import os
import torch
import math
import logging
import argparse
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from glob import glob
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers import AutoencoderDC
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from copy import deepcopy
from einops import rearrange, repeat

from models import get_models
from models.audio_proj import AudioProjModel
from datasets import get_dataset
from diffusion import create_diffusion
from utils import (
    update_ema,
    requires_grad,
    clip_grad_norm_,
    cleanup,
    find_model
)


class LetstalkTrainingModule(LightningModule):
    def __init__(self, args, logger: logging.Logger):
        super(LetstalkTrainingModule, self).__init__()
        self.args = args
        self.logging = logger
        self.model = get_models(args)
        if args.use_compile:
            self.model = torch.compile(self.model)

        self.ema = deepcopy(self.model)
        self.audioproj = AudioProjModel(
            seq_len=5,
            blocks=12,
            channels=args.audio_dim,
            intermediate_dim=512,
            output_dim=args.audio_dim,
            context_tokens=args.audio_token,
        )

        # Load pretrained model if specified
        if args.pretrained is not None:
            self._load_pretrained_parameters(args)
        self.logging.info(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        self.diffusion = create_diffusion(timestep_respacing="")
        if args.in_channels == 32:
            self.vae = AutoencoderDC.from_pretrained(args.pretrained_model_path, subfolder="vae")
        else:
            self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        self.lr_scheduler = None

        # Freeze model
        self.vae.requires_grad_(False)
        requires_grad(self.ema, False)

        update_ema(self.ema, self.model, decay=0)  # Ensure EMA is initialized with synced weights
        self.model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.audioproj.train()
        self.ema.eval()

    def _load_pretrained_parameters(self, args):
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            self.logging.info("Using ema ckpt!")
            checkpoint = checkpoint["ema"]

        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                self.logging.info("Ignoring: {}".format(k))
        self.logging.info(f"Successfully Load {len(pretrained_dict) / len(checkpoint.items()) * 100}% original pretrained model weights ")

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.logging.info(f"Successfully load model at {args.pretrained}!")

        if "audioproj" in checkpoint.keys():
            audioproj_dict = checkpoint["audioproj"]
            self.audioproj.load_state_dict(audioproj_dict)

    # def on_load_checkpoint(self, checkpoint):
    #     file_name = args.pretrained.split("/")[-1].split('.')[0]
    #     if file_name.isdigit():
    #         self.global_step = int(file_name)

    def add_noise_to_image(self, images):
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(images.size(0),), device=self.device)
        image_noise_sigma = torch.exp(image_noise_sigma)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None]
        return noisy_images

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alphas_cumprod = torch.from_numpy(self.diffusion.alphas_cumprod)
        alphas_cumprod = alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def training_step(self, batch, batch_idx):
        if "latent" in self.args.dataset:
            latents = batch["video"] * self.vae.config.scaling_factor
            ref_latents = batch["ref_latent"] * self.vae.config.scaling_factor
            motion_latents = batch["motions"] * self.vae.config.scaling_factor if self.args.initial_frames != 0 else None
        else:
            x = batch["video"]
            ref_image = batch["ref_image"]

            with torch.no_grad():
                b, _, _, _, _ = x.shape
                x = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
                if self.args.in_channels == 32:
                    latents = self.vae.encode(x)[0] * self.vae.config.scaling_factor
                else:
                    latents = self.vae.encode(x).latent_dist.sample() * self.vae.scaling_factor
                latents = rearrange(latents, "(b f) c h w -> b f c h w", b=b).contiguous()

                ref_image = self.add_noise_to_image(ref_image)
                if self.args.in_channels == 32:
                    ref_latents = self.vae.encode(ref_image)[0] * self.vae.config.scaling_factor
                else:
                    ref_latents = self.vae.encode(ref_image).latent_dist.sample().mul_(0.18215)

                if self.args.initial_frames != 0:
                    motions = batch["motions"]
                    motions = rearrange(motions, "b f c h w -> (b f) c h w").contiguous()
                    motions = self.add_noise_to_image(motions)
                    if self.args.in_channels == 32:
                        motion_latents = self.vae.encode(motions)[0] * self.vae.config.scaling_factor
                    else:
                        motion_latents = self.vae.encode(motions).latent_dist.sample() * self.vae.scaling_factor
                    motion_latents = rearrange(motion_latents, "(b f) c h w -> b f c h w", b=b).contiguous()

        ref_latents = repeat(ref_latents, "b c h w -> b f c h w", f=latents.size(1))
        model_kwargs = dict(y=None, cond=ref_latents)
        if self.args.initial_frames != 0:
            motion_timesteps = torch.randint(0, 50, (latents.shape[0],), device=latents.device).long()
            motion_noise = torch.randn_like(motion_latents)
            # add motion noise
            noisy_motion_latents = self.add_noise(
                motion_latents, motion_noise, motion_timesteps
            )

            b, f, c, h, w = noisy_motion_latents.shape
            rand_mask = torch.rand(h, w).to(device=noisy_motion_latents.device)
            mask = rand_mask > 0.25
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
            mask = mask.expand(b, f, c, h, w) 
            noisy_motion_latents = noisy_motion_latents * mask

            model_kwargs.update(motion=noisy_motion_latents)

        if "audio" in batch:
            audio_emb = batch["audio"]
            audio_emb = self.audioproj(audio_emb)
            model_kwargs.update(audio_embed=audio_emb)

        timesteps = torch.randint(0, self.diffusion.num_timesteps, (latents.shape[0],), device=self.device)
        loss_dict = self.diffusion.training_losses(self.model, latents, timesteps, model_kwargs)
        loss = loss_dict["loss"].mean()

        if self.global_step < self.args.start_clip_iter:
            gradient_norm = clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm, clip_grad=False)
        else:
            gradient_norm = clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm, clip_grad=True)

        self.log("train_loss", loss, prog_bar=True)
        self.log("gradient_norm", gradient_norm, prog_bar=True)
        self.log("train_step", self.global_step)

        if (self.global_step+1) % self.args.log_every == 0:
            self.logging.info(
                f"(step={self.global_step+1:07d}/epoch={self.current_epoch:04d}) Train Loss: {loss:.4f}, Gradient Norm: {gradient_norm:.4f}"
            )
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        update_ema(self.ema, self.model)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        epoch = self.trainer.current_epoch
        step = self.trainer.global_step
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "audioproj": self.audioproj.state_dict(),
        }
        torch.save(checkpoint, f"{checkpoint_dir}/last.ckpt")
        if step % self.args.ckpt_every == 0:
            torch.save(checkpoint, f"{checkpoint_dir}/epoch{epoch}-step{step}.ckpt")

    def configure_optimizers(self):
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.opt,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )
        return [self.opt], [self.lr_scheduler]


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def create_experiment_directory(args):
    os.makedirs(args.results_dir, exist_ok=True)        # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(os.path.join(args.results_dir, "*")))
    model_string_name = args.model.replace("/", "-")    # e.g., Letstalk-L/2 --> Letstalk-L-2 (for naming folders)
    num_frame_string = f"F{args.num_frames}S{args.frame_interval}"
    experiment_dir = os.path.join(                      # Create an experiment folder
        args.results_dir,
        f"{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"
    )
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")    # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir


def main(args):
    seed = args.global_seed
    torch.manual_seed(seed)

    # Determine if the current process is the main process (rank 0)
    is_main_process = (int(os.environ.get("LOCAL_RANK", 0)) == 0)
    # Setup an experiment folder and logger only if main process
    if is_main_process:
        experiment_dir, checkpoint_dir = create_experiment_directory(args)
        logger = create_logger(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, "config.yaml"))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        experiment_dir = os.getenv("EXPERIMENT_DIR", "default_path")
        checkpoint_dir = os.getenv("CHECKPOINT_DIR", "default_path")
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    tb_logger = TensorBoardLogger(experiment_dir, name="letstalk")

    # Create the dataset and dataloader
    dataset = get_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if is_main_process:
        logger.info(f"Dataset contains {len(dataset)} videos ({args.data_dir})")

    if args.in_channels == 32:
        sample_size = args.image_size // 32
    else:
        sample_size = args.image_size // 8
    args.latent_size = sample_size

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader)) // torch.cuda.device_count()
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # In multi GPUs mode, the real batchsize is local_batch_size * GPU numbers
    if is_main_process:
        logger.info(f"One epoch iteration {num_update_steps_per_epoch} steps")
        logger.info(f"Num train epochs: {num_train_epochs}")

    # Initialize the training module
    pl_module = LetstalkTrainingModule(args, logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-{train_loss:.2f}-{gradient_norm:.2f}",
        save_top_k=3,
        every_n_train_steps=args.ckpt_every,
        monitor="train_step",
        mode="max"
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        # devices=[0],
        strategy="auto",
        max_epochs=num_train_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor()],
        precision=args.precision if args.precision else "32-true",
    )

    trainer.fit(pl_module, train_dataloaders=loader, ckpt_path=args.resume_from_checkpoint if 
                args.resume_from_checkpoint else None)

    pl_module.model.eval()
    cleanup()
    if is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))