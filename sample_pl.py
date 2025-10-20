import os
import torch
import imageio
import logging
import argparse
from einops import rearrange, repeat
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from models import get_models
from models.audio_proj import AudioProjModel
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from utils import find_model, cleanup
from datasets import get_dataset


class LatteSamplingModule(LightningModule):
    def __init__(self, args, logger: logging.Logger):
        super(LatteSamplingModule, self).__init__()
        self.args = args
        self.logging = logger
        self.model = get_models(args).to(self.device)
        self.audioproj = AudioProjModel(
            seq_len=5,
            blocks=12,
            channels=args.audio_dim,
            intermediate_dim=512,
            output_dim=args.audio_dim,
            context_tokens=args.audio_token,
        )

        state_dict, audioproj_dict = find_model(args.pretrained)
        self.model.load_state_dict(state_dict)
        self.logging.info(f"Loaded model checkpoint from {args.pretrained}")
        self.audioproj.load_state_dict(audioproj_dict)

        self.diffusion = create_diffusion(str(args.num_sampling_steps))
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(self.device)

        self.model.eval()
        self.vae.eval()
        self.audioproj.eval()

    def sample(self, z, model_kwargs, sample_method='ddpm'):
        if sample_method == 'ddim':
            return self.diffusion.ddim_sample_loop(
                self.model.forward_with_cfg if self.args.cfg_scale > 1.0 else self.model.forward,
                z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device
            )
        elif sample_method == 'ddpm':
            return self.diffusion.p_sample_loop(
                self.model.forward_with_cfg if self.args.cfg_scale > 1.0 else self.model.forward,
                z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device
            )
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        video_names = batch["video_name"]
        if "latent" in self.args.dataset:
            ref_latents = batch["ref_latent"]
        else:
            ref_image = batch["ref_image"]
            ref_latents = self.vae.encode(ref_image).latent_dist.sample().mul_(0.18215)

        ref_latents = repeat(ref_latents, "b c h w -> b f c h w", f=self.args.clip_frames)
        model_kwargs = dict(y=None, cond=ref_latents)

        local_batch_size = ref_latents.size(0)
        z = torch.randn(local_batch_size, self.args.clip_frames, self.args.in_channels, self.args.latent_size, self.args.latent_size, device=self.device)
        if "audio" in batch:
            audio_emb = batch["audio"]
            audio_emb = self.audioproj(audio_emb)
            model_kwargs.update(audio_embed=audio_emb)

        if self.args.cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            model_kwargs.update(cfg_scale=self.args.cfg_scale)

        samples = self.sample(z, model_kwargs, sample_method=self.args.sample_method)

        samples = rearrange(samples, 'b f c h w -> (b f) c h w')
        samples = self.vae.decode(samples / 0.18215).sample
        samples = rearrange(samples, '(b f) c h w -> b f c h w', b=local_batch_size)

        for sample, video_name in zip(samples, video_names):
            video_ = ((sample * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
            video_save_path = os.path.join(self.args.save_video_path, f"{video_name}.mp4")
            imageio.mimwrite(video_save_path, video_, fps=self.args.fps, quality=9)
            self.logging.info(f"Saved video at {video_save_path}")

        return video_save_path


def main(args):
    # Setup logger
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    # Determine if the current process is the main process (rank 0)
    is_main_process = (int(os.environ.get("LOCAL_RANK", 0)) == 0)
    if is_main_process:
        os.makedirs(args.save_video_path, exist_ok=True)
        print(f"Saving .mp4 samples at {args.save_video_path}")

    # Create dataset and dataloader
    dataset = get_dataset(args)
    val_loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,  # Batch size 1 for sampling
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Validation set contains {len(dataset)} samples")

    sample_size = args.image_size // 8
    args.latent_size = sample_size

    # Initialize the sampling module
    pl_module = LatteSamplingModule(args, logger)

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],  # Specify GPU ids
        strategy="auto",
        logger=False,
        precision=args.precision if args.precision else "32-true",
    )

    # Run validation to generate samples
    trainer.validate(pl_module, dataloaders=val_loader)

    logger.info("Sampling completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample.yaml")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    main(omega_conf)