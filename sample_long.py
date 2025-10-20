import os
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from pathlib import Path
from omegaconf import OmegaConf
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from einops import rearrange, repeat
from models import get_models
from models.audio_proj import AudioProjModel
from preparation.audio_processor import AudioProcessor
from utils import find_model, combine_video_audio, tensor_to_video


def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
    audio_emb = torch.stack(concatenated_tensors, dim=0)
    return audio_emb


@torch.no_grad()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)
    state_dict, audioproj_dict = find_model(args.pretrained)
    model.load_state_dict(state_dict)
    model.eval()

    audioproj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=args.audio_dim,
        intermediate_dim=512,
        output_dim=args.audio_dim,
        context_tokens=args.audio_token,
    ).to(device)
    audioproj.load_state_dict(audioproj_dict)
    audioproj.eval()

    sample_rate = args.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = args.fps
    wav2vec_model_path = args.wav2vec
    audio_separator_model_file = args.audio_separator

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="sd-vae-ft-ema").to(device)
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(args.image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
        )

    vae.requires_grad_(False)
    audioproj.requires_grad_(False)
    model.requires_grad_(False)

    # Prepare output directory
    os.makedirs(args.save_video_path, exist_ok=True)

    # Iterate through reference image folder
    ref_image_folder = args.data_dir
    audio_folder = args.audio_dir
    ref_image_paths = glob.glob(os.path.join(ref_image_folder, "*.jpg"))
    print(f"===== Process folder: {args.data_dir} =====")

    tensor_result = []
    # Add progress bar for reference image processing
    for ref_image_path in ref_image_paths:
        audio_path = os.path.join(audio_folder, f"{Path(ref_image_path).stem}.wav")

        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for {audio_path}")
            continue

        ref_name = Path(ref_image_path).stem
        video_save_path = os.path.join(args.save_video_path, f"{ref_name}.mp4")
        if os.path.exists(video_save_path):
            print(f"Skip video {video_save_path}")
            continue

        clip_length = args.clip_frames
        with AudioProcessor(
                sample_rate,
                fps,
                wav2vec_model_path,
                os.path.dirname(audio_separator_model_file),
                os.path.basename(audio_separator_model_file),
                os.path.join(args.save_video_path, "audio_preprocess")
            ) as audio_processor:
                audio_emb, audio_length = audio_processor.preprocess(audio_path, clip_length)

        audio_emb = process_audio_emb(audio_emb)
        # Load reference image
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        ref_image_np = np.array(ref_image_pil)
        ref_image_tensor = transform(ref_image_np).unsqueeze(0).to(device)
        ref_latents = vae.encode(ref_image_tensor).latent_dist.sample().mul_(0.18215)
        ref_latents = repeat(ref_latents, "b c h w -> b f c h w", f=clip_length)

        times = audio_emb.shape[0] // clip_length

        concat_samples = []
        for t in tqdm(range(times), desc="Processing"):
            audio_tensor = audio_emb[
                t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
            ]

            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(
                device=audioproj.device, dtype=audioproj.dtype)
            audio_tensor = audioproj(audio_tensor)

            if len(concat_samples) == 0:
                # The first iteration
                initial_latents = ref_latents[:, 0:args.initial_frames, ...]

            z = torch.randn(1, args.clip_frames, 4, latent_size, latent_size, device=device)
            model_kwargs = dict(y=None, cond=ref_latents, motion=initial_latents, audio_embed=audio_tensor)
            if args.cfg_scale > 1.0:
                z = torch.cat([z, z], 0)
                model_kwargs.update(cfg_scale=args.cfg_scale)

            # Sample images:
            if args.sample_method == 'ddim':
                samples = diffusion.ddim_sample_loop(
                    model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
            elif args.sample_method == 'ddpm':
                samples = diffusion.p_sample_loop(
                    model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )

            initial_latents = samples[:, 0 - args.initial_frames:]

            # Decode and save samples
            samples = rearrange(samples, 'b f c h w -> (b f) c h w')
            samples = vae.decode(samples / 0.18215).sample.cpu()
            concat_samples.append(samples)
            torch.cuda.empty_cache()

        tensor_result = torch.cat(concat_samples)
        tensor_result = tensor_result[:audio_length]
        tensor_result = ((tensor_result * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        tensor_to_video(tensor_result, video_save_path, audio_path)
    print("Sampling completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    if args.data_dir is not None:
        omega_conf.data_dir = args.data_dir
    main(omega_conf)
