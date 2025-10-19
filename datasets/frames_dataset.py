import os
import cv2
import time
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        configs,
        temporal_sample=None,
        transform=None,
    ):
        self.data_dir = configs.data_dir
        self.datalists = [d for d in os.listdir(self.data_dir)]
        self.frame_interval = configs.frame_interval
        self.num_frames = configs.num_frames
        self.temporal_sample = temporal_sample
        self.transform = transform

        self.audio_dir = configs.audio_dir
        self.audio_margin = configs.audio_margin
        self.initial_frames = configs.initial_frames
        self.slide_window = (configs.in_channels == 4)

    def __len__(self):
        return len(self.datalists)

    def get_indices(self, total_frames):
        indices = np.arange(total_frames)
        # Randomly select frames
        start_index, end_index = self.temporal_sample(total_frames - self.audio_margin)
        selected_indices = torch.linspace(start_index, end_index - 1, self.num_frames + self.initial_frames, dtype=int)
        video_indices = selected_indices[self.initial_frames:]
        # start_index = random.randint(
        #     max(self.initial_frames, self.audio_margin), 
        #     total_frames - self.num_frames - self.audio_margin - 1,
        # )
        # selected_indices = indices[start_index - self.initial_frames: start_index + self.num_frames]
        # video_indices = torch.from_numpy(selected_indices[self.initial_frames:])

        # Choose a reference frame from the remaining frames
        remaining_indices = np.setdiff1d(indices, selected_indices)
        if len(remaining_indices) == 0:
            remaining_indices = indices
        ref_index = np.random.choice(remaining_indices)

        # Add the reference frame index to the selected_indices
        selected_indices_ = np.append(selected_indices, ref_index)
        return selected_indices_, video_indices

    def load_audio_emb(self, audio_emb_path, video_indices):
        # Extract wav hidden features
        audio_emb = torch.load(audio_emb_path)
        if self.slide_window:
            audio_indices = (
                torch.arange(2 * self.audio_margin + 1) - self.audio_margin
            )  # Generates [-2, -1, 0, 1, 2]
            center_indices = video_indices.unsqueeze(1) + audio_indices.unsqueeze(0)
            try:
                audio_tensor = audio_emb[center_indices]
            except:
                print(audio_emb_path)
                print(len(audio_emb))
        else:
            audio_tensor = audio_emb[video_indices]
        return audio_tensor


class VideoFramesDataset(BaseDataset):
    def load_images(self, folder_path, selected_indices):
        images = []
        for idx in selected_indices:
            img_path = os.path.join(folder_path, f"{idx+1:04d}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                raise FileNotFoundError(f"Image {img_path} not found.")
            images.append(img)
        return np.array(images)

    def __getitem__(self, index):
        sample = self.datalists[index]
        video_folder_path = os.path.join(self.data_dir, sample)
        audio_emb_path = "{}/{}.pt".format(self.audio_dir, sample)
        audio_emb = torch.load(audio_emb_path, weights_only=True)

        # Extract total frames count based on images available
        video_frames = len([name for name in os.listdir(video_folder_path) if name.endswith(".jpg")])
        total_frames = min(audio_emb.size(0), video_frames)

        selected_indices, video_indices = self.get_indices(total_frames)

        # Load the selected frames including the reference frame
        all_frames = self.load_images(video_folder_path, selected_indices)
        all_frames = torch.from_numpy(all_frames).permute(0, 3, 1, 2).contiguous()

        # Apply transformation if it exists
        if self.transform:
            all_frames = self.transform(all_frames)

        # Separate the reference frame, motions, and image_window
        ref_image = all_frames[-1]  # The last frame is the reference frame
        video_frames = all_frames[:-1]  # All frames except the last one

        motions = video_frames[:self.initial_frames]
        image_window = video_frames[self.initial_frames:]

        audio_indices = (
            torch.arange(2 * self.audio_margin + 1) - self.audio_margin
        )  # Generates [-2, -1, 0, 1, 2]
        center_indices = video_indices.unsqueeze(1) + audio_indices.unsqueeze(0)
        audio_tensor = audio_emb[center_indices].squeeze(1)
        return {"video": image_window, "audio": audio_tensor, "ref_image": ref_image, "motions": motions, "video_name": sample}


class FramesLatentDataset(BaseDataset):
    def load_latents(self, folder_path, selected_indices):
        latents = []
        for idx in selected_indices:
            latent_path = os.path.join(folder_path, f"{idx+1:04d}.npz")  # Change .pt to .npz
            latent_data = np.load(latent_path)['latent']  # Load latent data from .npz file
            latent = torch.tensor(latent_data)  # Convert numpy array to PyTorch tensor
            latents.append(latent)
        return torch.stack(latents)  # Return stacked latents as a tensor

    def __getitem__(self, index):
        sample = self.datalists[index]
        video_folder_path = os.path.join(self.data_dir, sample)
        audio_emb_path = "{}/{}.pt".format(self.audio_dir, sample)
        audio_emb = torch.load(audio_emb_path, weights_only=True)

        # Extract total frames count based on available .npz files
        video_frames = len([name for name in os.listdir(video_folder_path) if name.endswith(".npz")])
        total_frames = min(audio_emb.size(0), video_frames)

        selected_indices, video_indices = self.get_indices(total_frames)

        # Load the selected latents including the reference frame
        all_latents = self.load_latents(video_folder_path, selected_indices)

        # Separate the reference frame, motions, and image_window
        ref_latent = all_latents[-1]  # The last frame is the reference frame
        video_latents = all_latents[:-1]  # All frames except the last one

        motion_latents = video_latents[:self.initial_frames]
        latent_window = video_latents[self.initial_frames:]

        audio_indices = (
            torch.arange(2 * self.audio_margin + 1) - self.audio_margin
        )  # Generates [-2, -1, 0, 1, 2]
        center_indices = video_indices.unsqueeze(1) + audio_indices.unsqueeze(0)
        audio_tensor = audio_emb[center_indices].squeeze(1)
        return {"video": latent_window, "audio": audio_tensor, "ref_latent": ref_latent, "motions": motion_latents, "video_name": sample}


class VideoLatentDataset(BaseDataset):
    def load_latents(self, latent_path):
        latents = torch.load(latent_path)  # Load the entire latent tensor
        return latents

    def __getitem__(self, index):
        sample = self.datalists[index]
        video_latent_path = os.path.join(self.data_dir, sample)
        audio_emb_path = "{}/{}.pt".format(self.audio_dir, sample)
        audio_emb = torch.load(audio_emb_path, weights_only=True)

        # Load latents and calculate total frames
        latents = self.load_latents(video_latent_path)  # Latent shape: [num_frames, channels, height, width]
        video_frames = latents.shape[0]
        total_frames = min(audio_emb.size(0), video_frames)

        selected_indices, video_indices = self.get_indices(total_frames)

        # Select latent frames
        all_latents = latents[selected_indices]

        # Separate the reference frame, motions, and image_window
        ref_latent = all_latents[-1]  # The last frame is the reference frame
        video_latents = all_latents[:-1]  # All frames except the last one

        motion_latents = video_latents[:self.initial_frames]
        latent_window = video_latents[self.initial_frames:]

        audio_indices = (
            torch.arange(2 * self.audio_margin + 1) - self.audio_margin
        )  # Generates [-2, -1, 0, 1, 2]
        center_indices = video_indices.unsqueeze(1) + audio_indices.unsqueeze(0)
        audio_tensor = audio_emb[center_indices].squeeze(1)
        return {"video": latent_window, "audio": audio_tensor, "ref_latent": ref_latent, "motions": motion_latents, "video_name": sample}


class VideoDubbingDataset(BaseDataset):
    def __init__(
        self,
        configs,
        transform=None,
    ):
        self.data_dir = configs.data_dir
        self.datalists = [d for d in os.listdir(self.data_dir)]
        self.frame_interval = configs.frame_interval
        self.num_frames = configs.num_frames
        self.transform = transform
        self.audio_dir = configs.audio_dir
        self.audio_margin = configs.audio_margin

    def load_images(self, folder_path, selected_indices):
        images = []
        for idx in selected_indices:
            img_path = os.path.join(folder_path, f"{idx+1:04d}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                raise FileNotFoundError(f"Image {img_path} not found.")
            images.append(img)
        return np.array(images)

    def get_two_clip_indices(self, total_frames):
        # Calculate the total length needed for two clips and the gap between them
        total_needed = 2 * self.num_frames + self.num_frames  # 2 clips + gap
        
        # We need additional margin for audio indices at both ends
        total_needed_with_margin = total_needed + 2 * self.audio_margin
        
        # Check if video is long enough
        if total_frames < total_needed_with_margin:
            raise ValueError(f"Video has only {total_frames} frames, but need at least {total_needed_with_margin} frames (including audio margin)")
        
        # Randomly select a starting point that allows both clips to fit with audio margins
        max_start = total_frames - total_needed_with_margin
        start_frame = random.randint(self.audio_margin, max_start + self.audio_margin)
        
        # First clip indices (accounting for audio margin)
        first_clip_indices = range(start_frame, start_frame + self.num_frames)
        
        # Second clip starts after first clip + gap
        second_clip_start = start_frame + self.num_frames + self.num_frames
        second_clip_indices = range(second_clip_start, second_clip_start + self.num_frames)
        
        # Combine all indices needed for loading
        all_indices = list(first_clip_indices) + list(second_clip_indices)
        
        # Return the combined indices and the indices for each clip
        return all_indices, first_clip_indices, second_clip_indices

    def __getitem__(self, index):
        sample = self.datalists[index]
        video_folder_path = os.path.join(self.data_dir, sample)

        audio_emb_path = "{}/{}.pt".format(self.audio_dir, sample)
        audio_emb = torch.load(audio_emb_path, weights_only=True)

        # Extract total frames count based on images available
        video_frames = len([name for name in os.listdir(video_folder_path) if name.endswith(".jpg")])
        total_frames = min(audio_emb.size(0), video_frames)

        # Get indices for two clips
        all_indices, first_clip_indices, second_clip_indices = self.get_two_clip_indices(total_frames)
        
        # Load all needed frames
        all_frames = self.load_images(video_folder_path, all_indices)
        all_frames = torch.from_numpy(all_frames).permute(0, 3, 1, 2).contiguous()
        
        # Split into two clips
        first_clip_frames = all_frames[:self.num_frames]
        second_clip_frames = all_frames[self.num_frames:]
        
        # Apply transformation if it exists
        if self.transform:
            first_clip_frames = self.transform(first_clip_frames)
            second_clip_frames = self.transform(second_clip_frames)
        
        # Process audio for both clips
        def get_audio_tensor(video_indices):
            audio_indices = (
                torch.arange(2 * self.audio_margin + 1) - self.audio_margin
            )  # Generates [-2, -1, 0, 1, 2]
            center_indices = torch.tensor(video_indices).unsqueeze(1) + audio_indices.unsqueeze(0)
            return audio_emb[center_indices].squeeze(1)
        
        first_audio_tensor = get_audio_tensor(first_clip_indices)
        second_audio_tensor = get_audio_tensor(second_clip_indices)
        
        return {
            "first_video": first_clip_frames,
            "first_audio": first_audio_tensor,
            "second_video": second_clip_frames,
            "second_audio": second_audio_tensor,
            "video_name": sample
        }