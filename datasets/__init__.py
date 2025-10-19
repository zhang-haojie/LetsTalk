from torchvision import transforms
from datasets import video_transforms
from .frames_dataset import VideoFramesDataset, VideoLatentDataset, FramesLatentDataset, VideoDubbingDataset


def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop((args.num_frames + args.initial_frames) * args.frame_interval)

    transform = transforms.Compose([
        video_transforms.ToTensorVideo(),  # TCHW
        video_transforms.UCFCenterCropVideo(args.image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.dataset == "frame":
        return VideoFramesDataset(args, transform=transform, temporal_sample=temporal_sample)
    elif "frame" in args.dataset:
        return FramesLatentDataset(args, temporal_sample=temporal_sample)
    elif args.dataset ==  "video":
        return VideoDubbingDataset(args, transform=transform)
    return VideoLatentDataset(args, temporal_sample=temporal_sample)
