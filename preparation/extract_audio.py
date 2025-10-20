import os
import math
import argparse
import subprocess
from tqdm import tqdm

def main(video_root, audio_root):
    error_files = []
    video_files = []
    for root, _, files in os.walk(video_root):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))

    for video_path in tqdm(video_files):
        relative_path = os.path.relpath(video_path, video_root)
        wav_path = os.path.join(audio_root, os.path.splitext(relative_path)[0] + '.wav')

        if not os.path.exists(os.path.dirname(wav_path)):
            os.makedirs(os.path.dirname(wav_path))

        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn', '-acodec',
            "pcm_s16le", '-ar', '16000', '-ac', '2',
            wav_path
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from video {video_path}: {e}")
            error_files.append(video_path)
            continue

    for e_file in error_files:
        print(f"Error extracting audio from video {e_file}")

if __name__ == '__main__':
    video_root = "path/to/your/input/folder"
    audio_root = "path/to/your/output/folder"
    os.makedirs(audio_root, exist_ok=True)

    main(video_root, audio_root)
