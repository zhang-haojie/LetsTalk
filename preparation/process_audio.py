import os
import torch
from pathlib import Path
from tqdm import tqdm
from audio_processor import AudioProcessor


def get_audio_paths(source_dir: Path):
    """Get .wav files from the source directory."""
    return sorted([item for item in source_dir.iterdir() if item.is_file() and item.suffix == ".wav"])

def process_audio(audio_path: Path, output_dir: Path, audio_processor: AudioProcessor):
    """Process a single audio file and save its embedding."""
    audio_emb, _ = audio_processor.preprocess(audio_path)
    torch.save(audio_emb, os.path.join(output_dir, f"{audio_path.stem}.pt"))

def process_all_audios(input_audio_list, output_dir):
    """Process all audio files in the list."""
    wav2vec_model_path = "pretrained/wav2vec/wav2vec2-base-960h"
    audio_separator_model_file = "pretrained/audio_separator/Kim_Vocal_2.onnx"
    audio_processor = AudioProcessor(
        16000,
        25,
        wav2vec_model_path,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(output_dir, "vocals"),
        # only_last_features=True
    )
    error_files = []
    for audio_path in tqdm(input_audio_list, desc="Processing audios"):
        try:
            process_audio(audio_path, output_dir, audio_processor)
        except:
            error_files.append(audio_path)
    print("Error file:")
    for error_file in error_files:
        print(error_file)


if __name__ == "__main__":
    input_dir = Path("path/to/your/output/folder")  # Set your input directory
    output_dir = os.path.join(input_dir.parent, "audio_emb")
    os.makedirs(output_dir, exist_ok=True)

    audio_paths = get_audio_paths(input_dir)
    process_all_audios(audio_paths, output_dir)
