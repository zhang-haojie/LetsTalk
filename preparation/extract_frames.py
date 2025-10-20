import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def convert_video_to_images(video_path: Path, output_dir: Path):
    """
    Convert a video file into a sequence of images at 25 fps and save them in the output directory.
    
    Args:
        video_path (Path): The path to the input video file.
        output_dir (Path): The directory where the extracted images will be saved.

    Returns:
        None
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ffmpeg command to convert video to images at 25 fps
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', 'fps=25',
        str(output_dir / '%04d.jpg')  # Save images as 0001.png, 0002.png, etc.
    ]
    
    try:
        print(f"Running command: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting video {video_path} to images: {e}")
        raise


def process_videos_in_folder(folder_path: Path, output_root: Path, max_workers=4):
    """
    Traverse through all mp4 files in a folder and convert each video to images in parallel using threads.
    
    Args:
        folder_path (Path): The directory containing mp4 files.
        output_root (Path): The root directory where extracted frames will be saved.
        max_workers (int): Maximum number of threads for parallel processing.

    Returns:
        None
    """
    # Gather all mp4 files
    video_files = list(folder_path.glob('*.mp4'))
    
    # Use ThreadPoolExecutor to process videos in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_file in video_files:
            # Create a directory named after the video file (without extension)
            output_dir = output_root / video_file.stem
            print(f"Submitting video: {video_file.name} for processing")
            
            # Submit the task to the thread pool
            futures.append(executor.submit(convert_video_to_images, video_file, output_dir))
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exception that occurred during the task execution
            except Exception as e:
                print(f"Error during processing: {e}")


if __name__ == "__main__":
    # Define the input folder containing .mp4 videos and the output root directory
    input_folder = Path("path/to/your/input/folder")  # Replace with your input folder path
    output_folder = Path("path/to/your/output/folder")  # Replace with your desired output folder path

    # Start processing with multi-threading (adjust the number of threads by changing max_workers)
    process_videos_in_folder(input_folder, output_folder, max_workers=4)
