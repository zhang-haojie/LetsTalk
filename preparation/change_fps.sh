
SOURCE_FOLDER="/path/to/dataset/videos"
OUTPUT_FOLDER="/path/to/dataset/videos_25fps"
TARGET_FRAMERATE=25

mkdir -p "$OUTPUT_FOLDER"

export SOURCE_FOLDER OUTPUT_FOLDER TARGET_FRAMERATE

find "$SOURCE_FOLDER" -name '*.mp4' | parallel ffmpeg -i {} -r "$TARGET_FRAMERATE" "$OUTPUT_FOLDER/{/.}.mp4"

echo "Frame rate conversion completed for all videos."
