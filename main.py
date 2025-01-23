import os
import subprocess

# Function to ensure the results directory exists
def ensure_results_directory():
    if not os.path.exists("results"):
        os.makedirs("results")

# Function to extract audio from the video
def extract_audio(input_video, output_audio):
    subprocess.run([
        "ffmpeg", "-i", input_video, "-q:a", "0", "-map", "a", output_audio
    ])

# Function to create a mute version of the video
def create_mute_video(input_video, output_video):
    subprocess.run([
        "ffmpeg", "-i", input_video, "-an", output_video
    ])

# Main function
def process_video(input_video):
    # Create results directory
    ensure_results_directory()

    # Define output paths
    audio_output = "results/audio.mp3"
    video_output = "results/video_no_audio.mp4"

    # Process the video
    print("Extracting audio...")
    extract_audio(input_video, audio_output)
    print(f"Audio saved to {audio_output}")

    print("Creating mute video...")
    create_mute_video(input_video, video_output)
    print(f"Muted video saved to {video_output}")

    print("Processing complete.")

# Specify the input video file
input_video_file = "video.mp4"  # Replace with your input video filename

# Call the main function
process_video(input_video_file)
