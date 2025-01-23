import os
import subprocess
import torch
from transformers import pipeline
import whisper
import os
import webrtcvad
import numpy as np
from pydub import AudioSegment


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

def transcribe_audio_with_timestamps(audio_path, output_srt_path):
    """
    Transcribe audio file with precise timestamps using Whisper.
    """
    try:
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe with timestamps
        result = model.transcribe(audio_path, word_timestamps=True)
        
        # Write SRT file
        with open(output_srt_path, 'w', encoding='utf-8') as srt_file:
            for i, segment in enumerate(result['segments'], 1):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                # Format timestamp for SRT
                start_str = _format_timestamp(start_time)
                end_str = _format_timestamp(end_time)
                
                # Write SRT entry
                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_str} --> {end_str}\n")
                srt_file.write(f"{text}\n\n")
        
        print(f"Transcription saved to {output_srt_path}")
    except Exception as e:
        print(f"An error occurred during transcription: {e}")

def _format_timestamp(seconds):
    """
    Convert seconds to SRT timestamp format
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

# Specify the input video file
input_video_file = "video.mp4"  # Replace with your input video filename

# Call the main function
process_video(input_video_file)


input_audio = "results/audio.mp3"
output_srt = "results\\transcription.srt"
    
# Ensure the input audio exists
if os.path.exists(input_audio):
    transcribe_audio_with_timestamps(input_audio, output_srt)
else:
    print(f"Audio file not found: {input_audio}")