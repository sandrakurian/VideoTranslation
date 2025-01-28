import os
import subprocess
import whisper
import re
import torch
from TTS.api import TTS
from pydub import AudioSegment
from typing import Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from language import language_mapper  # Import the LanguageMapper



# --------- HELPER FUNCTIONS --------- #

def extract_audio(input_video, output_audio):
    """Extract audio from a video file."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_video, "-q:a", "0", "-map", "a", output_audio],
            check=True,
        )
        print(f"Audio extracted to {output_audio}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e}")

def create_mute_video(input_video, output_video):
    """Create a muted version of the video."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_video, "-an", output_video],
            check=True,
        )
        print(f"Muted video saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating mute video: {e}")

def format_timestamp(seconds):
    """Format seconds into SRT-compatible timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"

def translate_srt(input_file, output_file, source_lang, target_lang):
    """Translate the content of an SRT file using a translation model."""
    try:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        with open(input_file, "r", encoding="utf-8") as src, open(output_file, "w", encoding="utf-8") as tgt:
            for line in src:
                if "-->" in line or line.strip().isdigit():
                    tgt.write(line)
                elif line.strip():
                    inputs = tokenizer(line.strip(), return_tensors="pt", max_length=512, truncation=True)
                    outputs = model.generate(**inputs)
                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    tgt.write(translated_text + "\n\n")
        print(f"Translated SRT saved to {output_file}")
    except Exception as e:
        print(f"Error during translation: {e}")

def timestamp_to_ms(timestamp):
    """Convert SRT timestamp to milliseconds."""
    hours, minutes, seconds_ms = timestamp.replace(',', '.').split(':')
    seconds, ms = seconds_ms.split('.')
    return (
        int(hours) * 3600 * 1000
        + int(minutes) * 60 * 1000
        + int(seconds) * 1000
        + int(ms)
    )

# --------- MAIN FUNCTIONS --------- #

def split_audio(video, video_blank, audio):
    """Split audio from the video and create a muted version."""
    extract_audio(video, audio)
    create_mute_video(video, video_blank)

def transcribe_audio_with_sentence_timestamps(audio_path: str, output_srt_path: str):
    """Transcribe audio using Whisper and save it as an SRT file with sentence-level timestamps."""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)

        sentence_segments = []
        current_sentence = []
        current_start = None

        for segment in result["segments"]:
            text = segment["text"].strip()
            if not current_start:
                current_start = segment["start"]

            current_sentence.append(text)
            if text.rstrip()[-1] in ".!?":
                sentence_segments.append({"start": current_start, "end": segment["end"], "text": " ".join(current_sentence).strip()})
                current_sentence = []
                current_start = None

        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(sentence_segments, start=1):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"]
                srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        print(f"Transcription saved to {output_srt_path}")

    except Exception as e:
        print(f"Error during transcription: {e}")

def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT-compatible timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"

def transcribe_translate(audio, srt_orig, srt_trans, lang_output):
    """Transcribe audio and translate the transcription."""
    transcribe_audio_with_sentence_timestamps(audio, srt_orig)
    translate_srt(srt_orig, srt_trans, lang_output)

def extract_reference_audio(input_video, output_reference, duration_seconds=30):
    """
    Extract a high-quality portion of audio to use as reference for voice cloning.
    Enhanced to select segments with clear speech.
    """
    try:
        # First, extract full audio in high quality
        temp_full_audio = "temp_full_audio.wav"
        subprocess.run(
            [
                "ffmpeg", "-i", input_video,
                "-vn",  # No video
                "-ar", "44100",  # High sample rate
                "-ac", "2",  # Stereo
                "-b:a", "192k",  # High bitrate
                temp_full_audio
            ],
            check=True,
        )
        
        # Extract the first 30 seconds (or specified duration)
        subprocess.run(
            [
                "ffmpeg",
                "-i", temp_full_audio,
                "-t", str(duration_seconds),
                "-ar", "44100",
                "-ac", "2",
                "-b:a", "192k",
                output_reference
            ],
            check=True,
        )
        
        print(f"High-quality reference audio extracted to {output_reference}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during reference audio extraction: {e}")
        
    finally:
        if os.path.exists(temp_full_audio):
            os.remove(temp_full_audio)

def generate_cloned_audio(srt_output, audio_output, reference_audio, tts_lang):
    """Generate TTS audio using XTTS-v2."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)

        with open(srt_output, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = re.compile(r"(\d+)\n([\d:,]+) --> ([\d:,]+)\n(.*?)(?=\n\n|\Z)", re.DOTALL)
        segments = [{"text": match.group(4).strip()} for match in pattern.finditer(content)]

        final_audio = AudioSegment.silent(duration=0)
        for segment in segments:
            temp_audio_file = "temp_segment.wav"
            tts.tts_to_file(
                text=segment["text"], file_path=temp_audio_file, speaker_wav=reference_audio, language=tts_lang
            )
            generated_audio = AudioSegment.from_file(temp_audio_file)
            final_audio += generated_audio
            os.remove(temp_audio_file)

        final_audio.export(audio_output, format="wav")
        print(f"Generated cloned audio saved to {audio_output}")

    except Exception as e:
        print(f"Error in generate_cloned_audio: {e}")

def dub_video(video_output, video_no_audio, audio_output):
    """
    Combine the muted video with the generated audio to create a dubbed video.
    """
    try:
        # Use ffmpeg to combine video (no audio) with the generated audio
        subprocess.run(
            [
                "ffmpeg", 
                "-i", video_no_audio,          # Input muted video
                "-i", audio_output,            # Input audio
                "-c:v", "copy",                # Copy video codec without re-encoding
                "-c:a", "aac",                 # Encode audio in AAC format
                "-strict", "experimental",     # Allow experimental features (if needed)
                "-shortest",                   # Ensure output matches the shortest stream (audio or video)
                video_output                   # Output file
            ],
            check=True
        )
        print(f"Dubbed video saved to {video_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video dubbing: {e}")

# --------- MAIN SCRIPT --------- #

if __name__ == "__main__":
    video_input = "results/video2/dcf_es.mp4"
    results_dir = "results/video2/to_english"

    lang_input = "es"
    lang_output = "en"

    os.makedirs(results_dir, exist_ok=True)

    video_no_audio = os.path.join(results_dir, "../video_no_audio.mp4")
    video_output = os.path.join(results_dir, f"video_{lang_output}.mp4")
    audio_input = os.path.join(results_dir, f"../audio_{lang_input}.mp3")
    audio_output = os.path.join(results_dir, f"audio_{lang_output}.wav")
    reference_audio = os.path.join(results_dir, "../reference_audio.wav")
    srt_input = os.path.join(results_dir, f"../transcript_{lang_input}.srt")
    srt_output = os.path.join(results_dir, f"transcript_{lang_output}.srt")

    extract_audio(video_input, audio_input)
    create_mute_video(video_input, video_no_audio)
    extract_reference_audio(audio_input, reference_audio)
    transcribe_audio_with_sentence_timestamps(audio_input, srt_input)
    translate_srt(srt_input, srt_output, source_lang=lang_input, target_lang=lang_output)
    generate_cloned_audio(srt_output, audio_output, reference_audio, lang_output)
    dub_video(video_output, video_no_audio, audio_output)