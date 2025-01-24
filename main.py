import os
import subprocess
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re
from TTS.api import TTS
from pydub import AudioSegment


# Utility function to ensure paths work across platforms
def ensure_results_directory():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# Extract audio from video
def extract_audio(input_video, output_audio):
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_video, "-q:a", "0", "-map", "a", output_audio],
            check=True,
        )
        print(f"Audio extracted to {output_audio}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e}")


# Create a mute version of the video
def create_mute_video(input_video, output_video):
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_video, "-an", output_video],
            check=True,
        )
        print(f"Muted video saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating mute video: {e}")


# Transcribe audio with Whisper and save as SRT
def transcribe_audio_with_timestamps(audio_path, output_srt_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)

        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = _format_timestamp(segment["start"])
                end_time = _format_timestamp(segment["end"])
                text = segment["text"].strip()

                srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
        print(f"Transcription saved to {output_srt_path}")
    except Exception as e:
        print(f"Error during transcription: {e}")


# Helper: Format timestamps for SRT
def _format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"


# Translate SRT file
def translate_srt(input_file, output_file, source_lang="en", target_lang="fr"):
    try:
        # Load translation model
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Read and translate line by line
        with open(input_file, "r", encoding="utf-8") as src, open(
            output_file, "w", encoding="utf-8"
        ) as tgt:
            for line in src:
                if "-->" in line or line.strip().isdigit():
                    tgt.write(line)
                elif line.strip():
                    inputs = tokenizer(
                        line.strip(),
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                    )
                    outputs = model.generate(**inputs)
                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    tgt.write(translated_text + "\n\n")
        print(f"Translated SRT saved to {output_file}")
    except Exception as e:
        print(f"Error during translation: {e}")


# Main function
def process_video(input_video):
    results_dir = ensure_results_directory()
    audio_output = os.path.join(results_dir, "audio.mp3")
    video_output = os.path.join(results_dir, "video_no_audio.mp4")
    srt_output = os.path.join(results_dir, "transcription.srt")
    translated_srt_output = os.path.join(results_dir, "transcription_fr.srt")

    # Step 1: Extract audio and mute video
    extract_audio(input_video, audio_output)
    create_mute_video(input_video, video_output)

    # Step 2: Transcribe audio
    if os.path.exists(audio_output):
        transcribe_audio_with_timestamps(audio_output, srt_output)

        # Step 3: Translate SRT
        translate_srt(srt_output, translated_srt_output, source_lang="en", target_lang="fr")
    else:
        print(f"Audio file {audio_output} not found.")


def generate_precise_audio(srt_file, output_audio_file, language='fr'):
    """
    Generate audio with exact timing matching SRT timestamps
    """
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_audio_file), exist_ok=True)
    
    # Validate input file
    if not os.path.exists(srt_file):
        return
    
    try:
        # Initialize TTS model for French
        tts = TTS(model_name="tts_models/fr/css10/vits")
        
        # Read SRT file and parse timestamps and text
        segments = []
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use regex to parse SRT file
        pattern = re.compile(r'(\d+)\n([\d:,]+)\s*-->\s*([\d:,]+)\n(.*?)(?=\n\n|\Z)', re.DOTALL)
        
        for match in pattern.finditer(content):
            # Ignore index, parse timestamp and text
            start_time, end_time, text = match.group(2), match.group(3), match.group(4)
            
            # Convert timestamp to milliseconds
            start_ms = _timestamp_to_ms(start_time)
            end_ms = _timestamp_to_ms(end_time)
            
            segments.append({
                'text': text.strip(),
                'start_ms': start_ms,
                'end_ms': end_ms,
                'duration_ms': end_ms - start_ms
            })
        
        # Create a list to store individual audio segments
        audio_segments = []
        
        for segment in segments:
            # Generate audio for the specific segment
            temp_audio_file = f"temp_segment_{segments.index(segment)}.wav"
            tts.tts_to_file(text=segment['text'], file_path=temp_audio_file)
            
            # Load generated audio
            generated_audio = AudioSegment.from_wav(temp_audio_file)
            
            # Adjust speed if needed
            if generated_audio.duration_seconds * 1000 > segment['duration_ms']:
                # Speed up
                playback_speed = generated_audio.duration_seconds * 1000 / segment['duration_ms']
                generated_audio = generated_audio.speedup(playback_speed=playback_speed)
            elif generated_audio.duration_seconds * 1000 < segment['duration_ms']:
                # Pad with silence
                silence = AudioSegment.silent(duration=segment['duration_ms'] - int(generated_audio.duration_seconds * 1000))
                generated_audio += silence
            
            # Prepare silent audio before the segment if needed
            if not audio_segments:
                # First segment starts at 0
                padding_before = AudioSegment.silent(duration=segment['start_ms'])
                full_segment = padding_before + generated_audio
            else:
                # Calculate padding needed between previous segment and this one
                last_segment_end = audio_segments[-1]['end_ms']
                padding_duration = max(0, segment['start_ms'] - last_segment_end)
                padding = AudioSegment.silent(duration=padding_duration)
                full_segment = padding + generated_audio
            
            audio_segments.append({
                'audio': full_segment,
                'start_ms': segment['start_ms'],
                'end_ms': segment['start_ms'] + len(full_segment)
            })
            
            # Clean up temporary file
            os.remove(temp_audio_file)
        
        # Combine all segments
        final_audio = sum(segment['audio'] for segment in audio_segments)
        
        # Export final audio
        final_audio.export(output_audio_file, format="wav")
    
    except Exception as e:
        import traceback
        traceback.print_exc()

def _timestamp_to_ms(timestamp):
    """
    Convert SRT timestamp to milliseconds
    """
    hours, minutes, seconds_ms = timestamp.replace(',', '.').split(':')
    seconds, ms = seconds_ms.split('.')
    
    total_ms = (int(hours) * 3600 + 
                int(minutes) * 60 + 
                int(seconds)) * 1000 + \
               int(float(f"0.{ms}") * 1000)
    
    return total_ms

if __name__ == "__main__":
    # # Specify the input video file
    # input_video_file = "video.mp4"  # Replace with your video filename

    # if os.path.exists(input_video_file):
    #     process_video(input_video_file)
    # else:
    #     print(f"Input video file not found: {input_video_file}")

    # Specify full paths
    srt_file = os.path.join('results', 'transcription_fr.srt')
    output_audio_file = os.path.join('results', 'precise_french_audio.wav')
    
    generate_precise_audio(
        srt_file=srt_file, 
        output_audio_file=output_audio_file
    )