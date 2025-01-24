# Video Dubbing and Translation Tool

## Overview
This Python application automates the process of dubbing videos into a different language by extracting audio, transcribing, translating, and regenerating synchronized audio.

## Features
- Audio extraction from video
- Speech transcription using Whisper
- Subtitle translation 
- Text-to-Speech (TTS) audio generation
- Video dubbing with synchronized audio

## Requirements
- Python 3.8+
- Libraries:
  - whisper
  - transformers
  - TTS
  - pydub
  - ffmpeg

## Workflow
1. Extract original audio from video
2. Create a muted version of the original video
3. Transcribe audio to original language subtitles
4. Translate subtitles to target language
5. Generate new audio using TTS for translated text
6. Combine muted video with new audio track

## Current Limitations
- Single speaker support
- Translation accuracy dependent on model
- Voice cloning not yet implemented

## Future Improvements
- Multi-speaker voice differentiation
- Enhanced translation accuracy
- Voice cloning capabilities

## Usage
```python
# Example configuration
video_input = "video.mp4"
lang_input = "en"   # Original language
lang_output = "es"  # Target language

# Main processing steps
split_audio(video_input, video_no_audio, audio_input)
transcribe_translate(audio_input, srt_input, srt_output, lang_input, lang_output)
generate_audio(srt_output, audio_output, lang_output)
dub_video(video_output, video_no_audio, audio_output)
```
