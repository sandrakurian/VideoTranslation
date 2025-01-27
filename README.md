# Video Dubber

An AI-powered video dubbing application that can translate and clone voices across different languages while maintaining the original speaker's voice characteristics.

## Overview

This application takes a video in one language and creates a dubbed version in another language while preserving the original speaker's voice characteristics. It uses state-of-the-art AI models for transcription, translation, and voice synthesis to create natural-sounding dubbed videos.

## How It Works

The application follows a pipeline of operations:

1. **Reference Audio Extraction** (`extract_reference_audio`)
   - Extracts the first 30 seconds of audio from the input video
   - This sample serves as a reference for voice cloning, capturing the speaker's unique voice characteristics
   - Uses FFmpeg for high-quality audio extraction

2. **Audio-Video Separation** (`split_audio`)
   - Separates the audio track from the video
   - Creates a muted version of the original video
   - Enables parallel processing of audio and video streams

3. **Transcription and Translation** (`transcribe_translate`)
   - Uses OpenAI's Whisper model for accurate speech-to-text transcription
   - Creates timestamped transcriptions in SRT format
   - Translates the transcription using Helsinki-NLP's pre-trained models
   - Preserves timing information for synchronization

4. **Voice Cloning and Audio Generation** (`generate_cloned_audio`)
   - Utilizes Coqui's XTTS-v2 model for high-quality voice cloning
   - Generates translated speech while maintaining the original speaker's voice
   - Handles timing adjustments to match original video segments
   - Supports multiple languages while preserving natural speech patterns

5. **Video Dubbing** (`dub_video`)
   - Combines the muted video with the generated translated audio
   - Uses FFmpeg for precise audio-video synchronization
   - Preserves video quality through stream copying

## AI Models Used

### 1. Whisper (OpenAI)
- Used for speech recognition and transcription
- Chosen for its robust multilingual capabilities and high accuracy
- Provides precise timestamp information for better synchronization

### 2. Helsinki-NLP Translation Models
- Specifically using the `opus-mt` series from HuggingFace
- Selected for their:
  - Language-specific optimization
  - High-quality translations
  - Efficient processing
  - Support for multiple language pairs

### 3. XTTS-v2 (Coqui)
- Advanced text-to-speech model with voice cloning capabilities
- Selected for its ability to:
  - Clone voices with minimal reference audio
  - Generate natural-sounding speech in multiple languages
  - Maintain speaker identity across languages
  - Produce high-quality audio output

## Requirements

- Python 3.8+
- FFmpeg
- PyTorch
- Required Python packages:
  - whisper
  - transformers
  - TTS
  - pydub
  - torch

## Usage

```python
# Example usage
video_input = "path/to/input/video.mp4"
lang_input = "en"  # Source language
lang_output = "es" # Target language

# Create necessary directories
results_dir = "results/video1/to_spanish"
os.makedirs(results_dir, exist_ok=True)

# Run the dubbing pipeline
extract_reference_audio(video_input, reference_audio)
split_audio(video_input, video_no_audio, audio_input)
transcribe_translate(audio_input, srt_input, srt_output, lang_input, lang_output)
generate_cloned_audio(srt_output, audio_output, reference_audio, lang_output)
dub_video(video_output, video_no_audio, audio_output)
```

## Limitations and Considerations

- Quality of voice cloning depends on the reference audio quality
- Processing time varies based on video length and hardware capabilities
- Requires GPU for optimal performance with XTTS-v2
- Translation quality may vary for uncommon language pairs

## Future Improvements

- Add support for batch processing multiple videos
- Implement parallel processing for faster execution
- Add quality control checks for generated audio
- Expand language pair support
- Add fine-tuning options for voice cloning