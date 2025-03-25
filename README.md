# Bespoke Speaks

Bespoke Speaks is a natural voice conversation system that combines speech-to-text, large language models, and advanced text-to-speech capabilities to create a seamless conversational AI experience.

## Features

- **Natural Conversation Flow**: Streaming audio processing with support for user interruptions
- **High-Quality Text-to-Speech**: Using the CSM (Composition of Semantic Modules) model for lifelike speech generation
- **Advanced Language Understanding**: Integration with state-of-the-art LLMs (Default: Qwen 2.5-3B)
- **Modular Architecture**: Event-driven design for easy extensibility
- **Voice Context Preservation**: Maintains consistent voice characteristics throughout the conversation
- **Conversation Saving**: Save conversations as audio files with metadata
- **Automatic Device Selection**: CUDA GPU support with automatic fallback to CPU
- **Efficient Model Loading**: Support for Hugging Face Accelerate for optimized model loading

## Architecture

Bespoke Speaks follows a modular, event-driven architecture:

- **Audio Module**: Handles recording and playback with real-time voice activity detection
- **NLP Module**: Manages transcription (Whisper) and language model interactions (Qwen)
- **Conversation Module**: Maintains conversation state and handles interruptions
- **Utils Module**: Provides common utilities like event bus and prompt management

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU with 16GB+ RAM
- Working microphone and speakers/headphones
- FFmpeg installed on your system

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bespoke-speaks
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS (using Homebrew)
brew install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

5. Verify audio setup (optional but recommended):
```bash
python tests/soundtest.py
```

## Usage

Start the application with default settings:

```bash
python -m app.main
```

### Command Line Arguments

Available command-line options:

- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-file`: Specify log file path (default: bespoke.log)
- `--model`: Choose LLM model (default: Qwen/Qwen2.5-3B-Instruct)

Example with custom settings:

```bash
python -m app.main --log-level DEBUG --model Qwen/Qwen2.5-7B-Instruct
```

## Hardware Requirements

- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: CUDA-capable GPU with 8GB+ VRAM, 32GB RAM
- **Audio**: Working microphone and speakers/headphones
- **Storage**: At least 10GB free space for models and dependencies

## Development

The system is designed for modularity and extensibility:

1. **Event System**: Components communicate via the event bus system in `app/utils/event_bus.py`
2. **Adding Components**: Create new modules in the appropriate directory and integrate via events
3. **Voice Prompts**: Manage voice prompts through `app/utils/prompt_manager.py`
4. **Testing**: Run audio setup test with `tests/soundtest.py`

## Troubleshooting

Common issues and solutions:

1. **Audio Device Issues**:
   - Run `python tests/soundtest.py` to verify audio setup
   - Check if PulseAudio is running: `pulseaudio --start`
   - Verify microphone/speaker permissions

2. **Model Loading Issues**:
   - Ensure sufficient RAM/VRAM
   - Install accelerate: `pip install 'accelerate>=0.26.0'`
   - Try CPU-only mode if GPU memory is insufficient

3. **FFmpeg Missing**:
   - Install FFmpeg using package manager
   - Verify installation: `ffmpeg -version`

## License

This project uses models with various licenses:

- CSM 1B: [Apache 2.0 License](https://github.com/SesameAILabs/csm)
- Whisper: [MIT License](https://github.com/openai/whisper)
- Qwen 2.5: [Qwen License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## Acknowledgments

- CSM model from Sesame AI Labs
- Whisper model from OpenAI
- Qwen 2.5 model from Alibaba Cloud