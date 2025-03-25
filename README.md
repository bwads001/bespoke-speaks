# Bespoke Speaks

Bespoke Speaks is a natural voice conversation system that combines speech-to-text, large language models, and advanced text-to-speech capabilities to create a seamless conversational AI experience.

## Features

- **Natural Conversation Flow**: Streaming audio processing with support for user interruptions
- **High-Quality Text-to-Speech**: Using the CSM (Composition of Semantic Modules) model for lifelike speech generation
- **Advanced Language Understanding**: Integration with state-of-the-art LLMs (Default: Qwen 2.5-3B)
- **Modular Architecture**: Event-driven design for easy extensibility
- **Voice Context Preservation**: Maintains consistent voice characteristics throughout the conversation
- **Conversation Saving**: Save conversations as audio files with metadata

## Architecture

Bespoke Speaks follows a modular, event-driven architecture:

- **Audio Module**: Handles recording and playback
- **NLP Module**: Manages transcription and language model interactions
- **Conversation Module**: Maintains conversation state and handles interruptions
- **Utils Module**: Provides common utilities like event bus and prompt management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bespoke-speaks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have ffmpeg installed:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS (using Homebrew)
brew install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

## Usage

Run the application with default settings:

```bash
python -m app.main
```

### Command Line Arguments

The following command-line arguments are available:

- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-file`: Specify the log file path
- `--model`: Choose the LLM model to use

Example with custom settings:

```bash
python -m app.main --log-level DEBUG --model Qwen/Qwen2.5-7B-Instruct
```

## Development

The system is designed for modularity and extensibility:

1. **Event System**: Components communicate via the event bus system in `app/utils/event_bus.py`
2. **Adding New Components**: Create new modules in the appropriate directory and integrate them through the event system
3. **Custom Voice Prompts**: Add new voice prompts in the `app/utils/prompt_manager.py` file

## Hardware Requirements

- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: CUDA-capable GPU with 8GB+ VRAM, 32GB RAM
- **Audio**: Microphone and speakers/headphones

## License

This project uses models with various licenses:

- CSM 1B: [Apache 2.0 License](https://github.com/SesameAILabs/csm)
- Whisper: [MIT License](https://github.com/openai/whisper)
- Qwen 2.5: [Qwen License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## Acknowledgments

- CSM model from Sesame AI Labs
- Whisper model from OpenAI
- Qwen 2.5 model from Alibaba Cloud