import os
import argparse
import logging
import time
import sys
from typing import Optional, Any
import threading
import numpy as np

import torch

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Filter FFmpeg warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")

from app.utils.event_bus import EventBus, global_event_bus
from app.utils.prompt_manager import PromptManager
from app.audio.recorder import StreamingRecorder
from app.audio.speaker import Speaker
from app.nlp.transcriber import StreamingTranscriber
from app.nlp.llm_service import LLMService
from app.conversation.manager import ConversationManager
from app.conversation.context import ConversationContext
from app.conversation.interruption import InterruptionHandler
from app.utils.conversation_saver import ConversationSaver
# Import csm cleanup directly from service
from app.audio.csm_service import cleanup

class BespokeApp:
    """
    Main application class for Bespoke Speaks.
    
    This class orchestrates the components of the system and
    manages the overall application flow.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the application.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._setup_logging()
        self.logger = logging.getLogger("BespokeApp")
        
        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        print(f"🖥️  Using device: {self.device}")
        
        # Set up components
        self._setup_components()
        
        # Internal state
        self.is_running = False
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.get("log_file", "bespoke.log"))
            ]
        )
    
    def _setup_components(self) -> None:
        """Initialize and connect all components."""
        self.logger.info("Setting up components...")
        print("\n⚙️  Setting up components...")
        
        # Create and configure event bus
        self.event_bus = global_event_bus
        
        # Set up prompt manager and load initial prompt
        print("  • Loading voice prompts...")
        self.prompt_manager = PromptManager(sample_rate=24000)
        initial_prompt = self.prompt_manager.get_prompt_segment()
        
        # Set up conversation context
        print("  • Setting up conversation context...")
        self.context = ConversationContext(
            initial_prompt_segment=initial_prompt,
            event_bus=self.event_bus
        )
        
        # Set up audio components
        print("  • Initializing audio recorder...")
        self.recorder = StreamingRecorder(
            sample_rate=16000,
            event_bus=self.event_bus
        )
        
        print("  • Setting up TTS speaker...")
        self.speaker = Speaker(
            device=self.device,
            event_bus=self.event_bus
        )
        
        # Set up NLP components
        print("  • Preparing speech recognition...")
        self.transcriber = StreamingTranscriber(
            device=self.device,
            event_bus=self.event_bus
        )
        
        print("  • Loading language model...")
        self.llm_service = LLMService(
            device=self.device,
            event_bus=self.event_bus
        )
        
        # Set up conversation components
        print("  • Setting up conversation manager...")
        self.conversation_manager = ConversationManager(
            event_bus=self.event_bus
        )
        
        print("  • Setting up interruption detection...")
        self.interruption_handler = InterruptionHandler(
            event_bus=self.event_bus
        )
        
        # Set up conversation saver
        print("  • Setting up conversation saver...")
        self.conversation_saver = ConversationSaver(
            sample_rate=24000,
            event_bus=self.event_bus
        )
        
        # Connect components through events
        print("  • Connecting components...")
        self._connect_event_handlers()
        
        self.logger.info("All components initialized")
        print("✅ All components initialized successfully!\n")
    
    def _connect_event_handlers(self) -> None:
        """
        Connect components through event subscriptions.
        
        This is where we establish the event flow between components.
        """
        # Connect speech events to handlers
        self.event_bus.subscribe("speech_detected", self._handle_speech_detected)
        self.event_bus.subscribe("speech_ended", self._handle_speech_ended)
        
        # Connect response events
        self.event_bus.subscribe("response_ready", self._handle_response_ready)
        
        # Connect TTS events
        self.event_bus.subscribe("speak_response", self._handle_speak_response)
        
        # Connect transcriber to audio events
        self.transcriber.subscribe_to_audio_events()
        
        # Critical error handler - will exit the application
        self.event_bus.subscribe("critical_error", self._handle_critical_error)
    
    def _handle_speech_detected(self, data: Any) -> None:
        """
        Handle speech_detected events.
        
        Args:
            data: Event data (not used)
        """
        # Log and publish event for other components that might need it
        self.logger.debug("Speech detected in main app")
        # We could add additional processing here if needed in the future
        
    def _handle_speech_ended(self, audio_data: np.ndarray) -> None:
        """
        Handle speech_ended events by triggering transcription.
        
        Args:
            audio_data: Recorded audio data
        """
        # Log speech ended
        self.logger.info("Speech ended, transcribing audio")
        
        # We don't need to explicitly start transcription,
        # the transcriber is already subscribed to audio events
    
    def _handle_response_ready(self, response: str) -> None:
        """
        Handle response_ready events.
        
        Args:
            response: Generated text response
        """
        if not response or response.isspace():
            self.logger.warning("Empty response received")
            print("❌ Empty response received from language model")
            return
            
        self.logger.info(f"Response generated: {response}")
        print(f"\n\n🤖 Assistant: \"{response}\"")
        
        # Add to conversation history using conversation_manager
        # This will also trigger speech synthesis through the manager
        self.conversation_manager.handle_ai_message(response)
    
    def _handle_speak_response(self, text: str) -> None:
        """
        Handle speak_response events by triggering TTS.
        
        Args:
            text: Text to speak
        """
        # Trigger speech generation without context segments
        # The context is now handled internally by the csm_helper
        self.speaker.speak(text)
    
    def start(self) -> None:
        """Start the application."""
        self.logger.info("Starting Bespoke Speaks...")
        self.is_running = True
        
        try:
            print("\n🚀 Starting Bespoke Speaks...")
            
            # Prepare the greeting
            initial_greeting = "Hello! I'm your voice assistant. How can I help you today?"
            
            # Let's ensure our models are ready (they should already be initialized)
            print("\n📢 Verifying speech capability...")
            
            # Remove explicit reload of models - they are already loaded during component initialization
            # in the Speaker's __init__ method
            
            print("\n🔄 Preparing speech recognition...")
            self.transcriber._load_models()
            
            print("\n🔄 Preparing language model...")
            self.llm_service._load_models()
            
            # Initial greeting - first speak, then start listening
            print("\n🗣️  Initial greeting...")
            # Print greeting only ONCE here
            print(f"💬 AI: \"{initial_greeting}\"")
            
            # Publish AI message for other components
            # This will trigger the ConversationManager to handle it and publish a speak_response event
            self.event_bus.publish("ai_message", initial_greeting)
            
            # Wait for speech to complete before continuing
            # This blocks until speech generation and playback is done
            self.speaker.wait_for_playback_complete()
            
            # NOW start the recorder after the greeting is completed
            self.recorder.start_streaming()
            
            # No need to print additional UI messages, they're now in the recorder
            
            # Main application loop
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            print("\n👋 Interrupted by user. Shutting down...")
        except Exception as e:
            self.logger.error(f"Error in application: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
        finally:
            self.stop()
    
    def _main_loop(self) -> None:
        """Main application loop."""
        self.logger.info("Entering main loop")
        
        try:
            while self.is_running:
                # This is mostly an event-driven system, so we just need to
                # keep the main thread alive and responsive to keyboard interrupts
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, exiting main loop")
            raise
    
    def stop(self) -> None:
        """Stop the application and clean up resources."""
        self.logger.info("Stopping application...")
        print("\n🛑 Stopping application...")
        
        # Stop the recorder
        self.recorder.stop_streaming()
        
        # Stop any ongoing playback
        self.speaker.stop_playback()
        
        # Set state to not running
        self.is_running = False
        
        self.logger.info("Application stopped")
        print("✅ Application stopped successfully")

    def _handle_critical_error(self, error_message: str) -> None:
        """
        Handle critical errors that should terminate the application.
        
        Args:
            error_message: Description of the critical error
        """
        print(f"\n🛑 CRITICAL ERROR: {error_message}")
        print("🛑 Application cannot continue running")
        print("🛑 Shutting down...\n")
        
        # Set global flag to indicate we're in cleanup mode to prevent new operations
        import builtins
        setattr(builtins, "_bespoke_cleaning_up", True)
        
        # Attempt to clean up resources
        try:
            # Just stop recording, don't perform complex cleanups that might trigger more errors
            if hasattr(self, 'recorder') and self.recorder:
                # Use the proper method to stop the recorder
                if hasattr(self.recorder, 'stop_recording'):
                    self.recorder.stop_recording()
                
            # Stop playback but skip full cleanup on the speaker
            if hasattr(self, 'speaker') and self.speaker:
                # Just stop playback without full cleanup to avoid triggering more speak calls
                if hasattr(self.speaker, 'stop_playback'):
                    self.speaker.stop_playback()
            
            # We will skip the direct cleanup() call as it's likely to cause the same error
            # that triggered this handler in the first place
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        # Exit with error code
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bespoke Speaks - Voice Conversation System")
    
    parser.add_argument("--log-level", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO",
                        help="Set the logging level")
    
    parser.add_argument("--log-file",
                        default="bespoke.log",
                        help="Log file path")
    
    parser.add_argument("--model",
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="LLM model to use")
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    print("\n=== Bespoke Speaks - Initializing ===\n")
    
    args = parse_arguments()
    
    # Convert args to config dict
    config = {
        "log_level": args.log_level,
        "log_file": args.log_file,
        "model": args.model
    }
    
    # Create and start the application
    app = BespokeApp(config)
    app.start()


if __name__ == "__main__":
    main() 