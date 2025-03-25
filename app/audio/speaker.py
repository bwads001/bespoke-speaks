import torch
import sounddevice as sd
import logging
import threading
from typing import Optional, List, Callable
import time
import numpy as np
import torchaudio
from huggingface_hub import hf_hub_download

from app.utils.event_bus import EventBus
from app.utils.model_loader import ModelLoader
from app.models.generator import get_or_create_generator, prepare_prompt_segment, Segment

class Speaker:
    """
    Text-to-speech component using CSM model.
    
    This class handles text-to-speech conversion using the CSM model
    and manages audio playback with support for interruptions.
    """
    
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the speaker.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            event_bus: Event bus for communication with other components
        """
        self.device = device
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("Speaker")
        
        # Playback state
        self._is_speaking = False
        self._playback_thread: Optional[threading.Thread] = None
        self._should_stop_playback = False
        
        # Register for interruption events
        self.event_bus.subscribe("user_interruption", self._handle_interruption)
        
        # Attempt to load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """
        Initialize the CSM model if not already loaded.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        self._logger.info(f"Loading CSM model on {self.device}...")
        
        try:
            # Use the model loader utility with a custom load function
            def load_csm_components():
                # Get the generator and prompt segment
                generator = get_or_create_generator(self.device)
                prompt_segment = prepare_prompt_segment(self.device)
                
                return {
                    "generator": generator,
                    "prompt_segment": prompt_segment
                }
            
            # Use the unified model loader
            result = ModelLoader.load_model(
                load_function=load_csm_components,
                model_name="CSM 1B",
                device=self.device,
                model_type="speech",
                event_bus=self.event_bus
            )
            
            if not result:
                self._logger.error("Failed to load CSM model - speech generation is a critical feature")
                self.event_bus.publish("critical_error", "Text-to-speech system failed. Application cannot function properly without speech capability.")
                return False
            
            # Store references to the loaded components
            self.csm_generator = result["generator"]
            self.prompt_segment = result["prompt_segment"]
            
            self._logger.info("CSM model loaded successfully")
            return True
            
        except Exception as e:
            # Print detailed error information
            self._logger.error(f"Error loading CSM model: {e}", exc_info=True)
            
            # This is a critical failure
            self.event_bus.publish("critical_error", "Text-to-speech system failed. Application cannot function properly without speech capability.")
            return False
    
    def _handle_interruption(self, _: Optional[dict] = None) -> None:
        """Handle user interruption event."""
        self.stop_playback()
        self._logger.info("Playback interrupted by user")
        print("\n🤚 Playback interrupted by user")
    
    def speak(self, 
              text: str, 
              blocking: bool = False) -> None:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
            blocking: Whether to block until playback is complete
        """
        if not text:
            self._logger.warning("Empty text provided to speak")
            return
        
        # Log but don't print the text - let the caller handle display
        self._logger.info(f"Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Start playback in a separate thread if non-blocking
        if not blocking:
            self._playback_thread = threading.Thread(
                target=self._speak_impl,
                args=(text,)
            )
            self._playback_thread.daemon = True
            self._playback_thread.start()
        else:
            self._speak_impl(text)

    def _speak_impl(self, text: str) -> None:
        """
        Implementation of the speak method.
        
        Args:
            text: Text to speak
        """
        # Check if we're in cleanup mode and avoid execution
        try:
            import builtins
            if hasattr(builtins, "_bespoke_cleaning_up") and getattr(builtins, "_bespoke_cleaning_up"):
                self._logger.info("Skipping speech generation during application cleanup")
                return
        except:
            pass
            
        # Start playback
        self._is_speaking = True
        self._should_stop_playback = False
        
        # Notify that TTS has started
        self.event_bus.publish("tts_started", {"text": text})
        
        try:
            # Generate speech using direct approach with our generator
            self._logger.info(f"Generating speech for: {text}...")
            print(f"🔊 Generating speech... [ID: speech-gen-{id(text)}]")
            
            # Generate audio using direct model access - no wrapper service call
            generation_start = time.time()
            
            # Generate with direct model access
            audio_tensor = self.csm_generator.generate(
                text=text,
                speaker=1,
                context=[self.prompt_segment],
                max_audio_length_ms=10_000,
                temperature=0.7,
                topk=50,
            )
            
            # Convert to numpy for playback
            audio_np = audio_tensor.cpu().numpy()
            
            generation_time = time.time() - generation_start
            self._logger.info(f"Speech generation took {generation_time:.2f} seconds")
            
            # Check if we should stop (user may have interrupted during generation)
            if self._should_stop_playback:
                self._logger.info("Playback canceled before it started")
                return
                
            # Calculate audio duration
            sample_rate = self.csm_generator.sample_rate
            audio_duration = len(audio_np) / sample_rate
            self._logger.info(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Play audio with progress bar
            print(f"🎧 Playing response ({audio_duration:.1f}s)...")
            
            # Start playback
            sd.play(audio_np, sample_rate)
            
            # Show progress during playback
            start_play_time = time.time()
            
            # Monitor playback
            while not self._should_stop_playback and sd.get_stream().active:
                elapsed = time.time() - start_play_time
                progress = min(1.0, elapsed / audio_duration)
                bars = int(progress * 20)
                
                # Don't spam the console with progress updates
                if bars % 5 == 0:
                    print(f"\r🎧 Playing: {bars*5}% [{'▓'*bars}{' '*(20-bars)}]", end="", flush=True)
                    
                time.sleep(0.1)
                
            # If stopped early
            if self._should_stop_playback:
                sd.stop()
                print("\r🎧 Playback stopped                          ")
            else:
                # If completed normally
                print("\r🎧 Playback complete                         ")
                
            # Wait for playback to finish completely (max 0.5s)
            sd.wait(0.5)
            
            # Add a small pause after speech
            time.sleep(0.2)
            
        except Exception as e:
            error_msg = str(e)
            
            # Error reporting without fallback
            self._logger.error(f"Speech generation error: {e}", exc_info=True)
            print(f"\n❌ Speech generation error: {error_msg}")
            self.event_bus.publish("speak_error", error_msg)
            
        finally:
            self._is_speaking = False
            # Notify that TTS has finished, regardless of success or failure
            self.event_bus.publish("tts_finished", None)
    
    def stop_playback(self) -> None:
        """Stop any ongoing speech playback."""
        self._should_stop_playback = True
        
        # Stop audio playback
        sd.stop()
        
        self._logger.info("Playback stopped")
    
    def is_speaking(self) -> bool:
        """
        Check if currently speaking.
        
        Returns:
            True if currently speaking, False otherwise
        """
        return self._is_speaking

    def wait_for_playback_complete(self, timeout: float = 30.0) -> None:
        """
        Wait for current playback to complete.
        
        Args:
            timeout: Maximum seconds to wait
        """
        if not self._is_speaking:
            return
            
        start_time = time.time()
        
        while self._is_speaking and time.time() - start_time < timeout:
            time.sleep(0.1)

    def cleanup(self) -> None:
        """Clean up resources used by the speaker."""
        # Stop any ongoing playback
        self.stop_playback()
        
        # Wait for any ongoing speech to complete
        self.wait_for_playback_complete(timeout=2.0)
        
        # No explicit resource cleanup needed for CSM model
        # It will be garbage collected when the app exits
        self._logger.info("Speaker resources cleaned up") 