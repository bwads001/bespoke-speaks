import logging
import numpy as np
from typing import Optional, Callable

from app.utils.event_bus import EventBus

class InterruptionHandler:
    """
    Handles user interruptions during AI speech.
    
    This class detects when a user interrupts the AI's speech and
    triggers appropriate actions to create a more natural conversation flow.
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.05,
                 min_duration: float = 0.3,
                 sample_rate: int = 16000,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the interruption handler.
        
        Args:
            energy_threshold: Energy threshold to consider as speech
            min_duration: Minimum duration (seconds) of speech to consider as interruption
            sample_rate: Audio sample rate
            event_bus: Event bus for communication with other components
        """
        self.energy_threshold = energy_threshold
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("InterruptionHandler")
        
        # State tracking
        self._is_speaking = False
        self._speech_buffer = []
        self._consecutive_speech_chunks = 0
        self._min_speech_chunks = int(min_duration * sample_rate / 1600)  # Assuming 1600 samples per chunk (100ms)
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register handlers for interruption-related events."""
        self.event_bus.subscribe("speech_detected", self.check_for_interruption)
        self.event_bus.subscribe("tts_started", self._handle_tts_started)
        self.event_bus.subscribe("tts_finished", self._handle_tts_finished)
    
    def _handle_tts_started(self, _: Optional[dict] = None) -> None:
        """Handle TTS started event."""
        self._is_speaking = True
        self._consecutive_speech_chunks = 0
        self._speech_buffer = []
        self._logger.debug("TTS started, monitoring for interruptions")
    
    def _handle_tts_finished(self, _: Optional[dict] = None) -> None:
        """Handle TTS finished event."""
        self._is_speaking = False
        self._logger.debug("TTS finished, stopped monitoring for interruptions")
    
    def check_for_interruption(self, audio_chunk: np.ndarray) -> None:
        """
        Check if the user is interrupting the AI's speech.
        
        Args:
            audio_chunk: Chunk of audio data
        """
        if not self._is_speaking:
            # Not currently speaking, so no interruption possible
            return
            
        # Check if this chunk has significant energy
        is_valid_speech = self._is_valid_interruption(audio_chunk)
        
        if is_valid_speech:
            # Count consecutive speech chunks
            self._consecutive_speech_chunks += 1
            self._speech_buffer.append(audio_chunk)
            
            # If we have enough consecutive speech chunks, trigger an interruption
            if self._consecutive_speech_chunks >= self._min_speech_chunks:
                self._trigger_interruption()
        else:
            # Reset counter if not speech
            self._consecutive_speech_chunks = 0
            self._speech_buffer = []
    
    def _is_valid_interruption(self, audio_chunk: np.ndarray) -> bool:
        """
        Determine if audio chunk represents valid speech for interruption.
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            True if valid speech detected, False otherwise
        """
        # Simple energy-based detection
        energy = np.mean(np.abs(audio_chunk))
        return energy > self.energy_threshold
    
    def _trigger_interruption(self) -> None:
        """Trigger an interruption event."""
        self._logger.info("User interruption detected")
        
        # Publish the interruption event
        self.event_bus.publish("user_interruption", {
            "audio_buffer": self._speech_buffer.copy() if self._speech_buffer else None
        })
        
        # Reset state
        self._consecutive_speech_chunks = 0
        self._speech_buffer = []
        self._is_speaking = False 