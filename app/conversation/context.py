import logging
from typing import List, Optional
import torch

from app.models.generator import Segment
from app.utils.event_bus import EventBus

class ConversationContext:
    """
    Maintains context for voice generation consistency.
    
    This class manages audio segments that provide context to the CSM
    text-to-speech system, ensuring voice consistency throughout the conversation.
    """
    
    def __init__(self, 
                 initial_prompt_segment: Segment,
                 max_context_segments: int = 5,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the conversation context.
        
        Args:
            initial_prompt_segment: Initial voice prompt segment
            max_context_segments: Maximum number of segments to keep in context
            event_bus: Event bus for communication with other components
        """
        self.initial_prompt = initial_prompt_segment
        self.max_context_segments = max_context_segments
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("ConversationContext")
        
        # Generated speech segments history
        self.generated_segments: List[Segment] = []
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register handlers for context-related events."""
        self.event_bus.subscribe("get_tts_context", self.get_context_for_generation)
        self.event_bus.subscribe("speech_generated", self.add_segment)
    
    def add_segment(self, segment: Segment) -> None:
        """
        Add a speech segment to the context history.
        
        Args:
            segment: Speech segment to add
        """
        if not isinstance(segment, Segment):
            self._logger.warning("Invalid segment type provided")
            return
            
        self.generated_segments.append(segment)
        
        # Trim history if it exceeds the maximum size
        if len(self.generated_segments) > self.max_context_segments:
            self.generated_segments = self.generated_segments[-self.max_context_segments:]
    
    def get_context_for_generation(self, _: Optional[dict] = None) -> List[Segment]:
        """
        Get context segments for speech generation.
        
        Returns:
            List of context segments
        """
        # For now, only include the initial prompt to avoid KV cache issues
        # This is a temporary fix until we can properly handle multiple segments
        context = [self.initial_prompt]
        
        # Commented out to prevent KV cache errors
        # if self.generated_segments:
        #     # Get the most recent segments up to max_context_segments
        #     recent_segments = self.generated_segments[-self.max_context_segments:] if len(self.generated_segments) > self.max_context_segments else self.generated_segments
        #     context.extend(recent_segments)
        
        self._logger.debug(f"Providing {len(context)} context segments for TTS")
        return context
    
    def create_segment_from_audio(self, 
                                audio_tensor: torch.Tensor, 
                                text: str,
                                speaker_id: int) -> Segment:
        """
        Create a new segment from audio tensor and text.
        
        Args:
            audio_tensor: Audio tensor
            text: Text that was spoken
            speaker_id: ID of the speaker
            
        Returns:
            New segment
        """
        return Segment(text=text, speaker=speaker_id, audio=audio_tensor)
        
    def clear_context(self) -> None:
        """Clear the context history, keeping only the initial prompt."""
        self.generated_segments = []
        self._logger.info("Context history cleared") 