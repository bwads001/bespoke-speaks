import torch
import torchaudio
import json
import time
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from app.models.generator import Segment
from app.utils.event_bus import EventBus

class ConversationSaver:
    """
    Saves conversation audio and metadata to disk.
    
    This class provides functionality to save conversation audio
    and metadata for later playback or analysis.
    """
    
    def __init__(self, 
                 sample_rate: int = 24000,
                 output_dir: str = "conversations",
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the conversation saver.
        
        Args:
            sample_rate: Audio sample rate
            output_dir: Directory to save conversations to
            event_bus: Event bus for communication with other components
        """
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("ConversationSaver")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def save_audio(self, 
                  segments: List[Segment], 
                  filename: Optional[str] = None) -> str:
        """
        Save segments as a WAV file with timestamps and speaker annotations.
        
        Args:
            segments: List of audio segments to save
            filename: Optional filename to save to
            
        Returns:
            Path to the saved WAV file
        """
        if not segments:
            self._logger.warning("No segments provided to save")
            return ""
            
        # Generate filename if not provided
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"conversation_{timestamp}.wav"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
            
        try:
            # Concatenate audio segments
            all_audio = torch.cat([seg.audio for seg in segments], dim=0)
            
            # Save as WAV
            torchaudio.save(
                filepath,
                all_audio.unsqueeze(0).cpu(),
                self.sample_rate
            )
            
            # Save metadata alongside audio
            self._save_metadata(segments, filepath.replace('.wav', '.json'))
            
            self._logger.info(f"Audio saved to {filepath}")
            return filepath
            
        except Exception as e:
            self._logger.error(f"Error saving audio: {e}")
            return ""
    
    def _save_metadata(self, segments: List[Segment], filepath: str) -> None:
        """
        Save metadata for the conversation segments.
        
        Args:
            segments: List of segments
            filepath: Path to save metadata to
        """
        metadata = []
        
        # Calculate cumulative durations
        current_pos = 0
        
        for i, segment in enumerate(segments):
            # Calculate duration in seconds
            duration = segment.audio.shape[0] / self.sample_rate
            
            metadata.append({
                "index": i,
                "start_time": current_pos,
                "end_time": current_pos + duration,
                "duration": duration,
                "speaker": segment.speaker,
                "text": segment.text
            })
            
            current_pos += duration
        
        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        self._logger.debug(f"Metadata saved to {filepath}")
    
    def save_conversation_history(self, 
                                 history: List[Dict[str, Any]], 
                                 filename: Optional[str] = None) -> str:
        """
        Save conversation history to a JSON file.
        
        Args:
            history: Conversation history
            filename: Optional filename to save to
            
        Returns:
            Path to the saved JSON file
        """
        if not history:
            self._logger.warning("No conversation history provided to save")
            return ""
            
        # Generate filename if not provided
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"conversation_history_{timestamp}.json"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
            
        try:
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
                
            self._logger.info(f"Conversation history saved to {filepath}")
            return filepath
            
        except Exception as e:
            self._logger.error(f"Error saving conversation history: {e}")
            return "" 