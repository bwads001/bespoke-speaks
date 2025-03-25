import torch
import torchaudio
import logging
from typing import Dict, Any, Optional, Callable
import os

from huggingface_hub import hf_hub_download
from app.models.generator import Segment

class PromptManager:
    """
    Manages voice prompts for CSM text-to-speech.
    
    This class handles loading and management of voice prompts,
    which are used as context for the CSM model to maintain
    consistent voice characteristics.
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the prompt manager.
        
        Args:
            sample_rate: Target sample rate for prompts
        """
        self.sample_rate = sample_rate
        self._logger = logging.getLogger("PromptManager")
        self.prompts = self._load_available_prompts()
    
    def _load_available_prompts(self) -> Dict[str, Dict[str, Any]]:
        """
        Load information about available prompts.
        
        Returns:
            Dictionary of prompt information
        """
        return {
            "default": {
                "text": (
                    "like revising for an exam I'd have to try and like keep up the momentum because I'd "
                    "start really early I'd be like okay I'm gonna start revising now and then like "
                    "you're revising for ages and then I just like start losing steam I didn't do that "
                    "for the exam we had recently to be fair that was a more of a last minute scenario "
                    "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
                    "sort of start the day with this not like a panic but like a"
                ),
                "audio_path": "prompts/conversational_a.wav",
                "repo_id": "sesame/csm-1b"
            },
            # Additional prompts can be defined here
        }
    
    def get_prompt_segment(self, prompt_name: str = "default") -> Segment:
        """
        Load and prepare a prompt segment for a specified voice.
        
        Args:
            prompt_name: Name of the prompt to load
            
        Returns:
            Segment containing the prompt audio and text
        """
        if prompt_name not in self.prompts:
            self._logger.warning(f"Prompt '{prompt_name}' not found, using default")
            prompt_name = "default"
            
        prompt_info = self.prompts[prompt_name]
        
        try:
            # Download the prompt audio file from the Hugging Face Hub
            audio_path = hf_hub_download(
                repo_id=prompt_info["repo_id"],
                filename=prompt_info["audio_path"]
            )
            
            # Load the audio and convert to the right format
            audio_tensor = self._load_audio(audio_path)
            
            # Create and return the segment
            return Segment(text=prompt_info["text"], speaker=0, audio=audio_tensor)
            
        except Exception as e:
            self._logger.error(f"Error loading prompt '{prompt_name}': {e}")
            raise
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio from file and prepare for use with CSM.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Audio tensor
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Load audio
        audio_tensor, file_sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            
        # Extract the mono channel
        audio_tensor = audio_tensor.squeeze(0)
        
        # Resample if needed
        if file_sample_rate != self.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=file_sample_rate, 
                new_freq=self.sample_rate
            )
            
        return audio_tensor
    
    def list_available_prompts(self) -> Dict[str, str]:
        """
        Get a list of available prompts.
        
        Returns:
            Dictionary of prompt names and descriptions
        """
        return {name: info.get("description", "No description") 
                for name, info in self.prompts.items()} 