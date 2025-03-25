"""
CSM Generator implementation for Bespoke Speaks.

This is a modified version of the original CSM generator.py file,
optimized for direct use in the Bespoke Speaks application.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from app.models.csm_model import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from app.utils.watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    def _setup_generation(self, speaker: int, context: Optional[List[Segment]] = None):
        """Set up the model for generation with the given speaker and context."""
        # Reset KV cache
        self._model.reset_caches()
        
        self.eot_token = self._model.eot_token if hasattr(self._model, 'eot_token') else 0
        self.frame_rate = getattr(self._model, 'frame_rate', 1000 / 80)  # Default to 12.5Hz

    def _prepare_input(self, text: str, speaker: int):
        """Prepare the input tensor for generation."""
        # Tokenize the text
        input_tokens, input_mask = self._tokenize_text_segment(text, speaker)
        
        # Convert to batch format
        input_tensor = input_tokens.unsqueeze(0).to(self.device)
        token_mask = input_mask.unsqueeze(0).to(self.device)
        
        # Empty context mask
        context_mask = torch.zeros(1, 0, dtype=torch.bool, device=self.device)
        
        return input_tensor, token_mask, context_mask

    def _decode_audio_tokens(self, audio_tokens):
        """Decode audio tokens to audio waveform."""
        # Use the model's audio tokenizer if available
        if hasattr(self._model, '_audio_tokenizer') and self._model._audio_tokenizer is not None:
            audio = self._model._audio_tokenizer.decode(audio_tokens.permute(0, 2, 1)).squeeze(0).squeeze(0)
        elif hasattr(self, '_audio_tokenizer') and self._audio_tokenizer is not None:
            audio = self._audio_tokenizer.decode(audio_tokens.permute(0, 2, 1)).squeeze(0).squeeze(0)
        else:
            # Fall back to simple decoding if no audio tokenizer is available
            audio = audio_tokens.float().squeeze(0)
        
        # Apply watermark
        try:
            from app.audio.watermark import watermark, CSM_1B_GH_WATERMARK
            import torchaudio
            
            # This applies an imperceptible watermark to identify audio as AI-generated.
            # Watermarking ensures transparency, dissuades misuse, and enables traceability.
            # Please be a responsible AI citizen and keep the watermarking in place.
            # If using CSM 1B in another application, use your own private key and keep it secret.
            audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
            audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        except (ImportError, AttributeError) as e:
            # If watermarking fails for any reason, log and continue
            print(f"Warning: Audio watermarking unavailable: {str(e)}")
        
        return audio

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int = 0,
        context: Optional[List[Segment]] = None,
        max_audio_length_ms: int = 5000,
        temperature: float = 0.7,
        topk: int = 50,
        batch_size: int = 1,  # Default to single token generation for reliability
        use_kv_caching: bool = True
    ) -> torch.Tensor:
        """
        Generate audio for the given text.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID to use
            context: List of past segments to condition on
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            batch_size: Number of tokens to generate per forward pass
            use_kv_caching: Whether to use KV caching
            
        Returns:
            Generated audio tensor
        """
        if not text:
            return torch.zeros(1, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Reset KV cache
            self._model.reset_caches()
            
            # Process context segments if provided
            tokens, tokens_mask = [], []
            if context:
                for segment in context:
                    segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                    tokens.append(segment_tokens)
                    tokens_mask.append(segment_tokens_mask)
            
            # Tokenize the input text
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)
            
            # Concatenate all tokens
            prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
            
            # Set up for generation
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
            
            # Calculate max sequence length
            n_frames = int(max_audio_length_ms * 12.5 / 1000)  # 12.5 frames per second
            max_generation_len = min(n_frames, 1024)  # Cap at 1024 frames
            
            generation_start = time.time()
            samples = []
            
            # Generate tokens
            for _ in range(0, max_generation_len, batch_size):
                # Generate token(s)
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break  # end of sequence token
                    
                samples.append(sample)
                
                # Update tokens for next iteration
                token_update = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                mask_update = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                
                curr_tokens = token_update
                curr_tokens_mask = mask_update
                curr_pos = curr_pos[:, -1:] + 1
            
            generation_time = time.time() - generation_start
            print(f"Generated {len(samples)} tokens in {generation_time:.2f}s ({len(samples)/generation_time:.2f} tokens/s)")
            
            # Decode audio
            if not samples:
                return torch.zeros(1, dtype=torch.float32, device=self.device)
            
            # Decode from audio tokens to continuous audio using the Generator's audio tokenizer
            try:
                # Stack the audio tokens
                audio_tokens = torch.stack(samples)  # [T, B, K]
                # Permute to [B, K, T] for the decoder
                audio_tokens = audio_tokens.permute(1, 2, 0)
                
                # Use our own audio tokenizer
                audio = self._audio_tokenizer.decode(audio_tokens).squeeze(0).squeeze(0)
            except Exception as e:
                print(f"Error in audio decoding: {e}")
                return torch.zeros(1, dtype=torch.float32, device=self.device)
            
            # Apply watermark directly
            try:
                audio = self._apply_watermark(audio)
            except Exception as e:
                print(f"Warning: Audio watermarking unavailable: {str(e)}")
            
            return audio

    def _apply_watermark(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply watermark to the generated audio."""
        try:
            import torchaudio
            
            # This is a simplified version of watermarking
            # In a real implementation, this would apply a cryptographic watermark
            # For now, we're just adding a simple marker that doesn't affect audio quality
            
            # Add very subtle high-frequency tone at regular intervals that's inaudible
            # but can be detected through frequency analysis
            sample_rate = self.sample_rate
            length = audio.shape[0]
            
            # Create very quiet high frequency markers at regular intervals
            marker_freq = 16000  # High frequency near Nyquist limit
            marker_amplitude = 0.0005  # Very quiet
            marker_interval = int(0.5 * sample_rate)  # Every half second
            
            # Create time values for the markers
            for i in range(0, length, marker_interval):
                if i + 100 < length:  # Ensure we have enough samples
                    # Add a brief high frequency marker
                    t = torch.arange(100, device=self.device) / sample_rate
                    marker = marker_amplitude * torch.sin(2 * 3.14159 * marker_freq * t)
                    audio[i:i+100] = audio[i:i+100] + marker
            
            # Ensure we don't clip
            if audio.abs().max() > 0.99:
                audio = audio / (audio.abs().max() + 1e-6) * 0.99
            
            return audio
        except Exception as e:
            print(f"Error applying watermark: {e}")
            # Return original audio if watermarking fails
            return audio


def load_csm_1b(device: str = "cuda") -> Generator:
    """
    Load CSM 1B model for speech generation.
    
    Args:
        device: Device to load the model on ("cuda" or "cpu")
        
    Returns:
        Initialized Generator instance
    """
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model)
    return generator


# Keep a global instance for reuse across the application
_csm_generator = None

def get_or_create_generator(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Generator:
    """
    Get or create a global CSM generator instance.
    
    This function ensures we only load the model once in the application
    and reuse it across different components.
    
    Args:
        device: Device to load the model on
        
    Returns:
        CSM Generator instance
    """
    global _csm_generator
    
    if _csm_generator is None:
        _csm_generator = load_csm_1b(device)
        
    return _csm_generator


def prepare_prompt_segment(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Segment:
    """
    Prepare a prompt segment for CSM voice generation.
    
    Args:
        device: Device to load audio on
        
    Returns:
        Initialized prompt segment
    """
    generator = get_or_create_generator(device)
    
    # Load default prompt from HF Hub
    prompt_filepath = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav"
    )
    
    # Load and prepare audio
    audio_tensor, sample_rate = torchaudio.load(prompt_filepath)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    
    # Standard prompt text
    prompt_text = (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    )
    
    return Segment(text=prompt_text, speaker=0, audio=audio_tensor) 