import torch
import logging
import os
import threading
import importlib.util
from typing import List, Dict, Any, Optional, Callable

from transformers import AutoModelForCausalLM, AutoTokenizer
from app.utils.event_bus import EventBus
from app.utils.model_loader import ModelLoader

# Check if accelerate is available
has_accelerate = importlib.util.find_spec("accelerate") is not None

class LLMService:
    """
    Service for generating text responses using an LLM model.
    
    This class handles loading and managing LLM models for generating
    responses to user inputs, with support for conversation history.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the LLM model to use
            device: Device to run the model on ("cuda" or "cpu")
            event_bus: Event bus for communication with other components
        """
        self.model_name = model_name
        self.device = device
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("LLMService")
        
        # Lazily loaded models
        self._tokenizer = None
        self._model = None
        
        # Generation settings
        self.max_new_tokens = 256
        self.temperature = 0.7
        self.top_p = 0.9
        
        # Response generation state
        self._is_generating = False
        self._generation_thread: Optional[threading.Thread] = None
        
        # Subscribe to events
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register all event handlers."""
        # Only subscribe to generate_response events
        # Remove transcription_complete subscription to avoid double-triggering
        self.event_bus.subscribe("generate_response", self.handle_generate_response)
        
        self._logger.info("LLMService subscribed to events")
    
    def _load_models(self) -> bool:
        """
        Lazy-load the LLM model and tokenizer if not already loaded.
        
        Returns:
            True if models were loaded successfully, False otherwise
        """
        if self._model is not None and self._tokenizer is not None:
            return True
            
        self._logger.info(f"Loading LLM model {self.model_name}...")
        
        # Define model loading function
        def load_llm_models():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Different loading approaches depending on accelerate availability
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if has_accelerate:
                model_kwargs["device_map"] = "auto"
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # If accelerate is not available, move model to device manually
            if not has_accelerate:
                model = model.to(self.device)
                
            return {
                "model": model,
                "tokenizer": tokenizer
            }
        
        # Use the common model loader utility
        result = ModelLoader.load_model(
            load_function=load_llm_models,
            model_name=self.model_name.split('/')[-1],
            device=self.device,
            model_type="llm",
            event_bus=self.event_bus
        )
        
        if not result:
            return False
            
        self._model = result["model"]
        self._tokenizer = result["tokenizer"]
        return True
    
    def handle_generate_response(self, conversation_history: List[Dict[str, str]]) -> None:
        """
        Handle generate_response events from the conversation manager.
        
        Args:
            conversation_history: Complete conversation history
        """
        if not conversation_history:
            self._logger.warning("Empty conversation history received")
            return
            
        # Generate response asynchronously
        self._generation_thread = threading.Thread(
            target=self._generate_response_from_history,
            args=(conversation_history,)
        )
        self._generation_thread.daemon = True
        self._generation_thread.start()

    def _generate_response_from_history(self, conversation_history: List[Dict[str, str]]) -> None:
        """
        Generate a response from conversation history and publish it.
        
        Args:
            conversation_history: Full conversation history
        """
        self._is_generating = True
        
        try:
            # Load models if not already loaded
            if not self._load_models():
                self._logger.error("Failed to load LLM models")
                print("\n❌ Failed to generate response: model not loaded properly")
                self._is_generating = False
                return
            
            # Format messages for model
            prompt = self._tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Print short message about generation
            self._logger.info("Generating response from conversation history")
            print(f"\n🧠 Generating response...")
            
            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self._tokenizer.pad_token_id if self._tokenizer.pad_token_id else self._tokenizer.eos_token_id
                )
            
            # Decode only the newly generated tokens
            response = self._tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Handle empty responses
            if not response or response.isspace():
                response = "I'm not sure how to respond to that. Could you rephrase your question?"
                self._logger.warning("Generated empty response")
                print("\n⚠️ Generated empty response, using fallback")
            
            # Log the response
            self._logger.info(f"Generated response: {response}")
            
            # Publish response event to trigger next steps (e.g. text-to-speech)
            self.event_bus.publish("response_ready", response)
            
        except Exception as e:
            self._logger.error(f"Error generating response: {e}", exc_info=True)
            print(f"\n❌ Error generating response: {e}")
            
            # Publish fallback error response
            fallback = "I apologize, but I encountered an issue processing your request. Could we try again?"
            self.event_bus.publish("response_ready", fallback)
        finally:
            self._is_generating = False
    
    def is_generating(self) -> bool:
        """
        Check if currently generating a response.
        
        Returns:
            True if currently generating, False otherwise
        """
        return self._is_generating 