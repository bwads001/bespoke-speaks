import logging
import time
from typing import List, Dict, Any, Optional
import json
import os

from app.utils.event_bus import EventBus

class ConversationManager:
    """
    Manages conversation state and history.
    
    This class keeps track of the conversation history, handles
    turn-taking, and provides the conversation context to other components.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the conversation manager.
        
        Args:
            event_bus: Event bus for communication with other components
        """
        self.event_bus = event_bus or EventBus()
        self._logger = logging.getLogger("ConversationManager")
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initial system message
        self._system_message = {
            "role": "system", 
            "content": "You are a helpful AI assistant speaking with a user. Be concise and friendly in your responses."
        }
        
        # Internal state
        self._current_speaker: Optional[str] = None
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register handlers for conversation events."""
        self.event_bus.subscribe("user_message", self.handle_user_message)
        self.event_bus.subscribe("ai_message", self.handle_ai_message)
        self.event_bus.subscribe("get_conversation_history", self.get_chat_history)
        # Add direct subscription to transcription events
        self.event_bus.subscribe("transcription_complete", self.handle_transcription)
        self._logger.info("ConversationManager subscribed to events")
    
    def handle_user_message(self, message: str) -> None:
        """
        Handle a user message.
        
        Args:
            message: The user's message
        """
        if not message:
            self._logger.warning("Empty user message received")
            return
            
        self._logger.debug(f"User: {message}")
        
        # Add to conversation history
        self.add_message("user", message)
        self._current_speaker = "user"
    
    def handle_ai_message(self, message: str) -> None:
        """
        Handle an AI message.
        
        Args:
            message: The AI's message
        """
        if not message:
            self._logger.warning("Empty AI message received")
            return
            
        self._logger.debug(f"AI: {message}")
        
        # Add to conversation history
        self.add_message("assistant", message)
        self._current_speaker = "assistant"
        
        # Trigger text-to-speech
        self.event_bus.publish("speak_response", message)
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the speaker ("system", "user", or "assistant")
            content: The content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def get_chat_history(self, _: Optional[Any] = None) -> List[Dict[str, str]]:
        """
        Get the conversation history in chat format.
        
        Returns:
            List of messages in chat format
        """
        formatted_history = [self._system_message]
        
        for message in self.conversation_history:
            formatted_history.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        return formatted_history
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        self._logger.info("Conversation history cleared")
    
    def save_conversation(self, filepath: Optional[str] = None) -> str:
        """
        Save the conversation history to a file.
        
        Args:
            filepath: Optional filepath to save to
            
        Returns:
            Path to the saved file
        """
        if not filepath:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f"conversation_{timestamp}.json"
            
        # Make directories if needed
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2)
            
        self._logger.info(f"Conversation saved to {filepath}")
        return filepath
    
    def load_conversation(self, filepath: str) -> None:
        """
        Load conversation history from a file.
        
        Args:
            filepath: Path to the file to load from
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
                
            self._logger.info(f"Conversation loaded from {filepath}")
        except Exception as e:
            self._logger.error(f"Error loading conversation: {e}")
            raise 
    
    def handle_transcription(self, text: str) -> None:
        """
        Handle transcription_complete events directly.
        
        Args:
            text: Transcribed text from user speech
        """
        if not text or text.isspace():
            self._logger.warning("Empty transcription received")
            return
            
        self._logger.info(f"Transcription received: {text}")
        
        # Add to conversation as a user message
        self.handle_user_message(text)
        
        # Request a response from the language model
        print("🤔 Generating response...")
        self.event_bus.publish("generate_response", self.get_chat_history()) 