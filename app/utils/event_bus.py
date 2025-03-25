from typing import Callable, Dict, List, Any
import logging

class EventBus:
    """
    Event bus for communication between components.
    
    This class implements a simple event bus pattern to allow loose coupling
    between components. Components can subscribe to events and publish events
    without direct knowledge of each other.
    """
    
    def __init__(self):
        """Initialize an empty event bus."""
        self._subscribers: Dict[str, List[Callable]] = {}
        self._logger = logging.getLogger("EventBus")
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_type: The type of event to subscribe to
            callback: The callback function to be called when the event is published
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        self._logger.debug(f"Subscribed to event: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from an event.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: The callback function to remove
        """
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            self._logger.debug(f"Unsubscribed from event: {event_type}")
    
    def publish(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event.
        
        Args:
            event_type: The type of event to publish
            data: The data to pass to subscribers
        """
        if event_type not in self._subscribers:
            return
        
        for callback in self._subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                self._logger.error(f"Error in event handler for {event_type}: {e}")
    
    def clear_all_subscribers(self) -> None:
        """Remove all subscribers from the event bus."""
        self._subscribers.clear()
        self._logger.debug("Cleared all subscribers")


# Global event bus instance
global_event_bus = EventBus() 