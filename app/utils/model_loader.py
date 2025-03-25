import torch
import threading
import time
import logging
import inspect
from typing import Optional, Callable, Any, Dict, Type, Union

class ModelLoader:
    @staticmethod
    def load_model(
        load_function: Callable,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu", 
        model_type: str = "generic",
        event_bus: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified model loading with progress indication and error handling
        
        Args:
            load_function: Function that performs actual model loading
            model_name: Name/identifier of the model
            device: Device to load model on ("cuda" or "cpu")
            model_type: Type of model (for logging/display)
            event_bus: Optional event bus for publishing status
            **kwargs: Additional parameters to pass to load_function
            
        Returns:
            Dictionary containing loaded model components or None on failure
        """
        logger = logging.getLogger(f"{model_type.capitalize()}Loader") 
        
        # Show loading progress indicator
        loading_complete = False
        loading_thread = threading.Thread(
            target=ModelLoader._show_loading_progress,
            args=(model_type, lambda: loading_complete)
        )
        loading_thread.daemon = True
        loading_thread.start()
        
        try:
            print(f"\n🔄 Loading {model_type} model: {model_name}")
            
            # Check if the load_function accepts a device parameter
            sig = inspect.signature(load_function)
            if 'device' in sig.parameters:
                # Function accepts device parameter
                result = load_function(device=device, **kwargs)
            else:
                # Function doesn't accept device parameter
                result = load_function(**kwargs)
                
            loading_complete = True
            
            if loading_thread.is_alive():
                loading_thread.join(timeout=1.0)
                
            print(f"✅ {model_type.capitalize()} model loaded successfully")
            return result
            
        except Exception as e:
            loading_complete = True
            
            if loading_thread.is_alive():
                loading_thread.join(timeout=1.0)
                
            error_msg = f"Failed to load {model_type} model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"\n❌ {error_msg}")
            
            # Provide troubleshooting info based on error type
            ModelLoader._provide_troubleshooting(str(e))
            
            # Publish critical error if event bus provided
            if event_bus:
                critical = model_type.lower() in ["speech", "tts", "csm"]
                if critical:
                    event_bus.publish("critical_error", error_msg)
                    
            return None
            
    @staticmethod
    def _show_loading_progress(model_type: str, is_complete_fn: Callable[[], bool]):
        """Show loading progress indicator"""
        indicators = "|/-\\"
        i = 0
        
        while not is_complete_fn():
            print(f"\r📂 Loading {model_type} model... {indicators[i % len(indicators)]}", 
                  end="", flush=True)
            i += 1
            time.sleep(0.2)
            
    @staticmethod
    def _provide_troubleshooting(error_msg: str):
        """Provide relevant troubleshooting information based on error message"""
        if "CUDA out of memory" in error_msg:
            print("   Memory error detected. Try closing other applications to free GPU memory.")
        elif "module 'torch' has no attribute 'bfloat16'" in error_msg:
            print("   Your PyTorch version might not support bfloat16. Try upgrading PyTorch.")
        elif "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
            print("   Missing module. Ensure you've installed all requirements.")
        elif "requires Accelerate" in error_msg:
            print("   Please install accelerate: pip install 'accelerate>=0.26.0'") 