from langchain_ollama import ChatOllama
import time
import os
from typing import Optional, Generator
import requests
from functools import wraps
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class LLMConnectionError(Exception):
    """Custom exception for LLM connection issues."""
    pass


def retry_on_connection_error(max_retries: int = 3, delay: int = 2):
    """Decorator to retry on connection errors."""

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise LLMConnectionError(
                            f"Failed to connect after {max_retries} attempts: {str(e)}"
                        )
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


def check_ollama_status(host: str = "http://localhost:11434", model: str = "llama2") -> bool:
    """Check if Ollama service is available and model exists."""
    try:
        print(f"Checking Ollama service status at {host}...")
        # Check service status
        response = requests.get(f"{host}/api/version", timeout=5)
        if response.status_code != 200:
            print(f"❌ Ollama service returned status code: {response.status_code}")
            return False
        print("✅ Ollama service is running")

        print(f"Checking if model '{model}' exists...")
        # Check if model exists and pull if needed
        try:
            response = requests.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": "test"},
                timeout=5
            )
            print(f"✅ Model '{model}' is available")
            return True
        except requests.RequestException as model_error:
            error_msg = str(model_error).lower()
            if "model not found" in error_msg:
                print(f"⚠️ Model '{model}' not found. Attempting to pull the model...")
                try:
                    # Pull the model
                    pull_response = requests.post(
                        f"{host}/api/pull",
                        json={"name": model},
                        timeout=300
                    )
                    
                    if pull_response.status_code == 200:
                        # Wait a moment for the model to be ready
                        time.sleep(5)  # Initial wait after pull
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                # Verify model after pulling
                                verify_response = requests.post(
                                    f"{host}/api/generate",
                                    json={"model": model, "prompt": "test"},
                                    timeout=5
                                )
                                if verify_response.status_code == 200:
                                    print(f"✅ Successfully pulled and verified model '{model}'")
                                    return True
                                print(f"⚠️ Model verification attempt {attempt + 1} failed, retrying...")
                                time.sleep(5)  # Wait between retries
                            except requests.RequestException as verify_error:
                                print(f"⚠️ Verification attempt {attempt + 1} failed: {str(verify_error)}")
                                if attempt < max_retries - 1:
                                    time.sleep(5)  # Wait before next retry
                                continue
                        print(f"❌ Model verification failed after {max_retries} attempts")
                        return False
                    else:
                        print(f"❌ Failed to pull model '{model}'")
                        return False
                except requests.RequestException as pull_error:
                    print(f"❌ Error pulling model: {str(pull_error)}")
                    return False
            print(f"❌ Model error: {error_msg}")
            return False

    except requests.ConnectionError as e:
        print(f"❌ Connection Error: {str(e)}. Is Ollama running at {host}?")
        return False
    except requests.Timeout:
        print(f"❌ Timeout connecting to Ollama at {host}. Service may be busy or unreachable.")
        return False
    except requests.RequestException as e:
        print(f"❌ Error connecting to Ollama: {str(e)}")
        return False


@retry_on_connection_error(max_retries=3, delay=2)
def get_local_llm(host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                   model: str = os.getenv("OLLAMA_MODEL", "llama2"),
                   temperature: float = 0,
                   timeout: int = 30,
                   streaming: bool = True) -> Optional[ChatOllama]:
    """
    Get local LLM instance with error handling and connection checks.
    
    Args:
        host: Ollama API host URL
        model: Model to use (default: llama2:latest)
        temperature: Sampling temperature (default: 0)
        timeout: Request timeout in seconds (default: 30)
        
    Returns:
        ChatOllama instance or None if connection fails
        
    Raises:
        LLMConnectionError: If connection fails after retries
    """
    try:
        if not check_ollama_status(host, model):
            raise LLMConnectionError(
                f"Ollama service is not running or not accessible at {host} or model '{model}' not found. "
                "Please ensure Ollama is running and accessible and the model exists."
            )

        print(f"Initializing Ollama with model {model} at {host}")
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        llm = ChatOllama(
            base_url=host,
            model=model,
            temperature=temperature,
            timeout=timeout,
            streaming=streaming,
            callbacks=callbacks
        )

        try:
            print("Testing Ollama connection...")
            time.sleep(2)  # Give a moment for the model to be fully loaded
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Use a generator to test streaming
                    response = llm.invoke("test")
                    print("Successfully connected to Ollama!")
                    return llm
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "connection refused" in error_msg:
                        raise LLMConnectionError(
                            f"Cannot connect to Ollama at {host}. "
                            "Please ensure Ollama is running and the port is accessible."
                        )
                    elif "model not found" in error_msg:
                        print(f"⚠️ Connection test attempt {attempt + 1} failed: Model not found")
                        # Only try to pull on first attempt
                        if attempt == 0:
                            print(f"Attempting to pull model '{model}'...")
                            if check_ollama_status(host, model):
                                time.sleep(5)  # Wait for model to be ready
                                continue
                        elif attempt < max_retries - 1:
                            time.sleep(5)  # Wait before next retry
                            continue
                    else:
                        print(f"⚠️ Connection test attempt {attempt + 1} failed: {error_msg}")
                        if attempt < max_retries - 1:
                            time.sleep(5)  # Wait before next retry
                            continue
            
            # If we get here, all retries failed
            raise LLMConnectionError(
                f"Failed to establish connection after {max_retries} attempts. "
                f"Last error: {str(last_error)}"
            )

        except Exception as e:
            error_msg = f"Failed to initialize Ollama LLM at {host}: {str(e)}"
            print(error_msg)
            raise LLMConnectionError(error_msg)

    except LLMConnectionError:
        raise
    except Exception as e:
        error_msg = f"Failed to initialize Ollama LLM at {host}: {str(e)}"
        print(error_msg)
        raise LLMConnectionError(error_msg)


# Test the connection
if __name__ == "__main__":
    try:
        print("Testing Ollama connection...")
        if check_ollama_status():
            llm = get_local_llm()
            print("Successfully connected to Ollama!")
        else:
            print(
                "Failed to connect to Ollama. Please ensure the service is running."
            )
    except LLMConnectionError as e:
        print(f"Connection Error: {e}")