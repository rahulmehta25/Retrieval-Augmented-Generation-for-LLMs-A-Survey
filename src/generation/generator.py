import abc
import json
import requests
from typing import List, Dict, Any, Iterator
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class Generator(abc.ABC):
    """
    Abstract base class for LLM generation.
    """
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text based on the given prompt.
        """
        pass

    @abc.abstractmethod
    def stream_generate(self, prompt: str, **kwargs):
        """
        Generates text in a streaming fashion.
        """
        pass

class HuggingFaceGenerator(Generator):
    """
    Generator using HuggingFace models (e.g., local models).
    """
    def __init__(self, model_name: str = 'distilgpt2', device: str = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, **kwargs) -> str:
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        try:
            # The pipeline returns a list of dictionaries, take the generated_text from the first one
            result = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True, # Enable sampling for temperature to have effect
                pad_token_id=self.tokenizer.eos_token_id, # Handle padding for batching
                **kwargs
            )[0]['generated_text']

            # The pipeline returns the prompt concatenated with the generated text.
            # We need to extract only the new generated part.
            generated_text = result[len(prompt):].strip()
            
            # If no text was generated, return a fallback response
            if not generated_text:
                return "I don't have enough information to answer this question."
            
            # Check for repetitive patterns that indicate the model is stuck
            if generated_text.count(generated_text[:20]) > 3:
                return "I don't have enough information to answer this question."
                
            return generated_text
            
        except Exception as e:
            # Return a fallback response if generation fails
            return f"I encountered an error while generating the answer: {str(e)}"

    def stream_generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, **kwargs):
        # HuggingFace pipeline does not directly support streaming in a simple way for all models.
        # For true streaming, you would typically use the model's generate method with `stream=True`
        # or integrate with a framework like FastAPI for SSE.
        # This is a simplified placeholder.
        full_response = self.generate(prompt, max_new_tokens, temperature, **kwargs)
        yield full_response # Yield the full response as a single chunk for simplicity

class OpenAIGenerator(Generator):
    """
    Generator using OpenAI API.
    """
    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model_name = model_name
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

    def generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def stream_generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, **kwargs):
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class OllamaGenerator(Generator):
    """
    Generator using Ollama API for local LLM inference.
    
    Ollama provides a local HTTP API for running various open-source models
    like Gemma, Llama, Mistral, etc. on your local machine.
    """
    
    def __init__(self, model_name: str = 'gemma:2b', host: str = 'localhost', port: int = 11434):
        """
        Initialize Ollama generator.
        
        Args:
            model_name: Name of the Ollama model (e.g., 'gemma:2b', 'llama2', 'mistral')
            host: Ollama server host (default: localhost)
            port: Ollama server port (default: 11434)
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
        # Test connection to Ollama server
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running and accessible. Error: {str(e)}"
            )
    
    def _check_model_availability(self):
        """Check if the specified model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            
            if self.model_name not in available_models:
                raise ValueError(
                    f"Model '{self.model_name}' not found. Available models: {available_models}. "
                    f"Run 'ollama pull {self.model_name}' to download the model."
                )
        except requests.exceptions.RequestException as e:
            # If we can't check models, we'll try anyway and let the generate call fail
            pass
    
    def generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional parameters for Ollama API
            
        Returns:
            Generated text string
            
        Raises:
            ConnectionError: If Ollama server is not accessible
            ValueError: If model is not available
            RuntimeError: If generation fails
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        
        self._check_model_availability()
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=120  # Longer timeout for generation
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            # Handle empty or invalid responses
            if not generated_text:
                return "I don't have enough information to answer this question."
            
            # Check for repetitive patterns that indicate the model is stuck
            if len(generated_text) > 20 and generated_text.count(generated_text[:20]) > 3:
                return "I don't have enough information to answer this question."
            
            return generated_text
            
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again with a shorter prompt or lower max_new_tokens."
        except requests.exceptions.RequestException as e:
            return f"I encountered an error while generating the answer: {str(e)}"
        except json.JSONDecodeError:
            return "I received an invalid response from the model. Please try again."
        except Exception as e:
            return f"An unexpected error occurred during generation: {str(e)}"
    
    def stream_generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, **kwargs) -> Iterator[str]:
        """
        Generate text in streaming mode using Ollama API.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional parameters for Ollama API
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            ConnectionError: If Ollama server is not accessible
            ValueError: If model is not available
            RuntimeError: If generation fails
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        
        self._check_model_availability()
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line)
                        chunk_text = chunk_data.get('response', '')
                        
                        if chunk_text:
                            yield chunk_text
                        
                        # Check if generation is done
                        if chunk_data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue
                        
        except requests.exceptions.Timeout:
            yield "The request timed out. Please try again with a shorter prompt or lower max_new_tokens."
        except requests.exceptions.RequestException as e:
            yield f"I encountered an error while generating the answer: {str(e)}"
        except Exception as e:
            yield f"An unexpected error occurred during generation: {str(e)}"

class PromptTemplate:
    """
    Manages prompt templates for RAG.
    """
    def __init__(self, template: str = None):
        if template is None:
            # Use a simpler template that works better with smaller models
            self.template = """Based on this information: {context}

Question: {question}

Answer:"""
        else:
            self.template = template

    def format_prompt(self, question: str, context: List[str]) -> str:
        context_str = "\n".join(context)
        return self.template.format(context=context_str, question=question) 