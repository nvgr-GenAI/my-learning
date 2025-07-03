# Large Language Model APIs

## Introduction

Large Language Model APIs provide accessible interfaces to powerful AI capabilities without requiring extensive infrastructure or model training expertise. This guide covers the major LLM API providers, integration patterns, and best practices.

## Major API Providers

### OpenAI API

#### Overview

OpenAI offers several models through their API, including GPT-3.5, GPT-4, and specialized models for embeddings and moderation.

#### Authentication and Setup

```python
import openai
import os
from typing import List, Dict, Optional

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """Create a chat completion"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return response
        else:
            return response.choices[0].message.content
            
    def embeddings(self, texts: List[str], model: str = "text-embedding-ada-002"):
        """Generate embeddings for texts"""
        response = self.client.embeddings.create(
            model=model,
            input=texts
        )
        
        return [embedding.embedding for embedding in response.data]
    
    def completion(
        self,
        prompt: str,
        model: str = "text-davinci-003",
        temperature: float = 0.7,
        max_tokens: int = 100
    ):
        """Legacy completion endpoint"""
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].text
```

#### Usage Examples

```python
# Initialize client
openai_client = OpenAIClient()

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response = openai_client.chat_completion(messages, model="gpt-4")
print(response)

# Streaming response
stream = openai_client.chat_completion(messages, stream=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Generate embeddings
texts = ["Hello world", "Machine learning is fascinating"]
embeddings = openai_client.embeddings(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Anthropic Claude API

#### Setup and Authentication

```python
import anthropic
from typing import List, Dict

class ClaudeClient:
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system: Optional[str] = None
    ):
        """Create a chat completion with Claude"""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages
        )
        
        return response.content[0].text
    
    def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000
    ):
        """Stream completion with Claude"""
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
```

#### Usage Examples

```python
# Initialize Claude client
claude_client = ClaudeClient()

# Chat with Claude
messages = [
    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
]

response = claude_client.chat_completion(
    messages,
    system="You are an expert Python programmer."
)
print(response)

# Streaming response
for chunk in claude_client.stream_completion(messages):
    print(chunk, end="")
```

### Google PaLM API

#### Setup

```python
import google.generativeai as genai
from google.generativeai import GenerativeModel

class PaLMClient:
    def __init__(self, api_key: Optional[str] = None):
        genai.configure(api_key=api_key or os.getenv('GOOGLE_API_KEY'))
        self.model = GenerativeModel('gemini-pro')
        
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1000
    ):
        """Generate text with PaLM"""
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    def chat_completion(self, messages: List[Dict[str, str]]):
        """Chat completion with conversation history"""
        chat = self.model.start_chat(history=[])
        
        for message in messages[:-1]:  # Add history
            if message['role'] == 'user':
                chat.send_message(message['content'])
        
        # Send final message and get response
        response = chat.send_message(messages[-1]['content'])
        return response.text
```

### Hugging Face API

#### Setup and Usage

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import requests
import json

class HuggingFaceClient:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv('HUGGINGFACE_API_TOKEN')
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
    def query_hosted_model(self, model_id: str, inputs: str):
        """Query Hugging Face hosted model"""
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        
        payload = {"inputs": inputs}
        response = requests.post(API_URL, headers=self.headers, json=payload)
        
        return response.json()
    
    def load_local_model(self, model_name: str):
        """Load model for local inference"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return generator
    
    def generate_text(self, generator, prompt: str, max_length: int = 100):
        """Generate text using local model"""
        outputs = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        return outputs[0]['generated_text']

# Usage examples
hf_client = HuggingFaceClient()

# Query hosted model
response = hf_client.query_hosted_model(
    "microsoft/DialoGPT-medium",
    "Hello, how are you?"
)
print(response)

# Use local model
generator = hf_client.load_local_model("gpt2")
text = hf_client.generate_text(generator, "The future of AI is")
print(text)
```

## API Integration Patterns

### Unified API Interface

```python
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterator

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"

@dataclass
class CompletionRequest:
    prompt: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    system_message: Optional[str] = None

@dataclass
class CompletionResponse:
    text: str
    model: str
    provider: ModelProvider
    usage: Dict[str, int]
    metadata: Dict[str, Any]

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        pass
    
    @abstractmethod
    def stream_complete(self, request: CompletionRequest) -> Iterator[str]:
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        pass

class UnifiedLLMClient:
    def __init__(self):
        self.providers = {
            ModelProvider.OPENAI: OpenAIProvider(),
            ModelProvider.ANTHROPIC: AnthropicProvider(),
            ModelProvider.GOOGLE: GoogleProvider(),
            ModelProvider.HUGGINGFACE: HuggingFaceProvider()
        }
    
    def complete(
        self, 
        prompt: str, 
        provider: ModelProvider, 
        model: str,
        **kwargs
    ) -> CompletionResponse:
        """Unified completion interface"""
        request = CompletionRequest(
            prompt=prompt,
            model=model,
            **kwargs
        )
        
        provider_client = self.providers[provider]
        return provider_client.complete(request)
    
    def compare_providers(
        self, 
        prompt: str, 
        models: Dict[ModelProvider, str]
    ) -> Dict[ModelProvider, CompletionResponse]:
        """Compare responses across providers"""
        results = {}
        
        for provider, model in models.items():
            try:
                response = self.complete(prompt, provider, model)
                results[provider] = response
            except Exception as e:
                print(f"Error with {provider}: {e}")
                
        return results

# Usage
client = UnifiedLLMClient()

# Compare responses
models = {
    ModelProvider.OPENAI: "gpt-3.5-turbo",
    ModelProvider.ANTHROPIC: "claude-3-sonnet-20240229",
    ModelProvider.GOOGLE: "gemini-pro"
}

results = client.compare_providers(
    "Explain the concept of recursion in programming",
    models
)

for provider, response in results.items():
    print(f"{provider.value}: {response.text[:100]}...")
```

### Rate Limiting and Retry Logic

```python
import time
import asyncio
from functools import wraps
from typing import Callable, Any
import random

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_call
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
            
        self.last_call = time.time()

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Decorator for exponential backoff retry logic"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        raise last_exception
                    
                    # Calculate delay with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1) * delay
                    total_delay = delay + jitter
                    
                    print(f"Attempt {attempt + 1} failed, retrying in {total_delay:.2f}s")
                    time.sleep(total_delay)
                    
            raise last_exception
        return wrapper
    return decorator

class RobustLLMClient:
    def __init__(self, provider: LLMProvider, rate_limit: int = 60):
        self.provider = provider
        self.rate_limiter = RateLimiter(rate_limit)
    
    @retry_with_exponential_backoff(max_retries=3)
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Complete with rate limiting and retries"""
        self.rate_limiter.wait_if_needed()
        return self.provider.complete(request)
    
    async def batch_complete(
        self, 
        requests: List[CompletionRequest],
        max_concurrent: int = 5
    ) -> List[CompletionResponse]:
        """Process multiple requests with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request):
            async with semaphore:
                return await asyncio.to_thread(self.complete, request)
        
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
```

### Caching and Cost Optimization

```python
import hashlib
import json
import redis
from typing import Optional
import time

class LLMCache:
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.ttl = ttl
        
    def _generate_key(self, request: CompletionRequest) -> str:
        """Generate cache key from request"""
        # Create deterministic hash of request parameters
        request_dict = {
            'prompt': request.prompt,
            'model': request.model,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'system_message': request.system_message
        }
        
        request_json = json.dumps(request_dict, sort_keys=True)
        return hashlib.sha256(request_json.encode()).hexdigest()
    
    def get(self, request: CompletionRequest) -> Optional[CompletionResponse]:
        """Get cached response"""
        key = self._generate_key(request)
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                data = json.loads(cached_data)
                return CompletionResponse(**data)
        except Exception as e:
            print(f"Cache get error: {e}")
            
        return None
    
    def set(self, request: CompletionRequest, response: CompletionResponse):
        """Cache response"""
        key = self._generate_key(request)
        
        try:
            # Convert response to dict for JSON serialization
            response_dict = {
                'text': response.text,
                'model': response.model,
                'provider': response.provider.value,
                'usage': response.usage,
                'metadata': response.metadata
            }
            
            self.redis_client.setex(
                key, 
                self.ttl, 
                json.dumps(response_dict)
            )
        except Exception as e:
            print(f"Cache set error: {e}")

class CachedLLMClient:
    def __init__(self, provider: LLMProvider, cache: LLMCache):
        self.provider = provider
        self.cache = cache
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_cost': 0.0
        }
    
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Complete with caching"""
        # Try cache first
        cached_response = self.cache.get(request)
        if cached_response:
            self.stats['cache_hits'] += 1
            return cached_response
        
        # Make API call
        response = self.provider.complete(request)
        self.stats['cache_misses'] += 1
        
        # Update cost tracking
        self.stats['total_cost'] += self.calculate_cost(request, response)
        
        # Cache the response
        self.cache.set(request, response)
        
        return response
    
    def calculate_cost(self, request: CompletionRequest, response: CompletionResponse) -> float:
        """Calculate API call cost (simplified)"""
        # Cost calculation depends on provider and model
        cost_per_token = {
            'gpt-3.5-turbo': 0.002 / 1000,  # $0.002 per 1K tokens
            'gpt-4': 0.03 / 1000,           # $0.03 per 1K tokens
            'claude-3-sonnet': 0.003 / 1000  # Example rate
        }
        
        rate = cost_per_token.get(request.model, 0.001 / 1000)
        total_tokens = response.usage.get('total_tokens', 0)
        
        return rate * total_tokens
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'total_cost': self.stats['total_cost'],
            'estimated_savings': self.stats['cache_hits'] * 0.002  # Estimated
        }
```

## Advanced Usage Patterns

### Function Calling

```python
from typing import List, Dict, Any, Callable
import json

class FunctionCall:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

class FunctionManager:
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.function_definitions: List[FunctionCall] = []
    
    def register_function(
        self, 
        name: str, 
        description: str, 
        parameters: Dict[str, Any]
    ):
        """Decorator to register functions"""
        def decorator(func: Callable) -> Callable:
            self.functions[name] = func
            self.function_definitions.append(
                FunctionCall(name, description, parameters)
            )
            return func
        return decorator
    
    def execute_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered function"""
        if name not in self.functions:
            raise ValueError(f"Function {name} not found")
        
        func = self.functions[name]
        return func(**arguments)
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get all function definitions in OpenAI format"""
        return [func.to_openai_format() for func in self.function_definitions]

# Example function registration
function_manager = FunctionManager()

@function_manager.register_function(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]
    }
)
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Mock weather function"""
    return f"The weather in {location} is 72°{unit[0].upper()}"

@function_manager.register_function(
    name="calculate",
    description="Perform mathematical calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)
def calculate(expression: str) -> float:
    """Safe calculation function"""
    try:
        # Simple evaluation (in practice, use a safer parser)
        result = eval(expression)
        return float(result)
    except:
        return "Error in calculation"

class FunctionCallingClient:
    def __init__(self, llm_client: OpenAIClient, function_manager: FunctionManager):
        self.llm_client = llm_client
        self.function_manager = function_manager
    
    def chat_with_functions(
        self, 
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo"
    ) -> str:
        """Chat with function calling capability"""
        
        # First call with function definitions
        response = self.llm_client.client.chat.completions.create(
            model=model,
            messages=messages,
            functions=self.function_manager.get_function_definitions(),
            function_call="auto"
        )
        
        response_message = response.choices[0].message
        
        # Check if the model wants to call a function
        if response_message.function_call:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)
            
            # Execute the function
            function_result = self.function_manager.execute_function(
                function_name, function_args
            )
            
            # Add function call and result to messages
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": response_message.function_call.arguments
                }
            })
            
            messages.append({
                "role": "function",
                "name": function_name,
                "content": str(function_result)
            })
            
            # Get final response
            final_response = self.llm_client.client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        return response_message.content

# Usage example
openai_client = OpenAIClient()
function_client = FunctionCallingClient(openai_client, function_manager)

messages = [
    {"role": "user", "content": "What's the weather in New York and what's 25 * 4?"}
]

response = function_client.chat_with_functions(messages)
print(response)
```

### Streaming and Real-time Processing

```python
import asyncio
from typing import AsyncIterator
import websockets
import json

class StreamingClient:
    def __init__(self, llm_client: OpenAIClient):
        self.llm_client = llm_client
    
    async def stream_completion(
        self, 
        messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        """Async streaming completion"""
        stream = self.llm_client.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def websocket_chat_server(self, websocket, path):
        """WebSocket server for real-time chat"""
        try:
            async for message in websocket:
                data = json.loads(message)
                messages = data.get('messages', [])
                
                response_chunks = []
                
                async for chunk in self.stream_completion(messages):
                    response_chunks.append(chunk)
                    await websocket.send(json.dumps({
                        'type': 'chunk',
                        'content': chunk
                    }))
                
                # Send completion signal
                await websocket.send(json.dumps({
                    'type': 'complete',
                    'full_response': ''.join(response_chunks)
                }))
                
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
    
    def start_websocket_server(self, host="localhost", port=8765):
        """Start WebSocket server"""
        return websockets.serve(self.websocket_chat_server, host, port)

# Usage
async def chat_example():
    streaming_client = StreamingClient(OpenAIClient())
    
    messages = [
        {"role": "user", "content": "Tell me a story about a robot"}
    ]
    
    print("Streaming response:")
    async for chunk in streaming_client.stream_completion(messages):
        print(chunk, end='', flush=True)
    print("\n")

# Run the example
asyncio.run(chat_example())
```

## Error Handling and Monitoring

### Comprehensive Error Handling

```python
import logging
from typing import Optional, Dict, Any
import time

class LLMError(Exception):
    """Base class for LLM-related errors"""
    pass

class RateLimitError(LLMError):
    """Rate limit exceeded error"""
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after: {retry_after}s")

class ModelNotAvailableError(LLMError):
    """Model not available error"""
    pass

class TokenLimitError(LLMError):
    """Token limit exceeded error"""
    pass

class LLMErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle LLM API errors and return whether to retry"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(f"LLM Error: {error_type} - {str(error)}", extra=context)
        
        # Handle specific error types
        if isinstance(error, RateLimitError):
            if error.retry_after:
                time.sleep(error.retry_after)
            return True  # Retry
            
        elif isinstance(error, ModelNotAvailableError):
            self.logger.error("Model not available, trying fallback model")
            return False  # Don't retry, use fallback
            
        elif isinstance(error, TokenLimitError):
            self.logger.warning("Token limit exceeded, truncating input")
            return False  # Don't retry, need to modify request
            
        else:
            return False  # Unknown error, don't retry

class MonitoredLLMClient:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.error_handler = LLMErrorHandler()
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'response_times': []
        }
    
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Complete with comprehensive monitoring"""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            response = self.provider.complete(request)
            
            # Track success metrics
            self.metrics['successful_requests'] += 1
            self.metrics['total_tokens'] += response.usage.get('total_tokens', 0)
            
            response_time = time.time() - start_time
            self.metrics['response_times'].append(response_time)
            
            return response
            
        except Exception as e:
            self.metrics['failed_requests'] += 1
            
            context = {
                'model': request.model,
                'prompt_length': len(request.prompt),
                'request_id': id(request)
            }
            
            should_retry = self.error_handler.handle_error(e, context)
            
            if should_retry:
                return self.complete(request)  # Retry once
            else:
                raise e
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        response_times = self.metrics['response_times']
        
        return {
            'total_requests': self.metrics['total_requests'],
            'success_rate': self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1),
            'avg_response_time': sum(response_times) / max(len(response_times), 1),
            'total_tokens': self.metrics['total_tokens'],
            'error_distribution': self.error_handler.error_counts
        }
```

## Best Practices

### Security

```python
import os
from cryptography.fernet import Fernet

class SecureAPIClient:
    def __init__(self):
        # Encrypt API keys
        self.encryption_key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)
        
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for storage"""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key for use"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def sanitize_prompt(self, prompt: str) -> str:
        """Remove sensitive information from prompts"""
        import re
        
        # Remove email addresses
        prompt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', prompt)
        
        # Remove phone numbers
        prompt = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', prompt)
        
        # Remove credit card numbers (basic pattern)
        prompt = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', prompt)
        
        return prompt
```

### Cost Management

```python
class CostManager:
    def __init__(self, monthly_budget: float):
        self.monthly_budget = monthly_budget
        self.current_month_cost = 0.0
        self.daily_costs = {}
        
    def check_budget(self, estimated_cost: float) -> bool:
        """Check if request is within budget"""
        projected_cost = self.current_month_cost + estimated_cost
        return projected_cost <= self.monthly_budget
    
    def estimate_cost(self, request: CompletionRequest) -> float:
        """Estimate cost for a request"""
        # Token estimation (simplified)
        estimated_tokens = len(request.prompt.split()) * 1.3  # Rough estimate
        
        cost_per_token = {
            'gpt-3.5-turbo': 0.002 / 1000,
            'gpt-4': 0.03 / 1000,
            'claude-3-sonnet': 0.003 / 1000
        }
        
        rate = cost_per_token.get(request.model, 0.001 / 1000)
        return estimated_tokens * rate
    
    def track_usage(self, response: CompletionResponse, actual_cost: float):
        """Track actual usage and cost"""
        today = time.strftime('%Y-%m-%d')
        
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0
            
        self.daily_costs[today] += actual_cost
        self.current_month_cost += actual_cost
```

## Conclusion

LLM APIs provide powerful capabilities for integrating AI into applications. Key considerations include:

1. **Provider Selection**: Choose based on model capabilities, cost, and reliability
2. **Integration Patterns**: Use robust patterns for error handling and rate limiting
3. **Cost Optimization**: Implement caching and usage monitoring
4. **Security**: Protect API keys and sanitize inputs
5. **Monitoring**: Track performance and usage metrics

Success with LLM APIs requires careful planning, robust implementation, and continuous monitoring to ensure reliable, cost-effective operation.

## Further Reading

- OpenAI API Documentation
- Anthropic Claude API Guide
- Google AI Platform Documentation
- Hugging Face API Documentation
- "Building LLM Applications" best practices guides
