"""
LLM Text Generation Service

High-performance text generation service using OpenAI GPT OSS model on A100 GPUs.
Handles text generation with proper error handling and response formatting.
"""

from tetra_rp import remote
from config import get_llm_config


@remote(
    resource_config=get_llm_config(),
    dependencies=["transformers", "kernels", "torch", "accelerate"],
    hf_models_to_cache=["openai/gpt-oss-20b"],
    system_dependencies=["build-essential"]
)
class LLMTextGenerator:
    """LLM service for generating text content using OpenAI GPT OSS model."""
    
    def __init__(self) -> None:
        """Initialize the LLM text generation pipeline."""
        from transformers import pipeline
        
        model_id = "openai/gpt-oss-20b"
        
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        
        print("LLM Text Generator initialized on A100")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 256):
        """
        Generate text content from a given prompt.
        
        Args:
            prompt: Input text prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing generated text and metadata
        """
        print(f"Generating text for: '{prompt[:50]}...'")
        
        try:
            messages = [
                {"role": "system", "content": "You are an educator and motivational speaker."},
                {"role": "user", "content": prompt}
            ]
            
            outputs = self.pipe(messages, max_new_tokens=max_new_tokens)
            raw_response = outputs[0]["generated_text"][-1]
            
            generated_text = self._extract_content(raw_response)
            word_count = len(generated_text.split())
            
            print(f"Generated text: {generated_text[:100]}...")
            
            return {
                "original_prompt": prompt,
                "generated_text": generated_text,
                "text_length": len(generated_text),
                "word_count": word_count,
                "service": "llm_generator",
                "gpu_type": "A100",
                "success": True
            }
            
        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            return {
                "error": str(e),
                "original_prompt": prompt,
                "service": "llm_generator",
                "success": False
            }
    
    def _extract_content(self, raw_response) -> str:
        """Extract clean content from model response."""
        if isinstance(raw_response, dict) and 'content' in raw_response:
            content = raw_response['content']
            if "assistantfinal" in content:
                return content.split("assistantfinal", 1)[-1].strip()
            return content
        elif isinstance(raw_response, str):
            if "assistantfinal" in raw_response:
                return raw_response.split("assistantfinal", 1)[-1].strip()
            return raw_response
        else:
            return str(raw_response)