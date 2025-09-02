"""
LLM Text Generation Service

High-performance text generation service using Qwen3-4B-Instruct model on A100 GPUs.
Handles text generation with proper error handling and response formatting.
"""

from tetra_rp import remote
from config import get_llm_config


@remote(
    resource_config=get_llm_config(),
    dependencies=["transformers", "kernels", "torch", "accelerate"],
    system_dependencies=["build-essential"],
)
class LLMTextGenerator:
    """LLM service for generating text content using Qwen3-4B-Instruct model."""

    def __init__(self) -> None:
        """Initialize the LLM text generation with Qwen model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen3-4B-Instruct-2507"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        print("LLM Text Generator initialized with Qwen3-4B-Instruct on A100")

    def generate_text(self, prompt: str, max_new_tokens: int = 50):
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
                {
                    "role": "system",
                    "content": "You are an educator and motivational speaker. Answer just in 50 tokens.",
                },
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )

            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            word_count = len(generated_text.split())

            print(f"Generated text: {generated_text[:100]}...")

            return {
                "original_prompt": prompt,
                "generated_text": generated_text,
                "text_length": len(generated_text),
                "word_count": word_count,
                "service": "llm_generator",
                "gpu_type": "A100",
                "success": True,
            }

        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            return {
                "error": str(e),
                "original_prompt": prompt,
                "service": "llm_generator",
                "success": False,
            }
