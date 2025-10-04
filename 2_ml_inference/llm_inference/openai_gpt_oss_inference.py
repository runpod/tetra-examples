# OpenAI GPT OSS Inference Example with Tetra

# This example demonstrates how to run text generation using OpenAI's GPT OSS model
# with the Tetra framework. It showcases:
# - Model loading with Transformers pipeline
# - Text generation from conversational messages

import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup


gpu_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_80],
    name="example_openai_gpt_oss_inference",
)


@remote(
    resource_config=gpu_config,
    dependencies=["transformers", "kernels", "torch", "accelerate"],
    system_dependencies=["build-essential"]
)
class OpenAIGPTOSSInference:
    def __init__(self):
        from transformers import pipeline

        model_id = "openai/gpt-oss-20b"

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, messages, max_new_tokens=256):
        """Generate text using the pipeline"""
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
        )
        print(outputs[0]["generated_text"][-1])
        return outputs


async def main():
    print("🚀 OpenAI GPT OSS Inference")
    print("=" * 50)

    # Create instance
    print("\n1️⃣ Creating OpenAI GPT OSS instance...")
    gpt = OpenAIGPTOSSInference()

    # Generate text
    print("\n2️⃣ Generating text:")
    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]

    outputs = await gpt.generate(messages, max_new_tokens=256)
    print(f"Output: {outputs}")

    print("\n🎉 Generation completed!")


if __name__ == "__main__":
    asyncio.run(main())
