import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup, NetworkVolume

network_vol = NetworkVolume(
    name="openai_shared_storage",
    size=100
)




gpu_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_80],
    name="openai_gpt_oss_inference_5",
    networkVolume=network_vol,
    workersMax=1
)





@remote(
    resource_config=gpu_config,
    dependencies=["transformers", "kernels", "torch", "accelerate"],
)

class OpenAIGPTOSSInference:
    
    def __init__(self):
        from transformers import pipeline
        import torch
        
        
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
    print("🚀 Testing OpenAI GPT OSS Inference")
    print("=" * 50)
    
    # Create instance
    print("\n1️⃣ Creating OpenAI GPT OSS instance...")
    gpt = OpenAIGPTOSSInference()
    
    # Test generation
    print("\n2️⃣ Testing text generation:")
    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]
    
    outputs = await gpt.generate(messages, max_new_tokens=256)
    print(f"Output: {outputs}")
    
    print("\n🎉 Generation completed!")


if __name__ == "__main__":
    asyncio.run(main())
    