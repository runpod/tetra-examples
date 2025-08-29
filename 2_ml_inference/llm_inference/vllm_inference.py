# # Run Batch LLM Inference with vLLM and OPT-125M

# This example demonstrates how to run efficient batch text generation using vLLM using Tetra Runpod Framework.
# We  showcase vLLM's batching capabilities and model persistence.
# The script handles both single prompt generation and efficient batch processing.

# Key features:
# - Model persistence across multiple requests (load once, use many times)
# - Efficient batch processing for multiple prompts
# - Memory-optimized configuration for smaller GPU instances
# - Environment variable configuration for vLLM stability


import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup


gpu_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_16],
    name="example_vllm_inference",
)

@remote(
    resource_config=gpu_config,
    dependencies=["vllm"],
)
class MinimalVLLM:
    def __init__(self):
        from vllm import LLM, SamplingParams
        import os
        
        
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        
        self.llm = LLM(
            model="facebook/opt-125m",
            enforce_eager=True,  # Disable CUDA graphs
            gpu_memory_utilization=0.6,
            max_model_len=1024,
        )
        self.sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
        
        print("vLLM initialized successfully!")
    
    
    def test_generate(self):
        # Multiple test prompts
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "Once upon a time",
            "The future of AI is",
            "In the year 2030",
            "The best programming language is",
            "Climate change is",
            "My favorite hobby is"
        ]
        
        print(f"Generating for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "output": output.outputs[0].text
            })
        
        return results
    
    def generate_single(self, prompt: str):
        """Generate for a single prompt"""
        print(f"Generating for: '{prompt}'")
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        return {
            "prompt": prompt,
            "output": outputs[0].outputs[0].text
        }
    
    def generate_batch(self, prompts: list):
        """Generate for multiple prompts"""
        print(f"Batch generating for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "output": output.outputs[0].text
            })
        
        return results

async def main():
    print("🚀 Testing vLLM with MULTIPLE SEPARATE REQUESTS")
    print("=" * 60)
    
    # Create instance (model loads once - this is expensive!)
    print("\n1️⃣ Creating vLLM instance (loading model...)")
    llm = MinimalVLLM()
    
    # REQUEST 1: Single prompt
    print("\n2️⃣ REQUEST 1 - Single prompt generation:")
    result1 = await llm.generate_single("What is machine learning?")
    print(f"   Prompt: '{result1['prompt']}'")
    print(f"   Output: '{result1['output']}'")
    
    # REQUEST 2: Another single prompt (model already loaded!)
    print("\n3️⃣ REQUEST 2 - Another single prompt (fast!):")
    result2 = await llm.generate_single("How do computers work?")
    print(f"   Prompt: '{result2['prompt']}'")
    print(f"   Output: '{result2['output']}'")
    
    # REQUEST 3: Batch of prompts
    print("\n4️⃣ REQUEST 3 - Batch generation:")
    batch_prompts = [
        "The weather today is",
        "My favorite food is",
        "Technology will help"
    ]
    batch_results = await llm.generate_batch(batch_prompts)
    for i, result in enumerate(batch_results, 1):
        print(f"   {i}. '{result['prompt']}' → '{result['output']}'")
    
    # REQUEST 4: Another single prompt
    print("\n5️⃣ REQUEST 4 - Final single prompt:")
    result4 = await llm.generate_single("In conclusion,")
    print(f"   Prompt: '{result4['prompt']}'")
    print(f"   Output: '{result4['output']}'")
    
    # REQUEST 5: Original test with multiple prompts
    print("\n6️⃣ REQUEST 5 - Full test batch:")
    test_results = await llm.test_generate()
    print(f"   Generated {len(test_results)} responses in one batch")
    for i, result in enumerate(test_results[:3], 1):  # Show first 3
        print(f"   {i}. '{result['prompt']}' → '{result['output'][:50]}...'")
    print(f"   ... and {len(test_results) - 3} more")
    
    print("\n" + "=" * 60)
    print("🎉 ALL REQUESTS COMPLETED!")

if __name__ == "__main__":
    asyncio.run(main())