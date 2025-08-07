# # Run Stable Diffusion Image Generation with Tetra Runpod

# This example demonstrates how to run Stable Diffusion 1.5 for image generation using the Tetra Runpod framework.
# The script handles model initialization, image generation, and local file saving with proper memory management.

# Key features:
# - Memory-optimized SD 1.5 model (~4GB vs ~13GB for SDXL)
# - Automatic image saving with timestamps
# - JSON metadata export for each generation
# - CUDA memory management and cleanup
# - Attention slicing for reduced memory usage

# ## Import dependencies and configure GPU resources

import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup
from shared import shared_volume


gpu_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_16],
    name="example_stable_diffusion",
    networkVolume=shared_volume,
)


@remote(
    resource_config=gpu_config,
    dependencies=["diffusers", "torch", "transformers", "accelerate", "xformers"],
)
class SimpleSD:
    def __init__(self):
        from diffusers import StableDiffusionPipeline
        import torch
        import gc

        print("Initializing compact Stable Diffusion model...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable to save memory
            requires_safety_checker=False,
            low_cpu_mem_usage=True,  # Additional memory optimization
        )

        # Move to GPU and optimize
        self.pipe = self.pipe.to("cuda")
        # self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing()  # Additional memory saving

        # Clean up any leftover memory
        gc.collect()
        torch.cuda.empty_cache()

        print("Compact Stable Diffusion initialized successfully!")

    def generate_image(self, prompt: str, negative_prompt: str = "blurry, low quality"):
        """Generate a single image from prompt"""
        print(f"Generating image for: '{prompt}'")

        # Generate image with SD 1.5 (512x512 is native resolution)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=512,
            height=512,
        ).images[0]

        # Create output directory if it doesn't exist
        import datetime
        import os
        import json

        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save image locally with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"sd_generated_{timestamp}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        print(f"Image saved locally to: {image_path}")

        # Also convert to bytes if needed for transfer
        import io

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Create response data
        response_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image_path": image_path,
            "image_bytes": img_bytes,
            "image_size": len(img_bytes),
            "timestamp": timestamp,
            "generation_params": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
            },
            "message": "Image generated and saved locally!",
        }

        # Save response metadata to JSON file
        response_filename = f"sd_response_{timestamp}.json"
        response_path = os.path.join(output_dir, response_filename)

        # Create a copy without the bytes for JSON serialization
        response_json = response_data.copy()
        response_json["image_bytes"] = (
            f"<{len(img_bytes)} bytes>"  # Replace bytes with size info
        )

        with open(response_path, "w") as f:
            json.dump(response_json, f, indent=2)
        print(f"Response data saved to: {response_path}")

        return response_data


async def main():
    print("Testing Stable Diffusion with Simple Prompt")
    print("=" * 50)

    # Create instance (model loads once)
    print("\nLoading Stable Diffusion model...")
    sd = SimpleSD()
    print("Stable Diffusion model loaded!")

    # Single prompt generation
    print("\nGenerating image...")
    prompt = "A beautiful sunset over mountains, digital art, highly detailedand a cat"

    result = await sd.generate_image(prompt)

    # Save image locally on your machine
    import os

    local_dir = "local_images"
    os.makedirs(local_dir, exist_ok=True)

    local_image_path = os.path.join(
        local_dir, f"sd_generated_{result['timestamp']}.png"
    )
    with open(local_image_path, "wb") as f:
        f.write(result["image_bytes"])

    print(f"   Prompt: '{result['prompt']}'")
    print(f"   Result: {result['message']}")
    print(f"   Saved remotely to: {result['image_path']}")
    print(f"   Saved locally to: {local_image_path}")
    print(f"   File size: {result['image_size']:,} bytes")

    print("\n" + "=" * 50)
    print("🎉 IMAGE GENERATION COMPLETED!")
    print("🖼️  Image saved locally with timestamp!")
    print("📁 You can find your image in the current directory")


if __name__ == "__main__":
    asyncio.run(main())
