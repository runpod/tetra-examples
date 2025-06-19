


import asyncio
import base64
import io
from PIL import Image
from tetra_rp import remote, LiveServerless, GpuGroup

# Configuration for a GPU resource
sd_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_80],
    name="example_image_gen_server",
)








@remote(
    resource_config=sd_config,
    dependencies=["diffusers", "transformers", "torch", "accelerate", "safetensors"],
    system_dependencies=["git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"]
)
def generate_image_with_stable_diffusion(prompt, model_id="stabilityai/stable-diffusion-2-1", num_steps=50):
    """
    Generate an image using Stable Diffusion model.
    
    Args:
        prompt (str): Text prompt for image generation
        model_id (str): HuggingFace model identifier
        num_steps (int): Number of inference steps
    
    Returns:
        PIL.Image: Generated image
    """
    import torch
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPTextModel, CLIPTokenizer
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    # Generate image
    with torch.autocast(device):
        image = pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
    
    return image




async def main():
    # Generate an image
    print("Generating image...")
    result = await generate_image_with_stable_diffusion(
        prompt="Superman and batman fighting with spiderman",
        negative_prompt="blurry, distorted, low quality, text, watermark",
        width=768,
        height=512,
    )

    # Save the image
    img_data = base64.b64decode(result["image"])
    image = Image.open(io.BytesIO(img_data))

    # Save to file
    output_file = "knight.png"
    image.save(output_file)
    print(f"Image saved to {output_file}")
    print(f"Prompt: {result['prompt']}")
    print(f"Dimensions: {result['dimensions']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
