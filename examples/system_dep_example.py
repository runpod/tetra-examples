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
    system_dependencies=[
        "git",
        "wget",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    ],
)
def generate_image_with_stable_diffusion(
    prompt, model_id="stabilityai/stable-diffusion-2-1", num_steps=50
):
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

    # Load the Stable Diffusion pipeline (using float32 for stability)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for better numerical stability
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    # Enable memory optimizations
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    # Generate image without autocast to avoid numerical issues
    image = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images[0]

    return image


@remote(
    resource_config=sd_config,
    system_dependencies=["git", "wget", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1"]
)
def get_ffmpeg_version():
    import subprocess
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    first_line = result.stdout.splitlines()[0]
    return first_line


async def main():
    prompt = "Superman and batman fighting with spiderman"
    
    # Generate an image
    print("Generating image...")
    result_image = await generate_image_with_stable_diffusion(prompt=prompt)
    
    # Get FFmpeg version
    ffmpeg_version = await get_ffmpeg_version()
    print("Image generation completed.")
    print(f"FFmpeg version: {ffmpeg_version}")

    # Save the image directly (result_image is already a PIL Image)
    output_file = "knight.png"
    result_image.save(output_file)
    
    print(f"Image saved to {output_file}")
    print(f"Prompt: {prompt}")
    print(f"Dimensions: {result_image.size}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")