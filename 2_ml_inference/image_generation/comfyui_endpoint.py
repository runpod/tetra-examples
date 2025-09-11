# In this example, we will show how to run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) via a serverless endpoint to serve image gen for an example workflow.
# We'll use a premade example workflow from the [Runpod ComfyUI worker template](https://github.com/runpod-workers/worker-comfyui) on NVIDIA 5090 GPUs.

import asyncio
import json
from pathlib import Path
import base64

from tetra_rp import remote, ServerlessEndpoint, GpuGroup

gpu_config = ServerlessEndpoint(
    name="comfyui_example",
    gpus=[
        GpuGroup.ADA_32_PRO,
    ],
    imageName="runpod/worker-comfyui:5.4.1-flux1-schnell",
)

# we'll just use a simple premade workflow for this example
with open(Path(__file__).parent / "comfy_example.json") as f:
    comfy_workflow = json.load(f)

@remote(gpu_config, sync=True)
def run_comfy_workflow():
    return comfy_workflow

def process_output_to_local(comfy_endpoint_response: dict):
    output = comfy_endpoint_response.get("images")
    if not output:
        raise ValueError("no images returned in comfy response")
    
    output_path = Path(__file__).parent / "comfy_output_examples/"
    output_path.mkdir(parents=True, exist_ok=True)

    for image in output:
        with open(Path(__file__).parent / "comfy_output_examples" / image["filename"], "wb") as f:
            f.write(base64.b64decode(image["data"]))

    print("wrote generated images to ", output_path)

async def main():
    print("Generating images from remote endpoint...")

    comfy_response = await run_comfy_workflow()
    process_output_to_local(comfy_response)
        

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
