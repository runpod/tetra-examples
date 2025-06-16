import asyncio
from dotenv import load_dotenv
from tetra_rp import remote, LiveServerless, GpuGroup

# Load environment variables from .env file
load_dotenv()

# Configuration for a GPU resource
gpu_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_16],
    name="example_inspect_gpu",
)


@remote(
    resource_config=gpu_config,
    dependencies=["torch"],
)
def inspect_gpu():
    import torch

    if torch.cuda.is_available():
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            ),
            "memory_free_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "cuda_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
        }
    else:
        raise Exception("No CUDA-compatible GPU found.")


async def main():
    print("\nInspecting GPU...")
    gpu_info = await inspect_gpu()
    print(f"\nGPU Info: {gpu_info}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
