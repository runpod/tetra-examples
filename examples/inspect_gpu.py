import asyncio
from dotenv import load_dotenv
from tetra_rp import remote, LiveServerless

# Load environment variables from .env file
load_dotenv()

# Configuration for a GPU resource
gpu_config = LiveServerless(
    gpuIds="AMPERE_16",
    workersMax=1,
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


@remote(
    resource_config=gpu_config,
    dependencies=["torch"],
)
def run_tensor_test():
    import torch
    import time

    # Running a more advanced tensor test: large-scale 3D tensor operations

    # Create large 3D tensors on GPU
    a = torch.randn(500, 500, 500, device="cuda")
    b = torch.randn(500, 500, 500, device="cuda")

    # Start timing
    start = time.time()

    # Perform element-wise multiplication, summation, and matrix multiplication
    c = a * b
    d = torch.sum(c)
    e = torch.matmul(a.view(500, -1), b.view(-1, 500))

    torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()

    return {
        "elementwise_multiplication_completed": True,
        "summation_result": d.item(),
        "matrix_multiplication_shape": e.shape,
        "execution_time_seconds": round(end - start, 3),
    }


async def main():
    print("\nInspecting GPU...")
    gpu_info = await inspect_gpu()
    print(f"\nGPU Info: {gpu_info}")

    print("\nRunning tensor test...")
    test_result = await run_tensor_test()
    print(f"\nTest result: {test_result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
