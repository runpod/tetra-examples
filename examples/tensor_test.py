import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup


# Configuration for a GPU resource
gpu_config = LiveServerless(
    gpus=[
        GpuGroup.ADA_24,
        GpuGroup.ADA_48_PRO,
        GpuGroup.ADA_80_PRO,
    ],
    name="example_tensor_test",
)


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
    print("\nRunning tensor test...")
    test_result = await run_tensor_test()
    print(f"\nTest result: {test_result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
