import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup, CpuInstanceType


gpu_config = LiveServerless(
    name="example_hello_world_gpu",
    gpus=[
        GpuGroup.AMPERE_48,
    ],
)

cpu_config = LiveServerless(
    name="example_hello_world_cpu",
    instanceIds=[
        CpuInstanceType.CPU3C_1_2,
    ],
)


@remote(gpu_config)
def hello_world_gpu():
    print("Hello from the remote GPU function!")
    return "hello world"


@remote(cpu_config)
def hello_world_cpu():
    print("Hello from the remote CPU function!")
    return "hello world"


async def main():
    print("\nCalling hello_world functions...")
    result_gpu = await hello_world_gpu()
    print(f"\nResult (GPU): {result_gpu}")

    result_cpu = await hello_world_cpu()
    print(f"\nResult (CPU): {result_cpu}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
