import asyncio
from tetra_rp import remote, LiveServerless, CpuLiveServerless


gpu_config = LiveServerless(
    name="example_hello_world_gpu"
)

cpu_config = CpuLiveServerless(
    name="example_hello_world_cpu"
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
    await asyncio.gather(
        hello_world_gpu(),
        hello_world_cpu(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
