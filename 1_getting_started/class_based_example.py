import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup

gpu_config = LiveServerless(
    gpus=[GpuGroup.AMPERE_16],
    name="simple_example",
)

@remote(resource_config=gpu_config)
class SimpleTextGen:
    def __init__(self):
        print("Model loaded!")
    
    def generate(self, text: str):
        return f"Generated: {text}"

async def main():
    generator = SimpleTextGen()
    result = await generator.generate("Hello World")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())