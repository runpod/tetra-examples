import asyncio
from dotenv import load_dotenv
from tetra_rp import remote, LiveServerless

# Load environment variables from .env file
load_dotenv()

# Configuration for a simple resource
simple_config = LiveServerless(
    name="example_hello_world",
)


@remote(simple_config)
def hello_world():
    print("Hello from the remote function!")
    return "hello world"


async def main():
    print("\nCalling hello_world function...")
    result = await hello_world()
    print(f"\nResult: {result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
