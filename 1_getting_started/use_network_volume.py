import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup, NetworkVolume

# Option 1: Create a NetworkVolume object
network_vol = NetworkVolume(
    name="my_shared_storage",
    size=20  # 20GB
)

gpu_config_with_volume = LiveServerless(
    gpus=[GpuGroup.AMPERE_16],
    name="example_use_network_volume",
    networkVolume=network_vol
)

# Option 2: Use existing network volume ID directly
# gpu_config_with_volume = LiveServerless(
#     gpus=[GpuGroup.AMPERE_16],
#     name="example_use_network_volume",
#     networkVolumeId="your-existing-volume-id"
# )

@remote(resource_config=gpu_config_with_volume)
class VolumeExample:
    def __init__(self):
        print("Initialized with network volume mounted!")
    
    def write_to_volume(self, filename: str, content: str):
        """Write data to the network volume"""
        filepath = f"/runpod-volume/{filename}"
        with open(filepath, "w") as f:
            f.write(content)
        return f"Wrote to {filepath}"
    
    def read_from_volume(self, filename: str):
        """Read data from the network volume"""
        filepath = f"/runpod-volume/{filename}"
        try:
            with open(filepath, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "File not found"

async def main():
    storage = VolumeExample()
    
    # Write to volume
    result1 = await storage.write_to_volume("test.txt", "Hello from network volume!")
    print(result1)
    
    # Read from volume
    result2 = await storage.read_from_volume("test.txt")
    print(f"Read: {result2}")

if __name__ == "__main__":
    asyncio.run(main())