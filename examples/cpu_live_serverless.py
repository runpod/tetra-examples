import asyncio
from tetra_rp import remote, LiveServerless, CpuInstanceType

# Configure a Live CPU Serverless Endpoint
cpu_live_serverless = LiveServerless(
    name="example_cpu_live_serverless",
    instanceIds=[CpuInstanceType.CPU3G_1_4],
)


# Define a function to run on Runpod CPU
@remote(
    resource_config=cpu_live_serverless,
    dependencies=["pandas", "numpy"],
)
def cpu_data_processing(data):
    import pandas as pd
    import numpy as np
    import platform

    # Process data using CPU
    df = pd.DataFrame(data)

    return {
        "row_count": len(df),
        "column_count": len(df.columns) if not df.empty else 0,
        "mean_values": df.select_dtypes(include=[np.number]).mean().to_dict(),
        "system_info": platform.processor(),
        "platform": platform.platform(),
    }


# Define another function to run on Runpod CPU
@remote(
    resource_config=cpu_live_serverless,
    dependencies=["psutil",],
)
def inspect_cpu_machine():
    import platform
    import psutil

    print("CPU Information")
    print("----------------")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
    print(f"CPU Frequency: {psutil.cpu_freq().max:.2f} MHz")
    print()

    print("Memory Information")
    print("------------------")
    virtual_mem = psutil.virtual_memory()
    print(f"Total RAM: {virtual_mem.total / 1e9:.2f} GB")
    print(f"Available RAM: {virtual_mem.available / 1e9:.2f} GB")
    print(f"Used RAM: {virtual_mem.used / 1e9:.2f} GB")
    print()

    print("OS Information")
    print("--------------")
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print()

async def main_live_cpu_example():
    # Sample data
    sample_data = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "Bob", "age": 25, "score": 92},
        {"name": "Charlie", "age": 35, "score": 78},
        {"name": "Diana", "age": 28, "score": 88},
        {"name": "Ethan", "age": 40, "score": 95},
        {"name": "Fiona", "age": 32, "score": 81},
        {"name": "George", "age": 27, "score": 90},
        {"name": "Hannah", "age": 29, "score": 87},
        {"name": "Ian", "age": 31, "score": 83},
        {"name": "Julia", "age": 26, "score": 91},
    ]

    # Run the function on Runpod CPU
    result = await cpu_data_processing(sample_data)
    print("\n")
    print(f"Processed {result['row_count']} rows on {result['platform']}")
    print(f"Mean values: {result['mean_values']}")
    print("\n")

    # Run the other function to inspect the CPU machine
    await inspect_cpu_machine()


if __name__ == "__main__":
    try:
        asyncio.run(main_live_cpu_example())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
