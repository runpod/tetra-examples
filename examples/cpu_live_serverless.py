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


if __name__ == "__main__":
    try:
        asyncio.run(main_live_cpu_example())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
