# Accelerated data analytics with cudf
# This example shows some simple accelerated data analytics functionality using cudf and pandas.

# [cudf](https://github.com/rapidsai/cudf) is part of the [NVIDIA RAPIDs](https://rapids.ai/) project.
# RAPIDs provides simple APIs to accelerate common Python data analytics functions with GPUs.

# We create a Tetra LiveServerless GPU endpoint to run our code on Runpod infrastructure.
# In the example, we specify cudf as a dependent package, and initialize it with pandas.

# We use a network volume to store the raw data on a persistent disk. If you rerun the example, our code
# will automatically load it from disk instead of redownloading

# The CUDA version is restricted to values >12.0 as required by cudf.

import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup, NetworkVolume
from tetra_rp import CudaVersion

# first, define resources - a NetworkVolume will serve as persistent storage
network_volume = NetworkVolume(
        name="cudf_network_volume",
        size=30 # in GB
)

gpu_config = LiveServerless(
    name="cudf_gpu_example",
    gpus=[
        GpuGroup.AMPERE_24, GpuGroup.ADA_24,
    ],
    networkVolume=network_volume,
    cudaVersions=[CudaVersion(f"12.{minor_cuda_version}") for minor_cuda_version in range(0, 9)],
)

@remote(gpu_config, dependencies=["pandas", "cudf-cu12", "--extra-index-url=https://pypi.nvidia.com"])
class GpuDataFrame():
    def __init__(self):
        return

    def process_taxi_data_gpu(self):
        import cudf.pandas
        cudf.pandas.install()

        import pandas as pd
        from time import time
        import os

        filepath = "/runpod-volume/taxi_data.parquet"
        
        if os.path.exists(filepath):
            print("data already exists in network volume. Reading from network volume")
            _cached = True
            df = pd.read_parquet(filepath)
        else:
            print("data not preloaded on network volume. Downloading...")
            months = [f"{i:02d}" for i in range(1, 13)]
            dfs = [pd.read_parquet(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-{month}.parquet") for month in months]
            df = pd.concat(dfs, ignore_index=True)
            _cached = False


        print(df.head())
        print("number of records in dataset: ", len(df))

        start = time()
        print("starting some GPU accelerated analytics!")

        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
        df["year"] = df["tpep_pickup_datetime"].dt.year
        df["month"] = df["tpep_pickup_datetime"].dt.month
        df["day"] = df["tpep_pickup_datetime"].dt.day
        df["hour"] = df["tpep_pickup_datetime"].dt.hour


        agg = df.groupby(["hour"]).agg(
            trips=("passenger_count", "count"),
            avg_fare=("fare_amount", "mean"),
            avg_tip=("tip_amount", "mean")
        ).reset_index()

        end = time()
        
        print(f"Elapsed time: {end - start:.6f} seconds")
        print(agg.head())

        if not _cached:
            print("caching dataset to network volume")
            df.to_parquet(filepath)

        return

async def main():

    gpu_data_frame = GpuDataFrame()
    await gpu_data_frame.process_taxi_data_gpu()
    


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
