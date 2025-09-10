# # Structured Item Metadata Extraction using `instructor` and RunPod
# This example demonstrates how to use the `instructor` library to extract structured item metadata
# from unstructured product descriptions using a language model hosted on RunPod's serverless infrastructure.

# The power of structured output lies in its ability to transform messy, inconsistent product descriptions
# into clean, database-ready records. This enables downstream systems like inventory management,
# pricing engines, or e-commerce platforms to programmatically process product information
# that would otherwise require manual data entry.

# Tetra will handle deploying our inference engine as a server onto a Runpod GPU endpoint,
# and a client using the instructor library on a Runpod CPU `LiveServerless` resource.
# The image used for the GPU endpoint and Runpod's infra will generate a ready-made 
# OpenAI compatible endpoint serving the model we specify, and the CPU endpoint will send requests to it as a client.
# with the instructor library.

# We will use Qwen3-0.6B, beacause it's a compact model that's efficient for structured data extraction tasks
# When we configure our endpoint, we specify multiple GPU types to improve availability and reduce cold start times

import asyncio
import os
from tetra_rp import remote, ServerlessEndpoint, GpuGroup, LiveServerless, CpuInstanceType

# Retrieve the RunPod API key from environment variables
# This key is required to authenticate with RunPod's serverless API
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Configure the serverless GPU endpoint for our model deployment
gpu_serverless_endpoint = ServerlessEndpoint(
    gpus=[GpuGroup.AMPERE_24, GpuGroup.ADA_24],
    name="instructor_vllm_example_gpu",
    env={
        "MODEL_NAME": MODEL_NAME,
        "MAX_CONCURRENCY": "5",  # Our model is small, so we should be able to handle multiple concurrent requests
        "MAX_MODEL_LEN": "4096",
    },
    imageName="runpod/worker-v1-vllm:v2.8.0stable-cuda12.1.0",  # use the out of the box runpod vllm worker image
)

# Configure a Live CPU Serverless Endpoint
cpu_live_serverless = LiveServerless(
    name="instructor_vllm_example_cpu",
    env={"MODEL_NAME": MODEL_NAME},
    instanceIds=[CpuInstanceType.CPU3G_8_32],
)

# this will generate an empty first request, but we do this to initialize the endpoint
@remote(resource_config=gpu_serverless_endpoint)
def init_endpoint():
    return {"input": {"prompt": "hello, world"}}


# Client code that will interact with the GPU server and submit unstructed text in the form of requests.
@remote(resource_config=cpu_live_serverless, dependencies=["openai", "instructor"])
class RemoteCPUClient: 
    def __init__(self):
        print("initialized!")

    def request_item_classification(self, item: str, gpu_server_url: str, api_key: str):
        import instructor
        from openai import OpenAI
        from enum import Enum
        from pydantic import BaseModel, Field
        import os
        # Here, we make requests to the serverless GPU endpoint that we previously created

        # We create an instructor client that will make the requests; the vllm-based docker image
        # we used when creating this endpoint comes with out of the box openai-compatible routes
        print("attemping to make a request to url: ")
        client = instructor.from_openai(
            OpenAI(api_key=api_key, base_url=gpu_server_url),
        )

        # define classes with Pydantic that our deployed model will parse unstructed data into
        class CurrencyUnit(str, Enum):
            USD = "USD"
            EUR = "EUR"


        class ItemMetadata(BaseModel):
            item_name: str = Field(description="Product name of an item")
            item_sku: str = Field(description="SKU of an item")
            item_price: int = Field(description="Price of an item in provided currency unit")
            item_price_unit: CurrencyUnit = Field(
                description="Unit of currency (eg USD) for the provided item price"
            )

        # An instructor client will generate outputs with an API similar to regular chat completions,
        # but if we pass it our Pydantic model, it will parse the passed unstructured text object
        # into the provided model.
        resp = client.chat.completions.create(
            response_model=ItemMetadata,
            model=os.environ["MODEL_NAME"],
            messages=[
                {"role": "user", "content": f"Extract metadata from this example:\n {item}"}
            ],
        )
        return resp

    def classify_items(self, gpu_server_url: str, RUNPOD_API_KEY: str):
        items = [
            "50ft ethernet cable, 43ej4, $10",
            "NVIDIA 4090 GPU, f4fodw, 200 dollars",
            "NVIDIA B200, dcxdw, 50 euros",
        ]

        # Doing these serially is not the most efficient, but it works fine for our purposes once the worker is warm
        for item in items:
            print(self.request_item_classification(item, gpu_server_url, RUNPOD_API_KEY))

async def main():
    # local entrypoint - await our dummy function call to generate the remote sls endpoint
    print("creating remote endpoint and client")
    await init_endpoint()
    client = RemoteCPUClient()

    gpu_server_url = f"https://api.runpod.ai/v2/{gpu_serverless_endpoint.id}/openai/v1"
    print("GPU server url: ", gpu_server_url)

    # call our classification function that will run on a remote endpoint
    await client.classify_items(gpu_server_url, RUNPOD_API_KEY)

if __name__ == "__main__":
    asyncio.run(main())
