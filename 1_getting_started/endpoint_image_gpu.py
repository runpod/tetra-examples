# This is an example of a GPU Endpoint that executes payloads using runpod/mock-worker.
# 1. You provide a function that returns the payload
# 2. Wrap the payload functions in @remote
# 3. Assign a server configuration to the @remote decorator
# 4. Run the function (synchronously or asynchronously)
#
# Uses runpod/mock-worker Docker image
# This approach is similar to LiveServerless but only for payload processing
# See: https://github.com/runpod-workers/mock-worker

import asyncio
from tetra_rp import remote, ServerlessEndpoint


# Configuration for a GPU endpoint using runpod/mock-worker
sls_gpu_endpoint = ServerlessEndpoint(
    name="example_endpoint_image_gpu",
    imageName="runpod/mock-worker:dev",  # Docker image
)


@remote(sls_gpu_endpoint)
def get_async_response():
    """
    Returns a payload for async processing by the mock-worker.
    """
    return {
        "input": {
            "mock_return": "async hello world from GPU endpoint",
            "mock_delay": 0.5
        }
    }


@remote(sls_gpu_endpoint, sync=True)
def get_sync_response():
    """
    Returns a payload for sync processing by the mock-worker.
    """
    return {
        "input": {
            "mock_return": "sync hello world from GPU endpoint",
            "mock_delay": 0.2
        }
    }


async def main():
    print("=== GPU Endpoint with Mock Worker Examples ===")
    print("Using runpod/mock-worker Docker image on GPU instances\n")
    
    print("1. Async response...")
    result1 = await get_async_response()
    print(f"Async Result: {result1}\n")
    
    print("2. Sync response...")
    result2 = await get_sync_response()
    print(f"Sync Result: {result2}\n")
    
    print("✅ GPU endpoint mock-worker examples completed!")
        

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
