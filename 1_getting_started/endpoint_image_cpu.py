# This is an example of a CPU Endpoint that executes payloads using runpod/mock-worker.
# 1. You provide a function that returns the payload
# 2. Wrap the payload functions in @remote
# 3. Assign a server configuration to the @remote decorator
# 4. Run the function (synchronously or asynchronously)
#
# Uses runpod/mock-worker Docker image
# This approach is similar to LiveServerless but only for payload processing
# See: https://github.com/runpod-workers/mock-worker

import asyncio
from tetra_rp import remote, CpuServerlessEndpoint


# Configuration for a CPU endpoint using runpod/mock-worker
sls_cpu_endpoint = CpuServerlessEndpoint(
    name="example_endpoint_image_cpu",
    imageName="runpod/mock-worker:dev",  # Docker image
)


@remote(sls_cpu_endpoint)
def get_simple_mock_response():
    """
    Returns a simple payload for the mock-worker.
    The mock_return value will be echoed back as the result.
    """
    return {
        "input": {
            "mock_return": "Hello from CPU endpoint! This message was processed by runpod/mock-worker.",
            "mock_delay": 0.5  # Small delay to simulate processing
        }
    }


@remote(sls_cpu_endpoint, sync=True)
def get_list_response_payload():
    """
    Returns a payload that will return multiple values from mock-worker.
    Demonstrates synchronous processing with list output.
    """
    return {
        "input": {
            "mock_return": [
                "CPU task 1: Data preprocessing completed",
                "CPU task 2: Statistical analysis finished", 
                "CPU task 3: Report generation done"
            ],
            "mock_delay": 1.0  # Longer delay for sync processing
        }
    }


@remote(sls_cpu_endpoint)
def get_error_simulation_payload():
    """
    Returns a payload that simulates an error condition.
    Demonstrates error handling with CPU endpoints.
    """
    return {
        "input": {
            "mock_return": "This should not be returned because an error will occur first",
            "mock_error": True,  # Trigger an error in the mock worker
            "mock_delay": 0.2
        }
    }


@remote(sls_cpu_endpoint)
def get_complex_data_payload():
    """
    Returns a payload with complex data structure that gets echoed back.
    Demonstrates how mock-worker handles structured data.
    """
    return {
        "input": {
            "mock_return": {
                "cpu_analysis_results": {
                    "dataset_size": 1000,
                    "processing_time": "2.3 seconds",
                    "statistics": {
                        "mean": 45.7,
                        "median": 42.1,
                        "std_dev": 12.8
                    },
                    "cpu_info": {
                        "instance_type": "cpu5g-2-8",
                        "cores": 2,
                        "memory_gb": 8
                    },
                    "status": "completed"
                }
            },
            "mock_delay": 0.8
        }
    }


async def main():
    print("=== CPU Endpoint with Mock Worker Examples ===")
    print("Using runpod/mock-worker template on CPU instances\n")
    
    print("1. Simple message echo (async)...")
    result1 = await get_simple_mock_response()
    print(f"Simple Response: {result1}\n")
    
    print("2. List processing (sync)...")
    result2 = await get_list_response_payload()
    print(f"List Response: {result2}\n")
    
    print("3. Complex data structure (async)...")
    result3 = await get_complex_data_payload()
    print(f"Complex Data Response: {result3}\n")
    
    print("4. Error simulation (async)...")
    try:
        result4 = await get_error_simulation_payload()
        print(f"Error Response: {result4}\n")
    except Exception as error_result:
        print(f"Expected Error Caught: {error_result}\n")
    
    print("✅ All CPU endpoint mock-worker examples completed!")
    print("✅ The mock-worker successfully processed payloads on CPU instances")
        

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
