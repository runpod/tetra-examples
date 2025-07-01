# Example: Simple S3 Upload and Download with RunPod
# This script creates a simple image, uploads it to S3, and then downloads it back
# using the RunPod API. It demonstrates how to use the remote function decorator
# to run code on a remote GPU instance, and how to handle S3 Compatible API operations on Runpod with boto3.

# Set up your AWS credentials as environment variables:
# export AWS_ACCESS_KEY_ID=your_access_key_id
# export AWS_SECRET_ACCESS_KEY=your_secret_access_key
# replace RUNPOD_S3_ENDPOINT with your Runpod S3 endpoint if different,
# e.g., https://s3api-eur-is-1.runpod.io/
# replace <RUNPOD_VOLUME_ID> with your actual Runpod volume ID, e.g., p23s969vxz


import asyncio
import os
from tetra_rp import remote, LiveServerless, GpuGroup, PodTemplate

# Simple GPU config
gpu_config = LiveServerless(
    name="simple-s3-testssfsfkjkjs",
    gpus=[GpuGroup.AMPERE_24],
    template=PodTemplate(
        containerDiskInGb=30,
        env=[
            {"key": "AWS_ACCESS_KEY_ID", "value": os.getenv("AWS_ACCESS_KEY_ID", "")},
            {
                "key": "AWS_SECRET_ACCESS_KEY",
                "value": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            },
        ],
    ),
    workersMax=1,
)


@remote(resource_config=gpu_config, dependencies=["boto3", "pillow"])
def create_and_upload_image():
    """Create a simple colored image and upload to S3"""
    import boto3
    import os
    import io
    from PIL import Image

    print("Creating simple image...")

    # Create a simple 512x512 blue image
    image = Image.new("RGB", (512, 512), color="blue")

    # Convert to bytes
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    print("Uploading to S3...")

    # Upload to S3
    s3_client = boto3.client(
        "s3",
        endpoint_url="https://s3api-eur-is-1.runpod.io/",
        region_name="EUR-IS-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    s3_client.put_object(
        Bucket="p23s969vxz",  # Change this to your volume ID
        Key="test_image.png",
        Body=img_bytes,
        ContentType="image/png",
    )

    print("Upload complete!")
    return "s3://p23s969vxz/test_image.png"


def download_image_from_s3(s3_path: str, local_filename: str):
    """Download image from S3 to local file"""
    import boto3
    import os

    print(f"Downloading {s3_path} to {local_filename}...")

    # Extract bucket and key from s3 path
    # s3://bucket/key -> bucket, key
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]

    # Create S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url="https://s3api-eur-is-1.runpod.io/",
        region_name="EUR-IS-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        # Download the file
        s3_client.download_file(bucket, key, local_filename)
        print(f"Downloaded to {local_filename}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


# Add this to your main function:
async def main():
    print("Simple S3 Test")

    # Check credentials
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("Set AWS_ACCESS_KEY_ID environment variable")
        return

    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("Set AWS_SECRET_ACCESS_KEY environment variable")
        return

    # Run the remote function
    result = await create_and_upload_image()
    print(f"Result: {result}")

    # Download it locally
    download_image_from_s3(result, "downloaded_test_image.png")
    print("Complete workflow: Generate → Upload → Download!")


if __name__ == "__main__":
    asyncio.run(main())
