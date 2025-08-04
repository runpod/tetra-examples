import asyncio
from tetra_rp import remote, LiveServerless, CpuInstanceType

cpu_endpoint = LiveServerless(
    name="example_inspect_runtime_volume",
    instanceIds=[CpuInstanceType.CPU3G_1_4],
)


@remote(
    resource_config=cpu_endpoint,
    dependencies=[
        "psutil",
    ],
)
def inspect_runtime_volume():
    import platform
    import psutil
    import os
    import sys
    import subprocess

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

    print("Python Environment")
    print("------------------")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path[:3]}...")  # Show first 3 paths
    print()

    def list_directory_with_sizes(directory, indent="", max_depth=3, current_depth=0):
        """Recursively list directory contents with sizes."""
        if current_depth >= max_depth:
            return

        try:
            if not os.path.exists(directory):
                print(f"{indent}Directory does not exist")
                return

            items = os.listdir(directory)
            for item in sorted(items)[:20]:  # Limit to first 20 items per directory
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    # Get directory size using du -sh
                    try:
                        du_result = subprocess.run(
                            ["du", "-sh", item_path],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if du_result.returncode == 0:
                            size_info = du_result.stdout.split("\t")[0]
                            print(f"{indent}📁 {item}/ ({size_info})")
                        else:
                            print(f"{indent}📁 {item}/")
                    except:
                        print(f"{indent}📁 {item}/")

                    # Recursively list subdirectories, especially for runtimes
                    if item == "runtimes" or current_depth < 1:
                        list_directory_with_sizes(
                            item_path, indent + "  ", max_depth, current_depth + 1
                        )

                else:
                    size = os.path.getsize(item_path)
                    print(f"{indent}📄 {item} ({size} bytes)")

            if len(items) > 20:
                print(f"{indent}... and {len(items) - 20} more items")

        except PermissionError:
            print(f"{indent}Permission denied")
        except Exception as e:
            print(f"{indent}Error: {e}")

    print("Directory Listings with Sizes")
    print("-----------------------------")
    for directory in [
        "/runpod-volume",
    ]:
        print(f"\n{directory}:")
        list_directory_with_sizes(directory)
    print()


if __name__ == "__main__":
    try:
        asyncio.run(inspect_runtime_volume())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
