import asyncio
from tetra_rp import remote, LiveServerless, GpuGroup


# Configure a Live GPU Serverless Endpoint with AMPERE_16 GPU
gpu_endpoint = LiveServerless(
    name="example_config_gpu",
    gpus=[GpuGroup.AMPERE_16],
)


@remote(gpu_endpoint)
def inspect_container():
    import os
    import subprocess

    def get_gpu_info():
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            'name': parts[0],
                            'memory_mb': int(parts[1]),
                            'temperature_c': int(parts[2]) if parts[2] != 'N/A' else None,
                            'utilization_percent': int(parts[3]) if parts[3] != 'N/A' else None
                        })
                return gpus
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        return None

    def get_container_cpu_limit():
        """Get container CPU limit from cgroup."""
        try:
            # Try cgroups v2 first
            try:
                with open('/sys/fs/cgroup/cpu.max', 'r') as f:
                    cpu_max = f.read().strip()
                    if cpu_max != 'max':
                        quota, period = cpu_max.split()
                        return float(quota) / float(period)
            except FileNotFoundError:
                pass
            
            # Try cgroups v1
            try:
                with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
                    quota = int(f.read().strip())
                with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
                    period = int(f.read().strip())
                if quota > 0:
                    return quota / period
            except (FileNotFoundError, ValueError):
                pass
            
            # Try Docker-style CPU limits
            try:
                with open('/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us', 'r') as f:
                    quota = int(f.read().strip())
                with open('/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us', 'r') as f:
                    period = int(f.read().strip())
                if quota > 0:
                    return quota / period
            except (FileNotFoundError, ValueError):
                pass
                
            # Check cpuset for CPU allocation
            try:
                with open('/sys/fs/cgroup/cpuset/cpuset.cpus', 'r') as f:
                    cpus = f.read().strip()
                    if cpus and cpus != "":
                        # Parse CPU range/list (e.g., "0" or "0-3" or "0,2,4" or "2-5,8-11")
                        total_cpus = 0
                        for part in cpus.split(','):
                            part = part.strip()
                            if '-' in part:
                                # Handle ranges like "0-3" or "2-5"
                                start, end = part.split('-')
                                total_cpus += int(end) - int(start) + 1
                            else:
                                # Handle single CPU
                                total_cpus += 1
                        return float(total_cpus)
            except (FileNotFoundError, ValueError):
                pass
            
                
        except Exception:
            pass
        return None

    def get_container_memory_limit():
        """Get container memory limit from cgroup."""
        try:
            # Try cgroups v2 first
            try:
                with open('/sys/fs/cgroup/memory.max', 'r') as f:
                    mem_max = f.read().strip()
                    if mem_max != 'max':
                        return int(mem_max)
            except FileNotFoundError:
                pass
            
            # Try cgroups v1
            try:
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    limit = int(f.read().strip())
                    # If limit is very large, it's likely unlimited
                    if limit < (1 << 62):
                        return limit
            except (FileNotFoundError, ValueError):
                pass
        except Exception:
            pass
        return None

    def get_disk_usage(path="/"):
        """Get disk usage for the specified path."""
        try:
            statvfs = os.statvfs(path)
            total = statvfs.f_frsize * statvfs.f_blocks
            free = statvfs.f_frsize * statvfs.f_bavail
            used = total - free
            return total, used, free
        except (OSError, ValueError):
            return None, None, None

    print("Container Resource Information")
    print("=============================")
    print()

    print("GPU Information")
    print("---------------")
    gpu_info = get_gpu_info()
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            print(f"GPU {i}: {gpu['name']}")
            print(f"  Memory: {gpu['memory_mb']} MB ({gpu['memory_mb'] / 1024:.1f} GB)")
            if gpu['temperature_c'] is not None:
                print(f"  Temperature: {gpu['temperature_c']}C")
            if gpu['utilization_percent'] is not None:
                print(f"  Utilization: {gpu['utilization_percent']}%")
            if i < len(gpu_info) - 1:
                print()
    else:
        print("GPU: Unable to detect or nvidia-smi not available")
    print()

    print("CPU Information")
    print("----------------")
    container_cpu_limit = get_container_cpu_limit()
    if container_cpu_limit:
        print(f"vCPU cores: {container_cpu_limit:.1f}")
    else:
        print("vCPU cores: Unable to detect")
    print()

    print("Memory Information")
    print("------------------")
    container_mem_limit = get_container_memory_limit()
    if container_mem_limit:
        print(f"RAM: {container_mem_limit / 1e9:.1f} GB")
    else:
        print("RAM: Unable to detect")
    print()

    print("Disk Information")
    print("----------------")
    total_disk, used_disk, free_disk = get_disk_usage("/")
    if total_disk is not None and used_disk is not None and free_disk is not None:
        print(f"Disk total: {total_disk / 1e9:.1f} GB")
        print(f"Disk used: {used_disk / 1e9:.1f} GB")
        print(f"Disk free: {free_disk / 1e9:.1f} GB")
    else:
        print("Disk: Unable to detect")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(inspect_container())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
