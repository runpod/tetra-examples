import asyncio
from tetra_rp import remote, CpuLiveServerless, CpuInstanceType


# Configure a Live CPU Serverless Endpoint with 
# 8 vCPU, 16GB RAM, max 120GB container disk
cpu_endpoint = CpuLiveServerless(
    name="example_config_cpu",
    instanceIds=[CpuInstanceType.CPU5C_8_16],
)


@remote(cpu_endpoint)
def inspect_container():
    import os

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


if __name__ == "__main__":
    try:
        asyncio.run(inspect_container())
    except Exception as e:
        print(f"❌ An error occurred: {e}")
