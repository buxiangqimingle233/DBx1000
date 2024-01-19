import os

settings = {
    "/sys/devices/system/cpu/smt/control": "off",
    "/sys/devices/system/cpu/intel_pstate/no_turbo": "0",
    "/sys/kernel/mm/numa/demotion_enabled": "1",
    "/proc/sys/kernel/numa_balancing": "0",
    "/sys/module/damon_migrate/parameters/node1_is_toptier": "Y",
    "/sys/kernel/mm/neomem/neomem_scanning_enabled": "0",
    "/sys/module/damon_migrate/parameters/enabled": "N",
    "/sys/kernel/mm/neopebs/neopebs_enabled": "0",
    "/proc/sys/kernel/perf_cpu_time_max_percent": "0",
    "/proc/sys/vm/drop_caches": "3",
    "/proc/sys/vm/zone_reclaim_mode": "15",
    "/sys/kernel/mm/transparent_hugepage/enabled": "never",
    "/sys/kernel/mm/transparent_hugepage/defrag": "never",
    "/sys/devices/system/cpu/smt/control": "off"
}

def backup_vals():
    if not os.path.exists("./backup"):
        os.mkdir("./backup")
    for setting in settings:
        os.system(f"cp {setting} ./backup/{setting}.bak")


def reset_vals():
    print("reset all environment")
    # os.system("echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct")

    # for i in range(32):
    #     os.system(f"echo 1 > /sys/devices/system/cpu/cpu{i}/online")
    #     os.system(f"echo performance > /sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor")
