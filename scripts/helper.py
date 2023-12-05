import re
import subprocess
import datetime
import os

now = datetime.datetime.now()
strnow=now.strftime("%Y%m%d-%H%M%S")    # TODO: add time

def get_executable_name(cfgs):
    assert "WORKLOAD" in cfgs, "WORKLOAD not found in cfgs"
    assert "CC_ALG" in cfgs, "CC_ALG not found in cfgs"
    assert "LOG_ALGORITHM" in cfgs, "LOG_ALGORITHM not found in cfgs"

    return "rundb_" + cfgs['WORKLOAD'] + "_" + cfgs['CC_ALG'] + "_" + cfgs['LOG_ALGORITHM']

def get_result_home(cfg, arg, env):
    home = os.path.join(os.getcwd())
    sniper = env["SNIPER"]
    if sniper:
        result_home = os.path.join(home, "sniper-results")
    else:
        result_home = os.path.join(home, "host-results")
    return result_home

def get_work_name(cfg, arg, env):
    arg_value = "_".join(f"{key}{value}" for key, value in arg.items()) # TODO: really hard to read
    if env["SNIPER"]:
        return "sniper_" + get_executable_name(cfg) + "_" + arg_value + "_CC_" + str(env["SNIPER_CXL_LATENCY"])
    else:
        return "host_" + get_executable_name(cfg) + "_" + arg_value

# in cfgs: {name, value}
def replace_configs(filename, cfgs):
    with open(filename, 'r') as f:
        content = f.read()
    for param, value in cfgs.items():
        pattern = r"\#define\s*" + re.escape(param) + r'.*'
        replacement = "#define " + param + ' ' + str(value)
        content = re.sub(pattern, replacement, content)
        print("Replacing in {}: {} -> {}".format(filename, pattern, replacement))
    with open(filename, 'w') as f:
        f.write(content)

def get_cpu_freq():
    res = subprocess.check_output('lscpu', shell=True).decode().split('@')[1].strip()
    res = float(res.split('GHz')[0])
    print('Using CPU_FREQ', res)
    return res

CPU_FREQ = get_cpu_freq()

def merge(base, incoming):
    # Create a copy of the original dictionary
    new_dict = base.copy()
    for key, value in incoming.items():
        # Insert or update the key-value pair
        new_dict[key] = value
    return new_dict


# def replace(filename, pattern, replacement):
#     with open(filename, 'r') as f:
#         s = f.read()
#     new_s = re.sub(pattern, replacement, s)
#     if s != new_s:
#         print("Replacing in {}: {} -> {}".format(filename, pattern, replacement))
#     with open(filename, 'w') as f:
#         f.write(new_s)

# def get_executable_name(job):
#     return "rundb_" + job['WORKLOAD'] + "_" + job['CC_ALG'] + "_" + job['LOG_ALGORITHM']

# def get_work_name(job, run_args, cache_coherence_latency=0):
#     if SNIPER:
#         return "sniper_" + get_executable_name(job) + "_" + run_args + "_CC_" + str(cache_coherence_latency)
#     else:
#         return "host_" + get_executable_name(job) + "_" + run_args
