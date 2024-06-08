import re
import subprocess
import datetime
import os


GB = 1024**3
MB = 1024**2
KB = 1024
now = datetime.datetime.now()
strnow=now.strftime("%Y%m%d-%H%M%S")    # TODO: add time

def get_executable_name(cfgs):
    assert "WORKLOAD" in cfgs, "WORKLOAD not found in cfgs"
    assert "CC_ALG" in cfgs, "CC_ALG not found in cfgs"
    assert "LOG_ALGORITHM" in cfgs, "LOG_ALGORITHM not found in cfgs"
    ret = "rundb_" + cfgs['WORKLOAD'] + "_" + cfgs['CC_ALG'] + "_" + cfgs['LOG_ALGORITHM']

    if "INDEX_STRUCT" in cfgs:
        ret += "_" + cfgs['INDEX_STRUCT']

    return ret

def get_result_home(cfg, arg, env, exp, timestamp):
    home = os.path.join(os.getcwd())
    sniper = env["SNIPER"]
    if sniper:
        result_home = os.path.join(home, "sniper-results", timestamp + "_" + exp)
    else:
        result_home = os.path.join(home, "host-results", timestamp + "_" + exp)
    return result_home

def get_work_name(cfg, arg, env):
    arg_value = "_".join(f"{key}{value}" for key, value in arg.items()) # TODO: really hard to read
    if env["SNIPER"]:
        # Old experiments depend on this
        if "PRIMITIVE" not in env:
            ret = strnow + "_" + "sniper_" + get_executable_name(cfg) + "_" + arg_value + "_CC_" + str(env["SNIPER_CXL_LATENCY"]) + "_MEM_" + str(env["SNIPER_MEM_LATENCY"])
        # paperexps depend on this
        ret = strnow + "_" + "sniper_" + get_executable_name(cfg) + "_" + arg_value + "_CC_" + str(env["SNIPER_CXL_LATENCY"]) + "_MEM_" + str(env["SNIPER_MEM_LATENCY"]) + "_PRIMITIVE_" + env["PRIMITIVE"] + "_NNODE_" + str(env["NNODE"]) + "_THREAD_PER_NODE_" + str(env["THREAD_PER_NODE"])
    else:
        ret = strnow + "_" + "host_" + get_executable_name(cfg) + "_" + arg_value

    # suffix = [(cfg, "SIZE_PER_FIELD"), cfg, "INDEX_STRUCT"]
    if "SIZE_PER_FIELD" in cfg:
        ret += "_SIZE_PER_FIELD_" + str(cfg['SIZE_PER_FIELD'])
    # suffix = []
    # for a, key in suffix:
    #     if key in a:
    #         ret += "_" + key + "_" + str(a[key])

    # print("Work name: ", ret)
    return ret


def get_log_path(cfg, arg, env, exp, timestamp):
    result_home = get_result_home(cfg, arg, env, exp, timestamp)
    log_name = get_work_name(cfg, arg, env) + ".log"

    # Replace the timestamp before the first "_" with the given one
    result_home = re.sub(r'\d{8}-\d{6}', timestamp, result_home)
    log_name = re.sub(r'\d{8}-\d{6}', timestamp, log_name)

    return os.path.join(result_home, log_name)

def get_sniper_result_dir(cfg, arg, env, exp, timestamp):
    log_name = get_log_path(cfg, arg, env, exp, timestamp)
    return log_name[:-4]

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
