import itertools
import os
from helper import *
from plot_helper import *

# Args: Workload Parameters
# THREAD_CNT, MAX_TXNS_PER_THREAD, NUM_LOGGER, NUM_WH(TPCC)
# READ_PERC(YCSB), WRITE_PERC(YCSB), ZIPF_THETA(YCSB), REQ_PER_QUERY(YCSB), ROW_CNT(YCSB)
sniper_tpcc_args = {
    "WH1": '-t16 -Gx50 -Ln4 -n1 -r0.9 -w0.1 -z0 -R16 -s10485760',
    "WH4": '-t16 -Gx50 -Ln4 -n4 -r0.9 -w0.1 -z0 -R16 -s10485760',
    "WH16": '-t16 -Gx50 -Ln4 -n16 -r0.9 -w0.1 -z0 -R16 -s10485760',
}

# contended TPCC (1:16), contended TPCC (1:4), uncontended TPCC (1:1)
sniper_ycsb_args = {
    "UCR": '-t16 -Gx50 -Ln4 -n1 -r0.95 -w0.05 -z0 -R16 -s10485760',
    "CR": '-t16 -Gx50 -Ln4 -n1 -r0.95 -w0.05 -z0.8 -R16 -s10485760',
    "UCW": '-t16 -Gx50 -Ln4 -n1 -r0.05 -w0.95 -z0 -R16 -s10485760',
    "CW": '-t16 -Gx50 -Ln4 -n1 -r0.05 -w0.95 -z0.8 -R16 -s10485760',
}   # Write-Intensive Uncontended, Write-Intensive Conteded, Read-Intensive Contended, Read-Intensive Uncontended

host_tpcc_args = {
    "WH1": '-t16 -Gx1000 -Ln4 -n1 -r0.9 -w0.1 -z0 -R16 -s10485760',
    "WH4": '-t16 -Gx1000 -Ln4 -n4 -r0.9 -w0.1 -z0 -R16 -s10485760',
    "WH16": '-t16 -Gx1000 -Ln4 -n16 -r0.9 -w0.1 -z0 -R16 -s10485760',
}

# contended TPCC (1:16), contended TPCC (1:4), uncontended TPCC (1:1)
host_ycsb_args = {
    "UCR": '-t16 -Gx1000 -Ln4 -n1 -r0.95 -w0.05 -z0 -R16 -s10485760',
    "CR": '-t16 -Gx1000 -Ln4 -n1 -r0.95 -w0.05 -z0.8 -R16 -s10485760', 
    "UCW": '-t16 -Gx1000 -Ln4 -n1 -r0.05 -w0.95 -z0 -R16 -s10485760', 
    "CW": '-t16 -Gx1000 -Ln4 -n1 -r0.05 -w0.95 -z0.8 -R16 -s10485760',
}   # Write-Intensive Uncontended, Write-Intensive Conteded, Read-Intensive Contended, Read-Intensive Uncontended


DBMS_CFG = ["config-std.h", "config.h"]
SIM_API = ["sim_api-std.h", "sim_api.h"]

# ESSENTIAL CONFIGS
cfg_base = {
    "WORKLOAD": "YCSB",
    "CC_ALG": "HSTORE",
    "LOG_ALGORITHM": "LOG_BATCH",
    "SNIPER": 0
}

env_base = {
    "SNIPER_ROOT": "/home/wangzhao/snipersim",
    "SNIPER_CONFIG": "/home/wangzhao/experiments/DBx1000/cascade_lake.cfg",
    "SNIPER": 1,
    "SNIPER_CXL_LATENCY": 0,
    "SNIPER_MEM_LATENCY": 0,
}



arg_base = {
}

ARG_FLAG = {
    # Usage Options
    'PART_CNT': '-p',
    'VIRTUAL_PART_CNT': '-v',
    'THREAD_CNT': '-t',
    'QUERY_INTVL': '-q',
    'PRT_LAT_DISTR': '-d',
    'PART_ALLOC': '-a',
    'MEM_PAD': '-m',
    'ABORT_PENALTY (in ms)': '-Ga',
    'CENTRAL_MAN': '-Gc',
    'TS_ALLOC': '-Gt',
    'KEY_ORDER': '-Gk',
    'NO_DL': '-Gn',
    'TIMEOUT': '-Go',
    'DL_LOOP_DETECT': '-Gl',
    'MAX_TXNS_PER_THREAD': '-Gx',
    'TS_BATCH_ALLOC': '-Gb',
    'TS_BATCH_NUM': '-Gu',
    'OUTPUT_FILE': '-o',

    # YCSB Options
    'PART_PER_TXN': '-c',
    'PERC_MULTI_PART': '-e',
    'READ_PERC': '-r',
    'WRITE_PERC': '-w',
    'ZIPF_THETA': '-z',
    'SYNTH_TABLE_SIZE': '-s',
    'REQ_PER_QUERY': '-R',
    'FIELD_PER_TUPLE': '-f',

    # TPCC Options
    'NUM_WH': '-n',
    'PERC_PAYMENT': '-Tp',
    'WH_UPDATE': '-Tu',

    # Test Options
    'Test READ_WRITE': '-Ar',
    'Test CONFLICT': '-Ac',

    # Log Options
    'LOG_BUFFER_SIZE': '-Lb',
    'MAX_LOG_ENTRY_SIZE': '-Ld',
    'MAX_NUM_EPOCH': '-Le',
    'LOG_RECOVER': '-Lr',
    'NUM_LOGGER': '-Ln',
    'LOG_NO_FLUSH': '-Lf',
    'LOG_FLUSH_INTERVAL (in us)': '-Lt',
    'g_num_disk': '-LD'
}


def format_configs(base, config):
    fmt, list_of_values = zip(*config)
    combinations = list(itertools.product(*list_of_values))
    formated_config = [merge(base, {fmt[i]: comb[i] for i in range(len(fmt))}) for comb in combinations]
    return formated_config


def format_configs_decorator(func):
    def wrapper():
        cfgs, args, envs = func()
        cfgs, args, envs = format_configs(cfg_base, cfgs), format_configs(arg_base, args), format_configs(env_base, envs)
        return cfgs, args, envs
    return wrapper


def format_configs_decorator_supplement_rw(func):
    def wrapper():
        cfgs, args, envs = func()
        cfgs, args, envs = format_configs(cfg_base, cfgs), format_configs(arg_base, args), format_configs(env_base, envs)
        for arg in args:
            if "-r" in arg:
                arg['-w'] = 1 - arg['-r']
            elif "-w" in arg:
                arg['-r'] = 1 - arg['-w']
        return cfgs, args, envs
    return wrapper


def format_configs_decorator_filter_smp_cxl(func):
    def wrapper():
        cfgs, args, envs = func()
        cfgs, args, envs = format_configs(cfg_base, cfgs), format_configs(arg_base, args), format_configs(env_base, envs)
        for env in envs:
            assert env['SNIPER'] == 1
            if not ( (env['SNIPER_CXL_LATENCY'] == 0 and env['SNIPER_MEM_LATENCY'] == 0) or (env['SNIPER_CXL_LATENCY'] != 0 and env['SNIPER_MEM_LATENCY'] != 0)): # either both 0 or both non-zero
                envs.remove(env)
        return cfgs, args, envs
    return wrapper


@format_configs_decorator
def SNA_multipartition():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['HSTORE']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [0, 400]), # in ns
    ]

    # Args configs
    args = [
        ("-r", [0.5, 0.9]),
        ("-z", [0]),
        # ("-c", [1,2,4,6,8,10,12,14,16]),
        ("-c", [8]),
        ("-p", [16]),
        ("-v", [16]),
        ("-e", [0, 0.2, 0.4, 0.6, 0.8, 1]),
        ("-R", [16]),
        ("-Gx", [500]),
        ("-Ln", [4]),
        ("-t", [16]),
        ("-s", [2097152 * 16]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator
def SNA_coherence_sweep_ref():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['HSTORE']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [0]),
    ]

    # Args configs
    args = [
        ("-w", [0.5]),
        ("-z", [0.6]),
        ("-R", [10]),
        ("-Gx", [1000]),
        ("-Ln", [4]),
        ("-t", [8]),
        ("-s", [2097152 * 8]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator
def SNA_coherence_sweep():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['HSTORE', 'NO_WAIT', 'WAIT_DIE', 'MVCC']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [0, 70, 100, 200, 400, 800, 1600]), # in ns
    ]

    # Args configs
    args = [
        ("-w", [0.5]),
        ("-z", [0.6]),
        ("-R", [10]),
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-t", [8]),
        ("-s", [2097152 * 8]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_supplement_rw
def SDA_latency_breakdown():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['OCC']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
    ]

    # YCSB Options
    # 'PART_PER_TXN': '-c',
    # 'PERC_MULTI_PART': '-e',
    # 'READ_PERC': '-r',
    # 'WRITE_PERC': '-w',
    # 'ZIPF_THETA': '-z',
    # 'SYNTH_TABLE_SIZE': '-s',
    # 'REQ_PER_QUERY': '-R',
    # 'FIELD_PER_TUPLE': '-f',

# 0, 0.1, 0.2, 0.4
    # Args configs
    args = [
        ("-p", [1]),   # No partitioning
        ("-w", [0, 0.1, 0.2, 0.4]),
        ("-z", [0, 0.5, 0.7]),    # Zipf: high contention, low contention
        ("-R", [16]),        # Standard 16 req/txn
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-t", [16]),
        ("-s", [2097152 * 16]),
    ]

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def cxl_to_smp_slowdown_ycsb():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        # ("CC_ALG", ['HEKATON', 'TICTOC', 'SILO']),
        ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'TICTOC', 'SILO', 'MVCC']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [0, 847]), # in ns
        ("SNIPER_MEM_LATENCY", [0, 456]), # in ns
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("-z", [0, 0.2, 0.5, 0.7]),
        ("-R", [16]),
        ("-Gx", [100]),
        ("-Ln", [4]),
        ("-t", [48]),
        ("-s", [2097152 * 48]),
    ]   # Same with Denevaasdf

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def cc_respect2_read():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        # ("CC_ALG", ['HEKATON', 'TICTOC', 'SILO']),
        # ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'TICTOC', 'SILO', 'MVCC']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("-z", [0, 0.5, 0.7, 0.8, 0.9]),
        ("-R", [16]),
        ("-Gx", [100]),
        ("-Ln", [4]),
        ("-t", [48]),
        ("-s", [1024 * 1024 * 3]),
    ]   # Same with Denevaasdf

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def cxl_to_smp_slowdown_tpcc():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        # ("CC_ALG", ['WAIT_DIE', 'NO_WAIT']),
        # ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'HEKATON', 'TICTOC', 'SILO', 'MVCC']),
        ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'TICTOC', 'SILO', 'MVCC']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        # ("SNIPER_CXL_LATENCY", [456]),
        # ("SNIPER_MEM_LATENCY", [847]),
        ("SNIPER_CXL_LATENCY", [0, 847]), # in ns
        ("SNIPER_MEM_LATENCY", [0, 456]), # in ns
        # ("SNIPER_CXL_LATENCY", [0, 246]), # in ns
        # ("SNIPER_MEM_LATENCY", [0, 170]), # in ns
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [4, 8, 24, 48]),
        ("-Tp", [0.5]),
        ("-Gx", [100]),
        ("-t", [48]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


experiment_map = {
    "SNA_coherence_sweep": SNA_coherence_sweep,
    "SNA_coherence_sweep_ref": SNA_coherence_sweep_ref,
    "SNA_multipartition": SNA_multipartition,
    "SDA_latency_breakdown": SDA_latency_breakdown,
    "cxl_to_smp_slowdown_ycsb": cxl_to_smp_slowdown_ycsb,
    "cxl_to_smp_slowdown_tpcc": cxl_to_smp_slowdown_tpcc,
    "cc_respect2_read": cc_respect2_read
}
