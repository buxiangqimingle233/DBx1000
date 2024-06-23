import itertools
import os
from helper import *
from plot_helper import *

# Args: Workload Parameters
# THREAD_CNT, MAX_TXNS_PER_THREAD, NUM_LOGGER, NUM_WH(TPCC)
# READ_PERC(YCSB), WRITE_PERC(YCSB), ZIPF_THETA(YCSB), REQ_PER_QUERY(YCSB), ROW_CNT(YCSB)

DBMS_CFG = ["config-std.h", "config.h"]
SIM_API = ["sim_api-std.h", "sim_api.h"]


# ESSENTIAL CONFIGS
cfg_base = {
    "WORKLOAD": "YCSB",
    "CC_ALG": "HSTORE",
    "LOG_ALGORITHM": "LOG_BATCH",
    "SNIPER": 0,
}

env_base = {
    "SNIPER_ROOT": "/home/wangzhao/snipersim",
    "SNIPER_CONFIG": "/home/wangzhao/experiments/DBx1000/cascade_lake.cfg",
    "SNIPER": 1,
    "PRIMITIVE": "CXLVANILLA",       # CXLVANILLA, CXTNL
    "SNIPER_CXL_LATENCY": 0,
    "SNIPER_MEM_LATENCY": 0,
    "NNODE": 8,
    "THREAD_PER_NODE": 64 / 8
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


def format_configs_decorator_filter_smp_cxl(func):
    def wrapper():
        cfgs, args, envs = func()
        cfgs, args, envs = format_configs(cfg_base, cfgs), format_configs(arg_base, args), format_configs(env_base, envs)
        new_envs = envs.copy()
        for env in envs:
            # assert env['SNIPER'] == 1
            if not ((env['SNIPER_CXL_LATENCY'] == 0 and env['SNIPER_MEM_LATENCY'] == 0) or (env['SNIPER_CXL_LATENCY'] != 0 and env['SNIPER_MEM_LATENCY'] != 0)): # either both 0 or both non-zero
                new_envs.remove(env)
        new_cfgs = cfgs.copy()
        for cfg in cfgs:
            # if (cfg["INDEX_STRUCT"] == "IDX_BTREE"):
            if ("INDEX_STRUCT" in cfg) and (cfg["INDEX_STRUCT"] == "IDX_BTREE"):
                cfg["INIT_PARALLELISM"] = 1     # Index BTREE only support serialized initiation
        return cfgs, args, new_envs
    return wrapper


def format_configs_decorator_raw(func):
    def wrapper():
        cfgs, args, envs = func()
        cfgs, args, envs = format_configs(cfg_base, cfgs), format_configs(arg_base, args), format_configs(env_base, envs)
        new_envs = envs.copy()
        return cfgs, args, new_envs
    return wrapper


def format_configs_decorator_filter_scalability(func):
    def wrapper():
        cfgs, args, envs = func()
        cfgs, args, envs = format_configs(cfg_base, cfgs), format_configs(arg_base, args), format_configs(env_base, envs)
        new_envs = envs.copy()
        for env in envs:
            assert env['SNIPER'] == 1
            if not ((env['SNIPER_CXL_LATENCY'] == 0 and env['SNIPER_MEM_LATENCY'] == 0) or (env['SNIPER_CXL_LATENCY'] != 0 and env['SNIPER_MEM_LATENCY'] != 0)): # either both 0 or both non-zero
                new_envs.remove(env)
        return cfgs, args, new_envs
    return wrapper


@format_configs_decorator_filter_smp_cxl
def tput_ycsb():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'TICTOC', 'SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("PRIMITIVE", ["CXLVANILLA", "CXTNL"]),
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("SNIPER_CXL_LATENCY", [0, 847]), # in ns
        ("SNIPER_MEM_LATENCY", [0, 456]), # in ns
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.7]),
        ("-z", [0, 0.8]),
        ("-R", [16]),
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-t", [8 * 8]),
        ("-s", [16 * MB]),  # 16M * 1000B per node
    ]   # Keep same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def tput_tpcc():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'TICTOC', 'SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        # ("SNIPER_CXL_LATENCY", [0, 847]), # in ns
        # ("SNIPER_MEM_LATENCY", [0, 456]), # in ns
        ("SNIPER_CXL_LATENCY", [246]), # in ns
        ("SNIPER_MEM_LATENCY", [170]),
        ("PRIMITIVE", ["CXLVANILLA", "CXTNL"]),
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [8, 64]),
        ("-Tp", [0.5, 0]),
        ("-Gx", [50]),
        ("-t", [8 * 8]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_raw
def tput_tpcc_partition():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['OCC']),
        ("LOG_ALGORITHM", ['LOG_NO']),
        ("MPR", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        ("PRIMITIVE", ["CXTNL"]),
        ("NNODE", [4]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [32]),
        ("-Tp", [0]),
        ("-Gx", [50]),
        ("-t", [32]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def tput_tpcc_oracle():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'TICTOC', 'SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("PRIMITIVE", ["CXTNL"]),
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [64]),
        ("-Tp", [0.5]),
        ("-Gx", [50]),
        ("-t", [8 * 8]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def latency_tput_tpcc():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("PRIMITIVE", ["CXLVANILLA", "CXTNL"]),
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [1, 2, 3, 4, 5, 6, 7, 8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [8, 64]),
        ("-Tp", [0.5, 0]),
        ("-Gx", [50]),
        ("-t", [-1]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def latency_tput_tpcc_tight():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['OCC']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("PRIMITIVE", ["CXLVANILLA", "CXTNL"]),
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [64]),
        ("-Tp", [0]),
        ("-Gx", [50]),
        ("-t", [-1]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def latency_tput_ycsb():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("PRIMITIVE", ["CXLVANILLA", "CXTNL"]),
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [1, 2, 3, 4, 5, 6, 7, 8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.7]),
        ("-z", [0, 0.8]),
        ("-R", [16]),
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-t", [-1]),
        ("-s", [16 * MB]),  # 16M * 1000B per node
    ]   # Keep same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def debug_ycsb():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        # ("CC_ALG", ['OCC', 'WAIT_DIE', 'NO_WAIT', 'TICTOC', 'SILO']),
        ("CC_ALG", ['OCC', 'NO_WAIT']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("PRIMITIVE", ["CXLVANILLA", "CXTNL"]),
        ("SNIPER_CXL_LATENCY", [0, 847]), # in ns
        ("SNIPER_MEM_LATENCY", [0, 456]), # in ns
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.7]),
        ("-z", [0, 0.8]),
        ("-R", [16]),
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-t", [8 * 8]),
        ("-s", [16 * MB]),  # 16M * 1000B per node
    ]   # Keep same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def record_size_sensitivity():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
        ("SIZE_PER_FIELD", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])      # 10 fields per record, from 100B to 1000B
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("PRIMITIVE", ["CXLVANILLA", "CXTNL"]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.7]),
        ("-z", [0, 0.8]),
        ("-R", [16]),
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-t", [8 * 8]),
        ("-s", [16 * MB]),  # 16M records
    ]   # Keep same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def index_sensitivity_ycsb():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
        ("SIZE_PER_FIELD", [20]),      # 10 fields per record, from 100B to 1000B
        ("INDEX_STRUCT", ["IDX_HASH", "IDX_TREE"])
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("PRIMITIVE", ["CXTNL"]),
        ("SNIPER_CXL_LATENCY", [847, 0]), # in ns
        ("SNIPER_MEM_LATENCY", [456, 0]), # in ns
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.7]),
        ("-z", [0, 0.8]),
        ("-R", [16]),
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-t", [8 * 8]),
        ("-s", [16 * MB]),  # 16M records
    ]   # Keep same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def index_sensitivity_tpcc():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
        ("INDEX_STRUCT", ["IDX_HASH", "IDX_BTREE"])
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        # ("PRIMITIVE", ["CXTNL"]),
        ("PRIMITIVE", ["CXLVANILLA"]),
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [8, 64]),
        ("-Tp", [0.5, 0]),
        ("-Gx", [50]),
        ("-t", [8 * 8]),
        ("-Ln", [2]),
    ]   # Same with Deneva

    return cfgs, args, envs



@format_configs_decorator_filter_smp_cxl
def scalibity_tpcc():
    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
        ("INDEX_STRUCT", ["IDX_HASH"])
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        # ("SNIPER_CXL_LATENCY", [847, 0]), # in ns
        # ("SNIPER_MEM_LATENCY", [456, 0]), # in ns
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("PRIMITIVE", ["CXLVANILLA"]),
        ("NNODE", [1, 2, 4, 6, 8, 10, 12, 14, 16]),
        ("THREAD_PER_NODE", [8]),   # 8 worker + 1 logger
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-t_per_wh", [8, 1]),       # thread per warehouse, transform it at the decorator
        ("-Tp", [0, 0.5]),
        ("-t", [-1]),
        ("-Gx", [50]),
        ("-Ln", [2]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def ep_test():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [0, 847]), # in ns
        ("SNIPER_MEM_LATENCY", [0, 456]), # in ns
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("PRIMITIVE", ["CXTNL"]),
        ("NNODE", [4]),
        ("THREAD_PER_NODE", [8]),
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [8, 64]),
        ("-Tp", [0.5, 0]),
        ("-Gx", [500]),
        ("-t", [4 * 8]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def dram_latency_dist():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        ("PRIMITIVE", ["CXTNL"]),
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [8, 64]),
        ("-Tp", [0.5, 0]),
        ("-Gx", [500]),
        ("-t", [8 * 8]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def dram_latency_dist_ycsb():

    cfgs = [
        ("WORKLOAD", ['YCSB']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
        ("SIZE_PER_FIELD", [20]),      # 10 fields per record, from 100B to 1000B
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        # ("SNIPER_CXL_LATENCY", [246]), # in ns
        # ("SNIPER_MEM_LATENCY", [170]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        ("PRIMITIVE", ["CXTNL"]),
        ("NNODE", [8]),
        ("THREAD_PER_NODE", [8]),
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-w", [0, 0.7]),
        ("-z", [0, 0.8]),
        ("-R", [16]),
        ("-Gx", [300]),
        ("-Ln", [4]),
        ("-t", [8 * 8]),
        ("-s", [16 * MB]),  # 16M records
    ]   # Keep same with Deneva

    return cfgs, args, envs


@format_configs_decorator_filter_smp_cxl
def bus_bw():

    cfgs = [
        ("WORKLOAD", ['TPCC']),
        ("CC_ALG", ['SILO']),
        ("LOG_ALGORITHM", ['LOG_NO']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        ("SNIPER_CXL_LATENCY", [847]), # in ns
        ("SNIPER_MEM_LATENCY", [456]), # in ns
        ("PRIMITIVE", ["CXTNL", "CXL_VANILLA"]),
        ("NNODE", [16]),
        ("THREAD_PER_NODE", [8]),
    ]

    # Args configs
    args = [
        ("-p", [1]),
        ("-n", [128]),
        ("-Tp", [0]),
        ("-Gx", [500]),
        ("-t", [-1]),
        ("-Ln", [1]),
    ]   # Same with Deneva

    return cfgs, args, envs



experiment_map = {
    "tput_ycsb": tput_ycsb,
    "tput_tpcc": tput_tpcc,
    "tput_tpcc_partition": tput_tpcc_partition,
    "debug_ycsb": debug_ycsb,
    "record_size_sensitivity": record_size_sensitivity,
    "index_sensitivity_tpcc": index_sensitivity_tpcc,
    "index_sensitivity_ycsb": index_sensitivity_ycsb,
    "scalibity_tpcc": scalibity_tpcc,
    "ep_test": ep_test,
    "dram_latency_dist": dram_latency_dist,
    "dram_latency_dist_ycsb": dram_latency_dist_ycsb,
    "bus_bw": bus_bw,
    "latency_tput_tpcc": latency_tput_tpcc,
    "latency_tput_tpcc_tight": latency_tput_tpcc_tight,
    "latency_tput_ycsb": latency_tput_ycsb,
}

time_map = {
    "tput_tpcc": {
        "20240528-211236": "847-456 ns, 2KB/core SF, 8 channel",
        "20240531-114512": "246-170 ns, 16KB/core SF, 8 channel",
        "20240613-022305": "847-456 ns, 2 channel",
        "20240621-202656": "847-456 ns, 16KB/core SF, 8 channel, recent",
        "20240622-091024": "246-170 ns, 16KB/core SF, 8 channel, recent"
    },
    "tput_tpcc_rpc": {
        "20240621-000358": "847-456 ns, 16KB/core SF, 8 channel, compile snipersim with RPC=1, avoid caching CXL accesses",
        "20240621-183817": "Using latency insert from DBx1000 tpcc_txn.cpp:261/44"
    },
    "tput_ycsb": {
        "20240530-191418": "847-456 ns, 16KB/core SF, 200B record / tuple",
        "20240531-100934": "246-170 ns, 16KB/core SF, 200B record / tuple",
    },
    "record_size_sensitivity": {
        "20240531-152318": "847-456 ns, 16KB/core SF, YCSB",
    },
    "index_sensitivity_tpcc": {
        "20240603-003124": "847-456 ns, 16KB/core SF, CXTNL PRIMITIVE",
        "20240603-123540": "847-456 ns, 16KB/core SF, CXLVANILLA PRIMITIVE"
    },
    "scalibity_tpcc": {     # Well... a typo here: scalability -> scalibity
        # "20240605-135330": "847-456 ns, 16KB/core SF, CXTNL PRIMITIVE",
        "20240606-001743": "847-456 ns, 16KB/core SF, CXTNL PRIMITIVE"
    },
    "ep_test": {
        "20240604-144141": "847-456 ns, 16KB/core SF, CXTNL PRIMITIVE, 500 txns/thread, 4 Node, 32 Threads",
    },
    "dram_latency_dist": {
        "20240606-205131": "246-170 ns, 16KB/core SF, Enable Distributed DRAM Latency",
        "20240607-172212": "847-456 ns, 16KB/core SF, Enable Distributed DRAM Latency",
    },
    "dram_latency_dist_ycsb": {
        "20240607-222421": "847-456 ns, 16KB/core SF, Enable Distributed DRAM Latency",
    },
    "bus_bw": {
        "20240608-123348": "847-456 ns, 16KB/core SF, TPC-C Enable BUS Tracking, Enable Distributed DRAM Latency",
        # "20240608-162718": "847-456 ns, 16KB/core SF, TPC-C Enable BUS Tracking, Enable Distributed DRAM Latency, 16 nodes",
        "20240609-172705": "847-456 ns, 16KB/core SF, TPC-C Enable BUS Tracking, Enable Distributed DRAM Latency, CXL vanilla",
    },
    "latency_tput_tpcc": {
        "20240611-001805": "847-456 ns, no bus tracking, no uncore latency tracking, SILO"
    },
    "latency_tput_tpcc_tight": {
        "20240612-112843": "847-456 ns, no bus tracking, no uncore latency tracking, OCC, 2 channel in CXL"
    },
    "latency_tput_ycsb": {
        "20240611-041321": "847-456 ns, no bus tracking, no uncore latency tracking"
    }
}
