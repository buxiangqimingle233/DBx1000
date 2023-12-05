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

# Args: Hardware Parameters
# sniper_cxl_latencies = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 800, 1000]
sniper_cxl_latencies = [0, 100, 200, 400, 800, 1600]

# Args: DB Algorithms
# cc_algo = ['WAIT_DIE', 'NO_WAIT', 'HEKATON', 'SILO', 'TICTOC', "MVCC", "OCC"]
# cc_algo = ["HSTORE"]
cc_algo = ['WAIT_DIE', 'SILO']
# log_algo = ['LOG_NO', 'LOG_BATCH']
log_algo = ['LOG_BATCH']

DBMS_CFG = ["config-std.h", "config.h"]

# ESSENTIAL CONFIGS
cfg_base = {
    "WORKLOAD": "YCSB",
    "CC_ALG": "HSTORE",
    "LOG_ALGORITHM": "LOG_BATCH",
    "SNIPER": 0
}

env_base = {
    "SNIPER_ROOT": "/home/wangzhao/snipersim",
    "SNIPER_CONFIG": "/home/wangzhao/experiments/DBx1000/cxl_asplos.cfg",
    "SNIPER": 1,
    "SNIPER_CXL_LATENCY": 0
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


def hstore_network_sweep():
    # DB configs
    cfgs = [
        ("WORKLOAD", ['YCSB', 'TPCC']),
        ("CC_ALG", ['HSTORE']),
    ]

    # Env configs
    envs = [
        ("SNIPER", [1]),
        # ("SNIPER_CXL_LATENCY", [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]), # in ns
        ("SNIPER_CXL_LATENCY", [0, 100, 200]), # in ns
    ]

    # Args configs
    args = [
        ("-w", [0.5]),
        ("-z", [0.6]),
        ("-s", [10485760]),
        ("-R", [16]),
        ("-t", [16]),
        ("-Gx", [50]),
        ("-Ln", [4]),
        ("-n", [1]),
        ("-R", [16]),
        ("-s", [10485760]),
    ]

    cfgs, args, envs = format_configs(cfg_base, cfgs), format_configs(arg_base, args), format_configs(env_base, envs)
    return cfgs, args, envs

def hstore_network_sweep_plot():
    cfgs, args, envs = hstore_network_sweep()
    yvals = []
    xvals = [env["SNIPER_CXL_LATENCY"] for env in envs]
    vvals = [get_executable_name(cfg) for cfg in cfgs]

    for cfg, arg in itertools.product(cfgs, args):
        yvals_row = []
        for env in envs:
            result_home = get_result_home(cfg, arg, env)
            log_name = get_work_name(cfg, arg, env) + ".log"
            _, txn_cnt, _, run_time, _ = parse_log(os.path.join(result_home, log_name))
            throughput = txn_cnt / (run_time / arg["-t"])
            yvals_row.append(throughput)
        yvals.append(yvals_row)
    
    print_csv(yvals, xvals, vvals)
    draw_line_plot(yvals, xvals, vvals,  "HSTORE Network Sweep", "Throughput (txn/s)", "CXL Latency (ns)", "Benchmark", "./test.png")

experiment_map = {
    "hstore_network_sweep": hstore_network_sweep,
    "hstore_network_sweep_plot": hstore_network_sweep_plot,
}

hstore_network_sweep_plot()
