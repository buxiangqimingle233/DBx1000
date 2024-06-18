from experiments import experiment_map
from paperexps import experiment_map as paper_map
from paperexps import time_map
from helper import *
from plot_helper import *
import itertools, subprocess
import os, re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def SNA_coherence_sweep_plot():
    # cfgs, args, envs = SNA_coherence_sweep()
    cfgs, args, envs = experiment_map["SNA_coherence_sweep"]()
    yvals = []
    xvals = [env["SNIPER_CXL_LATENCY"] for env in envs]
    vvals = [get_executable_name(cfg) for cfg in cfgs]

    for cfg, arg in itertools.product(cfgs, args):
        yvals_row = []
        for env in envs:
            log_path = get_log_path(cfg, arg, env, "SNA_coherence_sweep", "")
            _, txn_cnt, _, run_time, _ = parse_log(log_path)
            throughput = txn_cnt / (run_time / arg["-t"])
            yvals_row.append(throughput)
        yvals.append(yvals_row)

    print_csv(yvals, xvals, vvals)
    draw_line_plot(yvals, xvals, vvals,  "Coherence Latency Sweep", "Throughput (txn/s)", "CXL Latency (ns)", "Benchmark", "./SNA_coherence_sweep.png")


def SNA_multipartition_plot():
    # cfgs, args, envs = SNA_multipartition()
    cfgs, args, envs = experiment_map["SNA_multipartition"]()
    yvals = []
    xvals = [arg["-e"] for arg in args]
    vvals = [env["SNIPER_CXL_LATENCY"] for env in envs]

    for cfg, env in itertools.product(cfgs, envs):
        yvals_row = []
        for arg in args:
            log_path = get_log_path(cfg, arg, env, "SNA_multipartition", "2021-05-25_17-00-00")
            _, txn_cnt, _, run_time, _ = parse_log(log_path)
            throughput = txn_cnt / (run_time / arg["-t"])
            yvals_row.append(throughput)
        yvals.append(yvals_row)

    print_csv(yvals, xvals, vvals)
    draw_line_plot(yvals, xvals, vvals,  "Coherence Latency Sweep", "Throughput (txn/s)", "Averaged Touched Partitions", "Benchmark", "./SNA_multipartition.png")


def SDA_latency_breakdown():
    cfgs, args, envs = experiment_map["SDA_latency_breakdown"]()
    for cfg, arg, env in itertools.product(cfgs, args, envs):
        log_path = get_log_path(cfg, arg, env, "SDA_latency_breakdown", "20231212-161356")
        _, txn_cnt, _, run_time, time_parts = parse_log(log_path)
        name = "_".join([f"{k}{v}" for k, v in arg.items()])
        org_time_parts = time_parts.copy()

        # remote_ratio = zipf
        remote_ratio = 0.5

        idx = time_parts["time_index"]
        tsm = time_parts["time_shared_metadata"]
        tsr = time_parts["time_shared_record"]
        # abort = time_parts["time_abort"]

        # print(idx, abort, tsm, tsr)
        # Summation of time parts seem less than run_time
        # scaled_run_time = run_time + tsm + tsr + idx + abort

        # Pass 2: Batch optimization (CALVIN-Like)
        # num_batch = scaled_run_time / arg['-t'] / (50 / 1000)
        # idx = org_time_parts["time_index"]
        # tsm = org_time_parts["time_shared_metadata"]
        # tsr = org_time_parts["time_shared_record"]
        # idx = idx + num_batch * remote_ratio * 2 / 1000                                   # batch * 2us RDMA two-sided rpc (Assume broadcast, one roundtrip/txn)

        # slowdown = scaled_run_time / run_time
        print("name: {} throughput: {}".format(name, txn_cnt / (run_time / arg['-t'])))
        # print(run_time / (arg['-Gx'] * arg['-t']))  # latency per txn
        # print(slowdown, scaled_run_time / arg['-t'], (time_parts["time_shared_metadata"] + tsm) / arg['-t'], (time_parts["time_shared_record"] + tsr) / arg['-t'], (time_parts["time_index"] + idx) / arg['-t'], sep=" ")


def cxl_to_smp_slowdown_ycsb_plot(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_ycsb"]()
    oracle, cxl = {}, {}
    for cfg, arg, env in itertools.product(cfgs, args, envs):
        # log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_ycsb", "20231221-170754")
        log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_ycsb", exec_time)
        _, txn_cnt, _, run_time, time_parts = parse_log(log_path)
        name = "_".join([f"{k}_{v}" for k, v in cfg.items()]) + "_".join([f"{k}{v}" for k, v in arg.items()])
        tput = txn_cnt / (run_time / arg['-t'])
        if env["SNIPER_CXL_LATENCY"] == 0 and env["SNIPER_MEM_LATENCY"] == 0:
            oracle[name] = tput
        elif env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] == 0:
            cxl[name] = tput

    for k in oracle:
        print(k, oracle[k] / cxl[k])


def cxl_to_smp_slowdown_tpcc_plot(exec_time_cxl, exec_time_smp=None):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()
    oracle, cxl = {}, {}

    if exec_time_smp is None:
        exec_time_smp = exec_time_cxl

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        # if arg['-n'] not in [4, 24, 48]:
        #     continue
        if env["SNIPER_CXL_LATENCY"] == 0 and env["SNIPER_MEM_LATENCY"] == 0:
            log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time_smp)
        elif env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0:
            log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time_cxl)
        # print(log_path)

        _, txn_cnt, txn_abort_cnt, run_time, time_parts = parse_log(log_path)
        txn_cnt = txn_cnt - txn_abort_cnt

        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])

        tput = txn_cnt / (run_time / arg['-t'])
        # print(name, tput, env["SNIPER_CXL_LATENCY"], env["SNIPER_MEM_LATENCY"])
        if env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] == 0:
            oracle[name] = tput
        elif env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0:
            cxl[name] = tput

    res = [max(oracle[k] / cxl[k], 1) for k in oracle]
    # res = [cxl[k] for k in oracle]
    # for k in oracle:
    #     print(k, cxl[k])

    draw_bar_plot(res, oracle.keys(), "CXL to SMP Slowdown", "Benchmark", "Slowdown", "./cxl_to_smp_slowdown_tpcc.png")

    # print(res)
    # print("avg: {}, max: {}, min: {}".format(sum(res) / len(res), max(res), min(res)))

    return res, list(oracle.keys())


def cxl_to_smp_slowdown_ycsb_plot(exec_time_cxl, exec_time_smp=None):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_ycsb"]()
    oracle, cxl = {}, {}

    if exec_time_smp is None:
        exec_time_smp = exec_time_cxl

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if arg['-w'] not in [0]:
            continue
        if env["SNIPER_CXL_LATENCY"] == 0 and env["SNIPER_MEM_LATENCY"] == 0:
            log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_ycsb", exec_time_smp)
        elif env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0:
            log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_ycsb", exec_time_cxl)
        # print(log_path)

        _, txn_cnt, txn_abort_cnt, run_time, time_parts = parse_log(log_path)
        txn_cnt = txn_cnt - txn_abort_cnt

        name = gen_simplified_name(cfg, arg, env, ['-w', '-z', 'CC_ALG'])

        tput = txn_cnt / (run_time / arg['-t'])
        # print(name, tput, env["SNIPER_CXL_LATENCY"], env["SNIPER_MEM_LATENCY"])
        if env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] == 0:
            oracle[name] = tput
        elif env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0:
            cxl[name] = tput

    res = [max(oracle[k] / cxl[k], 1) for k in oracle]
    # res = [cxl[k] for k in oracle]
    # for k in oracle:
    #     print(k, cxl[k])

    draw_bar_plot(res, oracle.keys(), "CXL to SMP Slowdown", "Benchmark", "Slowdown", "./cxl_to_smp_slowdown_ycsb.png")

    # print(res)
    # print("avg: {}, max: {}, min: {}".format(sum(res) / len(res), max(res), min(res)))

    return res, list(oracle.keys())


def cxl_breakdown_ycsb(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_ycsb"]()
    yvals, xval = [], []
    legend = []

    for cfg, arg, env in itertools.product(cfgs, args, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if arg['-z'] not in [0, 0.7]:
            continue
        log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_ycsb", exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)


        name = gen_simplified_name(cfg, arg, env, ['-w', 'CC_ALG', '-z'])

        # compute = time_parts['time_log'] + time_parts['time_query'] + time_parts['useful_work'] + time_parts['time_index']
        # txn_manager_tl = time_parts['time_man'] + time_parts['time_cleanup'] + time_parts['time_wait']
        # memory_layer = time_parts['time_shared_record'] + time_parts['time_shared_metadata']

        # yvals.append([compute, txn_manager_tl, memory_layer])
        # legend = ['Log-Query-Compute', 'Txn Manager (Thread-Local)', 'Memory Layer']
        version_management = (time_parts['time_shared_record'] + time_parts['time_man']) / run_time
        conflict_detection = time_parts['time_shared_metadata'] / run_time

        yvals.append(version_management)

        # yvals.append([version_management, conflict_detection])
        # legend = ['Version Management', 'Conflict Detection']

        xval.append(arg['-w'])
        legend.append(cfg['CC_ALG'])

        # print(name, sum(time_parts.values()) / run_time)

    prev_legend = None

    print(" ".join(map(str, xval)))
    for y, l in zip(yvals, legend):
        if prev_legend is not None and prev_legend != l:
            print(l)  # print a newline when the legend changes
        print(y, end=' ')
        prev_legend = l
    print(legend[-1])  # print a newline at the end

    # legend = list(time_parts.keys()) # Detailed breakdown

    return yvals, xval, legend


def cxl_latency_tpcc(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()
    yvals, xval = [], []
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if not (arg['-n'] in [4, 24, 48]):
            continue
        log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)
        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])

        yvals.append(run_time)
        xval.append(name)
    draw_bar_plot(yvals, xval, "CXL TPCC Latency", "Benchmark", "Latency (s)", "./cxl_tpcc_latency.png")
    return yvals, xval


def cxl_breakdown_tpcc(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()
    yvals, xval = [], []
    version_management, abort_rate, cc = [], [], []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        # if arg['-Tp'] == 1 or arg['-n'] not in [4, 24, 48]:
        #     continue
        log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])

        compute = time_parts['time_log'] + time_parts['time_query'] + time_parts['useful_work'] + time_parts['time_index']
        txn_manager_tl = time_parts['time_man'] + time_parts['time_cleanup'] + time_parts['time_wait']
        memory_layer = time_parts['time_shared_record'] + time_parts['time_shared_metadata']

        version_management.append((time_parts['time_shared_record'] + time_parts['time_man'] + time_parts['time_cleanup']) / run_time)
        abort_rate.append(abort_cnt / (txn_cnt + abort_cnt))
        cc.append(cfg['CC_ALG'])

        yvals.append([compute, txn_manager_tl, memory_layer])
        legend = ['Log-Query-Compute', 'Txn Manager (Thread-Local)', 'Memory Layer']

        # yvals.append([time_parts['time_shared_record'] + time_parts['time_man'] + time_parts['time_cleanup'], time_parts['time_shared_metadata']])
        # legend = ['Version Management', 'Conflict Detection']

        xval.append(name)
        # print(name, sum(time_parts.values()) / run_time)

    for i, row in enumerate(yvals):
        row_sum = sum(row)
        if row_sum == 0:
            continue  # Avoid division by zero
        yvals[i] = [element / row_sum for element in row]
        # print(yvals[i][2])

    # legend = list(time_parts.keys()) # Detailed breakdown
    draw_stacked_bar_plot(yvals, xval, legend, "CXL TPCC Breakdown", "Benchmark", " ", "./cxl_tpcc_breakdown.png", True)

    # Collapse with concurrency control
    draw_2d_scatter_with_legend(abort_rate, version_management, cc, "CXL TPCC Breakdown", "Abort Rate", "Version Management", "./cxl_tpcc_breakdown_scatter.png")
    dic = {}
    for v, a, c in zip(version_management, abort_rate, cc):
        if c not in dic:
            dic[c] = []
        else:
            dic[c].append([v, a])

    for c in dic:
        print(c, "\n".join(map(lambda x: str(x[0]) + " " + str(x[1]), dic[c])), sep=" ")

    return yvals, xval, legend


def cxl_cpi_tpcc(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()

    yvals, xval = [], []

    # legend = set()  # Create an empty set to store all keys
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        # if arg['-n'] not in [4, 24, 48]:
        #     continue

        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])

        cpi_plot_tool = os.path.join(env["SNIPER_ROOT"], "tools/cpistack.py")
        home = get_sniper_result_dir(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time)
        cpi_cmd = "{} {} {}".format("python2", cpi_plot_tool, "--abstime --partial roi-begin:roi-end")
        res = subprocess.Popen(cpi_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        data = parse_cpi_output(res)
        # Get the summation of all cores
        data = {key: sum(value.values()) for key, value in data.items()}
        yvals.append(data)

        # legend.update(data.keys())
        xval.append(name)

    # legend = list(legend)  # Convert the set to a list to preserve order

    # Detailed Description
    legend = ['sync-futex', 'sync-unscheduled', 'mem-remote', 'mem-l1d', 'mem-l3', 'mem-dram', 'base', 'other']
    # printed_legend = ['sync', 'mem-cxl', 'cache-processor', 'compute']
    printed_legend = ['sync', 'cxl-coherence', 'cxl-mem', 'processor-cache', 'compute']
    for i in range(len(yvals)):
        yvals[i] = [yvals[i].get(key, 0) for key in legend]     # filter
        yvals[i] = [
            yvals[i][0] + yvals[i][1],
            yvals[i][2],  # sync
            yvals[i][5],
            # yvals[i][2] + yvals[i][5], # mem-cxl
            yvals[i][3] + yvals[i][4],  # mem-private
            yvals[i][6] + yvals[i][7]   # compute
        ]
        yvals[i] = [element / sum(yvals[i]) for element in yvals[i]]

    # legend = ['sync-futex', 'sync-unscheduled', 'mem-remote', 'mem-l1d', 'mem-l3', 'mem-dram']
    # printed_legend = ['sync', 'mem-cxl-coherence',  'mem-cxl-memory', 'processor-cache']
    # for i in range(len(yvals)):
    #     yvals[i] = [yvals[i].get(key, 0) for key in legend]     # filter
    #     yvals[i] = [
    #         yvals[i][0] + yvals[i][1],  # sync
    #         yvals[i][2],  # mem-cxl-coherence
    #         yvals[i][3] + yvals[i][4],  # mem-private
    #         yvals[i][5],  # mem-cxl-memory
    #         yvals[i][6] + yvals[i][7]  # compute
    #     ]
    #     yvals[i] = [element / sum(yvals[i]) for element in yvals[i]]

    draw_stacked_bar_plot(yvals, xval, printed_legend, "CXL TPCC CPI", "Benchmark", "CPI", "./cxl_tpcc_cpi.png", True)
    return yvals, xval, printed_legend


def cxl_slowdown_version_management(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()
    vm_ratio, abort_rate, cc = [], [], []
    benchmarks = []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        # if arg['-Tp'] == 1 or arg['-n'] not in [4, 24, 48]:
        #     continue
        log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])

        vm_ratio.append((time_parts['time_shared_record'] + time_parts['time_man'] + time_parts['time_cleanup']) / run_time)
        abort_rate.append(abort_cnt / (txn_cnt + abort_cnt))
        cc.append(cfg['CC_ALG'])
        benchmarks.append(name)


        # if c not in dic:
        #     dic[c] = []
        # else:
        #     dic[c].append([v, a])

    # for c in dic:
    #     print(c, "\n".join(map(lambda x: str(x[0]) + " " + str(x[1]), dic[c])), sep=" ")

    # cpi_yvals, cpi_benchmarks, cpi_legend = cxl_tpcc_cpi(exec_time)
    slowdown, slowdown_benchmarks = cxl_to_smp_slowdown_tpcc_plot(exec_time)
    # print(cpi_benchmarks, slowdown_benchmarks, sep="\n")
    assert set(benchmarks) == set(slowdown_benchmarks)

    vm_ratio_dict = dict(zip(benchmarks, vm_ratio))
    slowdown_dict = dict(zip(slowdown_benchmarks, slowdown))
    res_x, res_y, res_l = [], [], []
    for bm in benchmarks:
        res_x.append(slowdown_dict[bm])
        res_y.append(vm_ratio_dict[bm])
        res_l.append(cc[benchmarks.index(bm)])

    dic = {}
    for v, a, c in zip(vm_ratio, slowdown, cc):
        print(v, a, c)
    draw_2d_scatter_with_legend(res_x, res_y, res_l, "CXL TPCC Breakdown", "Slowdown", "Version Management", "./cxl_tpcc_breakdown_scatter.png")


def cxl_slowdown_breakdown_tpcc(exec_time):
    # cpi_yvals, cpi_benchmarks, cpi_legend = cxl_tpcc_cpi(exec_time)
    breakdown_yvals, breakdown_benchmarks, breakdown_legend = cxl_breakdown_tpcc(exec_time)
    slowdown, slowdown_benchmarks = cxl_to_smp_slowdown_tpcc_plot(exec_time, None)
    # print(cpi_benchmarks, slowdown_benchmarks, sep="\n")
    assert set(breakdown_benchmarks) == set(slowdown_benchmarks)

    breakdown_dict = dict(zip(breakdown_benchmarks, breakdown_yvals))
    slowdown_dict = dict(zip(slowdown_benchmarks, slowdown))
    scaled_breakdown = []
    for benchmark in slowdown_benchmarks:
        scale = slowdown_dict[benchmark] / sum(breakdown_dict[benchmark])
        scaled_breakdown.append([scale * element for element in breakdown_dict[benchmark]])

    for i in range(len(scaled_breakdown)):
        for j in range(len(scaled_breakdown[i])):
            print(scaled_breakdown[i][j], end="\t")
        print()

    print(" ".join(breakdown_legend))
    print(" ".join(slowdown_benchmarks))
    draw_stacked_bar_plot(scaled_breakdown, slowdown_benchmarks, breakdown_legend, "CXL TPCC Slowdown", "Benchmark", "CPI", "./cxl_slowdown_breakdown.png", False)


def cxl_memstatus_ycsb(exec_time):
    bm = "cc_respect2_read"
    cfgs, args, envs = experiment_map[bm]()

    yvals_dict = {}
    yvals, xval, vval = [], [], []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if not (cfg['CC_ALG'] == 'SILO') or arg['-w'] not in [0.2, 0.4, 0.8]:
            continue
        name = gen_simplified_name(cfg, arg, env, ['-z', '-w', 'CC_ALG'])

        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        memstatus_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(memstatus_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        data = parse_memstatus_output(res)

        cpi_plot_tool = os.path.join(env["SNIPER_ROOT"], "tools/cpistack.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        cpi_cmd = "{} {} {}".format("python2", cpi_plot_tool, "--cpi --partial roi-begin:roi-end")
        res = subprocess.Popen(cpi_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        cpi = parse_cpi_output(res)
        # Get the summation of all cores
        cpi = {key: sum(value.values()) / len(value) for key, value in cpi.items()}

        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        # data['cxl-coherence-ratio'] = data['cxl-cache-cnt'] * 2      # 2: req-reply, arg['t']: total cores

        # CXL Overhead + Average memory access latency (3.2 GHz, 1 instr/cycle)
        cache_cnt =(data['l3-cxl-cache-overhead'] + data["directory-cxl-cache-overhead"]) / env["SNIPER_CXL_LATENCY"]
        # cxl_mem_cnt = data['cxl-mem-overhead'] / env["SNIPER_MEM_LATENCY"]
        cxl_mem_cnt = data["cxl-mem-read-cnt"]
        cache_overhead = (data['l3-cxl-cache-overhead'] + data["directory-cxl-cache-overhead"])
        mem_overhead = data["cxl-mem-overhead"]
        cxl_access_cnt = (data['cxl-mem-read-cnt'] + data["cxl-mem-write-cnt"])
        # yval = data["directory-cxl-cache-overhead"] / data["l3-cxl-cache-overhead"]
        # yval = (cache_overhead + mem_overhead) / cxl_mem_cnt + (cpi["mem-l3"] + cpi["mem-l1d"] + 66) / 3.2
        yval = (cpi["mem-l3"] + cpi["mem-l1d"] + cpi["mem-dram"])
        # yval = (cpi["mem-l3"] + cpi["mem-l1d"] + cpi["mem-dram"] + cpi["base"] + cpi.get("sync-futex", 0))


        # yval = cache_cnt / (txn_cnt - abort_cnt)
        # yval = (time_parts['time_shared_record'] + time_parts['time_shared_metadata']) / run_time


        # Get the arg['-w'] and arg['-z'] values
        w_val = arg['-w']
        z_val = arg['-z']
        # If the arg['-w'] value is not in xval, add it
        if w_val not in xval:
            xval.append(w_val)

        # If the arg['-z'] value is not in vval, add it
        if z_val not in vval:
            vval.append(z_val)

        # Add the yvals to the dictionary
        yvals_dict[(w_val, z_val)] = yval

        # print(abort_cnt / txn_cnt, data['cxl-cache-overhead'])
        # print(data['cxl-mem-read-cnt'], z_val, yval)
        # print(w_val, z_val, cache_cnt, cxl_mem_cnt)
        # print(w_val, z_val, yval)
        print(yval)

    yvals = [[yvals_dict[(w, z)] for z in sorted(vval)] for w in sorted(xval)]

    # Sort xval, vval, and yvals
    xval, yvals = zip(*sorted(zip(xval, yvals), key=lambda x: x[0]))
    vval, yvals = zip(*sorted(zip(vval, list(map(list, zip(*yvals)))), key=lambda x: x[0]))

    # Convert yvals back to a list of lists
    yvals = [list(y) for y in zip(*yvals)]

    # legend = ["load-hit-l1", "load-hit-l3", "load-from-dram", "load-from-remote-cache"]

    # Transpose
    # yvals = [list(row) for row in zip(*yvals)]
    for y in yvals:
        print(" ".join(map(str, y)))
    print(xval, vval, sep="\n\n")
    draw_line_plot(yvals, vval, xval,  "CXL Coherence Traffic", "CXL Coherence Traffic", "Write Ratio", "THETA", "./cxl_memstatus.png")
    return yvals, xval, vval


def cxl_ccnum_tpcc(exec_time):
    bm = "cxl_to_smp_slowdown_tpcc"
    cfgs, args, envs = experiment_map[bm]()

    yvals_dict = {}
    yvals, xval, vval = [], [], []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])

        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        memstatus_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(memstatus_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        data = parse_memstatus_output(res)
        
        print(data)
        # yval = data["l3-cxl-cache-overhead"] + data["load-from-remote-cache"]

        # yvals.append(yval)

    for y in yvals:
        print(" ".join(map(str, y)))
    return yvals, xval, vval


def cxl_sfquery_ycsb(exec_time):
    bm = "cc_respect2_read"
    cfgs, args, envs = experiment_map[bm]()

    yvals_dict = {}
    yvals, xval, vval = [], [], []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        # if not (cfg['CC_ALG'] == 'SILO') or arg['-z'] == 0.8:
        #     continue
        name = gen_simplified_name(cfg, arg, env, ['-z', '-w', 'CC_ALG'])

        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        memstatus_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(memstatus_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        data = parse_memstatus_output(res)
        
        yval = data["entries-allocated"]

        # Get the arg['-w'] and arg['-z'] values
        w_val = arg['-w']
        z_val = arg['-z']
        # If the arg['-w'] value is not in xval, add it
        if w_val not in xval:
            xval.append(w_val)

        # If the arg['-z'] value is not in vval, add it
        if z_val not in vval:
            vval.append(z_val)

        # Add the yvals to the dictionary
        yvals_dict[(w_val, z_val)] = yval


    yvals = [[yvals_dict[(w, z)] for z in sorted(vval)] for w in sorted(xval)]

    # Sort xval, vval, and yvals
    xval, yvals = zip(*sorted(zip(xval, yvals), key=lambda x: x[0]))
    vval, yvals = zip(*sorted(zip(vval, list(map(list, zip(*yvals)))), key=lambda x: x[0]))

    # Convert yvals back to a list of lists
    yvals = [list(y) for y in zip(*yvals)]

    # legend = ["load-hit-l1", "load-hit-l3", "load-from-dram", "load-from-remote-cache"]

    # Transpose
    yvals = [list(row) for row in zip(*yvals)]
    for y in yvals:
        print(" ".join(map(str, y)))
    print(xval, vval, sep="\n\n")
    draw_line_plot(yvals, xval, vval,  "CXL Coherence Traffic", "SF Queries", "Write Ratio", "THETA", "./cxl_sfquery.png")
    return yvals, xval, vval


def cxl_throughput_ycsb(exec_time):
    cfgs, args, envs = experiment_map["cc_respect2_read"]()
    yvals, xval = [], []
    legend = []

    for cfg, arg, env in itertools.product(cfgs, args, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if arg['-z'] not in [0.8]:
            continue
        log_path = get_log_path(cfg, arg, env, "cc_respect2_read", exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        # tput = (txn_cnt - abort_cnt) / (run_time / arg['-t'])
        # tput = abort_cnt
        name = gen_simplified_name(cfg, arg, env, ['-w', 'CC_ALG', '-z'])

        yvals.append(tput)
        xval.append(name)

    legend = ["throughput"]
    prev_legend = None

    print(" ".join(map(str, xval)))
    for y, l in zip(yvals, legend):
        if prev_legend is not None and prev_legend != l:
            print(l)  # print a newline when the legend changes
        print(y, end=' ')
        prev_legend = l
    print(legend[-1])  # print a newline at the end

    # legend = list(time_parts.keys()) # Detailed breakdown
    draw_bar_plot(yvals, xval, "CXL YCSB Throughput", "Benchmark", "Throughput (txn/s)", "./cxl_throughput_ycsb.png")
    return yvals, xval, legend


def cxl_breakdown_cpi_tpcc(exec_time):

    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()
    yvals, xval = [], []
    for cfg, arg, env in itertools.product(cfgs, args, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] == 0):
            continue
        # if arg['-Tp'] == 1 or arg['-n'] not in [4, 24, 48]:
        #     continue
        log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time)
        _, txn_cnt, _, run_time, time_parts = parse_log(log_path)

        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])

        # compute = time_parts['time_log'] + time_parts['time_query'] + time_parts['useful_work'] + time_parts['time_index']
        # txn_manager_tl = time_parts['time_man'] + time_parts['time_cleanup'] + time_parts['time_wait']
        # memory_layer = time_parts['time_shared_record'] + time_parts['time_shared_metadata']
        # print(time_parts)
        yvals.append(list(time_parts.values()))  # Convert values to list
        xval.append(name)
        legend = list(time_parts.keys())

    cpi_yvals, cpi_benchmarks, cpi_legend = cxl_cpi_tpcc(exec_time)
    breakdown_yvals, breakdown_benchmarks, breakdown_legend = yvals, xval, legend
    assert set(cpi_benchmarks) == set(breakdown_benchmarks)

    cpi_dict = dict(zip(cpi_benchmarks, cpi_yvals))
    breakdown_dict = dict(zip(breakdown_benchmarks, breakdown_yvals))

    res = []
    def get_val(vals, legend, name):
        return vals[legend.index(name)] if name in legend else 0

    for benchmark in breakdown_benchmarks:
        c, b = cpi_dict[benchmark], breakdown_dict[benchmark]
        # print(c, b)
        # sync = c[cpi_legend.index("sync")]
        sync = get_val(c, cpi_legend, "sync")
        # conflict_detect = get_val(b, breakdown_legend, "time_shared_metadata") * sync + get_val(b, breakdown_legend, "time_wait")
        total_time = sum(b)
        memory_copy = get_val(b, breakdown_legend, "time_shared_record")
        buffer_management = get_val(b, breakdown_legend, "time_man") + get_val(b, breakdown_legend, "time_cleanup")

        res.append([memory_copy / total_time, buffer_management / total_time])

    # # Regularization
    # for i in range(len(res)):
    #     res[i] = [element / sum(res[i]) for element in res[i]]

    legend = ["memory_copy", "buffer_management"]
    draw_stacked_bar_plot(res, breakdown_benchmarks, legend, "CXL TPCC CPI", "Benchmark", "CPI", "./cxl_breakdown_cpi.png", False)
    for i in range(len(res)):
        res[i] = sum(res[i])

    for i in range(0, len(res), 5):
        print(breakdown_benchmarks[i] + " " + " ".join(map(str, res[i:i + 5])))
    print(" ".join(breakdown_benchmarks))


# ================================================
# ======== Following codes are paper plots
# ================================================

def tput(exec_time):

    # bm = "tput_ycsb"
    # variables = ["-w", "-z", "CC_ALG"]

    bm = "tput_tpcc"
    variables = ["CC_ALG", "-n", "-Tp"]

    yvals, xval = [], []
    oracle, cxl_vanilla, cxtnl = {}, {}, {}    # local time + remote time

    cfgs, args, envs = paper_map[bm]()
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        # tput = (txn_cnt - abort_cnt) / (run_time / arg['-t']) / arg['-t']
        tput = abort_cnt / txn_cnt
        # tput = run_time / arg['-t']

        if env["SNIPER_CXL_LATENCY"] == 0 and env["SNIPER_MEM_LATENCY"] == 0 and env["PRIMITIVE"] == "CXTNL":
            oracle[name] = tput
        elif env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0:
            if env["PRIMITIVE"] == "CXLVANILLA":
                cxl_vanilla[name] = tput
            elif env["PRIMITIVE"] == "CXTNL":
                cxtnl[name] = tput

        # print(name, sum(time_parts.values()) / run_time)

    # legend = ["Local Time", "Remote Time"]
    ordered_name = sorted(cxtnl.keys(), reverse=True)
    res = [cxl_vanilla.get(k, 0) for k in ordered_name]
    res2 = [cxtnl.get(k, 0) for k in ordered_name]
    res3 = [oracle.get(k, 0) for k in ordered_name]

    # res = [max(cxtnl[k] / cxl_vanilla[k], 1) for k in sorted(oracle.keys(), reverse=True)]
    # res = [max(oracle[k] / cxtnl[k], 1) for k in sorted(oracle.keys())]
    # draw_bar_plot(res, sorted(oracle.keys()), "CXL to SMP Slowdown", "Benchmark", "Slowdown", "./tput_slowdown.png")

    for i, name in enumerate(ordered_name):
        if i % 5 == 0:
            print()
        print(name, res[i], res2[i], res3[i])


def local_remote_ratio(exec_time):
    # bm = "record_size_sensitivity"
    # variables = ["-w", "-z", "CC_ALG", "PRIMITIVE", "SIZE_PER_FIELD"]

    bm = "tput_ycsb"
    variables = ["-w", "-z", "CC_ALG"]

    key_word = "847-456"

    if exec_time is None:
        exec_time = get_exec_time(bm, key_word)


    cfgs, args, envs = paper_map[bm]()
    yvals, xval = [], []
    version_management, abort_rate, cc = [], [], []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if not env['PRIMITIVE'] == "CXTNL":
            continue
        # if not cfg['SIZE_PER_FIELD'] in [20, 100]:
        #     continue
        # if not (arg["-w"], arg['-z']) in [(0, 0), (0.7, 0.8)]:
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        name = gen_simplified_name(cfg, arg, env, variables)

        compute = time_parts['time_log'] + time_parts['time_query'] + time_parts['useful_work'] + time_parts['time_index']
        txn_manager_tl = time_parts['time_man'] + time_parts['time_cleanup'] + time_parts['time_wait']
        memory_layer = time_parts['time_shared_record'] + time_parts['time_shared_metadata']

        version_management.append((time_parts['time_shared_record'] + time_parts['time_man'] + time_parts['time_cleanup']) / run_time)
        abort_rate.append(abort_cnt / (txn_cnt + abort_cnt))
        cc.append(cfg['CC_ALG'])

        # yvals.append([compute, txn_manager_tl, memory_layer])
        yvals.append([time_parts['time_shared_record'], time_parts['time_shared_metadata'], compute + txn_manager_tl])
        legend = ['Log-Query-Compute', 'Txn Manager (Thread-Local)', 'Memory Layer']

        # yvals.append([time_parts['time_shared_record'] + time_parts['time_man'] + time_parts['time_cleanup'], time_parts['time_shared_metadata']])
        # legend = ['Version Management', 'Conflict Detection']

        xval.append(name)
        # print(name, sum(time_parts.values()) / run_time)
        # print(memory_layer / (compute + txn_manager_tl + memory_layer))
        print(name, abort_cnt / (txn_cnt + abort_cnt), compute, time_parts['time_man'], time_parts['time_cleanup'], time_parts['time_shared_record'], time_parts['time_shared_metadata'])

    for i, row in enumerate(yvals):
        row_sum = sum(row)
        if row_sum == 0:
            continue  # Avoid division by zero
        yvals[i] = [element / row_sum for element in row]
        # print(yvals[i][2])

    # legend = list(time_parts.keys()) # Detailed breakdown
    draw_stacked_bar_plot(yvals, xval, legend, "CXL TPCC Breakdown", "Benchmark", " ", "./cxl_tpcc_breakdown.png", True)


    return yvals, xval, legend


def mem_cpi(exec_time):

    # bm = "tput_tpcc"
    # variables = ["-n", "-Tp", "CC_ALG", "SNIPER_CXL_LATENCY"]

    bm = "tput_ycsb"
    variables = ["-w", "-z", "CC_ALG", "SNIPER_CXL_LATENCY"]
    key_word = "246"

    exec_time = get_exec_time(bm, key_word)

    # bm = "ep_test"
    # variables = ["CC_ALG", "-n", "-Tp", "PRIMITIVE"]

    cfgs, args, envs = paper_map[bm]()

    cxl_vanilla, cxtnl = {}, {}    # local time + remote time

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if not (cfg['CC_ALG'] == 'SILO'):
            continue
        name = gen_simplified_name(cfg, arg, env, variables)

        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        memstatus_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(memstatus_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        memstatus = parse_memstatus_output(res)

        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        ep_agent_status = parse_ep_agent(log_path)

        total_cxl_mem_access = memstatus['cxl-mem-read-cnt'] + memstatus['cxl-mem-write-cnt']
        total_type3_dram_access = ep_agent_status['benchmark']['read'] + ep_agent_status['benchmark']['write']

        yval = [memstatus['directory-cxl-cache-overhead'] / total_cxl_mem_access, memstatus['directory-cxl-bi-overhead'] / total_cxl_mem_access, (memstatus['cxl-mem-overhead']) / total_cxl_mem_access]

        if env["PRIMITIVE"] == "CXLVANILLA":
            cxl_vanilla[name] = yval
        elif env["PRIMITIVE"] == "CXTNL":
            cxtnl[name] = yval

    ordered_name = sorted(cxtnl.keys(), reverse=True)
    res = [cxl_vanilla.get(k, []) for k in ordered_name]
    res2 = [cxtnl.get(k, []) for k in ordered_name]
    for i, name in enumerate(ordered_name):
        # print(name, end=" ")
        print(" ".join(map(str, res[i])), " ".join(map(str, res2[i])), sep="\n")
        print()


def bus_packets(exec_time):

    bm = "tput_tpcc"
    variables = ["-n", "-Tp", "CC_ALG", "SNIPER_CXL_LATENCY"]

    # bm = "tput_ycsb"
    # variables = ["-w", "-z", "CC_ALG", "SNIPER_CXL_LATENCY"]

    cfgs, args, envs = paper_map[bm]()

    cxl_vanilla, cxtnl = {}, {}    # local time + remote time

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if not (cfg['CC_ALG'] == 'SILO'):
            continue
        name = gen_simplified_name(cfg, arg, env, variables)

        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        memstatus_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(memstatus_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        memstatus = parse_memstatus_output(res)


        cpi_plot_tool = os.path.join(env["SNIPER_ROOT"], "tools/cpistack.py")
        cpi_cmd = "{} {} {}".format("python2", cpi_plot_tool, "--cpi --partial roi-begin:roi-end")
        res = subprocess.Popen(cpi_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        cpi = parse_cpi_output(res)

        def avg(lst):
            return sum(lst) / len(lst)

        cpi = {key: avg(value.values()) for key, value in cpi.items()}

        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        yval = memstatus["directory-evict-cnt"] + memstatus["directory-cxl-cache-overhead"] / env["SNIPER_CXL_LATENCY"]
        print(yval)
        if env["PRIMITIVE"] == "CXLVANILLA":
            cxl_vanilla[name] = yval
        elif env["PRIMITIVE"] == "CXTNL":
            cxtnl[name] = yval

    ordered_name = sorted(cxtnl.keys(), reverse=True)
    res = [cxl_vanilla.get(k, []) for k in ordered_name]
    res2 = [cxtnl.get(k, []) for k in ordered_name]
    for i, name in enumerate(ordered_name):
        # print(" ".join(map(str, res[i])), " ".join(map(str, res2[i])), sep="\n")
        print(name, res[i], res2[i])


# Workload Sensitivity Tests
def record_sensitivity(exec_time):

    bm = "record_size_sensitivity"
    variables = ["-w", "-z", "CC_ALG", "SIZE_PER_FIELD"]

    cxl_vanilla, cxtnl = [], []    # local time + remote time

    cfgs, args, envs = paper_map[bm]()
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        tput = (txn_cnt - abort_cnt) / (run_time / arg['-t']) / arg['-t']

        if env["PRIMITIVE"] == "CXLVANILLA":
            cxl_vanilla.append((name, tput))
        elif env["PRIMITIVE"] == "CXTNL":
            cxtnl.append((name, tput))

        # print(name, sum(time_parts.values()) / run_time)

    # legend = ["Local Time", "Remote Time"]
    # ordered_name = sorted(cxtnl.keys(), reverse=True)
    # res = [cxl_vanilla.get(k, 0) for k in ordered_name]
    # res2 = [cxtnl.get(k, 0) for k in ordered_name]
    import numpy as np

    matrix = np.zeros((10, 4))

    # Fill the matrix column-wise
    for j in range(4):
        for i in range(10):
            matrix[i, j] = cxtnl[i + j * 10][1] / cxl_vanilla[i + j * 10][1]
            # matrix[i, j] = cxl_vanilla[i + j * 10][1]
    for i in range(10):
        for j in range(4):
            print(matrix[i, j], end=' ')
        print()

    # for i, (name, data) in enumerate(cxl_vanilla):
    #     print(name, cxtnl[i][1] / cxl_vanilla[i][1])


def index_sensitivity(exec_time):
    bm = "index_sensitivity_tpcc"
    variables = ["-n", "-Tp", "CC_ALG", "SNIPER_CXL_LATENCY", "INDEX_STRUCT"]

    # bm = "tput_ycsb"
    # variables = ["-w", "-z", "CC_ALG"]

    cfgs, args, envs = paper_map[bm]()

    idx_hash, idx_tree = [], []
    # vanilla, cxtnl = {}, {}
    names = []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        # tput = (txn_cnt - abort_cnt) / (run_time / arg['-t']) / arg['-t']
        tput = time_parts["time_index"]

        # if (env["PRIMITIVE"] == "CXLVANILLA"):
        #     vanilla.append(tput)
        # elif (env["PRIMITIVE"] == "CXTNL"):
        #     cxtnl.append(tput)
        # names.append(name)
        if (cfg["INDEX_STRUCT"] == "IDX_HASH"):
            names.append(name)
            idx_hash.append(tput)
        elif (cfg["INDEX_STRUCT"] == "IDX_BTREE"):
            idx_tree.append(tput)

    # print(names, idx_hash, idx_tree, sep="\n")
    for i, name in enumerate(names):
        print(name, idx_hash[i], idx_tree[i])


def scalability_sensitivity(exec_time):
    bm = "scalibity_tpcc"
    variables = ["-n", "-Tp", "CC_ALG", "SNIPER_CXL_LATENCY", "INDEX_STRUCT", "NNODE"]
    exec_time = get_exec_time(bm, "847")
    # bm = "tput_ycsb"
    # variables = ["-w", "-z", "CC_ALG"]

    cfgs, args, envs = paper_map[bm]()

    idx_hash, idx_tree = [], []
    # vanilla, cxtnl = {}, {}
    names, tputs = [], [[] for _ in range(9)]

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        # if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
        if not (env["SNIPER_CXL_LATENCY"] == 0 and env["SNIPER_MEM_LATENCY"] == 0):
            continue
        arg = arg.copy()

        if arg['-t'] == -1:
            arg['-t'] = env["THREAD_PER_NODE"] * env["NNODE"]
        if "-t_per_wh" in arg:
            assert cfg["WORKLOAD"] == "TPCC"
            arg["-n"] = int(arg['-t'] / arg['-t_per_wh'])
            arg.pop("-t_per_wh")

        # print(env["THREAD_PER_NODE"], env["NNODE"], env["THREAD_PER_NODE"]* env["NNODE"])

        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        tput = (txn_cnt - abort_cnt) / (run_time / arg['-t'])
        names.append(name)
        print(name, tput)
        # tputs.append(tput)
        tputs[env['NNODE'] // 2].append(tput)

    for i, line in enumerate(tputs):
        print(" ".join(map(str, line)))

    # for i, name in enumerate(names):
    #     if re.findall(r'NNODE-1$', name):
    #         print()
    #     print(name, tputs[i])


def ep_agent_breakdown(exec_time):
    # bm = "ep_test"
    bm = "dram_latency_dist"
    variables = ["CC_ALG", "-n", "-Tp", "PRIMITIVE"]

    # bm = "dram_latency_dist_ycsb"
    # variables = ["-w", "-z", "CC_ALG"]
    key_word = "847-456"

    if exec_time is None:
        exec_time = get_exec_time(bm, key_word)

    cfgs, args, envs = paper_map[bm]()
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if not (cfg['CC_ALG'] == 'SILO'):
            continue
        if not (env['PRIMITIVE'] == 'CXTNL'):
            continue
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        memstatus_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(memstatus_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        memstatus = parse_memstatus_output(res)

        ep_agent_status = parse_ep_agent(log_path)

        total_cxl_mem_access = memstatus['cxl-mem-read-cnt'] + memstatus['cxl-mem-write-cnt']
        total_type3_mem_access = memstatus['cxl-mem-type3-read-cnt'] + memstatus['cxl-mem-type3-write-cnt']
        total_type3_read = memstatus['cxl-mem-type3-read-cnt']
        total_type3_dram_access = ep_agent_status['benchmark']['read'] + ep_agent_status['benchmark']['write']

        total_flush = ep_agent_status['benchmark']['flush']

        vf_check = ep_agent_status['view-bloom-filter']['check']
        vf_check_hit = ep_agent_status['view-bloom-filter']['hit']
        vf_insert = ep_agent_status['view-bloom-filter']['add']

        cf_check = ep_agent_status['cache-bloom-filter']['check']
        cf_check_hit = ep_agent_status['cache-bloom-filter']['hit']
        cf_insert = ep_agent_status['cache-bloom-filter']['add']

        vat_check = vf_check_hit
        vat_insert = ep_agent_status['cuckoo-view-table']['inserts']
        vat_remove = ep_agent_status['cuckoo-view-table']['deletes']
        vat_conflict = ep_agent_status['cuckoo-view-table']['conflicts']

        # Only Consider Type3


        # VAT Manipulation
        vat_overhead = 50 * (vat_check + vat_insert + vat_remove)
        # VF Query
        vf_overhead = 4 * (vf_check + vf_insert)
        # CF Query
        cf_overhead = 4 * (cf_check + cf_insert)
        # DRAM Controller
        dram_ctrl_overhead = total_type3_dram_access * 50
        # GSync
        # gsync_overhead = ep_agent_status["benchmark"]["inv"] * 20

        total_cxl_dram_access = (memstatus['cxl-mem-overhead'] - (vat_overhead + vf_overhead + cf_overhead)) / env["SNIPER_MEM_LATENCY"]
        # Protocol Selection
        selection_overhead = total_cxl_dram_access * 2

        cxtnl_overhead = vat_overhead + max(vf_overhead, cf_overhead) + dram_ctrl_overhead + selection_overhead

        # DRAM Access Ratio
        dram_type3_to_cxl = total_type3_dram_access / total_cxl_dram_access
        # CPU Issued Ratio
        mem_type3_to_cxl = total_type3_mem_access / total_cxl_mem_access

        # Access Distribution
        legend = ["Bypass", "Partial Walk", "Complete Walk"]
        total = total_cxl_dram_access
        bypass = total_cxl_dram_access - total_type3_dram_access
        complete_walk = vat_check + vat_insert + vat_remove
        partial_walk = total - bypass - complete_walk

        print(name, (total_type3_read - total_type3_dram_access) / total_type3_read, partial_walk / total_type3_read, complete_walk / total_type3_read)

        print(name, partial_walk / total_type3_dram_access, vat_check / total, (vat_insert + vat_remove) / total)
        # print(name, total_cxl_dram_access / total_cxl_mem_access)
        # print(name, dram_type3_to_cxl, mem_type3_to_cxl, total_type3_dram_access / total_type3_mem_access, total_cxl_dram_access / total_cxl_mem_access)
        # print(name, selection_overhead / cxtnl_overhead, \
        #       max(vf_overhead, cf_overhead) / cxtnl_overhead, \
        #       vat_overhead / cxtnl_overhead, \
        #       dram_ctrl_overhead / cxtnl_overhead)



def cache_hit_ratio(exec_time):

    key_word = "847-456"
    bm = "tput_tpcc"
    variables = ["CC_ALG", "-n", "-Tp", "PRIMITIVE"]
    # bm = "tput_ycsb"
    # variables = ["-w", "-z", "CC_ALG", "PRIMITIVE"]
    # bm = "ep_test"
    # variables = ["CC_ALG", "-n", "-Tp", "PRIMITIVE"]


    if exec_time is None:
        exec_time = get_exec_time(bm, key_word)

    cfgs, args, envs = paper_map[bm]()
    cnt = 0
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if not (env["PRIMITIVE"] == "CXTNL"):
            continue
        # if not (cfg['CC_ALG'] == 'SILO'):
            # continue
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)

        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, bm, exec_time)
        memstatus_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(memstatus_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        memstatus = parse_memstatus_output(res)

        ep_agent_status = parse_ep_agent(log_path)

        total_cxl_mem_access = memstatus['cxl-mem-read-cnt'] + memstatus['cxl-mem-write-cnt']
        total_type3_mem_access = memstatus['cxl-mem-type3-read-cnt'] + memstatus['cxl-mem-type3-write-cnt']
        total_type3_dram_access = ep_agent_status['benchmark']['read'] + ep_agent_status['benchmark']['write']

        total_flush = ep_agent_status['benchmark']['flush']

        vf_check = ep_agent_status['view-bloom-filter']['check']
        vf_check_hit = ep_agent_status['view-bloom-filter']['hit']
        vf_insert = ep_agent_status['view-bloom-filter']['add']

        cf_check = ep_agent_status['cache-bloom-filter']['check']
        cf_check_hit = ep_agent_status['cache-bloom-filter']['hit']
        cf_insert = ep_agent_status['cache-bloom-filter']['add']

        vat_check = vf_check_hit
        vat_insert = ep_agent_status['cuckoo-view-table']['inserts']
        vat_remove = ep_agent_status['cuckoo-view-table']['deletes']
        vat_conflict = ep_agent_status['cuckoo-view-table']['conflicts']

        # Only Consider Type3


        # VAT Manipulation
        vat_overhead = 50 * (vat_check + vat_insert + vat_remove)
        # VF Query
        vf_overhead = 4 * (vf_check + vf_insert)
        # CF Query
        cf_overhead = 4 * (cf_check + cf_insert)
        # DRAM Controller
        dram_ctrl_overhead = total_type3_dram_access * 50
        # GSync
        # gsync_overhead = ep_agent_status["benchmark"]["inv"] * 20

        total_cxl_dram_access = (memstatus['cxl-mem-overhead'] - (vat_overhead + vf_overhead + cf_overhead)) / env["SNIPER_MEM_LATENCY"]
        # Protocol Selection
        selection_overhead = total_cxl_mem_access * 2

        cxtnl_overhead = vat_overhead + max(vf_overhead, cf_overhead) + dram_ctrl_overhead + selection_overhead

        # DRAM Access Ratio
        dram_type3_to_cxl = total_type3_dram_access / total_cxl_dram_access
        # CPU Issued Ratio
        mem_type3_to_cxl = total_type3_mem_access / total_cxl_mem_access

        # Access Distribution
        legend = ["Bypass", "Partial Walk", "Complete Walk"]
        total = total_cxl_dram_access
        bypass = total_cxl_dram_access - total_type3_dram_access
        complete_walk = vat_check + vat_insert + vat_remove
        partial_walk = total - bypass - complete_walk

        print(name, dram_type3_to_cxl)
        cnt += 1
        if not cnt % 5:
            print()

def latency_dist(exec_time):
    bm = "dram_latency_dist"
    variables = ["CC_ALG", "-n", "-Tp", "PRIMITIVE"]
    key_word = "847-456"

    if exec_time is None:
        exec_time = get_exec_time(bm, key_word)

    cfgs, args, envs = paper_map[bm]()
    res = [[] for _ in range(60)]
    names = []
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        # if not (env['PRIMITIVE'] == 'CXLVANILLA'):
        #     continue
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)
        dist = parse_latency_dist(log_path)['hist']
        for i in range(len(dist)):
            res[i].append(dist[i])
        names.append(name)
        # draw_bar_plot(dist['hist'], [i * 3000/60 for i in range(60)], "DRAM Latency Distribution", "Latency (ns)", "Frequency", "./dram_latency_dist.png")

    print(" ".join(names))
    for i in range(60):
        print(" ".join(map(str, res[i])))


def bus_tput(exec_time):
    bm = "bus_bw"
    variables = ["-n", "-t"]
    key_word = "vanilla"

    if exec_time is None:
        exec_time = get_exec_time(bm, key_word)
    cfgs, args, envs = paper_map[bm]()

    # Plot the data
    fig, ax = plt.subplots(figsize=(14, 8))

    def average(lst):
        return sum(lst) / len(lst)
        # return max(lst)
    yvals, xvals, names = [], [], []
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        arg['-t'] = env["THREAD_PER_NODE"] * env["NNODE"]
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        interval = 1e4 / 1e9        # in s
        try:
            bus_traffic = parse_bus_traffic(log_path, interval)
            # Calculate the average, p99, p75, and p25
            avg = np.mean(bus_traffic)
            p99 = np.percentile(bus_traffic, 99)
            p75 = np.percentile(bus_traffic, 75)
            p50 = np.percentile(bus_traffic, 50)
            p25 = np.percentile(bus_traffic, 25)

            print(name, avg, p99, p75, p50, p25)

            bus_traffic = bus_traffic[:len(bus_traffic) // 2]
            avg_field = 5
            bus_traffic = [average(bus_traffic[i:i + avg_field]) for i in range(0, len(bus_traffic), avg_field)]
            bus_traffic = [i for i in bus_traffic]  # 512 bits
            yvals.append(bus_traffic)
            # Set labels and title
            xvals = [i * interval * 1e3 for i in range(len(bus_traffic))]
            names.append(name)
        except: 
            yvals.append([0] * len(bus_traffic))
            names.append(name)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, y in enumerate(yvals):
        ax.plot(xvals, y, label=names[i])

    # ax.plot(xvals, yvals[0], label="16 nodes")
    # ax.plot(xvals, yvals[1], color="black", label="8 nodes")
    # ax.legend(["64 nodes", "8 nodes"], loc='upper center', fontsize='x-large')
    ax.set_xlabel("Timeline (ms)")
    ax.set_ylabel("BW Usage (%)")
    ax.set_ylim(bottom=0)
    ax.set_facecolor("none")  # Set background to no fill
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color="black")
    plt.show()
    plt.tight_layout()
    plt.savefig("./dram_bus_traffic.png")
    plt.clf()


def latency_tput_tpcc(exec_time):
    # bm = "latency_tput_tpcc_tight"
    # variables = ["CC_ALG", "-n", "-Tp", "PRIMITIVE"]
    bm = "latency_tput_ycsb"
    variables = ["-w", "-z", "CC_ALG", "-t"]
    key_word = "847-456"
    # key_word = "OCC"

    if exec_time is None:
        exec_time = get_exec_time(bm, key_word)

    cfgs, args, envs = paper_map[bm]()
    yvals, xvals = [], []
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env['PRIMITIVE'] == 'CXLVANILLA'):
            continue
        arg = arg.copy()
        if arg['-t'] == -1:
            arg['-t'] = env["THREAD_PER_NODE"] * env["NNODE"]
        log_path = get_log_path(cfg, arg, env, bm, exec_time)
        name = gen_simplified_name(cfg, arg, env, variables)
        _, txn_cnt, abort_cnt, run_time, time_parts = parse_log(log_path)
        tput = (txn_cnt - abort_cnt) / (run_time / arg['-t'])
        m = time_parts["median_latency"]
        print(name, tput, m)
        yvals.append(f"{tput} {m}")
        xvals.append(name)

    # for i in range(0, len(yvals), 8):
    #     print(" ".join(map(str, yvals[i:i + 8])))
        # print(" ".join(map(lambda x, yvals[i:i + 8])))
        # print(name, tput, time_parts["median_latency"])


import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot the results of an experiment.")
    parser.add_argument('experiment', type=str, help='The name of the experiment to plot')
    parser.add_argument('--exec_time', type=str, help='The execution time for experiments', default=None)

    args = parser.parse_args()

    # Get the function from globals and call it
    func = globals().get(args.experiment)
    if func and callable(func):
        func(args.exec_time)
    else:
        print(f"No function named {args.experiment} found.")
