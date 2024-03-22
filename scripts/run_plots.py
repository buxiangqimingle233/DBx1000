from experiments import experiment_map
from helper import *
from plot_helper import *
import itertools, subprocess
import os, re


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


def cxl_to_smp_slowdown_tpcc_plot(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()
    oracle, cxl = {}, {}
    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if arg['-n'] not in [4, 24, 48]:
            continue
        log_path = get_log_path(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time)
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
    slowdown, slowdown_benchmarks = cxl_to_smp_slowdown_tpcc_plot(exec_time)
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


def cxl_memstatus_tpcc(exec_time):
    cfgs, args, envs = experiment_map["cxl_to_smp_slowdown_tpcc"]()

    yvals, xval = [], []

    for arg, cfg, env in itertools.product(args, cfgs, envs):
        if not (env["SNIPER_CXL_LATENCY"] > 0 and env["SNIPER_MEM_LATENCY"] > 0):
            continue
        if arg['-Tp'] == 1 or arg['-n'] not in [4, 24, 48]:
            continue

        name = gen_simplified_name(cfg, arg, env, ['-n', 'CC_ALG'])
        
        memstatus_tool = os.path.join(env["SNIPER_ROOT"], "tools/gen_memstatus.py")
        home = get_sniper_result_dir(cfg, arg, env, "cxl_to_smp_slowdown_tpcc", exec_time)
        cpi_cmd = "{} {} {}".format("python2", memstatus_tool, "--partial roi-begin:roi-end")
        res = subprocess.Popen(cpi_cmd, shell=True, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        data = parse_memstatus_output(res)

        # Get the summation of all cores
        yvals.append(data)
        xval.append(name)

    legend = ["load-hit-l1", "load-hit-l3", "load-from-dram", "load-from-remote-cache"]

    # latency = [2, 33, 170, 246]
    latency = [2, 33, 410, 762]
    # latency = [1, 1, 1, 1]
    for i in range(len(yvals)):
        yvals[i] = [yvals[i].get(key, 0) * l for key, l in zip(legend, latency)]     # filter
        yvals[i] = [element / sum(yvals[i]) for element in yvals[i]]

    printed_legend = ["l1l2-hit", "llc-hit", "cxl-dram", "cxl-coherence"]
    draw_stacked_bar_plot(yvals, xval, printed_legend, "CXL TPCC CPI", "Benchmark", "CPI", "./cxl_tpcc_memstatus.png", True)
    return yvals, xval, printed_legend


def cxl_memstatus_cpi_tpcc(exec_time):
    cpi_yvals, cpi_benchmarks, cpi_legend = cxl_cpi_tpcc(exec_time)
    memstatus_yvals, memstatus_benchmarks, memstatus_legend = cxl_memstatus_tpcc(exec_time)
    assert set(cpi_benchmarks) == set(memstatus_benchmarks)

    cpi_dict = dict(zip(cpi_benchmarks, cpi_yvals))
    memstatus_dict = dict(zip(memstatus_benchmarks, memstatus_yvals))

    res_cpi = []
    res_legend = []
    for benchmark in cpi_benchmarks:
        c, m = cpi_dict[benchmark], memstatus_dict[benchmark]
        sync = c[cpi_legend.index("sync")]
        local_cache = m[memstatus_legend.index("l1l2-hit")] + m[memstatus_legend.index("llc-hit")]
        cxl_dram = m[memstatus_legend.index("cxl-dram")]
        cxl_coherence = m[memstatus_legend.index("cxl-coherence")]
        compute = c[cpi_legend.index("compute")]
        # FIXME: Check coherence v.s. sync


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


import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot the results of an experiment.")
    parser.add_argument('experiment', type=str, help='The name of the experiment to plot')
    parser.add_argument('--exec_time', type=str, help='The execution time for experiments')

    args = parser.parse_args()

    # Get the function from globals and call it
    func = globals().get(args.experiment)
    if func and callable(func):
        func(args.exec_time)
    else:
        print(f"No function named {args.experiment} found.")
