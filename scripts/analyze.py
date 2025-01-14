import re, os
import matplotlib.pyplot as plt
import numpy as np

def parse_log(file_path):
    config_info = {}
    txn_count_total = 0
    abort_count_total = 0
    run_time = 0
    time_parts = {}
    with open(file_path, 'r') as file:
        for line in file:
            config_match = re.search(r";\s*(\w+)\s+(\d+\.?\d*)", line)
            if config_match:
                config_key = config_match.group(1)
                config_value = config_match.group(2)
                config_info[config_key] = config_value

            if "txn_cnt" in line and "abort_cnt" in line and "summary" not in line:
                txn_count = int(re.search(r"txn_cnt=(\d+)", line).group(1))
                abort_count = int(re.search(r"abort_cnt=(\d+)", line).group(1))
                txn_count_total += txn_count
                abort_count_total += abort_count

            if "[summary]" in line:
                run_time = float(re.search(r"run_time=(\d+\.\d+)", line).group(1))
                time_parts['time_wait'] = float(re.search(r"time_wait=(\d+\.\d+)", line).group(1))
                time_parts['time_ts_alloc'] = float(re.search(r"time_ts_alloc=(\d+\.\d+)", line).group(1))
                time_parts['time_man'] = float(re.search(r"time_man=(\d+\.\d+)", line).group(1))
                time_parts['time_index'] = float(re.search(r"time_index=(\d+\.\d+)", line).group(1))
                time_parts['time_abort'] = float(re.search(r"time_abort=(\d+\.\d+)", line).group(1))
                time_parts['time_cleanup'] = float(re.search(r"time_cleanup=(\d+\.\d+)", line).group(1))
                time_parts['time_query'] = float(re.search(r"time_query=(\d+\.\d+)", line).group(1))

    return config_info, txn_count_total, abort_count_total, run_time, time_parts

def find_log_files(directory):
    log_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            log_files.append(filename)
    return log_files

def draw_figures(log_data):
    categories = list(next(iter(log_data.values())).keys())
    # colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#000000', '#ff7f0e']


    type_names = list(log_data.keys())
    type_values = np.zeros((len(categories), len(type_names)))

    for i, cat in enumerate(categories):
        type_values[i] = [log_data[name][cat] for name in type_names]

    cumulative_values = np.zeros(len(type_names))

    fig, ax = plt.subplots(figsize=(14, 8))

    for i in range(len(categories)):
        ax.bar(type_names, type_values[i], bottom=cumulative_values, color=colors[i], edgecolor='white', label=categories[i])
        cumulative_values += type_values[i]

    ax.set_xlabel('Transaction Types')
    ax.set_ylabel('Time Spent (s)')
    ax.set_title('Breakdown of Total Time Spent per Transaction Type')
    ax.legend(categories, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig('./breakdown_chart.png')

res = {}
for log_file in sorted(find_log_files(".")):
    wl_name = log_file[:-4]
    config_info, txn_count, abort_count, run_time, time_parts = parse_log(log_file)
    # FIXME: bugs in hstore
    if "HSTORE" in wl_name:
        continue

    # FIXME: time_man contains time_cleanup
    # for key, value in time_parts.items():
    #     if key == "time_man":
    #         time_parts[key] = value - time_parts['time_cleanup']

    # legalization
    summation = sum(time_parts.values())

    # FIXME: usework may be negative
    time_parts['useful_work'] = max(0, run_time - summation)
    print(log_file, time_parts)
    assert time_parts['useful_work'] >= 0

    # for key, val in time_parts.items():
    #     time_parts[key] = val / run_time
    res[wl_name] = time_parts

draw_figures(res)

    # print("Configuration Info:", config_info)
    # print("Total Transaction Count:", txn_count)
    # print("Total Abort Count:", abort_count)
    # print("Run Time:", run_time)
    # print("Time Parts:", time_parts)
    # exit()
    # res = {""}


