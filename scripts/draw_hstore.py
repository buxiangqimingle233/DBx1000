import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_log(file_path):
    config_info = {}
    txn_count_total = 0
    abort_count_total = 0
    run_time = 0
    time_parts = {}
    # print(file_path)
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
                time_parts['time_record'] = float(re.search(r"time_shared_record=(\d+\.\d+)", line).group(1))
                time_parts['time_metadata'] = float(re.search(r"time_shared_metadata=(\d+\.\d+)", line).group(1))
                time_parts['time_man'] = float(re.search(r"time_man=(\d+\.\d+)", line).group(1))
                time_parts['time_index'] = float(re.search(r"time_index=(\d+\.\d+)", line).group(1))
                # time_parts['time_abort'] = float(re.search(r"time_abort=(\d+\.\d+)", line).group(1))
                time_parts['time_cleanup'] = float(re.search(r"time_cleanup=(-?\d+\.\d+)", line).group(1))
                time_parts['time_query'] = float(re.search(r"time_query=(\d+\.\d+)", line).group(1))
                time_parts['time_log'] = float(re.search(r"time_log=(\d+\.\d+)", line).group(1))

    return config_info, txn_count_total, abort_count_total, run_time, time_parts


def find_log_files(directory):
    log_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            log_files.append(os.path.join(directory, filename))
    return log_files


def legalization_time(runtimes):
    for k, v in runtimes.items():
        v = [factor / (v[0] + 1e-10) for factor in v]
        runtimes[k] = v
    return runtimes

def calculate_throughput(num_thread, run_times):
    num_txn = num_thread * 50
    for k, v in run_times.items():
        v = [num_txn / (factor / num_thread) for factor in v]
        run_times[k] = v
    return run_times

def draw_figures(data, title):
    def select_evenly(my_list, num_items):
        step = len(my_list) // num_items
        return [my_list[i] for i in range(0, len(my_list), step)][:num_items]

    max_y, min_y = 0, 1145141919810
    for _, values in data.items():
        max_y = max(max_y, max(values))
        min_y = min(min_y, min(values))
    # Create a new figure with the same size of the example chart
    plt.figure(figsize=(5, 5))

    colors = plt.cm.tab20(np.linspace(0, 1, len(data)))
    # Plot each line with distinct markers and colors as per the example chart
    markers = ['s-', 'o-', '^-', 'D-', 'x-', 'P-', '>-', '<-']
    for (label, values), marker, color in zip(data.items(), markers, colors):
        # Assuming the x-axis represents a range of memory usage reduction levels
        # x_values = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 800, 1000]
        # x_values = select_evenly(x_values, 7)
        values = select_evenly(values, len(x_values))
        plt.plot(x_values, [value for value in values], marker, label=label, color=color)

    # Adding the target line and the shaded area
    # plt.axhline(y=90, color='r', linestyle='-', linewidth=2)
    # plt.text(5.5, 92, 'Target (90%)', color='r', fontsize=9, verticalalignment='bottom')
    # plt.fill_between(x_values, 90, 100, color='red', alpha=0.3)

    # Setting the axis labels and title as per the example chart style
    plt.xlabel('Cache Coherence Roundtrip (ns)')
    plt.ylabel('Throughput (tps)')
    plt.title('Throughput vs. Cache Coherence Roundtrip')

    # Adding a legend similar to the example chart
    plt.legend(loc='upper right', fontsize='small')

    # Adding grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Setting the same x and y axis limits as the example chart
    plt.xlim(min(x_values) * 0.9, max(x_values))
    plt.ylim(min_y * 0.8, max_y * 1.2)

    plt.tight_layout()
    plt.savefig(f'./hstore_{title.replace(" ", "_")}.png')


def list_and_cluster_log_files(directory):
    """
    Traverse a directory and list all files with '.log' extension.
    Cluster them based on the file name before the last underscore segment.
    Order the file names within each cluster based on the number after the last underscore, excluding the '.log' extension.

    :param directory: Path of the directory to be traversed
    :return: Dictionary with keys as cluster names and values as ordered list of file names
    """
    # Dictionary to store the clustered files
    clustered_files = defaultdict(list)

    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                # Check if the file contains an underscore
                if '_' in file:
                    # Split the file name to get the cluster name and the ordering number
                    parts = file.rsplit('_', 1)
                    cluster_name, order_part_with_extension = parts
                    # Remove the '.log' extension from the order part
                    order_part = order_part_with_extension.replace('.log', '')
                    try:
                        # Convert the order part to integer
                        order = int(order_part)
                        clustered_files[cluster_name].append((file, order))
                    except ValueError:
                        # Ignore files where the part after the last underscore is not an integer
                        continue
                else:
                    # For files without an underscore, use the entire filename as the cluster name
                    clustered_files[file].append((file, 0))

    # Sort the files in each cluster based on the order number
    for cluster in clustered_files:
        clustered_files[cluster].sort(key=lambda x: x[1])

    # Return the clustered files without order numbers
    return {cluster: [file for file, order in files] for cluster, files in clustered_files.items()}


breakdown = {}
runtimes = {}
directory_path = './sniper-results'
x_values = [0, 100, 200, 400, 800]
for benchmark, log_files in list_and_cluster_log_files(directory_path).items(): 
    # if "HSTORE" in benchmark and "YCSB" in benchmark:
    if "HSTORE" not in benchmark:
        breakdown[benchmark] = []
        runtimes[benchmark] = []
        for log_file in log_files:
            _, _, _, r, l = parse_log(os.path.join(directory_path, log_file))
            breakdown[benchmark].append(l)
            runtimes[benchmark].append(r)


# runtimes = legalization_time(runtimes)
# draw_figures(runtimes, 'runtime')
throughputs = calculate_throughput(16, runtimes)
draw_figures(throughputs, 'throughput')
