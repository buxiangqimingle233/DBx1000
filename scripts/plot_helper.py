import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys, os

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

            if "[summary]" in line:
                txn_count_match = re.search(r"txn_cnt=(\d+)", line)
                txn_count_total += float(txn_count_match.group(1)) if txn_count_match else 0

                abort_count_match = re.search(r"abort_cnt=(\d+)", line)
                abort_count_total += float(abort_count_match.group(1)) if abort_count_match else 0

                run_time_match = re.search(r"run_time=(\d+\.\d+)", line)
                run_time = float(run_time_match.group(1)) if run_time_match else 0

                # Breakdown
                for key in ['time_wait', 'time_ts_alloc', 'time_record', 'time_shared_metadata', 'time_man', 'time_index', 'time_cleanup', 'time_query', 'time_log']:
                    match = re.search(rf"{key}=(-?\d+\.\d+)", line)
                    time_parts[key] = float(match.group(1)) if match else 0

    return config_info, txn_count_total, abort_count_total, run_time, time_parts


def draw_line_plot(yval, xval, vval, title, xlabel, ylabel, vlabel, save_path):
    assert len(yval) == len(vval)
    colors = plt.cm.tab20(np.linspace(0, 1, len(vval)))
    markers = ['s-', 'o-', '^-', 'D-', 'x-', 'P-', '>-', '<-', '8-', 'p-', '*-']

    fig, ax = plt.subplots(figsize=(14, 8))
    for i in range(len(vval)):
        ax.plot(xval, yval[i], markers[i], color=colors[i], label=f'{vlabel}={vval[i]}')

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(vval, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path)

    plt.clf()


def print_csv(yval, xval, vval):
    # Print xvals, vvals, and yvals to the terminal in CSV format
    writer = csv.writer(sys.stdout, delimiter='\n')
    writer.writerow(['xvals'] + xval)
    writer = csv.writer(sys.stdout, delimiter=' ')
    writer.writerow([])

    writer.writerow(vval)
    # Transpose yvals and print each row
    yvals_transposed = list(map(list, zip(*yval)))
    for i, yvals_row in enumerate(yvals_transposed):
        writer.writerow(yvals_row)

# def parse_log(file_path):
#     config_info = {}
#     txn_count_total = 0
#     abort_count_total = 0
#     run_time = 0
#     time_parts = {}
#     # print(file_path)
#     with open(file_path, 'r') as file:
#         for line in file:
#             config_match = re.search(r";\s*(\w+)\s+(\d+\.?\d*)", line)
#             if config_match:
#                 config_key = config_match.group(1)
#                 config_value = config_match.group(2)
#                 config_info[config_key] = config_value

#             if "txn_cnt" in line and "abort_cnt" in line and "summary" not in line:
#                 txn_count = int(re.search(r"txn_cnt=(\d+)", line).group(1))
#                 abort_count = int(re.search(r"abort_cnt=(\d+)", line).group(1))
#                 txn_count_total += txn_count
#                 abort_count_total += abort_count

#             if "[summary]" in line:
#                 run_time = float(re.search(r"run_time=(\d+\.\d+)", line).group(1))
#                 time_parts['time_wait'] = float(re.search(r"time_wait=(\d+\.\d+)", line).group(1))
#                 time_parts['time_ts_alloc'] = float(re.search(r"time_ts_alloc=(\d+\.\d+)", line).group(1))
#                 time_parts['time_record'] = float(re.search(r"time_shared_record=(\d+\.\d+)", line).group(1))
#                 time_parts['time_metadata'] = float(re.search(r"time_shared_metadata=(\d+\.\d+)", line).group(1))
#                 time_parts['time_man'] = float(re.search(r"time_man=(\d+\.\d+)", line).group(1))
#                 time_parts['time_index'] = float(re.search(r"time_index=(\d+\.\d+)", line).group(1))
#                 # time_parts['time_abort'] = float(re.search(r"time_abort=(\d+\.\d+)", line).group(1))
#                 time_parts['time_cleanup'] = float(re.search(r"time_cleanup=(-?\d+\.\d+)", line).group(1))
#                 time_parts['time_query'] = float(re.search(r"time_query=(\d+\.\d+)", line).group(1))
#                 time_parts['time_log'] = float(re.search(r"time_log=(\d+\.\d+)", line).group(1))

#     return config_info, txn_count_total, abort_count_total, run_time, time_parts

parse_log("./env.sh")
