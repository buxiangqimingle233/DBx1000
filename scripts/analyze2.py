import re
import os
import matplotlib.pyplot as plt
import numpy as np
from plot_helper import parse_log


def find_log_files(directory):
    log_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            log_files.append(os.path.join(directory, filename))
    return log_files


def legalization_time(time_parts, run_time):
    # legalization
    summation = sum(time_parts.values())
    time_parts['useful_work'] = max(0, run_time - summation)
    for k, v in time_parts.items():
        time_parts[k] = v / max(run_time, summation)
    # FIXME: usework may be negative
    assert time_parts['useful_work'] >= 0


def draw_figures(log_data, title):
    categories = list(next(iter(log_data.values())).keys())
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#000000', '#ff7f0e']
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

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
    ax.set_title(title)
    ax.legend(categories, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(f'./breakdown_chart_{title.replace(" ", "_")}.png')
    plt.close(fig)

tpcc_logs = {}
ycsb_logs = {}
for log_file in sorted(find_log_files("./sniper-results")):
    if "HSTORE" not in log_file:
        continue
    if "TPCC" in log_file:
        _, _, _, runtime, tpcc_logs[log_file] = parse_log(log_file)
        # legalization_time(tpcc_logs[log_file], runtime)
    elif "YCSB" in log_file:
        print(log_file)
        _, _, _, runtime, ycsb_logs[log_file] = parse_log(log_file)
        assert "time_wait" in ycsb_logs[log_file]
        # legalization_time(ycsb_logs[log_file], runtime)


# draw_figures(tpcc_logs, 'TPCC Workload Breakdown')
draw_figures(ycsb_logs, 'YCSB Workload Breakdown')
