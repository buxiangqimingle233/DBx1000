import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

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
                txn_count_total += float(abort_count_match.group(1)) if abort_count_match else 0

                abort_count_match = re.search(r"abort_cnt=(\d+)", line)
                abort_count_total += float(abort_count_match.group(1)) if abort_count_match else 0

                run_time_match = re.search(r"run_time=(\d+\.\d+)", line)
                run_time = float(run_time_match.group(1)) if run_time_match else 0

                # Breakdown
                for key in ['time_wait', 'time_ts_alloc', 'time_shared_record', 'time_shared_metadata', 'time_man', 'time_index', 'time_cleanup', 'time_query', 'time_log']:
                    match = re.search(rf"{key}=(-?\d+\.\d+)", line)
                    time_parts[key] = float(match.group(1)) if match else 0
                
                # if run_time > sum(time_parts.values()):
                time_parts['useful_work'] = max(run_time - sum(time_parts.values()), 0)


    return config_info, txn_count_total, abort_count_total, run_time, time_parts


def gen_simplified_name(cfg, arg, env, viariables):
    arg_name = {k: arg[k] for k in viariables if k in arg}
    cfg_name = {k: cfg[k] for k in viariables if k in cfg}
    name = "_".join([f"{v}" for k, v in cfg_name.items()]) + "_".join([f"{v}" for k, v in arg_name.items()])
    return name


def draw_line_plot(yval, xval, vval, title, ylabel, xlabel, vlabel, save_path):
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
    # ax.set_xlim(left=0, right=1.25 * max(xval))
    ax.set_ylim(bottom=0)
    ax.legend(vval, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path)

    plt.clf()


def draw_bar_plot(yval, xval, title, x_label, y_label, save_path):
    assert len(yval) == len(xval)
    fig, ax = plt.subplots(dpi=300, figsize=(14, 8))
    ax.bar(xval, yval)

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Rotate x-axis labels
    plt.xticks(rotation=90, fontsize=8)
    # Adjust the bottom margin to make room for the rotated x-axis labels
    plt.subplots_adjust(bottom=0.35)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


def draw_stacked_bar_plot(yvals, xval, legend, title, x_label, y_label, save_path, regularized=True):
    """
    Draws a stacked bar plot with the given parameters.

    :param yvals: A list of lists, where each sublist contains the heights of each segment in a bar.
    :param xval: A list of x-axis categories.
    :param legend: A list of legend entries.
    :param title: The title of the plot.
    :param x_label: The label for the x-axis.
    :param y_label: The label for the y-axis.
    :param save_path: Path where to save the resulting plot image.
    """
    yvals = list(zip(*yvals))
    fig, ax = plt.subplots(dpi=300, figsize=(14, 8))

    # Get the Set3 colormap
    cmap = plt.cm.get_cmap('Set1')
    colors = cmap(range(len(legend)))

    # Bottom of the stacked bar starts at 0
    bottoms = [0] * len(xval)

    # Plot each stack in the bar
    for i, (vals, leg) in enumerate(zip(yvals, legend)):
        ax.bar(xval, vals, label=leg, bottom=bottoms, edgecolor='white', color=colors[i])
        # The next stack starts on top of the previous one
        bottoms = [left + height for left, height in zip(bottoms, vals)]

    if regularized:
        # Set the upper y-axis limit to 1
        ax.set_ylim(0, 1)

    # Adding labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Rotate x-axis labels
    plt.xticks(rotation=90, fontsize=8)
    # Adjust the bottom margin to make room for the rotated x-axis labels
    plt.subplots_adjust(bottom=0.35)
    # Add legend
    ax.legend()

    # Show the plot for verification
    plt.savefig(save_path)
    plt.clf()


def draw_stacked_bar_plot_subclass(yvals, xval, legend, title, x_label, y_label, save_path, regularized=True):
    """
    Draws a stacked bar plot with the given parameters.

    :param yvals: A list of lists, where each sublist contains the heights of each segment in a bar.
    :param xval: A list of x-axis categories.
    :param legend: A list of legend entries.
    :param title: The title of the plot.
    :param x_label: The label for the x-axis.
    :param y_label: The label for the y-axis.
    :param save_path: Path where to save the resulting plot image.
    """
    yvals = list(zip(*yvals))
    fig, ax = plt.subplots(dpi=300, figsize=(14, 8))

    # Bottom of the stacked bar starts at 0
    bottoms = [0] * len(xval)

    # Plot each stack in the bar
    for i, (vals, leg) in enumerate(zip(yvals, legend)):
        ax.bar(xval, vals, label=leg, bottom=bottoms, edgecolor='white')
        # The next stack starts on top of the previous one
        bottoms = [left + height for left, height in zip(bottoms, vals)]

    if regularized:
        # Set the upper y-axis limit to 1
        ax.set_ylim(0, 1)

    # Adding labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xticks(range(3))
    ax.set_xticklabels([i for i in range(3)])

    # Rotate x-axis labels
    plt.xticks(rotation=90, fontsize=8)
    # Adjust the bottom margin to make room for the rotated x-axis labels
    plt.subplots_adjust(bottom=0.35)
    # Add legend
    ax.legend()

    # Show the plot for verification
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

def parse_cpi_output(text):
    lines = text.splitlines()
    headers = re.split('\s\s+', lines[0].strip())[1:]  # Skip the first word ("Time (ms)")
    data = {}
    for line in lines[1:]:
        if line:
            words = line.split()
            data[words[0]] = {header: float(value) for header, value in zip(headers, words[1:])}
    # Transpose
    # data = {header: {key: value[header] for key, value in data.items()} for header in headers}
    return data

def parse_memstatus_output(text):
    # print(text)
    lines = text.splitlines()
    data = {}
    for line in lines:
        k, v = line.split(' ')
        data[k] = float(v)
    return data
