import os, os.path
import subprocess, datetime, time, signal
from tqdm import tqdm
import helper
from helper import get_work_name, get_executable_name, get_result_home, replace_configs
import itertools
# from experiments import experiment_map, DBMS_CFG, SIM_API
from paperexps import experiment_map, DBMS_CFG, SIM_API
import pickle

def compile_binary(cfg, env):
    print("Starting compilation for job: {}".format(str(cfg)))
    os.system("cp " + DBMS_CFG[0] + ' ' + DBMS_CFG[1])
    replace_configs(DBMS_CFG[1], cfg)

    os.system("cp " + SIM_API[0] + ' ' + SIM_API[1])
    if env["SNIPER"] and "PRIMITIVE" in env:
        primitve_choice = {"CXLVANILLA": 0, "CXTNL": 0}
        primitve_choice[env["PRIMITIVE"]] = 1
        replace_configs(SIM_API[1], primitve_choice)

    os.system("make clean > temp.out 2>&1")
    print("Running 'make' for the job...")
    ret = os.system("make -j8 > temp.out 2>&1")
    if ret != 0:
        print("ERROR in compiling job=")
        print(cfg)
        exit(0)

    new_exec_name = get_executable_name(cfg)
    if os.path.exists("rundb"):
        os.rename("rundb", new_exec_name)
        print("Compiled executable renamed to: {}".format(new_exec_name))
    else:
        print("ERROR: Compiled executable 'rundb' not found.")
        exit(0)

    print("PASS Compile\t\talg=%s,\tworkload=%s" % (cfg['CC_ALG'], cfg['WORKLOAD']))


def run_script_in_docker_container(container_name, script, max_retries=3, timeout_seconds=1800):
    # Check if the Docker container is running
    cmd = f"docker ps --filter name={container_name} --format '{{{{.Names}}}}'"
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()

    # If the Docker container is running, run the script inside the Docker container
    if output == container_name:
        for attempt in range(max_retries):
            cmd = f"docker exec -w {os.getcwd()} {container_name} {script}"
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                # Wait for the process to complete or timeout
                output, _ = process.communicate(timeout=timeout_seconds)
                return output
            except subprocess.TimeoutExpired:
                print(f"Attempt {attempt + 1}: Command timed out. Retrying...")
                process.kill()
                process.wait()

        # All attempts failed
        print(f"All attempts failed for command: {cmd}")
        print(f"Task Name: {script}")
        return None
    else:
        print(f"The Docker container {container_name} is not running.")
        return None


def run_binary(cfg: dict, arg: dict, env: dict, exp, retry=0):

    cfg = cfg.copy()
    arg = arg.copy()
    env = env.copy()

    sniper = env["SNIPER"]

    if arg['-t'] == -1:
        arg['-t'] = env["THREAD_PER_NODE"] * env["NNODE"]
    if "-t_per_wh" in arg:
        assert cfg["WORKLOAD"] == "TPCC"
        arg["-n"] = int(arg['-t'] / arg['-t_per_wh'])
        arg.pop("-t_per_wh")

    arg_value = " ".join(f"{key}{value}" for key, value in arg.items())
    work_name = get_work_name(cfg, arg, env)
    result_home = get_result_home(cfg, arg, env, exp, helper.strnow)
    os.makedirs(result_home, exist_ok=True)

    print("Running test for job: {}".format(work_name))

    # Create working directory
    home = os.getcwd()
    # return
    if sniper:  # run in sniper mode (in docker)
        result_dir = os.path.join(result_home, work_name)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        # assert os.getenv("SNIPER_ROOT") != None, "SNIPER_ROOT is not set!"
        assert "SNIPER_ROOT" in env, "SNIPER_ROOT is not set!"
        SNIPER_ROOT = env["SNIPER_ROOT"]
        sniper_bin = os.path.join(SNIPER_ROOT, "run-sniper")
        # COPY TO CWD
        SNIPER_CONFIG = env["SNIPER_CONFIG"]
        SNIPER_CXL_LATENCY = env["SNIPER_CXL_LATENCY"]
        SNIPER_MEM_LATENCY = env["SNIPER_MEM_LATENCY"]
        NTHREAD_PER_NODE = env["THREAD_PER_NODE"]
        NNODE = env["NNODE"]
        total_threads = NTHREAD_PER_NODE * NNODE
        if total_threads % 64 != 0:
            total_threads = total_threads + (64 - (total_threads % 64))  # Sniper Bugs: Make sure perf_model/dram/controllers_interleaving is a multiple of perf_model/l3_cache/shared_cores

        recorder_args = []
        # recorder_args += ["--no-cache-warming", "-d", result_dir, "--cache-only", "-n", total_threads]
        # recorder_args += ["-d", result_dir, "-n", 32]
        recorder_args += ["-d", result_dir, "--cache-only"]
        recorder_args += ["--roi"]
        recorder_args += ["-c", SNIPER_CONFIG]
        recorder_args += ["-g", f"perf_model/cxl/cxl_cache_roundtrip={SNIPER_CXL_LATENCY}"]
        recorder_args += ["-g", f"perf_model/cxl/cxl_mem_roundtrip={SNIPER_MEM_LATENCY}"]
        recorder_args += ["-g", f"general/total_cores={total_threads}"]
        recorder_args += ["-g", f"perf_model/dram/num_controllers={NNODE}"]

        cmd = sniper_bin + " " + " ".join(map(str, recorder_args)) + " -- " + os.path.join(home, get_executable_name(cfg)) + ' ' + arg_value
        print("Executing command: {}".format(cmd))
        output = run_script_in_docker_container("docker_sniper-dev-container_1", "pkill -f snipersim")
        output = run_script_in_docker_container("docker_sniper-dev-container_1", cmd)
        # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # process.wait()

    else:   # run in host mode
        cmd = "./" + get_executable_name(cfg) + ' ' + arg_value

        print("Executing command: {}".format(cmd))
        start = datetime.datetime.now()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        timeout = 100 # in seconds
        while process.poll() is None:
            time.sleep(1)
            now = datetime.datetime.now()
            if (now - start).seconds > timeout:
                os.kill(process.pid, signal.SIGKILL)
                os.waitpid(-1, os.WNOHANG)
                print("ERROR. Timeout cmd=%s" % cmd)
                exit(0)
        output, _ = process.communicate()

    with open(os.path.join(result_home, work_name + ".log"), "w") as f:
        f.write(output.decode())

    if "PASS" in output.decode():
        print("PASS execution. \talg=%s,\tworkload=%s(%s)" %
            (cfg["CC_ALG"], cfg["WORKLOAD"], work_name))
    else:
        print("FAILED execution. cmd = %s" % cmd)
        if retry < 3:
            print("Retrying: %s %sth..." % (cmd, retry))
            run_binary(cfg, arg, env, exp, retry + 1)
        else:
            raise Exception("Failed to run the binary. cmd = %s" % cmd)
        # exit(0)


def load_state():
    # Load the runtime state from a pickle file
    with open('runtime_state.pkl', 'rb') as f:
        state = pickle.load(f)
    print(f"Recovered state: {state}")
    return state


def run_all(exps, recover=False):
    jumpto = 0
    if recover:
        assert os.path.exists('runtime_state.pkl'), "No runtime state found. Please run the script without the --recover flag."
        state = load_state()
        jumpto = int(state['cnt'])
        helper.strnow = str(state['time'])
    try:
        cfgs, args, envs = experiment_map[exps]()
        total_jobs = len(cfgs) * len(args) * len(envs)
        progress_bar = tqdm(total=total_jobs, desc="Running Benchmarks", unit="config unit")

        for cfg, arg, env, in itertools.product(cfgs, args, envs):
            if env["SNIPER"]:
                cfg["SNIPER"] = 1
            if recover and progress_bar.n < jumpto:
                progress_bar.update(1)
                continue
            compile_binary(cfg, env)
            run_binary(cfg, arg, env, exps)
            progress_bar.update(1)

            with open('runtime_state.pkl', 'wb') as f:
                pickle.dump({'cnt': progress_bar.n, 'time': helper.strnow}, f)

    except Exception as e:
        print(f"An error occurred: {e}")
        # Save the runtime state to a pickle file
        with open('runtime_state.pkl', 'wb') as f:
            pickle.dump({'cnt': progress_bar.n, 'time': helper.strnow}, f)
        raise e


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('experiment', type=str, help='The name of the experiment to run')
    parser.add_argument('--recover', action='store_true', help='Recover from a previous run')

    args = parser.parse_args()

    if args.experiment in experiment_map:
        run_all(args.experiment, args.recover)
    else:
        print(f"Experiment {args.experiment} not found. Available experiments are: {list(experiment_map.keys())}")

# os.system('make clean > temp.out 2>&1')
# os.system('rm temp.out')
