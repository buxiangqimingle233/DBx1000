import os, sys, re, os.path
import re
import subprocess, datetime, time, signal
from tqdm import tqdm
from helper import *
import itertools
from experiments import experiment_map, DBMS_CFG

def compile_binary(cfg):
    print("Starting compilation for job: {}".format(str(cfg)))
    os.system("cp " + DBMS_CFG[0] + ' ' + DBMS_CFG[1])
    replace_configs(DBMS_CFG[1], cfg)

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


def run_script_in_docker_container(container_name, script, max_retries=3, timeout_seconds=240):
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


def run_binary(cfg: dict, arg: dict, env: dict):

    sniper = env["SNIPER"]
    arg_value = " ".join(f"{key}{value}" for key, value in arg.items())
    work_name = get_work_name(cfg, arg, env)

    print("Running test for job: {}".format(work_name))

    # Create working directory
    home = os.getcwd()
    result_home = get_result_home(cfg, arg, env)
    if not os.path.exists(result_home):
        os.mkdir(result_home)

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

        recorder_args = []
        recorder_args += ["--no-cache-warming", "-d", result_dir, "--cache-only", "-n", 32]
        recorder_args += ["--roi"]
        recorder_args += ["-c", SNIPER_CONFIG]
        recorder_args += ["-g", f"perf_model/cxl/cxl_cache_roundtrip={SNIPER_CXL_LATENCY}"]

        cmd = sniper_bin + " " + " ".join(map(str, recorder_args)) + " -- " + os.path.join(home, get_executable_name(cfg)) + ' ' + arg_value
        print("Executing command: {}".format(cmd))
        output = run_script_in_docker_container("docker_sniper-dev-container_1", cmd)
        # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # process.wait()

    else:   # run in host mode
        cmd = "./" + get_executable_name(cfg) + ' ' + arg_value

        print("Executing command: {}".format(cmd))
        start = datetime.datetime.now()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        timeout = 10 # in seconds
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
        exit(0)


def run_all(exps):
    cfgs, args, envs = experiment_map[exps]()
    total_jobs = len(cfgs) * len(args) * len(envs)
    progress_bar = tqdm(total=total_jobs, desc="Running Benchmarks", unit="config unit")
    for cfg, arg, env, in itertools.product(cfgs, args, envs):
        if env["SNIPER"]:
            cfg["SNIPER"] = 1
        compile_binary(cfg)
        run_binary(cfg, arg, env)
        progress_bar.update(1)

run_all("hstore_network_sweep")

# os.system('make clean > temp.out 2>&1')
# os.system('rm temp.out')
