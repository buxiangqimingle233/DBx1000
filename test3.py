import os, sys, re, os.path
import re
import platform
import subprocess, datetime, time, signal

SNIPER=True

jobs = {}
dbms_cfg = ["config-std.h", "config.h"]

# Compiling parameters
cc_algo = ['WAIT_DIE', 'NO_WAIT', 'HEKATON', 'SILO', 'TICTOC', "MVCC", "OCC"]
log_algo = ['LOG_NO', 'LOG_BATCH']

# Execution parameters
if SNIPER: 
    # THREAD_CNT, MAX_TXNS_PER_THREAD, NUM_LOGGER, NUM_WH(TPCC)
    # READ_PERC(YCSB), WRITE_PERC(YCSB), ZIPF_THETA(YCSB), REQ_PER_QUERY(YCSB), ROW_CNT(YCSB)
    tpcc_args = {
        "WH1": '-t16 -Gx50 -Ln4 -n1 -r0.9 -w0.1 -z0 -R16 -s10485760',
        "WH4": '-t16 -Gx50 -Ln4 -n4 -r0.9 -w0.1 -z0 -R16 -s10485760',
        "WH16": '-t16 -Gx50 -Ln4 -n16 -r0.9 -w0.1 -z0 -R16 -s10485760',
    }
        
    # contended TPCC (1:16), contended TPCC (1:4), uncontended TPCC (1:1)
    ycsb_args = {
        "UCR": '-t16 -Gx50 -Ln4 -n1 -r0.95 -w0.05 -z0 -R16 -s10485760',
        "CR": '-t16 -Gx50 -Ln4 -n1 -r0.95 -w0.05 -z0.8 -R16 -s10485760', 
        "UCW": '-t16 -Gx50 -Ln4 -n1 -r0.05 -w0.95 -z0 -R16 -s10485760', 
        "CW": '-t16 -Gx50 -Ln4 -n1 -r0.05 -w0.95 -z0.8 -R16 -s10485760', 
    }   # Write-Intensive Uncontended, Write-Intensive Conteded, Read-Intensive Contended, Read-Intensive Uncontended
else: 
    tpcc_args = {
        "WH1": '-t16 -Gx1000 -Ln4 -n1 -r0.9 -w0.1 -z0 -R16 -s10485760',
        "WH4": '-t16 -Gx1000 -Ln4 -n4 -r0.9 -w0.1 -z0 -R16 -s10485760',
        "WH16": '-t16 -Gx1000 -Ln4 -n16 -r0.9 -w0.1 -z0 -R16 -s10485760',
    }
        
    # contended TPCC (1:16), contended TPCC (1:4), uncontended TPCC (1:1)
    ycsb_args = {
        "UCR": '-t16 -Gx1000 -Ln4 -n1 -r0.95 -w0.05 -z0 -R16 -s10485760',
        "CR": '-t16 -Gx1000 -Ln4 -n1 -r0.95 -w0.05 -z0.8 -R16 -s10485760', 
        "UCW": '-t16 -Gx1000 -Ln4 -n1 -r0.05 -w0.95 -z0 -R16 -s10485760', 
        "CW": '-t16 -Gx1000 -Ln4 -n1 -r0.05 -w0.95 -z0.8 -R16 -s10485760', 
    }   # Write-Intensive Uncontended, Write-Intensive Conteded, Read-Intensive Contended, Read-Intensive Uncontended


def replace(filename, pattern, replacement):
    with open(filename, 'r') as f:
        s = f.read()
    new_s = re.sub(pattern, replacement, s)
    if s != new_s:
        print("Replacing in {}: {} -> {}".format(filename, pattern, replacement))
    with open(filename, 'w') as f:
        f.write(new_s)

def insert_job(cc_algo, log_algo, workload):
    jobs[workload + '_' + cc_algo + '_' + workload] = {
            "WORKLOAD": workload,
            "CC_ALG": cc_algo,
            "LOG_ALGORITHM": log_algo,
            "CPU_FREQ": CPU_FREQ,
        }


def get_exec_name(job):
    return "rundb_" + job['WORKLOAD'] + "_" + job['CC_ALG'] + "_" + job['LOG_ALGORITHM']

def get_log_name(job, run_args):
    return get_exec_name(job) + "_" + run_args + ".log"

def get_sniper_work_name(job, run_args):
    return "sniper_" + get_exec_name(job) + "_" + run_args

def test_compile(job, sniper=True):
    print("Starting compilation for job: {}".format(job))
    os.system("cp " + dbms_cfg[0] + ' ' + dbms_cfg[1])
    for param, value in job.items():
        pattern = r"\#define\s*" + re.escape(param) + r'.*'
        replacement = "#define " + param + ' ' + str(value)
        replace(dbms_cfg[1], pattern, replacement)

    if sniper:
        replace(dbms_cfg[1], r"\#define\s*SNIPER.*", "#define SNIPER 1")

    os.system("make clean > temp.out 2>&1")
    print("Running 'make' for the job...")
    ret = os.system("make -j8 > temp.out 2>&1")
    if ret != 0:
        print("ERROR in compiling job=")
        print(job)
        exit(0)
    
    new_exec_name = get_exec_name(job)
    if os.path.exists("rundb"):
        os.rename("rundb", new_exec_name)
        print("Compiled executable renamed to: {}".format(new_exec_name))
    else:
        print("ERROR: Compiled executable 'rundb' not found.")
        exit(0)

    print("PASS Compile\t\talg=%s,\tworkload=%s" % (job['CC_ALG'], job['WORKLOAD']))


def run_script_in_docker_container(container_name, script):
    # Check if the Docker container is running
    cmd = f"docker ps --filter name={container_name} --format '{{{{.Names}}}}'"
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    # If the Docker container is running, run the script inside the Docker container
    if output == container_name:
        cmd = f"docker exec -w {os.getcwd()} {container_name} {script}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        output, _ = process.communicate()
        return output
    else:
        print(f"The Docker container {container_name} is not running.")


def test_run(test='', job=None, sniper=True):

    app_flags = ""
    if test == 'read_write':
        app_flags = "-Ar -t1"
    elif test == 'conflict':
        app_flags = "-Ac -t4"
    
    if job['WORKLOAD'] == 'YCSB':
        args = ycsb_args
    elif job['WORKLOAD'] == 'TPCC':
        args = tpcc_args

    for name, arg in args.items():  # run in sniper mode
        print("Running test: {}, for job: {}".format(test, get_log_name(job, name)))
        
        # Create working directory  
        home = os.getcwd()
        if sniper: 
            result_home = os.path.join(home, "sniper-results")
        else:
            result_home = os.path.join(home, "host-results")
        if not os.path.exists(result_home):
            os.mkdir(result_home)

        if sniper:  # run in sniper mode (in docker)
            result_dir = os.path.join(result_home, get_sniper_work_name(job, name))
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            assert os.getenv("SNIPER_ROOT") != None, "SNIPER_ROOT is not set!"
            SNIPER_ROOT = os.getenv("SNIPER_ROOT")
            sniper_bin = os.path.join(SNIPER_ROOT, "run-sniper")
            sniper_config = os.path.join(SNIPER_ROOT, "config/cxl_asplos.cfg")

            recorder_args = []
            recorder_args += ["--no-cache-warming", "-d", result_dir, "--cache-only", "-n", 32]
            recorder_args += ["--roi"]
            recorder_args += ["-c", sniper_config]

            cmd = sniper_bin + " " + " ".join(map(str, recorder_args)) + " -- " + os.path.join(home, get_exec_name(job)) + ' ' + arg
            # print("Executing command: {}".format(cmd))
            output = run_script_in_docker_container("docker_sniper-dev-container_1", cmd)

            # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            # process.wait()
            
        else:   # run in host mode
            cmd = "./" + get_exec_name(job) + ' ' + arg

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
 
        with open(os.path.join(result_home, get_log_name(job, name)), "w") as f:
            f.write(output.decode())

        if "PASS" in output.decode():
            if test != '':
                print("PASS execution. \talg=%s,\tworkload=%s(%s)" %
                    (job["CC_ALG"], job["WORKLOAD"], test))
            else:
                print("PASS execution. \talg=%s,\tworkload=%s" %
                    (job["CC_ALG"], job["WORKLOAD"]))
        else: 
            print("FAILED execution. cmd = %s" % cmd)
            exit(0)


def run_all_test(jobs):
    for jobname, job in jobs.items():
        # test_compile(job, sniper=SNIPER)
        if job['WORKLOAD'] == 'TEST':
            test_run('read_write', job, sniper=SNIPER)
        else:
            test_run('', job)
            pass
    jobs = {}

# run YCSB tests
jobs = {}
for cc in cc_algo:
    for log in log_algo: 
        insert_job(cc, log, 'YCSB')
run_all_test(jobs)

# run TPCC tests
jobs = {}
for cc in cc_algo:
    for log in log_algo: 
        insert_job(cc, log, 'TPCC')
run_all_test(jobs)

os.system('make clean > temp.out 2>&1')
os.system('rm temp.out')
