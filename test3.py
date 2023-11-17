import os, sys, re, os.path
import platform
import subprocess, datetime, time, signal

def replace(filename, pattern, replacement):
    with open(filename, 'r') as f:
        s = f.read()
    new_s = re.sub(pattern, replacement, s)
    if s != new_s:
        print("Replacing in {}: {} -> {}".format(filename, pattern, replacement))
    with open(filename, 'w') as f:
        f.write(new_s)

jobs = {}
dbms_cfg = ["config-std.h", "config.h"]

# Compiling parameters
cc_algo = ['WAIT_DIE', 'NO_WAIT', 'HEKATON', 'SILO', 'TICTOC', "HSTORE", "OCC"]
log_algo = ['LOG_NO', 'LOG_BATCH']


# Execution parameters

# THREAD_CNT, MAX_TXNS_PER_THREAD, NUM_LOGGER, NUM_WH(TPCC)
# READ_PERC(YCSB), WRITE_PERC(YCSB), ZIPF_THETA(YCSB), REQ_PER_QUERY(YCSB), ROW_CNT(YCSB)
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


def insert_job(cc_algo, log_algo, workload):
    jobs[workload + '_' + cc_algo + '_' + workload] = {
            "WORKLOAD": workload,
            "CC_ALG": cc_algo,
            "LOG_ALGORITHM": log_algo,
        }

def get_exec_name(job):
    return "rundb_" + job['WORKLOAD'] + "_" + job['CC_ALG'] + "_" + job['LOG_ALGORITHM']

def get_log_name(job, exec_name):
    return get_exec_name(job) + "_" + exec_name + ".log"


def test_compile(job):
    print("Starting compilation for job: {}".format(job))
    os.system("cp " + dbms_cfg[0] + ' ' + dbms_cfg[1])
    for param, value in job.items():
        pattern = r"\#define\s*" + re.escape(param) + r'.*'
        replacement = "#define " + param + ' ' + str(value)
        replace(dbms_cfg[1], pattern, replacement)

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


def test_run(test='', job=None):
    print("Running test: {}, for job: {}".format(test, job))
    app_flags = ""
    if test == 'read_write':
        app_flags = "-Ar -t1"
    elif test == 'conflict':
        app_flags = "-Ac -t4"
    
    if job['WORKLOAD'] == 'YCSB':
        args = ycsb_args
    elif job['WORKLOAD'] == 'TPCC':
        args = tpcc_args
    
    for name, arg in args.items(): 
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

        with open(get_log_name(job, name), "w") as f:
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
        test_compile(job)
        if job['WORKLOAD'] == 'TEST':
            test_run('read_write', job)
        else:
            test_run('', job)
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
