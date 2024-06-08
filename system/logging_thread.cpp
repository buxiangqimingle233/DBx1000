#include "logging_thread.h"
#include "manager.h"
#include "wl.h"
#include "log.h"
#include <sys/types.h>
#include <aio.h>
#include <fcntl.h>
#include <errno.h>
#include <sstream>
#include "numa.h"

#if LOG_ALGORITHM != LOG_NO
#define UPDATE_RECOVER_LSN_DIRECT                                                                                \
	if (g_zipf_theta <= CONTENTION_THRESHOLD  || PER_WORKER_RECOVERY)                                                                    \
	{                                                                                                            \
		uint64_t rlv = UINT64_MAX;                                                                               \
		for (uint64_t i = 0; i < num_worker / POOL_SE_SPACE; i++)                                                \
		{                                                                                                        \
			if (SPSCPoolEnd[i * POOL_SE_SPACE] > SPSCPoolStart[i * POOL_SE_SPACE])                               \
			{                                                                                                    \
				uint64_t headLSN = SPSCPools[i][SPSCPoolStart[i * POOL_SE_SPACE] % g_poolsize_wait]->LSN[0] - 1; \
				if (headLSN < rlv)                                                                               \
					rlv = headLSN;                                                                               \
			}                                                                                                    \
			else                                                                                                 \
			{                                                                                                    \
				uint64_t temp = *log_manager->maxLVSPSC[logger_id][i];                                           \
				if (temp < rlv)                                                                                  \
					rlv = temp;                                                                                  \
			}                                                                                                    \
		}                                                                                                        \
		uint64_t tl = *log_manager->recoverLVSPSC_min[logger_id];                                                \
		if (tl < rlv)                                                                                            \
			ATOM_CAS(*log_manager->recoverLVSPSC_min[logger_id], tl, rlv);                                       \
	}

#define UPDATE_RECOVER_LSN_INDIRECT                                         \
	if (g_zipf_theta <= CONTENTION_THRESHOLD  || PER_WORKER_RECOVERY)                               \
	{                                                                       \
		uint64_t rlv = UINT64_MAX;                                          \
		for (uint64_t i = 0; i < num_worker / POOL_SE_SPACE; i++)           \
		{                                                                   \
			register auto rlvi = *log_manager->recoverLVSPSC[logger_id][i]; \
			if (rlv > rlvi)                                                 \
				rlv = rlvi;                                                 \
		}                                                                   \
		uint64_t tl = *log_manager->recoverLVSPSC_min[logger_id];           \
		if (tl < rlv)                                                       \
			ATOM_CAS(*log_manager->recoverLVSPSC_min[logger_id], tl, rlv);  \
	}

#if PER_WORKER_RECOVERY
#define UPDATE_RECOVER_LSN UPDATE_RECOVER_LSN_INDIRECT
#else
#define UPDATE_RECOVER_LSN UPDATE_RECOVER_LSN_DIRECT
#endif

#define BYPASS_WORKER false
// This switch is used to test the raw throughput of the log reader.

LoggingThread::LoggingThread()
{
}

void printLV(uint64_t *lv)
{
	cout << "LV:" << endl;
	for (uint i = 0; i < g_num_logger; i++)
	{
		cout << lv[i] << "  ";
	}
	cout << endl;
}

void LoggingThread::init()
{
	poolDone = false;
}

RC LoggingThread::run()
{
	//pthread_barrier_wait( &warmup_bar );
	if (LOG_ALGORITHM == LOG_BATCH && g_log_recover)
	{
		pthread_barrier_wait(&log_bar);
		return FINISH;
	}
	//uint64_t logging_start = get_sys_clock();

	glob_manager->set_thd_id(_thd_id);
	LogManager *logger;
	uint32_t logger_id = GET_THD_ID % g_num_logger;
	logger = log_manager[logger_id];
	

#if AFFINITY
	//#if LOG_ALGORITHM == LOG_TAURUS || LOG_ALGORITHM == LOG_BATCH

	uint64_t node_id = logger_id % NUMA_NODE_NUM;
	//uint64_t in_node_id = logger_id % logger_per_node;
	//uint64_t workers_per_logger = g_thread_cnt / g_num_logger;
	set_affinity((logger_id / NUMA_NODE_NUM) + node_id * NUM_CORES_PER_SLOT); // first CPU per socket
	//set_affinity(logger_id); // first CPU per socket
	printf("Setting logger %u to CPU node %lu\n", logger_id, (logger_id / NUMA_NODE_NUM) + node_id * NUM_CORES_PER_SLOT);
	int cpu = sched_getcpu();
	int node = numa_node_of_cpu(cpu);
	assert((uint64_t)node == node_id);
#endif
	// PAUSE // try to make affinity effective
	init();
	//#endif
	pthread_barrier_wait(&log_bar);
	uint64_t starttime = get_sys_clock();
	uint64_t total_log_data = 0;
	uint64_t flushcount = 0;
	if (g_log_recover)
	{ // recover
		while (true)
		{ //glob_manager->get_workload()->sim_done < g_thread_cnt) {
			uint64_t bytes = logger->tryReadLog();
			total_log_data += bytes;
			if (logger->iseof())
				break;
			if (bytes == 0)
			{
				// usleep(100);
			}
		}
		//poolDone = true;
	}
	else
	{	
		// cout << "PSN Flush Frequency: " << g_psn_flush_freq << endl;
		while (glob_manager->get_workload()->sim_done < g_thread_cnt)
		{
			uint32_t bytes = (uint32_t)logger->tryFlush();
			total_log_data += bytes;
			if (bytes == 0)
			{
				PAUSE;
			}
			else
			{
				flushcount++;
			}
			// update epoch periodically.
#if LOG_ALGORITHM == LOG_BATCH
			glob_manager->update_epoch();
#endif
		}
		//cout << "logging counter " << counter << endl;
		printf("logger thread %lu exit, total number (persistent) epochs: %lu, total number of flushes: %lu, logged bytes: %lu\n", _thd_id, glob_manager->get_epoch(), flushcount, total_log_data);
		logger->~LogManager();
		_mm_free(logger); // finish the rest
	}

	// FIXME: add stats here
	// INC_INT_STATS_V0(time_logging_thread, get_sys_clock() - starttime);
	// INC_FLOAT_STATS_V0(log_bytes, total_log_data);

	//INC_INT_STATS_V0(time_io, get_sys_clock() - starttime);
	//INC_INT_STATS(int_debug10, flushcount);
	return FINISH;
}

#endif
