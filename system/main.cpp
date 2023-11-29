#include "global.h"
#include "ycsb.h"
#include "tpcc.h"
#include "test.h"
#include "thread.h"
#include "manager.h"
#include "mem_alloc.h"
#include "query.h"
#include "plock.h"
#include "occ.h"
#include "vll.h"
#include "log.h"
#include "logging_thread.h"
#include "sim_api.h"

void * f(void *);
void * f_log(void *);

thread_t ** m_thds;
LoggingThread **logging_thds;

// defined in parser.cpp
void parser(int argc, char * argv[]);
void print_val();

/* HACK:
	1. workloads (yasb_wl, tpcc_wl) store INDEX + row_t (manager + record), in the_index and the_table
	2. transaction manager (ycsb_txn_man, tpcc_txn_man) stores basic data structures for concurrency control algorithms, 
		and implements transaction logic
	3. tread_t::run() initialize a transaction from the query_queue, get timestamp, invoke ycsb_txn_man.run_txn(), address aborts
	4. Different concurrency control protocols are implemented in: 
		a. thread_t::run() when getting timestamps
		b. row_t::manager states for version management states and locks
		c. txn_man::get_row() & row_t::get_row() when touching records
*/

int main(int argc, char* argv[])
{
	uint64_t mainstart = get_sys_clock();
	double mainstart_wallclock = get_wall_time();
	parser(argc, argv);
	print_val();
	
	mem_allocator.init(g_part_cnt, MEM_SIZE / g_part_cnt); 
	stats.init();
	glob_manager = (Manager *) _mm_malloc(sizeof(Manager), 64);
	glob_manager->init();
	if (g_cc_alg == DL_DETECT) 
		dl_detector.init();
	printf("mem_allocator initialized!\n");
	workload * m_wl;
	switch (WORKLOAD) {
		case YCSB :
			m_wl = new ycsb_wl; break;
		case TPCC :
			m_wl = new tpcc_wl; break;
		case TEST :
			m_wl = new TestWorkload; 
			((TestWorkload *)m_wl)->tick();
			break;
		default:
			assert(false);
	}
	m_wl->init();
	printf("workload initialized!\n");
	glob_manager->set_workload(m_wl);

	// Init logging data structure here
	string bench = "YCSB";
	if (WORKLOAD == TPCC)
	{
		bench = "TPCC_" + to_string(g_perc_payment);
	}
	log_manager = new LogManager *[g_num_logger];
	string type = (LOG_ALGORITHM == LOG_PARALLEL) ? "P" : "B";
	for (uint32_t i = 0; i < g_num_logger; i++)
	{
		log_manager[i] = (LogManager *)_mm_malloc(sizeof(LogManager), 64);
		new (log_manager[i]) LogManager(i);
		log_manager[i]->init("./logs/" + type + "D_log" + to_string(i) + "_" + to_string(g_num_logger) + "_" + bench + ".log");
	}

/* Seems no use, copied from logging version
	next_log_file_epoch = new uint32_t *[g_num_logger];
	for (uint32_t i = 0; i < g_num_logger; i++)
	{
		next_log_file_epoch[i] = (uint32_t *)MALLOC(sizeof(uint32_t), GET_THD_ID);
	}
	mem_allocator.init(g_part_cnt, MEM_SIZE / g_part_cnt);
*/

	uint64_t thd_cnt = g_thread_cnt;

	// Create working / logging threads here 
	pthread_t p_thds[thd_cnt - 1];
	pthread_t p_logs[g_num_logger];

	// Create and init pre-thread data structure here
	m_thds = new thread_t * [thd_cnt];
	logging_thds = new LoggingThread * [g_num_logger];

	for (uint32_t i = 0; i < thd_cnt; i++) {
		m_thds[i] = (thread_t *) _mm_malloc(sizeof(thread_t), 64);
		new (m_thds[i]) thread_t();
	}
	for (uint32_t i = 0; i < g_num_logger; i++) {
		logging_thds[i] = (LoggingThread *) _mm_malloc(sizeof(LoggingThread), 64);
		new (logging_thds[i]) LoggingThread();
	}

	// query_queue should be the last one to be initialized!!!
	// because it collects txn latency
	query_queue = (Query_queue *) _mm_malloc(sizeof(Query_queue), 64);
	if (WORKLOAD != TEST)
		query_queue->init(m_wl);
	printf("query_queue initialized!\n");

	// Initialize warmup bar, logging bar 
	pthread_barrier_init(&warmup_bar, NULL, g_thread_cnt );
	pthread_barrier_init(&worker_bar, NULL, g_thread_cnt);

#if LOG_ALGORITHM == LOG_NO
	pthread_barrier_init(&log_bar, NULL, g_thread_cnt);
#else
	pthread_barrier_init(&log_bar, NULL, g_num_logger + g_thread_cnt);
#endif

#if CC_ALG == HSTORE
	part_lock_man.init();
#elif CC_ALG == OCC
	occ_man.init();
#elif CC_ALG == VLL
	vll_man.init();
#endif

	// Init worker / logger thread data structures for warmup
	for (uint32_t i = 0; i < thd_cnt; i++) 
		m_thds[i]->init(i, m_wl);

#if LOG_ALGORITHM != LOG_NO
	for (uint32_t i = 0; i < g_num_logger; i++)
		logging_thds[i]->set_thd_id(i);
#endif

	if (WARMUP > 0) {
		printf("WARMUP start!\n");
		for (uint32_t i = 0; i < thd_cnt - 1; i++) {
			uint64_t vid = i;
			pthread_create(&p_thds[i], NULL, f, (void *)vid);
		}
		f((void *)(thd_cnt - 1));
		for (uint32_t i = 0; i < thd_cnt - 1; i++)
			pthread_join(p_thds[i], NULL);
		printf("WARMUP finished!\n");
	}
	warmup_finish = true;
	pthread_barrier_init( &warmup_bar, NULL, g_thread_cnt );
#ifndef NOGRAPHITE
	CarbonBarrierInit(&enable_barrier, g_thread_cnt);
#endif


/* We do not consider recovering here
	if (g_log_recover)
	{
		// change the order of threads.
		assert(LOG_ALGORITHM != LOG_NO);
		for (uint32_t i = 0; i < g_num_logger; i++)
		{
			uint64_t vid = i;
			pthread_create(&p_logs[i], NULL, f_log, (void *)vid);
		}
		for (uint32_t i = 0; i < thd_cnt - 1; i++)
		{
			uint64_t vid = i;
			pthread_create(&p_thds[i], NULL, f, (void *)vid);
		}
	}
*/
#ifdef SNIPER
	SimRoiStart();
#endif

	// spawn and run txns again.
	int64_t starttime = get_server_clock();
	// Let's start working
	for (uint32_t i = 0; i < thd_cnt - 1; i++) {
		uint64_t vid = i;
		pthread_create(&p_thds[i], NULL, f, (void *)vid);
	}
	if (LOG_ALGORITHM != LOG_NO) {
		for (uint32_t i = 0; i < g_num_logger; i++)
		{
			uint64_t vid = i;
			pthread_create(&p_logs[i], NULL, f_log, (void *)vid);
		}
	}
	f((void *)(thd_cnt - 1));


	for (uint32_t i = 0; i < thd_cnt - 1; i++) 
		pthread_join(p_thds[i], NULL);
	if (LOG_ALGORITHM != LOG_NO) 
		for (uint32_t i = 0; i < g_num_logger; i++)
			pthread_join(p_logs[i], NULL);
	int64_t endtime = get_server_clock();

#ifdef SNIPER
	SimRoiEnd();
#endif

	if (WORKLOAD != TEST) {
		printf("PASS! SimTime = %ld (cycles)\n", endtime - starttime);
		if (STATS_ENABLE)
			stats.print();
	} else {
		((TestWorkload *)m_wl)->summarize();
	}

	double sys_time_observed = float(get_sys_clock() - mainstart) / CPU_FREQ / 1e9;
	double wall_time_observed = get_wall_time() - mainstart_wallclock;
	cout << "Total time measured " << sys_time_observed << endl; // for CPU_FREQ calibration
	cout << "Total wall time observed " << wall_time_observed << endl;
	cout << "Estimated CPU_FREQ is " << (CPU_FREQ)*sys_time_observed / wall_time_observed << endl;

	return 0; 
}

void * f(void * id) {
	uint64_t tid = (uint64_t)id;
	m_thds[tid]->run();
	return NULL;
}

void *f_log(void *id)
{
#if LOG_ALGORITHM != LOG_NO
	uint64_t tid = (uint64_t)id;
	logging_thds[(tid + g_thread_cnt) % g_num_logger]->run();
#endif
	return NULL;
}
