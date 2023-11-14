#pragma once 

#include "global.h"
#include "helper.h"

class LoggingThread {
public:
	// logging threads have IDs higher than worker threads
	void set_thd_id(uint64_t thd_id) { _thd_id = thd_id + g_thread_cnt; }
#if LOG_ALGORITHM != LOG_NO
	LoggingThread();
#else
	LoggingThread(){}
#endif
	//void 		init(uint64_t thd_id, workload * workload);
	RC 			run();
	void		init();
	uint64_t _thd_id;

	volatile bool poolDone;
	// For SILO
	// the looging thread also manages the epoch number. 
};

extern LoggingThread ** logging_thds;