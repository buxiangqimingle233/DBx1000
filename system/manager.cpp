#include "manager.h"
#include "row.h"
#include "txn.h"
#include "pthread.h"

__thread uint64_t Manager::_thread_id;
__thread drand48_data Manager::_buffer;

void Manager::init() {
	timestamp = (uint64_t *) _mm_malloc(sizeof(uint64_t), 64);
	*timestamp = 1;
	_last_min_ts_time = 0;
	_min_ts = 0;
	_epoch = (uint64_t *) _mm_malloc(sizeof(uint64_t), 64);
	_last_epoch_update_time = (ts_t *) _mm_malloc(sizeof(uint64_t), 64);
	_max_epochs = new volatile uint64_t * [g_thread_cnt];

	// First epoch is epoch 1.
	*_epoch = 1;
	COMPILER_BARRIER

	// Initialize silo logging epochs
	*_last_epoch_update_time = get_sys_clock();
	all_ts = (ts_t volatile **) _mm_malloc(sizeof(ts_t *) * g_thread_cnt, 64);
	for (uint32_t i = 0; i < g_thread_cnt; i++) {
		all_ts[i] = (ts_t *) _mm_malloc(sizeof(ts_t), 64);
		_max_epochs[i] = (volatile uint64_t *) MALLOC(sizeof(uint64_t), GET_THD_ID);
		*_max_epochs[i] = 0;
	}

	_persistent_epoch = new volatile uint64_t * [g_num_logger];
	for (uint32_t i = 0; i < g_num_logger; i++) {
		_persistent_epoch[i] = (volatile uint64_t *) MALLOC(sizeof(uint64_t), GET_THD_ID);
		*_persistent_epoch[i] = 0;  // initialize
	}
	_epoch_mapping = new uint64_t * [g_thread_cnt];
	_active_epoch = new uint64_t [g_thread_cnt];

	for (uint32_t i = 0; i < g_thread_cnt; i++) {
		// TODO.  assume we never more than 1M epoches.  
		uint32_t max_epoch = g_max_num_epoch;
		_active_epoch[i] = 0;
		_epoch_mapping[i] = new uint64_t [max_epoch];
		memset(_epoch_mapping[i], -1, sizeof(uint64_t) * max_epoch);
	}

	_all_txns = new txn_man * [g_thread_cnt];
	for (UInt32 i = 0; i < g_thread_cnt; i++) {
		*all_ts[i] = UINT64_MAX;
		_all_txns[i] = NULL;
	}
	for (UInt32 i = 0; i < BUCKET_CNT; i++)
		pthread_mutex_init( &mutexes[i], NULL );
}

uint64_t 
Manager::get_ts(uint64_t thread_id) {
	// HACK: Timestamp implemented here
	if (g_ts_batch_alloc)
		assert(g_ts_alloc == TS_CAS);
	uint64_t time;
	uint64_t starttime = get_sys_clock();
	switch(g_ts_alloc) {
	case TS_MUTEX :
		pthread_mutex_lock( &ts_mutex );
		time = ++(*timestamp);
		pthread_mutex_unlock( &ts_mutex );
		break;
	case TS_CAS :
		if (g_ts_batch_alloc)
			time = ATOM_FETCH_ADD((*timestamp), g_ts_batch_num);
		else 
			time = ATOM_FETCH_ADD((*timestamp), 1);
		break;
	case TS_HW :
#ifndef NOGRAPHITE
		time = CarbonGetTimestamp();
#else
		assert(false);
#endif
		break;
	case TS_CLOCK :
		time = get_sys_clock() * g_thread_cnt + thread_id;
		break;
	default :
		assert(false);
	}
	INC_STATS(thread_id, time_ts_alloc, get_sys_clock() - starttime);
	return time;
}

ts_t Manager::get_min_ts(uint64_t tid) {
	uint64_t now = get_sys_clock();
	uint64_t last_time = _last_min_ts_time; 
	if (tid == 0 && now - last_time > MIN_TS_INTVL)
	{ 
		ts_t min = UINT64_MAX;
    		for (UInt32 i = 0; i < g_thread_cnt; i++)
			if (*all_ts[i] < min)
		    	    	min = *all_ts[i];
		if (min > _min_ts)
			_min_ts = min;
	}
	return _min_ts;
}

void Manager::add_ts(uint64_t thd_id, ts_t ts) {
	assert( ts >= *all_ts[thd_id] || 
		*all_ts[thd_id] == UINT64_MAX);
	*all_ts[thd_id] = ts;
}

void Manager::set_txn_man(txn_man * txn) {
	int thd_id = txn->get_thd_id();
	_all_txns[thd_id] = txn;
}


uint64_t Manager::hash(row_t * row) {
	uint64_t addr = (uint64_t)row / MEM_ALLIGN;
    return (addr * 1103515247 + 12345) % BUCKET_CNT;
}
 
void Manager::lock_row(row_t * row) {
	int bid = hash(row);
	pthread_mutex_lock( &mutexes[bid] );	
}

void Manager::release_row(row_t * row) {
	int bid = hash(row);
	pthread_mutex_unlock( &mutexes[bid] );
}
	
void
Manager::update_epoch()
{
	ts_t time = get_sys_clock();
	if (time - *_last_epoch_update_time > LOG_BATCH_TIME * 1000 * 1000) {
		*_epoch = *_epoch + 1;
		*_last_epoch_update_time = time;
	}
}


void
Manager::update_max_epoch(uint64_t epoch)
{
	*_max_epochs[GET_THD_ID] = epoch;
}

uint64_t 		
Manager::get_ready_epoch()
{
	uint64_t ready_epoch = (uint64_t)-1;
	for (uint32_t i = 0; i < g_thread_cnt; i++) {
		if (ready_epoch > *_max_epochs[i])
			ready_epoch = *_max_epochs[i];
	}
	if (ready_epoch > 1)
		return ready_epoch - 1;
	else 
		return 0;
}

void	
Manager::update_epoch_lsn_mapping(uint64_t epoch, uint64_t lsn)
{
	//assert(epoch[logger] == _active_epoch[logger] || epoch[logger] == _active_epoch[logger] + 1);
	uint32_t thd = GET_THD_ID;
	while (epoch > _active_epoch[thd]) {
		assert(_active_epoch[thd] < g_max_num_epoch);
		_epoch_mapping[thd][_active_epoch[thd] ++] = lsn;
//		printf("update_epoch_lsn_mapping.  epoch=%ld. active_epoch[%d] = %ld  => lsn=%ld\n", 
//			 epoch, thd, _active_epoch[thd], lsn);
	}
}

void 
Manager::update_persistent_epoch(uint32_t logger, uint64_t lsn)
{
	bool done = false;
	while (!done) {
//		printf("logger=%d. flush lsn = %ld\n", logger, lsn);
		for (uint32_t i = 0; i < g_thread_cnt; i++) {
			if (i % g_num_logger != logger) continue; 
			uint64_t l = _epoch_mapping[i][ *_persistent_epoch[logger] + 1 ];
			if (l == (uint64_t)-1 || (l != (uint64_t)-1 && l > lsn)) {
				done = true;
//				printf("!!!!!! l=%ld lsn=%ld. persistent_epoch=%ld\n", l, lsn, *_persistent_epoch[logger]);
				break;
			}
		}
//		printf("done = %d\n", done);
		if (!done)
			(*_persistent_epoch[logger]) ++;
	}
}
