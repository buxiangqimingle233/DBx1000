#include "txn.h"
#include "row.h"
#include "row_silo.h"
#include "mem_alloc.h"
#include "manager.h"

#if CC_ALG==SILO

void 
Row_silo::init(row_t * row) 
{
	_row = row;
#if ATOMIC_WORD
	_tid_word = 0;
#else 
	_latch = (pthread_mutex_t *) _mm_malloc(sizeof(pthread_mutex_t), 64);
	pthread_mutex_init( _latch, NULL );
	_tid = 0;
#endif
}

RC
Row_silo::access(txn_man * txn, TsType type, row_t * local_row) {
#if ATOMIC_WORD
	uint64_t v = 0;
	uint64_t v2 = 1;
	while (v2 != v) {
		v = _tid_word;
		while (v & LOCK_BIT) {
			PAUSE
			v = _tid_word;
		}
		// local_row->copy(_row);
		PROFILE_VOID(time_shared_record, local_row->copy_from_cxl, _row);
		COMPILER_BARRIER
		v2 = _tid_word;
	} 
	txn->last_tid = v & (~LOCK_BIT);
#else 
	lock();
	local_row->copy(_row);
	txn->last_tid = _tid;
	release();
#endif
	return RCOK;
}

bool
Row_silo::validate(ts_t tid, bool in_write_set) {
	SimAccessCXLType2();
#if ATOMIC_WORD
	uint64_t v = _tid_word;
	if (in_write_set) {
		bool res = (tid == (v & (~LOCK_BIT)));
		SimAccessReset();
		return res;
		// return tid == (v & (~LOCK_BIT));
	}
	if (v & LOCK_BIT) {
		SimAccessReset();
		return false;
	}
	else if (tid != (v & (~LOCK_BIT))) {
		SimAccessReset();
		return false;
	}
	else {
		SimAccessReset();
		return true;
	}
#else
	if (in_write_set)	
		return tid == _tid;
	if (!try_lock())
		return false;
	bool valid = (tid == _tid);
	release();
	return valid;
#endif
}

void
Row_silo::write(row_t * data, uint64_t tid) {
	// _row->copy(data);
	_row->copy_to_cxl(data);
	SimSyncWrite((unsigned long)_row->data, _row->get_tuple_size());
#if ATOMIC_WORD
	uint64_t v = _tid_word;
	M_ASSERT(tid > (v & (~LOCK_BIT)) && (v & LOCK_BIT), "tid=%ld, v & LOCK_BIT=%ld, v & (~LOCK_BIT)=%ld\n", tid, (v & LOCK_BIT), (v & (~LOCK_BIT)));
	_tid_word = (tid | LOCK_BIT); 
#else
	_tid = tid;
#endif
	SimAccessReset();
}

void
Row_silo::lock() {
#if ATOMIC_WORD
	SimAccessCXLType2();
	uint64_t v = _tid_word;
	while ((v & LOCK_BIT) || !__sync_bool_compare_and_swap(&_tid_word, v, v | LOCK_BIT)) {
		PAUSE
		v = _tid_word;
	}
#else
	pthread_mutex_lock( _latch );
#endif
	SimAccessReset();
}

void
Row_silo::release() {
	SimAccessCXLType2();
#if ATOMIC_WORD
	assert(_tid_word & LOCK_BIT);
	_tid_word = _tid_word & (~LOCK_BIT);
#else 
	pthread_mutex_unlock( _latch );
#endif
	SimAccessReset();
}

bool
Row_silo::try_lock()
{
	SimAccessCXLType2();
#if ATOMIC_WORD
	uint64_t v = _tid_word;
	if (v & LOCK_BIT)  {
		SimAccessReset();
		// already locked
		return false;
	}
	SimAccessReset();
	return __sync_bool_compare_and_swap(&_tid_word, v, (v | LOCK_BIT));
#else
	SimAccessReset();
	return pthread_mutex_trylock( _latch ) != EBUSY;
#endif
	
}

uint64_t 
Row_silo::get_tid()
{
	SimAccessCXLType2();
	assert(ATOMIC_WORD);
	SimAccessReset();
	return _tid_word & (~LOCK_BIT);
}

#endif
