#include "global.h"
#include "mem_alloc.h"
#include "stats.h"
#include "dl_detect.h"
#include "manager.h"
#include "query.h"
#include "plock.h"
#include "occ.h"
#include "vll.h"
#include "log.h"

mem_alloc mem_allocator;
Stats stats;
DL_detect dl_detector;
Manager * glob_manager;
Query_queue * query_queue;
Plock part_lock_man;
OptCC occ_man;
#if CC_ALG == VLL
VLLMan vll_man;
#endif 

// Logging
bool g_log_recover = LOG_RECOVER;
uint32_t g_num_logger = NUM_LOGGER;
uint32_t g_num_disk = 0;
bool g_no_flush = LOG_NO_FLUSH;
LogManager ** log_manager;  // FIXME: Initialize this 
bool g_ramdisk = false;
uint64_t g_flush_blocksize = FLUSH_BLOCK_SIZE;
uint32_t g_max_log_entry_size = MAX_LOG_ENTRY_SIZE;
uint32_t g_log_chunk_size = LOG_CHUNK_SIZE;
uint64_t g_max_num_epoch = MAX_NUM_EPOCH;
uint64_t g_flush_interval = LOG_FLUSH_INTERVAL;
uint64_t g_log_buffer_size = LOG_BUFFER_SIZE;

bool volatile warmup_finish = false;
bool volatile enable_thread_mem_pool = false;
pthread_barrier_t warmup_bar;
pthread_barrier_t log_bar;
pthread_barrier_t worker_bar;

// FIXME: Change Graphite to snipersim
#ifndef NOGRAPHITE
carbon_barrier_t enable_barrier;
#endif

uint64_t g_max_txns_per_thread = MAX_TXNS_PER_THREAD;

ts_t g_abort_penalty = ABORT_PENALTY;
bool g_central_man = CENTRAL_MAN;
UInt32 g_ts_alloc = TS_ALLOC;
bool g_key_order = KEY_ORDER;
bool g_no_dl = NO_DL;
ts_t g_timeout = TIMEOUT;
ts_t g_dl_loop_detect = DL_LOOP_DETECT;
bool g_ts_batch_alloc = TS_BATCH_ALLOC;
UInt32 g_ts_batch_num = TS_BATCH_NUM;

bool g_part_alloc = PART_ALLOC;
bool g_mem_pad = MEM_PAD;
UInt32 g_cc_alg = CC_ALG;
ts_t g_query_intvl = QUERY_INTVL;
UInt32 g_part_per_txn = PART_PER_TXN;
double g_perc_multi_part = PERC_MULTI_PART;
double g_read_perc = READ_PERC;
double g_write_perc = 1 - READ_PERC;
double g_zipf_theta = ZIPF_THETA;
bool g_prt_lat_distr = PRT_LAT_DISTR;
UInt32 g_part_cnt = PART_CNT;
UInt32 g_virtual_part_cnt = VIRTUAL_PART_CNT;
UInt32 g_thread_cnt = THREAD_CNT;
UInt64 g_synth_table_size = SYNTH_TABLE_SIZE;
UInt32 g_req_per_query = REQ_PER_QUERY;
UInt32 g_field_per_tuple = FIELD_PER_TUPLE;
UInt32 g_init_parallelism = INIT_PARALLELISM;

UInt32 g_num_wh = NUM_WH;
double g_perc_payment = PERC_PAYMENT;
bool g_wh_update = WH_UPDATE;
char * output_file = NULL;


map<string, string> g_params;

#if TPCC_SMALL
UInt32 g_max_items = 10000;
UInt32 g_cust_per_dist = 2000;
#else 
UInt32 g_max_items = 100000;
UInt32 g_cust_per_dist = 3000;
#endif
