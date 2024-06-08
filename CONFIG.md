# Code Navigation Logs

1. workloads (yasb_wl, tpcc_wl) store INDEX + row_t (manager + record), in the_index and the_table
2. transaction manager (ycsb_txn_man, tpcc_txn_man) stores basic data structures for concurrency control algorithms, 
    and implements transaction logic
3. thread_t::run() initialize a transaction from the query_queue, get timestamp, invoke ycsb_txn_man.run_txn(), address aborts
4. Different concurrency control protocols are implemented in: 
    a. thread_t::run() when getting timestamps
    b. row_t::manager states for version management states and locks
    c. txn_man::get_row() & row_t::get_row() when touching records
5. Adding a stat: 
    * Add field at Stats_thd / Stats_tmp 
    * Stats_thd::init() / Stats_tmp::init() to initialize the field
    * Stats::print(), create a collect variable, addup at the thread for-loop, print it

# Add stats
* SILO: manager::access (time_shared_row_cmt), manager::access->copy (time_shared_record)
* TICTOC: manager::access (time_shared_row_cmt), manager::access->copy (time_shared_record)
* OCC: manager::access (time_shared_row_cmt), manager::access->copy (time_shared_record)
* MVCC: manager::access (time_shared_row_cmt), manager::access->copy (time_shared_record)
* WAIT_DIE, NO_WAIT: manager::log_get (time_shared_row_cmt)


# Add Logging

* Log worker: LoggingThread             (periodically flushes logs to storage)
* Log entity: LogManager                (manages log records)
* APIs to transaction worker: txn_man::make_log   (packetize log records and send to LogManager)

* The chance to call make_log: in concurrency control protocols, after serialization point and correctness validation, before releasing locks
    * SILO: after r/w set validation, before release write locks
    * OCC: after validation, before cleanup
    * TICTOC: before write data
    * HEKATON: before postprocess, after readset validation
    * WAIT_DIE, NO_WAIT, DL_DETECT: at cleanup, before roll-back
    * HEKATON, HSTORE, VLL

# Time to add CXL write mark
* OCC: at return_row, after write data
* WAIT_DIE, NO_WAIT, DL_DETECT: at return_row, before release locks
* SILO: at Row_silo::write, which applies the updates as well as release the lock bit
* TICTOC: at Row_tictoc::write_data, after _row->copy
* MVCC, TimeStamp: Similar to OCC

# Copy-from-CXL or Copy-from-CXL
* OCC:  Row_occ::access-copy_from_cxl, Row_occ::write-copy_to_cxl
* TimeStamp: Row_ts::access-copy_from_cxl, Row_ts::update_buffer-copy_from_cxl, Row_ts::access-copy_to_cxl, Row_ts::update_buffer-copy_to_cxl, 
* WAIT_DIE, NO_WAIT, DL_DETECT: txn_man::get_row-copy_from_cxl, row_t::return_row-copy_to_cxl
* SILO: Row_silo::access-copy_from_cxl, Row_silo::write-copy_to_cxl
* TICTOC: Row_tictoc::access-copy_from_cxl, Row_tictoc::write_data-copy_to_cxl
* MVCC: Row_mvcc::access-copy_from_cxl, Row_mvcc::update_buffer-copy_to_cxl


# Time to flush Write Queue
* The commit queue chould be synchronous or asynchronous, the queue flushing is required if the queue is asynchronous
* For the current implementation, we assume the write commit is sync, i.e. the EP agent returns after all peer caches are invalidated. The timing overhead is added up immediately by the simulator. 


# Experiment Setup
* Contended TPC-C: 1 & 4 warehouses, 10% NewOrder and 15% Payment touch remote records
* Uncontended TPC-C: 1 warehouse/thread, 10% NewOrder and 15% Payment touch remote records
* Write-Intensive Conteded YCSB: 10M records * 100 bytes, 16 records per txn, Zipf=0.8, 95% write
* Write-Intensive Uncontended YCSB: 10M records * 100 bytes, 16 records per txn, Zipf=0.0, 95% write
* Read-Intensive Conteded YCSB: 10M records * 100 bytes, 16 records per txn, Zipf=0.8, 95% read
* Read-Intensive Uncontended YCSB: 10M records * 100 bytes, 16 records per txn, Zipf=0.0, 95% read


# Staring into the abyss: An evaluation of concurrency control with one thousand cores
## TPC-C

### Dataset
* 26MB * 1K warehouses & 100MB * 4 warehouses

### TXN
* Only support Payment and NewOrder
* Omit execution time for worker threads: each worker issues transactions without pausing
* 10% NewOrder and 15% Payment touch remote warehouses

## YCSB

### Dataset
* 10GB YCSB database with 100 byte-per-record * 2m * 48 records

### Dataset for CC Test
* 3M record * 100B = 300 MB records

### TXN
* 16 records per txn, all queries are independent
* Zipf: theta = 0, 0.6 and 0.8 (a hotspot of 10% tuples are accessed by ~40% and ~60% txns)


# Fast in-memory transaction processing using RDMA and HTM

## TPC-C
* New Order 45% + Payment 43% + Order Status 4% + Delivery 4% + Stock Level 4%
* One warehouse per thread
* Distributed transaction rates: 1%, 5%, 10%; also test the sensitivity to the distributed transaction rate (Figure 16, NewOrder Only)

# Polyjuice: High-Performance Transactions via Learned Concurrency Control

## TPC-C
* Standard Mixture ratio: 45% NewOrder + 43% Payment + 4% OrderStatus + 4% Delivery + 4% StockLevel
* 48 Thread with different # of warehouse
    * High contention: 1, 2, 4 warehouse
    * Low contention: 8, 16, 48 warehouse (one warehouse per thread)
* Avg latency, P99 latency, P90, P50, throughput


# Simulate Multiple Hosts
* DBx1000-Distributed mode


Please Check: 
* experiment.py: SNIPER_CXL_LATENCY & SNIPER_MEM_LATENCY
* sim_api.h: Strong & Weak
* address_home_lookup.cc:42 Directory Mapping


### ALL THINGS ARE WRONG: CC和Mem Latency反了。。。。
* 20240403-122323_cxl_to_smp_slowdown_tpcc: All Strongly Coherent, 456-847 ns
* 20240402-151004_cxl_to_smp_slowdown_tpcc: Strong-Weak Coherent, 456-847 ns, VBF + Cuckoo Filter, no back-invalidation
* 20240403-195216_cxl_to_smp_slowdown_ycsb: All Strongly Coherent, 456-847 ns, YCSB
* 20240408-104121_cxl_to_smp_slowdown_ycsb: Strong-Weak Coherent, 456-847 ns, VBF + Cuckoo Filter, no back-invalidation

### STH NEW。。。
* 20240423-220502_cxl_to_smp_slowdown_tpcc: All Strongly Coherent, 847-456 ns
* 20240424-090154_cxl_to_smp_slowdown_tpcc: Strong-Weak Coherent, 847-456 ns, VBF + Cuckoo Filter, no back-invalidation
* 20240427-223434_cxl_to_smp_slowdown_ycsb: All Strongly Coherent, 847-456 ns
* 20240426-113422_cc_respect2_read: YCSB, All Strongly Coherent, 847-456 ns


* 20240516-170810_cxl_to_smp_slowdown_tpcc: All Strongly Coherent, 246-170 ns
* 20240517-212920_cxl_to_smp_slowdown_tpcc: Strong-Weak Coherent, 246-170 ns, VBF + Cuckoo Filter, no back-invalidation

### Paper Figures: see eval for hardware setups
* 20240528-211236_tput_tpcc: 847-456 ns, 2KB/core SF
* 20240528-130806_tput_ycsb: 847-456 ns, 2KB/core SF, 1000B record / tuple
* 20240530-191418_tput_ycsb: 847-456 ns, 16KB/core SF, 200B record / tuple
* 20240531-100934_tput_ycsb: 246-170 ns, 16KB/core SF, 200B record / tuple
* 20240531-114512_tput_tpcc: 246-170 ns, 16KB/core SF

* 20240531-152318_record_size_sensitivity: 847-456 ns, 16KB/core SF, YCSB

* 20240603-003124_index_sensitivity_tpcc: 847-456 ns, 16KB/core SF, CXTNL PRIMITIVE
* 20240603-123540_index_sensitivity_tpcc: 847-456 ns, 16KB/core SF, CXLVANILLA PRIMITIVE

* 20240603-203317_scalability_tpcc: 847-456 ns, 16KB/core SF, CXTNL PRIMITIVE

EP Test Requires Large number of Transactions
* 20240604-144141_ep_test: 847-456 ns, 16KB/core SF, CXTNL PRIMITIVE, 500 txns/thread, 4 Node, 32 Threads
