# Code Navigation Logs

1. workloads (yasb_wl, tpcc_wl) store INDEX + row_t (manager + record), in the_index and the_table
2. transaction manager (ycsb_txn_man, tpcc_txn_man) stores basic data structures for concurrency control algorithms, 
    and implements transaction logic
3. thread_t::run() initialize a transaction from the query_queue, get timestamp, invoke ycsb_txn_man.run_txn(), address aborts
4. Different concurrency control protocols are implemented in: 
    a. thread_t::run() when getting timestamps
    b. row_t::manager states for version management states and locks
    c. txn_man::get_row() & row_t::get_row() when touching records


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
* 20GB YCSB database with 20m records
* primary key with 100 bytes records

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