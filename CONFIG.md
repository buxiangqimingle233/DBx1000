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