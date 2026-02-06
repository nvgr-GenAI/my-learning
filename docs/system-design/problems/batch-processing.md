# Design a Batch Processing System (Apache Spark, Hadoop)

A distributed batch data processing platform that processes massive datasets (100s of TB to PBs) across thousands of worker nodes using Apache Spark and Hadoop, with fault tolerance via lineage/checkpointing, resource management via YARN, and optimized shuffle operations for large-scale analytics and ETL workloads.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100 TB/day processing, 1000+ worker nodes, 10B+ records, petabyte-scale datasets |
| **Key Challenges** | Shuffle optimization, data skew, fault tolerance, resource scheduling, memory management, speculative execution |
| **Core Concepts** | DAG execution, lazy evaluation, RDD/DataFrame transformations, partitioning, shuffle, wide/narrow dependencies |
| **Companies** | Apache Spark, Hadoop MapReduce, AWS EMR, Databricks, Google Dataproc, Azure HDInsight, Cloudera |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Distributed Data Processing** | Process 100+ TB datasets across 1000+ nodes | P0 (Must have) |
    | **Data Partitioning** | Hash, range, and custom partitioning strategies | P0 (Must have) |
    | **Transformations** | Map, filter, reduce, join, groupBy, aggregations | P0 (Must have) |
    | **Fault Tolerance** | Lineage-based recovery and checkpointing | P0 (Must have) |
    | **Resource Management** | Dynamic resource allocation with YARN/Kubernetes | P0 (Must have) |
    | **Shuffle Operations** | Efficient hash-based shuffle with sort and merge | P0 (Must have) |
    | **Data Sources** | Read from HDFS, S3, Parquet, ORC, CSV, JSON | P0 (Must have) |
    | **SQL Support** | Spark SQL for structured data processing | P0 (Must have) |
    | **Memory Management** | Spill to disk, cache management, memory tuning | P1 (Should have) |
    | **Speculative Execution** | Re-run slow tasks on other nodes | P1 (Should have) |
    | **Dynamic Optimization** | Adaptive query execution, broadcast join hints | P1 (Should have) |
    | **Data Skew Handling** | Salting, repartitioning, adaptive execution | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Real-time stream processing (use Spark Streaming/Flink)
    - Interactive queries (use Presto/Trino for sub-second queries)
    - OLTP workloads (use traditional databases)
    - Machine learning pipelines (use specialized ML frameworks)
    - Data governance and catalog (use external tools)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Throughput** | 100 TB/day | Support large-scale batch ETL |
    | **Job Latency** | Minutes to hours | Batch processing, not real-time |
    | **Availability** | 99.9% uptime | Critical data pipelines |
    | **Fault Tolerance** | Automatic recovery | Handle node failures gracefully |
    | **Scalability** | 1000+ nodes | Linear scaling for data and compute |
    | **Resource Efficiency** | 70-80% cluster utilization | Optimize cost and performance |
    | **Data Locality** | 80%+ local reads | Minimize network transfer |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Processing scale:
    - Daily data volume: 100 TB/day
    - Records: 100 TB / 10 KB per record = 10B records/day
    - Processing time: 4 hours (peak ETL window)
    - Throughput: 100 TB / 4 hours = 25 TB/hour = 7 GB/sec

    Job characteristics:
    - Jobs per day: 500 jobs
    - Average job size: 200 GB
    - Tasks per job: 1,000 tasks (avg)
    - Task duration: 2 minutes (avg)
    - Total task executions: 500 × 1,000 = 500K tasks/day

    Data operations:
    - Reads: 100 TB from HDFS/S3
    - Shuffle: 30 TB (30% of data shuffled)
    - Writes: 80 TB (20% filtered out)
    - Compression ratio: 3:1 (Parquet/ORC)
    ```

    ### Storage Estimates

    ```
    Data storage (HDFS/S3):

    Raw data:
    - Daily ingestion: 100 TB
    - Retention: 90 days → 9 PB raw data
    - Replication factor: 3x → 27 PB total

    Processed data:
    - Daily output: 80 TB
    - Retention: 365 days → 29.2 PB processed
    - Replication: 3x → 87.6 PB total

    Intermediate data (shuffle):
    - Shuffle data: 30 TB/day
    - Temporary storage: 1 day → 30 TB
    - No replication (ephemeral) → 30 TB

    Metadata:
    - HDFS NameNode: 1 GB per 1M files
    - Files: 100 PB / 128 MB blocks = 800M blocks
    - Metadata: 800 MB
    - Block locations, replicas: 5 GB total

    Total storage: 27 PB + 87.6 PB + 30 TB = ~115 PB
    ```

    ### Bandwidth Estimates

    ```
    Network traffic:

    Data reads (from storage):
    - 100 TB / 4 hours = 7 GB/sec = 56 Gbps
    - With 1,000 workers: 56 Mbps per worker
    - Data locality: 80% local → 11.2 Gbps cross-node

    Shuffle traffic (between workers):
    - 30 TB shuffle / 4 hours = 2 GB/sec = 16 Gbps
    - Shuffle is all-to-all: High network utilization
    - Per worker: 16 Gbps / 1,000 = 16 Mbps

    Data writes (to storage):
    - 80 TB / 4 hours = 5.5 GB/sec = 44 Gbps
    - With replication (3x): 44 × 3 = 132 Gbps
    - Per worker: 132 Gbps / 1,000 = 132 Mbps

    Total bandwidth:
    - Aggregate: 56 + 16 + 132 = 204 Gbps
    - Per worker: 10 Gbps network interface
    ```

    ### Server Estimates

    ```
    Worker nodes (compute):
    - Total workers: 1,000 nodes
    - CPU: 32 cores per node (YARN containers)
    - Memory: 256 GB per node (executor memory + OS)
    - Disk: 12 × 4 TB HDDs = 48 TB per node (HDFS)
    - Network: 10 Gbps per node
    - Total cores: 32,000 cores
    - Total memory: 256 TB
    - Total disk: 48 PB raw (16 PB with 3x replication)

    Master nodes:
    - YARN ResourceManager: 2 nodes (HA)
    - HDFS NameNode: 2 nodes (HA)
    - Spark Master: 1 node (or YARN mode)
    - Per node: 64 cores, 512 GB RAM, 10 TB SSD

    Storage sizing:
    - HDFS: 48 PB raw / 3 replication = 16 PB usable
    - Meets requirement: 115 PB with compression

    Resource allocation:
    - Executors per node: 4 executors
    - Cores per executor: 5 cores (32 / 4 executors / 1.5 overhead)
    - Memory per executor: 50 GB (256 GB / 4 executors / 1.2 overhead)
    - Total executors: 1,000 nodes × 4 = 4,000 executors

    Total infrastructure:
    - Worker nodes: 1,000 nodes
    - Master nodes: 5 nodes
    - Total: 1,005 nodes
    ```

    ---

    ## Key Assumptions

    1. 100 TB data processed per day across 1,000 worker nodes
    2. 30% of data requires shuffle operations (joins, groupBy)
    3. 20% data reduction after filtering and aggregation
    4. 3:1 compression ratio with Parquet/ORC format
    5. Data locality at 80% (minimize network transfers)
    6. Task failure rate at 1% (speculative execution handles stragglers)
    7. 4-hour ETL window for daily batch processing
    8. HDFS replication factor of 3 for fault tolerance

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **DAG-based execution:** Jobs compiled to directed acyclic graphs of stages
    2. **Lazy evaluation:** Transformations build execution plan, actions trigger compute
    3. **Lineage-based fault tolerance:** Recompute lost partitions from source data
    4. **Data partitioning:** Distribute data across nodes for parallel processing
    5. **Shuffle optimization:** Minimize data movement between stages
    6. **Resource isolation:** YARN containers for multi-tenancy
    7. **Data locality:** Schedule tasks on nodes with data replicas

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            SparkApp[Spark Application<br/>Scala/Python/Java<br/>Submit job]
            SparkSQL[Spark SQL<br/>DataFrame API<br/>Catalyst optimizer]
        end

        subgraph "Cluster Manager - YARN"
            RM[Resource Manager<br/>Job scheduling<br/>Resource allocation<br/>HA: 2 nodes]
            NM1[Node Manager 1<br/>Container management<br/>Resource monitoring]
            NM2[Node Manager 2<br/>Container management<br/>Resource monitoring]
            NMN[Node Manager N<br/>Container management<br/>Resource monitoring]
        end

        subgraph "Spark Application Master"
            Driver[Spark Driver<br/>DAG Scheduler<br/>Task Scheduler<br/>Lineage tracking]
            DAG[DAG Scheduler<br/>Stage creation<br/>Dependency analysis]
            TaskSch[Task Scheduler<br/>Task assignment<br/>Locality preference]
        end

        subgraph "Executor Pool (Worker Node 1)"
            Ex1[Executor 1<br/>50 GB memory<br/>5 cores]
            Ex1Cache[(Block Manager<br/>Cache<br/>RDD partitions)]
            Ex1Task1[Task 1<br/>Map]
            Ex1Task2[Task 2<br/>Filter]
        end

        subgraph "Executor Pool (Worker Node 2)"
            Ex2[Executor 2<br/>50 GB memory<br/>5 cores]
            Ex2Cache[(Block Manager<br/>Cache<br/>RDD partitions)]
            Ex2Task1[Task 3<br/>GroupBy]
            Ex2Task2[Task 4<br/>Reduce]
        end

        subgraph "Executor Pool (Worker Node N)"
            ExN[Executor N<br/>50 GB memory<br/>5 cores]
            ExNCache[(Block Manager<br/>Cache<br/>RDD partitions)]
            ExNTask1[Task N<br/>Join]
            ExNTask2[Task N+1<br/>Aggregate]
        end

        subgraph "Distributed Storage - HDFS"
            NN[NameNode<br/>Metadata store<br/>Block locations<br/>HA: 2 nodes]
            DN1[(DataNode 1<br/>48 TB disk<br/>HDFS blocks)]
            DN2[(DataNode 2<br/>48 TB disk<br/>HDFS blocks)]
            DNN[(DataNode N<br/>48 TB disk<br/>HDFS blocks)]
        end

        subgraph "Shuffle Service"
            ShuffleWrite[Shuffle Write<br/>Sort and partition<br/>Disk-based]
            ShuffleRead[Shuffle Read<br/>Fetch from nodes<br/>Merge sorted runs]
            ShuffleMgr[Shuffle Manager<br/>Block locations<br/>Transfer coordination]
        end

        subgraph "External Storage"
            S3[(S3 Data Lake<br/>Raw data<br/>Parquet files)]
            Warehouse[(Data Warehouse<br/>Aggregated results<br/>Snowflake/Redshift)]
        end

        subgraph "Monitoring & Metrics"
            SparkUI[Spark Web UI<br/>Job progress<br/>DAG visualization<br/>Task metrics]
            Metrics[Metrics System<br/>Prometheus<br/>Executor stats<br/>Shuffle metrics]
            Logs[Log Aggregation<br/>YARN logs<br/>Executor logs]
        end

        SparkApp --> Driver
        SparkSQL --> Driver

        Driver --> RM
        RM --> NM1
        RM --> NM2
        RM --> NMN

        Driver --> DAG
        DAG --> TaskSch
        TaskSch --> Ex1
        TaskSch --> Ex2
        TaskSch --> ExN

        Ex1 --> Ex1Task1
        Ex1 --> Ex1Task2
        Ex1 --> Ex1Cache

        Ex2 --> Ex2Task1
        Ex2 --> Ex2Task2
        Ex2 --> Ex2Cache

        ExN --> ExNTask1
        ExN --> ExNTask2
        ExN --> ExNCache

        Ex1Task1 --> ShuffleWrite
        Ex2Task1 --> ShuffleRead
        ShuffleWrite --> ShuffleMgr
        ShuffleRead --> ShuffleMgr

        Ex1Task1 --> DN1
        Ex2Task1 --> DN2
        ExNTask1 --> DNN

        DN1 --> NN
        DN2 --> NN
        DNN --> NN

        S3 --> Ex1Task1
        Ex2Task2 --> Warehouse

        Driver --> SparkUI
        Ex1 --> Metrics
        Ex2 --> Metrics
        NM1 --> Logs

        style Driver fill:#90EE90
        style Ex1 fill:#FFE4B5
        style Ex2 fill:#FFE4B5
        style ExN fill:#FFE4B5
        style RM fill:#E6E6FA
        style NN fill:#FFF0F5
        style ShuffleMgr fill:#B0E0E6
        style DAG fill:#F0E68C
        style SparkUI fill:#FFE4E1
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Spark Driver** | Coordinates job execution, builds DAG, tracks lineage | MapReduce (higher latency), manual orchestration (complex) |
    | **YARN ResourceManager** | Multi-tenant resource allocation, isolation, scheduling | Standalone (no multi-tenancy), Kubernetes (less mature for big data) |
    | **DAG Scheduler** | Stage pipelining, optimize execution plan, shuffle boundaries | Fixed pipeline (inflexible), manual stages (error-prone) |
    | **RDD/DataFrame** | Lazy evaluation, lineage tracking, fault tolerance | Eager evaluation (no optimization), external storage (slow) |
    | **Block Manager** | In-memory caching, partition management, data locality | No caching (re-read data), external cache (network overhead) |
    | **Shuffle Manager** | Sort-based shuffle, reduce memory, handle large shuffles | Hash-based (memory explosion), external shuffle (complexity) |
    | **HDFS** | Distributed storage, replication, data locality | S3 (network latency), NFS (single point of failure), local disk (no fault tolerance) |
    | **Speculative Execution** | Handle stragglers, improve job completion time | Wait for slow tasks (long tail latency), kill and retry (wasteful) |

    **Key Trade-off:** We chose **lineage-based recovery** over checkpointing for most operations to minimize storage overhead. Checkpointing is used selectively for iterative algorithms where lineage chains become too long.

    ---

    ## Data Flow

    **Example: Word Count Job**

    ```
    1. Job Submission:
       - User submits Spark job: spark-submit wordcount.py
       - Driver starts on client or cluster
       - Driver requests resources from YARN ResourceManager

    2. Resource Allocation:
       - ResourceManager allocates containers on NodeManagers
       - Executors start in containers (4 executors × 1,000 nodes)
       - Executors register with Driver

    3. Data Loading:
       - Read text files from HDFS: sc.textFile("hdfs://data/input")
       - Driver queries NameNode for block locations
       - Tasks scheduled on nodes with data blocks (data locality)

    4. Transformations (Lazy):
       - flatMap: Split lines into words → Narrow dependency
       - map: Word → (word, 1) → Narrow dependency
       - reduceByKey: Count words → Wide dependency (shuffle)

    5. DAG Creation:
       - Driver analyzes dependencies
       - Stage 1: flatMap + map (no shuffle)
       - Stage 2: reduceByKey (shuffle boundary)

    6. Stage 1 Execution:
       - Tasks read HDFS blocks (local reads)
       - Apply flatMap and map transformations
       - Write shuffle outputs to local disk (partitioned by hash)
       - Shuffle write: 2,000 tasks × 200 partitions = 400K files

    7. Shuffle Phase:
       - Stage 2 tasks fetch shuffle data
       - Shuffle read: Pull from all Stage 1 executors
       - Sort and merge shuffle blocks
       - Network transfer: 30 TB shuffle data

    8. Stage 2 Execution:
       - reduceByKey aggregates word counts
       - Combine sorted partitions
       - Write results to HDFS

    9. Failure Handling:
       - Task fails on Node 5
       - Driver detects failure (heartbeat timeout)
       - Recompute lost partition using lineage
       - Re-execute failed task on Node 8
       - No impact on other tasks (partition independence)

    10. Job Completion:
        - All stages complete
        - Results written to HDFS
        - Executors shut down
        - Resources released to YARN
        - Driver exits with success code

    Processing time: 10 minutes for 1 TB text data
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 DAG Execution and Lazy Evaluation

    ### DAG Scheduler

    ```scala
    /**
     * DAG Scheduler: Core component that converts RDD lineage into physical execution plan
     *
     * Responsibilities:
     * 1. Stage creation based on shuffle boundaries
     * 2. Dependency analysis (narrow vs. wide dependencies)
     * 3. Task generation for each stage
     * 4. Failure recovery via lineage
     */

    class DAGScheduler(taskScheduler: TaskScheduler, mapOutputTracker: MapOutputTracker) {

      // Active jobs and stages
      private val activeJobs = new HashMap[Int, Job]
      private val activeStages = new HashMap[Int, Stage]
      private val shuffleToMapStage = new HashMap[Int, ShuffleMapStage]

      /**
       * Submit a job for execution
       * Called when an action is triggered (e.g., collect(), count(), saveAsTextFile())
       */
      def submitJob[T, U](
          rdd: RDD[T],
          func: (TaskContext, Iterator[T]) => U,
          partitions: Seq[Int],
          resultHandler: (Int, U) => Unit): JobWaiter[U] = {

        // Create job with unique ID
        val jobId = nextJobId.getAndIncrement()
        val job = new Job(jobId, rdd, func, partitions, resultHandler)

        activeJobs(jobId) = job

        // Build stage graph from RDD lineage
        val finalStage = createResultStage(rdd, partitions)
        job.finalStage = finalStage

        // Submit stages in topological order
        submitStage(finalStage)

        new JobWaiter(job)
      }

      /**
       * Create stages from RDD lineage
       * Stage boundary = shuffle dependency (wide dependency)
       */
      private def createResultStage(rdd: RDD[_], partitions: Seq[Int]): ResultStage = {
        val parents = getOrCreateParentStages(rdd)
        val stageId = nextStageId.getAndIncrement()
        new ResultStage(stageId, rdd, partitions, parents)
      }

      /**
       * Recursively create parent stages
       * Stop at shuffle boundaries
       */
      private def getOrCreateParentStages(rdd: RDD[_]): List[Stage] = {
        val parents = new ArrayBuffer[Stage]

        for (dep <- rdd.dependencies) {
          dep match {
            case shuffleDep: ShuffleDependency[_, _, _] =>
              // Wide dependency → create shuffle map stage
              parents += getOrCreateShuffleMapStage(shuffleDep)

            case _ =>
              // Narrow dependency → recurse to parent RDD
              parents ++= getOrCreateParentStages(dep.rdd)
          }
        }

        parents.toList
      }

      /**
       * Create shuffle map stage for shuffle dependency
       */
      private def getOrCreateShuffleMapStage(shuffleDep: ShuffleDependency[_, _, _]): ShuffleMapStage = {
        shuffleToMapStage.getOrElseUpdate(shuffleDep.shuffleId, {
          val rdd = shuffleDep.rdd
          val parents = getOrCreateParentStages(rdd)
          val stageId = nextStageId.getAndIncrement()

          val stage = new ShuffleMapStage(stageId, rdd, shuffleDep, parents)
          stage
        })
      }

      /**
       * Submit a stage for execution
       * Only submit if all parent stages are complete
       */
      private def submitStage(stage: Stage): Unit = {
        if (!activeStages.contains(stage.id)) {
          val missing = getMissingParentStages(stage)

          if (missing.isEmpty) {
            // All parents complete → submit this stage
            submitMissingTasks(stage)
            activeStages(stage.id) = stage
          } else {
            // Submit parent stages first (topological order)
            for (parent <- missing) {
              submitStage(parent)
            }
          }
        }
      }

      /**
       * Create and submit tasks for a stage
       */
      private def submitMissingTasks(stage: Stage): Unit = {
        val partitionsToCompute = stage.findMissingPartitions()

        val tasks: Seq[Task[_]] = stage match {
          case shuffleMapStage: ShuffleMapStage =>
            // Shuffle map tasks: Write shuffle outputs
            partitionsToCompute.map { partitionId =>
              new ShuffleMapTask(
                stageId = stage.id,
                taskBinary = broadcast(stage.rdd),
                partition = partitionId,
                locs = getPreferredLocs(stage.rdd, partitionId)
              )
            }

          case resultStage: ResultStage =>
            // Result tasks: Compute final results
            partitionsToCompute.map { partitionId =>
              new ResultTask(
                stageId = stage.id,
                taskBinary = broadcast(stage.rdd),
                partition = partitionId,
                locs = getPreferredLocs(stage.rdd, partitionId)
              )
            }
        }

        // Submit tasks to task scheduler
        taskScheduler.submitTasks(new TaskSet(tasks, stage.id))
      }

      /**
       * Get preferred locations for a partition (data locality)
       */
      private def getPreferredLocs(rdd: RDD[_], partition: Int): Seq[TaskLocation] = {
        rdd.preferredLocations(rdd.partitions(partition))
      }

      /**
       * Handle task completion
       */
      def handleTaskCompletion(event: CompletionEvent): Unit = {
        event.task match {
          case smt: ShuffleMapTask =>
            // Register shuffle output location
            val status = event.result.asInstanceOf[MapStatus]
            mapOutputTracker.registerMapOutput(smt.shuffleId, smt.partitionId, status)

            // Check if stage is complete
            val stage = activeStages(event.task.stageId)
            if (stage.isAvailable) {
              submitWaitingStages()
            }

          case rt: ResultTask =>
            // Handle result
            val job = activeJobs(rt.jobId)
            job.resultHandler(rt.partition, event.result)

            if (job.isComplete) {
              activeJobs.remove(job.jobId)
            }
        }
      }

      /**
       * Handle task failure
       * Recompute using lineage
       */
      def handleTaskFailure(event: TaskFailure): Unit = {
        val stage = activeStages(event.task.stageId)

        // Check if should retry
        if (event.attemptNumber < maxTaskAttempts) {
          // Resubmit failed task
          val task = event.task.copy(attemptNumber = event.attemptNumber + 1)
          taskScheduler.submitTasks(new TaskSet(Seq(task), stage.id))
        } else {
          // Max retries reached → fail stage
          abortStage(stage, s"Task ${event.task.id} failed ${maxTaskAttempts} times")
        }
      }
    }
    ```

    ### Lazy Evaluation Example

    ```scala
    // Spark transformations are lazy (build execution plan)
    val text = sc.textFile("hdfs://data/input")  // Lazy: No data read yet

    val words = text.flatMap(line => line.split(" "))  // Lazy: Plan recorded

    val pairs = words.map(word => (word, 1))  // Lazy: Plan extended

    val counts = pairs.reduceByKey(_ + _)  // Lazy: Shuffle boundary recorded

    // Action triggers execution (entire DAG executes)
    counts.saveAsTextFile("hdfs://data/output")  // Eager: Job submitted

    /**
     * Execution plan:
     *
     * Stage 1 (no shuffle):
     *   textFile → flatMap → map → shuffle write
     *
     * Stage 2 (shuffle read):
     *   shuffle read → reduceByKey → saveAsTextFile
     */
    ```

    ---

    ## 3.2 Data Partitioning

    ### Partitioning Strategies

    ```scala
    /**
     * Partitioner: Determines how data is distributed across partitions
     */

    // 1. Hash Partitioning (Default)
    // Key: Hash-based distribution, uniform load
    val data = sc.parallelize(1 to 1000)
    val pairs = data.map(x => (x, x * 2))

    // Partition by hash of key
    val partitioned = pairs.partitionBy(new HashPartitioner(100))
    // Partition = hash(key) % numPartitions

    // 2. Range Partitioning
    // Key: Sorted keys, efficient for range queries
    val rangePartitioned = pairs.partitionBy(new RangePartitioner(100, pairs))
    // Keys divided into ranges: [0-10), [10-20), [20-30), ...

    // 3. Custom Partitioning
    // Key: Domain-specific logic (e.g., geo-based)
    class GeoPartitioner(partitions: Int) extends Partitioner {
      def numPartitions: Int = partitions

      def getPartition(key: Any): Int = {
        val location = key.asInstanceOf[String]
        // Custom logic: Partition by geography
        location match {
          case loc if loc.startsWith("US") => 0
          case loc if loc.startsWith("EU") => 1
          case loc if loc.startsWith("ASIA") => 2
          case _ => 3
        }
      }
    }

    val geoPartitioned = pairs.partitionBy(new GeoPartitioner(4))
    ```

    ### Optimal Partition Count

    ```scala
    /**
     * Partition sizing guidelines:
     *
     * 1. Partition count:
     *    - Rule of thumb: 2-4 × number of cores
     *    - Example: 1,000 nodes × 32 cores × 3 = 96,000 partitions
     *
     * 2. Partition size:
     *    - Ideal: 100-200 MB per partition
     *    - Too small (< 10 MB): Task overhead dominates
     *    - Too large (> 1 GB): Memory pressure, long tasks
     *
     * 3. Shuffle partitions:
     *    - Default: 200 (spark.sql.shuffle.partitions)
     *    - For 100 TB data: 100 TB / 128 MB = 800,000 partitions
     *    - Set: spark.sql.shuffle.partitions = 800000
     */

    val spark = SparkSession.builder()
      .config("spark.sql.shuffle.partitions", "800000")
      .config("spark.default.parallelism", "96000")
      .getOrCreate()

    // Calculate partitions dynamically
    def calculatePartitions(dataSize: Long, partitionSizeMB: Int = 128): Int = {
      val partitionSizeBytes = partitionSizeMB * 1024 * 1024
      math.max(1, (dataSize / partitionSizeBytes).toInt)
    }

    val dataSize = 100L * 1024 * 1024 * 1024 * 1024  // 100 TB
    val numPartitions = calculatePartitions(dataSize)
    println(s"Recommended partitions: $numPartitions")  // 800,000
    ```

    ---

    ## 3.3 Shuffle Operations

    ### Shuffle Internals

    ```scala
    /**
     * Shuffle: Redistributing data across partitions
     *
     * Phases:
     * 1. Shuffle Write (Map side):
     *    - Each task writes output to local disk
     *    - Partitioned by hash of key
     *    - Sorted within each partition (optional)
     *
     * 2. Shuffle Read (Reduce side):
     *    - Fetch shuffle blocks from all map tasks
     *    - Merge sorted runs
     *    - Apply reduce function
     */

    // Wide transformation triggers shuffle
    val grouped = pairs.groupByKey()  // Shuffle: Hash partitioning
    val reduced = pairs.reduceByKey(_ + _)  // Shuffle: Hash + combine

    // Shuffle write locations
    class ShuffleWriter {
      val shuffleBlockResolver = new IndexShuffleBlockResolver()

      def write(records: Iterator[Product2[K, V]]): Unit = {
        val partitioner = dep.partitioner
        val numPartitions = partitioner.numPartitions

        // Create buckets for each partition
        val buckets = Array.fill(numPartitions)(new ArrayBuffer[(K, V)])

        // Partition records
        for (record <- records) {
          val partition = partitioner.getPartition(record._1)
          buckets(partition) += record
        }

        // Sort within partitions (optional, for sort-based shuffle)
        if (dep.keyOrdering.isDefined) {
          buckets.foreach(bucket => bucket.sortBy(_._1)(dep.keyOrdering.get))
        }

        // Write to disk
        for (i <- 0 until numPartitions) {
          val file = shuffleBlockResolver.getDataFile(shuffleId, mapId, i)
          writePartitionToFile(buckets(i), file)
        }

        // Write index file (partition boundaries)
        val lengths = buckets.map(_.size)
        shuffleBlockResolver.writeIndexFile(shuffleId, mapId, lengths)
      }
    }

    // Shuffle read from remote executors
    class ShuffleReader {
      def read(): Iterator[Product2[K, C]] = {
        val blockManager = SparkEnv.get.blockManager

        // Fetch shuffle blocks from all map tasks
        val blocks = for {
          mapId <- 0 until numMaps
        } yield {
          val blockId = ShuffleBlockId(shuffleId, mapId, reduceId)
          blockManager.getRemoteBlock(blockId)
        }

        // Merge sorted runs (if sort-based shuffle)
        val merged = mergeSortedBlocks(blocks)

        // Apply aggregation function
        val aggregated = applyAggregation(merged, dep.aggregator)

        aggregated
      }
    }
    ```

    ### Shuffle Optimization

    ```scala
    /**
     * Shuffle optimization techniques
     */

    // 1. Map-side combine (reduce shuffle data)
    val pairs = data.map(x => (x % 100, 1))
    val counts = pairs.reduceByKey(_ + _)  // Combines before shuffle
    // Without combine: 1B records shuffled
    // With combine: 100 records shuffled (100x reduction)

    // 2. Broadcast join (avoid shuffle for small tables)
    val largeDf = spark.read.parquet("hdfs://large")  // 10 TB
    val smallDf = spark.read.parquet("hdfs://small")  // 100 MB

    // Regular join: Both sides shuffle
    val joined1 = largeDf.join(smallDf, "key")  // 10 TB shuffle

    // Broadcast join: Only large side processes, no shuffle
    import org.apache.spark.sql.functions.broadcast
    val joined2 = largeDf.join(broadcast(smallDf), "key")  // 0 shuffle!

    // 3. Coalesce vs. Repartition
    val large = sc.parallelize(1 to 1000000, 10000)  // 10K partitions

    // Reduce partitions without shuffle (coalesce)
    val coalesced = large.coalesce(1000)  // Combine partitions locally

    // Repartition with shuffle (uniform distribution)
    val repartitioned = large.repartition(1000)  // Full shuffle

    // 4. Partition pruning (skip irrelevant partitions)
    val partitioned = data.partitionBy("date")
    partitioned.write.parquet("hdfs://output")

    // Only read relevant partitions
    val filtered = spark.read.parquet("hdfs://output")
      .filter("date = '2025-01-01'")  // Only 1 partition read!
    ```

    ---

    ## 3.4 Fault Tolerance

    ### Lineage-based Recovery

    ```scala
    /**
     * Lineage: Metadata tracking how RDD is derived from inputs
     *
     * Benefits:
     * 1. No checkpointing overhead for most operations
     * 2. Fine-grained recovery (only lost partitions)
     * 3. Automatic recovery (no user intervention)
     */

    // Example: Lineage graph
    val input = sc.textFile("hdfs://data/input")  // Lineage: HDFS path
    val mapped = input.map(line => line.length)   // Lineage: map(input)
    val filtered = mapped.filter(_ > 10)          // Lineage: filter(mapped)
    val sum = filtered.reduce(_ + _)              // Lineage: reduce(filtered)

    // Failure scenario:
    // - Task computing filtered partition 5 fails on Node 10
    // - Spark recomputes: input.partition(5).map(...).filter(...)
    // - Reads HDFS partition 5, applies transformations
    // - No impact on other partitions

    /**
     * Narrow dependencies: Fast recovery (local recomputation)
     * Wide dependencies: Slower recovery (may need shuffle re-execution)
     */

    // Narrow dependency: map, filter, union
    class OneToOneDependency[T](rdd: RDD[T]) extends NarrowDependency[T](rdd) {
      override def getParents(partitionId: Int): Seq[Int] = List(partitionId)
      // Recovery: Recompute only parent partition (fast)
    }

    // Wide dependency: groupByKey, reduceByKey, join
    class ShuffleDependency[K, V, C](
        rdd: RDD[_ <: Product2[K, V]],
        partitioner: Partitioner) extends Dependency[Product2[K, V]] {
      // Recovery: Re-fetch shuffle data or recompute all map outputs (slower)
    }
    ```

    ### Checkpointing

    ```scala
    /**
     * Checkpointing: Persist RDD to reliable storage
     *
     * Use cases:
     * 1. Long lineage chains (iterative algorithms)
     * 2. Wide dependencies (expensive to recompute)
     * 3. Stability (prevent cascading recomputation)
     */

    // Configure checkpoint directory
    sc.setCheckpointDir("hdfs://checkpoints")

    // Iterative algorithm: PageRank
    var ranks = pages.mapValues(_ => 1.0)

    for (i <- 1 to 10) {
      val contribs = links.join(ranks).values.flatMap {
        case (urls, rank) => urls.map(url => (url, rank / urls.size))
      }

      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)

      // Checkpoint every 5 iterations
      if (i % 5 == 0) {
        ranks.checkpoint()
        ranks.count()  // Materialize checkpoint
      }
    }

    /**
     * Checkpoint trade-offs:
     * - Pros: Fast recovery, truncate lineage
     * - Cons: I/O overhead, storage cost
     * - When to use: Lineage > 10 levels, or many wide dependencies
     */
    ```

    ### Speculative Execution

    ```scala
    /**
     * Speculative Execution: Re-run slow tasks on other nodes
     *
     * Problem: Stragglers (slow tasks due to bad nodes, data skew)
     * Solution: Launch duplicate tasks, use result from first to complete
     */

    // Enable speculative execution
    val conf = new SparkConf()
      .set("spark.speculation", "true")
      .set("spark.speculation.interval", "100ms")  // Check every 100ms
      .set("spark.speculation.multiplier", "1.5")  // Launch if 1.5x slower
      .set("spark.speculation.quantile", "0.75")   // Wait for 75% tasks

    /**
     * Algorithm:
     * 1. Monitor task progress (every 100ms)
     * 2. Calculate median task duration
     * 3. If task > 1.5x median and 75% tasks complete:
     *    → Launch speculative copy on different node
     * 4. Use result from first completed task
     * 5. Kill duplicate task
     *
     * Example:
     * - 1,000 tasks in stage
     * - 750 tasks complete in 2 minutes
     * - Task 500 running for 3 minutes (1.5x median)
     * - Launch speculative copy on Node 200
     * - Speculative copy completes in 1 minute
     * - Kill original task on Node 100
     */
    ```

    ---

    ## 3.5 Memory Management

    ```scala
    /**
     * Spark Memory Model: Unified memory management
     *
     * Memory regions:
     * 1. Execution memory: Shuffle, join, sort, aggregation
     * 2. Storage memory: Cache, broadcast variables
     * 3. User memory: User data structures
     * 4. Reserved memory: Spark internals (300 MB)
     *
     * Total executor memory = Execution + Storage + User + Reserved
     */

    val conf = new SparkConf()
      .set("spark.executor.memory", "50g")         // Total executor memory
      .set("spark.memory.fraction", "0.6")         // Execution + Storage: 60%
      .set("spark.memory.storageFraction", "0.5")  // Storage within unified: 50%

    /**
     * Memory calculation:
     * - Total: 50 GB
     * - Reserved: 300 MB
     * - Usable: 50 GB - 300 MB = 49.7 GB
     * - Execution + Storage: 49.7 GB × 0.6 = 29.82 GB
     * - Storage (initial): 29.82 GB × 0.5 = 14.91 GB
     * - Execution (initial): 29.82 GB × 0.5 = 14.91 GB
     * - User: 49.7 GB × 0.4 = 19.88 GB
     *
     * Unified memory: Storage can borrow from Execution and vice versa
     */

    // Spill to disk when memory full
    class ExternalSorter[K, V, C] {
      private val map = new PartitionedAppendOnlyMap[K, C]
      private var spilledMaps = new ArrayBuffer[SpilledFile]

      def insertAll(records: Iterator[Product2[K, V]]): Unit = {
        for (record <- records) {
          map.changeValue(record._1, update = _ => combiner(record._2))

          // Check memory usage
          if (map.estimateSize() > maxMemory) {
            spillToDisk()
          }
        }
      }

      def spillToDisk(): Unit = {
        // Sort in-memory data
        val sorted = map.destructiveSortedIterator(keyComparator)

        // Write to disk
        val file = File.createTempFile("spill", ".bin")
        val writer = new ObjectOutputStream(new FileOutputStream(file))

        for ((k, v) <- sorted) {
          writer.writeObject((k, v))
        }

        writer.close()
        spilledMaps += SpilledFile(file, map.size)

        // Clear in-memory map
        map.clear()
      }

      def iterator: Iterator[(K, C)] = {
        // Merge in-memory and spilled data
        val inMemory = map.iterator
        val onDisk = spilledMaps.map(f => readSpilledFile(f))

        mergeSort(inMemory +: onDisk)
      }
    }
    ```

=== "⚖️ Step 4: Scale & Optimize"

    ## Scalability Strategies

    ### Dynamic Resource Allocation

    ```scala
    /**
     * Dynamic Resource Allocation: Adjust executors based on workload
     *
     * Benefits:
     * 1. Better cluster utilization
     * 2. Cost savings (no idle executors)
     * 3. Automatic scaling
     */

    val conf = new SparkConf()
      .set("spark.dynamicAllocation.enabled", "true")
      .set("spark.dynamicAllocation.minExecutors", "10")
      .set("spark.dynamicAllocation.maxExecutors", "1000")
      .set("spark.dynamicAllocation.initialExecutors", "100")
      .set("spark.dynamicAllocation.executorIdleTimeout", "60s")
      .set("spark.dynamicAllocation.schedulerBacklogTimeout", "1s")

    /**
     * Algorithm:
     * 1. Start with initialExecutors (100)
     * 2. If tasks pending > 1 second → request more executors
     * 3. Scale up exponentially: +1, +2, +4, +8, ... (up to max)
     * 4. If executor idle > 60 seconds → remove executor
     * 5. Keep at least minExecutors
     *
     * Example scenario:
     * - Job starts: 100 executors
     * - Task backlog detected: Scale to 200, 400, 800, 1000
     * - Job completes: Idle executors removed
     * - Final: 10 executors (min)
     */
    ```

    ### Data Skew Handling

    ```python
    """
    Data Skew: Uneven distribution of data across partitions

    Problems:
    1. Few tasks process most data (hot keys)
    2. Other tasks finish quickly, sit idle
    3. Job latency dominated by slowest task

    Solutions:
    """

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand, concat, lit

    spark = SparkSession.builder.getOrCreate()

    # Example: Skewed join
    # Large table: 1B records, Small table: 1M records
    # Problem: Key "popular_product" appears 100M times
    large_df = spark.read.parquet("hdfs://large")
    small_df = spark.read.parquet("hdfs://small")

    # Solution 1: Salting (add random suffix to hot keys)
    salt_factor = 100  # Number of salt values

    # Add salt to large table
    large_salted = large_df.withColumn(
        "salted_key",
        concat(col("key"), lit("_"), (rand() * salt_factor).cast("int"))
    )

    # Replicate small table with all salt values
    salt_df = spark.range(salt_factor).selectExpr("id as salt")
    small_replicated = small_df.crossJoin(salt_df).withColumn(
        "salted_key",
        concat(col("key"), lit("_"), col("salt"))
    )

    # Join on salted key
    joined = large_salted.join(small_replicated, "salted_key")

    """
    Result: Hot key "popular_product" split into 100 keys:
    - popular_product_0: 1M records
    - popular_product_1: 1M records
    - ...
    - popular_product_99: 1M records

    Benefit: 100x parallelism for hot key
    """

    # Solution 2: Adaptive Query Execution (AQE)
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")

    """
    AQE detects skew at runtime:
    1. Partition > 256 MB and 5x median → skewed
    2. Split skewed partition into smaller chunks
    3. Dynamically adjust execution plan
    """

    # Solution 3: Broadcast join (if small table fits in memory)
    from pyspark.sql.functions import broadcast

    joined_broadcast = large_df.join(broadcast(small_df), "key")
    # No shuffle! Small table broadcasted to all executors
    ```

    ### Broadcast Variables

    ```scala
    /**
     * Broadcast Variables: Efficiently share read-only data across executors
     *
     * Use cases:
     * 1. Lookup tables (user profiles, product catalogs)
     * 2. Machine learning models
     * 3. Configuration data
     */

    // Without broadcast: Sent with every task (wasteful)
    val lookup = Map("A" -> 1, "B" -> 2, "C" -> 3)  // 1 GB lookup table

    val data = sc.parallelize(1 to 1000000, 10000)  // 10K tasks
    val result = data.map { x =>
      lookup.get(x.toString)  // Lookup table sent 10K times = 10 TB!
    }

    // With broadcast: Sent once per executor (efficient)
    val lookupBroadcast = sc.broadcast(lookup)  // 1 GB table

    val result2 = data.map { x =>
      lookupBroadcast.value.get(x.toString)  // Table sent to 4K executors = 4 GB
    }

    /**
     * Broadcast protocol:
     * 1. Driver serializes variable
     * 2. Breaks into blocks (4 MB each)
     * 3. Sends to executors via BitTorrent-like protocol
     * 4. Executors cache in memory/disk
     * 5. Tasks access local copy
     *
     * Limits:
     * - Max broadcast size: 8 GB (spark.driver.maxResultSize)
     * - Recommendation: Keep < 1 GB for efficiency
     */
    ```

    ### Partition Coalescing

    ```python
    """
    Partition Coalescing: Reduce partition count without full shuffle
    """

    # Scenario: Many small files (small partition problem)
    df = spark.read.csv("hdfs://input")  # 10,000 files, 10 MB each
    print(df.rdd.getNumPartitions())  # 10,000 partitions

    # Problem: 10,000 tasks for 100 GB data (overhead)
    # Ideal: 800 partitions (128 MB each)

    # Solution 1: Coalesce (no shuffle, local combine)
    df_coalesced = df.coalesce(800)
    # Combines multiple partitions on same executor
    # Fast, but may not be uniform

    # Solution 2: Repartition (full shuffle, uniform)
    df_repartitioned = df.repartition(800)
    # Shuffles data for uniform distribution
    # Slower, but balanced

    # When to use:
    # - Coalesce: Reduce partitions, data already balanced
    # - Repartition: Need uniform distribution or increase partitions
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Broadcast join** | Eliminate shuffle for small tables | Memory overhead (broadcast to all executors) |
    | **Map-side combine** | 10-100x reduction in shuffle data | CPU overhead for pre-aggregation |
    | **Partition pruning** | Skip irrelevant partitions (10-100x faster) | Requires partitioned data (storage overhead) |
    | **Adaptive execution** | Auto-optimize at runtime (2-5x faster) | Runtime overhead for statistics collection |
    | **Columnar storage** | 5-10x compression, faster reads | Write overhead (encoding, compression) |
    | **Speculative execution** | Reduce long tail latency (10-30%) | Wasted resources on duplicate tasks |

    ---

    ## Code Examples

    ### Spark DataFrame Example

    ```python
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, count, avg, sum, window

    # Create Spark session
    spark = SparkSession.builder \
        .appName("BatchProcessing") \
        .config("spark.executor.memory", "50g") \
        .config("spark.executor.cores", "5") \
        .config("spark.sql.shuffle.partitions", "800000") \
        .getOrCreate()

    # Read data from S3
    events = spark.read.parquet("s3://data-lake/events/")

    # Transformation: Filter, aggregate, join
    # Lazy evaluation: No computation yet
    filtered = events.filter(col("timestamp") >= "2025-01-01")

    # Aggregation by window
    aggregated = filtered.groupBy(
        window(col("timestamp"), "1 hour"),
        col("user_id")
    ).agg(
        count("*").alias("event_count"),
        sum("revenue").alias("total_revenue")
    )

    # Join with user profiles
    users = spark.read.parquet("s3://data-lake/users/")
    enriched = aggregated.join(users, "user_id", "left")

    # Action: Trigger execution
    enriched.write \
        .mode("overwrite") \
        .partitionBy("window") \
        .parquet("s3://data-lake/output/")

    # Spark UI: View DAG and execution plan
    # http://driver-node:4040
    ```

    ### Low-Level RDD Example

    ```scala
    import org.apache.spark.{SparkConf, SparkContext}

    val conf = new SparkConf().setAppName("WordCount")
    val sc = new SparkContext(conf)

    // Read text files
    val text = sc.textFile("hdfs://data/input", minPartitions = 10000)

    // Transformation: Split lines, count words
    val counts = text
      .flatMap(line => line.split("\\s+"))      // Split into words
      .map(word => (word.toLowerCase, 1))       // Create key-value pairs
      .reduceByKey(_ + _)                        // Aggregate by key (shuffle)
      .sortBy(_._2, ascending = false)           // Sort by count
      .take(100)                                 // Top 100 words

    // Save results
    sc.parallelize(counts).saveAsTextFile("hdfs://data/output")

    // Stop Spark context
    sc.stop()
    ```

    ### Advanced: Join Optimization

    ```scala
    import org.apache.spark.sql.functions._

    // Scenario: Join large fact table with dimension table
    val facts = spark.read.parquet("hdfs://facts")    // 10 TB
    val dims = spark.read.parquet("hdfs://dimensions")  // 10 GB

    // Strategy 1: Broadcast join (small dimension)
    val joined1 = facts.join(broadcast(dims), "dim_id")
    // Broadcast 10 GB to all executors, no shuffle

    // Strategy 2: Bucket join (pre-partitioned tables)
    facts.write
      .bucketBy(1000, "dim_id")
      .sortBy("dim_id")
      .saveAsTable("facts_bucketed")

    dims.write
      .bucketBy(1000, "dim_id")
      .sortBy("dim_id")
      .saveAsTable("dims_bucketed")

    val joined2 = spark.table("facts_bucketed")
      .join(spark.table("dims_bucketed"), "dim_id")
    // No shuffle! Data already co-located

    // Strategy 3: Salting for skewed joins
    val saltFactor = 100
    val factsSalted = facts.withColumn(
      "salt",
      (rand() * saltFactor).cast("int")
    )

    val dimsReplicated = dims
      .crossJoin(spark.range(saltFactor).toDF("salt"))

    val joined3 = factsSalted.join(
      dimsReplicated,
      factsSalted("dim_id") === dimsReplicated("dim_id") &&
      factsSalted("salt") === dimsReplicated("salt")
    )
    // Hot keys distributed across 100 partitions
    ```

    ---

    ## Monitoring Metrics

    ```python
    # Key metrics for batch processing health

    # Job metrics
    job_duration_seconds = "spark_job_duration_seconds"
    job_num_tasks = "spark_job_num_tasks"
    job_num_completed_tasks = "spark_job_num_completed_tasks"

    # Stage metrics
    stage_duration_seconds = "spark_stage_duration_seconds"
    stage_num_tasks = "spark_stage_num_tasks"
    stage_input_bytes = "spark_stage_input_bytes"
    stage_shuffle_read_bytes = "spark_stage_shuffle_read_bytes"
    stage_shuffle_write_bytes = "spark_stage_shuffle_write_bytes"

    # Task metrics
    task_duration_seconds = "spark_task_duration_seconds"
    task_result_size_bytes = "spark_task_result_size_bytes"
    task_failed_count = "spark_task_failed_count"

    # Executor metrics
    executor_memory_used = "spark_executor_memory_used_bytes"
    executor_disk_used = "spark_executor_disk_used_bytes"
    executor_gc_time = "spark_executor_gc_time_seconds"
    executor_active_tasks = "spark_executor_active_tasks"

    # Shuffle metrics
    shuffle_local_blocks_read = "spark_shuffle_local_blocks_read"
    shuffle_remote_blocks_read = "spark_shuffle_remote_blocks_read"
    shuffle_bytes_written = "spark_shuffle_bytes_written"
    shuffle_records_written = "spark_shuffle_records_written"

    # Alerting rules
    alerts = {
        "high_task_failure_rate": "rate(spark_task_failed_count[5m]) > 0.05",
        "high_gc_time": "spark_executor_gc_time_seconds / spark_executor_runtime_seconds > 0.1",
        "high_shuffle_spill": "spark_executor_disk_used_bytes > 100GB",
        "straggler_tasks": "spark_task_duration_seconds > 10 * avg(spark_task_duration_seconds)"
    }
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"What's the difference between RDD and DataFrame?"**
   - RDD: Low-level API, no schema, no optimization
   - DataFrame: High-level API, schema, Catalyst optimizer
   - Use DataFrame for structured data (10x faster)
   - Use RDD for unstructured or complex transformations

2. **"How do you handle data skew?"**
   - Salting: Add random suffix to hot keys
   - Two-phase aggregation: Local + global aggregation
   - Broadcast join: Avoid shuffle for small tables
   - Adaptive execution: Runtime detection and optimization

3. **"Explain narrow vs. wide dependencies"**
   - Narrow: Each parent partition used by at most one child (map, filter)
   - Wide: Multiple child partitions depend on one parent (groupBy, join)
   - Wide dependencies require shuffle (stage boundary)

4. **"How does Spark achieve fault tolerance?"**
   - Lineage: Track transformations, recompute lost partitions
   - Checkpointing: Persist to HDFS for long lineage chains
   - Speculative execution: Re-run slow tasks
   - Data replication: HDFS replicates blocks (3x)

5. **"When would you use checkpointing?"**
   - Iterative algorithms (PageRank, machine learning)
   - Long lineage chains (> 10 levels)
   - Many wide dependencies (expensive shuffle)
   - Trade-off: I/O overhead vs. recovery time

6. **"How do you optimize shuffle operations?"**
   - Reduce shuffle data: Map-side combine, pre-aggregation
   - Increase parallelism: More partitions (spark.sql.shuffle.partitions)
   - Use broadcast join: Avoid shuffle for small tables
   - Tune memory: Increase spark.memory.fraction for shuffle

7. **"What causes OOM errors in Spark?"**
   - Too few partitions (large partition size)
   - Excessive caching (eviction disabled)
   - Data skew (hot keys consume memory)
   - User memory leak (accumulator misuse)
   - Solution: Repartition, tune memory, fix skew, review code

8. **"How do you determine optimal partition count?"**
   - Rule: 2-4x number of cores
   - Partition size: 100-200 MB ideal
   - Too few: Poor parallelism, large memory
   - Too many: Task overhead, scheduler pressure

**Key Points to Mention:**

- DAG execution with lazy evaluation
- Lineage-based fault tolerance
- Data partitioning and locality
- Shuffle optimization (map-side combine, broadcast)
- Wide vs. narrow dependencies
- Speculative execution for stragglers
- Resource management with YARN
- Memory management and spill to disk

---

## Real-World Examples

**Uber (Trip Data Processing):**
- 100 PB daily trip data processed with Spark
- 10,000+ Spark jobs per day
- HDFS on thousands of nodes
- Use case: Fare calculation, surge pricing, route optimization

**Netflix (Recommendation Pipeline):**
- Spark processes viewing history (500 TB/day)
- Generate recommendations for 200M+ users
- Iterative algorithms: Matrix factorization
- Use case: Personalized recommendations, A/B testing

**Airbnb (Search Ranking):**
- Spark processes search logs (50 TB/day)
- Train ranking models on booking history
- Feature engineering with Spark SQL
- Use case: Personalized search results, pricing models

---

## Summary

**System Characteristics:**

- **Scale:** 100 TB/day, 1000 nodes, 10B records processed
- **Throughput:** 7 GB/sec sustained, 30 TB shuffle
- **Latency:** Minutes to hours (batch processing)
- **Fault Tolerance:** Lineage recovery, checkpointing, speculative execution

**Core Components:**

1. **Spark Driver:** DAG scheduler, task scheduler, lineage tracking
2. **YARN ResourceManager:** Resource allocation, container management
3. **Executors:** Task execution, block manager, caching
4. **Shuffle Manager:** Sort-based shuffle, spill to disk
5. **HDFS:** Distributed storage, replication, data locality
6. **Block Manager:** In-memory caching, partition management

**Key Design Decisions:**

- DAG execution with lazy evaluation (optimize before compute)
- Lineage-based fault tolerance (no checkpoint overhead)
- Data partitioning (parallelism and locality)
- Shuffle optimization (map-side combine, broadcast join)
- Speculative execution (handle stragglers)
- Dynamic resource allocation (scale with workload)
- Unified memory management (execution + storage)
- Columnar storage (Parquet/ORC for compression)

This design provides a scalable, fault-tolerant batch processing system capable of processing petabytes of data across thousands of nodes with optimized shuffle operations, efficient memory management, and comprehensive fault tolerance mechanisms.
