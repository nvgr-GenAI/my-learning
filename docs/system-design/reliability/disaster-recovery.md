# Disaster Recovery

Disaster recovery is what separates a company that survives a catastrophic failure from
one that doesn't. When an entire datacenter floods, a cloud region goes down, or a
ransomware attack encrypts your databases, your disaster recovery plan determines
whether you lose minutes of data or months of it — and whether recovery takes hours or
weeks.

The uncomfortable truth: **most companies discover their disaster recovery plan doesn't
work during the actual disaster.** Backups that were never tested turn out to be
corrupted. Failover scripts reference servers that no longer exist. Recovery procedures
assume manual steps by people who left the company two years ago. The only reliable
disaster recovery plan is one that is tested regularly and automated thoroughly.

---

## RTO and RPO: The Two Numbers That Define Your Strategy

Every disaster recovery plan is built around two metrics:

```
TIMELINE OF A DISASTER

Past ◄──────────────────────────── Present ────────────────────────► Future
                                      │
      ◄── RPO ──►                     │              ◄── RTO ──►
      │          │                  DISASTER          │          │
      │          │                  OCCURS            │          │
  Last backup  Data loss            HERE          Recovery     System
  taken here   boundary                           begins       fully
                                                               operational

RPO = How much data can you AFFORD TO LOSE?
RTO = How long can you AFFORD TO BE DOWN?
```

**RPO (Recovery Point Objective):** The maximum acceptable amount of data loss measured
in time. An RPO of 1 hour means you can tolerate losing up to 1 hour of data. An RPO
of zero means no data loss is acceptable.

**RTO (Recovery Time Objective):** The maximum acceptable downtime. An RTO of 4 hours
means the system must be operational within 4 hours of a disaster.

### The Cost-Recovery Trade-off

Tighter RPO and RTO cost exponentially more:

| RPO | RTO | Strategy | Approximate Cost | Example Use Case |
|---|---|---|---|---|
| 24 hours | 24 hours | Daily backups to cold storage | $ | Internal tools, logs |
| 4 hours | 4 hours | Frequent backups + warm standby | $$ | Content management systems |
| 1 hour | 1 hour | Continuous backup + hot standby | $$$ | E-commerce, SaaS products |
| Minutes | Minutes | Multi-region active-active | $$$$ | Financial trading, payments |
| Zero | Seconds | Synchronous replication + auto-failover | $$$$$ | Banking core, healthcare |

```
COST vs RECOVERY CAPABILITY

Cost $$$$$ │                                    ●  Zero RPO/RTO
           │                               ●       (sync replication,
           │                          ●              multi-region active)
           │                     ●
           │                ●
           │           ●
           │      ●
           │ ●
           └──────────────────────────────────────
           24h    12h    4h    1h   15m   0   ← RPO/RTO
```

The key business decision: **what is the cost of downtime versus the cost of
prevention?** An e-commerce site losing $100K/hour in revenue can justify expensive
multi-region failover. An internal wiki cannot.

---

=== "Backup Strategies"

    ## Backup Types

    ### Full Backup

    A complete copy of all data. Simple to restore but expensive in storage and time.

    ```
    FULL BACKUP SCHEDULE

    Sunday:  FULL BACKUP (100 GB, takes 4 hours)
    Monday:  FULL BACKUP (100 GB, takes 4 hours)
    Tuesday: FULL BACKUP (100 GB, takes 4 hours)
    ...

    Storage used per week: 700 GB
    Restore time: Fast (single restore operation)
    RPO: Up to 24 hours (last backup)
    ```

    ### Incremental Backup

    Only backs up data that changed since the **last backup** (full or incremental).
    Fast and storage-efficient, but restoration requires the full backup plus every
    subsequent incremental.

    ```
    INCREMENTAL BACKUP SCHEDULE

    Sunday:    FULL BACKUP (100 GB)
    Monday:    INCREMENTAL (+2 GB changed since Sunday)
    Tuesday:   INCREMENTAL (+3 GB changed since Monday)
    Wednesday: INCREMENTAL (+2 GB changed since Tuesday)
    ...

    Storage used per week: ~115 GB (vs 700 GB for daily fulls)
    Restore: Full + Mon + Tue + Wed + ... (chain of restores)
    Risk: If Tuesday's backup is corrupted, Wed-Sat are useless
    ```

    ### Differential Backup

    Backs up everything that changed since the **last full backup**. A middle ground:
    restoration needs only the full backup plus the latest differential.

    ```
    DIFFERENTIAL BACKUP SCHEDULE

    Sunday:    FULL BACKUP (100 GB)
    Monday:    DIFFERENTIAL (+2 GB changed since Sunday)
    Tuesday:   DIFFERENTIAL (+5 GB changed since Sunday)
    Wednesday: DIFFERENTIAL (+7 GB changed since Sunday)
    ...

    Storage: More than incremental, less than full
    Restore: Full + latest differential (just two operations)
    Simpler and safer than incremental chains
    ```

    | Strategy | Storage Cost | Backup Speed | Restore Speed | Restore Complexity |
    |---|---|---|---|---|
    | Full | Highest | Slowest | Fastest | Simple: single restore |
    | Incremental | Lowest | Fastest | Slowest | Complex: full + chain |
    | Differential | Medium | Medium | Medium | Simple: full + latest |

    ### The 3-2-1 Rule

    The industry standard for backup resilience:

    ```
    3 copies of your data
    │
    ├── 2 different storage media (e.g., disk + cloud)
    │
    └── 1 copy offsite (different geographic location)
    ```

    **Why:** A single backup on the same server is useless if the server is destroyed.
    Two backups in the same datacenter are useless if the datacenter floods. The offsite
    copy protects against regional disasters.

    **Modern extension (3-2-1-1-0):**
    - 3 copies, 2 media, 1 offsite
    - **1 copy offline/air-gapped** (protects against ransomware)
    - **0 errors** (verify backups regularly with test restores)

=== "Recovery Patterns"

    ## Disaster Recovery Architectures

    ### Cold Standby

    Infrastructure exists but is not running. In a disaster, you provision servers,
    restore from backups, and bring the system online.

    ```
    NORMAL OPERATION:
    ┌──────────────┐
    │ Primary Site  │ ← all traffic
    │ (running)     │
    └──────────────┘

    ┌──────────────┐
    │ DR Site       │ ← off, but backups stored here
    │ (powered off) │
    └──────────────┘

    DURING DISASTER:
    1. Detect failure (minutes to hours)
    2. Provision infrastructure at DR site (30-60 minutes)
    3. Restore from backups (hours)
    4. Verify and test (hours)
    5. Redirect traffic
    Total RTO: 8-24 hours
    ```

    **Cost:** Lowest — you only pay for storage of backups until disaster strikes.
    **RPO:** Hours (last backup). **RTO:** Hours to a full day.

    ### Warm Standby

    A scaled-down copy of the production environment runs continuously with data
    replication. During a disaster, scale up and redirect traffic.

    ```
    NORMAL OPERATION:
    ┌──────────────┐         ┌──────────────┐
    │ Primary Site  │──sync──→│ Warm Standby  │
    │ (full scale)  │  data   │ (minimal      │
    │ 10 servers    │         │  scale: 2)    │
    └──────────────┘         └──────────────┘

    DURING DISASTER:
    1. Detect failure (minutes)
    2. Scale up warm standby (10-30 minutes)
    3. Redirect traffic
    Total RTO: 30 minutes - 2 hours
    ```

    **Cost:** Moderate — running a small environment 24/7 plus data replication.
    **RPO:** Minutes (continuous replication). **RTO:** 30 minutes to 2 hours.

    ### Hot Standby (Active-Passive)

    A full-scale replica of production runs continuously, receiving replicated data in
    near-real-time. Failover is nearly instant.

    ```
    NORMAL OPERATION:
    ┌──────────────┐         ┌──────────────┐
    │ Primary Site  │──sync──→│ Hot Standby   │
    │ (active)      │  real-  │ (passive,     │
    │ handles all   │  time   │  full scale)  │
    │ traffic       │         │               │
    └──────────────┘         └──────────────┘

    DURING DISASTER:
    1. Detect failure (seconds-minutes)
    2. Redirect traffic to standby (seconds)
    Total RTO: 1-15 minutes
    ```

    **Cost:** High — running a full duplicate environment 24/7.
    **RPO:** Seconds to minutes. **RTO:** Minutes.

    ### Active-Active (Multi-Region)

    Both sites handle traffic simultaneously. If one fails, the other absorbs all
    traffic with no failover delay.

    ```
    NORMAL OPERATION:
    ┌──────────────┐  ◄── bi-directional ──►  ┌──────────────┐
    │ Region A      │      replication         │ Region B      │
    │ (50% traffic) │                          │ (50% traffic) │
    └──────────────┘                          └──────────────┘

    REGION A FAILS:
    ┌──────────────┐                          ┌──────────────┐
    │ Region A      │                          │ Region B      │
    │    💀 DOWN     │           ──────────────→│ (100% traffic)│
    └──────────────┘                          └──────────────┘
    No failover delay — Region B was already handling traffic.
    ```

    **Cost:** Highest — two full production environments plus complex data replication.
    **RPO:** Near-zero (synchronous replication) or seconds (async).
    **RTO:** Near-zero — the other region is already serving.

    | Strategy | RPO | RTO | Cost | Complexity |
    |---|---|---|---|---|
    | Cold standby | Hours | 8-24 hours | $ | Low |
    | Warm standby | Minutes | 30min - 2hr | $$ | Medium |
    | Hot standby | Seconds-minutes | 1-15 minutes | $$$ | High |
    | Active-active | Near-zero | Near-zero | $$$$ | Very high |

=== "Data Replication"

    ## Data Replication for DR

    The backup and replication strategy determines your RPO.

    ### Synchronous Replication

    Every write is confirmed by both the primary and the replica before acknowledging
    to the client. Guarantees zero data loss but adds latency.

    ```
    Client → Primary DB → Replica DB → Acknowledgment → Client
                          (must confirm)
    Latency: Primary write time + network round trip + replica write time
    RPO: ZERO (no data loss possible)
    Trade-off: Higher write latency, lower throughput
    ```

    **Used by:** Banking systems, stock exchanges, payment processing — where losing
    even a single transaction is unacceptable.

    **Cross-region latency problem:** Synchronous replication across regions adds
    50-200ms per write (speed of light between datacenters). Google Spanner solves this
    with TrueTime and Paxos, accepting the latency cost for global consistency.

    ### Asynchronous Replication

    Writes are acknowledged by the primary immediately. The replica receives updates
    with a delay (replication lag).

    ```
    Client → Primary DB → Acknowledgment → Client
                    │
                    └──→ Replica DB (catches up later, lag = seconds to minutes)

    RPO: Replication lag (typically seconds, can be minutes under load)
    Trade-off: No latency impact, but data loss up to lag amount
    ```

    **Used by:** Most web applications, content platforms, social networks — where
    losing a few seconds of data during a regional failure is acceptable.

    ### Semi-Synchronous Replication

    A compromise: wait for at least one replica to confirm, but not all. Provides
    stronger durability than async without the full latency cost of sync.

    **MySQL semi-synchronous replication** waits for at least one replica to acknowledge
    receiving the transaction log before confirming to the client. This guarantees that
    at least two copies exist, so failover to that replica loses no data.

---

## Real-World Disaster Recovery

### AWS Region Failure (US-East-1, 2017)

When US-East-1 experienced a major S3 outage (caused by an engineer accidentally
removing more servers than intended), services across the internet went down — even
services not hosted on AWS, because they depended on S3-hosted assets.

**Companies that survived:** Those with multi-region architectures (Netflix, which
operated across three regions). **Companies affected:** Those with single-region
deployments, even if they had "high availability" within one region.

**Lesson:** High availability within a single region does not protect against region
failure. True disaster recovery requires geographic distribution.

### GitLab Database Deletion (2017)

A GitLab engineer accidentally deleted the production database during a maintenance
operation. Their backup investigation revealed:

- **pg_dump backups:** Hadn't worked for months (silently failing)
- **LVM snapshots:** Not configured for the database server
- **Azure disk snapshots:** Not enabled
- **Replication:** The replica was being rebuilt and had no data

The only working backup was an LVM snapshot taken 6 hours earlier by coincidence.
GitLab lost 6 hours of production data.

**Lesson:** Backups that are not regularly tested are not backups. GitLab now runs
automated backup verification daily and publicly shares their disaster recovery
practices.

### Cloudflare (2020)

A configuration change intended for a subset of servers was accidentally deployed
globally, taking down Cloudflare's network for 27 minutes. Because Cloudflare sits
between users and millions of websites, the outage affected a significant portion of
the internet.

**Recovery:** Cloudflare had automated rollback procedures that reverted the change
within minutes of detection, but the global propagation of the fix took additional time.

**Lesson:** Configuration changes are a leading cause of outages. Canary deployments
(roll out to 1% of servers first) would have caught the issue before global impact.

---

## DR Testing: The Part Everyone Skips

A disaster recovery plan that has never been tested is a plan that will fail when you
need it most.

### Types of DR Tests

| Test Type | What You Do | Effort | Confidence |
|---|---|---|---|
| **Tabletop exercise** | Walk through the plan on paper, discuss scenarios | Low | Low |
| **Backup restore test** | Actually restore a backup to a new environment | Medium | Medium |
| **Partial failover** | Fail over a non-critical service to DR site | Medium | Medium-High |
| **Full failover** | Fail over entire production to DR site | High | High |
| **Chaos Day** | Simulate region failure during business hours | High | Highest |

### Testing Cadence

- **Backup verification:** Daily (automated — restore and validate checksums)
- **Tabletop exercise:** Quarterly (update runbooks, train new team members)
- **Partial failover:** Monthly (rotate which services are tested)
- **Full failover:** Annually (ideally bi-annually)

**Netflix** runs Chaos Kong (full region failover) regularly. **Google** runs DiRT
(Disaster Recovery Testing) annually, simulating scenarios like "what if the entire
corporate network goes down?" Their 2022 DiRT exercise simulated a solar storm
disrupting global communications.

---

## Key Takeaways

1. **Define RPO and RTO before choosing a strategy.** These are business decisions, not
   technical ones. The CFO must agree that losing 4 hours of data is acceptable before
   you build a 4-hour RPO solution.

2. **The 3-2-1 backup rule is the minimum.** Three copies, two media, one offsite. In
   the age of ransomware, add an air-gapped copy and verify with test restores.

3. **Untested backups are not backups.** Automate backup restoration testing. Verify
   data integrity daily. If you haven't restored a backup in the last month, you don't
   know if it works.

4. **Active-active is the gold standard but not always justified.** The cost and
   complexity are enormous. For most services, warm standby with sub-hour RPO/RTO is
   the right balance.

5. **Configuration changes cause more outages than hardware failures.** Canary
   deployments and automated rollback are the most cost-effective DR investments.

6. **DR is a team skill, not a document.** Runbooks go stale. Engineers leave. Regular
   testing keeps the team practiced and the procedures current. The team that can
   recover from a disaster is the one that practices recovery regularly.

---

## Related Topics

- [Fault Tolerance](fault-tolerance.md) — preventing disasters through redundancy and resilience
- [Chaos Engineering](chaos-engineering.md) — proactively testing disaster recovery readiness
- [Database Replication](../data/databases/replication.md) — data replication strategies in depth
- [Deployment Strategies](../deployment/strategies.md) — canary and blue-green deployments that reduce DR risk
