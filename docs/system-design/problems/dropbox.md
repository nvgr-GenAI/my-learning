# Design Dropbox (File Storage & Sync)

A cloud file storage and synchronization service that allows users to store files, sync across devices, and share with others.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 500M users, 2B files, 500 PB storage |
| **Key Challenges** | File chunking, conflict resolution, bandwidth optimization, version history |
| **Core Concepts** | Delta sync, chunking, deduplication, metadata management |
| **Companies** | Dropbox, Google Drive, OneDrive, Box, iCloud |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Upload Files** | Users can upload files to cloud | P0 (Must have) |
    | **Download Files** | Download files to any device | P0 (Must have) |
    | **Sync Across Devices** | Automatic file synchronization | P0 (Must have) |
    | **Share Files** | Share files/folders with other users | P0 (Must have) |
    | **Conflict Resolution** | Handle simultaneous edits | P0 (Must have) |
    | **Version History** | Restore previous file versions | P1 (Should have) |
    | **Offline Access** | Access files without internet | P1 (Should have) |
    | **Search** | Search files by name/content | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Real-time collaborative editing (Google Docs)
    - File preview generation
    - Mobile photo backup
    - Team admin features

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Reliability** | 99.99% uptime | Files must be accessible |
    | **Durability** | 99.999999999% (11 nines) | No data loss acceptable |
    | **Consistency** | Eventual consistency | Brief sync delays acceptable |
    | **Bandwidth Efficiency** | Delta sync only | Don't re-upload entire files |
    | **Latency** | < 2s for file metadata operations | Fast file browsing |
    | **Storage Optimization** | Deduplication | Save storage costs |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total Users: 500M
    Daily Active Users (DAU): 100M (20% engagement)

    File operations:
    - Uploads: 50M files/day (500 files/sec)
    - Downloads: 100M files/day (1,157 files/sec)
    - Syncs: 200M sync operations/day (2,315 syncs/sec)

    Average file size: 1 MB
    Daily upload volume: 50M × 1 MB = 50 TB/day
    Daily download volume: 100 TB/day
    ```

    ### Storage Estimates

    ```
    Total files: 2B files
    Average file size: 1 MB
    Total storage: 2B × 1 MB = 2 PB

    With deduplication (30% reduction):
    - Actual storage: 2 PB × 0.7 = 1.4 PB

    Version history (3 versions per file avg):
    - Additional: 2 PB × 3 = 6 PB
    - With compression: 6 PB × 0.5 = 3 PB

    Metadata:
    - Per file: 1 KB (name, path, hash, timestamps)
    - Total: 2B × 1 KB = 2 TB

    Total: 1.4 PB (current) + 3 PB (versions) + 2 TB (metadata) ≈ 4.4 PB
    ```

    ### Bandwidth Estimates

    ```
    Upload: 50 TB/day = 579 MB/sec ≈ 4.6 Gbps
    Download: 100 TB/day = 1.16 GB/sec ≈ 9.3 Gbps

    With delta sync (only 10% of file typically changes):
    - Effective upload: 0.46 Gbps
    - Effective download: 0.93 Gbps
    ```

    ---

    ## Key Assumptions

    1. Average file size: 1 MB
    2. 20% of users active daily
    3. Each user has 3 devices on average
    4. File deduplication saves 30%
    5. Delta sync reduces bandwidth by 90%
    6. Keep 3 versions per file on average

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Chunking** - Split files into 4 MB chunks for efficient sync
    2. **Deduplication** - Store identical chunks once
    3. **Delta sync** - Only upload changed chunks
    4. **Metadata-first** - Fast file browsing with lazy loading
    5. **Client-side intelligence** - Reduce server load

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Desktop[Desktop Client<br/>File Watcher]
            Mobile[Mobile App]
            Web[Web Client]
        end

        subgraph "Load Balancing"
            LB[Load Balancer]
        end

        subgraph "API Layer"
            Metadata_API[Metadata Service<br/>File operations]
            Sync_API[Sync Service<br/>Upload/download]
            Share_API[Sharing Service<br/>Permissions]
        end

        subgraph "Processing"
            Chunker[Chunking Service<br/>Split files]
            Dedup[Deduplication<br/>Hash matching]
            Version[Version Manager<br/>History tracking]
        end

        subgraph "Notification"
            Notify[Notification Service<br/>WebSocket/Long polling]
        end

        subgraph "Caching"
            Redis_Meta[Redis<br/>Metadata cache]
            Redis_Chunk[Redis<br/>Chunk metadata]
        end

        subgraph "Storage"
            Meta_DB[(Metadata DB<br/>MySQL<br/>Sharded)]
            Chunk_DB[(Chunk Index<br/>Cassandra)]
            S3[Object Storage<br/>S3<br/>File chunks]
        end

        subgraph "Message Queue"
            Queue[Message Queue<br/>Kafka]
        end

        Desktop --> LB
        Mobile --> LB
        Web --> LB

        LB --> Metadata_API
        LB --> Sync_API
        LB --> Share_API

        Sync_API --> Chunker
        Chunker --> Dedup
        Dedup --> S3
        Dedup --> Chunk_DB

        Metadata_API --> Meta_DB
        Metadata_API --> Redis_Meta

        Sync_API --> Queue
        Queue --> Notify
        Notify --> Desktop
        Notify --> Mobile

        Share_API --> Meta_DB

        Chunker --> Redis_Chunk
        Version --> Meta_DB

        style LB fill:#e1f5ff
        style Redis_Meta fill:#fff4e1
        style Redis_Chunk fill:#fff4e1
        style Meta_DB fill:#ffe1e1
        style Chunk_DB fill:#ffe1e1
        style S3 fill:#f3e5f5
        style Queue fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Chunking** | Enable delta sync (10x bandwidth savings), parallel upload/download | Whole file transfer (wasteful for large files) |
    | **S3** | Unlimited storage, 11 nines durability, cost-effective | Database BLOBs (expensive, doesn't scale) |
    | **Cassandra (Chunk Index)** | Fast chunk lookups, handles high write volume | MySQL (can't handle write volume), Redis (too expensive for large index) |
    | **MySQL (Metadata)** | ACID guarantees for file operations, complex queries | NoSQL (weak consistency, complex queries difficult) |
    | **WebSocket** | Real-time sync notifications | Polling (wasteful, higher latency) |
    | **Deduplication** | 30% storage savings, faster uploads | No dedup (higher costs, slower syncs) |

    **Key Trade-off:** We chose **eventual consistency** for file syncs but **strong consistency** for metadata operations (renames, deletes must be atomic).

    ---

    ## API Design

    ### 1. Upload File

    **Request:**
    ```http
    POST /api/v1/files/upload
    Content-Type: multipart/form-data
    Authorization: Bearer <token>

    file: <binary>
    path: "/Documents/report.pdf"
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created

    {
      "file_id": "file_abc123",
      "path": "/Documents/report.pdf",
      "size": 1048576,
      "hash": "sha256:abc123...",
      "chunks": 1,
      "version": 1,
      "modified_at": "2026-01-29T10:30:00Z"
    }
    ```

    ---

    ### 2. Sync Changes (Internal)

    **Request:**
    ```http
    POST /api/v1/sync/delta
    Authorization: Bearer <token>

    {
      "device_id": "device_xyz",
      "last_sync_timestamp": 1643712000,
      "changes": [
        {
          "file_id": "file_abc123",
          "action": "modified",
          "chunks_changed": [0, 2, 5],
          "new_hash": "sha256:def456..."
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "sync_id": "sync_123",
      "chunks_to_upload": [
        {
          "chunk_id": 0,
          "upload_url": "https://s3.../chunk_0?..."
        },
        {
          "chunk_id": 2,
          "upload_url": "https://s3.../chunk_2?..."
        }
      ],
      "conflicts": []
    }
    ```

    ---

    ## Database Schema

    ### Files Metadata (MySQL)

    ```sql
    -- Files table
    CREATE TABLE files (
        file_id BIGINT PRIMARY KEY AUTO_INCREMENT,
        user_id BIGINT NOT NULL,
        path VARCHAR(1024) NOT NULL,
        name VARCHAR(255) NOT NULL,
        size BIGINT NOT NULL,
        hash VARCHAR(64) NOT NULL,  -- SHA-256
        version INT DEFAULT 1,
        is_deleted BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_user_path (user_id, path),
        INDEX idx_hash (hash),
        UNIQUE KEY uniq_user_path (user_id, path, is_deleted)
    ) PARTITION BY HASH(user_id);

    -- File chunks mapping
    CREATE TABLE file_chunks (
        file_id BIGINT,
        chunk_index INT,
        chunk_hash VARCHAR(64),
        chunk_size INT,
        PRIMARY KEY (file_id, chunk_index),
        INDEX idx_chunk_hash (chunk_hash)
    );

    -- Version history
    CREATE TABLE file_versions (
        version_id BIGINT PRIMARY KEY AUTO_INCREMENT,
        file_id BIGINT,
        version INT,
        hash VARCHAR(64),
        size BIGINT,
        created_at TIMESTAMP,
        INDEX idx_file (file_id, version)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### File Upload Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant Chunker
        participant Dedup
        participant S3
        participant Meta_DB
        participant Notify

        Client->>API: Upload file
        API->>Chunker: Split into 4MB chunks

        loop For each chunk
            Chunker->>Chunker: Calculate SHA-256 hash
            Chunker->>Dedup: Check if chunk exists

            alt Chunk exists (dedup hit)
                Dedup-->>Chunker: Chunk ID (skip upload)
            else New chunk
                Chunker->>S3: Upload chunk
                S3-->>Chunker: Chunk uploaded
            end
        end

        Chunker->>Meta_DB: INSERT file metadata
        Chunker->>Meta_DB: INSERT chunk mappings
        Meta_DB-->>API: Success

        API->>Notify: File uploaded event
        Notify->>Client: Sync notification (other devices)

        API-->>Client: 201 Created
    ```

=== "🔍 Step 3: Deep Dive"

    === "🧩 File Chunking & Delta Sync"

        ## The Challenge

        **Problem:** 100 MB file modified (1 MB changed) → re-upload 100 MB? Wasteful!

        **Solution:** Chunking + delta sync

        ---

        ## Chunking Strategy

        **Fixed-size chunks (4 MB):**

        ```python
        import hashlib

        class FileChunker:
            """Split files into fixed-size chunks for efficient sync"""

            CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB

            def chunk_file(self, file_path: str) -> List[dict]:
                """
                Split file into chunks with hashes

                Returns:
                    List of {index, hash, size, data}
                """
                chunks = []
                chunk_index = 0

                with open(file_path, 'rb') as f:
                    while True:
                        chunk_data = f.read(self.CHUNK_SIZE)
                        if not chunk_data:
                            break

                        # Calculate SHA-256 hash
                        chunk_hash = hashlib.sha256(chunk_data).hexdigest()

                        chunks.append({
                            'index': chunk_index,
                            'hash': chunk_hash,
                            'size': len(chunk_data),
                            'data': chunk_data
                        })

                        chunk_index += 1

                return chunks

            def detect_changes(
                self,
                old_chunks: List[dict],
                new_chunks: List[dict]
            ) -> List[int]:
                """
                Detect which chunks changed

                Returns:
                    List of changed chunk indices
                """
                changed = []

                for i in range(max(len(old_chunks), len(new_chunks))):
                    if i >= len(old_chunks) or i >= len(new_chunks):
                        # Chunk added or removed
                        changed.append(i)
                    elif old_chunks[i]['hash'] != new_chunks[i]['hash']:
                        # Chunk modified
                        changed.append(i)

                return changed
        ```

        **Delta sync process:**

        1. Client modifies file locally
        2. Client chunks file, compares hashes with previous version
        3. Only upload changed chunks
        4. Server reconstructs file with new chunks + existing chunks

        **Bandwidth savings:**

        ```
        100 MB file, 1 MB changed:
        - Without delta sync: Upload 100 MB
        - With delta sync: Upload 4 MB (1 chunk)
        - Savings: 96 MB (96%)
        ```

    === "🔍 Deduplication"

        ## The Challenge

        **Problem:** Multiple users upload same file → store N copies? Wasteful!

        **Solution:** Content-based deduplication

        ---

        ## Deduplication Implementation

        ```python
        class ChunkDeduplicator:
            """Deduplicate file chunks across all users"""

            def __init__(self, db, s3):
                self.db = db
                self.s3 = s3

            def store_chunk(self, chunk_hash: str, chunk_data: bytes) -> str:
                """
                Store chunk with deduplication

                Args:
                    chunk_hash: SHA-256 hash of chunk
                    chunk_data: Chunk bytes

                Returns:
                    chunk_id: Storage location
                """
                # Check if chunk already exists
                existing = self.db.query(
                    "SELECT chunk_id, ref_count FROM chunks WHERE chunk_hash = %s",
                    (chunk_hash,)
                )

                if existing:
                    # Chunk exists - increment reference count
                    chunk_id = existing['chunk_id']
                    self.db.execute(
                        "UPDATE chunks SET ref_count = ref_count + 1 WHERE chunk_id = %s",
                        (chunk_id,)
                    )
                    logger.info(f"Dedup hit: Chunk {chunk_hash[:8]}... already exists")
                    return chunk_id

                # New chunk - store in S3
                chunk_id = str(uuid.uuid4())
                s3_key = f"chunks/{chunk_hash[:2]}/{chunk_hash}"

                self.s3.put_object(
                    Bucket='dropbox-chunks',
                    Key=s3_key,
                    Body=chunk_data
                )

                # Record in database
                self.db.execute("""
                    INSERT INTO chunks (chunk_id, chunk_hash, s3_key, size, ref_count)
                    VALUES (%s, %s, %s, %s, 1)
                """, (chunk_id, chunk_hash, s3_key, len(chunk_data)))

                return chunk_id

            def delete_chunk_reference(self, chunk_hash: str):
                """
                Decrement reference count, delete if no references

                Called when file deleted
                """
                result = self.db.execute(
                    "UPDATE chunks SET ref_count = ref_count - 1 WHERE chunk_hash = %s RETURNING ref_count",
                    (chunk_hash,)
                )

                if result['ref_count'] == 0:
                    # No more references - delete from S3
                    chunk = self.db.query(
                        "SELECT s3_key FROM chunks WHERE chunk_hash = %s",
                        (chunk_hash,)
                    )

                    self.s3.delete_object(
                        Bucket='dropbox-chunks',
                        Key=chunk['s3_key']
                    )

                    self.db.execute(
                        "DELETE FROM chunks WHERE chunk_hash = %s",
                        (chunk_hash,)
                    )

                    logger.info(f"Deleted chunk {chunk_hash[:8]}... (ref_count = 0)")
        ```

        **Storage savings:**

        - Popular files (company logos, common documents): High dedup rate
        - Personal photos: Low dedup rate
        - **Average: 30% storage savings**

    === "⚔️ Conflict Resolution"

        ## The Challenge

        **Problem:** User edits file on two devices offline, both sync → conflict!

        **Strategies:**

        1. **Last Write Wins (LWW):** Latest timestamp wins (data loss risk)
        2. **Version Branching:** Keep both versions, let user choose
        3. **Operational Transformation:** Merge changes (complex)

        **Dropbox approach: Version branching**

        ---

        ## Conflict Detection

        ```python
        class ConflictResolver:
            """Detect and resolve file conflicts"""

            def detect_conflict(
                self,
                file_id: str,
                device_id: str,
                local_version: int,
                local_hash: str
            ) -> bool:
                """
                Detect if file has conflict

                Returns:
                    True if conflict detected
                """
                # Get server's latest version
                server_file = db.query(
                    "SELECT version, hash FROM files WHERE file_id = %s",
                    (file_id,)
                )

                # No conflict if versions match
                if server_file['version'] == local_version:
                    return False

                # Conflict if hashes differ (different changes)
                if server_file['hash'] != local_hash:
                    return True

                return False

            def resolve_conflict(
                self,
                file_id: str,
                device_id: str,
                local_data: bytes
            ) -> dict:
                """
                Resolve conflict by creating conflict copy

                Returns:
                    {winner_file_id, conflict_file_id}
                """
                original_file = db.query(
                    "SELECT * FROM files WHERE file_id = %s",
                    (file_id,)
                )

                # Create conflict copy
                conflict_path = self._generate_conflict_path(
                    original_file['path'],
                    device_id
                )

                conflict_file_id = self.create_file(
                    user_id=original_file['user_id'],
                    path=conflict_path,
                    data=local_data
                )

                logger.info(f"Created conflict copy: {conflict_path}")

                return {
                    'winner_file_id': file_id,  # Server version wins
                    'conflict_file_id': conflict_file_id,
                    'conflict_path': conflict_path
                }

            def _generate_conflict_path(self, original_path: str, device_id: str) -> str:
                """
                Generate conflict file name

                Example: report.pdf → report (conflicted copy from Device-A).pdf
                """
                path_parts = original_path.rsplit('.', 1)
                if len(path_parts) == 2:
                    name, ext = path_parts
                    return f"{name} (conflicted copy from {device_id}).{ext}"
                else:
                    return f"{original_path} (conflicted copy from {device_id})"
        ```

        **User experience:**

        - Both versions preserved
        - User sees: `report.pdf` and `report (conflicted copy from Laptop).pdf`
        - User manually resolves by keeping desired version

    === "📜 Version History"

        ## The Challenge

        **Problem:** User accidentally deletes file or wants previous version.

        **Solution:** Keep version history (configurable retention)

        ---

        ## Version Management

        ```python
        class VersionManager:
            """Manage file version history"""

            MAX_VERSIONS = 30  # Keep last 30 versions

            def create_version(self, file_id: str, new_hash: str):
                """
                Create new version when file modified

                Args:
                    file_id: File being versioned
                    new_hash: Hash of new version
                """
                # Get current version number
                current = db.query(
                    "SELECT version FROM files WHERE file_id = %s",
                    (file_id,)
                )

                new_version = current['version'] + 1

                # Store old version in history
                db.execute("""
                    INSERT INTO file_versions (file_id, version, hash, size, created_at)
                    SELECT file_id, version, hash, size, modified_at
                    FROM files
                    WHERE file_id = %s
                """, (file_id,))

                # Update file to new version
                db.execute("""
                    UPDATE files
                    SET version = %s, hash = %s, modified_at = NOW()
                    WHERE file_id = %s
                """, (new_version, new_hash, file_id))

                # Cleanup old versions (keep only last N)
                self._cleanup_old_versions(file_id)

            def _cleanup_old_versions(self, file_id: str):
                """Delete versions beyond retention limit"""
                db.execute("""
                    DELETE FROM file_versions
                    WHERE file_id = %s
                    AND version NOT IN (
                        SELECT version FROM file_versions
                        WHERE file_id = %s
                        ORDER BY version DESC
                        LIMIT %s
                    )
                """, (file_id, file_id, self.MAX_VERSIONS))

            def restore_version(self, file_id: str, version: int) -> str:
                """
                Restore file to previous version

                Returns:
                    new_file_id: ID of restored file
                """
                # Get version data
                version_data = db.query("""
                    SELECT hash, size FROM file_versions
                    WHERE file_id = %s AND version = %s
                """, (file_id, version))

                # Create new current version from old version
                # This preserves history (don't actually rewind)
                self.create_version(file_id, version_data['hash'])

                return file_id
        ```

        **Storage optimization:**

        - Keep full versions for last 30 versions
        - Archive older versions (move to Glacier)
        - Delta-encode versions (only store diffs)

=== "⚡ Step 4: Scale & Optimize"

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Upload bandwidth** | ✅ Yes | Delta sync (10x reduction), parallel chunk upload |
    | **Storage** | ✅ Yes | Deduplication (30% savings), compression, tiered storage |
    | **Metadata queries** | 🟡 Moderate | Redis cache, MySQL sharding by user_id |
    | **Chunk lookups** | 🟡 Moderate | Cassandra (fast), Bloom filters for dedup |

    ---

    ## Cost Optimization

    **Monthly cost at 500M users:**

    | Component | Cost |
    |-----------|------|
    | **S3 storage** | $101,200 (4.4 PB × $0.023/GB) |
    | **EC2 (API)** | $86,400 (400 servers) |
    | **MySQL** | $43,200 (20 shards) |
    | **Cassandra** | $64,800 (50 nodes) |
    | **Redis** | $21,600 (100 nodes) |
    | **Bandwidth** | $27,000 (300 TB egress) |
    | **Total** | **$344,200/month** |

    **Revenue:** Dropbox charges $10/month per paying user. With 10% paid users: 50M × $10 = $500M/month. Infrastructure is 0.07% of revenue.

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **File chunking (4 MB)** - Enable delta sync, parallel transfers
    2. **Content deduplication** - 30% storage savings
    3. **Delta sync** - 90% bandwidth reduction
    4. **Version history** - Keep 30 versions per file
    5. **Conflict resolution** - Version branching (both versions preserved)
    6. **MySQL for metadata** - ACID guarantees, sharded by user_id

    ---

    ## Interview Tips

    ✅ **Start with chunking** - Core to efficient sync

    ✅ **Discuss deduplication** - Storage optimization

    ✅ **Conflict resolution** - Show understanding of distributed systems

    ✅ **Delta sync** - Bandwidth optimization critical

    ✅ **Metadata vs content** - Separate storage strategies

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle large files (100 GB)?"** | Chunking enables parallel upload, resume on failure, delta sync |
    | **"What if S3 fails?"** | Multi-region replication, failover to backup region |
    | **"How to share folder with 1000 users?"** | Shared folder metadata, permission inheritance, async notification |
    | **"How to implement offline editing?"** | Local cache, queue operations, sync when online, conflict resolution |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Dropbox, Google (Drive), Microsoft (OneDrive), Box

---

*Master this problem and you'll be ready for: Google Drive, OneDrive, Box, iCloud*
