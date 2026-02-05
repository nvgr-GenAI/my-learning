# Design a File Upload Service

Design a reliable file upload service that handles large files with resumable uploads, chunking, and progress tracking, similar to Google Drive or Dropbox upload functionality.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M users, 100K uploads/day, 10TB daily upload volume |
| **Key Challenges** | Large file handling, resumable uploads, deduplication, upload failures |
| **Core Concepts** | Chunking, multipart upload, presigned URLs, deduplication, retry logic |
| **Companies** | Dropbox, Google Drive, AWS S3, Microsoft OneDrive |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **File Upload** | Upload files up to 10GB | P0 (Must have) |
    | **Chunked Upload** | Split large files into chunks | P0 (Must have) |
    | **Resumable Upload** | Resume interrupted uploads | P0 (Must have) |
    | **Progress Tracking** | Show upload progress to user | P0 (Must have) |
    | **Deduplication** | Avoid storing duplicate files | P1 (Should have) |
    | **Metadata Storage** | Store file name, size, type, owner | P0 (Must have) |
    | **Upload Cancellation** | Cancel in-progress uploads | P1 (Should have) |

    **Explicitly Out of Scope:**

    - File versioning
    - File sharing and permissions
    - File preview/thumbnail generation
    - Virus scanning
    - Encryption at rest (simplified)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Reliability** | 99.9% | Uploads should not fail silently |
    | **Resumability** | 100% | Any upload can be resumed |
    | **Upload Speed** | Network-limited | Maximize bandwidth utilization |
    | **Storage Efficiency** | 30% savings | Deduplication reduces storage |
    | **Durability** | 99.999999999% (11 9s) | No data loss |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 10M users
    Upload rate: 1% of users upload per day = 100K uploads/day
    Average file size: 100 MB
    Large files (>1GB): 5% of uploads

    Upload requests per second:
    - 100K uploads/day / 86,400 sec = ~1.2 uploads/sec
    - Peak (3x): ~3.6 uploads/sec

    Chunk requests per second:
    - Average chunks per file (100MB / 5MB): 20 chunks
    - 1.2 uploads × 20 chunks = 24 chunk requests/sec
    - Peak: 72 chunk requests/sec
    ```

    ### Storage Estimates

    ```
    Daily upload volume:
    - 100K uploads × 100 MB average = 10 TB/day

    With deduplication (30% savings):
    - Actual storage: 10 TB × 0.7 = 7 TB/day

    Annual storage:
    - 7 TB × 365 = 2,555 TB = ~2.5 PB/year

    Metadata storage:
    - 100K uploads/day × 1 KB metadata = 100 MB/day
    - Annual: 36.5 GB metadata
    ```

    ### Bandwidth Estimates

    ```
    Upload bandwidth:
    - 1.2 uploads/sec × 100 MB = 120 MB/sec = 960 Mbps
    - Peak: 2.88 Gbps

    Chunk upload bandwidth (with parallelization):
    - 72 chunk requests/sec × 5 MB = 360 MB/sec = 2.88 Gbps
    ```

    ---

    ## Key Assumptions

    1. Files up to 10GB supported
    2. Chunk size: 5MB (configurable)
    3. S3 or similar object storage for file storage
    4. PostgreSQL for metadata
    5. Redis for tracking upload sessions

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Chunked uploads** - Split large files into manageable pieces
    2. **Resumable uploads** - Track progress and allow retry
    3. **Parallel uploads** - Upload multiple chunks simultaneously
    4. **Deduplication** - Content-based addressing to avoid duplicates
    5. **Presigned URLs** - Direct client-to-storage uploads

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client"
            Client[Web/Mobile Client]
            FileChunker[File Chunker]
            UploadManager[Upload Manager]
        end

        subgraph "Upload Service"
            API[API Server]
            UploadSession[Upload Session Manager]
            ChunkTracker[Chunk Upload Tracker]
            Dedup[Deduplication Service]
        end

        subgraph "Storage Layer"
            S3[(Object Storage<br/>S3/GCS)]
            MetaDB[(PostgreSQL<br/>Metadata)]
            Redis[(Redis<br/>Session State)]
        end

        subgraph "Background Jobs"
            Assembler[Chunk Assembler]
            HashCalc[Hash Calculator]
            Cleaner[Cleanup Worker]
        end

        Client --> FileChunker
        FileChunker --> UploadManager

        UploadManager --> API
        API --> UploadSession
        API --> ChunkTracker
        API --> Dedup

        UploadSession --> Redis
        ChunkTracker --> Redis
        Dedup --> MetaDB

        API -->|Presigned URL| UploadManager
        UploadManager -->|Direct upload| S3

        ChunkTracker --> Assembler
        Assembler --> S3
        Assembler --> MetaDB

        Dedup --> HashCalc
        HashCalc --> S3

        Cleaner --> S3
        Cleaner --> Redis

        style S3 fill:#fff4e1
        style MetaDB fill:#e1f5ff
        style Redis fill:#ffe1e1
    ```

    ---

    ## API Design

    ### 1. Initiate Upload Session

    **Request:**
    ```http
    POST /api/v1/uploads/initiate
    Content-Type: application/json

    {
      "file_name": "video.mp4",
      "file_size": 104857600,
      "file_hash": "sha256:abc123...",
      "content_type": "video/mp4",
      "chunk_size": 5242880
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "upload_id": "upload_xyz789",
      "file_exists": false,
      "total_chunks": 20,
      "chunk_size": 5242880,
      "expires_at": "2024-01-15T12:00:00Z"
    }
    ```

    **Response (Deduplication - File Exists):**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "upload_id": "upload_xyz789",
      "file_exists": true,
      "file_id": "file_abc456",
      "message": "File already exists, no upload needed"
    }
    ```

    ---

    ### 2. Get Presigned URL for Chunk

    **Request:**
    ```http
    POST /api/v1/uploads/{upload_id}/chunks/{chunk_number}/presigned-url
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "presigned_url": "https://s3.amazonaws.com/bucket/...",
      "chunk_number": 1,
      "expires_in": 3600,
      "upload_method": "PUT"
    }
    ```

    ---

    ### 3. Mark Chunk as Uploaded

    **Request:**
    ```http
    POST /api/v1/uploads/{upload_id}/chunks/{chunk_number}/complete
    Content-Type: application/json

    {
      "etag": "d41d8cd98f00b204e9800998ecf8427e",
      "chunk_hash": "sha256:def456..."
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "chunk_number": 1,
      "status": "completed",
      "uploaded_chunks": 1,
      "total_chunks": 20,
      "progress_percentage": 5.0
    }
    ```

    ---

    ### 4. Complete Upload

    **Request:**
    ```http
    POST /api/v1/uploads/{upload_id}/complete
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "file_id": "file_abc123",
      "file_name": "video.mp4",
      "file_size": 104857600,
      "file_url": "https://cdn.example.com/files/abc123",
      "status": "completed"
    }
    ```

    ---

    ### 5. Resume Upload

    **Request:**
    ```http
    GET /api/v1/uploads/{upload_id}/status
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "upload_id": "upload_xyz789",
      "status": "in_progress",
      "uploaded_chunks": [1, 2, 3, 5, 7],
      "pending_chunks": [4, 6, 8, 9, 10, ...],
      "total_chunks": 20,
      "progress_percentage": 25.0
    }
    ```

    ---

    ## Data Models

    ### Upload Session

    ```sql
    CREATE TABLE upload_sessions (
        upload_id VARCHAR(64) PRIMARY KEY,
        user_id BIGINT NOT NULL,
        file_name VARCHAR(255) NOT NULL,
        file_size BIGINT NOT NULL,
        file_hash VARCHAR(128),
        content_type VARCHAR(128),
        chunk_size INT NOT NULL,
        total_chunks INT NOT NULL,
        uploaded_chunks INT DEFAULT 0,
        status VARCHAR(32) NOT NULL, -- initiated, in_progress, completed, failed
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        expires_at TIMESTAMP NOT NULL,

        INDEX idx_user_status (user_id, status),
        INDEX idx_expires (expires_at)
    );
    ```

    ### Chunk Upload Tracking (Redis)

    ```
    Key: upload:{upload_id}:chunks
    Type: Bitmap
    Value: 1 bit per chunk (1 = uploaded, 0 = pending)

    Key: upload:{upload_id}:metadata
    Type: Hash
    Fields:
        - file_name
        - file_size
        - total_chunks
        - uploaded_chunks
        - status
        - expires_at
    ```

    ### File Metadata

    ```sql
    CREATE TABLE files (
        file_id VARCHAR(64) PRIMARY KEY,
        user_id BIGINT NOT NULL,
        file_name VARCHAR(255) NOT NULL,
        file_size BIGINT NOT NULL,
        file_hash VARCHAR(128) UNIQUE NOT NULL,
        content_type VARCHAR(128),
        storage_path VARCHAR(512) NOT NULL,
        upload_id VARCHAR(64),
        created_at TIMESTAMP DEFAULT NOW(),

        INDEX idx_user (user_id),
        INDEX idx_hash (file_hash)
    );
    ```

    ---

    ## Upload Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant Redis
        participant S3
        participant DB

        Client->>API: 1. Initiate upload (file metadata)
        API->>DB: Check if file_hash exists (dedup)

        alt File exists
            DB-->>API: File found
            API-->>Client: Return existing file_id
        else New file
            API->>Redis: Create upload session
            API->>DB: Store upload_session
            API-->>Client: Return upload_id, total_chunks

            loop For each chunk
                Client->>API: 2. Request presigned URL
                API-->>Client: Return presigned URL
                Client->>S3: 3. Upload chunk directly
                S3-->>Client: Upload success (ETag)
                Client->>API: 4. Mark chunk complete
                API->>Redis: Update chunk bitmap
            end

            Client->>API: 5. Complete upload
            API->>API: Verify all chunks uploaded
            API->>DB: Create file record
            API->>Redis: Delete upload session
            API-->>Client: Return file_id
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics

    ### 1. File Chunking Strategy

    ```python
    class FileChunker:
        def __init__(self, file_path: str, chunk_size: int = 5 * 1024 * 1024):
            self.file_path = file_path
            self.chunk_size = chunk_size
            self.file_size = os.path.getsize(file_path)
            self.total_chunks = math.ceil(self.file_size / chunk_size)

        def calculate_file_hash(self) -> str:
            """Calculate SHA-256 hash of entire file for deduplication"""
            sha256 = hashlib.sha256()
            with open(self.file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return f"sha256:{sha256.hexdigest()}"

        def get_chunk(self, chunk_number: int) -> bytes:
            """Read specific chunk from file"""
            offset = chunk_number * self.chunk_size
            with open(self.file_path, 'rb') as f:
                f.seek(offset)
                return f.read(self.chunk_size)

        def get_chunk_hash(self, chunk_number: int) -> str:
            """Calculate hash of specific chunk for verification"""
            chunk_data = self.get_chunk(chunk_number)
            return hashlib.sha256(chunk_data).hexdigest()
    ```

    ---

    ### 2. Upload Session Management

    ```python
    import redis
    from typing import Set

    class UploadSessionManager:
        def __init__(self, redis_client: redis.Redis):
            self.redis = redis_client
            self.session_ttl = 24 * 3600  # 24 hours

        def create_session(self, upload_id: str, total_chunks: int,
                          file_metadata: dict) -> dict:
            """Create new upload session"""
            # Store metadata
            self.redis.hset(
                f"upload:{upload_id}:metadata",
                mapping={
                    "file_name": file_metadata["file_name"],
                    "file_size": file_metadata["file_size"],
                    "total_chunks": total_chunks,
                    "uploaded_chunks": 0,
                    "status": "initiated"
                }
            )

            # Initialize chunk bitmap
            # All bits set to 0 initially (not uploaded)
            self.redis.setbit(f"upload:{upload_id}:chunks", total_chunks - 1, 0)

            # Set expiration
            self.redis.expire(f"upload:{upload_id}:metadata", self.session_ttl)
            self.redis.expire(f"upload:{upload_id}:chunks", self.session_ttl)

        def mark_chunk_uploaded(self, upload_id: str, chunk_number: int):
            """Mark specific chunk as uploaded"""
            # Set bit to 1
            self.redis.setbit(f"upload:{upload_id}:chunks", chunk_number, 1)

            # Increment uploaded_chunks counter
            self.redis.hincrby(f"upload:{upload_id}:metadata", "uploaded_chunks", 1)

        def get_uploaded_chunks(self, upload_id: str) -> Set[int]:
            """Get list of uploaded chunk numbers"""
            bitmap = self.redis.get(f"upload:{upload_id}:chunks")
            if not bitmap:
                return set()

            uploaded = set()
            for i, byte in enumerate(bitmap):
                for bit in range(8):
                    if byte & (1 << bit):
                        chunk_num = i * 8 + bit
                        uploaded.add(chunk_num)
            return uploaded

        def is_upload_complete(self, upload_id: str) -> bool:
            """Check if all chunks uploaded"""
            metadata = self.redis.hgetall(f"upload:{upload_id}:metadata")
            return int(metadata[b"uploaded_chunks"]) == int(metadata[b"total_chunks"])

        def get_progress(self, upload_id: str) -> dict:
            """Get upload progress"""
            metadata = self.redis.hgetall(f"upload:{upload_id}:metadata")
            uploaded = int(metadata[b"uploaded_chunks"])
            total = int(metadata[b"total_chunks"])

            return {
                "uploaded_chunks": uploaded,
                "total_chunks": total,
                "progress_percentage": (uploaded / total) * 100,
                "status": metadata[b"status"].decode()
            }
    ```

    ---

    ### 3. Deduplication

    **Content-based addressing:**
    - Calculate SHA-256 hash of entire file
    - Check if file with same hash exists
    - If exists, create new metadata entry pointing to existing file
    - Save storage space (30-50% for typical workloads)

    ```python
    class DeduplicationService:
        def check_duplicate(self, file_hash: str, user_id: int) -> Optional[str]:
            """Check if file already exists"""
            existing_file = db.query("""
                SELECT file_id, storage_path
                FROM files
                WHERE file_hash = ?
                LIMIT 1
            """, file_hash)

            if existing_file:
                # Create new metadata entry for this user
                file_id = self.create_file_reference(
                    user_id=user_id,
                    file_hash=file_hash,
                    storage_path=existing_file.storage_path
                )
                return file_id

            return None

        def create_file_reference(self, user_id: int, file_hash: str,
                                 storage_path: str) -> str:
            """Create metadata entry for deduplicated file"""
            file_id = generate_unique_id()
            db.execute("""
                INSERT INTO files (file_id, user_id, file_hash, storage_path)
                VALUES (?, ?, ?, ?)
            """, file_id, user_id, file_hash, storage_path)
            return file_id
    ```

    ---

    ### 4. Resumable Upload Implementation

    **Client-side resume logic:**

    ```python
    class ResumableUploader:
        def __init__(self, file_path: str, api_client):
            self.file_path = file_path
            self.api = api_client
            self.chunker = FileChunker(file_path)
            self.upload_id = None

        def upload(self):
            """Upload file with resume capability"""
            # 1. Initiate or resume upload
            if self.upload_id:
                status = self.api.get_upload_status(self.upload_id)
                uploaded_chunks = set(status["uploaded_chunks"])
            else:
                file_hash = self.chunker.calculate_file_hash()
                response = self.api.initiate_upload(
                    file_name=os.path.basename(self.file_path),
                    file_size=self.chunker.file_size,
                    file_hash=file_hash,
                    total_chunks=self.chunker.total_chunks
                )

                if response.get("file_exists"):
                    return response["file_id"]  # Deduplicated

                self.upload_id = response["upload_id"]
                uploaded_chunks = set()

            # 2. Upload pending chunks
            pending_chunks = set(range(self.chunker.total_chunks)) - uploaded_chunks

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for chunk_num in pending_chunks:
                    future = executor.submit(self.upload_chunk, chunk_num)
                    futures.append(future)

                # Wait for all chunks
                for future in futures:
                    future.result()

            # 3. Complete upload
            return self.api.complete_upload(self.upload_id)

        def upload_chunk(self, chunk_num: int, max_retries: int = 3):
            """Upload single chunk with retry logic"""
            chunk_data = self.chunker.get_chunk(chunk_num)
            chunk_hash = self.chunker.get_chunk_hash(chunk_num)

            for attempt in range(max_retries):
                try:
                    # Get presigned URL
                    url_response = self.api.get_presigned_url(
                        self.upload_id, chunk_num
                    )

                    # Upload directly to S3
                    response = requests.put(
                        url_response["presigned_url"],
                        data=chunk_data,
                        headers={"Content-Type": "application/octet-stream"}
                    )

                    etag = response.headers.get("ETag")

                    # Mark as complete
                    self.api.mark_chunk_complete(
                        self.upload_id, chunk_num, etag, chunk_hash
                    )

                    return

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## Scalability Improvements

    ### 1. Parallel Chunk Uploads

    **Optimize upload speed:**
    - Upload 4-8 chunks in parallel
    - Maximize bandwidth utilization
    - Trade-off: Client resource usage vs speed

    ```python
    # Client configuration
    PARALLEL_UPLOADS = 4  # Configurable based on network

    # Upload chunks in parallel
    with ThreadPoolExecutor(max_workers=PARALLEL_UPLOADS) as executor:
        futures = {executor.submit(upload_chunk, i): i
                  for i in pending_chunks}
    ```

    ---

    ### 2. Chunk Assembly (S3 Multipart Upload)

    **Use S3's native multipart upload:**

    ```python
    class S3ChunkAssembler:
        def assemble_file(self, upload_id: str, chunks: List[dict]):
            """Combine chunks into final file using S3 multipart"""
            # S3 multipart upload
            multipart_upload = s3_client.create_multipart_upload(
                Bucket=BUCKET,
                Key=f"files/{upload_id}"
            )

            parts = []
            for chunk in chunks:
                # Copy chunk to final file
                part = s3_client.upload_part_copy(
                    Bucket=BUCKET,
                    Key=f"files/{upload_id}",
                    CopySource=f"{BUCKET}/chunks/{chunk['key']}",
                    PartNumber=chunk['number'],
                    UploadId=multipart_upload['UploadId']
                )
                parts.append({
                    'PartNumber': chunk['number'],
                    'ETag': part['CopyPartResult']['ETag']
                })

            # Complete multipart upload
            s3_client.complete_multipart_upload(
                Bucket=BUCKET,
                Key=f"files/{upload_id}",
                UploadId=multipart_upload['UploadId'],
                MultipartUpload={'Parts': parts}
            )
    ```

    ---

    ### 3. Cleanup of Incomplete Uploads

    **Background worker to clean expired uploads:**

    ```python
    class UploadCleanupWorker:
        def clean_expired_uploads(self):
            """Remove expired upload sessions and chunks"""
            # Find expired uploads
            expired = db.query("""
                SELECT upload_id, status
                FROM upload_sessions
                WHERE expires_at < NOW()
                AND status IN ('initiated', 'in_progress')
            """)

            for upload in expired:
                # Delete chunks from S3
                s3_client.delete_objects(
                    Bucket=BUCKET,
                    Delete={
                        'Objects': [
                            {'Key': f"chunks/{upload.upload_id}/{i}"}
                            for i in range(upload.total_chunks)
                        ]
                    }
                )

                # Delete from Redis
                redis.delete(f"upload:{upload.upload_id}:chunks")
                redis.delete(f"upload:{upload.upload_id}:metadata")

                # Update database
                db.execute("""
                    UPDATE upload_sessions
                    SET status = 'expired'
                    WHERE upload_id = ?
                """, upload.upload_id)
    ```

    ---

    ### 4. Monitoring Metrics

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Upload Success Rate** | > 99% | < 95% |
    | **Average Upload Time** | < 2 min for 100MB | > 5 min |
    | **Chunk Upload Failure Rate** | < 1% | > 5% |
    | **Deduplication Rate** | 20-30% | N/A |
    | **Storage Efficiency** | 30% savings | < 20% |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Chunked uploads** - 5MB chunks for reliability and resumability
    2. **Presigned URLs** - Direct client-to-S3 uploads reduce server load
    3. **Redis bitmaps** - Efficient chunk tracking (1 bit per chunk)
    4. **Content-based deduplication** - SHA-256 hash to detect duplicates
    5. **S3 multipart upload** - Native cloud storage feature for assembly
    6. **Parallel uploads** - 4-8 concurrent chunks for speed

    ## Interview Tips

    ✅ **Discuss chunking strategy** - Why 5MB? Trade-offs of chunk size
    ✅ **Explain resumability** - How to track progress, handle failures
    ✅ **Cover deduplication** - Content-based addressing saves storage
    ✅ **Address failure cases** - Network errors, partial uploads, timeouts
    ✅ **Optimize upload speed** - Parallel chunks, direct S3 uploads

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle network failures?"** | Retry with exponential backoff, resume from last successful chunk |
    | **"What if chunk upload fails midway?"** | Retry failed chunk only, don't restart entire upload |
    | **"How to optimize for mobile networks?"** | Smaller chunks (1-2MB), adaptive chunk size based on bandwidth |
    | **"How to prevent abuse?"** | Rate limiting, quota per user, file size limits |
    | **"How to scale to millions of uploads?"** | Shard by user_id, use CDN for download, scale S3 storage |

    ## Real-World Examples

    - **Dropbox**: Chunked uploads with deduplication
    - **Google Drive**: Resumable uploads with progress tracking
    - **AWS S3**: Multipart upload API (reference implementation)
    - **YouTube**: Chunked video uploads with processing pipeline

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Dropbox, Google Drive, AWS, Microsoft

---

*This problem demonstrates file handling, resumable uploads, and cloud storage patterns essential for modern applications.*
