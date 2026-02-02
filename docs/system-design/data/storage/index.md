# Data Storage

**Choose the right storage for your data** | 💾 Object | 📦 Block | 📁 File | 🏊 Lake

---

## Overview

Storage systems are foundational to system design. Different storage types serve different use cases - from serving web assets to storing massive data sets for analytics.

**Key Question:** What are you storing and how will you access it?

---

## Storage Types Comparison

| Storage Type | Access Method | Use Case | Performance | Scale | Cost |
|--------------|---------------|----------|-------------|-------|------|
| **Object Storage** | HTTP API | Media, backups, static files | Medium | Unlimited | $ |
| **Block Storage** | Volume attach | Databases, VMs | Very High | TBs | $$$ |
| **File Storage** | Network share (NFS/SMB) | Shared files, home directories | High | TBs-PBs | $$ |
| **Data Lake** | Query engines | Analytics, big data | Low-Medium | Unlimited | $ |

---

## Object Storage

=== "What is it?"
    **Flat namespace for unstructured data accessed via HTTP**

    ```
    Structure:
    bucket/
    ├── images/
    │   ├── photo1.jpg (Object)
    │   └── photo2.png (Object)
    ├── videos/
    │   └── clip.mp4 (Object)
    └── documents/
        └── report.pdf (Object)

    Each object = Data + Metadata + Unique ID
    ```

    **Characteristics:**
    - **Flat namespace:** No true directories (simulated with prefixes)
    - **HTTP access:** GET, PUT, DELETE via REST API
    - **Metadata:** Custom key-value pairs per object
    - **Versioning:** Keep multiple versions of same object
    - **Immutable:** Objects are replaced, not modified

=== "Use Cases"
    **When to use Object Storage:**

    | Use Case | Why Object Storage? | Example |
    |----------|-------------------|---------|
    | **Static Website Hosting** | Cheap, scalable, CDN-friendly | Host React app on S3 |
    | **Media Storage** | Store images, videos at scale | Netflix stores videos |
    | **Backups & Archives** | Unlimited storage, durability | Database backups to S3 |
    | **Data Lakes** | Store raw data for analytics | Store logs, events for analysis |
    | **Application Assets** | Store user uploads | Profile pictures, documents |

    **Don't use for:**
    - ❌ Frequently updated files (high latency)
    - ❌ Database storage (not ACID compliant)
    - ❌ Applications needing POSIX filesystem
    - ❌ Low-latency requirements (< 10ms)

=== "Providers"
    **Major Object Storage Providers:**

    **AWS S3 (Simple Storage Service):**
    ```javascript
    // Upload object
    await s3.putObject({
        Bucket: 'my-bucket',
        Key: 'images/photo.jpg',
        Body: fileBuffer,
        ContentType: 'image/jpeg',
        Metadata: {
            'user-id': '12345',
            'upload-date': new Date().toISOString()
        }
    });

    // Retrieve object
    const object = await s3.getObject({
        Bucket: 'my-bucket',
        Key: 'images/photo.jpg'
    });
    ```

    **Pricing (AWS S3 Standard):**
    - Storage: $0.023 per GB/month
    - GET requests: $0.0004 per 1000 requests
    - PUT requests: $0.005 per 1000 requests

    **Google Cloud Storage:**
    ```python
    from google.cloud import storage

    # Upload blob
    client = storage.Client()
    bucket = client.bucket('my-bucket')
    blob = bucket.blob('images/photo.jpg')
    blob.upload_from_filename('local-photo.jpg')

    # Add metadata
    blob.metadata = {'user-id': '12345'}
    blob.patch()
    ```

    **Azure Blob Storage:**
    ```csharp
    // Upload blob
    BlobClient blobClient = new BlobClient(
        connectionString, "container", "photo.jpg");
    await blobClient.UploadAsync(stream);

    // Set metadata
    var metadata = new Dictionary<string, string> {
        { "user-id", "12345" }
    };
    await blobClient.SetMetadataAsync(metadata);
    ```

=== "Storage Classes"
    **Tiered storage for cost optimization:**

    | Class | Retrieval Time | Cost/GB/month | Use Case |
    |-------|---------------|---------------|----------|
    | **Standard** | Immediate | $0.023 | Frequently accessed |
    | **Infrequent Access** | Immediate | $0.0125 | Accessed < 1x/month |
    | **Glacier** | Minutes-hours | $0.004 | Archives, backups |
    | **Deep Archive** | 12 hours | $0.00099 | Long-term archives |

    **Lifecycle Policies:**
    ```json
    {
        "Rules": [{
            "Id": "Archive old logs",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "logs/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ]
        }]
    }
    ```

    **Cost Savings Example:**
    ```
    100GB of old logs moved to Glacier:
    Standard: 100GB × $0.023 = $2.30/month
    Glacier:  100GB × $0.004 = $0.40/month
    Savings:  $1.90/month (82% reduction)
    ```

=== "Best Practices"
    **Performance Optimization:**

    1. **Use CDN for frequently accessed objects**
       ```
       User → CloudFront (CDN) → S3
       Benefits: Lower latency, reduced S3 costs
       ```

    2. **Multipart uploads for large files**
       ```javascript
       // Files > 100MB should use multipart upload
       const upload = new AWS.S3.ManagedUpload({
           params: {
               Bucket: 'my-bucket',
               Key: 'large-video.mp4',
               Body: largeFileStream
           },
           partSize: 10 * 1024 * 1024, // 10MB parts
           queueSize: 4 // 4 concurrent uploads
       });
       ```

    3. **Optimize key naming for performance**
       ```
       ❌ Bad:  Sequential keys (creates hot partitions)
       user-1/photo.jpg
       user-2/photo.jpg
       user-3/photo.jpg

       ✅ Good: Random prefix (distributes load)
       a3b8/user-1/photo.jpg
       f7c2/user-2/photo.jpg
       9d4e/user-3/photo.jpg
       ```

    4. **Set proper caching headers**
       ```javascript
       await s3.putObject({
           Bucket: 'my-bucket',
           Key: 'images/logo.png',
           Body: buffer,
           CacheControl: 'public, max-age=31536000', // 1 year
           ContentType: 'image/png'
       });
       ```

---

## Decision Framework

=== "Choose Storage Type"
    ```
    Start here:

    ┌─ Need filesystem semantics?
    │
    ├─ Yes ─┬─ Multiple servers need access?
    │       │
    │       ├─ Yes → File Storage (NFS)
    │       │         Examples: EFS, Azure Files
    │       │
    │       └─ No → Block Storage (attached disk)
    │                 Examples: EBS, Persistent Disk
    │
    └─ No ──┬─ Need HTTP API access?
            │
            ├─ Yes ─┬─ Frequently accessed?
            │       │
            │       ├─ Yes → Object Storage (Standard)
            │       │         Examples: S3 Standard, GCS
            │       │
            │       └─ No → Object Storage (Archive)
            │                 Examples: S3 Glacier, Archive
            │
            └─ No → Data Lake
                      Examples: S3 + Athena, BigQuery
    ```

=== "Cost Comparison"
    **Cost per GB/month (approximate):**

    ```
    Most Expensive:
    ↑
    │ Block (SSD):        $0.10 - $0.125
    │ File Storage:       $0.30 (EFS Standard)
    │ Object (Standard):  $0.023 (S3)
    │ Object (Archive):   $0.004 (S3 Glacier)
    ↓
    Least Expensive:
    ```

    **Example: 1TB of photos**
    ```
    Block Storage:  $100-125/month
    File Storage:   $300/month
    Object Storage: $23/month ← Winner!
    Archive:        $4/month (if rarely accessed)
    ```

=== "Performance Comparison"
    **Latency:**

    | Storage | Read Latency | Write Latency | Throughput |
    |---------|-------------|---------------|------------|
    | **Block (SSD)** | < 1ms | < 1ms | 1000 MB/s |
    | **File (NFS)** | 1-10ms | 1-10ms | 100 MB/s |
    | **Object Storage** | 10-100ms | 10-100ms | 100 MB/s |
    | **Data Lake** | Seconds | Seconds | Varies |

---

## Interview Talking Points

**Q: When would you choose object storage over block storage?**

✅ **Strong Answer:**
> "I'd choose object storage for static assets like images, videos, and user uploads because it's significantly cheaper ($0.023/GB vs $0.10/GB) and scales to unlimited size. For example, storing user profile pictures on S3 instead of EBS saves 80% on storage costs. However, I'd use block storage for databases because they need low-latency random I/O (< 1ms) and filesystem semantics that object storage doesn't provide. The 10-100ms latency of S3 API calls would be too slow for database transactions."

---

## Related Topics

- [Databases](../databases/index.md) - Structured data storage
- [Caching Strategies](../caching/index.md) - In-memory storage
- [CDN](../../networking/cdn.md) - Content delivery
- [Scalability Patterns](../../scalability/patterns.md) - Scale storage systems

---

**Choose storage based on access patterns, not just data size! 💾**
