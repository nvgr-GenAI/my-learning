# Design an Image Hosting Service

Design an image hosting and delivery service similar to Imgur or Pinterest that handles image uploads, processing, CDN delivery, and multiple size variants (thumbnails, previews, full-size).

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 50M users, 10M images/day, 5PB total storage |
| **Key Challenges** | Image processing, CDN delivery, multiple formats, thumbnail generation |
| **Core Concepts** | CDN, image transformation, lazy loading, responsive images, metadata extraction |
| **Companies** | Instagram, Pinterest, Imgur, Flickr, Cloudinary |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Image Upload** | Upload images up to 20MB | P0 (Must have) |
    | **Image Storage** | Store original images | P0 (Must have) |
    | **Thumbnail Generation** | Create multiple size variants | P0 (Must have) |
    | **Image Delivery** | Serve images via CDN | P0 (Must have) |
    | **Format Conversion** | Convert to WebP, AVIF | P1 (Should have) |
    | **Metadata Extraction** | Extract EXIF, dimensions | P1 (Should have) |
    | **Direct Links** | Public URLs for sharing | P0 (Must have) |
    | **Image Deletion** | Remove uploaded images | P1 (Should have) |

    **Explicitly Out of Scope:**

    - Image editing/filters
    - Social features (likes, comments)
    - Albums/collections
    - Image search
    - Video hosting

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% | Users expect reliable image access |
    | **Latency** | < 200ms (CDN) | Fast image loading for good UX |
    | **Upload Speed** | < 5s for 5MB | Quick upload experience |
    | **Durability** | 99.999999999% | No image loss |
    | **Cost Efficiency** | Optimize storage | Billions of images = high cost |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 50M users
    Upload rate: 20% upload images = 10M uploads/day
    View rate: 100 image views per user/day = 5B views/day

    Uploads per second:
    - 10M uploads/day / 86,400 = ~116 uploads/sec
    - Peak (3x): 348 uploads/sec

    Views per second:
    - 5B views/day / 86,400 = ~57,870 views/sec
    - Peak (3x): 173,610 views/sec
    ```

    ### Storage Estimates

    ```
    Average image size: 2 MB
    Thumbnail sizes generated:
    - Small (150x150): 10 KB
    - Medium (500x500): 50 KB
    - Large (1200x1200): 200 KB
    - Original: 2 MB

    Per image storage:
    - Original: 2 MB
    - Variants: 260 KB
    - Total: ~2.26 MB per image

    Daily storage:
    - 10M images × 2.26 MB = 22.6 TB/day

    Annual storage:
    - 22.6 TB × 365 = 8.25 PB/year

    With compression (save 40%):
    - Actual: 4.95 PB/year
    ```

    ### Bandwidth Estimates

    ```
    Upload bandwidth:
    - 116 uploads/sec × 2 MB = 232 MB/sec = 1.86 Gbps
    - Peak: 5.58 Gbps

    Download bandwidth (CDN):
    - 57,870 views/sec × 200 KB (average) = 11.2 GB/sec = 89.6 Gbps
    - Peak: 268.8 Gbps (handled by CDN)
    ```

    ---

    ## Key Assumptions

    1. Image formats: JPEG, PNG, GIF, WebP
    2. Max file size: 20MB
    3. CDN for content delivery (CloudFront, Cloudflare)
    4. S3 for storage
    5. Async processing for thumbnails

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **CDN-first delivery** - Serve images from edge locations
    2. **Lazy processing** - Generate thumbnails on-demand or async
    3. **Format optimization** - WebP/AVIF for modern browsers
    4. **Metadata separation** - Store image metadata in database
    5. **URL-based transformations** - Specify size in URL

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Web[Web Browser]
            Mobile[Mobile App]
        end

        subgraph "CDN Layer"
            CDN[CDN<br/>CloudFront/Cloudflare]
            EdgeCache[Edge Cache]
        end

        subgraph "Upload Service"
            UploadAPI[Upload API]
            ImageValidator[Image Validator]
            StorageWriter[Storage Writer]
        end

        subgraph "Processing Pipeline"
            Queue[Job Queue<br/>SQS/RabbitMQ]
            ThumbnailWorker[Thumbnail Worker]
            MetadataExtractor[Metadata Extractor]
        end

        subgraph "Delivery Service"
            ImageAPI[Image API]
            TransformService[Transform Service]
            URLGenerator[URL Generator]
        end

        subgraph "Storage Layer"
            S3Original[(S3<br/>Original Images)]
            S3Processed[(S3<br/>Processed Images)]
            MetaDB[(PostgreSQL<br/>Metadata)]
            Cache[(Redis<br/>URL Cache)]
        end

        Web --> CDN
        Mobile --> CDN
        CDN --> EdgeCache

        Web --> UploadAPI
        Mobile --> UploadAPI

        UploadAPI --> ImageValidator
        ImageValidator --> StorageWriter
        StorageWriter --> S3Original
        StorageWriter --> Queue
        StorageWriter --> MetaDB

        Queue --> ThumbnailWorker
        Queue --> MetadataExtractor

        ThumbnailWorker --> S3Processed
        MetadataExtractor --> MetaDB

        EdgeCache -->|Cache miss| ImageAPI
        ImageAPI --> Cache
        ImageAPI --> TransformService
        TransformService --> S3Original
        TransformService --> S3Processed
        ImageAPI --> URLGenerator

        style CDN fill:#e1f5ff
        style S3Original fill:#fff4e1
        style S3Processed fill:#fff4e1
        style MetaDB fill:#ffe1e1
    ```

    ---

    ## API Design

    ### 1. Upload Image

    **Request:**
    ```http
    POST /api/v1/images
    Content-Type: multipart/form-data

    image: [binary data]
    title: "My Image"
    description: "A beautiful sunset"
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "image_id": "img_abc123",
      "urls": {
        "original": "https://cdn.example.com/img_abc123.jpg",
        "small": "https://cdn.example.com/img_abc123_s.jpg",
        "medium": "https://cdn.example.com/img_abc123_m.jpg",
        "large": "https://cdn.example.com/img_abc123_l.jpg"
      },
      "processing_status": "pending",
      "metadata": {
        "width": 3000,
        "height": 2000,
        "format": "jpeg",
        "size_bytes": 2048000
      }
    }
    ```

    ---

    ### 2. Get Image (URL-based transformation)

    **Original:**
    ```
    GET https://cdn.example.com/img_abc123.jpg
    ```

    **With transformations (query params):**
    ```
    GET https://cdn.example.com/img_abc123.jpg?w=800&h=600&fmt=webp&q=85

    Parameters:
    - w: width in pixels
    - h: height in pixels
    - fmt: format (jpeg, png, webp, avif)
    - q: quality (1-100)
    - fit: crop, contain, cover, fill
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: image/webp
    Cache-Control: public, max-age=31536000
    ETag: "abc123def456"

    [image binary data]
    ```

    ---

    ### 3. Get Image Metadata

    **Request:**
    ```http
    GET /api/v1/images/img_abc123/metadata
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "image_id": "img_abc123",
      "width": 3000,
      "height": 2000,
      "format": "jpeg",
      "size_bytes": 2048000,
      "exif": {
        "camera": "Canon EOS R5",
        "iso": 100,
        "aperture": "f/2.8",
        "shutter_speed": "1/250",
        "date_taken": "2024-01-15T14:30:00Z"
      },
      "upload_date": "2024-01-15T14:35:00Z",
      "views": 1250
    }
    ```

    ---

    ### 4. Delete Image

    **Request:**
    ```http
    DELETE /api/v1/images/img_abc123
    ```

    **Response:**
    ```http
    HTTP/1.1 204 No Content
    ```

    ---

    ## Data Models

    ### Images Table

    ```sql
    CREATE TABLE images (
        image_id VARCHAR(64) PRIMARY KEY,
        user_id BIGINT NOT NULL,
        title VARCHAR(255),
        description TEXT,

        -- Original image metadata
        original_url VARCHAR(512) NOT NULL,
        width INT NOT NULL,
        height INT NOT NULL,
        format VARCHAR(16) NOT NULL,
        size_bytes BIGINT NOT NULL,

        -- Storage location
        storage_path VARCHAR(512) NOT NULL,

        -- Processing status
        processing_status VARCHAR(32) NOT NULL, -- pending, processing, completed, failed

        -- EXIF data (JSONB for flexibility)
        exif_data JSONB,

        -- Stats
        views BIGINT DEFAULT 0,

        -- Timestamps
        uploaded_at TIMESTAMP DEFAULT NOW(),
        processed_at TIMESTAMP,

        INDEX idx_user (user_id),
        INDEX idx_uploaded (uploaded_at),
        INDEX idx_status (processing_status)
    );
    ```

    ### Image Variants Table

    ```sql
    CREATE TABLE image_variants (
        variant_id SERIAL PRIMARY KEY,
        image_id VARCHAR(64) NOT NULL,
        variant_type VARCHAR(32) NOT NULL, -- small, medium, large, custom
        width INT NOT NULL,
        height INT NOT NULL,
        format VARCHAR(16) NOT NULL,
        storage_path VARCHAR(512) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),

        FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
        INDEX idx_image (image_id),
        UNIQUE (image_id, variant_type)
    );
    ```

    ---

    ## Upload and Processing Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant Validator
        participant S3
        participant Queue
        participant Worker
        participant DB

        Client->>API: 1. Upload image
        API->>Validator: Validate image
        Validator->>Validator: Check format, size, dimensions

        alt Invalid image
            Validator-->>API: Error
            API-->>Client: 400 Bad Request
        else Valid image
            API->>S3: Store original image
            S3-->>API: Storage URL

            API->>DB: Save metadata
            API->>Queue: Enqueue processing job
            API-->>Client: 201 Created (processing_status: pending)

            Queue->>Worker: Process image
            Worker->>S3: Download original
            Worker->>Worker: Generate thumbnails
            Worker->>Worker: Extract EXIF
            Worker->>S3: Upload variants
            Worker->>DB: Update metadata & variants
            Worker->>DB: Set status: completed
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics

    ### 1. Image Processing Pipeline

    ```python
    from PIL import Image
    import io

    class ImageProcessor:
        VARIANTS = {
            "small": (150, 150),
            "medium": (500, 500),
            "large": (1200, 1200)
        }

        def process_image(self, image_id: str, original_path: str):
            """Process image: generate thumbnails, extract metadata"""
            # Download original
            image_data = s3.get_object(Bucket=BUCKET, Key=original_path)
            img = Image.open(io.BytesIO(image_data['Body'].read()))

            # Generate thumbnails
            for variant_name, size in self.VARIANTS.items():
                thumbnail = self.create_thumbnail(img, size)
                variant_path = f"{image_id}_{variant_name[0]}.jpg"

                # Upload to S3
                buffer = io.BytesIO()
                thumbnail.save(buffer, format='JPEG', quality=85, optimize=True)
                buffer.seek(0)

                s3.put_object(
                    Bucket=BUCKET,
                    Key=f"images/processed/{variant_path}",
                    Body=buffer,
                    ContentType='image/jpeg',
                    CacheControl='public, max-age=31536000'
                )

                # Save variant metadata
                db.execute("""
                    INSERT INTO image_variants
                    (image_id, variant_type, width, height, format, storage_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, image_id, variant_name, thumbnail.width, thumbnail.height,
                     'jpeg', variant_path)

            # Extract and save EXIF
            exif_data = self.extract_exif(img)
            db.execute("""
                UPDATE images
                SET exif_data = ?, processing_status = 'completed',
                    processed_at = NOW()
                WHERE image_id = ?
            """, json.dumps(exif_data), image_id)

        def create_thumbnail(self, img: Image.Image, size: tuple) -> Image.Image:
            """Create thumbnail maintaining aspect ratio"""
            # Calculate aspect ratio
            img_copy = img.copy()
            img_copy.thumbnail(size, Image.Resampling.LANCZOS)
            return img_copy

        def extract_exif(self, img: Image.Image) -> dict:
            """Extract EXIF metadata"""
            exif_data = {}
            try:
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = str(value)
            except:
                pass
            return exif_data
    ```

    ---

    ### 2. On-Demand Image Transformation

    **URL-based transformations for flexibility:**

    ```python
    class ImageTransformService:
        def transform_image(self, image_id: str, params: dict) -> bytes:
            """Transform image based on URL parameters"""
            # Check cache first
            cache_key = f"img:{image_id}:{hash(str(params))}"
            cached = redis.get(cache_key)
            if cached:
                return cached

            # Get original or closest variant
            image_path = self.get_best_source(image_id, params.get('w'), params.get('h'))
            img_data = s3.get_object(Bucket=BUCKET, Key=image_path)
            img = Image.open(io.BytesIO(img_data['Body'].read()))

            # Apply transformations
            if params.get('w') or params.get('h'):
                img = self.resize(img, params.get('w'), params.get('h'),
                                params.get('fit', 'contain'))

            # Convert format
            output_format = params.get('fmt', 'jpeg').upper()
            if output_format == 'WEBP':
                output_format = 'WEBP'

            # Apply quality
            quality = int(params.get('q', 85))

            # Save to buffer
            buffer = io.BytesIO()
            img.save(buffer, format=output_format, quality=quality, optimize=True)
            result = buffer.getvalue()

            # Cache transformed image
            redis.setex(cache_key, 3600, result)  # 1 hour cache

            return result

        def resize(self, img: Image.Image, width: int, height: int,
                  fit: str) -> Image.Image:
            """Resize image with different fit modes"""
            if fit == 'crop':
                return self.crop_to_fit(img, width, height)
            elif fit == 'contain':
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
                return img
            elif fit == 'cover':
                return self.cover_fit(img, width, height)
            else:
                return img.resize((width, height), Image.Resampling.LANCZOS)

        def crop_to_fit(self, img: Image.Image, width: int, height: int) -> Image.Image:
            """Crop image to exact dimensions"""
            img_ratio = img.width / img.height
            target_ratio = width / height

            if img_ratio > target_ratio:
                # Image is wider, crop width
                new_width = int(img.height * target_ratio)
                offset = (img.width - new_width) // 2
                img = img.crop((offset, 0, offset + new_width, img.height))
            else:
                # Image is taller, crop height
                new_height = int(img.width / target_ratio)
                offset = (img.height - new_height) // 2
                img = img.crop((0, offset, img.width, offset + new_height))

            return img.resize((width, height), Image.Resampling.LANCZOS)
    ```

    ---

    ### 3. Smart Format Selection

    **Serve modern formats to supported browsers:**

    ```python
    class FormatSelector:
        def select_format(self, accept_header: str, original_format: str) -> str:
            """Choose best format based on browser support"""
            # Check Accept header for format support
            accepts = accept_header.lower()

            # Prefer AVIF (best compression)
            if 'image/avif' in accepts:
                return 'avif'

            # Fallback to WebP (good compression, wide support)
            if 'image/webp' in accepts:
                return 'webp'

            # Use original format
            return original_format
    ```

    **Savings:**
    - WebP: 25-35% smaller than JPEG
    - AVIF: 50% smaller than JPEG
    - Significant bandwidth and storage savings

    ---

    ### 4. CDN Integration

    **Cache-Control headers:**

    ```python
    def set_cache_headers(response, image_type: str):
        """Set appropriate cache headers"""
        if image_type == 'original':
            # Original images never change
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        else:
            # Transformed images cache for shorter period
            response.headers['Cache-Control'] = 'public, max-age=86400'

        # Enable CDN caching
        response.headers['CDN-Cache-Control'] = 'max-age=31536000'

        # ETag for validation
        response.headers['ETag'] = generate_etag(image_id, params)
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## Optimization Techniques

    ### 1. Lazy Thumbnail Generation

    **Generate thumbnails on first request:**

    ```python
    def serve_image(image_id: str, variant: str):
        """Serve image with lazy generation"""
        # Check if variant exists
        variant_path = f"{image_id}_{variant}.jpg"

        try:
            # Try to get from S3
            img_data = s3.get_object(Bucket=BUCKET, Key=variant_path)
            return img_data['Body'].read()
        except s3.exceptions.NoSuchKey:
            # Variant doesn't exist, generate on-the-fly
            original_path = f"{image_id}.jpg"
            processor = ImageProcessor()
            thumbnail = processor.create_thumbnail_sync(original_path, variant)

            # Asynchronously save for future requests
            queue.enqueue('save_thumbnail', image_id, variant, thumbnail)

            return thumbnail
    ```

    ---

    ### 2. Progressive Image Loading

    **Serve low-quality placeholder first:**

    ```html
    <img
      src="image_abc123_tiny.jpg"  <!-- 10KB blurred placeholder -->
      data-src="image_abc123_m.jpg"  <!-- Full quality -->
      class="lazy-load"
      alt="Image"
    />

    <script>
    // Lazy load full quality when in viewport
    observer.observe(img);
    </script>
    ```

    ---

    ### 3. Storage Cost Optimization

    **S3 Lifecycle Policies:**

    ```yaml
    lifecycle_policy:
      - rule: "move-cold-images"
        transition:
          - days: 90
            storage_class: "STANDARD_IA"  # Infrequent Access
          - days: 365
            storage_class: "GLACIER"  # Archive
        filter:
          prefix: "images/original/"
    ```

    **Savings: 60-80% on storage costs for old images**

    ---

    ### 4. Monitoring Metrics

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **CDN Cache Hit Rate** | > 90% | < 80% |
    | **Image Processing Time** | < 10s | > 30s |
    | **Upload Success Rate** | > 99% | < 95% |
    | **P99 Image Load Time** | < 500ms | > 1s |
    | **Storage Cost per Image** | < $0.001 | > $0.005 |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **CDN-first delivery** - Serve from edge locations for low latency
    2. **Async processing** - Generate thumbnails in background
    3. **On-demand transforms** - URL-based image manipulation
    4. **Modern formats** - WebP/AVIF for better compression
    5. **S3 lifecycle policies** - Move old images to cheaper storage
    6. **Lazy generation** - Create variants on first request

    ## Interview Tips

    ✅ **Discuss CDN strategy** - Why CDN? Cache headers, invalidation
    ✅ **Explain image formats** - JPEG vs PNG vs WebP vs AVIF
    ✅ **Cover async processing** - Why async? Queue-based architecture
    ✅ **Address storage costs** - Lifecycle policies, compression
    ✅ **Optimize delivery** - Responsive images, lazy loading

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle image abuse?"** | Content moderation, rate limiting, virus scanning |
    | **"How to serve different sizes for mobile?"** | Responsive images (srcset), detect device, serve appropriate size |
    | **"How to reduce storage costs?"** | Compression, deduplication, S3 lifecycle policies, modern formats |
    | **"How to ensure fast image loads?"** | CDN, lazy loading, progressive images, WebP/AVIF |
    | **"How to scale to billions of images?"** | Shard by image_id, use object storage, CDN for delivery |

    ## Real-World Examples

    - **Instagram**: Image processing pipeline, CDN delivery
    - **Pinterest**: Responsive images, lazy loading
    - **Imgur**: Simple upload, direct links
    - **Cloudinary**: URL-based transformations (reference implementation)

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Instagram, Pinterest, Imgur, Cloudinary

---

*This problem demonstrates image processing, CDN integration, and content delivery optimization for media-rich applications.*
