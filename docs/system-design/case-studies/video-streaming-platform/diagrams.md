# High-Level Architecture and Diagrams

This section outlines the overall architecture of the video streaming platform using Mermaid diagrams to visualize the system's components and data flow.

## System Architecture

The system is designed as a set of decoupled microservices, each responsible for a specific business capability. This approach allows for independent scaling, development, and deployment.

```mermaid
graph TB
    subgraph "User-Facing"
        Users[Users]
    end

    subgraph "Content Delivery Network"
        CDN[Global CDN / Edge Servers]
    end

    subgraph "Core Backend Services (API Gateway)"
        LB[Load Balancer]
        API[API Gateway]
    end

    subgraph "Microservices"
        Auth[Auth Service]
        Video[Video Service]
        User[User Service]
        Rec[Recommendation Engine]
        Search[Search Service]
        Analytics[Analytics Service]
        Live[Live Streaming Service]
    end

    subgraph "Video Processing Pipeline"
        Upload[Upload Service]
        Processing[Video Processing Queue]
        Transcoding[Transcoding Workers]
    end

    subgraph "Data Stores"
        ObjectStore[Object Storage (S3/GCS)]
        MetadataDB[(Video Metadata DB)]
        UserDB[(User Database)]
        RecDB[(Graph/Vector DB)]
        SearchIndex[Search Index]
        DataWarehouse[(Analytics Warehouse)]
    end

    Users --> CDN
    Users --> LB
    LB --> API

    API --> Auth
    API --> Video
    API --> User
    API --> Rec
    API --> Search
    API --> Analytics
    API --> Live

    Video --> Upload
    Upload --> ObjectStore
    Upload --> Processing
    Processing --> Transcoding
    Transcoding --> ObjectStore
    Transcoding --> MetadataDB

    CDN --> ObjectStore

    User --> UserDB
    Video --> MetadataDB
    Rec --> RecDB
    Search --> SearchIndex
    Analytics --> DataWarehouse
```

## Data Flow: Video Upload & Processing

This diagram illustrates the sequence of events when a user uploads a new video.

```mermaid
sequenceDiagram
    participant User
    participant API Gateway
    participant Upload Service
    participant Object Storage
    participant Processing Queue
    participant Transcoding Service
    participant Metadata DB

    User->>API Gateway: 1. Upload Video Request (with metadata)
    API Gateway->>Upload Service: 2. Forward Request
    Upload Service->>Object Storage: 3. Upload Raw Video File
    Upload Service->>Metadata DB: 4. Create Initial Video Record (status: 'uploading')
    Upload Service-->>User: 5. Acknowledge Upload
    Upload Service->>Processing Queue: 6. Enqueue Processing Job (video_id, raw_file_path)
    
    Processing Queue->>Transcoding Service: 7. Dequeue Job
    Transcoding Service->>Object Storage: 8. Download Raw Video
    Transcoding Service->>Transcoding Service: 9. Transcode to Multiple Resolutions (e.g., 1080p, 720p)
    Transcoding Service->>Object Storage: 10. Upload Transcoded Segments
    Transcoding Service->>Metadata DB: 11. Update Video Record (status: 'processed', add URLs)
```
