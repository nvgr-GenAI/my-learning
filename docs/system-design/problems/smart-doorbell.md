# Design a Smart Doorbell System (Ring, Nest Hello)

A smart doorbell system that provides real-time video streaming, motion detection via PIR sensors and computer vision, two-way audio communication, cloud recording, and instant push notifications for enhanced home security.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M devices, 1M video streams/day, 5M motion events/day, 24/7 recording for premium users |
| **Key Challenges** | Video streaming (RTSP/WebRTC), motion detection (PIR + CV), cloud recording, battery optimization, two-way audio |
| **Core Concepts** | Video encoding (H.264/H.265), motion detection pipeline, CDN distribution, edge computing, battery management |
| **Companies** | Ring, Nest Hello, Arlo, Eufy, Blink, Wyze, SimpliSafe |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Live Video Streaming** | Real-time video feed accessible via mobile app | P0 (Must have) |
    | **Motion Detection** | Detect motion using PIR sensor + computer vision | P0 (Must have) |
    | **Two-Way Audio** | Speak to visitors through doorbell | P0 (Must have) |
    | **Push Notifications** | Instant alerts when motion detected or doorbell pressed | P0 (Must have) |
    | **Cloud Recording** | Store video clips in cloud (event-based or 24/7) | P0 (Must have) |
    | **Night Vision** | Infrared LEDs for low-light video | P0 (Must have) |
    | **Person Detection** | Distinguish people from other motion (cars, animals) | P1 (Should have) |
    | **Package Detection** | Detect when packages are delivered/removed | P1 (Should have) |
    | **Pre-roll Recording** | Buffer video before motion event | P1 (Should have) |
    | **Smart Responses** | Pre-recorded messages for visitors | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Facial recognition (privacy concerns, GDPR)
    - Lock integration (focus on doorbell only)
    - Multiple camera angles
    - License plate recognition
    - Audio event detection (glass breaking, etc.)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% uptime | Users rely on doorbell for security |
    | **Latency (Live Stream)** | < 3s end-to-end | Real-time communication requirement |
    | **Latency (Notifications)** | < 2s from motion detection | Security critical for timely alerts |
    | **Video Quality** | 1080p @ 30fps (1080p/720p/480p adaptive) | Clear identification of visitors |
    | **Battery Life** | 6-12 months (battery models) | Minimize maintenance |
    | **Recording Retention** | 30 days (standard), 60 days (premium) | Balance storage cost vs. utility |
    | **Motion Detection Accuracy** | > 95% precision (reduce false positives) | Avoid alert fatigue |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total devices: 10M smart doorbells deployed
    Active devices: 7M (70% online, 3M wired, 4M battery)
    Daily motion events: 5M events (0.5 events per device average)
    Daily video streams: 1M streams (0.1 streams per device)

    Video streaming:
    - Average stream duration: 60 seconds
    - Peak hours: 5-9pm (3x multiplier)
    - Concurrent streams (average): 1M × 60s / 86,400s ≈ 694 concurrent
    - Peak concurrent: 2,082 streams

    Motion detection:
    - Events per second (average): 5M / 86,400 ≈ 58 events/sec
    - Peak: 174 events/sec
    - Clips per event: 1 (10-30 seconds)
    - Storage per clip: 5 MB (720p, 20s, H.264)

    Push notifications:
    - Notification rate: 58/sec average, 174/sec peak
    - Notification payload: 2 KB (includes thumbnail)

    Doorbell button presses:
    - Daily: 2M presses (0.2 presses per device)
    - Per second: 23/sec average, 69/sec peak
    ```

    ### Storage Estimates

    ```
    Device metadata:
    - Per device: 3 KB (device_id, owner_id, settings, firmware_version, location)
    - Total: 10M × 3 KB = 30 GB

    User data:
    - Per user: 5 KB (user_id, name, email, phone, subscription, preferences)
    - Users: 8M (1.25 devices per user average)
    - Total: 8M × 5 KB = 40 GB

    Video recordings (event-based):
    - Clips per day: 5M events × 1 clip = 5M clips
    - Size per clip: 5 MB (720p, 20s, H.264)
    - Daily storage: 5M × 5 MB = 25 TB/day
    - Retention (30 days): 25 TB × 30 = 750 TB
    - Retention (60 days premium): 25 TB × 60 = 1,500 TB

    24/7 recording (premium, 20% of users):
    - Devices with 24/7: 2M devices
    - Bitrate: 2 Mbps (720p H.264)
    - Daily per device: 2 Mbps × 86,400s / 8 = 21.6 GB/day
    - Total daily: 2M × 21.6 GB = 43.2 PB/day
    - Retention (30 days): 43.2 PB × 30 = 1.3 EB

    Thumbnails:
    - Per event: 100 KB (JPEG)
    - Daily: 5M × 100 KB = 500 GB/day
    - Retention (30 days): 15 TB

    Total storage (event-based only): 750 TB + 15 TB = 765 TB
    Total storage (with 24/7): 1.3 EB (massive!)
    ```

    ### Bandwidth Estimates

    ```
    Live video streaming:
    - Concurrent streams: 694 average, 2,082 peak
    - Bitrate per stream: 2 Mbps (720p adaptive)
    - Total bandwidth: 694 × 2 Mbps = 1.4 Gbps average
    - Peak bandwidth: 2,082 × 2 Mbps = 4.2 Gbps

    Video upload (motion events):
    - Events per second: 58 average, 174 peak
    - Clip size: 5 MB
    - Upload duration: 30 seconds
    - Concurrent uploads: 58 × 30 = 1,740
    - Bandwidth: 1,740 × (5 MB × 8 / 30s) = 2.3 Gbps average

    24/7 recording upload:
    - Devices: 2M
    - Bitrate: 2 Mbps each
    - Total: 2M × 2 Mbps = 4 Tbps (use edge caching/CDN!)

    Total bandwidth (without 24/7): 1.4 + 2.3 = 3.7 Gbps
    Total bandwidth (with 24/7): 4 Tbps (requires CDN and edge storage)
    ```

    ### Memory Estimates (Caching)

    ```
    Active sessions:
    - Concurrent users: 100K (1.25% of users)
    - Session data: 10 KB per user
    - Total: 100K × 10 KB = 1 GB

    Device state cache:
    - Active devices: 1M (10% frequently accessed)
    - State data: 5 KB (battery, firmware, status)
    - Total: 1M × 5 KB = 5 GB

    Video streaming metadata:
    - Concurrent streams: 2,082 peak
    - Metadata: 50 KB (device_id, stream_url, quality, timestamp)
    - Total: 2,082 × 50 KB ≈ 100 MB

    Motion detection cache (recent events):
    - Recent events: 500K (last hour)
    - Per event: 2 KB (device_id, timestamp, thumbnail_url)
    - Total: 500K × 2 KB = 1 GB

    Total cache: 1 + 5 + 0.1 + 1 = 7.1 GB
    ```

    ---

    ## Key Assumptions

    1. Average motion events: 0.5 per device per day
    2. Average video stream duration: 60 seconds
    3. Battery life: 6-12 months (depends on activity)
    4. WiFi connectivity: 95% of devices have reliable WiFi
    5. Video retention: 30 days standard, 60 days premium
    6. 24/7 recording: 20% of users (premium subscription)
    7. Firmware updates: Monthly

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Low-latency streaming** - WebRTC for real-time video, RTSP for recordings
    2. **Edge processing** - Motion detection on device to save bandwidth
    3. **Adaptive bitrate** - Adjust video quality based on network conditions
    4. **Scalable storage** - S3 with lifecycle policies for cost optimization
    5. **Battery optimization** - Sleep modes, wake on motion, efficient encoding

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Mobile[Mobile App<br/>iOS/Android]
            Web[Web Dashboard]
            Alexa[Voice Assistant<br/>Alexa/Google]
        end

        subgraph "Doorbell Device"
            Camera[Camera<br/>1080p sensor]
            PIR[PIR Sensor<br/>Motion detection]
            Mic[Microphone<br/>Audio input]
            Speaker[Speaker<br/>Audio output]
            Button[Doorbell Button]
            IR_LED[IR LEDs<br/>Night vision]
            WiFi[WiFi Module<br/>2.4/5GHz]
            Encoder[Video Encoder<br/>H.264/H.265]
            MCU[MCU/SoC<br/>CV processing]
            Battery[Battery<br/>Rechargeable]
        end

        subgraph "API Gateway"
            LB[Load Balancer]
            API_GW[API Gateway<br/>Auth, Rate limiting]
            WebRTC_Signal[WebRTC Signaling<br/>Server]
        end

        subgraph "Core Services"
            Device_Service[Device Service<br/>Device management]
            Stream_Service[Streaming Service<br/>Live video]
            Recording_Service[Recording Service<br/>Cloud storage]
            Motion_Service[Motion Detection<br/>CV processing]
            Notification_Service[Notification Service<br/>Push/Email]
            AI_Service[AI/ML Service<br/>Person/package detection]
            Auth_Service[Auth Service<br/>User authentication]
        end

        subgraph "Streaming Infrastructure"
            Media_Server[Media Servers<br/>WebRTC/RTSP]
            TURN[TURN Servers<br/>NAT traversal]
            Transcoder[Transcoding<br/>Adaptive bitrate]
        end

        subgraph "Storage"
            S3_Video[S3 Video Storage<br/>Recordings]
            S3_Thumb[S3 Thumbnails<br/>Event snapshots]
            Glacier[S3 Glacier<br/>Old recordings]
            Device_DB[(Device DB<br/>PostgreSQL<br/>Devices, settings)]
            Event_DB[(Events DB<br/>Cassandra<br/>Motion events)]
            User_DB[(User DB<br/>PostgreSQL<br/>Users, subscriptions)]
        end

        subgraph "Caching"
            Redis_Session[Redis<br/>User sessions]
            Redis_Device[Redis<br/>Device state]
            CDN[CloudFront CDN<br/>Thumbnails, clips]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Events, clips]
        end

        subgraph "External Services"
            FCM[FCM/APNs<br/>Push notifications]
            Email_Service[SendGrid<br/>Email]
            Smart_Home[Smart Home APIs<br/>Alexa, Google]
            ML_Platform[ML Platform<br/>TensorFlow/PyTorch]
        end

        Mobile --> LB
        Web --> LB
        Alexa --> Smart_Home
        Smart_Home --> LB

        PIR --> MCU
        Button --> MCU
        Camera --> Encoder
        Encoder --> MCU
        Mic --> MCU
        IR_LED --> Camera
        MCU --> WiFi

        WiFi --> LB
        LB --> API_GW
        API_GW --> Device_Service
        API_GW --> Stream_Service
        API_GW --> Recording_Service
        API_GW --> Motion_Service
        API_GW --> Auth_Service

        Stream_Service --> WebRTC_Signal
        WebRTC_Signal --> Media_Server
        Media_Server --> TURN
        Media_Server --> Transcoder
        Media_Server <--> WiFi

        Device_Service --> Device_DB
        Device_Service --> Redis_Device

        Motion_Service --> Event_DB
        Motion_Service --> Kafka
        Motion_Service --> AI_Service
        AI_Service --> ML_Platform

        Recording_Service --> S3_Video
        Recording_Service --> S3_Thumb
        S3_Video --> Glacier

        Kafka --> Notification_Service
        Kafka --> Recording_Service
        Notification_Service --> FCM
        Notification_Service --> Email_Service

        Auth_Service --> User_DB
        Auth_Service --> Redis_Session

        S3_Thumb --> CDN
        S3_Video --> CDN
        CDN --> Mobile

        MCU --> Speaker
        Battery --> MCU

        style WiFi fill:#e8f5e9
        style Redis_Session fill:#fff4e1
        style Redis_Device fill:#fff4e1
        style Device_DB fill:#ffe1e1
        style Event_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style S3_Video fill:#e1f5fe
        style Kafka fill:#e8eaf6
        style CDN fill:#f3e5f5
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **WebRTC** | Low-latency peer-to-peer streaming (< 3s) | RTMP (higher latency ~10s), HLS (15-30s latency) |
    | **H.264 Encoding** | Wide compatibility, hardware acceleration | H.265 (better compression but less compatible), VP9 (no hardware support) |
    | **PIR Sensor** | Low power motion detection trigger | Computer vision only (drains battery too fast) |
    | **S3 + Glacier** | Cost-effective tiered storage ($0.023/GB → $0.004/GB) | Pure S3 (expensive for long retention), On-premise (doesn't scale) |
    | **Cassandra (Events)** | High write throughput for events (174/sec peak) | PostgreSQL (adequate but less scalable), DynamoDB (vendor lock-in) |
    | **CDN (CloudFront)** | Global video delivery, reduce origin load | Direct S3 (higher latency, egress costs), Custom edge servers (complexity) |

    **Key Trade-off:** We chose **WebRTC for live streaming** (low latency) but **RTSP/HLS for recordings** (cost-effective CDN delivery). This dual approach balances real-time communication with scalable playback.

    ---

    ## API Design

    ### 1. Start Live Video Stream

    **Request:**
    ```http
    POST /api/v1/devices/{device_id}/stream/start
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "quality": "720p",  // 1080p, 720p, 480p, auto
      "audio_enabled": true,
      "latency_mode": "real_time"  // real_time, normal, low_bandwidth
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "stream_id": "stream_abc123",
      "device_id": "dev_xyz789",
      "webrtc_offer": {
        "type": "offer",
        "sdp": "v=0\r\no=- 1234567890 1234567890 IN IP4 0.0.0.0\r\n..."
      },
      "ice_servers": [
        {
          "urls": "stun:stun.example.com:3478"
        },
        {
          "urls": "turn:turn.example.com:3478",
          "username": "user123",
          "credential": "pass123"
        }
      ],
      "expires_at": "2026-02-05T10:35:00Z"
    }
    ```

    **Design Notes:**

    - WebRTC SDP (Session Description Protocol) for peer negotiation
    - STUN/TURN servers for NAT traversal
    - Stream expires after 5 minutes of inactivity
    - Client sends ICE candidates for connection establishment

    ---

    ### 2. Get Motion Events

    **Request:**
    ```http
    GET /api/v1/devices/{device_id}/events?start_time=2026-02-01T00:00:00Z&end_time=2026-02-05T23:59:59Z&event_type=motion&limit=50&offset=0
    Authorization: Bearer <token>
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "events": [
        {
          "event_id": "evt_abc123",
          "device_id": "dev_xyz789",
          "event_type": "motion",  // motion, doorbell_press, person_detected, package_detected
          "timestamp": "2026-02-05T10:30:00Z",
          "duration_seconds": 20,
          "thumbnail_url": "https://cdn.example.com/thumbnails/evt_abc123.jpg",
          "video_url": "https://cdn.example.com/clips/evt_abc123.mp4",
          "detection_metadata": {
            "confidence": 0.95,
            "detected_objects": ["person"],
            "bounding_boxes": [
              {"x": 100, "y": 150, "w": 80, "h": 200, "label": "person"}
            ]
          },
          "pre_roll_seconds": 3
        },
        // ... more events
      ],
      "pagination": {
        "total": 250,
        "limit": 50,
        "offset": 0,
        "has_more": true
      }
    }
    ```

    ---

    ### 3. Send Two-Way Audio

    **Request:**
    ```http
    POST /api/v1/devices/{device_id}/audio/speak
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "audio_url": "https://app.example.com/audio/message_123.mp3",
      "duration_seconds": 5,
      "type": "live"  // live, pre_recorded
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "status": "playing",
      "audio_id": "audio_abc123",
      "device_id": "dev_xyz789"
    }
    ```

    **Design Notes:**

    - Audio streamed via WebRTC audio track (for live)
    - Pre-recorded messages cached on device
    - Opus codec for audio (low bitrate, high quality)

    ---

    ### 4. Configure Motion Detection

    **Request:**
    ```http
    PUT /api/v1/devices/{device_id}/settings/motion
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "enabled": true,
      "sensitivity": "medium",  // low, medium, high
      "detection_zones": [
        {
          "zone_id": "zone_1",
          "name": "Front Path",
          "polygon": [
            {"x": 0, "y": 0},
            {"x": 100, "y": 0},
            {"x": 100, "y": 100},
            {"x": 0, "y": 100}
          ],
          "enabled": true
        }
      ],
      "ai_detection": {
        "person_detection": true,
        "vehicle_detection": false,
        "animal_detection": false,
        "package_detection": true
      },
      "notification_cooldown_seconds": 60,
      "schedule": {
        "enabled": true,
        "time_windows": [
          {"start": "22:00", "end": "06:00", "days": ["all"]}
        ]
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "device_id": "dev_xyz789",
      "motion_settings": {
        "enabled": true,
        "sensitivity": "medium",
        "zones_count": 1,
        "ai_detection_enabled": true
      }
    }
    ```

    ---

    ## Database Schema

    ### Devices (PostgreSQL)

    ```sql
    -- Devices table
    CREATE TABLE devices (
        device_id UUID PRIMARY KEY,
        serial_number VARCHAR(50) UNIQUE NOT NULL,
        owner_id UUID NOT NULL REFERENCES users(user_id),
        device_name VARCHAR(100),
        model VARCHAR(50),
        firmware_version VARCHAR(20),
        hardware_version VARCHAR(20),
        installation_date DATE,

        -- Location
        location_lat DECIMAL(9,6),
        location_lng DECIMAL(9,6),
        timezone VARCHAR(50),

        -- Network
        wifi_ssid VARCHAR(100),
        local_ip_address INET,
        public_ip_address INET,
        connection_status VARCHAR(20),  -- online, offline, streaming
        last_seen_at TIMESTAMP,

        -- Power
        power_source VARCHAR(20),  -- battery, wired, solar
        battery_level INT,  -- 0-100
        battery_charging BOOLEAN,

        -- Settings
        video_quality VARCHAR(10),  -- 1080p, 720p, 480p, auto
        night_vision_enabled BOOLEAN DEFAULT true,
        motion_detection_enabled BOOLEAN DEFAULT true,
        audio_enabled BOOLEAN DEFAULT true,
        recording_mode VARCHAR(20),  -- event_only, 24_7, off

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        INDEX idx_owner (owner_id),
        INDEX idx_status (connection_status, last_seen_at)
    );

    -- Motion detection settings
    CREATE TABLE motion_settings (
        device_id UUID PRIMARY KEY REFERENCES devices(device_id),
        sensitivity VARCHAR(10),  -- low, medium, high
        detection_zones JSONB,  -- Array of polygons
        ai_detection_config JSONB,  -- {person: true, vehicle: false, ...}
        notification_cooldown_seconds INT DEFAULT 60,
        schedule JSONB,  -- Time windows for motion detection

        INDEX idx_device (device_id)
    );
    ```

    ---

    ### Motion Events (Cassandra)

    ```sql
    -- Time-series motion events
    CREATE TABLE motion_events (
        device_id UUID,
        event_time TIMESTAMP,
        event_id UUID,
        event_type VARCHAR(20),  -- motion, doorbell_press, person, package

        -- Video
        video_url TEXT,
        thumbnail_url TEXT,
        duration_seconds INT,
        video_size_bytes BIGINT,
        pre_roll_seconds INT,

        -- Detection metadata
        confidence DOUBLE,
        detected_objects LIST<TEXT>,
        bounding_boxes TEXT,  -- JSON serialized

        -- Context
        battery_level INT,
        recording_quality VARCHAR(10),

        PRIMARY KEY (device_id, event_time, event_id)
    ) WITH CLUSTERING ORDER BY (event_time DESC);

    -- User activity (for user-centric queries)
    CREATE TABLE user_motion_events (
        user_id UUID,
        event_time TIMESTAMP,
        event_id UUID,
        device_id UUID,
        event_type VARCHAR(20),
        thumbnail_url TEXT,

        PRIMARY KEY (user_id, event_time, event_id)
    ) WITH CLUSTERING ORDER BY (event_time DESC);
    ```

    ---

    ## Data Flow Diagrams

    ### Motion Detection Flow

    ```mermaid
    sequenceDiagram
        participant PIR as PIR Sensor
        participant MCU as Device MCU
        participant CV as CV Engine
        participant Cloud as Cloud API
        participant AI as AI Service
        participant S3 as S3 Storage
        participant User as Mobile App

        PIR->>MCU: Motion detected (wake signal)
        MCU->>MCU: Wake from sleep mode
        MCU->>CV: Start camera + video encoding

        CV->>CV: Capture pre-roll buffer (3s)
        CV->>CV: Run basic motion detection (edge)

        alt Significant motion detected
            CV->>Cloud: Upload video clip (H.264)
            Cloud->>S3: Store video + thumbnail
            Cloud->>AI: Analyze with ML model

            AI->>AI: Person detection (YOLO/SSD)
            AI->>AI: Generate bounding boxes
            AI-->>Cloud: Detection results

            Cloud->>Cloud: Create event record
            Cloud->>Kafka: Publish motion event

            Kafka->>Notification_Service: Motion event
            Notification_Service->>FCM: Send push notification
            FCM->>User: "Motion detected at Front Door"

            User->>Cloud: View clip
            Cloud->>S3: Get video URL (signed)
            S3-->>User: Stream video via CDN
        else False positive (low confidence)
            CV->>MCU: Discard buffer
            MCU->>MCU: Return to sleep
        end
    ```

    **Flow Explanation:**

    1. **PIR sensor trigger** - Low-power motion detection wakes MCU
    2. **Pre-roll buffer** - Capture 3 seconds before motion for context
    3. **Edge processing** - Basic motion detection on device (save bandwidth)
    4. **Cloud upload** - Only upload significant events
    5. **AI analysis** - Cloud-based person/package detection
    6. **Instant notification** - Push alert within 2 seconds
    7. **CDN delivery** - Fast video playback from edge cache

    ---

    ### Live Video Streaming Flow (WebRTC)

    ```mermaid
    sequenceDiagram
        participant User as Mobile App
        participant API as API Gateway
        participant Signal as WebRTC Signaling
        participant TURN as TURN Server
        participant Device as Doorbell

        User->>API: POST /devices/{id}/stream/start
        API->>API: Authenticate user
        API->>Device: Wake + prepare stream
        Device->>Device: Start camera encoding

        API->>Signal: Create WebRTC session
        Signal-->>User: WebRTC Offer (SDP)

        User->>User: Process SDP offer
        User->>Signal: WebRTC Answer (SDP)
        Signal->>Device: Forward SDP answer

        User->>Signal: ICE Candidates (NAT info)
        Signal->>Device: Forward ICE candidates
        Device->>Signal: ICE Candidates
        Signal->>User: Forward ICE candidates

        alt Direct P2P connection possible
            User<->>Device: Direct RTP stream (video/audio)
        else NAT/Firewall blocking
            User->>TURN: Connect via TURN relay
            TURN->>Device: Relay connection
            User<->>TURN: Relayed RTP stream
            TURN<->>Device: Relay bidirectional
        end

        User->>Device: Audio (speak to visitor)
        Device->>Speaker: Play audio output
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical Smart Doorbell subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Motion Detection Pipeline** | How to detect motion efficiently? | PIR sensor + edge CV + cloud AI |
    | **Video Streaming** | How to stream with low latency? | WebRTC with adaptive bitrate |
    | **Cloud Recording** | How to store massive video data? | S3 tiered storage with lifecycle |
    | **Battery Optimization** | How to last 6-12 months? | Sleep modes, efficient encoding, wake on PIR |

    ---

    === "🎥 Motion Detection Pipeline"

        ## The Challenge

        **Problem:** Detect meaningful motion (people, packages) while minimizing false positives (trees, cars, animals) and conserving battery.

        **Requirements:**
        - High accuracy: > 95% precision (reduce false alerts)
        - Low latency: < 2s from motion to notification
        - Battery efficient: Only wake for significant motion
        - Person detection: Distinguish humans from other motion

        ---

        ## Three-Tier Detection Architecture

        **Tier 1: PIR Sensor (Hardware)**

        ```python
        # Low-power hardware motion detection

        class PIRMotionSensor:
            """Passive Infrared sensor for initial motion detection"""

            def __init__(self):
                self.gpio_pin = 18  # PIR sensor connected to GPIO
                self.sensitivity_threshold = 0.6  # 60% change to trigger

            def detect_motion(self) -> bool:
                """
                Detect infrared changes (body heat)

                Returns:
                    True if motion detected, False otherwise

                Power: ~0.05 mA (ultra low power)
                Latency: ~100ms
                """
                # Read PIR sensor digital output
                motion_detected = GPIO.input(self.gpio_pin)

                if motion_detected:
                    # Wake main MCU
                    self.wake_mcu()
                    return True

                return False

            def wake_mcu(self):
                """Send interrupt to wake MCU from sleep"""
                GPIO.setup(WAKE_PIN, GPIO.OUT)
                GPIO.output(WAKE_PIN, GPIO.HIGH)
        ```

        **Tier 2: Edge Computer Vision (Device)**

        ```python
        import cv2
        import numpy as np

        class EdgeMotionDetection:
            """On-device motion detection using computer vision"""

            def __init__(self):
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=16,
                    detectShadows=True
                )
                self.min_contour_area = 5000  # Minimum pixels for motion
                self.frame_buffer = []
                self.pre_roll_frames = 90  # 3 seconds @ 30fps

            def detect_motion(self, frame: np.ndarray) -> dict:
                """
                Detect motion using background subtraction

                Args:
                    frame: Video frame (1080x1920x3)

                Returns:
                    {
                        'motion_detected': bool,
                        'confidence': float,
                        'motion_area': int (pixels),
                        'bounding_boxes': list
                    }

                Power: ~500 mA while processing
                Latency: ~50ms per frame
                """
                # Apply background subtraction
                fg_mask = self.bg_subtractor.apply(frame)

                # Remove noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Find contours
                contours, _ = cv2.findContours(
                    fg_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Filter small contours
                significant_contours = [
                    c for c in contours
                    if cv2.contourArea(c) > self.min_contour_area
                ]

                if not significant_contours:
                    return {
                        'motion_detected': False,
                        'confidence': 0.0,
                        'motion_area': 0,
                        'bounding_boxes': []
                    }

                # Calculate bounding boxes
                bounding_boxes = []
                total_area = 0

                for contour in significant_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)

                    bounding_boxes.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'area': int(area)
                    })
                    total_area += area

                # Calculate confidence based on motion area
                frame_area = frame.shape[0] * frame.shape[1]
                motion_percentage = total_area / frame_area
                confidence = min(motion_percentage * 10, 1.0)  # Scale to 0-1

                return {
                    'motion_detected': True,
                    'confidence': confidence,
                    'motion_area': total_area,
                    'bounding_boxes': bounding_boxes
                }

            def maintain_pre_roll_buffer(self, frame: np.ndarray):
                """
                Maintain circular buffer for pre-roll recording

                Keeps last 3 seconds of video in memory
                """
                self.frame_buffer.append(frame)

                if len(self.frame_buffer) > self.pre_roll_frames:
                    self.frame_buffer.pop(0)

            def get_pre_roll_clip(self) -> list:
                """Get pre-roll frames for video clip"""
                return self.frame_buffer.copy()
        ```

        **Tier 3: Cloud AI (Deep Learning)**

        ```python
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from PIL import Image
        import numpy as np

        class CloudAIDetection:
            """Cloud-based AI detection for person/package classification"""

            def __init__(self):
                # Load pre-trained Faster R-CNN model
                self.model = fasterrcnn_resnet50_fpn(pretrained=True)
                self.model.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)

                # COCO class labels
                self.COCO_CLASSES = {
                    1: 'person',
                    2: 'bicycle',
                    3: 'car',
                    28: 'suitcase',  # Package proxy
                    85: 'backpack'   # Package proxy
                }

                self.confidence_threshold = 0.7

            def detect_objects(self, video_url: str) -> dict:
                """
                Detect persons and packages in video clip

                Args:
                    video_url: S3 URL of video clip

                Returns:
                    {
                        'person_detected': bool,
                        'package_detected': bool,
                        'confidence': float,
                        'objects': list of detected objects
                    }

                Latency: ~500ms
                Cost: ~$0.001 per inference
                """
                # Extract keyframe from video
                frame = self._extract_keyframe(video_url)

                # Preprocess image
                image_tensor = self._preprocess_image(frame)

                # Run inference
                with torch.no_grad():
                    predictions = self.model([image_tensor.to(self.device)])

                # Parse predictions
                boxes = predictions[0]['boxes'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()

                # Filter by confidence
                high_conf_indices = scores > self.confidence_threshold
                boxes = boxes[high_conf_indices]
                labels = labels[high_conf_indices]
                scores = scores[high_conf_indices]

                # Categorize detections
                detected_objects = []
                person_detected = False
                package_detected = False
                max_confidence = 0.0

                for box, label, score in zip(boxes, labels, scores):
                    class_name = self.COCO_CLASSES.get(label, 'unknown')

                    detected_objects.append({
                        'class': class_name,
                        'confidence': float(score),
                        'bounding_box': {
                            'x': int(box[0]),
                            'y': int(box[1]),
                            'width': int(box[2] - box[0]),
                            'height': int(box[3] - box[1])
                        }
                    })

                    if class_name == 'person':
                        person_detected = True
                    elif class_name in ['suitcase', 'backpack']:
                        package_detected = True

                    max_confidence = max(max_confidence, score)

                return {
                    'person_detected': person_detected,
                    'package_detected': package_detected,
                    'confidence': float(max_confidence),
                    'objects': detected_objects,
                    'total_detections': len(detected_objects)
                }

            def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
                """Convert image to tensor for model input"""
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Convert to tensor and normalize
                image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
                image_tensor = image_tensor / 255.0

                return image_tensor

            def _extract_keyframe(self, video_url: str) -> np.ndarray:
                """Extract middle frame from video for analysis"""
                # Download video from S3
                cap = cv2.VideoCapture(video_url)

                # Get frame count and seek to middle
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

                ret, frame = cap.read()
                cap.release()

                return frame
        ```

        **Pipeline Integration:**

        ```python
        class MotionDetectionPipeline:
            """Orchestrate three-tier motion detection"""

            def __init__(self):
                self.pir_sensor = PIRMotionSensor()
                self.edge_cv = EdgeMotionDetection()
                self.cloud_ai = CloudAIDetection()
                self.video_encoder = H264Encoder()

            async def process_motion_event(self):
                """
                Full motion detection pipeline

                1. PIR sensor wakes device (hardware, ultra low power)
                2. Edge CV confirms motion (device, low power)
                3. Cloud AI classifies objects (cloud, high accuracy)
                """
                # Tier 1: PIR sensor detects motion
                if not self.pir_sensor.detect_motion():
                    return None

                # Wake camera and start encoding
                camera = Camera()
                camera.start()

                # Tier 2: Edge CV analysis
                frames = []
                motion_detected = False

                for i in range(30):  # 1 second @ 30fps
                    frame = camera.capture_frame()
                    self.edge_cv.maintain_pre_roll_buffer(frame)
                    frames.append(frame)

                    # Run motion detection every 5 frames
                    if i % 5 == 0:
                        result = self.edge_cv.detect_motion(frame)
                        if result['motion_detected'] and result['confidence'] > 0.3:
                            motion_detected = True

                if not motion_detected:
                    camera.stop()
                    return None  # False positive, return to sleep

                # Continue recording for 20 seconds
                for i in range(570):  # 19 more seconds
                    frame = camera.capture_frame()
                    frames.append(frame)

                camera.stop()

                # Encode video with pre-roll
                pre_roll_frames = self.edge_cv.get_pre_roll_clip()
                all_frames = pre_roll_frames + frames
                video_file = self.video_encoder.encode(all_frames)

                # Upload to cloud
                video_url = await self._upload_to_s3(video_file)

                # Tier 3: Cloud AI analysis
                ai_result = self.cloud_ai.detect_objects(video_url)

                # Create event record
                event = {
                    'event_id': str(uuid.uuid4()),
                    'timestamp': datetime.utcnow(),
                    'video_url': video_url,
                    'duration_seconds': len(all_frames) / 30,
                    'pre_roll_seconds': 3,
                    'detection': {
                        'person_detected': ai_result['person_detected'],
                        'package_detected': ai_result['package_detected'],
                        'confidence': ai_result['confidence'],
                        'objects': ai_result['objects']
                    }
                }

                return event
        ```

        ---

        ## Performance Metrics

        | Tier | Latency | Power | Accuracy | Cost |
        |------|---------|-------|----------|------|
        | **PIR Sensor** | 100ms | 0.05 mA | 85% (motion only) | $0 |
        | **Edge CV** | 50ms/frame | 500 mA | 90% (motion + area) | $0 |
        | **Cloud AI** | 500ms | 0 (cloud) | 98% (object class) | $0.001 |
        | **Pipeline Total** | < 2s | Avg 50 mA | 98% | $0.001/event |

    === "📡 Video Streaming Protocols"

        ## The Challenge

        **Problem:** Stream live video with low latency (< 3s) while supporting two-way audio and handling poor network conditions.

        **Requirements:**
        - Real-time streaming: < 3s end-to-end latency
        - Adaptive bitrate: Adjust quality for network conditions
        - Two-way audio: Full-duplex communication
        - NAT traversal: Work behind firewalls/routers

        ---

        ## WebRTC Architecture

        **Why WebRTC:**
        - Sub-second latency (typically 300-500ms)
        - Built-in NAT traversal (STUN/TURN)
        - Adaptive bitrate (automatic quality adjustment)
        - Two-way audio/video (full duplex)
        - Browser native (no plugins)

        **Implementation:**

        ```python
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from aiortc.contrib.media import MediaRecorder
        import asyncio

        class DoorbellVideoStream:
            """WebRTC video streaming for doorbell"""

            def __init__(self, device_id: str):
                self.device_id = device_id
                self.pc = RTCPeerConnection()
                self.video_track = None
                self.audio_track = None

                # STUN/TURN servers for NAT traversal
                self.ice_servers = [
                    {'urls': 'stun:stun.l.google.com:19302'},
                    {
                        'urls': 'turn:turn.example.com:3478',
                        'username': 'doorbell',
                        'credential': 'secret123'
                    }
                ]

            async def create_offer(self) -> dict:
                """
                Create WebRTC offer (SDP) for client

                Returns:
                    {
                        'type': 'offer',
                        'sdp': '<session_description>'
                    }
                """
                # Create video track from doorbell camera
                self.video_track = DoorbellVideoTrack(self.device_id)
                self.pc.addTrack(self.video_track)

                # Create audio track from doorbell microphone
                self.audio_track = DoorbellAudioTrack(self.device_id)
                self.pc.addTrack(self.audio_track)

                # Create SDP offer
                offer = await self.pc.createOffer()
                await self.pc.setLocalDescription(offer)

                return {
                    'type': self.pc.localDescription.type,
                    'sdp': self.pc.localDescription.sdp
                }

            async def process_answer(self, answer: dict):
                """
                Process WebRTC answer from client

                Args:
                    answer: SDP answer from mobile app
                """
                remote_description = RTCSessionDescription(
                    sdp=answer['sdp'],
                    type=answer['type']
                )
                await self.pc.setRemoteDescription(remote_description)

            async def add_ice_candidate(self, candidate: dict):
                """Add ICE candidate for NAT traversal"""
                await self.pc.addIceCandidate(candidate)

            def get_stats(self) -> dict:
                """
                Get streaming statistics

                Returns:
                    {
                        'video_bitrate': int (bps),
                        'audio_bitrate': int (bps),
                        'packet_loss': float (0-1),
                        'rtt': int (ms)
                    }
                """
                stats = self.pc.getStats()

                return {
                    'video_bitrate': stats.get('videoBitrate', 0),
                    'audio_bitrate': stats.get('audioBitrate', 0),
                    'packet_loss': stats.get('packetLoss', 0),
                    'rtt': stats.get('roundTripTime', 0)
                }


        class DoorbellVideoTrack(VideoStreamTrack):
            """Video track from doorbell camera"""

            def __init__(self, device_id: str):
                super().__init__()
                self.device_id = device_id
                self.camera = Camera(device_id)
                self.encoder = H264Encoder(bitrate=2_000_000)  # 2 Mbps

            async def recv(self):
                """
                Receive next video frame

                Returns:
                    VideoFrame with encoded H.264 data
                """
                # Capture frame from camera
                frame = await self.camera.capture_frame()

                # Encode with H.264
                encoded_frame = self.encoder.encode(frame)

                # Adaptive bitrate based on network conditions
                if self._should_reduce_bitrate():
                    self.encoder.set_bitrate(1_000_000)  # Drop to 1 Mbps

                return encoded_frame

            def _should_reduce_bitrate(self) -> bool:
                """Check if network requires lower bitrate"""
                stats = self.get_stats()

                # Reduce bitrate if packet loss > 5% or RTT > 500ms
                return stats['packet_loss'] > 0.05 or stats['rtt'] > 500


        class DoorbellAudioTrack(AudioStreamTrack):
            """Audio track from doorbell microphone"""

            def __init__(self, device_id: str):
                super().__init__()
                self.device_id = device_id
                self.microphone = Microphone(device_id)
                self.encoder = OpusEncoder()  # Opus for audio

            async def recv(self):
                """
                Receive next audio frame

                Returns:
                    AudioFrame with encoded Opus data
                """
                # Capture audio from microphone
                audio_data = await self.microphone.capture_audio()

                # Encode with Opus
                encoded_audio = self.encoder.encode(audio_data)

                return encoded_audio
        ```

        ---

        ## Signaling Server

        ```python
        from fastapi import FastAPI, WebSocket
        from fastapi.websockets import WebSocketDisconnect
        import json

        app = FastAPI()

        # Active streaming sessions
        streaming_sessions = {}

        @app.websocket("/ws/stream/{device_id}")
        async def webrtc_signaling(websocket: WebSocket, device_id: str):
            """
            WebRTC signaling server

            Handles SDP exchange and ICE candidate exchange
            """
            await websocket.accept()

            stream = DoorbellVideoStream(device_id)
            streaming_sessions[device_id] = stream

            try:
                while True:
                    # Receive message from client
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    if data['type'] == 'offer_request':
                        # Client requests offer
                        offer = await stream.create_offer()
                        await websocket.send_json({
                            'type': 'offer',
                            'offer': offer,
                            'ice_servers': stream.ice_servers
                        })

                    elif data['type'] == 'answer':
                        # Client sends answer
                        await stream.process_answer(data['answer'])
                        await websocket.send_json({
                            'type': 'ack',
                            'status': 'connected'
                        })

                    elif data['type'] == 'ice_candidate':
                        # ICE candidate from client
                        await stream.add_ice_candidate(data['candidate'])

                    elif data['type'] == 'get_stats':
                        # Client requests streaming stats
                        stats = stream.get_stats()
                        await websocket.send_json({
                            'type': 'stats',
                            'stats': stats
                        })

            except WebSocketDisconnect:
                # Client disconnected
                if device_id in streaming_sessions:
                    del streaming_sessions[device_id]
        ```

        ---

        ## Adaptive Bitrate

        ```python
        class AdaptiveBitrateController:
            """Dynamically adjust video quality based on network"""

            # Quality presets
            QUALITIES = {
                '1080p': {'width': 1920, 'height': 1080, 'bitrate': 4_000_000},
                '720p':  {'width': 1280, 'height': 720,  'bitrate': 2_000_000},
                '480p':  {'width': 854,  'height': 480,  'bitrate': 1_000_000},
                '360p':  {'width': 640,  'height': 360,  'bitrate': 500_000}
            }

            def __init__(self):
                self.current_quality = '720p'
                self.stats_history = []

            def adjust_quality(self, stats: dict) -> str:
                """
                Adjust quality based on network stats

                Args:
                    stats: {packet_loss, rtt, bitrate}

                Returns:
                    New quality level
                """
                self.stats_history.append(stats)

                # Keep last 10 seconds of stats
                if len(self.stats_history) > 10:
                    self.stats_history.pop(0)

                # Calculate average metrics
                avg_packet_loss = sum(s['packet_loss'] for s in self.stats_history) / len(self.stats_history)
                avg_rtt = sum(s['rtt'] for s in self.stats_history) / len(self.stats_history)

                # Downgrade quality if network is poor
                if avg_packet_loss > 0.1 or avg_rtt > 500:
                    # Very poor network
                    new_quality = '360p'
                elif avg_packet_loss > 0.05 or avg_rtt > 300:
                    # Poor network
                    new_quality = '480p'
                elif avg_packet_loss > 0.02 or avg_rtt > 200:
                    # Moderate network
                    new_quality = '720p'
                else:
                    # Good network
                    new_quality = '1080p'

                if new_quality != self.current_quality:
                    print(f"Switching quality: {self.current_quality} → {new_quality}")
                    self.current_quality = new_quality

                return new_quality
        ```

        ---

        ## Comparison: WebRTC vs RTSP vs HLS

        | Protocol | Latency | Use Case | Pros | Cons |
        |----------|---------|----------|------|------|
        | **WebRTC** | 300-500ms | Live streaming, two-way audio | Ultra-low latency, P2P, NAT traversal | Complex setup, browser dependency |
        | **RTSP** | 2-3s | Recording playback | Simple, widely supported | Higher latency, no NAT traversal |
        | **HLS** | 15-30s | CDN distribution | Scalable, CDN-friendly | Very high latency |

        **Our choice:** WebRTC for live, HLS for recorded clips (via CDN)

    === "☁️ Cloud Recording & Storage"

        ## The Challenge

        **Problem:** Store massive amounts of video data cost-effectively while ensuring fast retrieval.

        **Scale:**
        - 5M clips/day × 5 MB = 25 TB/day
        - 30-day retention = 750 TB
        - 24/7 recording (20% users) = 1.3 EB

        ---

        ## S3 Tiered Storage

        **Storage lifecycle:**

        ```python
        import boto3
        from datetime import datetime, timedelta

        class VideoStorageManager:
            """Manage video storage with S3 lifecycle policies"""

            def __init__(self):
                self.s3 = boto3.client('s3')
                self.bucket_name = 'doorbell-recordings'

                # Storage tiers (AWS S3)
                self.TIERS = {
                    'hot': {
                        'class': 'STANDARD',
                        'cost_per_gb': 0.023,
                        'retrieval_ms': 10
                    },
                    'warm': {
                        'class': 'STANDARD_IA',
                        'cost_per_gb': 0.0125,
                        'retrieval_ms': 100
                    },
                    'cold': {
                        'class': 'GLACIER_INSTANT_RETRIEVAL',
                        'cost_per_gb': 0.004,
                        'retrieval_ms': 1000
                    },
                    'archive': {
                        'class': 'DEEP_ARCHIVE',
                        'cost_per_gb': 0.00099,
                        'retrieval_hours': 12
                    }
                }

            async def upload_video(
                self,
                device_id: str,
                event_id: str,
                video_data: bytes
            ) -> str:
                """
                Upload video to S3

                Args:
                    device_id: Device identifier
                    event_id: Event identifier
                    video_data: H.264 encoded video

                Returns:
                    S3 URL
                """
                # Generate S3 key (partitioned by date)
                now = datetime.utcnow()
                s3_key = f"videos/{now.year}/{now.month:02d}/{now.day:02d}/{device_id}/{event_id}.mp4"

                # Upload to S3 (STANDARD tier initially)
                self.s3.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=video_data,
                    ContentType='video/mp4',
                    StorageClass='STANDARD',
                    Metadata={
                        'device_id': device_id,
                        'event_id': event_id,
                        'upload_time': now.isoformat()
                    }
                )

                # Generate CloudFront URL for fast delivery
                cloudfront_url = f"https://cdn.example.com/{s3_key}"

                return cloudfront_url

            def configure_lifecycle_policy(self):
                """
                Configure S3 lifecycle policy for automatic tiering

                Rules:
                - 0-7 days: STANDARD (hot, frequent access)
                - 7-30 days: STANDARD_IA (warm, occasional access)
                - 30-90 days: GLACIER_INSTANT (cold, rare access)
                - 90+ days: DEEP_ARCHIVE (archive, very rare access)
                """
                lifecycle_policy = {
                    'Rules': [
                        {
                            'Id': 'tier-to-infrequent-access',
                            'Status': 'Enabled',
                            'Prefix': 'videos/',
                            'Transitions': [
                                {
                                    'Days': 7,
                                    'StorageClass': 'STANDARD_IA'
                                }
                            ]
                        },
                        {
                            'Id': 'tier-to-glacier',
                            'Status': 'Enabled',
                            'Prefix': 'videos/',
                            'Transitions': [
                                {
                                    'Days': 30,
                                    'StorageClass': 'GLACIER_INSTANT_RETRIEVAL'
                                }
                            ]
                        },
                        {
                            'Id': 'tier-to-deep-archive',
                            'Status': 'Enabled',
                            'Prefix': 'videos/',
                            'Transitions': [
                                {
                                    'Days': 90,
                                    'StorageClass': 'DEEP_ARCHIVE'
                                }
                            ]
                        },
                        {
                            'Id': 'delete-old-videos',
                            'Status': 'Enabled',
                            'Prefix': 'videos/',
                            'Expiration': {
                                'Days': 365  # Delete after 1 year
                            }
                        }
                    ]
                }

                self.s3.put_bucket_lifecycle_configuration(
                    Bucket=self.bucket_name,
                    LifecycleConfiguration=lifecycle_policy
                )

            async def get_video_url(
                self,
                event_id: str,
                expiration_seconds: int = 3600
            ) -> str:
                """
                Generate signed URL for video access

                Args:
                    event_id: Event identifier
                    expiration_seconds: URL validity duration

                Returns:
                    Pre-signed S3 URL
                """
                # Query database for S3 key
                s3_key = self._get_s3_key_from_db(event_id)

                # Check storage class
                obj_metadata = self.s3.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                storage_class = obj_metadata.get('StorageClass', 'STANDARD')

                # If in DEEP_ARCHIVE, initiate restore
                if storage_class == 'DEEP_ARCHIVE':
                    self._initiate_restore(s3_key)
                    return None  # Video not immediately available

                # Generate pre-signed URL
                signed_url = self.s3.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': s3_key
                    },
                    ExpiresIn=expiration_seconds
                )

                return signed_url

            def _initiate_restore(self, s3_key: str):
                """Restore video from Deep Archive (12 hours)"""
                self.s3.restore_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    RestoreRequest={
                        'Days': 1,  # Keep restored for 1 day
                        'GlacierJobParameters': {
                            'Tier': 'Bulk'  # Cheapest, 12 hours
                        }
                    }
                )
        ```

        ---

        ## Cost Analysis

        **Monthly storage cost (event-based, 30-day retention):**

        ```
        Daily videos: 5M clips × 5 MB = 25 TB/day
        Retention: 30 days

        Storage breakdown:
        - 0-7 days (hot):    7 × 25 TB = 175 TB @ $0.023/GB = $4,025/month
        - 7-30 days (warm): 23 × 25 TB = 575 TB @ $0.0125/GB = $7,188/month

        Total: $11,213/month for 750 TB

        Per device: $11,213 / 10M = $0.0011/month per device
        ```

        **Monthly cost (24/7 recording, 30-day retention, 20% users):**

        ```
        Daily: 2M devices × 21.6 GB = 43.2 PB/day
        Retention: 30 days = 1.3 EB

        Storage breakdown:
        - 0-7 days:   7 × 43.2 PB = 302 PB @ $0.023/GB = $6.9M/month
        - 7-30 days: 23 × 43.2 PB = 994 PB @ $0.0125/GB = $12.4M/month

        Total: $19.3M/month for 1.3 EB

        Per 24/7 device: $19.3M / 2M = $9.65/month per device

        Revenue (assume $10/month subscription): $20M/month
        Storage cost as % of revenue: 96.5% (too high!)
        ```

        **Optimization needed:** Edge storage for 24/7 recording

        ---

        ## Edge Storage for 24/7 Recording

        ```python
        class EdgeStorageManager:
            """Local storage on doorbell device for 24/7 recording"""

            def __init__(self):
                self.storage_capacity_gb = 128  # 128 GB SD card
                self.retention_days = 7  # 7 days local storage
                self.bitrate = 1_000_000  # 1 Mbps (lower for local)

            def calculate_local_retention(self) -> int:
                """
                Calculate how many days of 24/7 recording fit on device

                Returns:
                    Days of retention
                """
                # Daily storage at 1 Mbps
                daily_gb = (self.bitrate / 8) * 86400 / (1024**3)

                # Days that fit in capacity
                days = self.storage_capacity_gb / daily_gb

                return int(days)

            def get_retention_days(self) -> int:
                """Get actual retention based on capacity"""
                max_days = self.calculate_local_retention()
                return min(max_days, self.retention_days)

        # Example calculation:
        # Bitrate: 1 Mbps
        # Daily storage: 1 Mbps × 86,400s / 8 / 1024^3 = 10.8 GB/day
        # 128 GB capacity: 128 / 10.8 = 11.8 days retention
        ```

        **Cost comparison:**

        | Storage | Cost/month/device | Retention | Notes |
        |---------|------------------|-----------|-------|
        | **Cloud 24/7** | $9.65 | 30 days | Too expensive |
        | **Edge + Cloud** | $1.50 | 7 days local + 30 days cloud events | Cost-effective |
        | **Edge only** | $0 | 7 days local | No cloud backup |

        **Best approach:** Edge storage for 24/7, cloud for events (motion/doorbell)

    === "🔋 Battery Optimization"

        ## The Challenge

        **Problem:** Maximize battery life (target: 6-12 months) while maintaining responsiveness.

        **Power consumption breakdown:**

        | Component | Active Current | Duty Cycle | Average |
        |-----------|---------------|------------|---------|
        | MCU/SoC (Qualcomm QCS605) | 2000 mA | 1% | 20 mA |
        | WiFi module | 300 mA | 5% | 15 mA |
        | Camera sensor | 500 mA | 0.1% | 0.5 mA |
        | Video encoder | 1000 mA | 0.1% | 1 mA |
        | PIR sensor | 0.05 mA | 100% | 0.05 mA |
        | IR LEDs (night) | 200 mA | 10% (nighttime) | 20 mA |
        | Speaker/Mic | 100 mA | 0.01% | 0.01 mA |
        | Sleep mode | 10 mA | ~94% | 10 mA |
        | **Total** | | | **~67 mA** |

        **Battery life:** 5000 mAh / 67 mA ≈ 75 hours ≈ 3 days (terrible!)

        ---

        ## Optimization Strategies

        ### 1. Deep Sleep with PIR Wake

        ```python
        # Firmware (Python-like pseudocode)

        class PowerManager:
            """Aggressive power management for battery life"""

            def __init__(self):
                self.sleep_mode = 'deep'  # deep, light, active
                self.wake_sources = ['pir', 'doorbell_button', 'wifi_packet']

            def enter_deep_sleep(self):
                """
                Enter ultra-low power mode

                Power: ~2 mA (PIR sensor + RTC + wake circuitry)
                Wake latency: ~500ms
                """
                # Power down all peripherals
                self.camera.power_off()
                self.wifi.power_off()
                self.encoder.power_off()
                self.speaker.power_off()

                # Keep only wake sources
                self.pir_sensor.enable()
                self.doorbell_button.enable()
                self.rtc.enable()

                # Enable wake-on-WiFi (magic packet)
                self.wifi.enable_wake_on_packet()

                # Enter deep sleep
                self.soc.enter_deep_sleep()

            def wake_from_sleep(self, wake_source: str):
                """
                Wake from deep sleep

                Args:
                    wake_source: 'pir', 'doorbell', 'wifi'
                """
                # Power up immediately needed components
                self.soc.wake()

                if wake_source == 'pir':
                    # Motion detected - start camera
                    self.camera.power_on()  # 200ms power-on delay
                    self.wifi.power_on()     # 500ms connection time
                    self.encoder.power_on()

                elif wake_source == 'doorbell':
                    # Doorbell pressed - start everything
                    self.camera.power_on()
                    self.wifi.power_on()
                    self.encoder.power_on()
                    self.speaker.power_on()

                elif wake_source == 'wifi':
                    # Remote access request
                    self.wifi.power_on()
        ```

        ### 2. Efficient Video Encoding

        ```python
        class EfficientVideoEncoder:
            """Hardware-accelerated H.264 encoding"""

            def __init__(self):
                self.encoder = HardwareH264Encoder()  # Use SoC encoder
                self.frame_skip_mode = False

                # Encoding profiles
                self.PROFILES = {
                    'high_quality': {
                        'resolution': (1920, 1080),
                        'fps': 30,
                        'bitrate': 4_000_000,
                        'power': 1000  # mA
                    },
                    'balanced': {
                        'resolution': (1280, 720),
                        'fps': 30,
                        'bitrate': 2_000_000,
                        'power': 600  # mA
                    },
                    'battery_saver': {
                        'resolution': (854, 480),
                        'fps': 15,
                        'bitrate': 500_000,
                        'power': 300  # mA
                    }
                }

            def encode_with_battery_optimization(
                self,
                frames: list,
                battery_level: int
            ) -> bytes:
                """
                Encode video with power-aware settings

                Args:
                    frames: List of video frames
                    battery_level: Current battery (0-100%)

                Returns:
                    Encoded H.264 video
                """
                # Choose profile based on battery level
                if battery_level > 50:
                    profile = self.PROFILES['balanced']
                elif battery_level > 20:
                    profile = self.PROFILES['battery_saver']
                else:
                    # Critical battery - aggressive savings
                    profile = self.PROFILES['battery_saver']
                    self.frame_skip_mode = True

                # Configure encoder
                self.encoder.set_resolution(profile['resolution'])
                self.encoder.set_fps(profile['fps'])
                self.encoder.set_bitrate(profile['bitrate'])

                # Skip frames if in battery saver mode
                if self.frame_skip_mode:
                    frames = frames[::2]  # Skip every other frame

                # Hardware encoding (10x more power efficient than software)
                encoded_video = self.encoder.encode(frames)

                return encoded_video
        ```

        ### 3. Intelligent WiFi Management

        ```python
        class WiFiPowerManager:
            """Optimize WiFi power consumption"""

            def __init__(self):
                self.wifi = WiFiModule()
                self.connection_state = 'disconnected'
                self.last_upload_time = None

            def optimize_wifi_power(self):
                """
                Use WiFi power save modes

                Modes:
                - Active: 300 mA (always on)
                - Power Save: 50 mA (listen interval)
                - Disconnected: 0 mA (off)
                """
                # Use 802.11 power save mode
                self.wifi.set_power_save_mode('max_power_save')

                # Listen interval: wake every 300ms to check for packets
                self.wifi.set_listen_interval(300)  # ms

                # Disconnect when idle
                if self._is_idle():
                    self.wifi.disconnect()
                    self.connection_state = 'disconnected'

            def _is_idle(self) -> bool:
                """Check if WiFi can be disconnected"""
                # Disconnect if no activity for 5 minutes
                if not self.last_upload_time:
                    return False

                idle_seconds = (datetime.now() - self.last_upload_time).total_seconds()
                return idle_seconds > 300

            async def batch_uploads(self, clips: list):
                """
                Batch multiple video uploads to save power

                Instead of: Connect → Upload → Disconnect (per clip)
                Do: Connect → Upload all → Disconnect (once)
                """
                # Connect once
                await self.wifi.connect()

                # Upload all clips
                for clip in clips:
                    await self._upload_clip(clip)

                # Disconnect after batch
                await self.wifi.disconnect()
        ```

        ### 4. Adaptive Night Vision

        ```python
        class AdaptiveNightVision:
            """Optimize IR LED power usage"""

            def __init__(self):
                self.ir_leds = IRLEDArray()
                self.light_sensor = AmbientLightSensor()

            def optimize_ir_power(self):
                """
                Adjust IR LED intensity based on ambient light

                Power: 0 mA (day) to 200 mA (full power night)
                """
                # Measure ambient light
                lux = self.light_sensor.read()

                if lux > 10:
                    # Daylight - no IR needed
                    self.ir_leds.set_intensity(0)
                elif lux > 1:
                    # Twilight - low IR
                    self.ir_leds.set_intensity(30)  # 30% → 60 mA
                elif lux > 0.1:
                    # Night - medium IR
                    self.ir_leds.set_intensity(60)  # 60% → 120 mA
                else:
                    # Dark night - full IR
                    self.ir_leds.set_intensity(100)  # 100% → 200 mA
        ```

        ---

        ## Optimized Battery Life

        **After optimizations:**

        | Component | Optimized Average |
        |-----------|------------------|
        | MCU/SoC (deep sleep) | 2 mA |
        | WiFi (power save + batching) | 3 mA |
        | Camera (on-demand) | 0.5 mA |
        | Video encoder (hardware) | 1 mA |
        | PIR sensor | 0.05 mA |
        | IR LEDs (adaptive) | 10 mA |
        | Speaker/Mic | 0.01 mA |
        | Sleep overhead | 2 mA |
        | **Total** | **~18.5 mA** |

        **Battery life:** 5000 mAh / 18.5 mA ≈ 270 hours ≈ **11 days**

        **Still not enough! Need bigger battery:**

        - 10,000 mAh battery: 540 hours ≈ **22.5 days** ≈ 0.75 months
        - 15,000 mAh battery: 810 hours ≈ **34 days** ≈ 1.1 months
        - **Target: 20,000 mAh battery → 45 days → 1.5 months**

        **To reach 6 months:** Need to reduce average to ~4.6 mA (aggressive!)

        **Additional optimizations:**
        - Solar panel: Add 100-200 mA during daytime
        - User configurable: Disable features to save power
        - Adaptive polling: Check for motion less frequently at night

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling from 1M devices to 10M devices.

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Video streaming** | ✅ Yes | WebRTC mesh + TURN relay pools |
    | **Video uploads** | ✅ Yes | Direct S3 upload (pre-signed URLs) |
    | **Motion AI** | 🟡 Moderate | GPU clusters + batch inference |
    | **Push notifications** | 🟢 No | FCM/APNs (managed service) |
    | **CDN delivery** | 🟢 No | CloudFront auto-scales |

    ---

    ## Horizontal Scaling

    ### Direct S3 Upload (Save bandwidth)

    ```python
    class DirectUploadManager:
        """Allow devices to upload directly to S3"""

        def __init__(self):
            self.s3 = boto3.client('s3')
            self.bucket_name = 'doorbell-recordings'

        def generate_upload_url(
            self,
            device_id: str,
            event_id: str
        ) -> dict:
            """
            Generate pre-signed URL for device to upload directly

            Benefits:
            - Reduce API server load (no video proxying)
            - Lower latency (direct to S3)
            - Lower bandwidth costs

            Returns:
                {
                    'upload_url': str,
                    'fields': dict (for POST upload)
                }
            """
            # Generate S3 key
            now = datetime.utcnow()
            s3_key = f"videos/{now.year}/{now.month:02d}/{now.day:02d}/{device_id}/{event_id}.mp4"

            # Generate pre-signed POST URL (valid for 10 minutes)
            response = self.s3.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields={
                    'Content-Type': 'video/mp4',
                    'x-amz-meta-device-id': device_id,
                    'x-amz-meta-event-id': event_id
                },
                Conditions=[
                    ['content-length-range', 1024, 10485760]  # 1KB - 10MB
                ],
                ExpiresIn=600
            )

            return {
                'upload_url': response['url'],
                'fields': response['fields'],
                's3_key': s3_key
            }
    ```

    ---

    ## CDN Strategy

    ```python
    class CDNManager:
        """Manage CloudFront CDN for video delivery"""

        def __init__(self):
            self.cloudfront = boto3.client('cloudfront')
            self.distribution_id = 'E1234567890ABC'

        def configure_cdn(self):
            """
            Configure CloudFront distribution

            Benefits:
            - Global edge caching (reduce latency)
            - Reduce S3 egress costs ($0.09/GB → $0.085/GB)
            - Automatic HTTPS
            """
            distribution_config = {
                'Origins': [
                    {
                        'Id': 's3-doorbell-recordings',
                        'DomainName': 'doorbell-recordings.s3.amazonaws.com',
                        'S3OriginConfig': {
                            'OriginAccessIdentity': 'origin-access-identity/cloudfront/ABCDEFG'
                        }
                    }
                ],
                'DefaultCacheBehavior': {
                    'TargetOriginId': 's3-doorbell-recordings',
                    'ViewerProtocolPolicy': 'redirect-to-https',
                    'AllowedMethods': ['GET', 'HEAD'],
                    'CachedMethods': ['GET', 'HEAD'],
                    'Compress': True,
                    'MinTTL': 0,
                    'DefaultTTL': 86400,  # 1 day
                    'MaxTTL': 31536000    # 1 year
                }
            }

        def invalidate_cache(self, s3_keys: list):
            """Invalidate CDN cache for updated videos"""
            self.cloudfront.create_invalidation(
                DistributionId=self.distribution_id,
                InvalidationBatch={
                    'Paths': {
                        'Quantity': len(s3_keys),
                        'Items': [f'/{key}' for key in s3_keys]
                    },
                    'CallerReference': str(time.time())
                }
            )
    ```

    ---

    ## Batch AI Inference

    ```python
    class BatchAIInference:
        """Batch AI inference for cost optimization"""

        def __init__(self):
            self.batch_size = 32
            self.pending_videos = []

        async def queue_video_for_inference(self, video_url: str):
            """
            Queue video for batch processing

            Instead of: 1 video → 1 GPU inference → 500ms
            Do: 32 videos → 1 GPU batch inference → 2s (16x more efficient)
            """
            self.pending_videos.append(video_url)

            if len(self.pending_videos) >= self.batch_size:
                await self._process_batch()

        async def _process_batch(self):
            """Process batch of videos"""
            batch = self.pending_videos[:self.batch_size]
            self.pending_videos = self.pending_videos[self.batch_size:]

            # Download videos in parallel
            videos = await asyncio.gather(*[
                self._download_video(url) for url in batch
            ])

            # Batch inference (GPU)
            results = self.model.predict_batch(videos)

            # Store results
            await asyncio.gather(*[
                self._store_result(url, result)
                for url, result in zip(batch, results)
            ])
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 10M devices:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (API servers)** | $28,800 (150 instances @ $6.40/day) |
    | **RDS PostgreSQL** | $8,640 (8 shards @ $1,080/month) |
    | **Cassandra (events)** | $21,600 (20 nodes @ $1,080/month) |
    | **Redis Cache** | $4,320 (5 nodes @ $864/month) |
    | **S3 Storage** | $11,213 (750 TB, event-based only) |
    | **CloudFront (CDN)** | $8,500 (100 TB egress @ $0.085/GB) |
    | **GPU (AI inference)** | $14,400 (10 p3.2xlarge @ $1,440/month) |
    | **Total** | **$97,473/month** |

    **Per-device cost:** $97,473 / 10M = **$0.0097/month = $0.12/year per device**

    **Revenue:** Assume $5/month subscription. 10M devices = $50M/month. Infrastructure is 0.19% of revenue.

    ---

    ## Monitoring & Alerting

    ```python
    class DoorbellSystemMonitoring:
        """Monitor system health and performance"""

        def __init__(self):
            self.cloudwatch = boto3.client('cloudwatch')

        def track_motion_event(
            self,
            device_id: str,
            latency_ms: float,
            ai_confidence: float
        ):
            """Track motion detection metrics"""
            self.cloudwatch.put_metric_data(
                Namespace='SmartDoorbell',
                MetricData=[
                    {
                        'MetricName': 'MotionDetectionLatency',
                        'Value': latency_ms,
                        'Unit': 'Milliseconds',
                        'Dimensions': [
                            {'Name': 'Service', 'Value': 'MotionDetection'}
                        ]
                    },
                    {
                        'MetricName': 'AIConfidence',
                        'Value': ai_confidence,
                        'Unit': 'None',
                        'Dimensions': [
                            {'Name': 'Service', 'Value': 'AIDetection'}
                        ]
                    }
                ]
            )

        def create_alarms(self):
            """Create CloudWatch alarms"""
            # High latency alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName='Doorbell-HighLatency',
                MetricName='MotionDetectionLatency',
                Namespace='SmartDoorbell',
                Statistic='Average',
                Period=300,
                EvaluationPeriods=2,
                Threshold=3000,  # 3 seconds
                ComparisonOperator='GreaterThanThreshold',
                AlarmActions=['arn:aws:sns:us-east-1:123456789:alerts']
            )

            # Low AI confidence alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName='Doorbell-LowAIConfidence',
                MetricName='AIConfidence',
                Namespace='SmartDoorbell',
                Statistic='Average',
                Period=300,
                EvaluationPeriods=2,
                Threshold=0.7,
                ComparisonOperator='LessThanThreshold',
                AlarmActions=['arn:aws:sns:us-east-1:123456789:alerts']
            )
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Three-tier motion detection** - PIR sensor + edge CV + cloud AI (balance power/accuracy)
    2. **WebRTC for live streaming** - Sub-3s latency for real-time communication
    3. **Direct S3 upload** - Devices upload directly to save bandwidth
    4. **S3 tiered storage** - Automatic lifecycle to reduce costs (hot → warm → cold)
    5. **Deep sleep with PIR wake** - Battery optimization (2 mA idle)
    6. **Edge storage for 24/7** - Local SD card to avoid cloud costs
    7. **Adaptive bitrate** - Adjust quality based on network conditions
    8. **CDN distribution** - CloudFront for fast global video delivery

    ---

    ## Interview Tips

    ✅ **Start with latency requirement** - Emphasize < 3s for live streaming

    ✅ **Discuss battery optimization deeply** - Show specific power calculations

    ✅ **Three-tier motion detection** - PIR → edge CV → cloud AI progression

    ✅ **Storage cost analysis** - Compare event-based vs 24/7 recording costs

    ✅ **Video streaming protocols** - WebRTC vs RTSP vs HLS trade-offs

    ✅ **Edge processing** - Why run CV on device (save bandwidth, faster response)

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle poor WiFi?"** | Adaptive bitrate (1080p → 480p), retry logic, local buffering, offline mode (view recordings later) |
    | **"What if battery dies?"** | Low battery alerts at 20%, wired power option, solar panel accessory |
    | **"Privacy concerns (camera)?"** | End-to-end encryption, user-controlled recording, LED indicator when streaming, GDPR compliance |
    | **"False motion alerts?"** | AI filters (person vs tree), detection zones (ignore street), cooldown period (60s between alerts) |
    | **"How to scale to 100M devices?"** | Shard databases, CDN for videos, batch AI inference, direct S3 uploads |
    | **"Video quality vs bandwidth?"** | Adaptive bitrate (WebRTC), H.265 codec (better compression), resolution presets (1080p/720p/480p) |
    | **"Night vision implementation?"** | IR LEDs (850nm), IR-cut filter removal, adaptive LED intensity based on ambient light |
    | **"Two-way audio latency?"** | WebRTC audio track (< 500ms), Opus codec (low bitrate), echo cancellation |

    ---

    ## Real-World Examples

    ### Ring Video Doorbell

    - **Architecture:** WiFi-connected, cloud-first design
    - **Battery:** 5,000-10,000 mAh, 6-12 months
    - **Features:** 1080p video, motion detection, two-way audio, night vision
    - **Storage:** Cloud recording with Ring Protect subscription ($3-10/month)
    - **AI:** Person detection, package detection (Ring Protect Plan)

    ### Nest Hello (Google)

    - **Architecture:** Wired power (no battery), always-on
    - **Features:** 24/7 recording, facial recognition, HDR video, 160° wide angle
    - **Storage:** Nest Aware subscription (10-30 days)
    - **AI:** Familiar face detection, activity zones, sound detection

    ### Arlo Video Doorbell

    - **Architecture:** Battery or wired, local storage option
    - **Battery:** 6 months typical
    - **Features:** Motion detection, night vision, two-way audio, siren
    - **Storage:** Local (USB hub) or cloud (Arlo Smart)

    ---

    ## Security Best Practices

    1. **Encryption everywhere** - E2E encryption for video streams
    2. **Signed URLs** - Pre-signed S3 URLs with expiration
    3. **Access control** - Only owner can view videos
    4. **Audit logging** - Log all video access
    5. **Privacy zones** - User can mask areas (neighbor's window)
    6. **LED indicator** - Show when camera is recording
    7. **Regular firmware updates** - Security patches

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Ring, Nest Hello, Arlo, Eufy, Blink

---

*Master this problem and you'll be ready for: IoT video systems, home security platforms, live streaming services*
