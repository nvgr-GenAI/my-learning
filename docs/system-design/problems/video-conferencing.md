# Design Video Conferencing (Zoom)

A real-time video conferencing platform that enables multi-party video calls, audio communication, screen sharing, chat, and recording capabilities with low latency and high quality.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 300M daily meeting participants, 10M concurrent meetings, 40 participants per meeting (avg) |
| **Key Challenges** | WebRTC implementation, SFU/MCU routing, <150ms latency, NAT traversal, adaptive bitrate, network resilience |
| **Core Concepts** | WebRTC, SFU (Selective Forwarding Unit), STUN/TURN servers, Opus codec, VP8/VP9, simulcast, jitter buffer |
| **Companies** | Zoom, Google Meet, Microsoft Teams, WebEx, Discord, Skype |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Video Call** | Real-time video communication (up to 1000 participants) | P0 (Must have) |
    | **Audio Call** | High-quality audio with noise suppression | P0 (Must have) |
    | **Screen Share** | Share screen/application window | P0 (Must have) |
    | **In-Meeting Chat** | Text chat during meetings | P0 (Must have) |
    | **Participant Controls** | Mute/unmute, video on/off, hand raise | P0 (Must have) |
    | **Recording** | Record meetings to cloud/local storage | P1 (Should have) |
    | **Breakout Rooms** | Split participants into smaller groups | P1 (Should have) |
    | **Virtual Backgrounds** | Replace/blur background | P2 (Nice to have) |
    | **Live Transcription** | Real-time speech-to-text | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - End-to-end encryption (E2EE) for all calls
    - Advanced AI features (noise cancellation ML models)
    - Webinar mode (>10K participants)
    - Marketplace/app integrations
    - Payment/billing system

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency** | < 150ms end-to-end | Critical for natural conversation (>300ms breaks flow) |
    | **Availability** | 99.9% uptime | Business meetings require high reliability |
    | **Video Quality** | 720p @ 30fps (HD), 1080p for screen share | Good quality without excessive bandwidth |
    | **Audio Quality** | 48kHz sampling, Opus codec, <50ms latency | Crystal clear audio is more critical than video |
    | **Bandwidth** | 1.5-3 Mbps per participant (adaptive) | Must work on residential internet |
    | **Packet Loss Tolerance** | < 5% acceptable, functional up to 10% | Networks are imperfect, must be resilient |
    | **Connection Time** | < 3 seconds to join meeting | Quick joining improves UX |
    | **Scalability** | 10M concurrent meetings, 300M daily participants | Handle global scale |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users: 300M meeting participants
    Concurrent meetings: 10M meetings
    Average meeting size: 30 participants (median: 5, p95: 100)
    Average meeting duration: 45 minutes

    Concurrent participants:
    - Total concurrent users: 10M meetings × 30 avg = 300M concurrent participants
    - Peak concurrent: 1.5x average = 450M concurrent (business hours overlap)

    Media streams:
    - Video streams per participant: 1 outbound, 3-49 inbound (gallery view)
    - Audio streams: Always active for all participants
    - Screen share: 1 additional stream when active (30% of meetings)

    Bandwidth per participant:
    - Video send: 1.5 Mbps (720p @ 30fps)
    - Video receive: 3-6 Mbps (multiple streams, adaptive)
    - Audio send/receive: 50-100 Kbps (Opus codec)
    - Screen share: 2-4 Mbps (1080p)
    - Signaling: ~10 Kbps (negligible)

    Total bandwidth:
    - Average per participant: 5 Mbps (bidirectional)
    - Total concurrent: 300M × 5 Mbps = 1,500 Pbps (1.5 Exabits/sec)
    - Through media servers (SFU): ~40% of total = 600 Pbps

    Signaling QPS:
    - Join/leave events: 300M participants / 45 min avg = 111K joins/sec
    - State updates (mute/unmute, video toggle): 3x = 333K events/sec
    - Chat messages: 50K msg/sec
    - Total signaling QPS: ~500K events/sec
    ```

    ### Storage Estimates

    ```
    Meeting metadata:
    - Meeting ID, participants, start/end time: 5 KB per meeting
    - Daily meetings: 10M concurrent × 2.5 (over 24h) = 25M meetings/day
    - Daily storage: 25M × 5 KB = 125 GB/day
    - 1 year: 125 GB × 365 = 45 TB

    Recording storage:
    - 20% of meetings recorded (5M recordings/day)
    - Average recording: 45 min × 2 Mbps (compressed) = 675 MB
    - Daily: 5M × 675 MB = 3.4 PB/day
    - 1 year: 3.4 PB × 365 = 1.24 EB (exabytes)
    - With deduplication/cleanup (90 day retention): ~112 PB

    Chat history:
    - 50K messages/sec × 86,400 sec = 4.3B messages/day
    - Message size: 1 KB (text + metadata)
    - Daily: 4.3B × 1 KB = 4.3 TB/day
    - 1 year: 4.3 TB × 365 = 1.57 PB

    Total: 45 TB (metadata) + 112 PB (recordings) + 1.57 PB (chat) ≈ 114 PB
    ```

    ### Bandwidth Estimates

    ```
    Media ingress (to SFU servers):
    - 300M participants × 1.5 Mbps (video) = 450 Tbps
    - 300M participants × 100 Kbps (audio) = 30 Tbps
    - Screen share (30% of meetings): 30M × 2 Mbps = 60 Tbps
    - Total ingress: 540 Tbps

    Media egress (from SFU to participants):
    - 300M participants × 4 Mbps (mixed video) = 1,200 Tbps
    - Audio: 30 Tbps
    - Screen share: 60 Tbps
    - Total egress: 1,290 Tbps

    Total bidirectional: 540 + 1,290 = 1,830 Tbps (1.83 Pbps)

    Note: This is distributed across global data centers, not single location.
    Per DC (20 regions): ~90 Tbps per region
    ```

    ### Memory Estimates

    ```
    Active meeting state:
    - 10M meetings × 100 KB (participants, state) = 1 TB

    Participant session state:
    - 300M participants × 50 KB (connection, streams) = 15 TB

    WebRTC peer connections (SFU):
    - 300M participants × 2 KB (connection state) = 600 GB

    Signaling server cache:
    - Room states, presence: 500 GB

    Jitter buffers (audio/video):
    - 300M participants × 1 MB (200ms buffer) = 300 TB

    Total memory: 1 TB + 15 TB + 600 GB + 500 GB + 300 TB ≈ 317 TB
    ```

    ---

    ## Key Assumptions

    1. Average meeting: 30 participants, 45 minutes duration
    2. Median meeting: 5 participants (most are small team calls)
    3. 95th percentile: 100 participants (large company meetings)
    4. Gallery view: Show 4-9 video streams simultaneously
    5. Network conditions: 70% good, 20% moderate, 10% poor
    6. Recording: 20% of meetings, 90-day retention
    7. Peak usage: Business hours (9am-5pm in each timezone)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **SFU architecture:** Selective Forwarding Unit for scalability (not MCU or P2P)
    2. **WebRTC for media:** Industry standard for real-time communication
    3. **Adaptive bitrate:** Adjust quality based on bandwidth and CPU
    4. **Network resilience:** Handle packet loss, jitter, and varying bandwidth
    5. **Low latency first:** Prioritize latency over perfect quality
    6. **Global edge network:** Media servers close to users

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Web[Web Browser<br/>WebRTC]
            Mobile[Mobile App<br/>Native WebRTC]
            Desktop[Desktop App<br/>Electron/Native]
        end

        subgraph "Edge Layer"
            TURN[TURN Servers<br/>NAT traversal]
            STUN[STUN Servers<br/>IP discovery]
        end

        subgraph "Signaling Layer"
            LB[Load Balancer<br/>WebSocket routing]
            Signal[Signaling Server<br/>Room management]
            Presence[Presence Service<br/>Online status]
        end

        subgraph "Media Layer"
            SFU[SFU Servers<br/>Selective Forwarding]
            Mixer[Audio Mixer<br/>Large meetings]
            Transcode[Transcoder<br/>Format conversion]
        end

        subgraph "Processing Layer"
            Record[Recording Service<br/>Cloud recording]
            AI[AI Services<br/>Noise suppression<br/>Virtual background]
            Transcribe[Transcription<br/>Speech-to-text]
        end

        subgraph "Application Services"
            API[REST API<br/>Meeting management]
            Chat[Chat Service<br/>In-meeting chat]
            Auth[Auth Service<br/>JWT tokens]
            Analytics[Analytics<br/>Quality metrics]
        end

        subgraph "Storage"
            Postgres[(PostgreSQL<br/>Meeting metadata)]
            Redis[(Redis<br/>Session state<br/>Presence)]
            S3[(S3/GCS<br/>Recordings<br/>Chat logs)]
            Cassandra[(Cassandra<br/>Analytics data)]
        end

        subgraph "Message Queue"
            Kafka[Kafka<br/>Event streaming]
        end

        Web --> STUN
        Mobile --> STUN
        Desktop --> STUN

        Web --> TURN
        Mobile --> TURN
        Desktop --> TURN

        Web --> LB
        Mobile --> LB
        Desktop --> LB

        LB --> Signal
        Signal --> Presence
        Signal --> Redis

        Web --> SFU
        Mobile --> SFU
        Desktop --> SFU

        SFU --> Mixer
        SFU --> Transcode
        SFU --> Record
        SFU --> AI

        Signal --> Kafka
        SFU --> Kafka

        Kafka --> Analytics
        Kafka --> Transcribe

        API --> Postgres
        API --> Redis
        Signal --> Postgres

        Record --> S3
        Chat --> S3
        Chat --> Kafka

        Analytics --> Cassandra

        Auth --> API
        Auth --> Signal

        style STUN fill:#e8f5e9
        style TURN fill:#e8f5e9
        style LB fill:#e1f5ff
        style Signal fill:#e1f5ff
        style SFU fill:#fff4e1
        style Mixer fill:#fff4e1
        style Redis fill:#ffe1e1
        style Postgres fill:#ffe1e1
        style S3 fill:#f3e5f5
        style Kafka fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **SFU (not MCU)** | Scales to 100+ participants, low latency (<150ms), each client gets original quality | MCU (high CPU, mixes all streams), P2P (only works for 2-4 participants) |
    | **WebRTC** | Industry standard, built-in browsers, handles NAT traversal, adaptive bitrate | Custom protocol (reinventing wheel, no browser support), RTMP (high latency) |
    | **TURN servers** | 10-15% of users behind restrictive NATs/firewalls need relay | Direct P2P (fails for corporate networks), VPN (bad UX) |
    | **Redis for state** | Fast session state (< 1ms), pub/sub for signaling, participant presence | Database (too slow), in-memory on server (not distributed) |
    | **Kafka for events** | Reliable event streaming for analytics, recording, transcription | Direct calls (no replay), RabbitMQ (lower throughput) |
    | **PostgreSQL** | Meeting metadata, user accounts, structured data with ACID | NoSQL (need transactions for billing), Cassandra (overkill for metadata) |

    **Key Trade-off:** We chose **SFU over MCU** for scalability and quality. MCU mixes all streams (high CPU cost), SFU just forwards (low cost, better quality).

    ---

    ## API Design

    ### 1. Create Meeting

    **Request:**
    ```http
    POST /api/v1/meetings
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "topic": "Q1 Planning Meeting",
      "start_time": "2026-02-03T10:00:00Z",
      "duration": 60,                      // minutes
      "password": "abc123",                 // Optional
      "settings": {
        "video": true,
        "audio": true,
        "screen_share": true,
        "recording": "cloud",               // "cloud", "local", "none"
        "waiting_room": true,
        "mute_on_entry": false
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "meeting_id": "123-456-789",
      "meeting_url": "https://zoom.us/j/123456789",
      "join_url": "https://zoom.us/j/123456789?pwd=xyz",
      "password": "abc123",
      "host_key": "987654",
      "created_at": "2026-02-02T15:00:00Z",
      "settings": {
        "video": true,
        "audio": true,
        "screen_share": true,
        "recording": "cloud",
        "waiting_room": true
      }
    }
    ```

    ---

    ### 2. Join Meeting (Signaling)

    **WebSocket Connection:**
    ```javascript
    // Client connects to signaling server
    const ws = new WebSocket('wss://signal.zoom.us/meetings/123456789');

    // Authenticate
    ws.send(JSON.stringify({
      type: 'join',
      meeting_id: '123456789',
      user_id: 'user_abc',
      display_name: 'John Doe',
      token: '<jwt_token>',
      capabilities: {
        video: true,
        audio: true,
        screen_share: true
      }
    }));
    ```

    **Server Response:**
    ```json
    {
      "type": "joined",
      "participant_id": "part_123",
      "sfu_server": "sfu-us-west-1.zoom.us:5000",
      "ice_servers": [
        {
          "urls": "stun:stun.zoom.us:3478"
        },
        {
          "urls": "turn:turn.zoom.us:3478",
          "username": "user123",
          "credential": "temp_cred"
        }
      ],
      "participants": [
        {
          "participant_id": "part_456",
          "display_name": "Jane Smith",
          "video": true,
          "audio": false,
          "is_host": true
        }
      ]
    }
    ```

    **Design Notes:**

    - WebSocket for signaling (join, leave, mute, etc.)
    - Return ICE servers (STUN/TURN) for NAT traversal
    - Return SFU server address for media routing
    - List current participants for UI rendering

    ---

    ### 3. WebRTC Media Exchange

    **SDP Offer (Client → SFU):**
    ```javascript
    // Create WebRTC peer connection
    const pc = new RTCPeerConnection({
      iceServers: ice_servers
    });

    // Add local media tracks
    localStream.getTracks().forEach(track => {
      pc.addTrack(track, localStream);
    });

    // Create and send offer
    const offer = await pc.createOffer({
      offerToReceiveAudio: true,
      offerToReceiveVideo: true
    });
    await pc.setLocalDescription(offer);

    ws.send(JSON.stringify({
      type: 'offer',
      sdp: offer.sdp
    }));
    ```

    **SDP Answer (SFU → Client):**
    ```json
    {
      "type": "answer",
      "sdp": "v=0\r\no=- 123456 2 IN IP4 0.0.0.0\r\n..."
    }
    ```

    **ICE Candidates (Bidirectional):**
    ```json
    {
      "type": "ice-candidate",
      "candidate": "candidate:1 1 UDP 2130706431 192.168.1.100 54321 typ host",
      "sdpMLineIndex": 0,
      "sdpMid": "audio"
    }
    ```

    ---

    ### 4. In-Meeting Actions

    **Mute/Unmute Audio:**
    ```javascript
    ws.send(JSON.stringify({
      type: 'audio-state',
      participant_id: 'part_123',
      enabled: false  // muted
    }));
    ```

    **Start Screen Share:**
    ```javascript
    // Get screen share stream
    const screenStream = await navigator.mediaDevices.getDisplayMedia({
      video: { width: 1920, height: 1080, frameRate: 30 }
    });

    // Add to existing peer connection
    const screenTrack = screenStream.getVideoTracks()[0];
    const sender = pc.addTrack(screenTrack, screenStream);

    ws.send(JSON.stringify({
      type: 'screen-share-start',
      participant_id: 'part_123'
    }));
    ```

    ---

    ## Database Schema

    ### Meetings (PostgreSQL)

    ```sql
    -- Meetings table
    CREATE TABLE meetings (
        meeting_id VARCHAR(20) PRIMARY KEY,
        host_user_id BIGINT NOT NULL,
        topic VARCHAR(255),
        password_hash VARCHAR(255),
        start_time TIMESTAMP,
        duration_minutes INT,
        status VARCHAR(20), -- scheduled, in_progress, ended
        settings JSONB,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_host_user (host_user_id),
        INDEX idx_start_time (start_time),
        INDEX idx_status (status)
    );

    -- Participants table
    CREATE TABLE participants (
        participant_id VARCHAR(50) PRIMARY KEY,
        meeting_id VARCHAR(20) NOT NULL,
        user_id BIGINT,
        display_name VARCHAR(100),
        join_time TIMESTAMP,
        leave_time TIMESTAMP,
        duration_seconds INT,
        is_host BOOLEAN DEFAULT FALSE,
        recording_consent BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id),
        INDEX idx_meeting_id (meeting_id),
        INDEX idx_user_id (user_id)
    );

    -- Meeting events (for analytics)
    CREATE TABLE meeting_events (
        event_id BIGSERIAL PRIMARY KEY,
        meeting_id VARCHAR(20),
        participant_id VARCHAR(50),
        event_type VARCHAR(50), -- join, leave, mute, unmute, screen_share
        event_data JSONB,
        timestamp TIMESTAMP DEFAULT NOW(),
        INDEX idx_meeting_id (meeting_id),
        INDEX idx_timestamp (timestamp)
    ) PARTITION BY RANGE (timestamp);
    ```

    ---

    ### Session State (Redis)

    ```redis
    # Active meeting state
    HSET meeting:123456789
      host_id "user_abc"
      participant_count 30
      start_time "2026-02-02T10:00:00Z"
      sfu_server "sfu-us-west-1"
      recording "true"

    # Participant state
    HSET participant:part_123
      meeting_id "123456789"
      display_name "John Doe"
      audio "true"
      video "true"
      screen_share "false"
      join_time "2026-02-02T10:05:00Z"

    # Participant list (for quick lookup)
    SADD meeting:123456789:participants "part_123" "part_456" "part_789"

    # Presence (online status)
    SETEX presence:user_abc 60 "online"  # TTL 60 seconds, refresh every 30s

    # SFU load balancing
    HSET sfu:us-west-1
      active_meetings 1200
      active_participants 35000
      cpu_usage 65
      bandwidth_gbps 45
    ```

    ---

    ### Recordings (S3/GCS)

    ```
    s3://zoom-recordings/
    ├── 2026/
    │   ├── 02/
    │   │   ├── 02/
    │   │   │   ├── meeting_123456789_20260202_100000.mp4
    │   │   │   ├── meeting_123456789_20260202_100000_audio.m4a
    │   │   │   ├── meeting_123456789_20260202_100000_chat.txt
    │   │   │   └── meeting_123456789_20260202_100000_transcript.vtt
    ```

    **Metadata:**
    ```sql
    CREATE TABLE recordings (
        recording_id VARCHAR(50) PRIMARY KEY,
        meeting_id VARCHAR(20) NOT NULL,
        file_path TEXT,
        file_size_bytes BIGINT,
        duration_seconds INT,
        format VARCHAR(20), -- mp4, m4a, webm
        status VARCHAR(20), -- processing, completed, failed
        created_at TIMESTAMP DEFAULT NOW(),
        FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id),
        INDEX idx_meeting_id (meeting_id)
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Join Meeting Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant SignalServer
        participant Redis
        participant STUN
        participant TURN
        participant SFU
        participant Postgres

        Client->>SignalServer: WebSocket: JOIN meeting
        SignalServer->>SignalServer: Authenticate JWT
        SignalServer->>Redis: Check meeting exists
        Redis-->>SignalServer: Meeting state

        SignalServer->>Redis: Add participant to meeting
        SignalServer->>Postgres: Log join event

        SignalServer->>SignalServer: Select least-loaded SFU
        SignalServer->>Redis: Update SFU load

        SignalServer-->>Client: JOINED (SFU address, ICE servers, participants)

        Client->>STUN: STUN request (discover public IP)
        STUN-->>Client: Your public IP:port

        Client->>SFU: WebRTC Offer (SDP)
        SFU-->>Client: WebRTC Answer (SDP)

        Client->>SFU: ICE candidates
        SFU-->>Client: ICE candidates

        alt Direct connection succeeds
            Client->>SFU: DTLS handshake (direct)
            Client->>SFU: SRTP media (audio/video)
        else Direct fails (NAT/firewall)
            Client->>TURN: Allocate relay
            TURN-->>Client: Relay address
            Client->>TURN: SRTP media
            TURN->>SFU: Relay media
        end

        SFU->>SignalServer: Participant connected
        SignalServer->>Client: Broadcast: New participant joined
    ```

    **Flow Explanation:**

    1. **WebSocket signaling** - Client connects to signaling server for control messages
    2. **Authenticate** - Verify JWT token, check meeting permissions
    3. **SFU selection** - Choose least-loaded SFU server in nearest region
    4. **ICE negotiation** - STUN for IP discovery, establish peer connection
    5. **Fallback to TURN** - If direct connection fails, use TURN relay (10-15% of users)
    6. **Media streaming** - Once connected, stream audio/video via SRTP

    ---

    ### Media Routing Flow (SFU)

    ```mermaid
    sequenceDiagram
        participant Alice
        participant SFU
        participant Bob
        participant Charlie
        participant Recording

        Alice->>SFU: Video stream (720p, 1.5 Mbps)
        Alice->>SFU: Audio stream (Opus, 100 Kbps)

        SFU->>SFU: Decode video to identify simulcast layers
        SFU->>SFU: Buffer 200ms for jitter

        Bob->>SFU: Request: Alice's video (360p - slow network)
        SFU->>SFU: Transcode 720p → 360p OR select simulcast layer
        SFU->>Bob: Alice video (360p, 600 Kbps)
        SFU->>Bob: Alice audio (Opus, 100 Kbps)

        Charlie->>SFU: Request: Alice's video (720p - good network)
        SFU->>Charlie: Alice video (720p, 1.5 Mbps)
        SFU->>Charlie: Alice audio (Opus, 100 Kbps)

        alt Recording enabled
            SFU->>Recording: Mixed video + audio
            Recording->>Recording: Encode to MP4
            Recording->>S3: Upload recording
        end

        Note over SFU: SFU forwards, doesn't decode/mix<br/>(except for recording/transcoding)
    ```

    **Flow Explanation:**

    1. **Upload once** - Each participant uploads 1 video + 1 audio stream
    2. **Simulcast** - Send multiple quality layers (720p, 360p, 180p)
    3. **SFU forwards** - SFU routes appropriate quality to each receiver
    4. **Adaptive bitrate** - Downgrade quality if network degrades
    5. **Recording** - SFU sends copy to recording service for MP4 encoding

    ---

    ### Screen Share Flow

    ```mermaid
    sequenceDiagram
        participant Alice
        participant SignalServer
        participant SFU
        participant Bob
        participant Charlie

        Alice->>Alice: getDisplayMedia() - select screen/window
        Alice->>SignalServer: START_SCREEN_SHARE
        SignalServer->>SignalServer: Check permission (host or allowed)
        SignalServer-->>Alice: APPROVED

        Alice->>SFU: Add screen track to peer connection
        Alice->>SFU: Screen stream (1080p, 2-4 Mbps)

        SignalServer->>Bob: SCREEN_SHARE_STARTED (Alice)
        SignalServer->>Charlie: SCREEN_SHARE_STARTED (Alice)

        Bob->>SFU: Subscribe to Alice's screen track
        SFU->>Bob: Screen stream (1080p)

        Charlie->>SFU: Subscribe to Alice's screen track
        SFU->>Charlie: Screen stream (1080p)

        Note over SFU: Screen share gets priority bandwidth<br/>May downgrade video quality to preserve screen

        Alice->>SignalServer: STOP_SCREEN_SHARE
        SignalServer->>Bob: SCREEN_SHARE_STOPPED (Alice)
        SignalServer->>Charlie: SCREEN_SHARE_STOPPED (Alice)
        SFU->>SFU: Remove screen track
    ```

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical video conferencing subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **WebRTC & SFU** | How to route media for 100+ participants? | SFU architecture with simulcast, selective forwarding |
    | **NAT Traversal** | How to connect through firewalls/NATs? | STUN/TURN servers, ICE framework |
    | **Quality Adaptation** | How to handle poor networks? | Adaptive bitrate, simulcast, packet loss recovery |
    | **Scalability** | How to scale to 10M concurrent meetings? | Regional SFU clusters, load balancing, edge network |

    ---

    === "🎥 WebRTC & SFU Architecture"

        ## The Challenge

        **Problem:** 100 participants in meeting = 100×99 = 9,900 peer connections (mesh P2P). Impossible!

        **Approaches:**

        | Approach | How it works | Pros | Cons | Use Case |
        |----------|--------------|------|------|----------|
        | **Mesh P2P** | Every peer connects to every other peer | Simple, no server | Doesn't scale (>4 participants) | 1-on-1 calls |
        | **MCU (Multipoint Control Unit)** | Server mixes all streams into one | Bandwidth efficient for clients | High CPU cost, fixed layouts, added latency | Legacy systems |
        | **SFU (Selective Forwarding Unit)** | Server forwards streams without mixing | Scalable, low latency, flexible layouts | Higher client bandwidth | **Modern systems (Zoom, Meet)** |

        ---

        ## SFU Architecture

        **Concept:** Each participant sends 1 stream to SFU, receives N streams from SFU.

        ```
        Participant uploads: 1 video + 1 audio (1.6 Mbps)
        Participant downloads: N videos + N audio (N × 500 Kbps avg)

        For 10 participants:
        - Upload: 1.6 Mbps
        - Download: 9 × 500 Kbps = 4.5 Mbps
        - Total: 6.1 Mbps (feasible on residential internet)
        ```

        **Benefits:**

        - **Scalable:** SFU CPU is low (just forwarding packets)
        - **Low latency:** No encoding/decoding, just forwarding (<10ms overhead)
        - **Quality control:** Each receiver gets quality based on their bandwidth
        - **Flexible layouts:** Client decides which streams to render

        ---

        ## SFU Implementation

        ```python
        import asyncio
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from collections import defaultdict

        class SFUServer:
            """
            Selective Forwarding Unit for video conferencing

            Routes media streams without transcoding (except for bandwidth adaptation)
            """

            def __init__(self):
                # meeting_id -> {participant_id -> peer_connection}
                self.meetings = defaultdict(dict)

                # meeting_id -> {participant_id -> {track_id -> MediaStreamTrack}}
                self.tracks = defaultdict(lambda: defaultdict(dict))

                # Statistics
                self.stats = {
                    'active_meetings': 0,
                    'active_participants': 0,
                    'bandwidth_mbps': 0
                }

            async def join_meeting(self, meeting_id: str, participant_id: str) -> RTCPeerConnection:
                """
                Create WebRTC peer connection for new participant

                Args:
                    meeting_id: Meeting identifier
                    participant_id: Participant identifier

                Returns:
                    RTCPeerConnection for media exchange
                """
                pc = RTCPeerConnection()
                self.meetings[meeting_id][participant_id] = pc

                # Handle incoming tracks (video/audio from participant)
                @pc.on('track')
                async def on_track(track):
                    logger.info(f"Received {track.kind} track from {participant_id}")

                    # Store track for forwarding to other participants
                    self.tracks[meeting_id][participant_id][track.id] = track

                    # Forward this track to all other participants
                    await self._forward_track_to_participants(
                        meeting_id,
                        participant_id,
                        track
                    )

                    # Handle track end
                    @track.on('ended')
                    async def on_ended():
                        logger.info(f"Track {track.id} ended")
                        del self.tracks[meeting_id][participant_id][track.id]

                # Handle connection state changes
                @pc.on('connectionstatechange')
                async def on_connection_state_change():
                    logger.info(f"Connection state: {pc.connectionState}")

                    if pc.connectionState == 'failed':
                        await self.leave_meeting(meeting_id, participant_id)

                self.stats['active_participants'] += 1
                if len(self.meetings[meeting_id]) == 1:
                    self.stats['active_meetings'] += 1

                return pc

            async def _forward_track_to_participants(
                self,
                meeting_id: str,
                sender_id: str,
                track
            ):
                """
                Forward track from one participant to all others in meeting

                This is the core SFU logic: selective forwarding
                """
                meeting_participants = self.meetings[meeting_id]

                for participant_id, pc in meeting_participants.items():
                    # Don't send back to sender
                    if participant_id == sender_id:
                        continue

                    # Check if participant wants this track (based on subscription)
                    if not self._should_forward_track(meeting_id, participant_id, track):
                        continue

                    # Add track to peer connection
                    try:
                        pc.addTrack(track)
                        logger.info(f"Forwarding {track.kind} from {sender_id} to {participant_id}")
                    except Exception as e:
                        logger.error(f"Failed to forward track: {e}")

            def _should_forward_track(
                self,
                meeting_id: str,
                participant_id: str,
                track
            ) -> bool:
                """
                Determine if track should be forwarded to participant

                Optimization: Don't forward if participant is in gallery view
                and this track is not in visible grid
                """
                # Get participant's subscription preferences
                subscriptions = self._get_subscriptions(meeting_id, participant_id)

                # Gallery view: only send tracks for visible participants
                if subscriptions.get('layout') == 'gallery':
                    visible_participants = subscriptions.get('visible_participants', [])
                    if len(visible_participants) > 0:
                        # Only forward if track sender is in visible list
                        return self._get_track_sender(track) in visible_participants

                # Speaker view: only send active speaker + screen share
                elif subscriptions.get('layout') == 'speaker':
                    active_speaker = self._get_active_speaker(meeting_id)
                    track_sender = self._get_track_sender(track)

                    if track.kind == 'video':
                        # Only forward active speaker or screen share
                        return (track_sender == active_speaker or
                                self._is_screen_share(track))

                # Audio: always forward all audio tracks
                if track.kind == 'audio':
                    return True

                return True  # Default: forward all tracks

            async def handle_sdp_offer(
                self,
                meeting_id: str,
                participant_id: str,
                offer: RTCSessionDescription
            ) -> RTCSessionDescription:
                """
                Handle WebRTC SDP offer from client

                Args:
                    meeting_id: Meeting identifier
                    participant_id: Participant identifier
                    offer: SDP offer from client

                Returns:
                    SDP answer to send back to client
                """
                pc = await self.join_meeting(meeting_id, participant_id)

                # Set remote description (client's offer)
                await pc.setRemoteDescription(offer)

                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                logger.info(f"Created SDP answer for {participant_id}")

                return pc.localDescription

            async def leave_meeting(self, meeting_id: str, participant_id: str):
                """
                Handle participant leaving meeting
                """
                if participant_id in self.meetings[meeting_id]:
                    pc = self.meetings[meeting_id][participant_id]
                    await pc.close()

                    del self.meetings[meeting_id][participant_id]
                    del self.tracks[meeting_id][participant_id]

                    self.stats['active_participants'] -= 1

                    # Remove meeting if empty
                    if len(self.meetings[meeting_id]) == 0:
                        del self.meetings[meeting_id]
                        self.stats['active_meetings'] -= 1

                    logger.info(f"Participant {participant_id} left meeting {meeting_id}")

            def get_stats(self) -> dict:
                """Get SFU statistics"""
                return {
                    'active_meetings': self.stats['active_meetings'],
                    'active_participants': self.stats['active_participants'],
                    'bandwidth_mbps': self._calculate_bandwidth(),
                    'cpu_usage': self._get_cpu_usage()
                }
        ```

        ---

        ## Simulcast

        **Problem:** Participants have different bandwidth (mobile 3G vs fiber). Sending 720p to everyone wastes bandwidth.

        **Solution:** Send multiple quality layers simultaneously (simulcast).

        ```javascript
        // Client: Send 3 quality layers
        const pc = new RTCPeerConnection();

        pc.addTransceiver('video', {
          direction: 'sendrecv',
          sendEncodings: [
            { rid: 'high', maxBitrate: 1500000 },  // 720p @ 30fps
            { rid: 'medium', maxBitrate: 600000 }, // 360p @ 30fps
            { rid: 'low', maxBitrate: 200000 }     // 180p @ 15fps
          ]
        });
        ```

        **SFU behavior:**

        ```python
        def select_quality_layer(participant_bandwidth: int, cpu_load: float) -> str:
            """
            Select appropriate simulcast layer based on conditions

            Args:
                participant_bandwidth: Available bandwidth in Kbps
                cpu_load: Client CPU usage (0.0 - 1.0)

            Returns:
                Layer to forward: 'high', 'medium', or 'low'
            """
            # If CPU is overloaded, downgrade quality
            if cpu_load > 0.8:
                return 'low'

            # Select based on bandwidth
            if participant_bandwidth > 1200:
                return 'high'
            elif participant_bandwidth > 400:
                return 'medium'
            else:
                return 'low'
        ```

        **Benefits:**

        - **Bandwidth savings:** Mobile users get 180p instead of 720p
        - **Fast adaptation:** Switch layers without re-encoding
        - **Quality preservation:** Desktop users still get 720p

        ---

        ## Audio Mixing

        **Problem:** For large meetings (100+ participants), downloading 100 audio streams = 10 Mbps audio alone.

        **Solution:** Mix audio on server for large meetings.

        ```python
        class AudioMixer:
            """
            Mix multiple audio streams into one for large meetings

            Used when participant count > 50
            """

            def __init__(self, max_voices: int = 5):
                self.max_voices = max_voices  # Mix only top 5 loudest
                self.sample_rate = 48000
                self.channels = 1  # Mono

            def mix_audio_frames(self, audio_frames: List[np.ndarray]) -> np.ndarray:
                """
                Mix multiple audio frames into single output

                Args:
                    audio_frames: List of audio frames (numpy arrays)

                Returns:
                    Mixed audio frame
                """
                if not audio_frames:
                    return np.zeros(960, dtype=np.int16)  # 20ms silence

                # Calculate volume (RMS) for each frame
                volumes = [np.sqrt(np.mean(frame**2)) for frame in audio_frames]

                # Select top N loudest speakers
                top_indices = np.argsort(volumes)[-self.max_voices:]
                top_frames = [audio_frames[i] for i in top_indices]

                # Mix: average of top frames
                mixed = np.mean(top_frames, axis=0).astype(np.int16)

                # Prevent clipping
                max_val = np.max(np.abs(mixed))
                if max_val > 32767:
                    mixed = (mixed * 32767 / max_val).astype(np.int16)

                return mixed
        ```

    === "🔒 NAT Traversal (STUN/TURN)"

        ## The Challenge

        **Problem:** Most users are behind NATs/firewalls. How do two peers connect?

        ```
        Alice (NAT) -----> [Internet] <----- Bob (NAT)
        Private IP: 192.168.1.5       Private IP: 10.0.0.10
        Public IP: ???                Public IP: ???
        ```

        **Requirements:**

        - Discover public IP address
        - Punch hole through NAT (for direct connection)
        - Fallback to relay if direct connection fails

        ---

        ## ICE (Interactive Connectivity Establishment)

        **Framework:** Try multiple connection methods in order:

        1. **Host candidate:** Direct connection (same network)
        2. **Server reflexive:** Through NAT (STUN)
        3. **Relay candidate:** Through TURN server (fallback)

        ---

        ## STUN (Session Traversal Utilities for NAT)

        **Purpose:** Discover your public IP and port.

        ```python
        import socket
        import struct

        class STUNClient:
            """
            STUN client for NAT traversal

            Discovers public IP address and port
            """

            STUN_SERVERS = [
                ('stun.zoom.us', 3478),
                ('stun.l.google.com', 19302)
            ]

            def __init__(self):
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.settimeout(3)

            def get_public_address(self) -> tuple:
                """
                Send STUN request and parse response

                Returns:
                    (public_ip, public_port) tuple
                """
                # STUN Binding Request
                transaction_id = os.urandom(12)
                message = self._build_stun_request(transaction_id)

                # Try each STUN server
                for server_addr in self.STUN_SERVERS:
                    try:
                        self.sock.sendto(message, server_addr)
                        response, _ = self.sock.recvfrom(1024)

                        # Parse response
                        public_ip, public_port = self._parse_stun_response(response)

                        logger.info(f"Public address: {public_ip}:{public_port}")
                        return (public_ip, public_port)

                    except socket.timeout:
                        logger.warning(f"STUN server {server_addr} timeout")
                        continue

                raise Exception("All STUN servers failed")

            def _build_stun_request(self, transaction_id: bytes) -> bytes:
                """Build STUN Binding Request"""
                message_type = 0x0001  # Binding Request
                message_length = 0
                magic_cookie = 0x2112A442

                header = struct.pack(
                    '!HHI12s',
                    message_type,
                    message_length,
                    magic_cookie,
                    transaction_id
                )

                return header

            def _parse_stun_response(self, response: bytes) -> tuple:
                """
                Parse STUN Binding Response

                Returns:
                    (ip, port) from XOR-MAPPED-ADDRESS attribute
                """
                # Parse header
                msg_type, msg_len, magic, txn_id = struct.unpack(
                    '!HHI12s',
                    response[:20]
                )

                # Parse attributes
                offset = 20
                while offset < len(response):
                    attr_type, attr_len = struct.unpack(
                        '!HH',
                        response[offset:offset+4]
                    )

                    # XOR-MAPPED-ADDRESS (0x0020)
                    if attr_type == 0x0020:
                        # Parse address
                        family, xor_port, xor_ip = struct.unpack(
                            '!HHI',
                            response[offset+4:offset+12]
                        )

                        # XOR with magic cookie
                        port = xor_port ^ (magic >> 16)
                        ip = socket.inet_ntoa(
                            struct.pack('!I', xor_ip ^ magic)
                        )

                        return (ip, port)

                    offset += 4 + attr_len

                raise Exception("XOR-MAPPED-ADDRESS not found")
        ```

        **STUN Flow:**

        ```mermaid
        sequenceDiagram
            participant Client
            participant NAT
            participant STUN

            Client->>NAT: Binding Request (from 192.168.1.5:54321)
            NAT->>NAT: Create mapping (54321 -> 88.77.66.55:12345)
            NAT->>STUN: Binding Request (from 88.77.66.55:12345)

            STUN->>STUN: Read source IP:port from packet
            STUN->>NAT: Binding Response (your address: 88.77.66.55:12345)
            NAT->>Client: Binding Response

            Client->>Client: Now I know my public address!
        ```

        ---

        ## TURN (Traversal Using Relays around NAT)

        **Purpose:** Relay media when direct connection fails (10-15% of cases).

        **When needed:**

        - Symmetric NATs (changes port for each destination)
        - Corporate firewalls blocking UDP
        - Carrier-grade NAT (CGNAT)

        ```python
        class TURNServer:
            """
            TURN server for relaying media when P2P fails

            Allocates relay addresses and forwards packets
            """

            def __init__(self):
                # client_addr -> relay_addr
                self.allocations = {}

                # relay_addr -> {peer_addr -> permission}
                self.permissions = defaultdict(dict)

                # Statistics
                self.stats = {
                    'active_relays': 0,
                    'bytes_relayed': 0
                }

            async def handle_allocation_request(
                self,
                client_addr: tuple,
                username: str,
                password: str
            ) -> tuple:
                """
                Allocate relay address for client

                Args:
                    client_addr: Client's IP:port
                    username: Authentication username
                    password: Authentication password

                Returns:
                    Allocated relay address (IP, port)
                """
                # Authenticate
                if not self._authenticate(username, password):
                    raise Exception("Authentication failed")

                # Allocate relay address
                relay_port = self._allocate_port()
                relay_addr = (self.server_ip, relay_port)

                # Store allocation
                self.allocations[client_addr] = {
                    'relay_addr': relay_addr,
                    'allocated_at': time.time(),
                    'lifetime': 600,  # 10 minutes
                    'bytes_relayed': 0
                }

                self.stats['active_relays'] += 1

                logger.info(f"Allocated relay {relay_addr} for {client_addr}")

                return relay_addr

            async def handle_send_indication(
                self,
                client_addr: tuple,
                peer_addr: tuple,
                data: bytes
            ):
                """
                Relay data from client to peer

                Args:
                    client_addr: Source client address
                    peer_addr: Destination peer address
                    data: Data to relay
                """
                allocation = self.allocations.get(client_addr)
                if not allocation:
                    logger.error(f"No allocation for {client_addr}")
                    return

                relay_addr = allocation['relay_addr']

                # Check permission
                if not self._has_permission(relay_addr, peer_addr):
                    logger.warning(f"No permission to send to {peer_addr}")
                    return

                # Forward data
                await self._send_udp(relay_addr, peer_addr, data)

                # Update stats
                allocation['bytes_relayed'] += len(data)
                self.stats['bytes_relayed'] += len(data)

            async def handle_channel_bind(
                self,
                client_addr: tuple,
                peer_addr: tuple,
                channel_number: int
            ):
                """
                Create channel for efficient relaying

                Channels use 4-byte header instead of STUN overhead
                """
                allocation = self.allocations.get(client_addr)
                if not allocation:
                    return

                relay_addr = allocation['relay_addr']

                # Create channel binding
                self.permissions[relay_addr][peer_addr] = {
                    'channel': channel_number,
                    'created_at': time.time()
                }

                logger.info(f"Bound channel {channel_number} for {peer_addr}")

            def _allocate_port(self) -> int:
                """Allocate available UDP port"""
                # Simple implementation: increment from 49152
                return 49152 + len(self.allocations)

            def _authenticate(self, username: str, password: str) -> bool:
                """Authenticate TURN request"""
                # In production: check against user database
                # Use time-limited credentials for security
                return True

            def _has_permission(self, relay_addr: tuple, peer_addr: tuple) -> bool:
                """Check if client has permission to send to peer"""
                return peer_addr in self.permissions[relay_addr]
        ```

        **TURN Flow:**

        ```mermaid
        sequenceDiagram
            participant Alice
            participant TURN
            participant Bob

            Alice->>TURN: Allocate Request
            TURN->>TURN: Allocate relay port 12345
            TURN-->>Alice: Relay address: turn.zoom.us:12345

            Alice->>TURN: Create Permission for Bob
            TURN-->>Alice: Permission granted

            Bob->>TURN: Allocate Request
            TURN->>TURN: Allocate relay port 12346
            TURN-->>Bob: Relay address: turn.zoom.us:12346

            Alice->>TURN: Send to Bob (via relay 12345)
            TURN->>Bob: Forward data (from relay 12345)

            Bob->>TURN: Send to Alice (via relay 12346)
            TURN->>Alice: Forward data (from relay 12346)
        ```

        **Cost impact:**

        - Direct connection: No server cost for media
        - TURN relay: 10-15% of users × 5 Mbps × $0.05/GB = significant cost
        - Optimization: Use TURN only when necessary

    === "📊 Quality Adaptation"

        ## The Challenge

        **Problem:** Network conditions vary (WiFi, 4G, Ethernet). How to maintain quality without freezing?

        **Scenarios:**

        - **Good network:** 50 Mbps, <10ms latency, 0% packet loss → 1080p @ 30fps
        - **Moderate network:** 2 Mbps, 50ms latency, 2% packet loss → 360p @ 15fps
        - **Poor network:** 500 Kbps, 200ms latency, 8% packet loss → Audio only

        ---

        ## Adaptive Bitrate Algorithm

        ```python
        class AdaptiveBitrateController:
            """
            Dynamically adjust video quality based on network conditions

            Monitors: bandwidth, packet loss, RTT, jitter
            Adjusts: resolution, framerate, bitrate
            """

            def __init__(self):
                self.current_quality = 'high'
                self.target_bitrate = 1500  # Kbps

                # Moving averages
                self.avg_bandwidth = 3000  # Kbps
                self.avg_packet_loss = 0.0
                self.avg_rtt = 50  # ms

                # Thresholds
                self.QUALITY_LEVELS = {
                    'high': {'resolution': '720p', 'fps': 30, 'bitrate': 1500},
                    'medium': {'resolution': '360p', 'fps': 30, 'bitrate': 600},
                    'low': {'resolution': '180p', 'fps': 15, 'bitrate': 200}
                }

            def update_stats(self, stats: dict):
                """
                Update network statistics

                Args:
                    stats: {bandwidth_kbps, packet_loss_rate, rtt_ms}
                """
                alpha = 0.3  # Smoothing factor

                # Exponential moving average
                self.avg_bandwidth = (
                    alpha * stats['bandwidth_kbps'] +
                    (1 - alpha) * self.avg_bandwidth
                )
                self.avg_packet_loss = (
                    alpha * stats['packet_loss_rate'] +
                    (1 - alpha) * self.avg_packet_loss
                )
                self.avg_rtt = (
                    alpha * stats['rtt_ms'] +
                    (1 - alpha) * self.avg_rtt
                )

            def select_quality(self) -> dict:
                """
                Select optimal quality level based on network conditions

                Returns:
                    Quality settings: {resolution, fps, bitrate}
                """
                # Calculate network score (0-100)
                score = self._calculate_network_score()

                # Select quality based on score
                if score >= 80:
                    new_quality = 'high'
                elif score >= 50:
                    new_quality = 'medium'
                else:
                    new_quality = 'low'

                # Hysteresis: avoid flapping
                if new_quality != self.current_quality:
                    if self._should_change_quality(new_quality, score):
                        logger.info(f"Quality change: {self.current_quality} -> {new_quality}")
                        self.current_quality = new_quality

                return self.QUALITY_LEVELS[self.current_quality]

            def _calculate_network_score(self) -> float:
                """
                Calculate overall network quality score

                Returns:
                    Score from 0 (terrible) to 100 (perfect)
                """
                # Bandwidth score (0-40 points)
                bandwidth_score = min(40, (self.avg_bandwidth / 3000) * 40)

                # Packet loss score (0-30 points)
                packet_loss_penalty = self.avg_packet_loss * 10
                packet_loss_score = max(0, 30 - packet_loss_penalty * 3)

                # RTT score (0-30 points)
                rtt_penalty = max(0, self.avg_rtt - 50)
                rtt_score = max(0, 30 - rtt_penalty / 10)

                total_score = bandwidth_score + packet_loss_score + rtt_score

                logger.debug(f"Network score: {total_score:.1f} "
                           f"(BW: {bandwidth_score:.1f}, "
                           f"Loss: {packet_loss_score:.1f}, "
                           f"RTT: {rtt_score:.1f})")

                return total_score

            def _should_change_quality(self, new_quality: str, score: float) -> bool:
                """
                Hysteresis to prevent quality flapping

                Require 10-point buffer before changing quality
                """
                current_level = list(self.QUALITY_LEVELS.keys()).index(self.current_quality)
                new_level = list(self.QUALITY_LEVELS.keys()).index(new_quality)

                # Downgrading: change immediately (avoid freezing)
                if new_level > current_level:
                    return True

                # Upgrading: require stable good conditions (10-point buffer)
                if new_level < current_level:
                    threshold = 80 if new_quality == 'high' else 50
                    return score > threshold + 10

                return False
        ```

        ---

        ## Packet Loss Recovery

        **Techniques:**

        1. **Forward Error Correction (FEC)**
        2. **Negative Acknowledgment (NACK)**
        3. **Frame dropping/freezing trade-off**

        ```python
        class PacketLossRecovery:
            """
            Recover from packet loss using FEC and NACK
            """

            def __init__(self):
                self.fec_enabled = True
                self.nack_enabled = True

                # Statistics
                self.packets_sent = 0
                self.packets_lost = 0
                self.packets_recovered = 0

            def add_fec(self, media_packets: List[bytes]) -> List[bytes]:
                """
                Add Forward Error Correction (FEC) packets

                Use XOR redundancy: every 10 media packets get 2 FEC packets
                Can recover from 2 lost packets without retransmission

                Args:
                    media_packets: List of RTP media packets

                Returns:
                    media_packets + FEC packets
                """
                if not self.fec_enabled:
                    return media_packets

                fec_packets = []

                # Group into blocks of 10
                for i in range(0, len(media_packets), 10):
                    block = media_packets[i:i+10]

                    # XOR all packets in block
                    fec_data = self._xor_packets(block)

                    # Create 2 FEC packets (20% overhead)
                    fec_packets.append(self._create_fec_packet(fec_data, block_id=i//10))
                    fec_packets.append(self._create_fec_packet(fec_data, block_id=i//10, offset=1))

                logger.debug(f"Added {len(fec_packets)} FEC packets for {len(media_packets)} media packets")

                return media_packets + fec_packets

            def recover_lost_packet(
                self,
                received_packets: List[bytes],
                fec_packets: List[bytes],
                lost_seq: int
            ) -> bytes:
                """
                Recover lost packet using FEC

                Args:
                    received_packets: Packets successfully received in block
                    fec_packets: FEC packets for this block
                    lost_seq: Sequence number of lost packet

                Returns:
                    Recovered packet data
                """
                # XOR all received packets with FEC packet
                recovered = self._xor_packets(received_packets + fec_packets[:1])

                self.packets_recovered += 1
                logger.info(f"Recovered packet {lost_seq} using FEC")

                return recovered

            def send_nack(self, lost_packets: List[int]) -> dict:
                """
                Request retransmission of lost packets

                Use RTCP NACK feedback message

                Args:
                    lost_packets: List of lost sequence numbers

                Returns:
                    NACK message to send
                """
                if not self.nack_enabled:
                    return None

                # RTCP NACK message
                nack_message = {
                    'type': 'NACK',
                    'media_ssrc': self.ssrc,
                    'lost_packets': lost_packets[:16]  # Max 16 per NACK
                }

                logger.info(f"Sending NACK for packets: {lost_packets}")

                return nack_message

            def handle_packet_loss(
                self,
                expected_seq: int,
                received_seq: int
            ) -> str:
                """
                Decide how to handle packet loss

                Args:
                    expected_seq: Expected sequence number
                    received_seq: Actually received sequence number

                Returns:
                    Action: 'fec', 'nack', 'skip'
                """
                packets_lost = received_seq - expected_seq
                self.packets_lost += packets_lost

                # Single packet loss: try FEC first, then NACK
                if packets_lost == 1:
                    return 'fec'

                # Burst loss (2-5 packets): NACK immediately
                elif packets_lost <= 5:
                    return 'nack'

                # Heavy loss (>5 packets): skip (NACK won't help)
                else:
                    logger.warning(f"Heavy packet loss ({packets_lost}), skipping recovery")
                    return 'skip'

            def _xor_packets(self, packets: List[bytes]) -> bytes:
                """XOR multiple packets for FEC"""
                if not packets:
                    return b''

                result = bytearray(packets[0])
                for packet in packets[1:]:
                    for i in range(min(len(result), len(packet))):
                        result[i] ^= packet[i]

                return bytes(result)
        ```

        ---

        ## Jitter Buffer

        **Purpose:** Smooth out network jitter (varying delay).

        ```python
        class JitterBuffer:
            """
            Buffer incoming packets to smooth out jitter

            Trade latency for smoothness
            """

            def __init__(self, target_delay_ms: int = 100):
                self.target_delay = target_delay_ms / 1000.0  # seconds
                self.buffer = []  # (seq, timestamp, packet)
                self.last_played_seq = -1

                # Adaptive buffering
                self.min_delay = 20 / 1000.0  # 20ms
                self.max_delay = 200 / 1000.0  # 200ms

            def add_packet(self, seq: int, timestamp: float, packet: bytes):
                """
                Add packet to jitter buffer

                Args:
                    seq: RTP sequence number
                    timestamp: Arrival timestamp
                    packet: Audio/video packet data
                """
                # Insert in order by sequence number
                bisect.insort(self.buffer, (seq, timestamp, packet))

                # Limit buffer size (prevent memory bloat)
                if len(self.buffer) > 100:
                    self.buffer.pop(0)

            def get_next_packet(self, current_time: float) -> bytes:
                """
                Get next packet to play

                Args:
                    current_time: Current timestamp

                Returns:
                    Packet data to play, or None if buffer underrun
                """
                if not self.buffer:
                    return None

                seq, arrival_time, packet = self.buffer[0]

                # Calculate when packet should be played
                play_time = arrival_time + self.target_delay

                # Time to play this packet?
                if current_time >= play_time:
                    self.buffer.pop(0)
                    self.last_played_seq = seq

                    # Adapt buffer size based on jitter
                    self._adapt_buffer_size(current_time - arrival_time)

                    return packet

                # Not yet time
                return None

            def _adapt_buffer_size(self, actual_delay: float):
                """
                Adjust target delay based on observed jitter

                Increase if underruns, decrease if overrunning
                """
                # If we're consistently early, decrease buffer
                if actual_delay < self.target_delay - 0.020:  # 20ms early
                    self.target_delay = max(
                        self.min_delay,
                        self.target_delay * 0.95
                    )

                # If we're consistently late, increase buffer
                elif actual_delay > self.target_delay + 0.050:  # 50ms late
                    self.target_delay = min(
                        self.max_delay,
                        self.target_delay * 1.05
                    )

                logger.debug(f"Jitter buffer delay: {self.target_delay*1000:.1f}ms")
        ```

    === "🌍 Scalability & Load Balancing"

        ## The Challenge

        **Scale:** 10M concurrent meetings, 300M participants, 600 Tbps media traffic.

        **Requirements:**

        - Distribute load across regions
        - Route users to nearest SFU
        - Handle SFU failures gracefully
        - Balance meetings across SFUs

        ---

        ## SFU Clustering

        **Architecture:**

        ```
        Global deployment: 20 regions × 50 SFUs = 1,000 SFU servers

        Each SFU:
        - 10,000 concurrent meetings
        - 300,000 concurrent participants
        - 60 Tbps bandwidth
        - 32 vCPUs, 128 GB RAM
        ```

        **Region selection:**

        ```python
        class RegionSelector:
            """
            Select optimal region for meeting based on participants' locations
            """

            REGIONS = {
                'us-west-1': {'location': (37.4, -122.1), 'capacity': 15_000_000},
                'us-east-1': {'location': (39.0, -77.5), 'capacity': 15_000_000},
                'eu-west-1': {'location': (53.3, -6.3), 'capacity': 12_000_000},
                'ap-southeast-1': {'location': (1.3, 103.8), 'capacity': 10_000_000},
                # ... 16 more regions
            }

            def select_region(self, participants: List[dict]) -> str:
                """
                Select region that minimizes average latency

                Args:
                    participants: List of {user_id, lat, lon}

                Returns:
                    Optimal region ID
                """
                if not participants:
                    return 'us-west-1'  # Default

                # Calculate centroid of participants
                avg_lat = sum(p['lat'] for p in participants) / len(participants)
                avg_lon = sum(p['lon'] for p in participants) / len(participants)

                # Find closest region
                best_region = None
                best_distance = float('inf')

                for region_id, region_info in self.REGIONS.items():
                    # Check capacity
                    if not self._has_capacity(region_id):
                        continue

                    # Calculate distance
                    distance = self._haversine_distance(
                        (avg_lat, avg_lon),
                        region_info['location']
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_region = region_id

                logger.info(f"Selected region {best_region} for {len(participants)} participants")

                return best_region

            def _haversine_distance(self, coord1: tuple, coord2: tuple) -> float:
                """Calculate distance between two coordinates in km"""
                lat1, lon1 = coord1
                lat2, lon2 = coord2

                R = 6371  # Earth radius in km

                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)

                a = (math.sin(dlat/2)**2 +
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                     math.sin(dlon/2)**2)

                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

                return R * c

            def _has_capacity(self, region_id: str) -> bool:
                """Check if region has available capacity"""
                # Query load balancer for current load
                current_load = redis.get(f"region:{region_id}:load")
                capacity = self.REGIONS[region_id]['capacity']

                return current_load < capacity * 0.9  # Keep 10% headroom
        ```

        ---

        ## SFU Load Balancer

        ```python
        class SFULoadBalancer:
            """
            Balance meetings across SFUs in a region

            Uses least-loaded algorithm with health checks
            """

            def __init__(self, redis_client):
                self.redis = redis_client

            def select_sfu(self, region_id: str, meeting_size: int) -> str:
                """
                Select SFU for new meeting

                Args:
                    region_id: Target region (e.g., 'us-west-1')
                    meeting_size: Expected participant count

                Returns:
                    SFU server address
                """
                # Get all SFUs in region
                sfus = self._get_region_sfus(region_id)

                # Filter healthy SFUs
                healthy_sfus = [sfu for sfu in sfus if self._is_healthy(sfu)]

                if not healthy_sfus:
                    raise Exception(f"No healthy SFUs in region {region_id}")

                # Select least loaded
                best_sfu = min(healthy_sfus, key=lambda sfu: self._get_load_score(sfu))

                logger.info(f"Selected SFU {best_sfu} for meeting (size: {meeting_size})")

                return best_sfu

            def _get_region_sfus(self, region_id: str) -> List[str]:
                """Get all SFU servers in region"""
                # In production: query service discovery (Consul, etcd)
                sfu_list = self.redis.smembers(f"region:{region_id}:sfus")
                return [sfu.decode() for sfu in sfu_list]

            def _is_healthy(self, sfu_address: str) -> bool:
                """Check if SFU is healthy"""
                # Check heartbeat (last heartbeat < 10 seconds ago)
                last_heartbeat = self.redis.get(f"sfu:{sfu_address}:heartbeat")

                if not last_heartbeat:
                    return False

                age = time.time() - float(last_heartbeat)
                return age < 10

            def _get_load_score(self, sfu_address: str) -> float:
                """
                Calculate load score for SFU

                Lower score = less loaded = more preferred
                """
                stats = self.redis.hgetall(f"sfu:{sfu_address}:stats")

                # Parse stats
                active_meetings = int(stats.get(b'active_meetings', 0))
                active_participants = int(stats.get(b'active_participants', 0))
                cpu_usage = float(stats.get(b'cpu_usage', 0))
                bandwidth_gbps = float(stats.get(b'bandwidth_gbps', 0))

                # Weighted score
                score = (
                    active_meetings * 1.0 +
                    active_participants * 0.1 +
                    cpu_usage * 0.5 +
                    bandwidth_gbps * 0.2
                )

                return score

            def update_sfu_stats(self, sfu_address: str, stats: dict):
                """
                Update SFU statistics (called by SFU every 5 seconds)

                Args:
                    sfu_address: SFU server address
                    stats: {active_meetings, active_participants, cpu_usage, bandwidth_gbps}
                """
                # Store stats
                self.redis.hset(
                    f"sfu:{sfu_address}:stats",
                    mapping={
                        'active_meetings': stats['active_meetings'],
                        'active_participants': stats['active_participants'],
                        'cpu_usage': stats['cpu_usage'],
                        'bandwidth_gbps': stats['bandwidth_gbps']
                    }
                )

                # Update heartbeat
                self.redis.set(
                    f"sfu:{sfu_address}:heartbeat",
                    time.time()
                )

                logger.debug(f"Updated stats for SFU {sfu_address}")
        ```

        ---

        ## SFU Failure Handling

        **Problem:** SFU crashes, all meetings on that server lost?

        **Solution:** Migrate meetings to another SFU.

        ```python
        class SFUFailoverHandler:
            """
            Handle SFU failures by migrating meetings to healthy SFUs
            """

            def __init__(self, signaling_server, load_balancer):
                self.signaling = signaling_server
                self.lb = load_balancer

            async def handle_sfu_failure(self, failed_sfu: str):
                """
                Migrate all meetings from failed SFU to healthy SFUs

                Args:
                    failed_sfu: Address of failed SFU server
                """
                logger.error(f"SFU failure detected: {failed_sfu}")

                # Get all meetings on failed SFU
                meetings = self._get_sfu_meetings(failed_sfu)

                logger.info(f"Migrating {len(meetings)} meetings from failed SFU")

                # Migrate each meeting
                for meeting_id in meetings:
                    await self._migrate_meeting(meeting_id, failed_sfu)

            async def _migrate_meeting(self, meeting_id: str, old_sfu: str):
                """
                Migrate single meeting to new SFU

                Steps:
                1. Select new SFU
                2. Notify all participants to reconnect
                3. Participants establish new WebRTC connections
                """
                # Get meeting info
                meeting = self._get_meeting_info(meeting_id)
                participants = meeting['participants']
                region = meeting['region']

                # Select new SFU
                new_sfu = self.lb.select_sfu(region, len(participants))

                logger.info(f"Migrating meeting {meeting_id}: {old_sfu} -> {new_sfu}")

                # Update meeting state
                self._update_meeting_sfu(meeting_id, new_sfu)

                # Notify all participants to reconnect
                for participant_id in participants:
                    await self.signaling.send_message(
                        participant_id,
                        {
                            'type': 'sfu-migration',
                            'new_sfu': new_sfu,
                            'reason': 'server_failure'
                        }
                    )

                # Client automatically reconnects to new SFU
                # WebRTC peer connection re-established
                # Meeting continues with minimal disruption (2-3 second freeze)
        ```

        **Client handling:**

        ```javascript
        // Client receives migration message
        ws.onmessage = async (event) => {
          const message = JSON.parse(event.data);

          if (message.type === 'sfu-migration') {
            console.log('SFU migration requested:', message.new_sfu);

            // Close old peer connection
            peerConnection.close();

            // Create new peer connection to new SFU
            peerConnection = new RTCPeerConnection({
              iceServers: iceServers
            });

            // Re-add local media tracks
            localStream.getTracks().forEach(track => {
              peerConnection.addTrack(track, localStream);
            });

            // Create new offer
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);

            // Send to signaling server (will route to new SFU)
            ws.send(JSON.stringify({
              type: 'offer',
              sdp: offer.sdp
            }));

            console.log('Reconnected to new SFU');
          }
        };
        ```

=== "⚡ Step 4: Scale & Optimize"

    ## Overview

    Scaling from 1M to 300M daily participants.

    **Scaling challenges:**

    - **Media bandwidth:** 600 Tbps through SFU servers
    - **Signaling load:** 500K events/sec (join/leave/mute/etc.)
    - **Recording storage:** 3.4 PB/day
    - **Global latency:** <150ms from anywhere

    ---

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **SFU bandwidth** | ✅ Yes | 1,000 SFU servers across 20 regions, each 60 Tbps |
    | **WebSocket connections** | ✅ Yes | 300 signaling servers (1M connections each), sticky sessions |
    | **Recording encoding** | ✅ Yes | Distributed encoding (1 encoder per 100 meetings), GPU acceleration |
    | **TURN relay cost** | ✅ Yes | Optimize NAT traversal (ICE-TCP, IPv6), reduce TURN usage to <10% |
    | **Signaling latency** | 🟡 Approaching | Redis pub/sub for low-latency message routing (<5ms) |

    ---

    ## Optimization Strategies

    ### 1. Simulcast Optimization

    **Problem:** Sending 3 quality layers = 3x bandwidth.

    **Solution:** Only enable simulcast when needed.

    ```python
    def should_enable_simulcast(meeting_size: int, participant_network: str) -> bool:
        """
        Enable simulcast only for large meetings or mixed networks

        Args:
            meeting_size: Number of participants
            participant_network: 'excellent', 'good', 'poor'

        Returns:
            True if simulcast should be enabled
        """
        # Small meetings (< 5 people): no simulcast needed
        if meeting_size < 5:
            return False

        # Large meetings: always use simulcast
        if meeting_size > 20:
            return True

        # Medium meetings: use if any participant has poor network
        if participant_network == 'poor':
            return True

        return False
    ```

    **Savings:** 60% of meetings don't need simulcast → 40% bandwidth reduction.

    ---

    ### 2. Video Pause for Off-Screen Participants

    **Problem:** Gallery view shows 9 participants, but downloading 50 video streams.

    **Solution:** Only download visible streams.

    ```javascript
    // Client: Subscribe only to visible participants
    function updateVisibleParticipants(visibleIds) {
      const currentIds = new Set(subscribedStreams.keys());
      const newIds = new Set(visibleIds);

      // Unsubscribe from off-screen participants
      for (const id of currentIds) {
        if (!newIds.has(id)) {
          const stream = subscribedStreams.get(id);
          stream.getTracks().forEach(track => track.stop());
          subscribedStreams.delete(id);

          // Notify server to stop sending
          ws.send(JSON.stringify({
            type: 'unsubscribe',
            participant_id: id
          }));
        }
      }

      // Subscribe to newly visible participants
      for (const id of newIds) {
        if (!currentIds.has(id)) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            participant_id: id,
            quality: 'medium'  // Gallery view doesn't need 1080p
          }));
        }
      }
    }

    // Update when scrolling gallery
    galleryElement.addEventListener('scroll', () => {
      const visibleIds = getVisibleParticipantIds();
      updateVisibleParticipants(visibleIds);
    });
    ```

    **Savings:** 50 participants, show 9 → 82% bandwidth reduction.

    ---

    ### 3. Audio-Only Mode for Large Meetings

    **Problem:** 1,000 participant webinar = impossible to show all videos.

    **Solution:** Webinar mode (host video + audience audio).

    ```python
    def select_meeting_mode(participant_count: int) -> str:
        """
        Select meeting mode based on size

        Returns:
            'p2p', 'sfu', 'webinar'
        """
        if participant_count <= 4:
            return 'p2p'  # Peer-to-peer
        elif participant_count <= 100:
            return 'sfu'  # Full video conferencing
        else:
            return 'webinar'  # Host video + audio only
    ```

    **Webinar mode:**

    - Host + panelists: Video enabled
    - Audience: Audio only (receive), can "raise hand" to join panel
    - Bandwidth: 100 Kbps per audience member (vs 2-5 Mbps in SFU mode)

    ---

    ### 4. Recording Optimization

    **Problem:** 20% of meetings recorded = 3.4 PB/day storage.

    **Solution:** Optimized encoding + cleanup.

    ```python
    class RecordingOptimizer:
        """
        Optimize recording storage and encoding
        """

        QUALITY_PRESETS = {
            'high': {'resolution': '1080p', 'bitrate': 3000, 'size_mb_per_hour': 1350},
            'medium': {'resolution': '720p', 'bitrate': 1500, 'size_mb_per_hour': 675},
            'low': {'resolution': '480p', 'bitrate': 800, 'size_mb_per_hour': 360}
        }

        def select_recording_quality(self, meeting_type: str, duration_min: int) -> str:
            """
            Select appropriate recording quality

            Args:
                meeting_type: 'webinar', 'meeting', 'personal'
                duration_min: Meeting duration in minutes

            Returns:
                Quality preset: 'high', 'medium', 'low'
            """
            # Webinars: always high quality (will be shared widely)
            if meeting_type == 'webinar':
                return 'high'

            # Long meetings: medium quality (save storage)
            if duration_min > 60:
                return 'medium'

            # Short meetings: high quality
            return 'high'

        def cleanup_old_recordings(self):
            """
            Delete recordings older than retention period

            Retention:
            - Free accounts: 30 days
            - Paid accounts: 90 days
            - Enterprise: Custom (180 days)
            """
            cutoff_date = datetime.now() - timedelta(days=90)

            old_recordings = db.query(
                "SELECT recording_id, file_path FROM recordings "
                "WHERE created_at < %s AND account_type = 'paid'",
                (cutoff_date,)
            )

            for recording in old_recordings:
                # Delete from S3
                s3.delete_object(
                    Bucket='zoom-recordings',
                    Key=recording['file_path']
                )

                # Delete from database
                db.execute(
                    "DELETE FROM recordings WHERE recording_id = %s",
                    (recording['recording_id'],)
                )

            logger.info(f"Cleaned up {len(old_recordings)} old recordings")
    ```

    ---

    ## Cost Optimization

    **Monthly cost at 300M daily participants:**

    | Component | Cost | Notes |
    |-----------|------|-------|
    | **SFU servers** | $2,160,000 | 1,000 servers × $2,160/month (32 vCPU, 128 GB) |
    | **TURN servers** | $324,000 | 300 servers (10-15% of traffic) |
    | **Signaling servers** | $216,000 | 300 servers (WebSocket) |
    | **Recording encoders** | $540,000 | 500 GPU instances for encoding |
    | **PostgreSQL** | $54,000 | Managed PostgreSQL (50 nodes) |
    | **Redis cluster** | $108,000 | 200 Redis nodes for session state |
    | **S3 storage** | $2,800,000 | 112 PB × $25/TB |
    | **Bandwidth** | $1,800,000 | 600 Tbps × $0.05/GB egress |
    | **Total** | **$8,002,000/month** | ($0.027 per participant per day) |

    **Optimization impact:**

    - Simulcast optimization: -40% bandwidth = -$720K/month
    - Visible streams only: -30% bandwidth = -$540K/month
    - Recording cleanup: -50% storage = -$1,400K/month
    - **Total savings: $2,660K/month (33% reduction)**

    ---

    ## Monitoring & Alerting

    **Key metrics:**

    | Metric | Target | Alert Threshold | Impact |
    |--------|--------|-----------------|--------|
    | **Audio Latency (P95)** | < 80ms | > 150ms | Bad UX, conversation breaks |
    | **Video Latency (P95)** | < 150ms | > 300ms | Lip sync issues |
    | **Packet Loss (P95)** | < 1% | > 5% | Quality degradation |
    | **Connection Success Rate** | > 99% | < 95% | Users can't join |
    | **SFU CPU Usage** | < 70% | > 85% | Need more capacity |
    | **TURN Usage** | < 15% | > 25% | High relay costs |
    | **Recording Success** | > 99.9% | < 99% | Lost recordings |

    ---

    ## Quality Metrics (QoS)

    ```python
    class QualityMonitor:
        """
        Monitor call quality metrics and trigger alerts
        """

        def __init__(self):
            self.metrics_buffer = []

        def record_quality_stats(self, stats: dict):
            """
            Record quality statistics from participant

            Args:
                stats: {
                    participant_id, meeting_id,
                    audio_latency_ms, video_latency_ms,
                    packet_loss_rate, jitter_ms,
                    audio_bitrate, video_bitrate
                }
            """
            # Store in Cassandra for analytics
            cassandra.insert(
                "INSERT INTO call_quality_metrics "
                "(participant_id, meeting_id, timestamp, metrics) "
                "VALUES (?, ?, ?, ?)",
                (stats['participant_id'], stats['meeting_id'],
                 datetime.now(), json.dumps(stats))
            )

            # Check for issues
            self._check_quality_issues(stats)

        def _check_quality_issues(self, stats: dict):
            """Alert on quality issues"""
            issues = []

            # High latency
            if stats['audio_latency_ms'] > 150:
                issues.append(f"High audio latency: {stats['audio_latency_ms']}ms")

            # High packet loss
            if stats['packet_loss_rate'] > 0.05:
                issues.append(f"High packet loss: {stats['packet_loss_rate']*100:.1f}%")

            # Low bitrate (quality degraded)
            if stats['video_bitrate'] < 300:
                issues.append(f"Low video bitrate: {stats['video_bitrate']} kbps")

            if issues:
                # Alert participant (show notification)
                self._notify_poor_connection(stats['participant_id'], issues)

                # Log for analytics
                logger.warning(f"Quality issues for {stats['participant_id']}: {issues}")
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **SFU architecture:** Scalability + quality (not P2P or MCU)
    2. **WebRTC standard:** Browser support, NAT traversal built-in
    3. **STUN/TURN for connectivity:** 10-15% need TURN relay
    4. **Simulcast for adaptation:** Multiple quality layers for varied bandwidth
    5. **Regional SFU clusters:** Low latency by being close to users
    6. **Adaptive bitrate:** Maintain quality on poor networks
    7. **Jitter buffer:** Trade 100ms latency for smoothness

    ---

    ## Interview Tips

    ✅ **Explain SFU vs MCU vs P2P** - Critical architecture decision

    ✅ **Discuss WebRTC in depth** - Core technology, explain signaling vs media

    ✅ **NAT traversal complexity** - STUN/TURN, ICE framework

    ✅ **Quality adaptation** - Adaptive bitrate, simulcast, packet loss recovery

    ✅ **Scalability** - How to handle 10M concurrent meetings

    ✅ **Network resilience** - Jitter buffer, FEC, NACK

    ✅ **Cost optimization** - TURN relay costs, bandwidth, storage

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"Why SFU instead of MCU?"** | SFU: low CPU (just forward), low latency, flexible layouts. MCU: high CPU (mixing), fixed layouts, added latency |
    | **"How to handle NAT traversal?"** | STUN for IP discovery, TURN relay for restrictive NATs (10-15%), ICE framework tries all options |
    | **"How to maintain quality on poor networks?"** | Adaptive bitrate (downgrade resolution/fps), simulcast (multiple layers), FEC/NACK for packet loss, jitter buffer |
    | **"How to scale to 1000 participants?"** | Webinar mode: host video + audience audio, selective subscription (only download visible streams) |
    | **"How to reduce latency?"** | SFU close to users (<50ms), minimize jitter buffer (<100ms), skip TURN when possible (direct P2P) |
    | **"How to handle SFU failures?"** | Migrate meetings to healthy SFU, participants reconnect automatically (2-3s freeze) |

    ---

    ## Key Tradeoffs

    | Tradeoff | Decision | Reasoning |
    |----------|----------|-----------|
    | **SFU vs MCU** | SFU | Scalability and quality over simplicity |
    | **Latency vs Quality** | Latency | <150ms critical, can degrade quality if needed |
    | **Bandwidth vs Participants** | Bandwidth | Limit video streams for large meetings |
    | **Direct vs TURN** | Try direct first | TURN is expensive, only use when necessary |
    | **Audio vs Video** | Audio priority | Audio must always work, video can be disabled |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Zoom, Google Meet, Microsoft Teams, WebEx, Discord, Skype

---

*Master this problem and you'll be ready for: Twitch, Discord voice/video, Google Hangouts, Slack calls, WhatsApp video*
