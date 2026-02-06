# Design an AI Voice Assistant (Siri, Google Assistant)

An intelligent voice-activated assistant that processes natural language voice commands, understands user intent, executes actions, and provides spoken responses. The system handles wake word detection, speech recognition, natural language understanding, dialogue management, action execution, and text-to-speech synthesis.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M daily active users, 1B voice queries/day, 500M concurrent device sessions, 100K queries/sec peak |
| **Key Challenges** | Always-on wake word detection, low-latency speech-to-text, intent classification with context, multi-turn dialogue, privacy-preserving edge processing, multi-language support |
| **Core Concepts** | Wake word detection, Automatic Speech Recognition (ASR/Whisper), Natural Language Understanding (NLU/BERT), Intent classification, Slot filling, Dialogue state tracking, Text-to-speech (TTS), Neural vocoders, Edge computing |
| **Companies** | Apple (Siri), Google (Google Assistant), Amazon (Alexa), Microsoft (Cortana), Samsung (Bixby) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Wake Word Detection** | Always-on listening for "Hey Siri", "OK Google" trigger phrase | P0 (Must have) |
    | **Speech-to-Text (STT)** | Convert spoken audio to text with high accuracy | P0 (Must have) |
    | **Intent Classification** | Identify user's goal (play music, set alarm, get weather) | P0 (Must have) |
    | **Entity Extraction** | Extract parameters (song name, time, location) from query | P0 (Must have) |
    | **Action Execution** | Execute commands (call API, control device, query knowledge base) | P0 (Must have) |
    | **Text-to-Speech (TTS)** | Generate natural-sounding spoken responses | P0 (Must have) |
    | **Multi-turn Dialogue** | Maintain context across conversation (follow-up questions) | P0 (Must have) |
    | **Context Management** | Remember user preferences, location, history | P0 (Must have) |
    | **Voice Matching** | Identify speaker for personalized responses | P1 (Should have) |
    | **Multi-language Support** | Support 50+ languages with accent variations | P1 (Should have) |
    | **Offline Mode** | Basic commands work without internet (timers, alarms) | P1 (Should have) |
    | **Noise Cancellation** | Handle background noise, multiple speakers | P1 (Should have) |
    | **Third-party Skills** | Extensible plugin system (Alexa Skills, Google Actions) | P2 (Nice to have) |
    | **Proactive Suggestions** | Contextual recommendations based on time/location | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Music streaming infrastructure (assume integration with Spotify/Apple Music)
    - Smart home device control (focus on voice processing pipeline)
    - E-commerce/shopping cart management
    - Video calling/screen sharing
    - Model training infrastructure (assume pre-trained models)
    - Content creation (music generation, image generation)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Wake Word)** | < 200ms detection time | Instant response feel |
    | **Latency (STT)** | < 500ms streaming recognition | Real-time transcription |
    | **Latency (NLU + Action)** | < 300ms intent classification | Total voice-to-action under 1s |
    | **Latency (TTS)** | < 400ms first audio chunk | Immediate response playback |
    | **Accuracy (Wake Word)** | < 0.1% false positive, > 99% true positive | Minimize accidental triggers |
    | **Accuracy (STT)** | > 95% Word Error Rate (WER) | High transcription quality |
    | **Accuracy (Intent)** | > 90% intent classification | Understand user correctly |
    | **Availability** | 99.99% uptime | Always accessible assistant |
    | **Privacy** | On-device processing where possible, encrypted transmission | User data protection |
    | **Battery Impact** | < 2% battery drain per hour (wake word listening) | All-day device usage |
    | **Scalability** | Handle 10x traffic spikes (new feature launches) | Support viral adoption |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 100M users
    Monthly Active Users (MAU): 200M users

    Voice queries:
    - Average queries per DAU: 10 queries/day
    - Daily queries: 100M × 10 = 1B queries/day
    - Query QPS: 1B / 86,400 = ~11,600 queries/sec average
    - Peak QPS: 5x average (morning/evening) = ~58,000 queries/sec

    Wake word detections:
    - Devices per user: 2 devices (phone, smart speaker)
    - Active devices: 100M × 2 = 200M devices
    - Wake word triggers per device/day: 10
    - Wake word events: 200M × 10 = 2B events/day
    - Wake word QPS: 2B / 86,400 = ~23,000/sec

    Audio processing:
    - Average query duration: 3 seconds audio
    - Average audio size: 3s × 16kHz × 2 bytes = 96 KB per query
    - Daily audio ingress: 1B × 96 KB = 96 TB/day
    - Audio throughput: 96 TB / 86,400 = ~1.1 GB/sec

    Speech-to-text:
    - STT requests: 1B queries/day
    - STT QPS: ~11,600/sec
    - Average transcription: 15 words, 100 characters
    - Daily transcription data: 1B × 100 bytes = 100 GB/day

    Intent classification:
    - NLU requests: 1B queries/day (same as STT)
    - NLU QPS: ~11,600/sec
    - Intent distribution:
      - Simple queries (weather, time, facts): 40%
      - Commands (alarms, reminders, timers): 30%
      - Smart home control: 15%
      - Music/media playback: 10%
      - Conversational (multi-turn): 5%

    Multi-turn dialogues:
    - Dialogues per day: 5% × 1B = 50M dialogues
    - Average turns per dialogue: 3 turns
    - Dialogue messages: 50M × 3 = 150M messages
    - Active dialogue sessions: 50M / 86,400 = ~580 sessions/sec

    Text-to-speech:
    - TTS requests: 1B responses/day (one per query)
    - TTS QPS: ~11,600/sec
    - Average response: 20 words, 5 seconds audio
    - Response audio: 5s × 24kHz × 2 bytes = 240 KB per response
    - Daily TTS audio: 1B × 240 KB = 240 TB/day
    - TTS throughput: 240 TB / 86,400 = ~2.8 GB/sec

    Action executions:
    - API calls (weather, calendar, etc.): 500M/day
    - Smart home commands: 150M/day
    - Knowledge base queries: 400M/day
    - Music playback: 100M/day
    - Total actions: 1.15B/day

    Voice matching:
    - Voice profile checks: 1B queries/day
    - Profile comparisons: ~11,600/sec
    - Average profile size: 10 KB (voice embeddings)

    Read/Write ratio: 20:1 (queries >> profile updates)
    ```

    ### Storage Estimates

    ```
    User profiles:
    - Active users: 200M MAU
    - Profile data: 5 KB (preferences, settings, linked accounts)
    - 200M × 5 KB = 1 TB

    Voice profiles (speaker recognition):
    - Users with voice profiles: 150M (75% of MAU)
    - Voice embeddings: 10 KB per user
    - 150M × 10 KB = 1.5 TB

    Query history:
    - Queries: 1B/day
    - Query record: 1 KB (audio_id, text, intent, timestamp, user_id)
    - Daily: 1B × 1 KB = 1 TB/day
    - 90-day retention: 1 TB × 90 = 90 TB

    Audio recordings (opt-in):
    - Recordings saved: 10% of queries (for quality improvement)
    - Audio size: 96 KB per query
    - Daily: 100M × 96 KB = 9.6 TB/day
    - 30-day retention: 9.6 TB × 30 = 288 TB

    Dialogue state (active sessions):
    - Active sessions: 10M concurrent
    - State per session: 5 KB (context, history, entities)
    - 10M × 5 KB = 50 GB (in-memory)

    User context:
    - Location history: 200M users × 100 KB = 20 TB
    - App usage patterns: 200M users × 50 KB = 10 TB
    - Calendar/contacts metadata: 200M users × 200 KB = 40 TB

    Intent models:
    - NLU models: 50 languages × 500 MB = 25 TB
    - Wake word models: 50 languages × 50 MB = 2.5 TB
    - TTS models: 50 languages × 1 GB = 50 TB

    Knowledge base:
    - Facts, entities, Q&A pairs: 100 TB
    - Embeddings for semantic search: 50 TB

    Third-party skills:
    - Skill definitions: 100K skills × 100 KB = 10 GB
    - Skill metadata: 100K × 10 KB = 1 GB

    Total: 1 TB (profiles) + 1.5 TB (voice) + 90 TB (history) + 288 TB (audio) + 70 TB (context) + 77.5 TB (models) + 150 TB (knowledge) + 11 GB (skills) ≈ 678 TB
    ```

    ### Compute Estimates

    ```
    Wake word detection (on-device):
    - Devices: 200M active
    - Model size: 50 MB (small neural network)
    - Inference: Continuous, ~5ms per frame
    - Power: < 100mW per device (dedicated DSP)

    Speech-to-text (cloud):
    - STT QPS: 11,600 queries/sec
    - Model: Whisper-large (1.5B params) or similar
    - Inference time: 500ms per 3-second audio (with GPU)
    - GPUs needed: 11,600 / (1/0.5) = 5,800 concurrent requests
    - With batching (8x): ~725 GPUs (A100 80GB)
    - Cost: 725 × $2.50/hour = $1,812/hour = $43,500/day = $1.3M/month

    Intent classification (cloud):
    - NLU QPS: 11,600/sec
    - Model: BERT-large (340M params) for intent + slot filling
    - Inference time: 50ms per query
    - Throughput per GPU: 20 queries/sec
    - GPUs needed: 11,600 / 20 = 580 GPUs
    - With batching (4x): ~145 GPUs (A100)
    - Cost: 145 × $2.50/hour = $362/hour = $8,700/day = $260K/month

    Text-to-speech (cloud):
    - TTS QPS: 11,600/sec
    - Model: Neural vocoder (WaveNet, Tacotron 2)
    - Inference time: 400ms per 5-second audio
    - Throughput per GPU: 2.5 queries/sec
    - GPUs needed: 11,600 / 2.5 = 4,640 GPUs
    - With batching/caching (5x): ~930 GPUs (A100)
    - Cost: 930 × $2.50/hour = $2,325/hour = $55,800/day = $1.67M/month

    Action execution (CPUs):
    - API orchestration: 11,600 req/sec
    - CPU-bound workload
    - Servers: ~500 application servers (c5.4xlarge equivalent)
    - Cost: 500 × $0.68/hour = $340/hour = $8,160/day = $245K/month

    Dialogue management (in-memory):
    - Active sessions: 10M concurrent
    - State tracking: Redis cluster
    - Memory: 50 GB (distributed)
    - Servers: ~50 Redis instances (r5.2xlarge)
    - Cost: 50 × $0.504/hour = $25/hour = $600/day = $18K/month

    Total compute cost: $1.3M + $260K + $1.67M + $245K + $18K = ~$3.5M/month
    ```

    ### Bandwidth Estimates

    ```
    Audio ingress (wake word → cloud):
    - Triggered queries: 11,600/sec × 96 KB = 1.1 GB/sec ≈ 8.8 Gbps
    - Compressed (Opus codec, 3:1): ~3 Gbps

    STT processing:
    - Audio → STT service: 1.1 GB/sec
    - STT → NLU: 11,600/sec × 100 bytes = 1.2 MB/sec ≈ 10 Mbps

    NLU processing:
    - Text → NLU: 1.2 MB/sec
    - NLU → Action executor: 11,600/sec × 500 bytes = 5.8 MB/sec ≈ 46 Mbps

    Action execution:
    - API calls: 11,600/sec × 2 KB = 23 MB/sec ≈ 184 Mbps
    - Responses: 11,600/sec × 5 KB = 58 MB/sec ≈ 464 Mbps

    TTS egress:
    - TTS audio: 11,600/sec × 240 KB = 2.8 GB/sec ≈ 22 Gbps
    - Compressed (Opus): ~7 Gbps

    Dialogue state sync:
    - Active sessions: 10M
    - State updates: 580/sec × 5 KB = 2.9 MB/sec ≈ 23 Mbps

    Total ingress: ~3 Gbps (audio)
    Total egress: ~7 Gbps (TTS responses)
    Internal: ~1 Gbps (service communication)
    ```

    ### Memory Estimates (Caching)

    ```
    Wake word model cache (on-device):
    - Model: 50 MB per device
    - Total: 200M × 50 MB = 10 PB (distributed across devices)

    STT model cache (cloud):
    - Model weights: 3 GB (Whisper-large)
    - Loaded replicas: 725 GPUs × 3 GB = 2.2 TB

    NLU model cache (cloud):
    - Model weights: 1.3 GB (BERT-large)
    - Loaded replicas: 145 GPUs × 1.3 GB = 189 GB

    TTS model cache (cloud):
    - Model weights: 2 GB (neural vocoder)
    - Loaded replicas: 930 GPUs × 2 GB = 1.86 TB

    Dialogue state cache:
    - Active sessions: 10M × 5 KB = 50 GB

    User profile cache:
    - Hot profiles: 50M users × 5 KB = 250 GB

    Intent cache (common queries):
    - Cached responses: 10M queries × 500 bytes = 5 GB

    Voice profile cache:
    - Active voice profiles: 20M × 10 KB = 200 GB

    Knowledge base cache:
    - Hot facts: 10M entities × 2 KB = 20 GB

    Total cache: 2.2 TB + 189 GB + 1.86 TB + 50 GB + 250 GB + 5 GB + 200 GB + 20 GB ≈ 4.8 TB (cloud memory)
    ```

    ---

    ## Key Assumptions

    1. Average user makes 10 voice queries per day
    2. Each query is ~3 seconds of audio (96 KB)
    3. 5% of queries are multi-turn dialogues (average 3 turns)
    4. Wake word detection runs on-device with dedicated DSP
    5. STT/NLU/TTS processed in cloud for accuracy (except offline mode)
    6. 10% of audio saved for quality improvement (opt-in)
    7. 75% of users have voice profiles for speaker recognition
    8. Common queries cached to reduce compute (30% cache hit rate)
    9. Peak traffic 5x average during morning (7-9am) and evening (6-8pm)
    10. Multi-language support requires separate models per language

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Edge-cloud hybrid:** On-device wake word detection, cloud-based STT/NLU/TTS for accuracy
    2. **Streaming pipeline:** Low-latency streaming ASR and incremental intent recognition
    3. **Context-aware:** Maintain dialogue state and user context across multi-turn conversations
    4. **Modular architecture:** Separate concerns (STT, NLU, action execution, TTS) for independent scaling
    5. **Privacy-first:** Minimal data transmission, encryption, opt-in recording, on-device processing where possible
    6. **Multi-language:** Language detection and routing to appropriate models
    7. **Fault tolerance:** Graceful degradation (offline mode, cached responses)

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Edge Device (Phone/Speaker)"
            Mic[Microphone<br/>Audio capture]
            WakeWord[Wake Word Detector<br/>Small neural network<br/>On-device DSP]
            AudioBuffer[Audio Buffer<br/>Circular buffer]
            Codec[Audio Codec<br/>Opus compression]
            Speaker[Speaker<br/>Audio playback]
            LocalCache[Local Cache<br/>Offline commands]
        end

        subgraph "API Gateway"
            LB[Load Balancer<br/>WebSocket/gRPC]
            Auth[Auth Service<br/>Device authentication]
            RateLimit[Rate Limiter<br/>Per-user quotas]
            Router[Request Router<br/>Route by language/region]
        end

        subgraph "Speech-to-Text Service"
            STT_LB[STT Load Balancer<br/>Streaming]
            STT_Pool[ASR Engine Pool<br/>Whisper/Wav2Vec<br/>GPU-accelerated]
            VAD[Voice Activity<br/>Detection]
            LangDetect[Language Detection<br/>Identify spoken language]
        end

        subgraph "Natural Language Understanding"
            NLU_Service[NLU Service<br/>BERT-based]
            Intent[Intent Classifier<br/>Multi-class classification]
            SlotFill[Slot Filling<br/>Entity extraction]
            Context[Context Manager<br/>Dialogue history]
        end

        subgraph "Dialogue Management"
            DM[Dialogue Manager<br/>State machine]
            StateTracker[State Tracker<br/>Track conversation]
            PolicyNet[Policy Network<br/>Next action]
            PersonaMgr[Persona Manager<br/>Response style]
        end

        subgraph "Action Execution Layer"
            ActionRouter[Action Router<br/>Intent → handler]
            KnowledgeBase[Knowledge Base<br/>Facts, Q&A]
            APIOrchestrator[API Orchestrator<br/>External services]
            SmartHome[Smart Home<br/>Device control]
            MediaControl[Media Control<br/>Music/podcasts]
            Calendar[Calendar/Reminders<br/>Personal data]
        end

        subgraph "Text-to-Speech Service"
            TTS_LB[TTS Load Balancer]
            TTS_Pool[TTS Engine Pool<br/>Neural vocoder<br/>GPU-accelerated]
            Voice[Voice Selection<br/>Per user preference]
            ProsodyMod[Prosody Modulation<br/>Emotional tone]
        end

        subgraph "User Services"
            UserProfile[User Profile Service<br/>Preferences, settings]
            VoiceProfile[Voice Profile Service<br/>Speaker recognition]
            ContextDB[Context Service<br/>Location, history]
            PermissionMgr[Permission Manager<br/>Privacy controls]
        end

        subgraph "Caching Layer"
            Redis_Session[Redis<br/>Dialogue state]
            Redis_Intent[Redis<br/>Intent cache]
            Redis_TTS[Redis<br/>TTS response cache]
            Redis_User[Redis<br/>User profile cache]
        end

        subgraph "Storage"
            UserDB[(User DB<br/>PostgreSQL<br/>Profiles)]
            QueryDB[(Query History<br/>Cassandra<br/>Time-series)]
            AudioStore[(Audio Storage<br/>S3<br/>Encrypted)]
            ModelRegistry[(Model Registry<br/>S3<br/>ML models)]
            KnowledgeDB[(Knowledge Graph<br/>Neo4j<br/>Entities)]
        end

        subgraph "Analytics & ML"
            Analytics[Analytics Service<br/>Query patterns]
            Telemetry[Telemetry<br/>Performance metrics]
            FeedbackLoop[Feedback Loop<br/>Model improvement]
            ABTest[A/B Testing<br/>Model variants]
        end

        %% Flow connections
        Mic --> WakeWord
        WakeWord -->|Wake word detected| AudioBuffer
        AudioBuffer --> Codec
        Codec --> LB

        LB --> Auth
        Auth --> RateLimit
        RateLimit --> Router

        Router --> STT_LB
        STT_LB --> VAD
        VAD --> LangDetect
        LangDetect --> STT_Pool

        STT_Pool --> NLU_Service
        NLU_Service --> Intent
        NLU_Service --> SlotFill
        Intent --> Context
        SlotFill --> Context

        Context --> DM
        DM --> StateTracker
        StateTracker --> PolicyNet
        PolicyNet --> PersonaMgr

        PersonaMgr --> ActionRouter
        ActionRouter --> KnowledgeBase
        ActionRouter --> APIOrchestrator
        ActionRouter --> SmartHome
        ActionRouter --> MediaControl
        ActionRouter --> Calendar

        ActionRouter --> TTS_LB
        TTS_LB --> Voice
        Voice --> ProsodyMod
        ProsodyMod --> TTS_Pool

        TTS_Pool --> Codec
        Codec --> Speaker

        %% User services
        VoiceProfile -.->|Verify speaker| STT_Pool
        UserProfile -.->|Get preferences| DM
        ContextDB -.->|Get context| Context
        PermissionMgr -.->|Check access| ActionRouter

        %% Caching
        Redis_Session <--> StateTracker
        Redis_Intent <--> Intent
        Redis_TTS <--> TTS_Pool
        Redis_User <--> UserProfile

        %% Storage
        UserProfile <--> UserDB
        Context <--> QueryDB
        STT_Pool -.->|Save audio| AudioStore
        STT_Pool <--> ModelRegistry
        KnowledgeBase <--> KnowledgeDB

        %% Analytics
        STT_Pool --> Telemetry
        Intent --> Analytics
        TTS_Pool --> FeedbackLoop

        style WakeWord fill:#e1f5ff
        style STT_Pool fill:#fff4e1
        style Intent fill:#e8f5e9
        style DM fill:#f3e5f5
        style TTS_Pool fill:#ffe0e0
    ```

    ---

    ## Component Descriptions

    ### 1. Edge Device Layer

    **Wake Word Detector:**
    - Small neural network (< 50 MB) running on dedicated DSP
    - Continuously processes audio in 50ms frames
    - Optimized for power efficiency (< 100mW)
    - High true positive (> 99%), low false positive (< 0.1%)
    - Triggers audio buffer capture on detection

    **Audio Buffer:**
    - Circular buffer holding last 2 seconds of audio (pre-trigger)
    - Captures audio after wake word until voice activity ends
    - Sends complete utterance to cloud for processing

    **Codec:**
    - Opus codec for efficient compression (3:1 ratio)
    - 16kHz sampling rate for speech
    - Encrypted transmission (TLS 1.3)

    ---

    ### 2. Speech-to-Text (STT) Service

    **ASR Engine:**
    - Whisper (OpenAI) or Wav2Vec 2.0 (Meta) models
    - Streaming recognition with partial results
    - Multi-language support (50+ languages)
    - GPU-accelerated inference (A100 GPUs)
    - Achieves < 5% Word Error Rate (WER)

    **Voice Activity Detection (VAD):**
    - Detects speech segments in audio stream
    - Filters silence and background noise
    - Triggers end-of-utterance processing

    **Language Detection:**
    - Identifies spoken language in first 500ms
    - Routes to language-specific ASR model
    - Supports code-switching (bilingual speakers)

    ---

    ### 3. Natural Language Understanding (NLU)

    **Intent Classifier:**
    - BERT-based transformer model (340M params)
    - Multi-class classification (500+ intent classes)
    - Examples: `play_music`, `set_alarm`, `get_weather`, `answer_question`
    - Confidence score for intent (reject if < 0.7)

    **Slot Filling:**
    - Named Entity Recognition (NER) for parameter extraction
    - Extracts entities: song_name, time, location, contact_name, etc.
    - Uses CRF (Conditional Random Field) on top of BERT

    **Context Manager:**
    - Maintains dialogue history (last 5 turns)
    - Resolves anaphora ("Play that song again" → previous song entity)
    - Tracks user state (location, current activity)

    ---

    ### 4. Dialogue Management

    **Dialogue Manager:**
    - Finite state machine for conversation flow
    - Handles clarification questions ("Which alarm did you mean?")
    - Supports multi-turn dialogues with memory

    **State Tracker:**
    - Tracks conversation state (beliefs about user intent)
    - Updates slot values across turns
    - Stored in Redis for fast access

    **Policy Network:**
    - Decides next action (answer, clarify, execute, fallback)
    - Reinforcement learning trained policy
    - Handles ambiguity and errors gracefully

    ---

    ### 5. Action Execution

    **Action Router:**
    - Routes intents to appropriate backend service
    - Handles authentication and authorization
    - Executes actions (API calls, database queries, device commands)

    **Knowledge Base:**
    - Graph database (Neo4j) with entities and relationships
    - Question answering with semantic search
    - Retrieval-Augmented Generation (RAG) for factual queries

    **External APIs:**
    - Weather services, calendar integration, smart home APIs
    - Timeout and retry logic for resilience
    - Response formatting for natural language

    ---

    ### 6. Text-to-Speech (TTS)

    **TTS Engine:**
    - Neural vocoder (WaveNet, Tacotron 2, or FastSpeech 2)
    - Generates natural-sounding speech from text
    - Multiple voices (male, female, accents)
    - Streaming synthesis for low latency (first audio chunk < 400ms)

    **Prosody Modulation:**
    - Adjusts pitch, rhythm, emphasis for natural delivery
    - Emotional tone based on response type (apologetic, cheerful)
    - Handles punctuation and special formatting

    ---

    ### 7. User Services

    **Voice Profile:**
    - Speaker recognition using voice embeddings (x-vectors, d-vectors)
    - Identifies user for personalized responses
    - Handles multi-user households (separate profiles)

    **User Profile:**
    - Stores preferences (language, voice, linked accounts)
    - Privacy settings (recording opt-in, data retention)
    - Personalization data (favorite music, frequent contacts)

    ---

    ## Request Flow

    **End-to-End Flow:**

    1. **Wake Word Detection (on-device, 200ms):**
       - User says "Hey Siri" or "OK Google"
       - On-device neural network detects wake word
       - Audio buffer starts capturing audio

    2. **Audio Capture & Transmission (300ms):**
       - Capture complete utterance (e.g., "What's the weather in Seattle?")
       - Compress audio with Opus codec (96 KB → 32 KB)
       - Send encrypted audio to cloud via WebSocket

    3. **Speech-to-Text (500ms):**
       - Voice Activity Detection filters silence
       - Language detection identifies English
       - Whisper model transcribes: "what's the weather in seattle"
       - Streaming partial results sent during transcription

    4. **Natural Language Understanding (300ms):**
       - Intent classifier: `get_weather` (confidence: 0.95)
       - Slot filling: `location=Seattle, WA`
       - Context manager adds user location context

    5. **Dialogue Management (100ms):**
       - State tracker updates dialogue state
       - Policy network decides: execute action (no clarification needed)

    6. **Action Execution (200ms):**
       - Action router calls weather API with location=Seattle
       - Receives response: {"temp": 55°F, "condition": "rainy"}
       - Formats response: "It's 55 degrees and rainy in Seattle"

    7. **Text-to-Speech (400ms):**
       - TTS engine synthesizes response audio
       - First audio chunk streamed after 400ms
       - Voice modulation applied for natural delivery

    8. **Audio Playback (2s):**
       - Compressed audio sent to device
       - Device plays response through speaker

    **Total latency:** 200ms (wake) + 300ms (audio) + 500ms (STT) + 300ms (NLU) + 100ms (DM) + 200ms (action) + 400ms (TTS) = **2 seconds** (within 3s target)

    ---

    ## API Design

    ### WebSocket API (Streaming)

    **Connect:**
    ```
    WS wss://api.assistant.example.com/v1/stream
    Headers:
      Authorization: Bearer <device_token>
      X-Device-ID: <device_uuid>
      X-User-ID: <user_id>
    ```

    **Client → Server Messages:**

    ```json
    // Start session
    {
      "type": "session_start",
      "session_id": "sess_123",
      "language": "en-US",
      "voice_profile_id": "vp_456"
    }

    // Stream audio chunks
    {
      "type": "audio_chunk",
      "session_id": "sess_123",
      "audio": "<base64_encoded_audio>",
      "sequence": 1
    }

    // End utterance
    {
      "type": "audio_end",
      "session_id": "sess_123"
    }
    ```

    **Server → Client Messages:**

    ```json
    // Partial transcription
    {
      "type": "stt_partial",
      "session_id": "sess_123",
      "text": "what's the weather",
      "is_final": false
    }

    // Final transcription
    {
      "type": "stt_final",
      "session_id": "sess_123",
      "text": "what's the weather in seattle",
      "confidence": 0.98
    }

    // Intent recognized
    {
      "type": "nlu_result",
      "session_id": "sess_123",
      "intent": "get_weather",
      "confidence": 0.95,
      "slots": {
        "location": "Seattle, WA"
      }
    }

    // Response text
    {
      "type": "response_text",
      "session_id": "sess_123",
      "text": "It's 55 degrees and rainy in Seattle"
    }

    // TTS audio chunks
    {
      "type": "tts_chunk",
      "session_id": "sess_123",
      "audio": "<base64_encoded_audio>",
      "sequence": 1
    }

    // Session complete
    {
      "type": "session_end",
      "session_id": "sess_123"
    }
    ```

    ---

    ### REST API

    **Process Query (non-streaming):**
    ```
    POST /v1/query
    Content-Type: multipart/form-data

    Parameters:
      audio: <audio_file> (WAV, 16kHz, mono)
      language: en-US
      user_id: user_123
      session_id: sess_456 (optional, for multi-turn)

    Response:
    {
      "query_id": "q_789",
      "transcription": "what's the weather in seattle",
      "intent": "get_weather",
      "slots": {
        "location": "Seattle, WA"
      },
      "response_text": "It's 55 degrees and rainy in Seattle",
      "response_audio_url": "https://cdn.example.com/tts/abc123.mp3",
      "session_id": "sess_456"
    }
    ```

    **Get User Profile:**
    ```
    GET /v1/users/{user_id}/profile

    Response:
    {
      "user_id": "user_123",
      "language": "en-US",
      "voice_preference": "female_voice_1",
      "linked_accounts": {
        "spotify": "linked",
        "calendar": "linked"
      },
      "privacy_settings": {
        "save_audio": false,
        "personalization": true
      }
    }
    ```

    **Update Dialogue Context:**
    ```
    POST /v1/dialogue/context
    Content-Type: application/json

    {
      "session_id": "sess_456",
      "context": {
        "last_song_played": "song_789",
        "last_location": "Seattle, WA",
        "user_intent_history": ["get_weather", "play_music"]
      }
    }

    Response:
    {
      "session_id": "sess_456",
      "context_updated": true
    }
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Wake Word Detection

    Wake word detection is the always-on listening component that triggers the voice assistant. It must be highly accurate (low false positives/negatives), power-efficient, and run on-device for privacy.

    ### Architecture

    **Small Neural Network:**
    - Typically 1-5 layers, < 100K parameters
    - Input: Mel-frequency cepstral coefficients (MFCCs) or raw waveform
    - Output: Binary classification (wake word detected / not detected)
    - Models: Depthwise separable convolutions, LSTM, GRU, or attention

    **Processing Pipeline:**
    1. Audio captured at 16 kHz
    2. Compute features (MFCCs) every 50ms frame
    3. Feed frames to neural network
    4. Threshold probability (> 0.9 = detected)
    5. Post-processing to reduce false positives (temporal smoothing)

    ### Implementation

    ```python
    import torch
    import torch.nn as nn
    import torchaudio

    class WakeWordDetector(nn.Module):
        """
        Lightweight wake word detection model using depthwise separable convolutions.
        Optimized for on-device inference with minimal parameters.
        """
        def __init__(self, n_mfcc=40, hidden_dim=128, output_dim=1):
            super().__init__()

            # Depthwise separable conv blocks (efficient)
            self.conv1 = nn.Sequential(
                nn.Conv1d(n_mfcc, n_mfcc, kernel_size=5, padding=2, groups=n_mfcc),
                nn.Conv1d(n_mfcc, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )

            self.conv2 = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )

            # Temporal attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, output_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            # x: (batch, n_mfcc, time_steps)
            x = self.conv1(x)
            x = self.conv2(x)

            # Transpose for attention: (batch, time, features)
            x = x.transpose(1, 2)

            # Compute attention weights
            attn_weights = torch.softmax(self.attention(x), dim=1)

            # Weighted sum
            x = torch.sum(x * attn_weights, dim=1)

            # Classification
            return self.classifier(x)

    class WakeWordDetectionPipeline:
        """
        End-to-end wake word detection pipeline.
        Processes audio stream in real-time.
        """
        def __init__(self, model_path, threshold=0.9):
            self.model = WakeWordDetector()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

            self.threshold = threshold
            self.sample_rate = 16000
            self.frame_length = 0.05  # 50ms frames
            self.n_mfcc = 40

            # MFCC transform
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
            )

            # Circular buffer for temporal smoothing
            self.detection_buffer = []
            self.buffer_size = 5  # Require 5 consecutive detections

        def process_audio_chunk(self, audio_chunk):
            """
            Process single audio chunk (50ms).

            Args:
                audio_chunk: Tensor of shape (samples,) at 16kHz

            Returns:
                is_detected: bool, True if wake word detected
                confidence: float, detection confidence
            """
            # Compute MFCCs
            mfccs = self.mfcc_transform(audio_chunk.unsqueeze(0))  # (1, n_mfcc, time)

            # Inference
            with torch.no_grad():
                prob = self.model(mfccs).item()

            # Temporal smoothing (reduce false positives)
            self.detection_buffer.append(prob > self.threshold)
            if len(self.detection_buffer) > self.buffer_size:
                self.detection_buffer.pop(0)

            # Detect if majority of recent frames are positive
            is_detected = sum(self.detection_buffer) >= (self.buffer_size * 0.6)

            return is_detected, prob

        def stream_detection(self, audio_stream):
            """
            Continuously process audio stream and detect wake word.

            Args:
                audio_stream: Generator yielding audio chunks

            Yields:
                (timestamp, is_detected, confidence) tuples
            """
            frame_samples = int(self.frame_length * self.sample_rate)

            for timestamp, audio_chunk in audio_stream:
                # Ensure correct chunk size
                if len(audio_chunk) != frame_samples:
                    continue

                is_detected, confidence = self.process_audio_chunk(audio_chunk)

                if is_detected:
                    yield (timestamp, True, confidence)
                    # Clear buffer after detection to avoid multiple triggers
                    self.detection_buffer.clear()

    # Example usage
    if __name__ == "__main__":
        # Initialize pipeline
        detector = WakeWordDetectionPipeline(
            model_path="wake_word_model.pth",
            threshold=0.9
        )

        # Simulate audio stream (in production, this comes from microphone)
        def audio_stream_generator():
            # Load test audio file
            waveform, sr = torchaudio.load("test_audio.wav")

            frame_samples = int(0.05 * sr)
            for i in range(0, waveform.shape[1] - frame_samples, frame_samples):
                chunk = waveform[0, i:i+frame_samples]
                timestamp = i / sr
                yield timestamp, chunk

        # Detect wake word
        for timestamp, detected, confidence in detector.stream_detection(audio_stream_generator()):
            print(f"[{timestamp:.2f}s] Wake word detected! Confidence: {confidence:.3f}")
            print("Starting voice query processing...")
            break
    ```

    **Key Optimizations:**
    - **Depthwise separable convolutions:** Reduce parameters by 8-10x vs standard convolutions
    - **Temporal smoothing:** Require multiple consecutive positive frames to reduce false positives
    - **Dedicated DSP:** Run on low-power Digital Signal Processor (not main CPU) for battery efficiency
    - **Quantization:** Use INT8 quantization to reduce model size and inference time

    **Trade-offs:**
    - **Accuracy vs Power:** Larger models are more accurate but consume more power
    - **False Positives vs False Negatives:** Tuning threshold balances accidental triggers vs missed queries
    - **On-device vs Cloud:** On-device protects privacy but limits model complexity

    ---

    ## 3.2 Speech-to-Text (Streaming ASR)

    Automatic Speech Recognition (ASR) converts spoken audio to text. Modern systems use transformer-based models like Whisper (OpenAI) or Wav2Vec 2.0 (Meta) for high accuracy.

    ### Architecture

    **Whisper Model:**
    - Encoder-decoder transformer architecture
    - Encoder: Process audio features (80-channel log-Mel spectrogram)
    - Decoder: Generate text tokens autoregressively
    - Trained on 680K hours of multilingual audio
    - Sizes: tiny (39M), base (74M), small (244M), medium (769M), large (1.5B params)

    **Streaming Recognition:**
    - Chunk audio into overlapping segments (30s windows)
    - Process incrementally, emit partial transcriptions
    - Use language model for context and corrections

    ### Implementation

    ```python
    import whisper
    import torch
    import numpy as np
    from collections import deque

    class StreamingASR:
        """
        Streaming Automatic Speech Recognition using Whisper.
        Processes audio in chunks and emits partial/final transcriptions.
        """
        def __init__(self, model_name="base", device="cuda", language="en"):
            self.model = whisper.load_model(model_name).to(device)
            self.device = device
            self.language = language

            # Streaming parameters
            self.sample_rate = 16000
            self.chunk_duration = 10  # Process 10s chunks
            self.chunk_overlap = 2     # 2s overlap for context

            # Audio buffer
            self.audio_buffer = deque(maxlen=int(self.chunk_duration * self.sample_rate))

            # Transcription state
            self.previous_text = ""
            self.final_text = ""

        def preprocess_audio(self, audio_chunk):
            """
            Convert audio to format expected by Whisper.

            Args:
                audio_chunk: numpy array, shape (samples,), 16kHz mono

            Returns:
                Preprocessed audio tensor
            """
            # Ensure float32 and normalize
            audio = audio_chunk.astype(np.float32) / 32768.0

            # Pad or trim to 30s (Whisper's expected input length)
            target_length = 30 * self.sample_rate
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            return audio

        def transcribe_chunk(self, audio_chunk, is_final=False):
            """
            Transcribe single audio chunk.

            Args:
                audio_chunk: numpy array of audio samples
                is_final: bool, whether this is the final chunk

            Returns:
                dict with 'text', 'is_final', 'confidence'
            """
            # Preprocess
            audio = self.preprocess_audio(audio_chunk)

            # Transcribe
            options = {
                "language": self.language,
                "task": "transcribe",
                "fp16": torch.cuda.is_available(),
                "beam_size": 5 if is_final else 1,  # Use beam search for final
                "best_of": 5 if is_final else 1
            }

            result = self.model.transcribe(audio, **options)

            # Extract text and confidence
            text = result["text"].strip()

            # For streaming, only return new text (remove overlap with previous)
            if not is_final:
                # Remove common prefix with previous transcription
                new_text = self._extract_new_text(text, self.previous_text)
                self.previous_text = text
            else:
                new_text = text
                self.final_text = text

            return {
                "text": new_text,
                "full_text": text,
                "is_final": is_final,
                "confidence": self._compute_confidence(result)
            }

        def _extract_new_text(self, current_text, previous_text):
            """
            Extract new words that weren't in previous partial transcription.
            """
            current_words = current_text.split()
            previous_words = previous_text.split()

            # Find where texts diverge
            common_length = 0
            for i, (curr, prev) in enumerate(zip(current_words, previous_words)):
                if curr == prev:
                    common_length = i + 1
                else:
                    break

            # Return new words
            new_words = current_words[common_length:]
            return " ".join(new_words)

        def _compute_confidence(self, result):
            """
            Compute average confidence from segment-level probabilities.
            """
            if "segments" in result:
                confidences = []
                for segment in result["segments"]:
                    if "avg_logprob" in segment:
                        # Convert log probability to confidence
                        conf = np.exp(segment["avg_logprob"])
                        confidences.append(conf)

                return np.mean(confidences) if confidences else 0.0
            return 0.0

        def stream_transcribe(self, audio_stream):
            """
            Process audio stream and yield partial/final transcriptions.

            Args:
                audio_stream: Generator yielding (audio_chunk, is_final) tuples

            Yields:
                Transcription results
            """
            for audio_chunk, is_final in audio_stream:
                # Add to buffer
                self.audio_buffer.extend(audio_chunk)

                # Transcribe when we have enough audio or final chunk
                if len(self.audio_buffer) >= self.chunk_duration * self.sample_rate or is_final:
                    audio_array = np.array(self.audio_buffer)

                    result = self.transcribe_chunk(audio_array, is_final=is_final)

                    yield result

                    if is_final:
                        # Reset state
                        self.audio_buffer.clear()
                        self.previous_text = ""

    class LanguageDetector:
        """
        Detect spoken language from audio using Whisper's built-in detection.
        """
        def __init__(self, model_name="base", device="cuda"):
            self.model = whisper.load_model(model_name).to(device)
            self.device = device

        def detect_language(self, audio_chunk, top_k=3):
            """
            Detect language from audio chunk.

            Args:
                audio_chunk: numpy array, first 3-5 seconds of audio
                top_k: return top k languages

            Returns:
                List of (language_code, probability) tuples
            """
            # Preprocess audio
            audio = audio_chunk.astype(np.float32) / 32768.0
            audio = whisper.pad_or_trim(audio)

            # Detect language
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            _, probs = self.model.detect_language(mel)

            # Get top k languages
            top_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]

            return top_langs

    # Example usage
    if __name__ == "__main__":
        # Initialize ASR
        asr = StreamingASR(model_name="base", device="cuda", language="en")

        # Simulate audio stream (in production, this comes from microphone/network)
        def audio_stream_generator():
            # Load audio file
            audio, sr = librosa.load("query_audio.wav", sr=16000)

            # Chunk into 1s segments
            chunk_size = 16000  # 1 second
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                is_final = (i + chunk_size >= len(audio))
                yield chunk, is_final

        # Transcribe stream
        print("Streaming transcription:")
        for result in asr.stream_transcribe(audio_stream_generator()):
            if result["is_final"]:
                print(f"\nFinal: {result['full_text']}")
                print(f"Confidence: {result['confidence']:.3f}")
            else:
                print(f"Partial: {result['text']}", end=" ", flush=True)

        # Language detection
        print("\n\nLanguage detection:")
        detector = LanguageDetector(model_name="base", device="cuda")
        audio_sample, _ = librosa.load("query_audio.wav", sr=16000, duration=3)
        top_langs = detector.detect_language(audio_sample, top_k=3)
        for lang, prob in top_langs:
            print(f"  {lang}: {prob:.3f}")
    ```

    **Key Optimizations:**
    - **Streaming with overlap:** Process chunks with overlap to avoid cutting words
    - **Beam search for final:** Use greedy decoding for speed (partial), beam search for accuracy (final)
    - **GPU batching:** Batch multiple concurrent requests for throughput
    - **Model quantization:** Use INT8 quantization to reduce memory and latency
    - **Language detection caching:** Detect once per session, reuse for subsequent chunks

    ---

    ## 3.3 Natural Language Understanding (Intent Classification & Slot Filling)

    NLU extracts meaning from transcribed text: what the user wants (intent) and the parameters (slots/entities).

    ### Architecture

    **BERT-based NLU:**
    - Pre-trained BERT model fine-tuned for intent + slot filling
    - Intent classification: Multi-class classification head on [CLS] token
    - Slot filling: Token classification (BIO tagging) on each word
    - Joint training improves performance (intents help slot prediction)

    **Intent Taxonomy:**
    - 500+ intent classes grouped into domains
    - Examples: `play_music`, `set_alarm`, `get_weather`, `send_message`, `answer_question`

    **Slot Types:**
    - Categorical: song_name, artist, contact_name, location
    - Temporal: time, date, duration
    - Numeric: temperature, quantity

    ### Implementation

    ```python
    import torch
    import torch.nn as nn
    from transformers import BertModel, BertTokenizer

    class JointIntentSlotModel(nn.Module):
        """
        Joint Intent Classification and Slot Filling model.
        Uses BERT encoder with two classification heads.
        """
        def __init__(self, num_intents, num_slots, bert_model="bert-base-uncased", dropout=0.3):
            super().__init__()

            # BERT encoder
            self.bert = BertModel.from_pretrained(bert_model)
            hidden_size = self.bert.config.hidden_size

            # Intent classification head (use [CLS] token)
            self.intent_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_intents)
            )

            # Slot filling head (token classification)
            self.slot_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_slots)
            )

        def forward(self, input_ids, attention_mask):
            """
            Forward pass.

            Args:
                input_ids: (batch, seq_len)
                attention_mask: (batch, seq_len)

            Returns:
                intent_logits: (batch, num_intents)
                slot_logits: (batch, seq_len, num_slots)
            """
            # BERT encoding
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            # [CLS] token for intent
            cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
            intent_logits = self.intent_classifier(cls_output)

            # All tokens for slots
            sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            slot_logits = self.slot_classifier(sequence_output)

            return intent_logits, slot_logits

    class NLUService:
        """
        Natural Language Understanding service.
        Performs intent classification and slot filling.
        """
        def __init__(self, model_path, intent_labels, slot_labels, device="cuda"):
            self.device = device
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            # Load model
            self.model = JointIntentSlotModel(
                num_intents=len(intent_labels),
                num_slots=len(slot_labels)
            ).to(device)

            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Label mappings
            self.intent_labels = intent_labels
            self.slot_labels = slot_labels
            self.intent_id_to_label = {i: label for i, label in enumerate(intent_labels)}
            self.slot_id_to_label = {i: label for i, label in enumerate(slot_labels)}

        def predict(self, text):
            """
            Predict intent and slots for input text.

            Args:
                text: str, transcribed user query

            Returns:
                dict with intent, confidence, and extracted slots
            """
            # Tokenize
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            # Inference
            with torch.no_grad():
                intent_logits, slot_logits = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"]
                )

            # Intent prediction
            intent_probs = torch.softmax(intent_logits, dim=-1)
            intent_id = torch.argmax(intent_probs, dim=-1).item()
            intent_confidence = intent_probs[0, intent_id].item()
            intent_label = self.intent_id_to_label[intent_id]

            # Slot prediction (BIO tagging)
            slot_ids = torch.argmax(slot_logits, dim=-1)[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

            slots = self._extract_slots(tokens, slot_ids)

            return {
                "text": text,
                "intent": intent_label,
                "intent_confidence": intent_confidence,
                "slots": slots
            }

        def _extract_slots(self, tokens, slot_ids):
            """
            Extract slot values from BIO-tagged tokens.

            Args:
                tokens: list of subword tokens
                slot_ids: list of slot label IDs

            Returns:
                dict of {slot_name: slot_value}
            """
            slots = {}
            current_slot = None
            current_value = []

            for token, slot_id in zip(tokens, slot_ids):
                # Skip special tokens
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue

                slot_label = self.slot_id_to_label[slot_id]

                if slot_label == "O":  # Outside any slot
                    if current_slot:
                        # Save previous slot
                        slots[current_slot] = self._merge_subwords(current_value)
                        current_slot = None
                        current_value = []

                elif slot_label.startswith("B-"):  # Begin slot
                    if current_slot:
                        # Save previous slot
                        slots[current_slot] = self._merge_subwords(current_value)

                    current_slot = slot_label[2:]  # Remove "B-" prefix
                    current_value = [token]

                elif slot_label.startswith("I-"):  # Inside slot
                    if current_slot == slot_label[2:]:
                        current_value.append(token)
                    else:
                        # Mismatch, treat as new slot
                        if current_slot:
                            slots[current_slot] = self._merge_subwords(current_value)
                        current_slot = slot_label[2:]
                        current_value = [token]

            # Save last slot
            if current_slot:
                slots[current_slot] = self._merge_subwords(current_value)

            return slots

        def _merge_subwords(self, subword_tokens):
            """
            Merge BERT subword tokens into complete words.
            """
            text = " ".join(subword_tokens)
            text = text.replace(" ##", "")  # Remove subword markers
            return text.strip()

    class ContextManager:
        """
        Manages dialogue context and resolves references.
        Tracks conversation history and user state.
        """
        def __init__(self, session_id, user_id):
            self.session_id = session_id
            self.user_id = user_id

            # Dialogue history (last N turns)
            self.history = []
            self.max_history = 5

            # Slot memory (carry over slots from previous turns)
            self.slot_memory = {}

            # User context
            self.user_context = {
                "location": None,
                "last_song": None,
                "last_contact": None,
                "last_entity": None
            }

        def update_context(self, nlu_result):
            """
            Update context with new NLU result.
            Resolve references and fill missing slots.

            Args:
                nlu_result: dict from NLUService.predict()

            Returns:
                Updated nlu_result with resolved slots
            """
            # Add to history
            self.history.append({
                "text": nlu_result["text"],
                "intent": nlu_result["intent"],
                "slots": nlu_result["slots"]
            })

            if len(self.history) > self.max_history:
                self.history.pop(0)

            # Resolve references (e.g., "that song", "her", "there")
            resolved_slots = self._resolve_references(nlu_result["slots"])

            # Merge with slot memory (carry-over from previous turn)
            merged_slots = {**self.slot_memory, **resolved_slots}

            # Update slot memory
            self.slot_memory = merged_slots

            # Update user context
            self._update_user_context(merged_slots)

            # Return updated result
            return {
                **nlu_result,
                "slots": merged_slots,
                "context": self.user_context
            }

        def _resolve_references(self, slots):
            """
            Resolve anaphoric references (pronouns, demonstratives).
            """
            resolved = slots.copy()

            # Example: "that song" → previous song entity
            if "song_name" in slots and slots["song_name"].lower() in ["that", "it", "the same"]:
                if self.user_context["last_song"]:
                    resolved["song_name"] = self.user_context["last_song"]

            # Example: "call her" → previous contact
            if "contact_name" in slots and slots["contact_name"].lower() in ["him", "her", "them"]:
                if self.user_context["last_contact"]:
                    resolved["contact_name"] = self.user_context["last_contact"]

            # Example: "weather there" → previous location
            if "location" in slots and slots["location"].lower() in ["there", "that place"]:
                if self.user_context["location"]:
                    resolved["location"] = self.user_context["location"]

            return resolved

        def _update_user_context(self, slots):
            """Update user context with new slot values."""
            if "location" in slots:
                self.user_context["location"] = slots["location"]
            if "song_name" in slots:
                self.user_context["last_song"] = slots["song_name"]
            if "contact_name" in slots:
                self.user_context["last_contact"] = slots["contact_name"]

    # Example usage
    if __name__ == "__main__":
        # Intent and slot labels
        intent_labels = ["play_music", "set_alarm", "get_weather", "send_message", "answer_question"]
        slot_labels = ["O", "B-song_name", "I-song_name", "B-artist", "I-artist",
                      "B-time", "I-time", "B-location", "I-location", "B-contact_name", "I-contact_name"]

        # Initialize NLU service
        nlu = NLUService(
            model_path="nlu_model.pth",
            intent_labels=intent_labels,
            slot_labels=slot_labels,
            device="cuda"
        )

        # Initialize context manager
        context = ContextManager(session_id="sess_123", user_id="user_456")

        # Example 1: Play music
        result1 = nlu.predict("play bohemian rhapsody by queen")
        result1 = context.update_context(result1)
        print("Query 1:", result1)
        # Output: {intent: "play_music", slots: {song_name: "bohemian rhapsody", artist: "queen"}}

        # Example 2: Follow-up (reference resolution)
        result2 = nlu.predict("play that song again")
        result2 = context.update_context(result2)
        print("Query 2:", result2)
        # Output: {intent: "play_music", slots: {song_name: "bohemian rhapsody"}}  # Resolved!

        # Example 3: Weather query
        result3 = nlu.predict("what's the weather in seattle")
        result3 = context.update_context(result3)
        print("Query 3:", result3)
        # Output: {intent: "get_weather", slots: {location: "seattle"}}
    ```

    **Key Optimizations:**
    - **Joint training:** Train intent + slot models together for better performance
    - **Context tracking:** Maintain dialogue history for reference resolution
    - **Caching:** Cache common intent-slot combinations
    - **Model distillation:** Distill large BERT to smaller model (DistilBERT, MobileBERT) for lower latency

    ---

    ## 3.4 Dialogue Management & Multi-turn Conversations

    Dialogue management decides how the system should respond: answer immediately, ask for clarification, or execute an action.

    ### Architecture

    **Components:**
    1. **State Tracker:** Tracks conversation state (beliefs about user goals)
    2. **Policy Network:** Decides next action (reinforcement learning)
    3. **Response Generator:** Generates natural language responses

    **Dialogue States:**
    - Intent understood, all slots filled → Execute action
    - Intent understood, slots missing → Ask clarification questions
    - Intent ambiguous → Offer choices
    - Intent unknown → Fallback response

    ### Implementation

    ```python
    from enum import Enum
    from typing import Dict, List, Optional
    import random

    class DialogueAction(Enum):
        """Possible dialogue actions."""
        EXECUTE = "execute"           # Execute user request
        CLARIFY_SLOT = "clarify_slot"  # Ask for missing slot value
        CONFIRM = "confirm"           # Confirm before executing
        DISAMBIGUATE = "disambiguate"  # Choose between multiple options
        FALLBACK = "fallback"         # Cannot understand request
        CHITCHAT = "chitchat"         # Casual conversation

    class DialogueState:
        """
        Represents the current state of a dialogue.
        """
        def __init__(self, session_id: str):
            self.session_id = session_id

            # Current intent and slots
            self.intent: Optional[str] = None
            self.intent_confidence: float = 0.0
            self.slots: Dict[str, str] = {}

            # Required slots for current intent
            self.required_slots: List[str] = []
            self.filled_slots: List[str] = []

            # Dialogue action
            self.last_action: Optional[DialogueAction] = None

            # Turn count
            self.turn_count: int = 0

            # Clarification state
            self.pending_clarification: Optional[str] = None

        def update_from_nlu(self, nlu_result: Dict):
            """Update state from NLU result."""
            self.intent = nlu_result["intent"]
            self.intent_confidence = nlu_result["intent_confidence"]
            self.slots.update(nlu_result["slots"])

            self.turn_count += 1

        def is_complete(self) -> bool:
            """Check if all required slots are filled."""
            return all(slot in self.slots for slot in self.required_slots)

        def get_missing_slots(self) -> List[str]:
            """Get list of missing required slots."""
            return [slot for slot in self.required_slots if slot not in self.slots]

    class DialogueManager:
        """
        Manages multi-turn dialogue and decides next action.
        """
        def __init__(self):
            # Intent to required slots mapping
            self.intent_schemas = {
                "play_music": ["song_name", "artist"],
                "set_alarm": ["time"],
                "get_weather": ["location"],
                "send_message": ["contact_name", "message_content"],
                "make_call": ["contact_name"],
                "set_reminder": ["reminder_content", "time"],
                "search_web": ["query"]
            }

            # Slot clarification prompts
            self.clarification_prompts = {
                "song_name": "What song would you like to play?",
                "artist": "Which artist?",
                "time": "What time?",
                "location": "Which location?",
                "contact_name": "Who would you like to contact?",
                "message_content": "What should the message say?",
                "reminder_content": "What should I remind you about?",
                "query": "What would you like to search for?"
            }

            # Active dialogue states (session_id -> DialogueState)
            self.active_dialogues: Dict[str, DialogueState] = {}

        def process_turn(self, session_id: str, nlu_result: Dict) -> Dict:
            """
            Process a single dialogue turn.

            Args:
                session_id: Unique session identifier
                nlu_result: NLU result with intent and slots

            Returns:
                dict with action, response_text, and execution details
            """
            # Get or create dialogue state
            if session_id not in self.active_dialogues:
                state = DialogueState(session_id)
                self.active_dialogues[session_id] = state
            else:
                state = self.active_dialogues[session_id]

            # Update state from NLU
            state.update_from_nlu(nlu_result)

            # Check if answering clarification question
            if state.pending_clarification:
                return self._handle_clarification_response(state, nlu_result)

            # Get required slots for intent
            state.required_slots = self.intent_schemas.get(state.intent, [])

            # Decide next action
            action = self._select_action(state)

            # Generate response
            response = self._generate_response(state, action)

            return response

        def _select_action(self, state: DialogueState) -> DialogueAction:
            """
            Select next dialogue action using policy.

            Args:
                state: Current dialogue state

            Returns:
                DialogueAction
            """
            # Low confidence intent → fallback
            if state.intent_confidence < 0.7:
                return DialogueAction.FALLBACK

            # Chitchat intents
            if state.intent in ["greeting", "goodbye", "thanks"]:
                return DialogueAction.CHITCHAT

            # Check if all required slots filled
            missing_slots = state.get_missing_slots()

            if not missing_slots:
                # All slots filled → execute
                # Optional: confirm for sensitive actions
                if state.intent in ["send_message", "make_call"] and state.last_action != DialogueAction.CONFIRM:
                    return DialogueAction.CONFIRM
                else:
                    return DialogueAction.EXECUTE
            else:
                # Missing slots → clarify
                return DialogueAction.CLARIFY_SLOT

        def _generate_response(self, state: DialogueState, action: DialogueAction) -> Dict:
            """
            Generate response based on action.

            Args:
                state: Current dialogue state
                action: Selected dialogue action

            Returns:
                dict with response details
            """
            response = {
                "session_id": state.session_id,
                "action": action.value,
                "intent": state.intent,
                "slots": state.slots,
                "response_text": "",
                "should_execute": False,
                "end_session": False
            }

            if action == DialogueAction.EXECUTE:
                response["response_text"] = self._generate_execution_response(state)
                response["should_execute"] = True
                response["end_session"] = True

            elif action == DialogueAction.CLARIFY_SLOT:
                missing_slot = state.get_missing_slots()[0]
                response["response_text"] = self.clarification_prompts.get(
                    missing_slot,
                    f"Could you specify the {missing_slot}?"
                )
                state.pending_clarification = missing_slot
                state.last_action = action

            elif action == DialogueAction.CONFIRM:
                response["response_text"] = self._generate_confirmation_prompt(state)
                state.last_action = action

            elif action == DialogueAction.FALLBACK:
                response["response_text"] = "I'm sorry, I didn't understand that. Could you rephrase?"
                response["end_session"] = False

            elif action == DialogueAction.CHITCHAT:
                response["response_text"] = self._generate_chitchat_response(state)
                response["end_session"] = True

            return response

        def _generate_execution_response(self, state: DialogueState) -> str:
            """Generate confirmation message for execution."""
            intent = state.intent
            slots = state.slots

            templates = {
                "play_music": f"Playing {slots.get('song_name', 'music')} by {slots.get('artist', 'the artist')}",
                "set_alarm": f"Alarm set for {slots.get('time')}",
                "get_weather": f"Getting weather for {slots.get('location')}",
                "send_message": f"Sending message to {slots.get('contact_name')}",
                "make_call": f"Calling {slots.get('contact_name')}",
                "set_reminder": f"Reminder set for {slots.get('time')}: {slots.get('reminder_content')}"
            }

            return templates.get(intent, "Executing your request")

        def _generate_confirmation_prompt(self, state: DialogueState) -> str:
            """Generate confirmation question for sensitive actions."""
            intent = state.intent
            slots = state.slots

            if intent == "send_message":
                return f"Do you want to send '{slots.get('message_content')}' to {slots.get('contact_name')}?"
            elif intent == "make_call":
                return f"Should I call {slots.get('contact_name')}?"
            else:
                return "Should I proceed?"

        def _generate_chitchat_response(self, state: DialogueState) -> str:
            """Generate response for chitchat intents."""
            templates = {
                "greeting": ["Hello!", "Hi there!", "Hey, how can I help?"],
                "goodbye": ["Goodbye!", "See you later!", "Have a great day!"],
                "thanks": ["You're welcome!", "Happy to help!", "Anytime!"]
            }

            responses = templates.get(state.intent, ["I'm here to help!"])
            return random.choice(responses)

        def _handle_clarification_response(self, state: DialogueState, nlu_result: Dict) -> Dict:
            """
            Handle user response to clarification question.
            Extract slot value from response.
            """
            # Extract slot value from user response
            # (In production, use NER or regex to extract value)
            slot_name = state.pending_clarification
            slot_value = nlu_result.get("text", "").strip()

            # Update slots
            state.slots[slot_name] = slot_value
            state.pending_clarification = None

            # Continue dialogue
            return self.process_turn(state.session_id, nlu_result)

        def end_session(self, session_id: str):
            """End dialogue session and clean up state."""
            if session_id in self.active_dialogues:
                del self.active_dialogues[session_id]

    # Example usage
    if __name__ == "__main__":
        dm = DialogueManager()
        session_id = "sess_123"

        # Turn 1: Play music (missing artist)
        nlu_result_1 = {
            "text": "play bohemian rhapsody",
            "intent": "play_music",
            "intent_confidence": 0.95,
            "slots": {"song_name": "bohemian rhapsody"}
        }
        response_1 = dm.process_turn(session_id, nlu_result_1)
        print(f"Turn 1: {response_1['response_text']}")
        # Output: "Which artist?"

        # Turn 2: Provide artist
        nlu_result_2 = {
            "text": "queen",
            "intent": "play_music",
            "intent_confidence": 0.90,
            "slots": {"artist": "queen"}
        }
        response_2 = dm.process_turn(session_id, nlu_result_2)
        print(f"Turn 2: {response_2['response_text']}")
        # Output: "Playing bohemian rhapsody by queen"

        # Turn 3: New session - send message (confirm)
        session_id_2 = "sess_456"
        nlu_result_3 = {
            "text": "send a message to john saying i'll be late",
            "intent": "send_message",
            "intent_confidence": 0.92,
            "slots": {"contact_name": "john", "message_content": "i'll be late"}
        }
        response_3 = dm.process_turn(session_id_2, nlu_result_3)
        print(f"Turn 3: {response_3['response_text']}")
        # Output: "Do you want to send 'i'll be late' to john?"
    ```

    **Key Concepts:**
    - **Slot carryover:** Remember slots from previous turns
    - **Clarification:** Ask for missing required information
    - **Confirmation:** Confirm before sensitive actions (send message, make call)
    - **Fallback:** Handle low-confidence or unknown intents gracefully
    - **Multi-turn tracking:** Maintain state across conversation

    ---

    ## 3.5 Text-to-Speech (Neural TTS)

    Text-to-Speech converts response text to natural-sounding audio. Modern systems use neural vocoders for high-quality synthesis.

    ### Architecture

    **Two-stage TTS:**
    1. **Acoustic Model:** Converts text to mel-spectrogram (Tacotron 2, FastSpeech 2)
    2. **Vocoder:** Converts mel-spectrogram to audio waveform (WaveNet, WaveGlow, HiFi-GAN)

    **Key Features:**
    - Natural prosody (rhythm, intonation)
    - Multiple voices and accents
    - Emotional tone control
    - Streaming synthesis (incremental audio generation)

    ### Implementation

    ```python
    import torch
    import numpy as np
    from scipy.io import wavfile
    import re

    class TextNormalizer:
        """
        Normalize text for TTS (expand abbreviations, numbers, etc.).
        """
        def __init__(self):
            # Number to text mapping
            self.num_to_text = {
                "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
            }

        def normalize(self, text: str) -> str:
            """
            Normalize text for TTS.

            Args:
                text: Raw text

            Returns:
                Normalized text
            """
            # Convert to lowercase
            text = text.lower()

            # Expand common abbreviations
            text = text.replace("mr.", "mister")
            text = text.replace("mrs.", "misses")
            text = text.replace("dr.", "doctor")
            text = text.replace("st.", "street")
            text = text.replace("etc.", "etcetera")

            # Expand numbers (simple version)
            text = re.sub(r'\b(\d+)\b', lambda m: self._number_to_words(int(m.group(1))), text)

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        def _number_to_words(self, num: int) -> str:
            """Convert number to words (simplified)."""
            if num < 10:
                return self.num_to_text[str(num)]
            elif num < 20:
                teens = ["ten", "eleven", "twelve", "thirteen", "fourteen",
                        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
                return teens[num - 10]
            elif num < 100:
                tens = ["", "", "twenty", "thirty", "forty", "fifty",
                       "sixty", "seventy", "eighty", "ninety"]
                return tens[num // 10] + ("" if num % 10 == 0 else " " + self.num_to_text[str(num % 10)])
            else:
                # For larger numbers, use built-in conversion or library
                return str(num)

    class NeuralTTS:
        """
        Neural Text-to-Speech system.
        Uses pre-trained acoustic model + vocoder.
        """
        def __init__(self, model_path: str, device: str = "cuda"):
            self.device = device
            self.normalizer = TextNormalizer()

            # Load TTS model (placeholder - in production use Tacotron2/FastSpeech2)
            # For demo, we'll use a simplified interface
            # In practice: self.model = load_pretrained_tts_model(model_path)

            self.sample_rate = 22050

            # Voice parameters
            self.voice_id = 0  # Default voice
            self.speaking_rate = 1.0  # Normal speed
            self.pitch_shift = 0.0  # No pitch shift

        def synthesize(self, text: str, streaming: bool = False):
            """
            Synthesize speech from text.

            Args:
                text: Text to synthesize
                streaming: If True, yield audio chunks incrementally

            Returns:
                Audio waveform (numpy array) or generator of chunks
            """
            # Normalize text
            normalized_text = self.normalizer.normalize(text)

            # In production, call actual TTS model:
            # mel_spectrogram = self.acoustic_model(normalized_text)
            # audio = self.vocoder(mel_spectrogram)

            # Placeholder: generate synthetic audio (for demo)
            # In real system, replace with actual model inference
            duration = len(normalized_text) * 0.1  # Rough estimate
            num_samples = int(duration * self.sample_rate)

            if streaming:
                # Streaming synthesis: yield chunks as they're generated
                chunk_size = self.sample_rate // 4  # 250ms chunks
                for i in range(0, num_samples, chunk_size):
                    chunk = self._generate_audio_chunk(normalized_text, i, chunk_size)
                    yield chunk
            else:
                # Non-streaming: return full audio
                audio = self._generate_full_audio(normalized_text, num_samples)
                return audio

        def _generate_audio_chunk(self, text: str, offset: int, chunk_size: int):
            """
            Generate single audio chunk (placeholder).
            In production, this calls the TTS model incrementally.
            """
            # Placeholder: sine wave (replace with actual TTS)
            t = np.arange(offset, offset + chunk_size) / self.sample_rate
            frequency = 440  # A4 note
            audio_chunk = 0.3 * np.sin(2 * np.pi * frequency * t)
            return audio_chunk.astype(np.float32)

        def _generate_full_audio(self, text: str, num_samples: int):
            """
            Generate full audio waveform (placeholder).
            """
            # Placeholder: sine wave (replace with actual TTS)
            t = np.arange(num_samples) / self.sample_rate
            frequency = 440
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            return audio.astype(np.float32)

        def set_voice(self, voice_id: int):
            """Set voice ID (different speakers)."""
            self.voice_id = voice_id

        def set_speaking_rate(self, rate: float):
            """Set speaking rate (0.5 = slow, 1.0 = normal, 2.0 = fast)."""
            self.speaking_rate = rate

        def set_pitch(self, pitch_shift: float):
            """Set pitch shift in semitones."""
            self.pitch_shift = pitch_shift

    class TTSService:
        """
        High-level TTS service with caching and streaming.
        """
        def __init__(self, model_path: str, cache_size: int = 10000):
            self.tts = NeuralTTS(model_path, device="cuda")

            # Response cache (common phrases)
            self.cache = {}
            self.cache_size = cache_size

        def synthesize_response(self, text: str, streaming: bool = True, use_cache: bool = True):
            """
            Synthesize response with caching.

            Args:
                text: Response text
                streaming: Enable streaming synthesis
                use_cache: Use cache for common responses

            Returns:
                Audio or audio stream
            """
            # Check cache
            if use_cache and text in self.cache:
                return self.cache[text]

            # Synthesize
            if streaming:
                # Return generator for streaming
                return self.tts.synthesize(text, streaming=True)
            else:
                # Synthesize full audio
                audio = self.tts.synthesize(text, streaming=False)

                # Cache if small enough
                if len(self.cache) < self.cache_size:
                    self.cache[text] = audio

                return audio

        def synthesize_to_file(self, text: str, output_path: str):
            """Synthesize and save to file."""
            audio = self.tts.synthesize(text, streaming=False)

            # Convert float to int16
            audio_int = (audio * 32767).astype(np.int16)

            # Save to WAV file
            wavfile.write(output_path, self.tts.sample_rate, audio_int)

    # Example usage
    if __name__ == "__main__":
        # Initialize TTS service
        tts_service = TTSService(model_path="tts_model.pth")

        # Example 1: Non-streaming synthesis
        response_text = "It's 55 degrees and rainy in Seattle"
        audio = tts_service.synthesize_response(response_text, streaming=False)
        print(f"Generated audio: {len(audio)} samples, {len(audio) / 22050:.2f} seconds")

        # Save to file
        tts_service.synthesize_to_file(response_text, "response.wav")

        # Example 2: Streaming synthesis
        print("\nStreaming synthesis:")
        for i, audio_chunk in enumerate(tts_service.synthesize_response(response_text, streaming=True)):
            print(f"Chunk {i}: {len(audio_chunk)} samples")
            # In production, send chunk to client immediately

        # Example 3: Voice customization
        tts_service.tts.set_voice(1)  # Female voice
        tts_service.tts.set_speaking_rate(1.2)  # Slightly faster
        audio_custom = tts_service.synthesize_response("Hello, how can I help you?", streaming=False)
    ```

    **Key Optimizations:**
    - **Caching:** Cache common responses (greetings, confirmations) to avoid re-synthesis
    - **Streaming:** Generate audio incrementally, start playback before full synthesis
    - **Parallel synthesis:** Pre-generate common responses in parallel
    - **Model quantization:** Use INT8/FP16 for faster inference
    - **Voice cloning:** Fine-tune on user's voice for personalized responses

=== "⚡ Step 4: Scale & Optimize"

    ## 4.1 Edge vs Cloud Processing

    **Trade-offs:**

    | Component | Edge (On-device) | Cloud |
    |-----------|------------------|-------|
    | **Wake Word** | ✅ Always (privacy, latency) | ❌ Too slow, privacy concerns |
    | **STT** | ⚠️ Limited accuracy, offline mode only | ✅ High accuracy, multi-language |
    | **NLU** | ⚠️ Simple intents only (alarms, timers) | ✅ Complex intents, context |
    | **Action Execution** | ✅ Device control, basic queries | ✅ API calls, knowledge base |
    | **TTS** | ⚠️ Lower quality, limited voices | ✅ High quality, many voices |

    **Hybrid Strategy:**
    - Run wake word + basic commands on-device (offline mode)
    - Send complex queries to cloud for accuracy
    - Cache frequent responses on-device

    **Implementation:**

    ```python
    class HybridVoiceAssistant:
        """
        Hybrid edge-cloud voice assistant.
        Handles simple queries on-device, complex queries in cloud.
        """
        def __init__(self, device_id: str):
            self.device_id = device_id

            # On-device models (lightweight)
            self.local_wake_word = WakeWordDetector("local_model.pth")
            self.local_nlu = None  # Optional: small on-device NLU

            # Cloud connection
            self.cloud_connected = True

            # Offline capability (cached responses)
            self.offline_handlers = {
                "set_timer": self._handle_timer_offline,
                "set_alarm": self._handle_alarm_offline,
                "cancel_alarm": self._handle_cancel_alarm_offline,
                "what_time": self._handle_time_offline
            }

        def process_query(self, audio_chunk):
            """
            Process voice query with edge-cloud hybrid approach.
            """
            # Step 1: Wake word detection (always on-device)
            wake_detected, confidence = self.local_wake_word.process_audio_chunk(audio_chunk)

            if not wake_detected:
                return None

            # Step 2: Capture audio
            full_audio = self._capture_utterance()

            # Step 3: Decide edge vs cloud
            if self.cloud_connected:
                # Cloud processing (high accuracy)
                return self._process_in_cloud(full_audio)
            else:
                # Edge processing (offline mode)
                return self._process_on_device(full_audio)

        def _process_in_cloud(self, audio):
            """Send to cloud for high-accuracy processing."""
            # Send to cloud STT/NLU/TTS pipeline
            response = self._call_cloud_api(audio)

            # Cache for offline use
            self._cache_response(response)

            return response

        def _process_on_device(self, audio):
            """
            Process on-device (offline mode).
            Limited to basic commands.
            """
            # Use lightweight on-device STT (lower accuracy)
            text = self._local_stt(audio)

            # Simple pattern matching for intents
            intent = self._match_intent_offline(text)

            if intent in self.offline_handlers:
                # Execute offline handler
                return self.offline_handlers[intent](text)
            else:
                # Cannot handle offline
                return {
                    "error": "offline",
                    "message": "This command requires internet connection"
                }

        def _match_intent_offline(self, text):
            """Simple pattern matching for offline intents."""
            text_lower = text.lower()

            if "timer" in text_lower:
                return "set_timer"
            elif "alarm" in text_lower:
                if "cancel" in text_lower or "delete" in text_lower:
                    return "cancel_alarm"
                else:
                    return "set_alarm"
            elif "time" in text_lower:
                return "what_time"
            else:
                return "unknown"

        def _handle_timer_offline(self, text):
            """Handle timer command offline."""
            # Extract duration (simple regex)
            import re
            match = re.search(r'(\d+)\s*(minute|second|hour)', text)

            if match:
                duration = int(match.group(1))
                unit = match.group(2)
                # Set timer locally
                return {
                    "action": "set_timer",
                    "duration": duration,
                    "unit": unit,
                    "response": f"Timer set for {duration} {unit}s"
                }
            else:
                return {"error": "Could not understand timer duration"}

        def _handle_alarm_offline(self, text):
            """Handle alarm command offline."""
            # Simplified alarm parsing
            return {
                "action": "set_alarm",
                "response": "Alarm set (offline mode)"
            }

        def _handle_time_offline(self, text):
            """Handle time query offline."""
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return {
                "action": "tell_time",
                "response": f"It's {current_time}"
            }
    ```

    ---

    ## 4.2 Model Optimization (Quantization & Pruning)

    **Techniques:**

    **1. Quantization:** Reduce model precision (FP32 → INT8)
    - 4x smaller model size
    - 2-4x faster inference
    - Minimal accuracy loss (< 1%)

    **2. Pruning:** Remove unnecessary weights
    - 50-90% smaller models
    - Faster inference
    - Requires retraining

    **3. Knowledge Distillation:** Train small model to mimic large model
    - DistilBERT: 40% smaller, 60% faster, 97% of BERT's performance
    - MobileBERT: Optimized for mobile devices

    **Implementation:**

    ```python
    import torch
    from torch.quantization import quantize_dynamic, quantize_static

    class ModelOptimizer:
        """
        Optimize models for production deployment.
        """
        @staticmethod
        def quantize_dynamic(model):
            """
            Dynamic quantization (weights only).
            Best for LSTM/Transformer models.
            """
            quantized_model = quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            return quantized_model

        @staticmethod
        def measure_latency(model, input_tensor, num_runs=100):
            """Measure inference latency."""
            import time

            # Warmup
            for _ in range(10):
                _ = model(input_tensor)

            # Measure
            torch.cuda.synchronize()
            start = time.time()

            for _ in range(num_runs):
                _ = model(input_tensor)

            torch.cuda.synchronize()
            end = time.time()

            avg_latency = (end - start) / num_runs * 1000  # ms
            return avg_latency

        @staticmethod
        def compare_models(original_model, optimized_model, test_input):
            """Compare original vs optimized model."""
            print("Original model:")
            original_latency = ModelOptimizer.measure_latency(original_model, test_input)
            print(f"  Latency: {original_latency:.2f} ms")
            print(f"  Size: {ModelOptimizer.get_model_size(original_model):.2f} MB")

            print("\nOptimized model:")
            optimized_latency = ModelOptimizer.measure_latency(optimized_model, test_input)
            print(f"  Latency: {optimized_latency:.2f} ms")
            print(f"  Size: {ModelOptimizer.get_model_size(optimized_model):.2f} MB")

            print(f"\nSpeedup: {original_latency / optimized_latency:.2f}x")
            print(f"Size reduction: {ModelOptimizer.get_model_size(original_model) / ModelOptimizer.get_model_size(optimized_model):.2f}x")

        @staticmethod
        def get_model_size(model):
            """Get model size in MB."""
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            size_mb = (param_size + buffer_size) / (1024 ** 2)
            return size_mb

    # Example: Quantize NLU model
    if __name__ == "__main__":
        # Load original model
        nlu_model = JointIntentSlotModel(num_intents=100, num_slots=50)
        nlu_model.load_state_dict(torch.load("nlu_model.pth"))
        nlu_model.eval()

        # Quantize
        quantized_model = ModelOptimizer.quantize_dynamic(nlu_model)

        # Compare
        test_input = torch.randint(0, 1000, (1, 50))  # (batch, seq_len)
        test_mask = torch.ones(1, 50)

        ModelOptimizer.compare_models(nlu_model, quantized_model, (test_input, test_mask))

        # Save quantized model
        torch.save(quantized_model.state_dict(), "nlu_model_quantized.pth")
    ```

    ---

    ## 4.3 Caching Strategies

    **Multi-level caching:**

    **1. Intent Cache:** Cache common query → intent + slots mappings
    - "What's the weather?" → always `get_weather` intent
    - 30% cache hit rate reduces NLU compute by 30%

    **2. Response Cache:** Cache complete responses for common queries
    - "What time is it?" → pre-generated TTS audio
    - Eliminates TTS synthesis for frequent queries

    **3. User Profile Cache:** Cache user preferences in memory
    - Avoid database lookups for every query

    **Implementation:**

    ```python
    import redis
    import hashlib
    import json

    class VoiceAssistantCache:
        """
        Multi-level caching for voice assistant.
        """
        def __init__(self, redis_host="localhost", redis_port=6379):
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

            # Cache TTLs
            self.intent_cache_ttl = 3600  # 1 hour
            self.response_cache_ttl = 1800  # 30 minutes
            self.user_profile_ttl = 600  # 10 minutes

        def get_cached_intent(self, text: str):
            """Get cached intent + slots for query text."""
            cache_key = f"intent:{self._hash_text(text)}"
            cached = self.redis.get(cache_key)

            if cached:
                return json.loads(cached)
            return None

        def cache_intent(self, text: str, nlu_result: dict):
            """Cache intent + slots."""
            cache_key = f"intent:{self._hash_text(text)}"
            self.redis.setex(
                cache_key,
                self.intent_cache_ttl,
                json.dumps(nlu_result)
            )

        def get_cached_response(self, text: str):
            """Get cached TTS response audio."""
            cache_key = f"response:{self._hash_text(text)}"
            cached = self.redis.get(cache_key)

            if cached:
                # In production, this would be audio binary data
                return cached
            return None

        def cache_response(self, text: str, audio_data):
            """Cache TTS response."""
            cache_key = f"response:{self._hash_text(text)}"
            self.redis.setex(
                cache_key,
                self.response_cache_ttl,
                audio_data
            )

        def get_user_profile(self, user_id: str):
            """Get cached user profile."""
            cache_key = f"user:{user_id}"
            cached = self.redis.get(cache_key)

            if cached:
                return json.loads(cached)
            return None

        def cache_user_profile(self, user_id: str, profile: dict):
            """Cache user profile."""
            cache_key = f"user:{user_id}"
            self.redis.setex(
                cache_key,
                self.user_profile_ttl,
                json.dumps(profile)
            )

        def _hash_text(self, text: str) -> str:
            """Generate hash for cache key."""
            normalized = text.lower().strip()
            return hashlib.md5(normalized.encode()).hexdigest()
    ```

    ---

    ## 4.4 Multi-language Support

    **Challenges:**
    - Train separate models per language (50+ languages)
    - Handle code-switching (bilingual speakers)
    - Language-specific TTS voices

    **Architecture:**
    - Detect language in first 500ms of audio
    - Route to language-specific STT/NLU/TTS models
    - Maintain language context in dialogue state

    **Optimization:**
    - Multilingual models (mBERT, XLM-R) for resource-constrained languages
    - Shared acoustic models with language-specific adapters

    ---

    ## 4.5 Privacy & Security

    **Privacy Measures:**

    **1. On-device wake word:** No audio sent to cloud until wake word detected

    **2. Opt-in recording:** Audio only saved if user opts in for quality improvement

    **3. Encryption:** TLS 1.3 for all audio transmission

    **4. Differential privacy:** Add noise to usage analytics

    **5. Data retention:** Delete audio after 30 days (configurable)

    **Security:**

    **1. Device authentication:** Each device has unique token

    **2. Rate limiting:** Prevent abuse (max 100 queries/user/hour)

    **3. Content filtering:** Block malicious commands (phishing, malware)

    ```python
    class PrivacyManager:
        """
        Manage privacy settings and data retention.
        """
        def __init__(self, user_id: str):
            self.user_id = user_id
            self.settings = self._load_privacy_settings()

        def should_save_audio(self) -> bool:
            """Check if audio should be saved."""
            return self.settings.get("save_audio", False)

        def should_personalize(self) -> bool:
            """Check if personalization enabled."""
            return self.settings.get("personalization", True)

        def anonymize_query(self, query_data: dict):
            """Anonymize query data for analytics."""
            # Remove PII
            anonymized = query_data.copy()
            anonymized.pop("user_id", None)
            anonymized.pop("device_id", None)

            # Add differential privacy noise
            # (simplified - in production, use proper DP library)
            return anonymized
    ```

    ---

    ## 4.6 Monitoring & Analytics

    **Key Metrics:**

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | Wake word false positive rate | < 0.1% | > 0.2% |
    | STT Word Error Rate (WER) | < 5% | > 7% |
    | Intent classification accuracy | > 90% | < 85% |
    | End-to-end latency (p95) | < 2s | > 3s |
    | TTS audio quality (MOS score) | > 4.0 | < 3.5 |
    | Dialogue success rate | > 85% | < 75% |

    **Logging:**
    - Log all queries with intent, slots, latency
    - Sample audio for quality checks (opt-in only)
    - A/B test new models continuously

    ---

    ## 4.7 Scalability Patterns

    **Horizontal Scaling:**
    - STT: Add GPU instances, load balance by language
    - NLU: Stateless service, easy to scale
    - TTS: Cache common responses, batch synthesis

    **Auto-scaling triggers:**
    - Scale STT when queue depth > 100 requests
    - Scale NLU when CPU > 70%
    - Scale TTS when latency p95 > 500ms

    **Database sharding:**
    - Shard user profiles by user_id hash
    - Shard query history by timestamp (time-series)

---

## Additional Considerations

### 1. Wake Word Customization
- Allow users to customize wake word ("Hey [Name]")
- Train personalized wake word models
- Balance customization with false positive rate

### 2. Emotional Intelligence
- Detect user emotion from voice (sentiment analysis)
- Adjust response tone (empathetic, cheerful, apologetic)
- Handle frustration gracefully (escalate to fallback)

### 3. Accessibility
- Support for users with speech impairments
- Visual feedback for hearing-impaired users
- Alternative input methods (text, touch)

### 4. Testing & Quality Assurance
- Regression testing with diverse accents, ages, genders
- Adversarial testing (background noise, cross-talk)
- Continuous monitoring of accuracy metrics

### 5. Compliance
- GDPR compliance (data deletion, export)
- COPPA for children's privacy
- Accessibility standards (WCAG)

---

## References & Further Reading

**Papers:**
- Whisper (OpenAI): "Robust Speech Recognition via Large-Scale Weak Supervision"
- BERT: "BERT: Pre-training of Deep Bidirectional Transformers"
- WaveNet: "WaveNet: A Generative Model for Raw Audio"
- Tacotron 2: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"

**System Designs:**
- Apple Siri architecture: On-device + cloud hybrid
- Google Assistant: Duplex for natural conversations
- Amazon Alexa: Skills platform and third-party integrations

**Libraries:**
- OpenAI Whisper: State-of-art ASR
- Hugging Face Transformers: BERT, DistilBERT for NLU
- Coqui TTS: Open-source neural TTS
- PyTorch: Deep learning framework

---

## Summary

This AI Voice Assistant design covers:

1. **Requirements:** 100M DAU, 1B queries/day, < 2s end-to-end latency, > 95% STT accuracy, > 90% intent accuracy
2. **Architecture:** Edge-cloud hybrid with wake word on-device, STT/NLU/TTS in cloud, WebSocket streaming
3. **Deep Dive:** Wake word detection (small CNN), streaming ASR (Whisper), joint intent-slot NLU (BERT), dialogue management (state tracking), neural TTS (Tacotron + WaveNet)
4. **Optimization:** Model quantization (4x faster), multi-level caching (30% cache hit), edge processing (offline mode), privacy-preserving design

**Key Takeaways:**
- Wake word must be on-device for privacy and latency
- Streaming ASR with partial results improves perceived latency
- Joint intent-slot models outperform separate classifiers
- Dialogue state tracking enables multi-turn conversations
- Edge-cloud hybrid balances accuracy and privacy
- Caching and quantization are critical for scale

**Trade-offs:**
- On-device vs Cloud: Privacy/latency vs accuracy/features
- Model size vs accuracy: Smaller models faster but less accurate
- Streaming vs batch: Lower latency vs higher throughput
- Personalization vs privacy: Better UX vs data collection

This design supports Apple Siri, Google Assistant, Amazon Alexa-scale voice assistants with high accuracy, low latency, and strong privacy protections.
