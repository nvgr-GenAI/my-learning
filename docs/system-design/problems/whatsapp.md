# Design WhatsApp (Messaging Platform)

A real-time messaging platform supporting text, media, voice/video calls, and group chats with end-to-end encryption for billions of users.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 2B users, 100B messages/day, 1B groups |
| **Key Challenges** | Real-time delivery, end-to-end encryption, message persistence, multimedia transfer |
| **Core Concepts** | WebSocket, message queue, last-seen tracking, delivery receipts, media compression |
| **Companies** | Meta (WhatsApp/Messenger), Telegram, Signal, WeChat, Line |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Send/Receive Messages** | One-to-one text messaging | P0 (Must have) |
    | **Group Chat** | Send messages to multiple users | P0 (Must have) |
    | **Media Sharing** | Photos, videos, documents | P0 (Must have) |
    | **Delivery Receipts** | Sent, delivered, read status | P0 (Must have) |
    | **Last Seen** | Show when user was last online | P1 (Should have) |
    | **Voice/Video Calls** | Real-time audio/video | P1 (Should have) |
    | **Stories** | 24-hour ephemeral content | P2 (Nice to have) |
    | **End-to-End Encryption** | Secure message transmission | P0 (Must have) |

    **Explicitly Out of Scope** (mention in interview):

    - Payment integration (WhatsApp Pay)
    - Business API features
    - Multi-device sync (focus on mobile)
    - Message search (complex indexing)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% uptime | Messaging is critical communication |
    | **Latency** | < 100ms message delivery | Real-time messaging expectation |
    | **Consistency** | Eventual consistency acceptable | Brief delays OK, messages must arrive |
    | **Security** | End-to-end encrypted | Privacy is core to WhatsApp |
    | **Durability** | Messages stored until delivered | No message loss |
    | **Scalability** | Billions of concurrent users | Global scale |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Total Users: 2B monthly active
    Daily Active Users (DAU): 1B (50% engagement)

    Messages:
    - Messages per DAU: 100 messages/day (average)
    - Daily messages: 1B × 100 = 100B messages/day
    - Message QPS: 100B / 86,400 = 1.16M messages/sec
    - Peak QPS: 3x = 3.47M messages/sec

    Group messages:
    - 20% of messages are in groups
    - Average group size: 10 members
    - Effective messages: 20B × 10 = 200B fanout/day
    - Total fanout: 100B + 200B = 300B operations/day

    Media sharing:
    - 30% of messages include media
    - Daily media: 30B files/day
    - Media QPS: 347K files/sec
    ```

    ### Storage Estimates

    ```
    Text messages:
    - Average message size: 100 bytes (text only)
    - Daily: 100B × 100 bytes = 10 TB/day
    - With 30-day retention: 300 TB

    Media files:
    - Photos: 20B/day × 500 KB = 10 PB/day
    - Videos: 8B/day × 5 MB = 40 PB/day
    - Documents: 2B/day × 1 MB = 2 PB/day
    - Total media: 52 PB/day
    - With 30-day retention: 1,560 PB (1.56 exabytes)

    Metadata:
    - User data: 2B × 10 KB = 20 TB
    - Group data: 1B groups × 5 KB = 5 TB

    Total: 300 TB (text) + 1.56 EB (media) + 25 TB (metadata) ≈ 1.56 exabytes
    ```

    ### Bandwidth Estimates

    ```
    Upload bandwidth:
    - Text: 10 TB/day = 115 MB/sec ≈ 1 Gbps
    - Media: 52 PB/day = 602 GB/sec ≈ 4.8 Tbps

    Download bandwidth:
    - Same as upload (each message downloaded once)
    - Total: ~4.8 Tbps

    Compressed: With media compression (50%), actual ~2.4 Tbps
    ```

    ### Memory Estimates (Caching)

    ```
    Online users:
    - Concurrent online: 500M users
    - Session data: 500M × 5 KB = 2.5 TB

    Recent messages (hot cache):
    - Last 1 hour messages: 100B / 24 = 4.2B messages
    - 4.2B × 100 bytes = 420 GB

    Delivery queue (pending messages):
    - Offline users: 1B users
    - Average pending: 50 messages
    - 1B × 50 × 100 bytes = 5 TB

    Total cache: 2.5 TB + 420 GB + 5 TB ≈ 8 TB
    ```

    ---

    ## Key Assumptions

    1. Average message size: 100 bytes (text)
    2. 30% of messages include media
    3. 20% of messages are in groups (avg 10 members)
    4. 50% of users online at any time
    5. Message retention: 30 days on server, forever on client
    6. End-to-end encryption mandatory

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **WebSocket for real-time** - Persistent connections for instant delivery
    2. **Message queue for reliability** - Persist until delivered
    3. **End-to-end encryption** - Client-side encryption, server cannot read
    4. **Media in object storage** - Separate text from media
    5. **Multi-datacenter** - Global distribution for low latency

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Client_A[User A<br/>Mobile App]
            Client_B[User B<br/>Mobile App]
        end

        subgraph "Edge Layer"
            LB[Load Balancer]
            CDN[CDN<br/>Media delivery]
        end

        subgraph "Application Layer"
            WS[WebSocket Servers<br/>Real-time messaging]
            API[API Servers<br/>REST endpoints]
            Media_Service[Media Service<br/>Upload/download]
        end

        subgraph "Message Processing"
            Message_Queue[Message Queue<br/>Kafka/RabbitMQ]
            Delivery_Worker[Delivery Worker<br/>Process & route]
        end

        subgraph "Caching"
            Redis_Session[Redis<br/>User sessions]
            Redis_Message[Redis<br/>Message buffer]
        end

        subgraph "Storage"
            Message_DB[(Message DB<br/>Cassandra<br/>Time-series)]
            User_DB[(User DB<br/>PostgreSQL)]
            S3[Object Storage<br/>S3<br/>Media files]
        end

        subgraph "Notification"
            Push_Service[Push Notification<br/>APNs/FCM]
        end

        Client_A --> LB
        Client_B --> LB
        Client_A --> CDN
        Client_B --> CDN

        LB --> WS
        LB --> API

        WS --> Message_Queue
        WS --> Redis_Session
        API --> Media_Service

        Message_Queue --> Delivery_Worker
        Delivery_Worker --> WS
        Delivery_Worker --> Message_DB
        Delivery_Worker --> Push_Service

        Media_Service --> S3
        CDN --> S3

        WS --> Redis_Message
        Delivery_Worker --> Redis_Message

        API --> User_DB

        Push_Service --> Client_A
        Push_Service --> Client_B

        style LB fill:#e1f5ff
        style CDN fill:#e8f5e9
        style Redis_Session fill:#fff4e1
        style Redis_Message fill:#fff4e1
        style Message_DB fill:#ffe1e1
        style User_DB fill:#ffe1e1
        style S3 fill:#f3e5f5
        style Message_Queue fill:#e8eaf6
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **WebSocket** | Real-time bidirectional messaging, persistent connection | HTTP polling (wasteful, high latency), XMPP (more complex) |
    | **Kafka** | High-throughput message queue (3.47M msg/sec), replay capability | RabbitMQ (lower throughput), SQS (higher latency) |
    | **Cassandra** | Write-heavy workload (300B ops/day), time-series data | PostgreSQL (can't handle write volume), MongoDB (weaker consistency) |
    | **Redis** | Fast message buffer for online users (< 1ms), session management | No cache (higher database load, slower delivery) |
    | **S3** | Unlimited media storage, durability (11 nines) | Database BLOBs (expensive, doesn't scale) |

    **Key Trade-off:** We chose **eventual consistency for message delivery** (brief delays acceptable) but **strong ordering within conversations** (messages appear in order sent).

    ---

    ## API Design

    ### 1. Send Message

    **Request:**
    ```http
    POST /api/v1/messages
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "recipient_id": "user_456",
      "content": "SGVsbG8gV29ybGQh",  // Base64 encoded encrypted message
      "message_type": "text",  // text, image, video, audio, document
      "media_url": null,
      "timestamp": 1643712000000
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "message_id": "msg_abc123",
      "status": "sent",
      "timestamp": 1643712000000
    }
    ```

    **Design Notes:**

    - Content is encrypted client-side (server doesn't decrypt)
    - Message persisted in queue immediately
    - Status updates via WebSocket

    ---

    ### 2. WebSocket Message Flow

    **Client sends (via WebSocket):**
    ```json
    {
      "type": "send_message",
      "message_id": "msg_abc123",
      "recipient_id": "user_456",
      "content": "SGVsbG8gV29ybGQh",
      "timestamp": 1643712000000
    }
    ```

    **Server acknowledges:**
    ```json
    {
      "type": "ack",
      "message_id": "msg_abc123",
      "status": "sent"
    }
    ```

    **Recipient receives:**
    ```json
    {
      "type": "new_message",
      "message_id": "msg_abc123",
      "sender_id": "user_123",
      "content": "SGVsbG8gV29ybGQh",
      "timestamp": 1643712000000
    }
    ```

    **Recipient sends delivery receipt:**
    ```json
    {
      "type": "delivery_receipt",
      "message_id": "msg_abc123",
      "status": "delivered"  // or "read"
    }
    ```

    ---

    ## Database Schema

    ### Messages (Cassandra)

    ```sql
    -- Messages table (time-series)
    CREATE TABLE messages (
        user_id UUID,
        conversation_id UUID,
        timestamp TIMESTAMP,
        message_id UUID,
        sender_id UUID,
        content BLOB,  -- Encrypted
        message_type TEXT,
        media_url TEXT,
        status TEXT,  -- sent, delivered, read
        PRIMARY KEY ((user_id, conversation_id), timestamp, message_id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC);

    -- Conversations (for chat list)
    CREATE TABLE conversations (
        user_id UUID,
        conversation_id UUID,
        last_message_id UUID,
        last_message_timestamp TIMESTAMP,
        last_message_preview TEXT,
        unread_count INT,
        PRIMARY KEY (user_id, conversation_id)
    );
    ```

    **Why Cassandra:**
    - **Write-heavy:** 300B operations/day
    - **Time-series:** Messages ordered by timestamp
    - **Horizontal scaling:** Add nodes easily

    ---

    ## Data Flow Diagrams

    ### Message Send Flow

    ```mermaid
    sequenceDiagram
        participant Sender
        participant WS_Server
        participant Kafka
        participant Delivery_Worker
        participant Message_DB
        participant Recipient_WS
        participant Recipient

        Sender->>WS_Server: Send message (WebSocket)
        WS_Server->>WS_Server: Validate, assign message_id
        WS_Server->>Kafka: Publish message event
        WS_Server-->>Sender: ACK (sent)

        Kafka->>Delivery_Worker: Process message
        Delivery_Worker->>Message_DB: Store message
        Message_DB-->>Delivery_Worker: Success

        alt Recipient online
            Delivery_Worker->>Recipient_WS: Forward message
            Recipient_WS->>Recipient: Deliver (WebSocket)
            Recipient-->>Recipient_WS: Delivery receipt
            Recipient_WS->>Delivery_Worker: Receipt
            Delivery_Worker->>Message_DB: Update status: delivered
            Delivery_Worker->>WS_Server: Notify sender
            WS_Server->>Sender: Status update (delivered)
        else Recipient offline
            Delivery_Worker->>Message_DB: Store in pending queue
            Delivery_Worker->>Push_Service: Send push notification
            Push_Service->>Recipient: Push notification
        end
    ```

    **Flow Explanation:**

    1. **Sender sends** - Via WebSocket, get immediate ACK
    2. **Persist in Kafka** - Ensures message not lost
    3. **Worker processes** - Store in database
    4. **Deliver if online** - Forward via WebSocket
    5. **Push if offline** - APNs/FCM notification
    6. **Delivery receipt** - Update status, notify sender

=== "🔍 Step 3: Deep Dive"

    ## Overview

    This section dives deep into four critical WhatsApp subsystems.

    | Topic | Key Question | Solution |
    |-------|-------------|----------|
    | **Message Delivery** | How to ensure reliable delivery? | Message queue + retry + idempotency |
    | **End-to-End Encryption** | How to implement E2EE at scale? | Signal Protocol + key exchange |
    | **Group Messaging** | How to send to 256 members efficiently? | Fan-out optimization + encryption keys |
    | **Media Handling** | How to transfer 52 PB/day of media? | Client-side compression + S3 + CDN |

    ---

    === "📨 Message Delivery System"

        ## The Challenge

        **Problem:** Deliver 1.16M messages/sec reliably, even if recipient offline.

        **Requirements:**

        - **Exactly-once delivery:** No duplicates, no loss
        - **Ordered delivery:** Messages arrive in send order
        - **Fast delivery:** < 100ms when online
        - **Reliable:** Persist until delivered

        ---

        ## Message Queue Architecture

        **Using Kafka for reliable messaging:**

        ```python
        class MessageDeliveryService:
            """Reliable message delivery with Kafka"""

            def __init__(self):
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=['kafka1:9092', 'kafka2:9092'],
                    acks='all',  # Wait for all replicas
                    retries=3,
                    max_in_flight_requests_per_connection=1  # Ensure ordering
                )
                self.redis = redis.Redis()

            def send_message(self, sender_id: str, recipient_id: str, content: bytes):
                """
                Send message via Kafka

                Args:
                    sender_id: Message sender
                    recipient_id: Message recipient
                    content: Encrypted message content

                Returns:
                    message_id
                """
                # Generate unique message ID
                message_id = str(uuid.uuid4())
                conversation_id = self._get_conversation_id(sender_id, recipient_id)

                # Create message object
                message = {
                    'message_id': message_id,
                    'conversation_id': conversation_id,
                    'sender_id': sender_id,
                    'recipient_id': recipient_id,
                    'content': base64.b64encode(content).decode(),
                    'timestamp': int(time.time() * 1000),
                    'status': 'sent'
                }

                # Publish to Kafka (partitioned by recipient for ordering)
                self.kafka_producer.send(
                    'messages',
                    key=recipient_id.encode(),  # Ensures ordering per recipient
                    value=json.dumps(message).encode()
                )

                # Send ACK to sender immediately
                return message_id

            def process_message(self, message: dict):
                """
                Worker processes message from Kafka

                Delivers to recipient if online, stores if offline
                """
                message_id = message['message_id']
                recipient_id = message['recipient_id']

                # Check idempotency (prevent duplicate delivery)
                if self.redis.exists(f"delivered:{message_id}"):
                    logger.info(f"Message {message_id} already delivered")
                    return

                # Store in database (Cassandra)
                self._store_message(message)

                # Check if recipient online
                if self._is_user_online(recipient_id):
                    # Deliver via WebSocket
                    success = self._deliver_via_websocket(recipient_id, message)

                    if success:
                        # Mark as delivered
                        self.redis.setex(f"delivered:{message_id}", 86400, 1)
                        self._update_status(message_id, 'delivered')
                else:
                    # Store in pending queue
                    self._store_pending(recipient_id, message_id)

                    # Send push notification
                    self._send_push_notification(recipient_id, message)

            def _deliver_via_websocket(self, user_id: str, message: dict) -> bool:
                """Send message to online user via WebSocket"""
                ws_connection = websocket_manager.get_connection(user_id)

                if ws_connection:
                    ws_connection.send(json.dumps({
                        'type': 'new_message',
                        'message': message
                    }))
                    return True

                return False

            def _store_pending(self, user_id: str, message_id: str):
                """Store message in pending queue for offline user"""
                self.redis.lpush(f"pending:{user_id}", message_id)
                logger.info(f"Message {message_id} stored in pending queue for {user_id}")

            def deliver_pending_messages(self, user_id: str):
                """
                Deliver all pending messages when user comes online

                Called when user connects
                """
                # Get all pending message IDs
                pending_ids = self.redis.lrange(f"pending:{user_id}", 0, -1)

                for message_id in pending_ids:
                    # Fetch message from database
                    message = self._fetch_message(message_id)

                    # Deliver via WebSocket
                    success = self._deliver_via_websocket(user_id, message)

                    if success:
                        # Remove from pending queue
                        self.redis.lrem(f"pending:{user_id}", 1, message_id)
                        self._update_status(message_id, 'delivered')

                logger.info(f"Delivered {len(pending_ids)} pending messages to {user_id}")
        ```

        ---

        ## Idempotency & Deduplication

        **Problem:** Network issues cause retries → duplicate messages.

        **Solution:** Idempotency keys

        ```python
        def handle_message_send(sender_id: str, content: bytes, idempotency_key: str):
            """
            Handle message send with idempotency

            Args:
                idempotency_key: Client-generated unique ID for this send
            """
            # Check if we've already processed this request
            existing_message_id = redis.get(f"idempotency:{idempotency_key}")

            if existing_message_id:
                # Already processed, return existing message_id
                return existing_message_id.decode()

            # Process message
            message_id = send_message(sender_id, content)

            # Store idempotency mapping (24h TTL)
            redis.setex(f"idempotency:{idempotency_key}", 86400, message_id)

            return message_id
        ```

    === "🔐 End-to-End Encryption"

        ## The Challenge

        **Problem:** Encrypt messages so only sender and recipient can read (not even server).

        **Solution:** Signal Protocol (also used by WhatsApp, Signal)

        ---

        ## Signal Protocol Overview

        **Components:**

        1. **Identity Keys:** Long-term public/private key pair per user
        2. **Pre-keys:** One-time use public keys
        3. **Session Keys:** Ephemeral keys per conversation
        4. **Ratcheting:** Forward secrecy (past messages safe if key compromised)

        **Key Exchange (Simplified):**

        ```
        Alice wants to message Bob:

        1. Alice fetches Bob's public identity key & pre-key from server
        2. Alice generates session key using:
           - Her private identity key
           - Bob's public identity key
           - Bob's pre-key
        3. Alice encrypts message with session key
        4. Bob decrypts using his private keys
        5. Both sides ratchet keys for forward secrecy
        ```

        ---

        ## Implementation

        ```python
        from cryptography.hazmat.primitives.asymmetric import x25519
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        class E2EEManager:
            """End-to-end encryption using Signal Protocol (simplified)"""

            def __init__(self):
                self.redis = redis.Redis()

            def generate_identity_keys(self, user_id: str):
                """
                Generate long-term identity key pair for user

                Called once per user on account creation
                """
                # Generate X25519 key pair
                private_key = x25519.X25519PrivateKey.generate()
                public_key = private_key.public_key()

                # Store private key securely (encrypted with user's password)
                # In reality, private key NEVER leaves client device
                private_bytes = private_key.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption()
                )

                # Store public key on server
                public_bytes = public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )

                db.execute(
                    "INSERT INTO user_keys (user_id, public_identity_key) VALUES (%s, %s)",
                    (user_id, public_bytes)
                )

                return private_key, public_key

            def encrypt_message(
                self,
                sender_private_key: x25519.X25519PrivateKey,
                recipient_public_key: bytes,
                plaintext: str
            ) -> bytes:
                """
                Encrypt message using recipient's public key

                Args:
                    sender_private_key: Sender's private key
                    recipient_public_key: Recipient's public key
                    plaintext: Message to encrypt

                Returns:
                    Encrypted message bytes
                """
                # Load recipient's public key
                recipient_key = x25519.X25519PublicKey.from_public_bytes(recipient_public_key)

                # Perform key exchange (ECDH)
                shared_secret = sender_private_key.exchange(recipient_key)

                # Derive encryption key from shared secret
                encryption_key = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=b'whatsapp-encryption'
                ).derive(shared_secret)

                # Encrypt with AES-GCM
                aesgcm = AESGCM(encryption_key)
                nonce = os.urandom(12)
                ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)

                # Return nonce + ciphertext
                return nonce + ciphertext

            def decrypt_message(
                self,
                recipient_private_key: x25519.X25519PrivateKey,
                sender_public_key: bytes,
                ciphertext: bytes
            ) -> str:
                """
                Decrypt message using sender's public key

                Args:
                    recipient_private_key: Recipient's private key
                    sender_public_key: Sender's public key
                    ciphertext: Encrypted message (nonce + ciphertext)

                Returns:
                    Decrypted plaintext
                """
                # Load sender's public key
                sender_key = x25519.X25519PublicKey.from_public_bytes(sender_public_key)

                # Perform key exchange
                shared_secret = recipient_private_key.exchange(sender_key)

                # Derive decryption key
                decryption_key = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=b'whatsapp-encryption'
                ).derive(shared_secret)

                # Extract nonce and ciphertext
                nonce = ciphertext[:12]
                encrypted_data = ciphertext[12:]

                # Decrypt
                aesgcm = AESGCM(decryption_key)
                plaintext_bytes = aesgcm.decrypt(nonce, encrypted_data, None)

                return plaintext_bytes.decode()
        ```

        **Security properties:**

        - **Confidentiality:** Only sender and recipient can read
        - **Forward secrecy:** Past messages safe if key compromised
        - **Authentication:** Verify sender identity
        - **Server-blind:** Server cannot decrypt messages

    === "👥 Group Messaging"

        ## The Challenge

        **Problem:** Send message to 256-member group efficiently with E2EE.

        **Naive approach:** Encrypt separately for each member. **Slow:** 256 encryptions per message.

        **Solution:** Sender Keys Protocol

        ---

        ## Sender Keys Protocol

        **Concept:** Group member generates a symmetric key, shares encrypted version with all members.

        **How it works:**

        1. **Alice joins group:** Generates sender key SK_Alice
        2. **Alice shares:** Encrypts SK_Alice with each member's public key, sends to all
        3. **Alice sends message:** Encrypts with SK_Alice (one encryption)
        4. **Members decrypt:** Each uses their copy of SK_Alice

        **Implementation:**

        ```python
        class GroupMessaging:
            """Group messaging with Sender Keys Protocol"""

            def send_group_message(
                self,
                sender_id: str,
                group_id: str,
                content: str
            ):
                """
                Send message to group

                Args:
                    sender_id: Message sender
                    group_id: Target group
                    content: Message content
                """
                # Get sender's key for this group
                sender_key = self._get_sender_key(sender_id, group_id)

                # Encrypt message with sender key (AES-256)
                encrypted_content = self._encrypt_with_sender_key(sender_key, content)

                # Get all group members
                members = db.query(
                    "SELECT user_id FROM group_members WHERE group_id = %s",
                    (group_id,)
                )

                # Fan-out to all members (parallel)
                with ThreadPoolExecutor(max_workers=50) as executor:
                    futures = []
                    for member in members:
                        future = executor.submit(
                            self._deliver_to_member,
                            member['user_id'],
                            group_id,
                            encrypted_content
                        )
                        futures.append(future)

                    # Wait for all deliveries
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Failed to deliver to member: {e}")

            def _get_sender_key(self, sender_id: str, group_id: str) -> bytes:
                """Get or generate sender key for group"""
                key = redis.get(f"sender_key:{sender_id}:{group_id}")

                if not key:
                    # Generate new sender key (AES-256)
                    key = os.urandom(32)
                    redis.setex(f"sender_key:{sender_id}:{group_id}", 86400, key)

                return key
        ```

        **Benefits:**

        - **Efficient:** One encryption instead of 256
        - **E2EE maintained:** Each member's copy encrypted with their key
        - **Scalable:** Works for large groups

    === "📷 Media Handling"

        ## The Challenge

        **Problem:** Transfer 52 PB/day of media (photos/videos).

        **Requirements:**

        - **Compression:** Reduce size without quality loss
        - **Fast uploads:** Chunked, resumable
        - **CDN delivery:** Low latency worldwide
        - **Cost-effective:** Storage optimization

        ---

        ## Media Upload Pipeline

        ```python
        class MediaService:
            """Handle media upload and delivery"""

            def upload_media(
                self,
                user_id: str,
                media_type: str,
                file_data: bytes
            ) -> str:
                """
                Upload media file

                Args:
                    user_id: Uploader
                    media_type: image, video, document
                    file_data: File bytes

                Returns:
                    media_url
                """
                # Generate unique media ID
                media_id = str(uuid.uuid4())

                # Compress based on type
                if media_type == 'image':
                    compressed = self._compress_image(file_data)
                elif media_type == 'video':
                    compressed = self._compress_video(file_data)
                else:
                    compressed = file_data  # No compression for documents

                # Upload to S3
                s3_key = f"media/{user_id}/{media_id}"
                s3.put_object(
                    Bucket='whatsapp-media',
                    Key=s3_key,
                    Body=compressed,
                    ContentType=f'{media_type}/*'
                )

                # Generate CDN URL
                media_url = f"https://cdn.whatsapp.com/{s3_key}"

                return media_url

            def _compress_image(self, image_data: bytes) -> bytes:
                """Compress image (JPEG quality 85%)"""
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(image_data))

                # Resize if too large (max 1600x1600)
                max_size = (1600, 1600)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Compress to JPEG
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=85, optimize=True)

                return output.getvalue()

            def _compress_video(self, video_data: bytes) -> bytes:
                """Compress video using H.265"""
                # Use FFmpeg for compression
                # Reduce bitrate, resolution for mobile
                # Implementation details omitted
                pass
        ```

        **Compression savings:**

        - Images: 500 KB → 150 KB (70% reduction)
        - Videos: 5 MB → 2 MB (60% reduction)
        - Total savings: 52 PB → 21 PB/day (60% reduction)

=== "⚡ Step 4: Scale & Optimize"

    ## Bottleneck Identification

    | Component | Bottleneck? | Solution |
    |-----------|------------|----------|
    | **Message throughput** | ✅ Yes | Kafka cluster (100 brokers), partitioning |
    | **WebSocket connections** | ✅ Yes | 1,000 servers (500K connections each) |
    | **Database writes** | ✅ Yes | Cassandra cluster (500 nodes) |
    | **Media storage** | ✅ Yes | S3 with lifecycle (move to Glacier after 30 days) |

    ---

    ## Cost Optimization

    **Monthly cost at 2B users:**

    | Component | Cost |
    |-----------|------|
    | **EC2 (WebSocket/API)** | $1,080,000 (5,000 servers) |
    | **Kafka cluster** | $216,000 (100 brokers) |
    | **Cassandra** | $810,000 (500 nodes) |
    | **Redis cache** | $108,000 (500 nodes) |
    | **S3 storage** | $484,000 (21 PB × $0.023/GB) |
    | **CDN** | $1,785,000 (85 PB egress × $0.021/GB) |
    | **Total** | **$4.5M/month** |

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **WebSocket for real-time** - Persistent connections, < 100ms delivery
    2. **Kafka for reliability** - High throughput (3.47M msg/sec), replay capability
    3. **End-to-end encryption** - Signal Protocol, server-blind
    4. **Sender keys for groups** - Efficient group messaging
    5. **Media compression** - 60% size reduction, quality maintained
    6. **Cassandra for messages** - Write-heavy (300B ops/day), time-series

    ---

    ## Interview Tips

    ✅ **Start with scale** - 100B messages/day is massive write load

    ✅ **Discuss E2EE** - How to implement at scale

    ✅ **Group messaging** - Sender keys protocol

    ✅ **Message delivery** - Reliability, ordering, idempotency

    ✅ **Media handling** - Compression, CDN strategy

    ---

    ## Common Follow-Up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to ensure message ordering?"** | Kafka partitioning by recipient, in-order processing |
    | **"What if user switches devices?"** | Multi-device sync using device keys, session management |
    | **"How to handle message deletion?"** | Client-side deletion, server deletion after 30 days |
    | **"How to scale WebSocket connections?"** | 1,000 servers (500K each), connection routing by user_id hash |

---

**Difficulty:** 🔴 Hard | **Interview Time:** 60-75 minutes | **Companies:** Meta, Telegram, Signal, WeChat, Line

---

*Master this problem and you'll be ready for: Messenger, Telegram, Discord, Slack*
