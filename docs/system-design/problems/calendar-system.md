# Design a Calendar System

Design a comprehensive calendar system similar to Google Calendar or Microsoft Outlook that handles event management, recurring events, conflict detection, availability checking, time zone handling, and calendar sharing.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐ Medium | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 100M users, 500M events/day, 1B availability checks/day |
| **Key Challenges** | Recurring events, conflict detection, time zones, RSVP tracking, availability |
| **Core Concepts** | RRULE (RFC 5545), conflict resolution, free/busy, time zone conversion, notification system |
| **Companies** | Google Calendar, Microsoft Outlook, Apple Calendar, Calendly, Zoom |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Event CRUD** | Create, read, update, delete events | P0 (Must have) |
    | **Recurring Events** | Daily, weekly, monthly, custom patterns | P0 (Must have) |
    | **Conflict Detection** | Detect overlapping events | P0 (Must have) |
    | **Availability Check** | Check free/busy status | P0 (Must have) |
    | **Invitations & RSVP** | Invite attendees, track responses | P0 (Must have) |
    | **Time Zone Support** | Handle multiple time zones | P0 (Must have) |
    | **Reminders** | Email/push notifications before events | P1 (Should have) |
    | **Calendar Sharing** | Share calendars with others | P1 (Should have) |

    **Explicitly Out of Scope:**

    - Video conferencing integration
    - Task management
    - Calendar themes/customization
    - Contact management
    - File attachments (simplified)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.9% | Users rely on calendars daily |
    | **Consistency** | Strong | No double-booking, accurate RSVP |
    | **Latency** | < 200ms for queries | Fast event lookup and availability check |
    | **Scalability** | 100M users | Support enterprise scale |
    | **Data Integrity** | 100% | No lost events or incorrect recurring patterns |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 100M users
    Events created per user/day: 2 events = 200M events/day
    Events viewed per user/day: 20 views = 2B views/day
    Availability checks: 10 per user/day = 1B checks/day

    Event creation rate:
    - 200M events/day / 86,400 sec = ~2,315 creates/sec
    - Peak (3x): 6,945 creates/sec

    Event view rate:
    - 2B views/day / 86,400 sec = ~23,150 views/sec
    - Peak (3x): 69,450 views/sec

    Availability checks:
    - 1B checks/day / 86,400 sec = ~11,575 checks/sec
    - Peak (3x): 34,725 checks/sec
    ```

    ### Storage Estimates

    ```
    Event record size:
    - Event metadata: 1 KB (title, description, location, times)
    - Recurrence rule: 200 bytes
    - Attendees (avg 3): 300 bytes
    - Total per event: ~1.5 KB

    Daily storage:
    - 200M events × 1.5 KB = 300 GB/day

    Annual storage:
    - 300 GB × 365 = 109.5 TB/year

    With recurring event optimization (store rule, not instances):
    - Actual storage: ~40 TB/year (60% reduction)

    Metadata caching (Redis):
    - Active events (30 days): 6 TB
    - Hot cache (today + 7 days): 2.4 TB
    ```

    ### Database Queries

    ```
    Read queries (views + availability):
    - 23,150 + 11,575 = ~34,725 reads/sec
    - Peak: 104,175 reads/sec

    Write queries (creates + updates):
    - Creates: 2,315/sec
    - Updates (5% of events): 116/sec
    - RSVPs: 1,000/sec
    - Total: ~3,431 writes/sec
    - Peak: 10,293 writes/sec

    Read:Write ratio = 10:1
    ```

    ---

    ## Key Assumptions

    1. Events stored for 5 years
    2. Average event duration: 1 hour
    3. Recurring events: 30% of all events
    4. Average attendees per event: 3
    5. Time zones: Support all IANA time zones
    6. Reminders: 15 min, 1 hour, 1 day before event

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Event normalization** - Store recurring events as rules, not instances
    2. **Time zone agnostic storage** - Store in UTC, display in user's time zone
    3. **Conflict detection** - Check overlaps on event creation/update
    4. **Distributed caching** - Cache active events and availability
    5. **Event-driven notifications** - Queue-based reminder system

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Web[Web Client]
            Mobile[Mobile App]
            Sync[Sync Client]
        end

        subgraph "API Gateway"
            Gateway[API Gateway<br/>Rate Limiting]
        end

        subgraph "Application Services"
            EventAPI[Event Service]
            AvailAPI[Availability Service]
            RecurAPI[Recurrence Service]
            ConflictAPI[Conflict Detector]
            NotifAPI[Notification Service]
            ShareAPI[Sharing Service]
        end

        subgraph "Processing"
            EventProc[Event Processor]
            ReminderWorker[Reminder Worker]
            RecurExpander[Recurrence Expander]
            ConflictChecker[Conflict Checker]
        end

        subgraph "Storage Layer"
            EventDB[(PostgreSQL<br/>Events & Recurrence)]
            AttendeesDB[(PostgreSQL<br/>Attendees & RSVP)]
            Cache[(Redis<br/>Event Cache)]
            Queue[Message Queue<br/>RabbitMQ/SQS]
        end

        subgraph "External"
            Email[Email Service<br/>SendGrid]
            Push[Push Notification<br/>FCM/APNS]
        end

        Web --> Gateway
        Mobile --> Gateway
        Sync --> Gateway

        Gateway --> EventAPI
        Gateway --> AvailAPI
        Gateway --> ShareAPI

        EventAPI --> RecurAPI
        EventAPI --> ConflictAPI
        EventAPI --> EventProc
        AvailAPI --> Cache
        AvailAPI --> EventDB

        EventProc --> Queue
        Queue --> ReminderWorker
        Queue --> RecurExpander

        RecurAPI --> RecurExpander
        ConflictAPI --> ConflictChecker

        EventAPI --> EventDB
        EventAPI --> AttendeesDB
        EventAPI --> Cache

        ReminderWorker --> NotifAPI
        NotifAPI --> Email
        NotifAPI --> Push

        RecurExpander --> EventDB
        ConflictChecker --> EventDB
        ConflictChecker --> Cache

        style EventDB fill:#e1f5ff
        style AttendeesDB fill:#e1f5ff
        style Cache fill:#ffe1e1
        style Queue fill:#fff4e1
    ```

    ---

    ## API Design

    ### 1. Create Event

    **Request:**
    ```http
    POST /api/v1/events
    Content-Type: application/json

    {
      "title": "Team Standup",
      "description": "Daily team sync",
      "location": "Conference Room A",
      "start_time": "2024-01-15T10:00:00Z",
      "end_time": "2024-01-15T10:30:00Z",
      "time_zone": "America/New_York",
      "attendees": [
        {"email": "alice@example.com", "optional": false},
        {"email": "bob@example.com", "optional": true}
      ],
      "recurrence": {
        "frequency": "DAILY",
        "interval": 1,
        "until": "2024-12-31T23:59:59Z",
        "by_day": ["MO", "TU", "WE", "TH", "FR"]
      },
      "reminders": [
        {"minutes_before": 15, "method": "email"},
        {"minutes_before": 5, "method": "push"}
      ]
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "event_id": "evt_abc123",
      "title": "Team Standup",
      "start_time": "2024-01-15T10:00:00Z",
      "end_time": "2024-01-15T10:30:00Z",
      "time_zone": "America/New_York",
      "is_recurring": true,
      "recurrence_id": "rec_xyz789",
      "conflicts": [],
      "attendees": [
        {
          "email": "alice@example.com",
          "status": "pending",
          "optional": false
        }
      ],
      "calendar_url": "https://calendar.example.com/events/evt_abc123"
    }
    ```

    ---

    ### 2. Check Availability (Free/Busy)

    **Request:**
    ```http
    POST /api/v1/availability/check
    Content-Type: application/json

    {
      "users": ["alice@example.com", "bob@example.com"],
      "start_time": "2024-01-15T09:00:00Z",
      "end_time": "2024-01-15T17:00:00Z",
      "time_zone": "America/New_York"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "time_zone": "America/New_York",
      "availability": {
        "alice@example.com": {
          "free_slots": [
            {"start": "2024-01-15T09:00:00Z", "end": "2024-01-15T10:00:00Z"},
            {"start": "2024-01-15T11:00:00Z", "end": "2024-01-15T12:00:00Z"}
          ],
          "busy_slots": [
            {"start": "2024-01-15T10:00:00Z", "end": "2024-01-15T11:00:00Z"},
            {"start": "2024-01-15T14:00:00Z", "end": "2024-01-15T15:00:00Z"}
          ]
        },
        "bob@example.com": {
          "free_slots": [...],
          "busy_slots": [...]
        }
      },
      "common_free_slots": [
        {"start": "2024-01-15T11:00:00Z", "end": "2024-01-15T12:00:00Z"}
      ]
    }
    ```

    ---

    ### 3. Update Event (Single Instance)

    **Request:**
    ```http
    PATCH /api/v1/events/evt_abc123
    Content-Type: application/json

    {
      "instance_date": "2024-01-16T10:00:00Z",  // For recurring events
      "update_scope": "this_instance",  // or "all_future" or "all"
      "start_time": "2024-01-16T11:00:00Z",
      "end_time": "2024-01-16T11:30:00Z",
      "title": "Team Standup (Modified)"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "event_id": "evt_abc123",
      "exception_id": "exc_def456",  // Created for single instance
      "message": "Event instance updated",
      "conflicts": []
    }
    ```

    ---

    ### 4. RSVP to Event

    **Request:**
    ```http
    POST /api/v1/events/evt_abc123/rsvp
    Content-Type: application/json

    {
      "response": "accepted",  // accepted, declined, tentative
      "comment": "Looking forward to it!"
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "event_id": "evt_abc123",
      "attendee": "alice@example.com",
      "status": "accepted",
      "updated_at": "2024-01-15T08:00:00Z"
    }
    ```

    ---

    ### 5. Get Events for Date Range

    **Request:**
    ```http
    GET /api/v1/events?start=2024-01-15T00:00:00Z&end=2024-01-22T23:59:59Z&time_zone=America/New_York
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "events": [
        {
          "event_id": "evt_abc123",
          "title": "Team Standup",
          "start_time": "2024-01-15T10:00:00Z",
          "end_time": "2024-01-15T10:30:00Z",
          "is_recurring": true,
          "attendees_count": 5,
          "my_status": "accepted",
          "conflicts": false
        },
        // Recurring event instances expanded
        {
          "event_id": "evt_abc123",
          "instance_date": "2024-01-16T10:00:00Z",
          "title": "Team Standup",
          "start_time": "2024-01-16T10:00:00Z",
          "end_time": "2024-01-16T10:30:00Z",
          "is_recurring": true
        }
      ],
      "total": 42
    }
    ```

    ---

    ## Data Models

    ### Events Table

    ```sql
    CREATE TABLE events (
        event_id VARCHAR(64) PRIMARY KEY,
        calendar_id BIGINT NOT NULL,
        creator_id BIGINT NOT NULL,

        -- Event details
        title VARCHAR(255) NOT NULL,
        description TEXT,
        location VARCHAR(512),

        -- Time (stored in UTC)
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        time_zone VARCHAR(64) NOT NULL,
        all_day BOOLEAN DEFAULT FALSE,

        -- Recurrence
        is_recurring BOOLEAN DEFAULT FALSE,
        recurrence_id VARCHAR(64),  -- Links to recurrence_rules
        parent_event_id VARCHAR(64),  -- For recurring series

        -- Status
        status VARCHAR(32) DEFAULT 'confirmed',  -- confirmed, cancelled, tentative
        visibility VARCHAR(32) DEFAULT 'default',  -- default, public, private

        -- Metadata
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),

        INDEX idx_calendar_time (calendar_id, start_time, end_time),
        INDEX idx_creator (creator_id),
        INDEX idx_recurrence (recurrence_id),
        INDEX idx_time_range (start_time, end_time)
    );
    ```

    ### Recurrence Rules Table (RRULE)

    ```sql
    CREATE TABLE recurrence_rules (
        recurrence_id VARCHAR(64) PRIMARY KEY,
        event_id VARCHAR(64) NOT NULL,

        -- RFC 5545 RRULE components
        frequency VARCHAR(16) NOT NULL,  -- DAILY, WEEKLY, MONTHLY, YEARLY
        interval INT DEFAULT 1,
        count INT,  -- Number of occurrences
        until TIMESTAMP,  -- End date

        -- By-rules (stored as JSON arrays)
        by_day VARCHAR(255),  -- ["MO", "WE", "FR"]
        by_month_day VARCHAR(255),  -- [1, 15, -1]
        by_month VARCHAR(255),  -- [1, 6, 12]
        by_set_pos VARCHAR(255),  -- [1, -1] for first/last

        -- Week start day
        week_start VARCHAR(2) DEFAULT 'MO',

        -- Timezone for recurrence
        time_zone VARCHAR(64) NOT NULL,

        created_at TIMESTAMP DEFAULT NOW(),

        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        INDEX idx_event (event_id)
    );
    ```

    ### Event Exceptions Table

    ```sql
    CREATE TABLE event_exceptions (
        exception_id VARCHAR(64) PRIMARY KEY,
        parent_event_id VARCHAR(64) NOT NULL,
        original_start_time TIMESTAMP NOT NULL,  -- The instance being modified

        -- Modified fields (NULL = use parent)
        title VARCHAR(255),
        description TEXT,
        location VARCHAR(512),
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        status VARCHAR(32),  -- cancelled for deleted instances

        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),

        FOREIGN KEY (parent_event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        INDEX idx_parent (parent_event_id, original_start_time),
        UNIQUE (parent_event_id, original_start_time)
    );
    ```

    ### Attendees Table

    ```sql
    CREATE TABLE event_attendees (
        attendee_id SERIAL PRIMARY KEY,
        event_id VARCHAR(64) NOT NULL,
        user_id BIGINT,
        email VARCHAR(255) NOT NULL,

        -- RSVP status
        status VARCHAR(32) DEFAULT 'pending',  -- pending, accepted, declined, tentative
        response_comment TEXT,
        optional BOOLEAN DEFAULT FALSE,

        -- Notifications
        reminded BOOLEAN DEFAULT FALSE,

        responded_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW(),

        FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE,
        INDEX idx_event (event_id),
        INDEX idx_user (user_id),
        INDEX idx_email (email),
        UNIQUE (event_id, email)
    );
    ```

    ### Calendar Sharing Table

    ```sql
    CREATE TABLE calendar_shares (
        share_id SERIAL PRIMARY KEY,
        calendar_id BIGINT NOT NULL,
        shared_with_user_id BIGINT NOT NULL,
        permission VARCHAR(32) NOT NULL,  -- view, edit, admin

        created_at TIMESTAMP DEFAULT NOW(),

        INDEX idx_calendar (calendar_id),
        INDEX idx_user (shared_with_user_id),
        UNIQUE (calendar_id, shared_with_user_id)
    );
    ```

    ---

    ## Event Flow Diagrams

    ### Create Recurring Event Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant RecurService
        participant ConflictCheck
        participant DB
        participant Cache
        participant Queue

        Client->>API: Create recurring event
        API->>RecurService: Parse RRULE
        RecurService->>RecurService: Validate recurrence pattern

        RecurService->>ConflictCheck: Check first instance conflicts
        ConflictCheck->>DB: Query overlapping events
        ConflictCheck-->>RecurService: No conflicts

        RecurService->>DB: Insert event record
        RecurService->>DB: Insert recurrence_rule

        RecurService->>Cache: Cache event + rule
        RecurService->>Queue: Schedule reminders

        RecurService-->>API: Event created
        API-->>Client: Return event_id + recurrence_id
    ```

    ### Availability Check Flow

    ```mermaid
    sequenceDiagram
        participant Client
        participant API
        participant AvailService
        participant Cache
        participant DB

        Client->>API: Check availability for users
        API->>AvailService: Get free/busy for time range

        loop For each user
            AvailService->>Cache: Get cached events
            alt Cache hit
                Cache-->>AvailService: Cached events
            else Cache miss
                AvailService->>DB: Query events in range
                DB-->>AvailService: Events
                AvailService->>AvailService: Expand recurring events
                AvailService->>Cache: Cache expanded events
            end
        end

        AvailService->>AvailService: Calculate free/busy slots
        AvailService->>AvailService: Find common free time
        AvailService-->>API: Availability data
        API-->>Client: Return free/busy slots
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics

    ### 1. Recurring Event Implementation (RRULE)

    **RFC 5545 (iCalendar) RRULE Format:**

    ```python
    from datetime import datetime, timedelta
    from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY, YEARLY, MO, TU, WE, TH, FR

    class RecurrenceEngine:
        """Handle recurring event generation using RRULE"""

        FREQUENCY_MAP = {
            'DAILY': DAILY,
            'WEEKLY': WEEKLY,
            'MONTHLY': MONTHLY,
            'YEARLY': YEARLY
        }

        WEEKDAY_MAP = {
            'MO': MO, 'TU': TU, 'WE': WE,
            'TH': TH, 'FR': FR, 'SA': SA, 'SU': SU
        }

        def generate_occurrences(self, event: dict, recurrence_rule: dict,
                                start_date: datetime, end_date: datetime) -> list:
            """Generate event occurrences for date range"""

            # Parse RRULE
            freq = self.FREQUENCY_MAP[recurrence_rule['frequency']]
            interval = recurrence_rule.get('interval', 1)
            count = recurrence_rule.get('count')
            until = recurrence_rule.get('until')

            # Parse by_day
            by_day = None
            if recurrence_rule.get('by_day'):
                by_day = [self.WEEKDAY_MAP[d] for d in recurrence_rule['by_day']]

            # Generate occurrences
            rule = rrule(
                freq=freq,
                interval=interval,
                count=count,
                until=until,
                byweekday=by_day,
                dtstart=event['start_time']
            )

            # Filter to requested range
            occurrences = []
            for dt in rule:
                if dt > end_date:
                    break
                if dt >= start_date:
                    duration = event['end_time'] - event['start_time']
                    occurrences.append({
                        'event_id': event['event_id'],
                        'instance_date': dt,
                        'start_time': dt,
                        'end_time': dt + duration,
                        'title': event['title'],
                        'is_recurring_instance': True
                    })

            return occurrences

        def parse_rrule_string(self, rrule_str: str) -> dict:
            """Parse iCalendar RRULE string

            Example: FREQ=WEEKLY;INTERVAL=1;BYDAY=MO,WE,FR;UNTIL=20241231T235959Z
            """
            parts = rrule_str.split(';')
            rule = {}

            for part in parts:
                if '=' not in part:
                    continue

                key, value = part.split('=', 1)

                if key == 'FREQ':
                    rule['frequency'] = value
                elif key == 'INTERVAL':
                    rule['interval'] = int(value)
                elif key == 'COUNT':
                    rule['count'] = int(value)
                elif key == 'UNTIL':
                    rule['until'] = datetime.strptime(value, '%Y%m%dT%H%M%SZ')
                elif key == 'BYDAY':
                    rule['by_day'] = value.split(',')
                elif key == 'BYMONTHDAY':
                    rule['by_month_day'] = [int(d) for d in value.split(',')]
                elif key == 'BYMONTH':
                    rule['by_month'] = [int(m) for m in value.split(',')]

            return rule

        def build_rrule_string(self, recurrence: dict) -> str:
            """Build iCalendar RRULE string"""
            parts = [f"FREQ={recurrence['frequency']}"]

            if recurrence.get('interval', 1) > 1:
                parts.append(f"INTERVAL={recurrence['interval']}")

            if recurrence.get('count'):
                parts.append(f"COUNT={recurrence['count']}")

            if recurrence.get('until'):
                until_str = recurrence['until'].strftime('%Y%m%dT%H%M%SZ')
                parts.append(f"UNTIL={until_str}")

            if recurrence.get('by_day'):
                days = ','.join(recurrence['by_day'])
                parts.append(f"BYDAY={days}")

            return ';'.join(parts)
    ```

    **Example RRULE patterns:**

    ```python
    # Daily standup (weekdays only)
    {
        "frequency": "DAILY",
        "interval": 1,
        "by_day": ["MO", "TU", "WE", "TH", "FR"],
        "until": "2024-12-31T23:59:59Z"
    }

    # Weekly team meeting (every Monday)
    {
        "frequency": "WEEKLY",
        "interval": 1,
        "by_day": ["MO"],
        "count": 52  # 1 year
    }

    # Monthly all-hands (first Friday of each month)
    {
        "frequency": "MONTHLY",
        "interval": 1,
        "by_day": ["FR"],
        "by_set_pos": [1]  # First occurrence
    }

    # Bi-weekly sprint planning (every other Monday)
    {
        "frequency": "WEEKLY",
        "interval": 2,
        "by_day": ["MO"]
    }
    ```

    ---

    ### 2. Conflict Detection

    ```python
    class ConflictDetector:
        """Detect and resolve calendar conflicts"""

        def check_conflicts(self, user_id: int, new_event: dict,
                          exclude_event_id: str = None) -> list:
            """Check if new event conflicts with existing events"""

            # Query overlapping events
            conflicts = db.query("""
                SELECT e.event_id, e.title, e.start_time, e.end_time
                FROM events e
                JOIN event_attendees ea ON e.event_id = ea.event_id
                WHERE ea.user_id = ?
                AND e.event_id != COALESCE(?, '')
                AND e.status != 'cancelled'
                AND (
                    -- New event starts during existing event
                    (? >= e.start_time AND ? < e.end_time)
                    OR
                    -- New event ends during existing event
                    (? > e.start_time AND ? <= e.end_time)
                    OR
                    -- New event completely contains existing event
                    (? <= e.start_time AND ? >= e.end_time)
                )
            """, user_id, exclude_event_id,
                new_event['start_time'], new_event['start_time'],
                new_event['end_time'], new_event['end_time'],
                new_event['start_time'], new_event['end_time'])

            return conflicts

        def check_recurring_conflicts(self, user_id: int, event: dict,
                                     recurrence: dict, check_range_days: int = 90) -> dict:
            """Check conflicts for recurring event (limited range)"""

            # Generate instances for next N days
            engine = RecurrenceEngine()
            end_date = datetime.utcnow() + timedelta(days=check_range_days)
            instances = engine.generate_occurrences(
                event, recurrence,
                start_date=event['start_time'],
                end_date=end_date
            )

            # Check each instance
            conflicts_by_date = {}
            for instance in instances:
                conflicts = self.check_conflicts(user_id, instance)
                if conflicts:
                    conflicts_by_date[instance['instance_date']] = conflicts

            return {
                'has_conflicts': len(conflicts_by_date) > 0,
                'total_conflicts': len(conflicts_by_date),
                'conflicts_by_date': conflicts_by_date,
                'checked_until': end_date
            }

        def get_availability_matrix(self, user_ids: list, start_time: datetime,
                                   end_time: datetime, interval_minutes: int = 30) -> dict:
            """Get availability matrix for multiple users"""

            # Time slots
            slots = []
            current = start_time
            while current < end_time:
                slots.append(current)
                current += timedelta(minutes=interval_minutes)

            # Get all events for users in range
            availability = {}
            for user_id in user_ids:
                events = self.get_user_events(user_id, start_time, end_time)
                user_availability = self.calculate_free_busy(events, slots)
                availability[user_id] = user_availability

            # Find common free slots
            common_free = self.find_common_free_slots(availability, slots)

            return {
                'time_zone': 'UTC',
                'interval_minutes': interval_minutes,
                'slots': slots,
                'availability': availability,
                'common_free_slots': common_free
            }

        def calculate_free_busy(self, events: list, slots: list) -> dict:
            """Calculate free/busy for time slots"""
            busy_slots = set()

            for event in events:
                for slot in slots:
                    slot_end = slot + timedelta(minutes=30)
                    # Check if slot overlaps with event
                    if (slot < event['end_time'] and slot_end > event['start_time']):
                        busy_slots.add(slot)

            free_slots = [s for s in slots if s not in busy_slots]
            busy_slots_list = sorted(list(busy_slots))

            return {
                'free_slots': free_slots,
                'busy_slots': busy_slots_list,
                'total_free_minutes': len(free_slots) * 30,
                'total_busy_minutes': len(busy_slots_list) * 30
            }
    ```

    ---

    ### 3. Time Zone Handling

    ```python
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import pytz

    class TimeZoneHandler:
        """Handle time zone conversions and DST"""

        def store_event(self, event_data: dict) -> dict:
            """Convert event times to UTC for storage"""

            # User provides time in their timezone
            user_tz = ZoneInfo(event_data['time_zone'])

            # Parse time in user's timezone
            start_local = datetime.fromisoformat(event_data['start_time'])
            end_local = datetime.fromisoformat(event_data['end_time'])

            # Localize to user's timezone
            start_aware = start_local.replace(tzinfo=user_tz)
            end_aware = end_local.replace(tzinfo=user_tz)

            # Convert to UTC for storage
            start_utc = start_aware.astimezone(ZoneInfo('UTC'))
            end_utc = end_aware.astimezone(ZoneInfo('UTC'))

            return {
                'start_time': start_utc.isoformat(),
                'end_time': end_utc.isoformat(),
                'time_zone': event_data['time_zone']  # Store original TZ
            }

        def display_event(self, event: dict, target_timezone: str) -> dict:
            """Convert event from UTC to user's timezone"""

            target_tz = ZoneInfo(target_timezone)

            # Parse UTC times
            start_utc = datetime.fromisoformat(event['start_time'])
            end_utc = datetime.fromisoformat(event['end_time'])

            # Convert to target timezone
            start_local = start_utc.astimezone(target_tz)
            end_local = end_utc.astimezone(target_tz)

            return {
                'start_time': start_local.isoformat(),
                'end_time': end_local.isoformat(),
                'time_zone': target_timezone,
                'start_time_display': start_local.strftime('%Y-%m-%d %I:%M %p %Z'),
                'is_dst': start_local.dst() != timedelta(0)
            }

        def handle_recurring_dst(self, event: dict, recurrence: dict) -> list:
            """Handle DST transitions for recurring events"""

            # Example: Meeting at 9 AM in America/New_York
            # During DST: 9 AM EDT = 1 PM UTC
            # During Standard: 9 AM EST = 2 PM UTC

            # Generate occurrences in local time, then convert to UTC
            engine = RecurrenceEngine()
            tz = ZoneInfo(event['time_zone'])

            # Generate in local time
            local_times = engine.generate_occurrences(event, recurrence)

            # Convert each to UTC (handles DST automatically)
            utc_times = []
            for local_time in local_times:
                local_aware = local_time['start_time'].replace(tzinfo=tz)
                utc_time = local_aware.astimezone(ZoneInfo('UTC'))
                utc_times.append(utc_time)

            return utc_times
    ```

    ---

    ### 4. Notification System

    ```python
    import asyncio
    from datetime import datetime, timedelta

    class NotificationScheduler:
        """Schedule and send event reminders"""

        def schedule_reminders(self, event: dict, attendees: list, reminders: list):
            """Schedule reminders for event"""

            for reminder in reminders:
                for attendee in attendees:
                    # Calculate reminder time
                    reminder_time = event['start_time'] - timedelta(
                        minutes=reminder['minutes_before']
                    )

                    # Schedule job
                    job_data = {
                        'event_id': event['event_id'],
                        'attendee_email': attendee['email'],
                        'reminder_time': reminder_time,
                        'method': reminder['method'],  # email, push, sms
                        'event_title': event['title'],
                        'event_start': event['start_time']
                    }

                    # Add to queue with delay
                    delay_seconds = (reminder_time - datetime.utcnow()).total_seconds()
                    if delay_seconds > 0:
                        queue.enqueue_at(reminder_time, 'send_reminder', job_data)

        def schedule_recurring_reminders(self, event: dict, recurrence: dict,
                                        reminders: list, days_ahead: int = 30):
            """Schedule reminders for recurring event (next N days)"""

            engine = RecurrenceEngine()
            end_date = datetime.utcnow() + timedelta(days=days_ahead)
            instances = engine.generate_occurrences(
                event, recurrence,
                start_date=datetime.utcnow(),
                end_date=end_date
            )

            # Schedule reminders for each instance
            for instance in instances:
                self.schedule_reminders(instance, event['attendees'], reminders)

        async def send_reminder(self, job_data: dict):
            """Send reminder notification"""

            # Get attendee preferences
            attendee = db.get_attendee(job_data['attendee_email'])

            # Format message
            message = self.format_reminder_message(job_data)

            # Send based on method
            if job_data['method'] == 'email':
                await self.send_email_reminder(attendee, message)
            elif job_data['method'] == 'push':
                await self.send_push_reminder(attendee, message)
            elif job_data['method'] == 'sms':
                await self.send_sms_reminder(attendee, message)

            # Mark as sent
            db.mark_reminder_sent(job_data['event_id'], attendee['email'])

        def format_reminder_message(self, job_data: dict) -> dict:
            """Format reminder message"""

            time_until = job_data['event_start'] - datetime.utcnow()

            if time_until.total_seconds() < 3600:
                time_str = f"in {int(time_until.total_seconds() / 60)} minutes"
            else:
                time_str = f"in {int(time_until.total_seconds() / 3600)} hours"

            return {
                'subject': f"Reminder: {job_data['event_title']} {time_str}",
                'body': f"Your event '{job_data['event_title']}' starts {time_str}",
                'event_id': job_data['event_id'],
                'action_url': f"https://calendar.example.com/events/{job_data['event_id']}"
            }
    ```

    ---

    ### 5. Calendar Sharing & Permissions

    ```python
    class CalendarSharingService:
        """Handle calendar sharing and permissions"""

        PERMISSIONS = {
            'view': ['read'],
            'edit': ['read', 'create', 'update'],
            'admin': ['read', 'create', 'update', 'delete', 'share']
        }

        def share_calendar(self, calendar_id: int, owner_id: int,
                          share_with_email: str, permission: str):
            """Share calendar with another user"""

            # Get or create user
            shared_user = db.get_user_by_email(share_with_email)
            if not shared_user:
                # Send invitation email
                shared_user = self.invite_user(share_with_email)

            # Create share record
            db.execute("""
                INSERT INTO calendar_shares
                (calendar_id, shared_with_user_id, permission)
                VALUES (?, ?, ?)
                ON CONFLICT (calendar_id, shared_with_user_id)
                DO UPDATE SET permission = ?
            """, calendar_id, shared_user['user_id'], permission, permission)

            # Notify user
            self.notify_calendar_shared(shared_user, calendar_id, permission)

        def check_permission(self, user_id: int, calendar_id: int,
                           action: str) -> bool:
            """Check if user has permission for action"""

            # Owner has all permissions
            calendar = db.get_calendar(calendar_id)
            if calendar['owner_id'] == user_id:
                return True

            # Check shared permissions
            share = db.query("""
                SELECT permission
                FROM calendar_shares
                WHERE calendar_id = ? AND shared_with_user_id = ?
            """, calendar_id, user_id)

            if not share:
                return False

            # Check if permission includes action
            allowed_actions = self.PERMISSIONS.get(share['permission'], [])
            return action in allowed_actions

        def get_free_busy_access(self, calendar_id: int, requester_id: int) -> bool:
            """Check if user can see free/busy (less restrictive than full view)"""

            # Free/busy is available if:
            # 1. User has view permission
            # 2. Calendar is marked as public
            # 3. Organization-wide access (same domain)

            calendar = db.get_calendar(calendar_id)

            # Public calendar
            if calendar['visibility'] == 'public':
                return True

            # Has view permission
            if self.check_permission(requester_id, calendar_id, 'read'):
                return True

            # Same organization
            owner = db.get_user(calendar['owner_id'])
            requester = db.get_user(requester_id)
            if owner['email'].split('@')[1] == requester['email'].split('@')[1]:
                return True

            return False
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## Scalability Strategies

    ### 1. Caching Strategy

    **Multi-level caching:**

    ```python
    class CalendarCache:
        """Multi-level caching for calendar data"""

        def __init__(self, redis_client, local_cache):
            self.redis = redis_client
            self.local = local_cache
            self.event_ttl = 3600  # 1 hour
            self.availability_ttl = 300  # 5 minutes

        def get_events_for_range(self, calendar_id: int, start: datetime,
                                end: datetime) -> list:
            """Get events with caching"""

            # Cache key
            cache_key = f"cal:{calendar_id}:events:{start.date()}:{end.date()}"

            # Try local cache (L1)
            events = self.local.get(cache_key)
            if events:
                return events

            # Try Redis (L2)
            cached = self.redis.get(cache_key)
            if cached:
                events = json.loads(cached)
                self.local.set(cache_key, events, ttl=60)
                return events

            # Query database (L3)
            events = db.get_events(calendar_id, start, end)

            # Expand recurring events
            expanded = self.expand_recurring_events(events, start, end)

            # Cache in Redis
            self.redis.setex(cache_key, self.event_ttl, json.dumps(expanded))
            self.local.set(cache_key, expanded, ttl=60)

            return expanded

        def invalidate_event(self, event_id: str):
            """Invalidate cache when event changes"""

            # Get event details
            event = db.get_event(event_id)

            # Invalidate relevant cache entries
            # For recurring events, invalidate broader range
            if event['is_recurring']:
                # Invalidate next 90 days
                for i in range(90):
                    date = datetime.utcnow().date() + timedelta(days=i)
                    pattern = f"cal:{event['calendar_id']}:events:*{date}*"
                    keys = self.redis.keys(pattern)
                    if keys:
                        self.redis.delete(*keys)
            else:
                # Invalidate specific date
                date = event['start_time'].date()
                pattern = f"cal:{event['calendar_id']}:events:*{date}*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)

        def cache_availability(self, user_id: int, date: datetime.date,
                             availability: dict):
            """Cache availability data (short TTL)"""
            cache_key = f"avail:{user_id}:{date}"
            self.redis.setex(cache_key, self.availability_ttl,
                           json.dumps(availability))
    ```

    ---

    ### 2. Database Optimization

    **Sharding strategy:**

    ```python
    class CalendarSharding:
        """Shard calendars by user_id"""

        def __init__(self, num_shards=16):
            self.num_shards = num_shards

        def get_shard(self, user_id: int) -> int:
            """Determine shard for user"""
            return user_id % self.num_shards

        def get_shard_db(self, user_id: int):
            """Get database connection for shard"""
            shard_id = self.get_shard(user_id)
            return db_pool.get_connection(f"calendar_shard_{shard_id}")
    ```

    **Optimized queries:**

    ```sql
    -- Create materialized view for common availability queries
    CREATE MATERIALIZED VIEW user_daily_availability AS
    SELECT
        ea.user_id,
        DATE(e.start_time) as event_date,
        e.start_time,
        e.end_time,
        e.status
    FROM events e
    JOIN event_attendees ea ON e.event_id = ea.event_id
    WHERE e.status = 'confirmed'
    AND e.start_time >= CURRENT_DATE - INTERVAL '7 days'
    AND e.start_time <= CURRENT_DATE + INTERVAL '90 days';

    -- Refresh daily
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_daily_availability;

    -- Index for fast lookups
    CREATE INDEX idx_user_daily_avail
    ON user_daily_availability(user_id, event_date);
    ```

    ---

    ### 3. Recurring Event Optimization

    **Lazy expansion:**

    ```python
    class LazyRecurrenceExpander:
        """Expand recurring events on-demand"""

        def get_events_optimized(self, calendar_id: int, start: datetime,
                                end: datetime) -> list:
            """Get events with optimized recurring expansion"""

            # Get all events in range (including recurring)
            events = db.query("""
                SELECT e.*, r.*
                FROM events e
                LEFT JOIN recurrence_rules r ON e.recurrence_id = r.recurrence_id
                WHERE e.calendar_id = ?
                AND (
                    -- Non-recurring events in range
                    (e.is_recurring = false
                     AND e.start_time >= ?
                     AND e.end_time <= ?)
                    OR
                    -- Recurring events that might have instances in range
                    (e.is_recurring = true
                     AND e.start_time <= ?
                     AND (r.until IS NULL OR r.until >= ?))
                )
            """, calendar_id, start, end, end, start)

            result = []
            engine = RecurrenceEngine()

            for event in events:
                if event['is_recurring']:
                    # Expand only for requested range
                    instances = engine.generate_occurrences(
                        event, event['recurrence_rule'], start, end
                    )
                    result.extend(instances)
                else:
                    result.append(event)

            return result
    ```

    ---

    ### 4. Monitoring Metrics

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Event Creation Latency** | < 100ms | > 500ms |
    | **Availability Check Latency** | < 200ms | > 1s |
    | **Cache Hit Rate** | > 80% | < 60% |
    | **Conflict Detection Time** | < 50ms | > 200ms |
    | **Notification Delivery Rate** | > 99% | < 95% |
    | **RSVP Update Latency** | < 100ms | > 300ms |

    ---

    ## Performance Optimizations

    ### 1. Batch Operations

    ```python
    def batch_create_events(events: list) -> list:
        """Create multiple events in single transaction"""

        with db.transaction():
            event_ids = []
            for event in events:
                event_id = db.insert_event(event)
                event_ids.append(event_id)

                # Batch insert attendees
                if event.get('attendees'):
                    db.bulk_insert_attendees(event_id, event['attendees'])

            return event_ids
    ```

    ### 2. Connection Pooling

    ```python
    # Database connection pool
    db_pool = ConnectionPool(
        min_connections=10,
        max_connections=100,
        connection_timeout=30
    )

    # Redis connection pool
    redis_pool = redis.ConnectionPool(
        host='localhost',
        port=6379,
        max_connections=50
    )
    ```

    ### 3. Asynchronous Processing

    ```python
    async def process_event_updates(event_id: str, updates: dict):
        """Process event updates asynchronously"""

        # Update event
        await db.update_event_async(event_id, updates)

        # Parallel tasks
        await asyncio.gather(
            invalidate_cache_async(event_id),
            notify_attendees_async(event_id, updates),
            recheck_conflicts_async(event_id)
        )
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **RRULE for recurrence** - RFC 5545 standard for recurring events
    2. **UTC storage** - Store all times in UTC, display in user's timezone
    3. **Lazy expansion** - Expand recurring events on-demand, not at creation
    4. **Multi-level caching** - Local + Redis for active events
    5. **Event exceptions** - Handle single instance modifications efficiently
    6. **Queue-based notifications** - Scheduled reminders with retry logic

    ## Interview Tips

    ✅ **Discuss RRULE** - Explain RFC 5545, common patterns (daily, weekly, etc.)
    ✅ **Explain time zones** - UTC storage, DST handling, display conversion
    ✅ **Cover conflict detection** - Overlapping events, free/busy calculation
    ✅ **Address scalability** - Sharding, caching, lazy expansion
    ✅ **Handle edge cases** - DST transitions, deleted instances, all-day events

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to handle DST transitions?"** | Store in UTC, recurrence in local time, convert per instance |
    | **"How to modify single recurring instance?"** | Event exceptions table, override specific fields |
    | **"How to find common free time for 10 people?"** | Parallel availability check, merge free slots, cache results |
    | **"How to prevent double-booking?"** | Pessimistic locking on event creation, conflict check in transaction |
    | **"How to scale to 1B events?"** | Shard by calendar_id, archive old events, materialized views |
    | **"How to handle timezones for recurring events?"** | Store DTSTART timezone, apply to each occurrence |

    ## Real-World Examples

    - **Google Calendar**: RRULE recurrence, conflict detection, free/busy
    - **Microsoft Outlook**: Exchange protocol, room booking, availability
    - **Apple Calendar**: CalDAV protocol, iCloud sync
    - **Calendly**: Availability-first design, time slot booking

    ## Advanced Topics

    ### 1. Calendar Synchronization (CalDAV)

    ```python
    # CalDAV protocol for cross-platform sync
    class CalDAVSync:
        def sync_calendar(self, calendar_id: int, last_sync: datetime):
            """Sync changes since last sync"""
            changes = db.get_changes_since(calendar_id, last_sync)
            return {
                'added': changes['new_events'],
                'modified': changes['updated_events'],
                'deleted': changes['deleted_events'],
                'sync_token': generate_sync_token()
            }
    ```

    ### 2. Smart Scheduling

    ```python
    def suggest_meeting_times(attendees: list, duration_minutes: int,
                             preferences: dict) -> list:
        """AI-powered meeting time suggestions"""

        # Get availability for all attendees
        availability = get_multi_user_availability(attendees)

        # Apply preferences (work hours, preferred times)
        filtered = apply_preferences(availability, preferences)

        # Rank by quality (all available, no conflicts, good time)
        ranked = rank_time_slots(filtered)

        return ranked[:5]  # Top 5 suggestions
    ```

    ### 3. Room/Resource Booking

    ```python
    class ResourceBooking:
        """Book conference rooms and resources"""

        def book_resource(self, resource_id: str, event_id: str,
                         start_time: datetime, end_time: datetime) -> bool:
            """Book resource with conflict check"""

            # Check availability
            conflicts = self.check_resource_conflicts(
                resource_id, start_time, end_time
            )

            if conflicts:
                return False

            # Pessimistic lock for booking
            with db.transaction():
                # Double-check availability
                if self.check_resource_conflicts(resource_id, start_time, end_time):
                    return False

                # Create booking
                db.insert_resource_booking(resource_id, event_id, start_time, end_time)
                return True
    ```

---

**Difficulty:** 🟡 Medium | **Interview Time:** 45-60 minutes | **Companies:** Google, Microsoft, Apple, Zoom, Calendly

---

*This problem demonstrates event management, recurrence patterns, time zone handling, and conflict resolution essential for calendar and scheduling systems.*
