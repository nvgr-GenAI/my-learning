# Problem Scope and Requirements

The first step in a system design interview is to clarify the scope and gather detailed requirements.

## 📋 Problem Statement

Design a global video streaming platform, similar to YouTube or Netflix, capable of handling millions of concurrent users and storing exabytes of video content.

### Core Features to Consider:
- **Video Lifecycle:** Users can upload, process, and stream videos.
- **User Experience:** The platform must provide low-latency global delivery, personalized recommendations, and support for various devices (mobile, web, TV).
- **Scale & Reliability:** The system must be highly available and scalable to handle a massive user base.

## 🎯 Requirements Gathering

### Functional Requirements

- **Video Upload:** Users can upload videos. The system should handle large files and various formats (e.g., MP4, MOV).
- **Video Processing (Transcoding):** Videos must be converted into multiple resolutions and formats (e.g., 1080p, 720p, 480p) to support adaptive bitrate streaming.
- **Video Playback:** Users can stream videos with minimal start time and buffering.
- **Search & Discovery:** Users can search for videos and discover new content through recommendations.
- **User & Profile Management:** Users can create accounts, manage profiles, and view their watch history.
- **Live Streaming:** The platform should support real-time streaming events.

### Non-Functional Requirements

- **High Availability:** The system must achieve 99.9% uptime.
- **Low Latency:** Video start time should be under 2 seconds, with global CDN latency under 100ms.
- **Scalability:** The system must support billions of users and 500+ hours of video uploaded per minute.
- **Durability:** Uploaded videos must never be lost. Data should be backed up and replicated.
- **Consistency:** While strong consistency is not required for all features (e.g., view counts), metadata and user information should be reasonably consistent. Eventual consistency is acceptable for many parts of the system.
- **Cost-Effectiveness:** The design should optimize for storage and compute costs.
