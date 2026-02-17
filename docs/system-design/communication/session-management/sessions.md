# Session Management

HTTP is a stateless protocol. Every request arrives at the server with no memory of what came before it, as if the user is a stranger each time. Yet modern web applications must remember who you are, what is in your shopping cart, and whether you have permission to view a page. Session management bridges this gap by associating a persistent identity with a series of stateless requests.

The fundamental design decision is where to store that state. Server-side sessions keep data on infrastructure you control, giving you the power to revoke access instantly but requiring shared storage as you scale horizontally. Client-side sessions embed state in tokens the browser carries, eliminating server storage entirely but making immediate revocation difficult. Every session architecture is a negotiation between these two poles.

## Session Storage Comparison

| Approach | Where State Lives | Horizontal Scalability | Instant Revocation | Practical Size Limit |
| --- | --- | --- | --- | --- |
| Server-side (Redis) | Redis cluster | High with shared store | Yes, delete the key | Megabytes per session |
| Server-side (Database) | PostgreSQL / MySQL | Moderate, DB is bottleneck | Yes, delete the row | Limited by DB capacity |
| JWT (signed token) | Client cookie or header | Excellent, no server lookup | No, must wait for expiry | ~4 KB (cookie limit) |
| Signed cookie | Client cookie | Excellent, no server lookup | No, must wait for expiry | 4 KB hard limit |

=== "Server-Side Sessions"

    Server-side sessions follow a simple principle: the server keeps all session data and hands the client nothing more than an opaque identifier. When a request arrives, the server looks up that identifier in a shared store and retrieves the full session context.

    The request lifecycle looks like this:

    ```
    Browser                Load Balancer          App Server           Redis
      |                        |                      |                  |
      |--- GET /dashboard ---->|                      |                  |
      |    Cookie: sid=a3f9... |                      |                  |
      |                        |--- forward --------->|                  |
      |                        |                      |--- GET sid ------>|
      |                        |                      |<-- session data --|
      |                        |                      |                  |
      |                        |<-- HTML response ----|                  |
      |<-- 200 OK ------------|                      |                  |
    ```

    Without a centralized store, horizontal scaling forces you into sticky sessions, where the load balancer routes each user to the same server that created their session. This creates a fragile coupling: if that server goes down, the user loses their session. A centralized store like Redis eliminates this problem entirely. Any application server can serve any request because they all read from the same place.

    Shopify runs one of the largest Redis session stores in e-commerce, managing over 80 million active sessions during peak sales events like Black Friday. Their architecture uses Redis Cluster with automatic failover, ensuring that even if a Redis node fails, sessions are preserved on replica nodes without any user-visible disruption.

    Session ID generation is a security-critical operation. A predictable session ID allows an attacker to guess valid sessions and hijack them. The session ID must come from a cryptographically secure random number generator (such as `/dev/urandom` or `secrets.token_hex` in Python) with at least 128 bits of entropy. A typical implementation produces a 64-character hex string:

    ```python
    import secrets
    session_id = secrets.token_hex(32)  # 256-bit, 64 hex chars
    ```

    The server then stores session data keyed by this ID with a TTL matching the desired session lifetime, and sets the ID as an HTTP cookie on the response.

=== "Client-Side Sessions"

    Client-side sessions reverse the storage model. Instead of keeping state on the server, the server encodes session data into a token and hands it to the client. The most widely adopted format is the JSON Web Token (JWT), which has three Base64URL-encoded parts separated by dots:

    ```
    eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIiwiZXhwIjoxNzA3MH0.SflKxwRJSMeKKF2QT4fwpM
    |___ Header ___|     |_______ Payload _______|     |_____ Signature _____|

    Header:    {"alg": "HS256", "typ": "JWT"}
    Payload:   {"sub": "user123", "role": "admin", "exp": 1707000000}
    Signature: HMAC-SHA256(header + "." + payload, server_secret)
    ```

    The server never needs to look anything up. It verifies the signature using its secret key, checks the expiration claim, and trusts the payload. This makes JWTs ideal for distributed systems where many services need to validate identity without calling a central session store.

    The token lifecycle follows a pattern of short-lived access tokens paired with longer-lived refresh tokens. When the access token expires (typically after 15 to 30 minutes), the client uses the refresh token to obtain a new one without requiring the user to log in again. Refresh token rotation is the recommended practice: each time a refresh token is used, the server issues a new one and invalidates the old one. If an attacker steals a refresh token, the legitimate user's next refresh attempt will fail, signaling a compromise.

    ```
    Client                          Auth Server
      |                                  |
      |--- POST /token (credentials) --->|
      |<-- access_token + refresh_token --|
      |                                  |
      |  ... access_token expires ...    |
      |                                  |
      |--- POST /refresh (refresh_v1) -->|
      |<-- new access + refresh_v2 ------|
      |    (refresh_v1 invalidated)      |
      |                                  |
      |  ... if attacker uses v1 ...     |
      |                                  |
      |--- POST /refresh (refresh_v1) -->|
      |<-- 401 DENIED (token reuse!) ----|
      |    (all tokens for user revoked) |
    ```

    Choosing between JWTs and opaque tokens depends on your architecture:

    | Characteristic | JWT (Self-Contained) | Opaque Token |
    | --- | --- | --- |
    | Server lookup required | No | Yes, every request |
    | Instant revocation | No (requires blocklist) | Yes, delete from store |
    | Token size | 500+ bytes (grows with claims) | ~32 bytes (just an ID) |
    | Cross-service validation | Yes, any service with the key | No, must call issuer |
    | Best suited for | Microservices, APIs | Monoliths, high-security apps |

    Auth0, one of the largest identity platforms, processes over 4.5 billion logins per year using JWT-based flows. Their architecture issues short-lived access tokens (5 to 15 minutes) and stores refresh tokens server-side in encrypted form, combining the scalability of JWTs with the revocation capability of server-side storage.

=== "Stateless Design"

    Stateless architectures achieve horizontal scalability by ensuring that no single server holds unique state. Any server in the pool can handle any request, which means you can add or remove servers freely, route traffic through any load balancer algorithm, and recover from server failures without losing user context.

    The key principle is extracting all state to external stores. Rather than keeping session data in application memory, you push it to Redis, a database, or the client token itself. The application server becomes a pure computation layer that reads state in, processes the request, and writes state out.

    ```
    Stateful (fragile):              Stateless (scalable):

    Server A [session data]          Server A [no state]---+
    Server B [session data]          Server B [no state]---+---> Redis
    Server C [session data]          Server C [no state]---+
         |                                |
    If Server A dies,                If Server A dies,
    its sessions are lost            Redis still has all sessions
    ```

    Cookie-based sessions are the simplest form of stateless design for web applications. The server serializes session data, signs it cryptographically, and sets it as a cookie. On each request the browser sends the cookie back, and the server verifies the signature and reads the data without any external lookup. This approach works well when session data is small (under 4 KB) and does not contain secrets, since the data is visible to the client even though it cannot be tampered with.

    Netflix operates thousands of stateless microservices that collectively serve over 260 million subscribers. Each service validates incoming JWTs independently using a shared public key, with no centralized session store. When a user logs in, the authentication service issues a signed token, and every downstream service (recommendations, playback, billing) trusts that token without calling back to the auth service. This design allows Netflix to deploy, scale, and restart individual services without any impact on user sessions.

=== "Security"

    Session security failures can give an attacker full access to a user's account. The most common attacks target the session identifier itself or the transport mechanism that carries it.

    **Session fixation** occurs when an attacker sets a known session ID in a victim's browser before the victim logs in. The victim authenticates, and the server associates the attacker's chosen session ID with the now-authenticated account. The attacker then uses that same session ID to access the account. The defense is straightforward: regenerate the session ID immediately after successful authentication. The old ID becomes meaningless, and only the new ID (which only the legitimate user's browser holds) maps to the authenticated session.

    **Session hijacking** happens when an attacker intercepts or steals a valid session ID. Cookie security flags are the primary defense layer. Setting these flags correctly ensures that session cookies travel only over encrypted channels and cannot be read by client-side scripts.

    | Cookie Flag | Value | What It Prevents |
    | --- | --- | --- |
    | `Secure` | Required | Transmission over unencrypted HTTP |
    | `HttpOnly` | Required | JavaScript access via `document.cookie` (blocks XSS theft) |
    | `SameSite` | `Lax` or `Strict` | Cross-site request forgery (CSRF) |
    | `Domain` | Your domain only | Cookie leaking to sibling subdomains |
    | `Path` | `/` or specific path | Cookie sent to unrelated application paths |
    | `Max-Age` | Seconds | Indefinite session persistence (set reasonable expiry) |

    A properly secured session cookie looks like this:

    ```
    Set-Cookie: sid=a3f9c8...; Secure; HttpOnly; SameSite=Lax; Path=/; Max-Age=3600
    ```

    **CSRF protection** through the `SameSite` attribute has simplified what once required anti-CSRF tokens. With `SameSite=Lax`, the browser will not send the session cookie on cross-origin POST requests, which blocks the most common CSRF attack vector. `SameSite=Strict` goes further by blocking the cookie on all cross-origin navigations, though this can interfere with legitimate flows like clicking a link from an email.

    **Session timeout strategies** balance security with usability. Idle timeouts expire the session after a period of inactivity (commonly 15 to 30 minutes for sensitive applications, 30 to 60 minutes for general use). Absolute timeouts set a hard upper bound on session lifetime regardless of activity (typically 8 to 24 hours), forcing re-authentication even for continuously active users. Banking applications often combine a 15-minute idle timeout with an 8-hour absolute timeout, while social media platforms may allow sessions to persist for days with rolling expiration that resets on each visit.

## Key Takeaways

1. Server-side sessions (Redis or database) give you full control over revocation and data size, but require a shared store that every application server can reach. Redis is the dominant choice because its sub-millisecond reads match the latency budget of session lookups.

2. JWTs eliminate server-side storage and enable independent validation across microservices, but they cannot be revoked before expiry without maintaining a server-side blocklist, which partially negates their stateless advantage.

3. Stateless architecture scales horizontally by extracting all session state to external stores or client tokens. The application server becomes interchangeable, which simplifies deployment, auto-scaling, and failure recovery.

4. Session ID generation must use cryptographically secure randomness with at least 128 bits of entropy. Predictable session IDs are the root cause of session hijacking attacks.

5. Cookie security flags (Secure, HttpOnly, SameSite) are non-negotiable for any session cookie. Each flag closes a specific attack vector, and omitting any one of them leaves a gap.

6. The hybrid approach, short-lived JWTs for authentication combined with server-side refresh token storage for revocation, is the pattern used by most large-scale identity providers because it captures the scalability benefits of stateless tokens while retaining the ability to cut off compromised sessions.

## Related Topics

- [Authentication Patterns](../../security/authentication.md) -- how identity is verified before a session begins
- [Common Attacks](../../security/common-attacks.md) -- XSS, CSRF, and other threats that target sessions
- [Caching](../../data/caching/index.md) -- Redis and Memcached patterns used for session storage
