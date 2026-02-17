# API Security

APIs are the primary attack surface of modern applications. Every public endpoint is a door that attackers can probe — testing for authentication bypasses, injection vulnerabilities, and resource exhaustion. API security is about controlling who can access your API, how often, and ensuring that malicious input never reaches your application logic.

---

=== "Rate Limiting"

    ## Rate Limiting

    Without rate limiting, a single client can overwhelm your API — whether through a deliberate DDoS attack, a buggy script in a retry loop, or a competitor scraping your data. Rate limiting caps how many requests a client can make in a given time window.

    ### Common Strategies

    ```
    Fixed Window:
      100 requests per 15 minutes
      Counter resets at the window boundary

      Problem: 100 requests at 14:59 + 100 at 15:00 = 200 in 1 minute

    Sliding Window:
      100 requests in any rolling 15-minute period
      Smoother enforcement, no burst at window boundaries

    Token Bucket:
      Bucket holds 100 tokens, refills at 10/second
      Each request costs 1 token
      Allows short bursts while enforcing average rate
    ```

    | Algorithm | Burst Handling | Memory | Best For |
    |---|---|---|---|
    | **Fixed Window** | Poor (boundary bursts) | Low | Simple APIs |
    | **Sliding Window** | Good (smooth) | Medium | Most production APIs |
    | **Token Bucket** | Excellent (controlled bursts) | Low | APIs needing burst tolerance |

    ### What to Limit

    Different endpoints need different limits. Login endpoints need strict limits (5-10 per 15 minutes) to prevent brute-force attacks. Read endpoints can be more generous (1000 per minute). Write endpoints sit in between.

    **GitHub's API** allows 5,000 authenticated requests per hour, 60 unauthenticated. **Stripe's API** defaults to 100 requests per second with higher limits for specific endpoints. **Twitter's API** varies by endpoint — 300 tweets/3 hours for posting, 900 requests/15 minutes for reading.

    Rate limit responses should include headers telling the client their status:

    ```
    HTTP/1.1 429 Too Many Requests
    X-RateLimit-Limit: 100
    X-RateLimit-Remaining: 0
    X-RateLimit-Reset: 1635724800
    Retry-After: 30
    ```

=== "API Authentication"

    ## API Authentication Methods

    ### API Keys

    The simplest form of API authentication — a unique string that identifies the caller. API keys are sent in a header (`X-API-Key: abc123`) or as a query parameter.

    **API keys are for identification, not authentication.** They identify which application is making the request and enable per-client rate limiting and usage tracking. But they're not user-specific — an API key doesn't tell you which user within an organization is making the call.

    **When to use:** Machine-to-machine communication, third-party developer access, public API usage tracking.

    **When not to use:** User authentication (use OAuth or JWTs), anything where you need to know which human initiated the request.

    ### OAuth 2.0 for APIs

    For APIs that act on behalf of users, OAuth tokens provide scoped, user-specific, revocable access. See [Authentication](authentication.md) for the full OAuth flow.

    The key advantage over API keys: OAuth tokens carry **scopes** that limit what the token holder can do. A token with `read:repos` scope can list repositories but not delete them, even if the underlying user has delete permissions.

=== "Input Validation & Security"

    ## Input Validation

    Every piece of data from a client is untrusted. Validate at the API boundary before it touches your business logic or database.

    ### Defense in Depth

    ```
    Client Request
         │
         ▼
    ┌─────────────────────┐
    │  API Gateway         │ ← Rate limiting, authentication
    ├─────────────────────┤
    │  Schema Validation   │ ← Request body matches expected shape
    ├─────────────────────┤
    │  Input Sanitization  │ ← Strip/escape dangerous characters
    ├─────────────────────┤
    │  Business Logic      │ ← Application-level authorization
    ├─────────────────────┤
    │  Parameterized Query │ ← Database interaction (never concat input into SQL)
    └─────────────────────┘
    ```

    **Key principles:**

    - **Validate types and ranges.** An age field should accept integers 0-150, not arbitrary strings. An email should match a known pattern. A quantity shouldn't be negative.
    - **Whitelist, don't blacklist.** Define what's allowed, not what's blocked. Blacklists always have gaps.
    - **Validate at the server, not the client.** Client validation is for UX. Server validation is for security. Attackers bypass client validation trivially.

    ## CORS (Cross-Origin Resource Sharing)

    Browsers block JavaScript from making requests to a different domain than the page's origin — the **same-origin policy**. CORS is the mechanism that allows controlled exceptions.

    ```
    User visits evil.com
      │
      evil.com JavaScript tries to call api.yoursite.com
      │
      Browser checks: does api.yoursite.com allow requests from evil.com?
      │
      ├─ CORS headers say yes → request allowed
      └─ No CORS headers / not listed → request blocked
    ```

    **Configuration matters.** Setting `Access-Control-Allow-Origin: *` defeats the purpose — it allows any site to make requests to your API. Specify exact allowed origins: `Access-Control-Allow-Origin: https://yourfrontend.com`.

    ## HTTPS Enforcement

    All API traffic must use HTTPS. Enforce this at multiple levels:

    - **Redirect HTTP to HTTPS** at the load balancer or reverse proxy
    - **HSTS header** (`Strict-Transport-Security`) tells browsers to always use HTTPS, even if the user types `http://`
    - **Reject non-HTTPS requests** at the API level for sensitive endpoints

---

## API Security Checklist

| Category | Practice |
|---|---|
| **Authentication** | Require authentication on all non-public endpoints |
| **Rate limiting** | Enforce per-client limits, stricter on auth endpoints |
| **Input validation** | Validate all input server-side, use parameterized queries |
| **HTTPS** | Enforce TLS on all endpoints, use HSTS |
| **CORS** | Whitelist specific origins, never use wildcard for sensitive APIs |
| **Error responses** | Don't leak internal details (stack traces, SQL errors) |
| **Logging** | Log authentication failures, rate limit triggers, unusual patterns |
| **Versioning** | Deprecate insecure API versions, don't maintain them indefinitely |

---

## Key Takeaways

1. **Rate limiting is your first line of defense.** It protects against brute-force attacks, scraping, and accidental overload. Use sliding window or token bucket for most APIs.

2. **API keys identify, OAuth authenticates.** Use API keys for tracking and rate limiting. Use OAuth tokens when you need user-level permissions and scoping.

3. **Validate everything at the server boundary.** Never trust client input. Validate types, ranges, and formats before the data reaches business logic.

4. **HTTPS is non-negotiable.** All API traffic must be encrypted. No exceptions for "internal" services — use mTLS between services.

5. **Error messages should be helpful to legitimate users and useless to attackers.** Return "Invalid credentials" rather than "User not found" or "Wrong password."

---

## Related Topics

- **[Authentication](authentication.md)** — JWT, OAuth, and session-based API authentication
- **[Encryption](encryption.md)** — TLS and data protection
- **[Common Attacks](common-attacks.md)** — SQL injection, XSS, and other attacks targeting APIs
- **[Load Balancers](../networking/load-balancers.md)** — rate limiting and traffic management at the infrastructure level
