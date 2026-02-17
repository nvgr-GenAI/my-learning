# Authentication

Authentication answers the question: **who are you?** Before a system can decide what you're allowed to do, it must verify your identity. This seems simple — enter a username and password, and you're in — but the challenge is maintaining that identity across subsequent requests without asking users to log in every time they click a link.

The fundamental tension in authentication design is between security and convenience. More secure methods add friction; more convenient methods expand the attack surface. Every authentication system navigates this trade-off.

---

## How Authentication Works Across Requests

HTTP is stateless — each request is independent, with no memory of previous ones. Authentication systems must bolt identity onto this stateless protocol. There are two fundamentally different approaches.

```
Session-Based (stateful):

Client ──→ Login ──→ Server creates session, stores in Redis
  │                        │
  │←── Set-Cookie: sid=abc123
  │
  │──→ GET /profile (Cookie: sid=abc123)
  │                        │
  │                   Server looks up sid=abc123 in Redis
  │                   → finds {userId: 42, role: "admin"}
  │←── 200 OK


Token-Based (stateless):

Client ──→ Login ──→ Server creates signed JWT
  │                        │
  │←── {token: "eyJhbG..."}
  │
  │──→ GET /profile (Authorization: Bearer eyJhbG...)
  │                        │
  │                   Server verifies JWT signature
  │                   → extracts {userId: 42, role: "admin"}
  │←── 200 OK
```

The key difference: with sessions, the server **remembers** who you are. With tokens, the server **recalculates** who you are from the token itself.

---

=== "Session-Based"

    ## Session-Based Authentication

    The server stores session data (user ID, role, preferences) in a session store — typically Redis or a database — and gives the client a session ID in a cookie. Each subsequent request includes this cookie, and the server looks up the session.

    **How it works:**
    1. User submits credentials
    2. Server validates against stored password hash (bcrypt, argon2)
    3. Server creates a session object in Redis with a random ID
    4. Session ID sent to client as an httpOnly, secure cookie
    5. Client automatically includes cookie on every request
    6. Server looks up session by ID to identify the user

    **Strengths:**

    - **Instant revocation.** Delete the session from Redis and the user is immediately logged out. No waiting for token expiration.
    - **Small cookie size.** The cookie contains only a session ID (32-64 bytes), not the actual user data.
    - **Server control.** The server can track active sessions, force logout across all devices, and implement "log out everywhere."

    **Weaknesses:**

    - **Scaling requires shared state.** With multiple servers, they all need access to the same session store. This is why Redis is almost universal — **Instagram**, **GitHub**, and **Shopify** all use Redis for session storage.
    - **CSRF vulnerability.** Browsers automatically attach cookies to requests, so a malicious site can trigger authenticated requests on behalf of the user. Mitigation: CSRF tokens.
    - **Not ideal for mobile/API.** Cookie-based auth works naturally in browsers but is awkward for mobile apps and third-party API consumers.

=== "Token-Based (JWT)"

    ## Token-Based Authentication (JWT)

    Instead of storing session data on the server, the server encodes user information into a signed token that the client stores and sends with each request. The server verifies the token's signature to confirm it hasn't been tampered with.

    A JWT (JSON Web Token) has three parts:

    ```
    header.payload.signature

    Header:  {"alg": "HS256", "typ": "JWT"}
    Payload: {"userId": 42, "role": "admin", "exp": 1735689600}
    Signature: HMAC-SHA256(header + "." + payload, secret_key)
    ```

    The payload is **not encrypted** — anyone can decode it. The signature only proves the server created it and nobody modified it.

    **Strengths:**

    - **Stateless.** No server-side storage needed. Any server with the signing key can verify the token. This is why JWTs are the default for microservices architectures.
    - **Cross-domain.** Tokens can be sent to any domain in an Authorization header, unlike cookies which are domain-scoped.
    - **Scalable.** No shared session store means no scaling bottleneck. **Uber**, **Spotify**, and most modern APIs use JWTs.

    **Weaknesses:**

    - **No instant revocation.** Once issued, a JWT is valid until it expires. If a user's account is compromised, you can't invalidate the stolen token without maintaining a blocklist — which defeats the stateless benefit.
    - **Token size.** A JWT with typical claims is 500-1000 bytes, sent with every request.
    - **Storage risk.** If stored in localStorage, any XSS attack can steal the token.

    ### The Refresh Token Pattern

    Short-lived access tokens (15 minutes) paired with long-lived refresh tokens (7 days) balance security and convenience:

    ```
    Login → receive access token (15min) + refresh token (7 days)

    Normal requests: send access token in Authorization header

    When access token expires:
      Client ──→ POST /refresh (refresh token)
      Server ──→ verify refresh token in database
      Server ──→ issue new access token
      Client ──→ continue with new access token
    ```

    The refresh token is stored server-side (in a database), so it can be revoked. The access token remains stateless for performance. This is the pattern used by **Google**, **Auth0**, and most production JWT implementations.

=== "OAuth 2.0"

    ## OAuth 2.0

    OAuth 2.0 is a **delegated authorization** protocol — it lets users grant third-party applications limited access to their accounts on another service, without sharing their password. "Login with Google" is the most visible example, but OAuth also powers API integrations between services.

    ### Authorization Code Flow

    The most secure OAuth flow, used by server-side applications:

    ```
    1. User clicks "Login with Google"
                │
    2. App redirects to Google's authorization page
                │
    3. User logs in to Google and approves requested scopes
                │
    4. Google redirects back to app with authorization code
                │
    5. App's backend exchanges code for access token (server-to-server)
                │
    6. App uses access token to fetch user profile from Google
                │
    7. App creates its own session/JWT for the user
    ```

    The critical security feature: the access token is exchanged server-to-server in step 5, never exposed to the browser. The browser only sees the authorization code, which is single-use and short-lived.

    ### OAuth Roles

    | Role | Who | Example |
    |---|---|---|
    | **Resource Owner** | The user | You, the person logging in |
    | **Client** | The application requesting access | A third-party app |
    | **Authorization Server** | Issues tokens after authentication | Google's auth server |
    | **Resource Server** | Hosts the protected data | Google's user info API |

    ### When to Use OAuth

    - **Social login.** Let users sign in with existing Google/GitHub/Apple accounts. Reduces friction (no new password to create) and shifts password security to providers who invest heavily in it.
    - **Third-party API access.** Let users grant your app access to their data on other platforms — GitHub repos, Google Calendar, Stripe accounts.
    - **Not for internal authentication.** If you control both the frontend and backend, OAuth adds unnecessary complexity. Use sessions or JWTs directly.

=== "SSO"

    ## SSO (Single Sign-On)

    SSO allows users to log in once and access multiple applications without re-authenticating. This is an enterprise staple — employees at large companies access dozens of internal tools (email, HR portal, project management, code repositories) with a single login.

    ```
    User → SSO Provider (Okta, Auth0, Azure AD)
                  │
             Login once
                  │
             ┌────┼────┬────────┐
             │    │    │        │
             ▼    ▼    ▼        ▼
           Email  HR  Jira    GitHub
           (no additional login needed)
    ```

    ### SAML Flow

    SAML (Security Assertion Markup Language) is the dominant SSO protocol in enterprise environments:

    1. User accesses App A
    2. App A redirects to the SSO identity provider (IdP)
    3. User authenticates at the IdP (if not already authenticated)
    4. IdP generates a SAML assertion (signed XML document with user identity and attributes)
    5. User is redirected back to App A with the assertion
    6. App A validates the assertion's signature and logs the user in
    7. When user accesses App B, the IdP sees the existing session and immediately issues an assertion — no second login

    **Okta** processes over 1 billion authentications per month. **Microsoft Azure AD** provides SSO for over 500,000 organizations. **Google Workspace** uses SSO to connect Gmail, Drive, Calendar, and dozens of other services.

---

## Choosing an Authentication Method

| Scenario | Best Choice | Why |
|---|---|---|
| Traditional web app, single server | **Sessions** | Simple, full server control |
| API consumed by mobile + web + third parties | **JWT** | Stateless, cross-platform |
| "Login with Google/GitHub" needed | **OAuth 2.0** | Delegated auth, no password management |
| Enterprise with many internal apps | **SSO (SAML/OIDC)** | One login for everything |
| Microservices architecture | **JWT + OAuth** | Stateless verification across services |

Most production systems combine methods. A common pattern: OAuth for initial login, JWT for API authentication, sessions for the web frontend.

---

## Security Best Practices

**Password storage.** Never store plaintext passwords. Use bcrypt or argon2 with a cost factor that takes ~250ms to hash. This makes brute-force attacks computationally expensive.

**Multi-factor authentication (MFA).** Something you know (password) + something you have (phone/hardware key). **Duo** processes billions of MFA requests per year. Google found that security keys (like YubiKey) prevent 100% of phishing attacks, compared to 76% for SMS-based MFA.

**Token storage.** Store JWTs in httpOnly cookies (immune to XSS) rather than localStorage. If you must use localStorage, keep access tokens short-lived (15 minutes) and refresh tokens in httpOnly cookies.

**Rate limiting on auth endpoints.** Login endpoints should have strict rate limits — 5-10 attempts per 15 minutes per IP — to prevent brute-force attacks. See [API Security](api-security.md).

---

## Key Takeaways

1. **Sessions for simplicity, JWTs for scalability.** Use sessions when you need server-side control and instant revocation. Use JWTs when you need stateless authentication across multiple services.

2. **OAuth is for delegation, not internal auth.** Use OAuth when you need "Login with X" or third-party API access. Use sessions or JWTs for your own authentication.

3. **SSO is essential at enterprise scale.** When users access more than 3-4 applications, SSO pays for itself in reduced password fatigue and centralized access control.

4. **Combine the refresh token pattern with JWTs.** Short-lived access tokens (15min) + revocable refresh tokens (7 days) give you both stateless performance and the ability to revoke access.

5. **MFA is the single most impactful security improvement.** It stops the vast majority of account takeover attacks, regardless of which authentication method you use.

---

## Related Topics

- **[Authorization](authorization.md)** — what authenticated users are allowed to do
- **[API Security](api-security.md)** — rate limiting, API keys, and input validation
- **[Encryption](encryption.md)** — protecting data in transit and at rest
- **[Common Attacks](common-attacks.md)** — XSS, CSRF, and how they target authentication
