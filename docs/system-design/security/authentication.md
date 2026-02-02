# Authentication

**Verify user identity** | 🔑 Session | 🎟️ Token | 🔐 OAuth | 🌐 SSO

---

## Overview

Authentication is the process of verifying that users are who they claim to be. It's the first line of defense in application security.

**Key Question:** How do we securely identify users across requests?

---

## Authentication Methods Comparison

| Method | Storage | Scalability | Security | Use Case |
|--------|---------|-------------|----------|----------|
| **Session-based** | Server memory/DB | Medium | Good | Traditional web apps |
| **Token-based (JWT)** | Client-side | Excellent | Good | APIs, microservices |
| **OAuth 2.0** | Third-party | Excellent | Excellent | Social login |
| **SSO** | Centralized | Excellent | Excellent | Enterprise apps |

---

## Session-Based Authentication

=== "How It Works"
    **Server stores session state**

    ```
    1. User logs in with credentials
       ↓
    2. Server validates credentials
       ↓
    3. Server creates session, stores in memory/Redis
       ↓
    4. Server sends session ID in cookie
       ↓
    5. Client includes cookie in subsequent requests
       ↓
    6. Server looks up session to verify user
    ```

=== "Implementation"
    ```javascript
    const express = require('express');
    const session = require('express-session');
    const RedisStore = require('connect-redis')(session);
    const redis = require('redis');

    const app = express();
    const redisClient = redis.createClient();

    // Configure session middleware
    app.use(session({
        store: new RedisStore({ client: redisClient }),
        secret: 'your-secret-key',
        resave: false,
        saveUninitialized: false,
        cookie: {
            secure: true,      // HTTPS only
            httpOnly: true,    // No JavaScript access
            maxAge: 24 * 60 * 60 * 1000  // 24 hours
        }
    }));

    // Login endpoint
    app.post('/login', async (req, res) => {
        const { username, password } = req.body;
        
        // Validate credentials
        const user = await db.users.findOne({ username });
        if (!user || !await bcrypt.compare(password, user.passwordHash)) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        // Create session
        req.session.userId = user.id;
        req.session.username = user.username;
        
        res.json({ message: 'Logged in successfully' });
    });

    // Protected route
    app.get('/profile', (req, res) => {
        if (!req.session.userId) {
            return res.status(401).json({ error: 'Not authenticated' });
        }
        
        res.json({
            userId: req.session.userId,
            username: req.session.username
        });
    });

    // Logout
    app.post('/logout', (req, res) => {
        req.session.destroy((err) => {
            if (err) {
                return res.status(500).json({ error: 'Logout failed' });
            }
            res.clearCookie('connect.sid');
            res.json({ message: 'Logged out successfully' });
        });
    });
    ```

=== "Advantages"
    - ✅ Server has full control (can revoke sessions)
    - ✅ Smaller cookie size (just session ID)
    - ✅ Easy to implement
    - ✅ Stateful (server knows active users)

=== "Disadvantages"
    - ❌ Scaling challenges (session stored on server)
    - ❌ Requires shared session store (Redis) for multiple servers
    - ❌ CSRF vulnerability (needs CSRF tokens)
    - ❌ Not ideal for APIs (stateful)

---

## Token-Based (JWT)

=== "How It Works"
    **Stateless authentication with signed tokens**

    ```
    1. User logs in with credentials
       ↓
    2. Server validates credentials
       ↓
    3. Server creates JWT with user claims
       ↓
    4. Server signs JWT with secret key
       ↓
    5. Client stores JWT (localStorage/cookie)
       ↓
    6. Client includes JWT in Authorization header
       ↓
    7. Server verifies signature and extracts claims
    ```

    **JWT Structure:**
    ```
    header.payload.signature

    Header:
    {
        "alg": "HS256",
        "typ": "JWT"
    }

    Payload:
    {
        "userId": 123,
        "username": "alice",
        "exp": 1735689600  // Expiration time
    }

    Signature:
    HMACSHA256(
        base64UrlEncode(header) + "." + base64UrlEncode(payload),
        secret
    )
    ```

=== "Implementation"
    ```javascript
    const jwt = require('jsonwebtoken');
    const bcrypt = require('bcrypt');

    const SECRET_KEY = process.env.JWT_SECRET;

    // Login endpoint
    app.post('/login', async (req, res) => {
        const { username, password } = req.body;
        
        // Validate credentials
        const user = await db.users.findOne({ username });
        if (!user || !await bcrypt.compare(password, user.passwordHash)) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        // Create JWT
        const token = jwt.sign(
            {
                userId: user.id,
                username: user.username,
                role: user.role
            },
            SECRET_KEY,
            { expiresIn: '24h' }
        );
        
        res.json({ token });
    });

    // Authentication middleware
    function authenticateToken(req, res, next) {
        const authHeader = req.headers['authorization'];
        const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

        if (!token) {
            return res.status(401).json({ error: 'No token provided' });
        }

        jwt.verify(token, SECRET_KEY, (err, user) => {
            if (err) {
                return res.status(403).json({ error: 'Invalid token' });
            }
            req.user = user;
            next();
        });
    }

    // Protected route
    app.get('/profile', authenticateToken, (req, res) => {
        res.json({
            userId: req.user.userId,
            username: req.user.username,
            role: req.user.role
        });
    });

    // Refresh token
    app.post('/refresh', authenticateToken, (req, res) => {
        const newToken = jwt.sign(
            {
                userId: req.user.userId,
                username: req.user.username,
                role: req.user.role
            },
            SECRET_KEY,
            { expiresIn: '24h' }
        );
        
        res.json({ token: newToken });
    });
    ```

=== "Refresh Tokens"
    **Handle token expiration gracefully:**

    ```javascript
    // Two token strategy
    function generateTokens(user) {
        // Short-lived access token (15 minutes)
        const accessToken = jwt.sign(
            { userId: user.id, username: user.username },
            ACCESS_SECRET,
            { expiresIn: '15m' }
        );

        // Long-lived refresh token (7 days)
        const refreshToken = jwt.sign(
            { userId: user.id },
            REFRESH_SECRET,
            { expiresIn: '7d' }
        );

        // Store refresh token in database
        await db.refreshTokens.create({
            userId: user.id,
            token: refreshToken,
            expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
        });

        return { accessToken, refreshToken };
    }

    // Refresh endpoint
    app.post('/refresh', async (req, res) => {
        const { refreshToken } = req.body;

        // Verify refresh token
        let payload;
        try {
            payload = jwt.verify(refreshToken, REFRESH_SECRET);
        } catch (err) {
            return res.status(401).json({ error: 'Invalid refresh token' });
        }

        // Check if token exists in database
        const tokenExists = await db.refreshTokens.findOne({
            userId: payload.userId,
            token: refreshToken
        });

        if (!tokenExists) {
            return res.status(401).json({ error: 'Token revoked' });
        }

        // Generate new access token
        const user = await db.users.findById(payload.userId);
        const accessToken = jwt.sign(
            { userId: user.id, username: user.username },
            ACCESS_SECRET,
            { expiresIn: '15m' }
        );

        res.json({ accessToken });
    });
    ```

=== "Advantages"
    - ✅ Stateless (no server-side storage)
    - ✅ Perfect for APIs and microservices
    - ✅ Works across domains
    - ✅ Scalable (no shared session store needed)

=== "Disadvantages"
    - ❌ Larger payload (token can be 500+ bytes)
    - ❌ Can't revoke tokens easily (until expiration)
    - ❌ Token stored in localStorage (XSS risk)
    - ❌ Need refresh token strategy

---

## OAuth 2.0

=== "What is OAuth?"
    **Delegated authorization protocol**

    **Use Case:** "Login with Google/Facebook/GitHub"

    ```
    Flow:
    1. User clicks "Login with Google"
       ↓
    2. Redirect to Google's authorization page
       ↓
    3. User approves access
       ↓
    4. Google redirects back with authorization code
       ↓
    5. Exchange code for access token
       ↓
    6. Use token to access user's Google data
    ```

=== "Implementation"
    ```javascript
    const passport = require('passport');
    const GoogleStrategy = require('passport-google-oauth20').Strategy;

    // Configure Google OAuth
    passport.use(new GoogleStrategy({
        clientID: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        callbackURL: 'http://localhost:3000/auth/google/callback'
    },
    async (accessToken, refreshToken, profile, done) => {
        // Find or create user
        let user = await db.users.findOne({ googleId: profile.id });
        
        if (!user) {
            user = await db.users.create({
                googleId: profile.id,
                email: profile.emails[0].value,
                name: profile.displayName,
                avatar: profile.photos[0].value
            });
        }
        
        return done(null, user);
    }));

    // Login route
    app.get('/auth/google',
        passport.authenticate('google', {
            scope: ['profile', 'email']
        })
    );

    // Callback route
    app.get('/auth/google/callback',
        passport.authenticate('google', { failureRedirect: '/login' }),
        (req, res) => {
            // Create JWT for our app
            const token = jwt.sign(
                { userId: req.user.id },
                SECRET_KEY,
                { expiresIn: '24h' }
            );
            res.redirect(`/dashboard?token=${token}`);
        }
    );
    ```

=== "Advantages"
    - ✅ No password management (delegated to provider)
    - ✅ Better UX (one-click login)
    - ✅ Access to user's data (with permission)
    - ✅ Trusted providers (Google, GitHub, etc.)

=== "Disadvantages"
    - ❌ Dependency on third-party
    - ❌ Privacy concerns (data sharing)
    - ❌ Complex implementation
    - ❌ Provider outage affects your app

---

## SSO (Single Sign-On)

=== "What is SSO?"
    **One login for multiple applications**

    ```
    Enterprise Scenario:
    Login once → Access all apps

    User → SSO Provider (Okta/Auth0) → App1, App2, App3
    ```

=== "SAML Flow"
    ```
    1. User accesses App1
       ↓
    2. App1 redirects to SSO provider
       ↓
    3. User logs in at SSO provider (once)
       ↓
    4. SSO provider generates SAML assertion
       ↓
    5. User redirected back to App1 with assertion
       ↓
    6. App1 validates assertion and logs user in
       ↓
    7. User accesses App2
       ↓
    8. App2 redirects to SSO provider
       ↓
    9. SSO provider sees existing session
       ↓
    10. Immediately redirects back to App2 (no login needed!)
    ```

=== "Advantages"
    - ✅ Single login for all apps
    - ✅ Centralized user management
    - ✅ Better security (one strong password)
    - ✅ Improved UX

=== "Disadvantages"
    - ❌ Complex to set up
    - ❌ Single point of failure
    - ❌ Vendor lock-in

---

## Best Practices

### Password Security
```javascript
const bcrypt = require('bcrypt');

// Hash password (during registration)
const saltRounds = 10;
const passwordHash = await bcrypt.hash(password, saltRounds);

// Verify password (during login)
const isValid = await bcrypt.compare(password, user.passwordHash);

// Password requirements
function validatePassword(password) {
    return password.length >= 12 &&
           /[a-z]/.test(password) &&
           /[A-Z]/.test(password) &&
           /[0-9]/.test(password) &&
           /[^a-zA-Z0-9]/.test(password);
}
```

### Multi-Factor Authentication (MFA)
```javascript
const speakeasy = require('speakeasy');

// Generate secret for user
const secret = speakeasy.generateSecret({ name: 'MyApp' });

// User scans QR code and enters first code
const verified = speakeasy.totp.verify({
    secret: secret.base32,
    encoding: 'base32',
    token: userEnteredCode
});

if (verified) {
    // Save secret to user's account
    await db.users.update(userId, { mfaSecret: secret.base32 });
}

// During login
const isValid = speakeasy.totp.verify({
    secret: user.mfaSecret,
    encoding: 'base32',
    token: req.body.mfaCode
});
```

---

## Interview Talking Points

**Q: Session-based vs Token-based authentication - when to use each?**

✅ **Strong Answer:**
> "I'd use session-based authentication for traditional server-rendered web apps where the same server handles both rendering and API requests. It's simpler to implement and gives the server full control to revoke sessions. However, for APIs, mobile apps, or microservices, I'd use token-based (JWT) authentication because it's stateless and scales better - you don't need a shared session store like Redis. The trade-off is you can't easily revoke JWTs before they expire, so I'd implement a refresh token strategy with short-lived access tokens (15 minutes) and longer refresh tokens stored in the database that can be revoked."

**Q: How do you securely store JWT tokens on the client?**

✅ **Strong Answer:**
> "The most secure approach is to store JWTs in httpOnly cookies, which prevents XSS attacks since JavaScript can't access them. However, this requires CSRF protection. Alternatively, storing in localStorage is convenient but vulnerable to XSS - any malicious script can steal the token. A hybrid approach I use is: store short-lived access tokens in memory (lost on refresh), and refresh tokens in httpOnly cookies. On page load, use the refresh token to get a new access token. This limits the XSS exposure window while maintaining UX."

---

## Related Topics

- [Authorization](authorization.md) - Access control
- [API Security](api-security.md) - Protect APIs
- [Encryption](encryption.md) - Secure data transmission
- [Common Attacks](common-attacks.md) - Security vulnerabilities

---

**Authentication is the foundation of security! 🔑**
