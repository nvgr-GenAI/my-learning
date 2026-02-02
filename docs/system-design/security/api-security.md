# API Security

**Protect your APIs** | 🚦 Rate Limiting | 🔑 API Keys | 🛡️ Validation

---

## Rate Limiting

=== "Why Rate Limit?"
    - Prevent abuse (DDoS attacks)
    - Ensure fair usage
    - Control costs (paid APIs)
    - Protect backend resources

=== "Implementation"
    ```javascript
    const rateLimit = require('express-rate-limit');
    const RedisStore = require('rate-limit-redis');
    const redis = require('redis');

    const client = redis.createClient();

    // Configure rate limiter
    const limiter = rateLimit({
        store: new RedisStore({ client }),
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 100, // Limit each IP to 100 requests per windowMs
        message: 'Too many requests, please try again later.',
        standardHeaders: true,
        legacyHeaders: false
    });

    // Apply to all routes
    app.use('/api/', limiter);

    // Stricter limit for login
    const strictLimiter = rateLimit({
        windowMs: 15 * 60 * 1000,
        max: 5, // Only 5 login attempts per 15 minutes
        skipSuccessfulRequests: true
    });

    app.post('/api/login', strictLimiter, loginHandler);
    ```

---

## API Keys

```javascript
// Generate API key
const crypto = require('crypto');

function generateAPIKey() {
    return crypto.randomBytes(32).toString('hex');
}

// Middleware to validate API key
function validateAPIKey(req, res, next) {
    const apiKey = req.headers['x-api-key'];
    
    if (!apiKey) {
        return res.status(401).json({ error: 'API key required' });
    }

    const valid = await db.apiKeys.findOne({
        key: apiKey,
        active: true
    });

    if (!valid) {
        return res.status(403).json({ error: 'Invalid API key' });
    }

    req.apiKey = valid;
    next();
}

app.use('/api/', validateAPIKey);
```

---

**Security is everyone's responsibility! 🛡️**
