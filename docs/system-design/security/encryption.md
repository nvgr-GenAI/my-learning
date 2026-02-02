# Encryption

**Protect data in transit and at rest** | 🔐 TLS | 💾 At-Rest | 🔑 Keys

---

## Overview

Encryption transforms readable data into unreadable ciphertext using cryptographic algorithms. Only those with the decryption key can read it.

**Two main types:**
- **Encryption in transit:** Protect data moving over networks (TLS/SSL)
- **Encryption at rest:** Protect stored data (database, files)

---

## TLS/SSL (Transport Layer Security)

=== "How HTTPS Works"
    ```
    1. Client → Server: Hello (supported ciphers)
    2. Server → Client: Certificate + Public Key
    3. Client verifies certificate (trusted CA?)
    4. Client generates session key
    5. Client encrypts session key with server's public key
    6. Server decrypts with private key
    7. Both use session key for symmetric encryption
    
    Result: Encrypted communication channel
    ```

=== "Implementation"
    ```javascript
    // Node.js HTTPS server
    const https = require('https');
    const fs = require('fs');

    const options = {
        key: fs.readFileSync('/path/to/private-key.pem'),
        cert: fs.readFileSync('/path/to/certificate.pem'),
        // Modern TLS configuration
        minVersion: 'TLSv1.2',
        ciphers: [
            'ECDHE-RSA-AES128-GCM-SHA256',
            'ECDHE-RSA-AES256-GCM-SHA384'
        ].join(':')
    };

    https.createServer(options, (req, res) => {
        res.writeHead(200);
        res.end('Secure connection!');
    }).listen(443);
    ```

=== "Best Practices"
    - ✅ Use TLS 1.2 or 1.3 (disable older versions)
    - ✅ Strong cipher suites only
    - ✅ HSTS header (force HTTPS)
    - ✅ Certificate from trusted CA
    - ✅ Auto-renewal (Let's Encrypt)

---

## Encryption At Rest

=== "Database Encryption"
    ```javascript
    const crypto = require('crypto');

    // Encryption key (store in env variable)
    const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY; // 32 bytes
    const IV_LENGTH = 16;

    function encrypt(text) {
        const iv = crypto.randomBytes(IV_LENGTH);
        const cipher = crypto.createCipheriv(
            'aes-256-cbc',
            Buffer.from(ENCRYPTION_KEY),
            iv
        );
        
        let encrypted = cipher.update(text);
        encrypted = Buffer.concat([encrypted, cipher.final()]);
        
        return iv.toString('hex') + ':' + encrypted.toString('hex');
    }

    function decrypt(text) {
        const parts = text.split(':');
        const iv = Buffer.from(parts.shift(), 'hex');
        const encryptedText = Buffer.from(parts.join(':'), 'hex');
        
        const decipher = crypto.createDecipheriv(
            'aes-256-cbc',
            Buffer.from(ENCRYPTION_KEY),
            iv
        );
        
        let decrypted = decipher.update(encryptedText);
        decrypted = Buffer.concat([decrypted, decipher.final()]);
        
        return decrypted.toString();
    }

    // Usage
    const sensitiveData = 'SSN: 123-45-6789';
    const encrypted = encrypt(sensitiveData);
    // Store encrypted in database

    const decrypted = decrypt(encrypted);
    // Use decrypted data
    ```

=== "Key Management"
    ```javascript
    // AWS KMS example
    const AWS = require('aws-sdk');
    const kms = new AWS.KMS();

    async function encryptWithKMS(plaintext) {
        const params = {
            KeyId: 'alias/my-key',
            Plaintext: plaintext
        };
        
        const result = await kms.encrypt(params).promise();
        return result.CiphertextBlob.toString('base64');
    }

    async function decryptWithKMS(ciphertext) {
        const params = {
            CiphertextBlob: Buffer.from(ciphertext, 'base64')
        };
        
        const result = await kms.decrypt(params).promise();
        return result.Plaintext.toString();
    }
    ```

---

**Encrypt sensitive data always! 🔐**
