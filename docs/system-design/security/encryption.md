# Encryption

Encryption transforms readable data into unintelligible ciphertext that can only be reversed with the correct key. It protects data in two situations: **in transit** (moving across a network) and **at rest** (stored on disk). These are different problems with different solutions, but both are essential — encrypting only one is like locking your front door while leaving the windows open.

---

## Symmetric vs Asymmetric Encryption

Understanding this distinction is fundamental to everything else in encryption.

```
Symmetric (one key):
  Encrypt: plaintext + key → ciphertext
  Decrypt: ciphertext + same key → plaintext

  Fast. Used for bulk data encryption.
  Problem: how do you share the key securely?


Asymmetric (two keys):
  Encrypt: plaintext + public key → ciphertext
  Decrypt: ciphertext + private key → plaintext

  Slow. Used for key exchange and digital signatures.
  Solves the key sharing problem — public key can be shared openly.
```

| Property | Symmetric (AES) | Asymmetric (RSA, ECDSA) |
|---|---|---|
| **Speed** | Fast (GB/s) | Slow (KB/s) |
| **Key sharing** | Must be secret | Public key is shareable |
| **Use case** | Encrypting data | Key exchange, signatures |
| **Key size** | 128-256 bits | 2048-4096 bits |

In practice, systems use **both together**: asymmetric encryption to securely exchange a symmetric key, then symmetric encryption for the actual data. This is exactly how TLS works.

---

=== "In Transit (TLS)"

    ## Encryption in Transit (TLS)

    TLS (Transport Layer Security) protects data moving between a client and server. When you see HTTPS in a URL, TLS is at work. It provides three guarantees: **confidentiality** (nobody can read the data), **integrity** (nobody can modify it), and **authentication** (you're talking to who you think you are).

    ### The TLS Handshake

    ```
    Client                              Server
      │                                    │
      │──→ ClientHello                     │  (supported ciphers, TLS version)
      │                                    │
      │    ServerHello + Certificate ←──│  (chosen cipher, server's public key)
      │                                    │
      │    Verify certificate              │  (is it signed by a trusted CA?)
      │                                    │
      │──→ Key exchange material           │  (encrypted with server's public key)
      │                                    │
      │←─→ Both derive session key         │  (symmetric key for this connection)
      │                                    │
      │══════ Encrypted traffic ══════│  (all data encrypted with session key)
    ```

    After the handshake, all data flows through the symmetric session key — fast encryption for the actual data transfer, after asymmetric encryption solved the key exchange problem.

    **TLS 1.3** (the current version) reduces the handshake to one round trip (down from two in TLS 1.2), improving latency. **Cloudflare** reported a 30% reduction in connection time after enabling TLS 1.3 across their network.

    ### mTLS (Mutual TLS)

    Standard TLS only authenticates the server — the client verifies the server's certificate. **Mutual TLS** adds client authentication: both sides present certificates and verify each other. This is common in microservices architectures where services need to verify each other's identity.

    **Istio** and other service meshes use mTLS to encrypt all inter-service traffic automatically. **Stripe** requires mTLS for certain high-security API integrations.

=== "At Rest"

    ## Encryption at Rest

    Data stored on disk — in databases, file systems, backups, and logs — needs encryption to protect against physical theft, unauthorized access to storage, and data breaches.

    ### Levels of At-Rest Encryption

    ```
    Full Disk Encryption (FDE):
      Entire disk encrypted. Protects against physical theft.
      OS handles transparently — applications don't know.
      AWS EBS encryption, Azure Disk Encryption, BitLocker.

    Database-Level Encryption (TDE):
      Database encrypts data files on disk.
      Queries work normally — decryption is transparent.
      PostgreSQL TDE, SQL Server TDE, Oracle TDE.

    Application-Level Encryption:
      Application encrypts specific fields before storing.
      Only the application can decrypt — not even DB admins.
      Credit card numbers, SSNs, medical records.

      plaintext: "SSN: 123-45-6789"
          ↓ encrypt with AES-256
      stored: "a4f2b8c9e1d3..."
          ↓ decrypt when needed
      displayed: "SSN: 123-45-6789"
    ```

    Each level protects against different threats:

    | Level | Protects Against | Doesn't Protect Against |
    |---|---|---|
    | **Full disk** | Physical theft, decommissioned hardware | DB admin access, SQL injection |
    | **Database TDE** | Stolen backup files, unauthorized file access | DB admin access, application compromise |
    | **Application-level** | DB admin snooping, SQL injection data theft | Application server compromise |

    **Best practice: layer them.** AWS RDS enables disk encryption by default. The database can add TDE. And the application encrypts the most sensitive fields (PII, payment data). **Stripe** encrypts card numbers at the application level with per-merchant keys — even if an attacker compromises the database, the encrypted data is useless without the corresponding keys.

=== "Key Management"

    ## Key Management

    Encryption is only as secure as the keys. A perfectly encrypted database is worthless if the key is hardcoded in the application source or stored in the same database.

    ### Principles

    - **Separate keys from data.** The encryption key should never live in the same system as the encrypted data.
    - **Rotate keys regularly.** If a key is compromised, only data encrypted since the last rotation is at risk. Automated rotation (every 90 days) limits exposure.
    - **Use envelope encryption.** Encrypt data with a data key, then encrypt the data key with a master key. This limits how much data needs to be re-encrypted during key rotation — you only re-encrypt the data key.

    ```
    Envelope Encryption:

    Master Key (stored in KMS, never leaves)
         │
         └──→ encrypts Data Key
                  │
                  └──→ encrypts actual data

    To decrypt: KMS decrypts data key → data key decrypts data
    ```

    ### Key Management Services

    **AWS KMS** (Key Management Service) stores master keys in hardware security modules (HSMs) that are tamper-resistant. The master key never leaves the HSM — encryption and decryption happen inside it. Google Cloud KMS and Azure Key Vault work similarly.

    **HashiCorp Vault** is the standard for managing secrets (API keys, database credentials, encryption keys) in a platform-agnostic way. It provides dynamic secrets (generate database credentials on demand, auto-revoke after a TTL), encryption as a service, and detailed audit logs.

---

## Hashing vs Encryption

These are different operations, and confusing them leads to security vulnerabilities.

| Property | Encryption | Hashing |
|---|---|---|
| **Reversible?** | Yes (with the key) | No (one-way) |
| **Purpose** | Protect data that needs to be read later | Verify data without storing it |
| **Use case** | Credit card numbers, PII | Passwords, data integrity |
| **Algorithm** | AES-256, ChaCha20 | bcrypt, argon2, SHA-256 |

**Passwords must be hashed, never encrypted.** If passwords are encrypted, anyone with the key can decrypt all of them. If hashed with bcrypt (cost factor ~12, taking ~250ms per hash), an attacker who steals the database must brute-force each password individually.

---

## Key Takeaways

1. **TLS everywhere.** All traffic — external and internal — should be encrypted in transit. TLS 1.3 is the current standard. Use mTLS between services.

2. **Layer at-rest encryption.** Disk encryption + database encryption + application-level encryption for sensitive fields. Each layer protects against different threats.

3. **Never manage keys yourself.** Use AWS KMS, Google Cloud KMS, or HashiCorp Vault. Keys should live in HSMs and never be exposed in application code or configuration files.

4. **Hash passwords, encrypt everything else.** Passwords are verified, not decrypted — use bcrypt or argon2. Sensitive data that needs to be read back (credit cards, SSNs) gets encrypted.

5. **Envelope encryption for scalability.** Encrypt data with data keys, encrypt data keys with master keys. This makes key rotation practical at any scale.

---

## Related Topics

- **[Authentication](authentication.md)** — TLS protects credentials in transit, hashing protects stored passwords
- **[API Security](api-security.md)** — HTTPS enforcement and API key protection
- **[Common Attacks](common-attacks.md)** — man-in-the-middle attacks that encryption prevents
