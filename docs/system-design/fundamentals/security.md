# Security in System Design: Building Secure Systems

## 🎯 Security Fundamentals

Security is not an afterthought—it must be built into every layer of your system architecture. This section covers the essential security principles and practices for designing secure, scalable systems.

## 🛡️ Defense in Depth

### Layered Security Model
**Multiple security layers provide comprehensive protection**

**Application Layer**:
- Input validation and sanitization
- Authentication and authorization
- Session management
- Error handling and logging

**Network Layer**:
- Firewalls and network segmentation
- VPNs and encrypted communications
- Intrusion detection systems
- DDoS protection

**Infrastructure Layer**:
- Operating system hardening
- Access controls and permissions
- Patch management
- Physical security

**Data Layer**:
- Encryption at rest and in transit
- Database security
- Backup encryption
- Data masking and tokenization

## 🔐 Authentication and Authorization

### Authentication Methods
**Verify user identity**

**Password-Based Authentication**:
```
Username + Password → Hash Comparison → Access Granted/Denied
```

**Best Practices**:
- Use strong password policies
- Implement account lockout mechanisms
- Store password hashes (bcrypt, scrypt, Argon2)
- Never store plaintext passwords

**Multi-Factor Authentication (MFA)**:
```
Factor 1: Something you know (password)
Factor 2: Something you have (phone, token)
Factor 3: Something you are (biometrics)
```

**Implementation Example**:
```
1. User enters username/password
2. System validates credentials
3. System sends SMS/email with code
4. User enters verification code
5. System validates code and grants access
```

**Token-Based Authentication**:
```
JWT Token Structure:
Header.Payload.Signature

Example:
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

### Authorization Models
**Control access to resources**

**Role-Based Access Control (RBAC)**:
```
User → Role → Permissions
├── Admin → Full access to all resources
├── Manager → Read/write access to team resources
└── User → Read-only access to personal resources
```

**Attribute-Based Access Control (ABAC)**:
```
Access Decision = f(
  User Attributes,
  Resource Attributes,
  Environment Attributes,
  Action Attributes
)

Example:
Allow if:
- User.department == "Finance" AND
- Resource.type == "financial_report" AND
- Environment.time == "business_hours" AND
- Action == "read"
```

**OAuth 2.0 Flow**:
```
1. Client requests authorization from resource owner
2. Resource owner grants authorization
3. Client exchanges authorization grant for access token
4. Client uses access token to access protected resources
```

## 🔒 Data Protection

### Encryption
**Protect data confidentiality**

**Encryption at Rest**:
- Database encryption (TDE - Transparent Data Encryption)
- File system encryption
- Backup encryption
- Key management systems

**Encryption in Transit**:
- TLS/SSL for web traffic
- VPN for network communications
- Database connection encryption
- Message queue encryption

**Key Management**:
```
Key Hierarchy:
Master Key → Data Encryption Keys → Encrypted Data

Key Rotation:
1. Generate new key
2. Encrypt new data with new key
3. Re-encrypt existing data (optional)
4. Retire old key after grace period
```

### Data Classification
**Categorize data by sensitivity**

**Classification Levels**:
- **Public**: No harm if disclosed
- **Internal**: Moderate risk if disclosed
- **Confidential**: High risk if disclosed
- **Restricted**: Severe risk if disclosed

**Handling Requirements**:
```
Public Data:
- Standard backup procedures
- Basic access controls

Confidential Data:
- Encryption required
- Access logging
- Regular access reviews
- Secure disposal

Restricted Data:
- Strong encryption
- Multi-factor authentication
- Continuous monitoring
- Data loss prevention
```

## 🌐 Network Security

### Network Segmentation
**Isolate network traffic**

**DMZ (Demilitarized Zone)**:
```
Internet → Firewall → DMZ (Web Servers) → Internal Firewall → Internal Network
```

**Micro-segmentation**:
```
Application Tier → Database Tier
       ↓               ↓
   Firewall Rules   Firewall Rules
```

**Zero Trust Architecture**:
```
Principles:
- Never trust, always verify
- Assume breach has occurred
- Verify explicitly for every access
- Use least privilege access
```

### API Security
**Secure service-to-service communication**

**API Gateway Security**:
- Rate limiting and throttling
- API key management
- Request/response validation
- Threat detection

**HTTPS Everywhere**:
```
HTTP Request → SSL/TLS Encryption → HTTPS
Benefits:
- Data encryption in transit
- Authentication of server
- Data integrity verification
```

**API Security Headers**:
```
Security Headers:
- Content-Security-Policy
- X-Frame-Options
- X-Content-Type-Options
- Strict-Transport-Security
- X-XSS-Protection
```

## 🛡️ Input Validation and Sanitization

### Common Vulnerabilities
**OWASP Top 10 security risks**

**SQL Injection**:
```
Vulnerable:
query = "SELECT * FROM users WHERE username = '" + username + "'"

Secure:
query = "SELECT * FROM users WHERE username = ?"
preparedStatement.setString(1, username)
```

**Cross-Site Scripting (XSS)**:
```
Vulnerable:
output = "<div>" + userInput + "</div>"

Secure:
output = "<div>" + escapeHtml(userInput) + "</div>"
```

**Cross-Site Request Forgery (CSRF)**:
```
Protection:
- CSRF tokens
- SameSite cookies
- Referrer validation
```

### Input Validation Strategies
**Validate all inputs**

**Validation Types**:
- **Syntax validation**: Format and structure
- **Semantic validation**: Business logic rules
- **Length validation**: Size constraints
- **Type validation**: Data type checking

**Example Validation**:
```json
{
  "email": {
    "type": "string",
    "format": "email",
    "maxLength": 254
  },
  "age": {
    "type": "integer",
    "minimum": 0,
    "maximum": 150
  },
  "password": {
    "type": "string",
    "minLength": 8,
    "pattern": "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]"
  }
}
```

## 🔍 Monitoring and Logging

### Security Monitoring
**Detect and respond to threats**

**Security Information and Event Management (SIEM)**:
```
Log Sources:
├── Application logs
├── System logs
├── Network logs
├── Security device logs
└── Database logs

↓

SIEM System:
├── Log aggregation
├── Correlation rules
├── Threat detection
├── Incident response
└── Compliance reporting
```

**Key Metrics to Monitor**:
- Failed login attempts
- Privilege escalation attempts
- Data access patterns
- Network anomalies
- System resource usage

### Audit Logging
**Track security-relevant events**

**What to Log**:
- Authentication events
- Authorization decisions
- Data access and modifications
- Administrative actions
- System configuration changes

**Log Format Example**:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "eventType": "authentication",
  "result": "success",
  "userId": "user123",
  "sourceIP": "192.168.1.100",
  "userAgent": "Mozilla/5.0...",
  "sessionId": "sess456",
  "resource": "/api/users/profile"
}
```

## 🚨 Incident Response

### Incident Response Plan
**Prepare for security incidents**

**Response Phases**:
1. **Preparation**: Develop response procedures
2. **Detection**: Identify security incidents
3. **Analysis**: Assess impact and scope
4. **Containment**: Limit damage and prevent spread
5. **Eradication**: Remove threat from system
6. **Recovery**: Restore normal operations
7. **Lessons Learned**: Improve future response

**Response Team Roles**:
- **Incident Commander**: Overall response coordination
- **Security Analyst**: Technical investigation
- **Communications Lead**: Stakeholder communication
- **Legal/Compliance**: Regulatory requirements
- **IT Operations**: System recovery

### Disaster Recovery
**Recover from security incidents**

**Recovery Strategies**:
- **Backup and restore**: Regular system backups
- **Failover systems**: Redundant infrastructure
- **Incident isolation**: Contain affected systems
- **Communication plan**: Notify stakeholders

**Recovery Time Objectives**:
- **RTO (Recovery Time Objective)**: How long to restore service
- **RPO (Recovery Point Objective)**: How much data loss is acceptable

## 🔐 Secure Development Practices

### Security by Design
**Build security into development process**

**Secure Development Lifecycle (SDLC)**:
```
1. Requirements → Security requirements
2. Design → Threat modeling
3. Implementation → Secure coding
4. Testing → Security testing
5. Deployment → Security configuration
6. Maintenance → Security monitoring
```

**Threat Modeling**:
```
STRIDE Framework:
- Spoofing: Impersonation attacks
- Tampering: Data modification
- Repudiation: Denial of actions
- Information Disclosure: Data leaks
- Denial of Service: Availability attacks
- Elevation of Privilege: Unauthorized access
```

### Code Security
**Secure coding practices**

**Common Security Issues**:
- Buffer overflows
- Race conditions
- Memory leaks
- Improper error handling
- Insecure random number generation

**Security Testing**:
- **Static Analysis**: Code review tools
- **Dynamic Analysis**: Runtime testing
- **Penetration Testing**: Simulated attacks
- **Dependency Scanning**: Third-party vulnerabilities

## 🛠️ Security Tools and Technologies

### Security Scanning
**Automated vulnerability detection**

**Vulnerability Scanners**:
- **Network scanners**: Nmap, Nessus
- **Web application scanners**: OWASP ZAP, Burp Suite
- **Container scanners**: Clair, Trivy
- **Infrastructure scanners**: OpenVAS, Qualys

**Scanning Schedule**:
- **Continuous**: Critical systems
- **Weekly**: Production systems
- **Monthly**: Development systems
- **Quarterly**: Comprehensive scans

### Security Automation
**Automate security processes**

**DevSecOps Pipeline**:
```
Code → Security Scan → Build → Security Test → Deploy → Monitor
```

**Automated Security Controls**:
- Patch management
- Configuration management
- Access provisioning/deprovisioning
- Security monitoring alerts
- Incident response workflows

## 🌍 Compliance and Privacy

### Regulatory Compliance
**Meet legal and industry requirements**

**Common Regulations**:
- **GDPR**: European data protection
- **CCPA**: California privacy law
- **PCI DSS**: Payment card security
- **HIPAA**: Healthcare data protection
- **SOX**: Financial reporting

**Compliance Controls**:
```
GDPR Requirements:
- Data minimization
- Purpose limitation
- Consent management
- Right to erasure
- Data portability
- Privacy by design
```

### Privacy Protection
**Protect personal information**

**Privacy Techniques**:
- **Data anonymization**: Remove identifying information
- **Data pseudonymization**: Replace identifiers with pseudonyms
- **Data masking**: Hide sensitive data in non-production
- **Differential privacy**: Add noise to protect individuals

## 🎯 Security Best Practices

### Design Principles
1. **Principle of Least Privilege**: Grant minimum necessary access
2. **Fail Securely**: Default to deny access on errors
3. **Defense in Depth**: Multiple security layers
4. **Security by Design**: Build security from the start
5. **Assume Breach**: Plan for security incidents

### Implementation Guidelines
1. **Regular security updates**: Keep systems patched
2. **Strong authentication**: Multi-factor where possible
3. **Encrypt sensitive data**: At rest and in transit
4. **Monitor continuously**: Watch for suspicious activity
5. **Train users**: Security awareness education

### Common Mistakes to Avoid
1. **Security through obscurity**: Don't rely on secrecy alone
2. **Ignoring insider threats**: Monitor internal users
3. **Weak password policies**: Enforce strong passwords
4. **Unencrypted communications**: Use TLS/SSL everywhere
5. **Insufficient logging**: Log security events properly

Security is an ongoing process, not a one-time implementation. Regular assessment, updating, and improvement of security measures is essential for maintaining a secure system architecture.
