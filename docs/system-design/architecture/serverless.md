# Serverless Architecture

**No server management** | ☁️ Cloud-Native | 💰 Pay-per-use | ⚡ Auto-scale

---

## Overview

Serverless architecture lets you build and run applications without managing servers. You write code, deploy it to a cloud provider, and it runs on-demand with automatic scaling.

**Key Principle:** You don't manage infrastructure - the cloud provider does.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Cloud Provider                        │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    │
│  │  Function  │    │  Function  │    │  Function  │    │
│  │  Instance  │    │  Instance  │    │  Instance  │    │
│  │   (Auto)   │    │   (Auto)   │    │   (Auto)   │    │
│  └────────────┘    └────────────┘    └────────────┘    │
│         ↑                  ↑                  ↑          │
│         └──────────────────┼──────────────────┘          │
│                            │                              │
│                    ┌───────────────┐                      │
│                    │  API Gateway  │                      │
│                    └───────┬───────┘                      │
└────────────────────────────┼──────────────────────────────┘
                             │
                     ┌───────┴───────┐
                     │    Client     │
                     │  (Web/Mobile) │
                     └───────────────┘

Automatic:
- Scaling (0 → 1000 instances)
- Load balancing
- High availability
- Server management
```

---

## Core Concepts

=== "Functions as a Service (FaaS)"
    **Code runs in stateless compute containers**

    ```javascript
    // AWS Lambda Function
    exports.handler = async (event) => {
        // 1. Parse request
        const { userId, action } = JSON.parse(event.body);

        // 2. Execute business logic
        if (action === 'getProfile') {
            const user = await dynamoDB.get({ Key: { userId } });
            return {
                statusCode: 200,
                body: JSON.stringify(user)
            };
        }

        // 3. Return response
        return {
            statusCode: 400,
            body: JSON.stringify({ error: 'Invalid action' })
        };
    };
    ```

    **Characteristics:**
    - **Stateless:** No state between invocations
    - **Short-lived:** Max execution time (AWS: 15 min)
    - **Event-driven:** Triggered by events (HTTP, queue, schedule)
    - **Auto-scaling:** From 0 to 1000s of instances

=== "Backend as a Service (BaaS)"
    **Use managed services instead of building**

    ```javascript
    // Traditional: Build your own
    - Authentication server
    - Database server
    - File storage server
    - Email server

    // Serverless: Use managed services
    - Auth: AWS Cognito, Auth0, Firebase Auth
    - Database: DynamoDB, Firestore, MongoDB Atlas
    - Storage: S3, Cloud Storage
    - Email: SendGrid, AWS SES
    - Search: Algolia, Elasticsearch Service
    ```

=== "Triggers"
    **Events that invoke functions**

    | Trigger Type | Example | Use Case |
    |-------------|---------|----------|
    | **HTTP Request** | API Gateway | REST APIs, webhooks |
    | **Queue Message** | SQS, Pub/Sub | Async processing |
    | **File Upload** | S3, Cloud Storage | Image processing |
    | **Database Change** | DynamoDB Streams | Data sync, notifications |
    | **Schedule** | CloudWatch Events | Cron jobs, reports |
    | **Authentication** | Cognito triggers | User signup flow |

=== "Cold Start"
    **Delay when function hasn't run recently**

    ```
    Cold Start (first request):
    ┌──────────────────────────────────┐
    │ 1. Provision container (100ms)   │
    │ 2. Load code (50ms)               │
    │ 3. Initialize runtime (100ms)     │
    │ 4. Run function (50ms)            │
    └──────────────────────────────────┘
    Total: 300ms

    Warm Start (subsequent requests):
    ┌──────────────────────────────────┐
    │ 1. Run function (50ms)            │
    └──────────────────────────────────┘
    Total: 50ms (6x faster!)

    Container stays warm for ~15 minutes
    ```

---

## Serverless Components

=== "AWS Serverless Stack"
    ```
    ┌─────────────────────────────────────────┐
    │ Client (Web/Mobile)                     │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────┐
    │ CloudFront (CDN)                        │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────┐
    │ API Gateway (HTTP routing)              │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────┐
    │ Lambda (Business logic)                 │
    │ - Node.js, Python, Go, Java...          │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌───────┬──────────┬──────────┬──────────┐
    │       │          │          │          │
    │  S3   │ DynamoDB │   RDS    │  SQS     │
    │(Files)│ (NoSQL)  │  (SQL)   │(Queues)  │
    └───────┴──────────┴──────────┴──────────┘
    ```

    **Deployment:**
    ```yaml
    # serverless.yml (Serverless Framework)
    service: my-api

    provider:
      name: aws
      runtime: nodejs18.x

    functions:
      getUser:
        handler: handler.getUser
        events:
          - http:
              path: users/{id}
              method: get
    ```

=== "Google Cloud Serverless"
    ```
    ┌─────────────────────────────────────────┐
    │ Cloud Functions (FaaS)                  │
    │ - Python, Node.js, Go, Java             │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌───────┬──────────┬──────────┬──────────┐
    │       │          │          │          │
    │  GCS  │Firestore │  Cloud   │  Pub/Sub │
    │(Files)│ (NoSQL)  │   SQL    │(Messages)│
    └───────┴──────────┴──────────┴──────────┘
    ```

=== "Azure Serverless"
    ```
    ┌─────────────────────────────────────────┐
    │ Azure Functions (FaaS)                  │
    │ - C#, JavaScript, Python, Java          │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌───────┬──────────┬──────────┬──────────┐
    │       │          │          │          │
    │ Blob  │  Cosmos  │   SQL    │  Service │
    │Storage│   DB     │ Database │   Bus    │
    └───────┴──────────┴──────────┴──────────┘
    ```

---

## When to Use Serverless

| Scenario | Recommended? | Reason |
|----------|--------------|--------|
| **Spiky traffic** | ✅ Yes | Auto-scales, pay only for usage |
| **Event processing** | ✅ Yes | Natural fit (file uploads, webhooks) |
| **Microservices** | ✅ Yes | Each function is a microservice |
| **Startups/MVPs** | ✅ Yes | No infrastructure management |
| **Scheduled jobs** | ✅ Yes | Cron without servers |
| **Low latency required** | ❌ No | Cold starts add latency |
| **Long-running tasks** | ❌ No | 15-minute timeout (AWS Lambda) |
| **Stateful workloads** | ❌ No | Functions are stateless |
| **Predictable, constant load** | ⚠️ Maybe | Traditional servers might be cheaper |

---

## Advantages

### ✅ **No Server Management**

You write code, cloud provider handles everything else:

```
You DON'T manage:
❌ Server provisioning
❌ OS updates
❌ Security patches
❌ Scaling configuration
❌ Load balancers
❌ Monitoring setup
❌ High availability

You ONLY write:
✅ Business logic
```

### ✅ **Auto-Scaling**

Scales from 0 to millions automatically:

```
Traffic Pattern:
Time    Requests/sec  Lambda Instances
8am     10            1
12pm    1000          100
6pm     10000         1000
2am     0             0  (No cost!)
```

**No capacity planning needed!**

### ✅ **Pay-per-Use**

Only pay when code runs:

```
Traditional Server:
$50/month for 24/7 running
(Even if idle 95% of time)

Serverless:
$0.20 for 1 million requests
$0 when not used
```

**Example (AWS Lambda):**
- Free tier: 1M requests/month
- After: $0.20 per 1M requests
- Execution: $0.0000166667 per GB-second

### ✅ **Built-in High Availability**

Cloud provider handles HA:

```
Multi-AZ deployment (automatic)
┌───────────┐  ┌───────────┐  ┌───────────┐
│  AZ us-    │  │  AZ us-    │  │  AZ us-    │
│  east-1a  │  │  east-1b  │  │  east-1c  │
│           │  │           │  │           │
│ Function  │  │ Function  │  │ Function  │
└───────────┘  └───────────┘  └───────────┘

One AZ fails? Traffic routes to others automatically
```

---

## Disadvantages

### ❌ **Cold Start Latency**

First request can be slow:

```javascript
// Latency comparison
Warm Lambda:     50ms
Cold Lambda:     300-500ms (Node.js)
Cold Lambda:     1-3 seconds (Java)

// For latency-sensitive APIs, this is problematic
```

**Mitigation:**
```javascript
// 1. Keep functions warm
setInterval(() => {
    axios.get('https://my-api.com/health'); // Ping every 5 minutes
}, 5 * 60 * 1000);

// 2. Use provisioned concurrency (AWS)
// Keeps X instances always warm (costs more)

// 3. Choose lightweight runtime (Node.js, Python, Go)
```

### ❌ **Vendor Lock-In**

Hard to migrate between cloud providers:

```javascript
// AWS Lambda
exports.handler = async (event, context) => { ... }

// Google Cloud Functions
exports.myFunction = (req, res) => { ... }

// Azure Functions
module.exports = async function (context, req) { ... }

// Different signatures, different deployment, different tools
```

**Mitigation:** Use abstraction layer (Serverless Framework, Terraform)

### ❌ **Execution Time Limits**

Functions can't run forever:

| Provider | Max Execution Time |
|----------|-------------------|
| AWS Lambda | 15 minutes |
| Google Cloud Functions | 9 minutes (gen 1), 60 minutes (gen 2) |
| Azure Functions | 10 minutes (Consumption plan) |

```javascript
// ❌ Won't work: Video transcoding (30 minutes)
exports.transcodeVideo = async (event) => {
    await ffmpeg.transcode(largeVideo); // Takes 30 minutes
    // Function times out at 15 minutes!
};

// ✅ Solution: Use Step Functions or batch processing
```

### ❌ **Debugging Difficulty**

Hard to debug without production environment:

```
Local development:
- Can't replicate AWS environment exactly
- Different behavior than production
- Hard to test triggers (S3, DynamoDB streams)

Production debugging:
- Must rely on logs (CloudWatch)
- Can't SSH into server (no server!)
- Limited debugging tools
```

### ❌ **Stateless Constraints**

Can't maintain state between requests:

```javascript
// ❌ Won't work: In-memory cache
let cache = {};

exports.handler = async (event) => {
    if (cache[event.key]) {
        return cache[event.key]; // Cache doesn't persist!
    }
    cache[event.key] = await fetchData(event.key);
    return cache[event.key];
};

// ✅ Solution: Use external cache (Redis, DynamoDB)
const redis = new Redis();

exports.handler = async (event) => {
    let data = await redis.get(event.key);
    if (!data) {
        data = await fetchData(event.key);
        await redis.set(event.key, data);
    }
    return data;
};
```

---

## Common Patterns

=== "API Backend"
    **REST API using Lambda + API Gateway**

    ```javascript
    // GET /users/:id
    exports.getUser = async (event) => {
        const userId = event.pathParameters.id;

        const user = await dynamoDB.get({
            TableName: 'Users',
            Key: { id: userId }
        });

        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(user)
        };
    };

    // POST /users
    exports.createUser = async (event) => {
        const userData = JSON.parse(event.body);

        await dynamoDB.put({
            TableName: 'Users',
            Item: { id: uuid(), ...userData }
        });

        return {
            statusCode: 201,
            body: JSON.stringify({ message: 'User created' })
        };
    };
    ```

    **Architecture:**
    ```
    Client → API Gateway → Lambda → DynamoDB
    ```

=== "Image Processing"
    **Process images on upload**

    ```javascript
    // Triggered when file uploaded to S3
    exports.processImage = async (event) => {
        // Get uploaded file info
        const bucket = event.Records[0].s3.bucket.name;
        const key = event.Records[0].s3.object.key;

        // Download image
        const image = await s3.getObject({ Bucket: bucket, Key: key });

        // Resize image
        const resized = await sharp(image.Body)
            .resize(800, 600)
            .toBuffer();

        // Upload resized image
        await s3.putObject({
            Bucket: bucket,
            Key: `resized/${key}`,
            Body: resized
        });
    };
    ```

    **Flow:**
    ```
    User uploads → S3 → Triggers Lambda → Resize → Save to S3
    ```

=== "Scheduled Tasks"
    **Cron jobs without servers**

    ```javascript
    // Runs every night at 2 AM
    exports.cleanupOldData = async (event) => {
        const cutoffDate = Date.now() - (30 * 24 * 60 * 60 * 1000); // 30 days ago

        // Delete old records
        await dynamoDB.scan({
            TableName: 'Logs',
            FilterExpression: 'timestamp < :cutoff',
            ExpressionAttributeValues: { ':cutoff': cutoffDate }
        }).then(result => {
            // Delete items
        });

        console.log('Cleanup complete');
    };
    ```

    **Configuration:**
    ```yaml
    functions:
      cleanup:
        handler: handler.cleanupOldData
        events:
          - schedule: cron(0 2 * * ? *)  # 2 AM daily
    ```

=== "Event Processing"
    **Process queue messages**

    ```javascript
    // Triggered by SQS message
    exports.processOrder = async (event) => {
        for (const record of event.Records) {
            const order = JSON.parse(record.body);

            // Process order
            await inventory.reserve(order.items);
            await payment.charge(order.userId, order.total);
            await email.send(order.userId, 'Order confirmed');

            // Delete message from queue (automatic if function succeeds)
        }
    };
    ```

    **Flow:**
    ```
    Order Service → SQS → Lambda → Process → Update DB
    ```

---

## Cost Optimization

### **Right-Size Memory**

More memory = faster execution but higher cost:

```javascript
// Test different memory configurations
256 MB:  1000ms execution = $0.000002083
512 MB:  600ms execution  = $0.000001875 (cheaper!)
1024 MB: 400ms execution  = $0.000002667
```

**Find sweet spot:** Sometimes more memory = lower cost

### **Reduce Function Size**

Smaller packages = faster cold starts:

```javascript
// ❌ Bad: Include entire AWS SDK (70MB)
const AWS = require('aws-sdk');

// ✅ Good: Import only what you need
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
```

### **Use Caching**

Cache expensive computations:

```javascript
// Cache at container level (survives warm starts)
let cachedData = null;

exports.handler = async (event) => {
    if (!cachedData) {
        cachedData = await expensiveOperation();
    }
    return cachedData;
};
```

### **Batch Processing**

Process multiple items per invocation:

```javascript
// ❌ Expensive: 1 Lambda per item (1000 items = $0.20)
for (const item of items) {
    await lambda.invoke({ FunctionName: 'process', Payload: item });
}

// ✅ Cheap: Batch of 100 items (1000 items = $0.002)
const batches = chunk(items, 100);
for (const batch of batches) {
    await lambda.invoke({ FunctionName: 'processBatch', Payload: batch });
}
```

---

## Best Practices

### ✅ **Do's**

1. **Keep functions small and focused**
   ```javascript
   // ✅ Good: Single responsibility
   - getUserById()
   - createUser()
   - deleteUser()

   // ❌ Bad: God function
   - handleAllUserOperations()
   ```

2. **Use environment variables for configuration**
   ```javascript
   const TABLE_NAME = process.env.TABLE_NAME;
   const API_KEY = process.env.API_KEY;
   ```

3. **Implement proper error handling**
   ```javascript
   exports.handler = async (event) => {
       try {
           return await processEvent(event);
       } catch (error) {
           console.error('Error:', error);
           return {
               statusCode: 500,
               body: JSON.stringify({ error: error.message })
           };
       }
   };
   ```

4. **Use structured logging**
   ```javascript
   console.log(JSON.stringify({
       level: 'INFO',
       requestId: event.requestContext.requestId,
       message: 'Processing order',
       orderId: order.id
   }));
   ```

### ❌ **Don'ts**

1. **Don't use Lambda for long-running tasks**
   - Video encoding: Use ECS/Batch instead
   - Large data processing: Use EMR/Glue
   - WebSockets: Use API Gateway WebSocket or AppSync

2. **Don't ignore cold starts**
   - Profile different runtimes
   - Consider provisioned concurrency for critical APIs
   - Use CDN caching where possible

3. **Don't create fat functions**
   - Keep deployment package < 50MB
   - Remove unused dependencies
   - Use Lambda Layers for shared code

---

## Real-World Examples

=== "Netflix"
    **Use case:** Video encoding, data processing

    - 1000+ Lambda functions
    - Process billions of events daily
    - Save millions on infrastructure

=== "Coca-Cola"
    **Use case:** Vending machine IoT

    - Serverless APIs
    - React to vending machine events
    - Scale from 0 to peak automatically

=== "iRobot"
    **Use case:** Robot vacuum cloud**

    - Process sensor data from millions of robots
    - Serverless data pipeline
    - Cost-effective scaling

---

## Interview Talking Points

**Q: When would you choose serverless over traditional servers?**

✅ **Strong Answer:**
> "I'd choose serverless for event-driven workloads with variable traffic, like image processing on upload or webhook handlers. The auto-scaling and pay-per-use model makes sense when you can't predict load or when traffic is sporadic. For example, a background job that runs when users upload files - you might process 10 images one day and 10,000 the next. With serverless, you pay only for what you use and don't manage any infrastructure. However, I'd avoid it for latency-sensitive APIs (due to cold starts) or long-running tasks (15-minute timeout on AWS Lambda)."

**Q: How do you handle cold starts in serverless applications?**

✅ **Strong Answer:**
> "I'd first minimize cold starts by keeping deployment packages small and choosing lightweight runtimes like Node.js or Python. For critical endpoints, I'd use provisioned concurrency to keep instances warm, though this costs more. I'd also implement caching at the API Gateway level for frequently accessed data and use CloudFront CDN for static content. For non-critical paths, I'd accept the cold start trade-off since it only affects 1-5% of requests and the cost savings are significant. I'd monitor P99 latency to ensure cold starts don't exceed SLA requirements."

---

## Related Topics

- [Microservices Architecture](microservices.md) - Serverless enables microservices
- [Event-Driven Architecture](event-driven.md) - Natural fit with serverless
- [API Design](../communication/api-design/index.md) - Design serverless APIs
- [Cloud Providers](../deployment/index.md) - Deployment strategies

---

**Focus on code, not servers! ☁️**
