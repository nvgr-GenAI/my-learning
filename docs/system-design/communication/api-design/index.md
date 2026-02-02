# API Design

**Design APIs that developers love** | 🔌 REST | 📊 GraphQL | ⚡ gRPC | 🔄 WebSocket

---

## Overview

APIs are the contracts between services. Good API design makes systems easier to build, maintain, and scale.

**Key Principle:** Design for your API consumers, not just your implementation.

---

## API Paradigms Comparison

| Paradigm | Protocol | Data Format | Use Case | Performance |
|----------|----------|-------------|----------|-------------|
| **REST** | HTTP | JSON | Public APIs, CRUD | Good |
| **GraphQL** | HTTP | JSON | Complex data fetching | Good |
| **gRPC** | HTTP/2 | Protobuf | Internal services | Excellent |
| **WebSocket** | WebSocket | JSON/Binary | Real-time, bidirectional | Excellent |

---

## REST (Representational State Transfer)

=== "Principles"
    **Core REST principles:**

    1. **Resource-based:** URLs represent resources
    2. **HTTP methods:** Use GET, POST, PUT, DELETE correctly
    3. **Stateless:** Each request contains all needed information
    4. **Cacheable:** Responses indicate if they can be cached

    ```
    REST URL Structure:
    https://api.example.com/v1/users/123/orders/456

    Breakdown:
    - api.example.com: Domain
    - v1: API version
    - users: Resource collection
    - 123: Resource ID (user)
    - orders: Nested resource
    - 456: Order ID
    ```

=== "HTTP Methods"
    **CRUD operations mapped to HTTP:**

    | Method | Operation | Idempotent? | Safe? | Example |
    |--------|-----------|-------------|-------|---------|
    | **GET** | Read | ✅ Yes | ✅ Yes | GET /users/123 |
    | **POST** | Create | ❌ No | ❌ No | POST /users |
    | **PUT** | Update/Replace | ✅ Yes | ❌ No | PUT /users/123 |
    | **PATCH** | Partial Update | ❌ No | ❌ No | PATCH /users/123 |
    | **DELETE** | Delete | ✅ Yes | ❌ No | DELETE /users/123 |

    **Idempotent:** Same request multiple times = same result  
    **Safe:** No side effects (read-only)

=== "Examples"
    **Well-designed REST API:**

    ```javascript
    // List users (with pagination)
    GET /api/v1/users?page=1&limit=20
    Response:
    {
        "data": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ],
        "pagination": {
            "page": 1,
            "limit": 20,
            "total": 150,
            "totalPages": 8
        }
    }

    // Get single user
    GET /api/v1/users/123
    Response:
    {
        "id": 123,
        "name": "Alice",
        "email": "alice@example.com",
        "createdAt": "2025-01-01T00:00:00Z"
    }

    // Create user
    POST /api/v1/users
    Body:
    {
        "name": "Charlie",
        "email": "charlie@example.com"
    }
    Response: 201 Created
    {
        "id": 456,
        "name": "Charlie",
        "email": "charlie@example.com"
    }

    // Update user
    PUT /api/v1/users/123
    Body:
    {
        "name": "Alice Smith",
        "email": "alice.smith@example.com"
    }

    // Partial update
    PATCH /api/v1/users/123
    Body:
    {
        "email": "newemail@example.com"
    }

    // Delete user
    DELETE /api/v1/users/123
    Response: 204 No Content
    ```

=== "Best Practices"
    **REST API design guidelines:**

    1. **Use nouns, not verbs**
       ```
       ✅ GET /users/123
       ❌ GET /getUser?id=123

       ✅ POST /orders
       ❌ POST /createOrder
       ```

    2. **Use plural names for collections**
       ```
       ✅ GET /users
       ❌ GET /user
       ```

    3. **Nest resources logically**
       ```
       ✅ GET /users/123/orders
       ❌ GET /orders?userId=123 (acceptable, but less semantic)
       ```

    4. **Version your API**
       ```
       ✅ /api/v1/users
       ✅ /api/v2/users
       ```

    5. **Use HTTP status codes correctly**
       ```
       200 OK - Successful GET, PUT, PATCH
       201 Created - Successful POST
       204 No Content - Successful DELETE
       400 Bad Request - Invalid input
       401 Unauthorized - Not authenticated
       403 Forbidden - Authenticated but not authorized
       404 Not Found - Resource doesn't exist
       500 Internal Server Error - Server error
       ```

    6. **Provide filtering, sorting, pagination**
       ```
       GET /users?status=active&sort=-createdAt&page=1&limit=20

       - status=active: Filter
       - sort=-createdAt: Sort descending by created date
       - page=1&limit=20: Pagination
       ```

=== "Disadvantages"
    **REST limitations:**

    - ❌ **Over-fetching:** Get more data than needed
    - ❌ **Under-fetching:** Need multiple requests
    - ❌ **No schema:** Must maintain documentation
    - ❌ **Versioning complexity:** Breaking changes require new versions

    ```javascript
    // Over-fetching problem
    GET /users/123
    // Returns: id, name, email, address, phone, preferences, ...
    // But you only wanted the name!

    // Under-fetching problem
    GET /users/123        // Get user
    GET /users/123/posts  // Get user's posts
    GET /posts/1/comments // Get post comments
    // 3 requests just to show a user profile!
    ```

---

## GraphQL

=== "What is GraphQL?"
    **Query language for your API**

    **Key Innovation:** Clients specify exactly what data they need

    ```graphql
    # Client query
    query {
        user(id: 123) {
            name
            email
            posts {
                title
                comments {
                    text
                    author {
                        name
                    }
                }
            }
        }
    }

    # Single request gets all needed data!
    ```

=== "Schema"
    **Strongly typed schema:**

    ```graphql
    # Define types
    type User {
        id: ID!
        name: String!
        email: String!
        posts: [Post!]!
    }

    type Post {
        id: ID!
        title: String!
        content: String!
        author: User!
        comments: [Comment!]!
    }

    type Comment {
        id: ID!
        text: String!
        author: User!
    }

    # Define queries
    type Query {
        user(id: ID!): User
        users(limit: Int, offset: Int): [User!]!
        post(id: ID!): Post
    }

    # Define mutations
    type Mutation {
        createUser(name: String!, email: String!): User!
        updateUser(id: ID!, name: String, email: String): User!
        deleteUser(id: ID!): Boolean!
    }
    ```

=== "Implementation"
    **GraphQL server example:**

    ```javascript
    const { ApolloServer, gql } = require('apollo-server');

    // Schema
    const typeDefs = gql`
        type User {
            id: ID!
            name: String!
            email: String!
        }

        type Query {
            users: [User!]!
            user(id: ID!): User
        }

        type Mutation {
            createUser(name: String!, email: String!): User!
        }
    `;

    // Resolvers
    const resolvers = {
        Query: {
            users: () => database.users.findAll(),
            user: (parent, { id }) => database.users.findById(id)
        },
        Mutation: {
            createUser: (parent, { name, email }) => {
                return database.users.create({ name, email });
            }
        }
    };

    // Start server
    const server = new ApolloServer({ typeDefs, resolvers });
    server.listen().then(({ url }) => {
        console.log(`Server ready at ${url}`);
    });
    ```

=== "Advantages"
    **Why use GraphQL:**

    - ✅ **No over-fetching:** Get exactly what you need
    - ✅ **No under-fetching:** Single request for complex data
    - ✅ **Strongly typed:** Auto-generated documentation
    - ✅ **Versioning:** Add fields without breaking changes
    - ✅ **Introspection:** Clients can discover schema

    ```graphql
    # Only get what you need
    query {
        user(id: 123) {
            name  # Just the name, nothing else
        }
    }

    # Complex nested query in single request
    query {
        user(id: 123) {
            name
            posts(limit: 5) {
                title
                comments(limit: 3) {
                    text
                }
            }
        }
    }
    ```

=== "Disadvantages"
    **GraphQL challenges:**

    - ❌ **Caching complexity:** HTTP caching doesn't work well
    - ❌ **Query complexity:** Clients can write expensive queries
    - ❌ **Learning curve:** More complex than REST
    - ❌ **Overhead:** Overkill for simple APIs

    ```graphql
    # Expensive query (N+1 problem)
    query {
        users {           # 1 query
            posts {       # N queries (one per user)
                comments {  # N*M queries
                    author { # N*M*K queries
                        name
                    }
                }
            }
        }
    }
    # Solution: Use DataLoader to batch queries
    ```

---

## gRPC

=== "What is gRPC?"
    **High-performance RPC framework using Protocol Buffers**

    **Key Features:**
    - HTTP/2 based
    - Binary protocol (faster than JSON)
    - Strongly typed
    - Code generation

    ```protobuf
    // user.proto
    syntax = "proto3";

    service UserService {
        rpc GetUser (UserRequest) returns (UserResponse);
        rpc ListUsers (ListUsersRequest) returns (ListUsersResponse);
        rpc CreateUser (CreateUserRequest) returns (UserResponse);
    }

    message UserRequest {
        int32 id = 1;
    }

    message UserResponse {
        int32 id = 1;
        string name = 2;
        string email = 3;
    }
    ```

=== "Implementation"
    **gRPC server example:**

    ```javascript
    // Node.js server
    const grpc = require('@grpc/grpc-js');
    const protoLoader = require('@grpc/proto-loader');

    // Load proto file
    const packageDefinition = protoLoader.loadSync('user.proto');
    const userProto = grpc.loadPackageDefinition(packageDefinition);

    // Implement service
    const userService = {
        getUser: (call, callback) => {
            const userId = call.request.id;
            const user = database.users.findById(userId);
            callback(null, user);
        },
        listUsers: (call, callback) => {
            const users = database.users.findAll();
            callback(null, { users });
        },
        createUser: (call, callback) => {
            const { name, email } = call.request;
            const user = database.users.create({ name, email });
            callback(null, user);
        }
    };

    // Start server
    const server = new grpc.Server();
    server.addService(userProto.UserService.service, userService);
    server.bindAsync(
        '0.0.0.0:50051',
        grpc.ServerCredentials.createInsecure(),
        () => server.start()
    );
    ```

    **gRPC client:**
    ```javascript
    const client = new userProto.UserService(
        'localhost:50051',
        grpc.credentials.createInsecure()
    );

    // Call RPC method
    client.getUser({ id: 123 }, (error, user) => {
        if (error) console.error(error);
        else console.log(user);
    });
    ```

=== "Streaming"
    **4 types of RPCs:**

    ```protobuf
    service StreamService {
        // Unary (request-response)
        rpc GetUser (UserRequest) returns (UserResponse);

        // Server streaming (one request, stream of responses)
        rpc ListUsers (ListRequest) returns (stream User);

        // Client streaming (stream of requests, one response)
        rpc UploadLogs (stream LogEntry) returns (UploadResponse);

        // Bidirectional streaming
        rpc Chat (stream Message) returns (stream Message);
    }
    ```

    **Example: Server streaming**
    ```javascript
    // Server sends stream of results
    listUsers: (call) => {
        const users = database.users.findAll();
        for (const user of users) {
            call.write(user); // Stream each user
        }
        call.end();
    }

    // Client receives stream
    const call = client.listUsers({});
    call.on('data', (user) => {
        console.log('Received user:', user);
    });
    call.on('end', () => {
        console.log('Stream ended');
    });
    ```

=== "Advantages"
    **Why use gRPC:**

    - ✅ **Performance:** 7-10x faster than REST (binary protocol)
    - ✅ **Streaming:** Built-in support for streaming
    - ✅ **Code generation:** Auto-generate client/server code
    - ✅ **Type safety:** Strong typing with Protobuf
    - ✅ **Multiplexing:** HTTP/2 enables concurrent requests

    **Benchmark:**
    ```
    REST (JSON):  1000 requests/sec, 500KB/sec
    gRPC (Protobuf): 10,000 requests/sec, 100KB/sec
    ```

=== "Disadvantages"
    **gRPC limitations:**

    - ❌ **Browser support:** Limited (needs grpc-web)
    - ❌ **Debugging:** Binary format hard to inspect
    - ❌ **Learning curve:** Protobuf syntax
    - ❌ **Human-readability:** Not JSON (can't curl easily)

    **When NOT to use:**
    - Public APIs (use REST)
    - Browser clients (use REST/GraphQL)
    - Simple CRUD (REST is simpler)

---

## WebSocket

=== "What is WebSocket?"
    **Full-duplex communication channel over TCP**

    **Key Feature:** Bidirectional, real-time communication

    ```
    HTTP:      Client → Server → Client (request-response)
    WebSocket: Client ↔ Server (persistent connection)
    ```

=== "Use Cases"
    **When to use WebSocket:**

    | Use Case | Why WebSocket? |
    |----------|---------------|
    | **Chat applications** | Real-time bidirectional messages |
    | **Live updates** | Push updates to clients instantly |
    | **Collaborative editing** | Google Docs-style collaboration |
    | **Gaming** | Low-latency game state updates |
    | **Stock tickers** | Real-time price updates |

=== "Implementation"
    **WebSocket server:**

    ```javascript
    const WebSocket = require('ws');
    const wss = new WebSocket.Server({ port: 8080 });

    // Handle connections
    wss.on('connection', (ws) => {
        console.log('Client connected');

        // Handle messages from client
        ws.on('message', (message) => {
            console.log('Received:', message);
            
            // Broadcast to all clients
            wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(message);
                }
            });
        });

        // Handle disconnect
        ws.on('close', () => {
            console.log('Client disconnected');
        });

        // Send welcome message
        ws.send('Welcome to the chat!');
    });
    ```

    **WebSocket client:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8080');

    // Connection opened
    ws.addEventListener('open', (event) => {
        ws.send('Hello Server!');
    });

    // Listen for messages
    ws.addEventListener('message', (event) => {
        console.log('Message from server:', event.data);
    });

    // Send message
    document.getElementById('sendBtn').addEventListener('click', () => {
        ws.send('Hello!');
    });
    ```

=== "Advantages"
    - ✅ **Real-time:** Instant bidirectional communication
    - ✅ **Low latency:** No HTTP overhead
    - ✅ **Efficient:** Single connection for many messages
    - ✅ **Push-based:** Server can initiate communication

=== "Disadvantages"
    - ❌ **Complexity:** Harder than HTTP
    - ❌ **Scaling:** Requires stateful servers
    - ❌ **Load balancing:** Need sticky sessions
    - ❌ **Debugging:** Harder to inspect traffic

---

## Decision Framework

=== "Choose API Type"
    ```
    Start here:

    ┌─ Real-time bidirectional communication needed?
    │
    ├─ Yes → WebSocket
    │         Examples: Chat, gaming, live collaboration
    │
    └─ No ──┬─ Internal microservices?
            │
            ├─ Yes → gRPC
            │         High performance, type safety
            │
            └─ No ──┬─ Complex data fetching from client?
                    │
                    ├─ Yes → GraphQL
                    │         Flexible queries, reduce requests
                    │
                    └─ No → REST
                              Simple, widely understood
    ```

=== "Performance Comparison"
    **Latency & Throughput:**

    | API Type | Latency | Throughput | Payload Size |
    |----------|---------|------------|--------------|
    | **WebSocket** | < 1ms | Very High | Small |
    | **gRPC** | 1-5ms | Very High | Small (binary) |
    | **GraphQL** | 10-50ms | Medium | Medium (JSON) |
    | **REST** | 10-100ms | Medium | Medium (JSON) |

=== "Use Case Matrix"
    | Scenario | Best Choice | Why? |
    |----------|-------------|------|
    | **Public API** | REST | Widely understood, easy to use |
    | **Mobile app** | GraphQL or REST | Flexible queries or simplicity |
    | **Microservices** | gRPC | Performance, type safety |
    | **Real-time chat** | WebSocket | Bidirectional, low latency |
    | **Admin dashboard** | REST or GraphQL | CRUD operations |
    | **IoT devices** | gRPC or MQTT | Efficient binary protocol |

---

## Interview Talking Points

**Q: When would you choose GraphQL over REST?**

✅ **Strong Answer:**
> "I'd choose GraphQL when the client needs flexible data fetching or when the API powers multiple clients (web, mobile, desktop) with different data needs. For example, a mobile app might want minimal data to save bandwidth, while a desktop app wants more details - GraphQL lets both use one API with different queries. However, I'd use REST for simple CRUD APIs or public APIs where simplicity matters more than flexibility. GraphQL adds complexity (caching, query cost analysis) that isn't worth it for straightforward use cases."

**Q: Why is gRPC faster than REST?**

✅ **Strong Answer:**
> "gRPC is faster because it uses Protocol Buffers (binary format) instead of JSON, which is smaller and faster to serialize/deserialize. It also uses HTTP/2 which enables multiplexing - multiple requests over a single TCP connection. In benchmarks, gRPC can be 7-10x faster than REST. For example, a JSON payload might be 500 bytes while the equivalent Protobuf message is 50 bytes. However, gRPC isn't suitable for public APIs or browser clients without grpc-web, so REST remains better for those use cases despite being slower."

---

## Related Topics

- [Microservices Architecture](../../architecture/microservices.md) - Service communication
- [API Gateway](../../../networking/index.md) - API routing
- [Rate Limiting](../../security/api-security.md) - Protect APIs
- [Caching](../../data/caching/index.md) - Cache API responses

---

**Good API design is an investment that pays off forever! 🔌**
