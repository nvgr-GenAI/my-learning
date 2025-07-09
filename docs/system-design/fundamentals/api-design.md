# API Design: Building Robust Interfaces

## 🎯 What is API Design?

API (Application Programming Interface) design is the process of creating interfaces that enable different software systems to communicate effectively. Good API design makes your system easy to use, understand, and integrate with other systems.

## 🏛️ REST API Design Principles

### Richardson Maturity Model
**The four levels of REST API maturity**

**Level 0: The Swamp of POX (Plain Old XML)**
- Single endpoint for all operations
- All communication via POST
- No use of HTTP methods or status codes

**Level 1: Resources**
- Multiple endpoints for different resources
- Still using POST for everything
- Basic resource identification

**Level 2: HTTP Verbs**
- Proper use of HTTP methods (GET, POST, PUT, DELETE)
- Appropriate HTTP status codes
- Resource-based URLs

**Level 3: Hypermedia Controls (HATEOAS)**
- Self-descriptive messages
- Links to related resources
- Discoverability through hypermedia

### RESTful URL Design
**Best practices for resource naming**

**Good URL patterns**:
```
GET /users                    # Get all users
GET /users/123               # Get specific user
POST /users                  # Create new user
PUT /users/123               # Update user
DELETE /users/123            # Delete user

GET /users/123/orders        # Get orders for user
POST /users/123/orders       # Create order for user
GET /orders/456              # Get specific order
```

**Avoid these patterns**:
```
POST /getUsers               # Use GET instead
GET /users/delete/123        # Use DELETE instead
POST /users/123/updateName   # Use PUT instead
```

**URL Design Guidelines**:
- Use nouns, not verbs
- Use plural nouns for collections
- Use hierarchical structure for relationships
- Keep URLs predictable and consistent
- Use query parameters for filtering and pagination

### HTTP Methods and Status Codes
**Proper usage of HTTP semantics**

**HTTP Methods**:
- **GET**: Retrieve data (safe and idempotent)
- **POST**: Create new resources
- **PUT**: Update existing resources (idempotent)
- **PATCH**: Partial updates
- **DELETE**: Remove resources (idempotent)

**Status Codes**:
- **2xx Success**: 200 OK, 201 Created, 204 No Content
- **3xx Redirection**: 301 Moved Permanently, 304 Not Modified
- **4xx Client Error**: 400 Bad Request, 401 Unauthorized, 404 Not Found
- **5xx Server Error**: 500 Internal Server Error, 503 Service Unavailable

## 🔒 Authentication and Authorization

### Authentication Methods
**Verifying user identity**

**API Keys**:
```
GET /api/users
Authorization: Bearer your-api-key-here
```

**Use cases**:
- Simple service-to-service communication
- Rate limiting and usage tracking
- Basic access control

**JWT (JSON Web Tokens)**:
```
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user123",
    "exp": 1640995200,
    "iat": 1640908800,
    "roles": ["user", "admin"]
  }
}
```

**Benefits**:
- Stateless authentication
- Contains user information
- Can be verified without database lookup
- Supports expiration

**OAuth 2.0**:
```
1. User clicks "Login with Google"
2. Redirect to Google authorization server
3. User grants permission
4. Google returns authorization code
5. Exchange code for access token
6. Use token to access protected resources
```

### Authorization Patterns
**Controlling access to resources**

**Role-Based Access Control (RBAC)**:
```
User → Role → Permissions
├── Admin → Full access
├── Manager → Read/Write specific resources
└── User → Read-only access
```

**Attribute-Based Access Control (ABAC)**:
```
Decision based on:
- User attributes (department, clearance level)
- Resource attributes (classification, owner)
- Environment attributes (time, location)
- Action attributes (read, write, delete)
```

## 🚦 Rate Limiting and Throttling

### Rate Limiting Algorithms
**Control request frequency**

**Token Bucket Algorithm**:
```
Bucket capacity: 100 tokens
Refill rate: 10 tokens/second
Request consumes: 1 token

if (bucket.tokens >= 1) {
    bucket.tokens -= 1
    processRequest()
} else {
    rejectRequest(429, "Too Many Requests")
}
```

**Sliding Window**:
```
Track requests in time windows
Window size: 1 minute
Max requests: 100

current_window_count = countRequestsInWindow()
if (current_window_count < 100) {
    processRequest()
} else {
    rejectRequest()
}
```

**Fixed Window**:
```
Time window: 00:00-00:59
Max requests: 100
Reset at: 01:00

if (requests_this_minute < 100) {
    processRequest()
} else {
    rejectRequest()
}
```

### Rate Limiting Headers
**Communicate limits to clients**

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640908800
X-RateLimit-Window: 3600
```

## 🔄 Versioning Strategies

### URL Versioning
**Version in the URL path**

```
https://api.example.com/v1/users
https://api.example.com/v2/users
```

**Advantages**:
- Clear and explicit
- Easy to cache
- Simple to route

**Disadvantages**:
- Multiple URLs for same resource
- Breaks REST principles

### Header Versioning
**Version in HTTP headers**

```
GET /users
Accept: application/vnd.api+json;version=1
```

**Advantages**:
- Clean URLs
- Follows REST principles
- Flexible content negotiation

**Disadvantages**:
- Less visible
- Harder to test manually

### Query Parameter Versioning
**Version as query parameter**

```
GET /users?version=1
```

**Advantages**:
- Simple to implement
- Easy to default to latest version

**Disadvantages**:
- Can be ignored by clients
- Clutters URLs

### Content Negotiation
**Version through Accept headers**

```
GET /users
Accept: application/vnd.example.v1+json
```

## 📝 Request and Response Design

### Request Design
**Structure input data effectively**

**JSON Request Example**:
```json
POST /users
Content-Type: application/json

{
  "firstName": "John",
  "lastName": "Doe",
  "email": "john@example.com",
  "dateOfBirth": "1990-01-15",
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zipCode": "10001"
  }
}
```

**Best practices**:
- Use consistent naming conventions (camelCase or snake_case)
- Validate input data
- Provide clear error messages
- Support partial updates with PATCH

### Response Design
**Structure output data consistently**

**Success Response**:
```json
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": 123,
  "firstName": "John",
  "lastName": "Doe",
  "email": "john@example.com",
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-15T10:30:00Z"
}
```

**Error Response**:
```json
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
}
```

### Pagination
**Handle large datasets efficiently**

**Offset-based pagination**:
```json
GET /users?page=2&limit=10

{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 10,
    "total": 1000,
    "totalPages": 100
  }
}
```

**Cursor-based pagination**:
```json
GET /users?cursor=eyJpZCI6MTIz&limit=10

{
  "data": [...],
  "pagination": {
    "cursor": "eyJpZCI6MTMz",
    "hasMore": true,
    "limit": 10
  }
}
```

## 🌐 API Gateway Pattern

### What is an API Gateway?
**Single entry point for all API requests**

**Responsibilities**:
- Request routing to appropriate services
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Monitoring and analytics
- API versioning and deprecation

**Architecture**:
```
Client Apps → API Gateway → Microservices
                    ├── User Service
                    ├── Product Service
                    ├── Order Service
                    └── Payment Service
```

### Benefits of API Gateway
**Centralized cross-cutting concerns**

**For Clients**:
- Single endpoint to remember
- Consistent API experience
- Built-in rate limiting
- Unified authentication

**For Services**:
- Simplified service interfaces
- Centralized logging and monitoring
- Load balancing and failover
- A/B testing capabilities

**For Operations**:
- Centralized configuration
- Easy API analytics
- Version management
- Security policies

### API Gateway Features
**Common capabilities**

**Request Routing**:
```
/api/v1/users/* → User Service
/api/v1/products/* → Product Service
/api/v1/orders/* → Order Service
```

**Request Transformation**:
```
Client Request:
GET /api/v1/user-profile

Gateway transforms to:
GET /users/123/profile
Authorization: Bearer token
```

**Response Aggregation**:
```
Client requests: GET /api/v1/dashboard

Gateway fetches:
- User info from User Service
- Recent orders from Order Service
- Recommendations from Product Service

Returns combined response
```

## 🔧 GraphQL vs REST

### GraphQL Advantages
**Flexible data fetching**

**Single Request for Multiple Resources**:
```graphql
query {
  user(id: "123") {
    name
    email
    orders {
      id
      total
      items {
        product {
          name
          price
        }
      }
    }
  }
}
```

**Type Safety**:
```graphql
type User {
  id: ID!
  name: String!
  email: String!
  orders: [Order!]!
}

type Order {
  id: ID!
  total: Float!
  items: [OrderItem!]!
}
```

### When to Use GraphQL
**Ideal scenarios**:
- Complex data requirements
- Multiple client types (mobile, web, desktop)
- Rapidly changing requirements
- Development teams want flexibility

### When to Use REST
**Traditional scenarios**:
- Simple CRUD operations
- Caching is important
- Team familiar with REST
- Integration with existing systems

## 🔍 API Documentation

### OpenAPI/Swagger Specification
**Describe your API structure**

```yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: API for managing users

paths:
  /users:
    get:
      summary: Get all users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/User'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string
          format: email
      required:
        - id
        - name
        - email
```

### Documentation Best Practices
**Make your API easy to understand**

**Essential Elements**:
- Clear endpoint descriptions
- Request/response examples
- Error code explanations
- Authentication requirements
- Rate limiting information

**Interactive Documentation**:
- Try-it-out functionality
- Code samples in multiple languages
- Real-time API testing
- Postman collections

## 🚀 API Testing Strategies

### Testing Levels
**Comprehensive API testing**

**Unit Tests**:
```javascript
describe('User API', () => {
  it('should create a user', async () => {
    const userData = {
      name: 'John Doe',
      email: 'john@example.com'
    };
    
    const response = await request(app)
      .post('/users')
      .send(userData)
      .expect(201);
      
    expect(response.body.name).toBe(userData.name);
    expect(response.body.email).toBe(userData.email);
  });
});
```

**Integration Tests**:
```javascript
describe('User Integration', () => {
  it('should handle complete user workflow', async () => {
    // Create user
    const user = await createUser(userData);
    
    // Get user
    const fetchedUser = await getUser(user.id);
    
    // Update user
    const updatedUser = await updateUser(user.id, updates);
    
    // Delete user
    await deleteUser(user.id);
  });
});
```

**Contract Tests**:
```javascript
// Using Pact for contract testing
describe('User Service Contract', () => {
  it('should return user data in expected format', async () => {
    await provider
      .given('User exists')
      .uponReceiving('A request for user data')
      .withRequest({
        method: 'GET',
        path: '/users/123'
      })
      .willRespondWith({
        status: 200,
        headers: {
          'Content-Type': 'application/json'
        },
        body: {
          id: 123,
          name: 'John Doe',
          email: 'john@example.com'
        }
      });
  });
});
```

## 🎯 Best Practices Summary

### Design Principles
1. **Consistency**: Use consistent naming and patterns
2. **Simplicity**: Keep interfaces simple and intuitive
3. **Flexibility**: Design for future changes
4. **Performance**: Optimize for common use cases
5. **Security**: Implement proper authentication and authorization

### Common Mistakes to Avoid
1. **Exposing internal structure**: Keep implementation details hidden
2. **Ignoring HTTP semantics**: Use proper methods and status codes
3. **Inconsistent error handling**: Standardize error responses
4. **Poor documentation**: Keep documentation up to date
5. **No versioning strategy**: Plan for API evolution

### Success Metrics
- **Developer Experience**: Easy to understand and use
- **Performance**: Fast response times
- **Reliability**: High uptime and error handling
- **Scalability**: Handles growing load
- **Security**: Protects against common attacks

Good API design is crucial for system integration, developer productivity, and long-term maintainability. Invest time in getting it right from the start.
