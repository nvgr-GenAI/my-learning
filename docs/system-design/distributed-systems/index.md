# Distributed Systems 🔗

Learn the principles and patterns for building systems that span multiple machines and networks. This section covers microservices, consistency models, consensus algorithms, and distributed system challenges.

## 🎯 What You'll Learn

Distributed systems introduce unique challenges and opportunities. Master the concepts and patterns needed to build reliable, scalable systems that work across multiple machines, data centers, and even geographic regions.

<div class="grid cards" markdown>

- :material-hexagon-multiple: **Microservices**

    ---

    Service decomposition, communication patterns, and orchestration

    [Design microservices →](microservices.md)

- :material-sync: **Consistency Models**

    ---

    ACID, BASE, eventual consistency, and consensus algorithms

    [Ensure consistency →](consistency.md)
    
    Service decomposition, boundaries, communication patterns
    
    [Design microservices →](microservices.md)

-   :material-message-arrow-right: **Service Communication**
    
    ---
    
    REST, gRPC, message queues, event streaming
    
    [Connect services →](communication.md)

-   :material-api: **API Design & Gateway**
    
    ---
    
    API patterns, versioning, rate limiting, API gateways
    
    [Design APIs →](api-design.md)

-   :material-message-processing: **Message Queues & Streaming**
    
    ---
    
    Async messaging, event sourcing, pub/sub patterns
    
    [Handle messages →](messaging.md)

-   :material-animation: **Event-Driven Architecture**
    
    ---
    
    Event sourcing, CQRS, saga patterns, choreography
    
    [Build event-driven →](event-driven.md)

-   :material-radar: **Service Discovery & Config**
    
    ---
    
    Service registry, configuration management, service mesh
    
    [Manage services →](service-management.md)

</div>

## 🏗️ Microservices vs Monolith

### Architecture Comparison

| Aspect | Monolith | Microservices |
|--------|----------|---------------|
| **Deployment** | Single deployable unit | Independent deployments |
| **Scaling** | Scale entire application | Scale individual services |
| **Technology** | Single tech stack | Polyglot programming |
| **Team Structure** | Single team | Multiple small teams |
| **Complexity** | Simple initially | Complex distributed system |
| **Data Management** | Shared database | Database per service |

### When to Choose Each

```python
class ArchitectureDecision:
    """Framework for choosing architecture"""
    
    def choose_monolith_when(self):
        """Scenarios where monolith is better"""
        return {
            'team_size': 'Small team (< 10 developers)',
            'system_complexity': 'Simple business logic',
            'performance_requirements': 'Low latency critical',
            'deployment_frequency': 'Infrequent deployments',
            'operational_maturity': 'Limited DevOps experience',
            'startup_phase': 'Early stage, rapid prototyping'
        }
    
    def choose_microservices_when(self):
        """Scenarios where microservices are better"""
        return {
            'team_size': 'Large organization (multiple teams)',
            'system_complexity': 'Complex business domains',
            'scalability_requirements': 'Different scaling needs per service',
            'technology_diversity': 'Need for different technologies',
            'deployment_frequency': 'Frequent, independent deployments',
            'fault_isolation': 'Need to isolate failures'
        }
    
    def hybrid_approach(self):
        """Gradual migration strategy"""
        return {
            'start_with': 'Modular monolith',
            'extract_services': 'Extract services when boundaries are clear',
            'strangler_pattern': 'Gradually replace monolith components',
            'database_decomposition': 'Split shared database last'
        }
```

## 🔄 Service Communication Patterns

### Synchronous Communication

```python
import requests
import grpc
from typing import Optional, Dict, Any

class SynchronousCommunication:
    """Synchronous service communication patterns"""
    
    def __init__(self):
        self.timeout = 30  # seconds
        self.retry_attempts = 3
    
    def rest_api_call(self, service_url: str, endpoint: str, data: Optional[Dict] = None):
        """HTTP REST API communication"""
        url = f"{service_url}/{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                if data:
                    response = requests.post(url, json=data, timeout=self.timeout)
                else:
                    response = requests.get(url, timeout=self.timeout)
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt == self.retry_attempts - 1:
                    raise
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                if attempt == self.retry_attempts - 1:
                    raise
    
    def grpc_call(self, stub, method_name: str, request):
        """gRPC service communication"""
        try:
            method = getattr(stub, method_name)
            response = method(request, timeout=self.timeout)
            return response
        except grpc.RpcError as e:
            print(f"gRPC call failed: {e.code()} - {e.details()}")
            raise
    
    def circuit_breaker_call(self, service_call, fallback_response=None):
        """Circuit breaker pattern for service calls"""
        try:
            return service_call()
        except Exception as e:
            print(f"Service call failed: {e}")
            if fallback_response:
                return fallback_response
            raise
```

### Asynchronous Communication

```python
import asyncio
import json
from typing import Callable, Dict, Any
from datetime import datetime

class AsynchronousCommunication:
    """Asynchronous service communication using message queues"""
    
    def __init__(self, message_broker):
        self.broker = message_broker
        self.event_handlers = {}
    
    def publish_event(self, event_type: str, event_data: Dict[Any, Any]):
        """Publish event to message broker"""
        event = {
            'type': event_type,
            'data': event_data,
            'timestamp': datetime.utcnow().isoformat(),
            'id': self._generate_event_id()
        }
        
        self.broker.publish(event_type, json.dumps(event))
        print(f"Published event: {event_type}")
    
    def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.broker.subscribe(event_type, self._handle_message)
    
    def _handle_message(self, message_body: str):
        """Handle incoming message"""
        try:
            event = json.loads(message_body)
            event_type = event['type']
            
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    # Handle asynchronously to avoid blocking
                    asyncio.create_task(handler(event))
            
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return str(uuid.uuid4())

# Example usage
class OrderService:
    def __init__(self, message_system):
        self.messaging = message_system
        
        # Subscribe to events
        self.messaging.subscribe_to_event('payment_completed', self.handle_payment_completed)
        self.messaging.subscribe_to_event('inventory_reserved', self.handle_inventory_reserved)
    
    def create_order(self, order_data):
        """Create order and publish events"""
        order_id = self._save_order(order_data)
        
        # Publish order created event
        self.messaging.publish_event('order_created', {
            'order_id': order_id,
            'user_id': order_data['user_id'],
            'items': order_data['items'],
            'total_amount': order_data['total']
        })
        
        return order_id
    
    async def handle_payment_completed(self, event):
        """Handle payment completion event"""
        order_id = event['data']['order_id']
        print(f"Payment completed for order {order_id}")
        
        # Update order status
        self._update_order_status(order_id, 'paid')
        
        # Publish next event
        self.messaging.publish_event('order_confirmed', {
            'order_id': order_id
        })
    
    async def handle_inventory_reserved(self, event):
        """Handle inventory reservation event"""
        order_id = event['data']['order_id']
        print(f"Inventory reserved for order {order_id}")
        
        # Proceed with order processing
        self._update_order_status(order_id, 'processing')
```

## 🔌 API Design Patterns

### RESTful API Design

```python
from flask import Flask, request, jsonify
from typing import Dict, List, Optional

class RESTfulAPIDesign:
    """RESTful API design best practices"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup RESTful routes"""
        # Resource-based URLs
        self.app.route('/api/v1/users', methods=['GET'])(self.get_users)
        self.app.route('/api/v1/users', methods=['POST'])(self.create_user)
        self.app.route('/api/v1/users/<int:user_id>', methods=['GET'])(self.get_user)
        self.app.route('/api/v1/users/<int:user_id>', methods=['PUT'])(self.update_user)
        self.app.route('/api/v1/users/<int:user_id>', methods=['DELETE'])(self.delete_user)
        
        # Nested resources
        self.app.route('/api/v1/users/<int:user_id>/orders', methods=['GET'])(self.get_user_orders)
        self.app.route('/api/v1/users/<int:user_id>/orders', methods=['POST'])(self.create_user_order)
    
    def get_users(self):
        """GET /api/v1/users - List users with pagination"""
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        
        # Pagination
        offset = (page - 1) * limit
        users = self._get_users_from_db(offset, limit)
        total = self._count_users()
        
        return jsonify({
            'data': users,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'pages': (total + limit - 1) // limit
            }
        })
    
    def create_user(self):
        """POST /api/v1/users - Create new user"""
        data = request.json
        
        # Validation
        if not data or not data.get('email'):
            return jsonify({'error': 'Email is required'}), 400
        
        try:
            user_id = self._create_user_in_db(data)
            return jsonify({
                'id': user_id,
                'message': 'User created successfully'
            }), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def get_user(self, user_id: int):
        """GET /api/v1/users/{id} - Get specific user"""
        user = self._get_user_from_db(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'data': user})
    
    def api_versioning_strategy(self):
        """API versioning strategies"""
        return {
            'url_versioning': '/api/v1/users vs /api/v2/users',
            'header_versioning': 'Accept: application/vnd.api+json;version=1',
            'parameter_versioning': '/api/users?version=1',
            'content_negotiation': 'Accept: application/vnd.company.user-v1+json'
        }
    
    def error_handling_standards(self):
        """Standardized error responses"""
        return {
            'error_format': {
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': 'Invalid input data',
                    'details': [
                        {
                            'field': 'email',
                            'message': 'Email is required'
                        }
                    ]
                }
            },
            'http_status_codes': {
                200: 'Success',
                201: 'Created',
                400: 'Bad Request',
                401: 'Unauthorized',
                403: 'Forbidden',
                404: 'Not Found',
                500: 'Internal Server Error'
            }
        }
```

### GraphQL API Design

```python
import graphene
from graphene import ObjectType, String, Int, List, Field

class GraphQLAPIDesign:
    """GraphQL API design patterns"""
    
    class User(ObjectType):
        id = Int()
        name = String()
        email = String()
        orders = List(lambda: Order)
        
        def resolve_orders(self, info):
            # Lazy loading of related data
            return get_orders_for_user(self.id)
    
    class Order(ObjectType):
        id = Int()
        user_id = Int()
        total = String()
        status = String()
        user = Field(User)
        
        def resolve_user(self, info):
            return get_user_by_id(self.user_id)
    
    class Query(ObjectType):
        users = List(User)
        user = Field(User, id=Int(required=True))
        orders = List(Order, status=String())
        
        def resolve_users(self, info):
            return get_all_users()
        
        def resolve_user(self, info, id):
            return get_user_by_id(id)
        
        def resolve_orders(self, info, status=None):
            if status:
                return get_orders_by_status(status)
            return get_all_orders()
    
    class CreateUser(graphene.Mutation):
        class Arguments:
            name = String(required=True)
            email = String(required=True)
        
        user = Field(User)
        success = String()
        
        def mutate(self, info, name, email):
            user = create_user(name, email)
            return CreateUser(user=user, success="User created successfully")
    
    class Mutation(ObjectType):
        create_user = CreateUser.Field()
    
    def schema_definition(self):
        """Complete GraphQL schema"""
        return graphene.Schema(
            query=self.Query,
            mutation=self.Mutation
        )
    
    def advantages_over_rest(self):
        """GraphQL advantages"""
        return {
            'single_endpoint': 'One endpoint for all operations',
            'flexible_queries': 'Client specifies exactly what data needed',
            'no_over_fetching': 'Reduces data transfer',
            'strong_typing': 'Schema defines exact data structure',
            'introspection': 'Self-documenting API'
        }
```

## 📨 Message Queue Patterns

### Message Queue Implementation

```python
import asyncio
import json
import time
from enum import Enum
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Message:
    id: str
    type: str
    payload: Dict[str, Any]
    priority: MessagePriority
    created_at: float
    retry_count: int = 0
    max_retries: int = 3

class MessageQueue:
    """Advanced message queue implementation"""
    
    def __init__(self):
        self.queues = {}  # topic -> priority queue
        self.consumers = {}  # topic -> list of consumers
        self.dead_letter_queue = []
        self.processing_messages = set()
    
    def publish(self, topic: str, message: Message):
        """Publish message to topic"""
        if topic not in self.queues:
            self.queues[topic] = []
        
        # Insert message based on priority
        self.queues[topic].append(message)
        self.queues[topic].sort(key=lambda m: m.priority.value, reverse=True)
        
        print(f"Published message {message.id} to topic {topic}")
    
    def subscribe(self, topic: str, consumer: Callable[[Message], bool]):
        """Subscribe consumer to topic"""
        if topic not in self.consumers:
            self.consumers[topic] = []
        
        self.consumers[topic].append(consumer)
        
        # Start processing messages for this topic
        asyncio.create_task(self._process_topic(topic))
    
    async def _process_topic(self, topic: str):
        """Process messages for a specific topic"""
        while True:
            if topic in self.queues and self.queues[topic]:
                message = self.queues[topic].pop(0)
                
                if message.id not in self.processing_messages:
                    self.processing_messages.add(message.id)
                    await self._process_message(topic, message)
                    self.processing_messages.discard(message.id)
            
            await asyncio.sleep(0.1)  # Prevent busy waiting
    
    async def _process_message(self, topic: str, message: Message):
        """Process individual message"""
        consumers = self.consumers.get(topic, [])
        
        for consumer in consumers:
            try:
                success = await self._call_consumer(consumer, message)
                
                if success:
                    print(f"Message {message.id} processed successfully")
                    return
                else:
                    await self._handle_processing_failure(topic, message)
                    
            except Exception as e:
                print(f"Consumer error: {e}")
                await self._handle_processing_failure(topic, message)
    
    async def _call_consumer(self, consumer: Callable, message: Message) -> bool:
        """Call consumer function safely"""
        try:
            if asyncio.iscoroutinefunction(consumer):
                return await consumer(message)
            else:
                return consumer(message)
        except Exception:
            return False
    
    async def _handle_processing_failure(self, topic: str, message: Message):
        """Handle message processing failure"""
        message.retry_count += 1
        
        if message.retry_count <= message.max_retries:
            # Retry with exponential backoff
            delay = 2 ** message.retry_count
            await asyncio.sleep(delay)
            
            # Re-queue message
            self.publish(topic, message)
            print(f"Retrying message {message.id}, attempt {message.retry_count}")
        else:
            # Move to dead letter queue
            self.dead_letter_queue.append(message)
            print(f"Message {message.id} moved to dead letter queue")

# Example usage
class OrderProcessingService:
    def __init__(self, message_queue: MessageQueue):
        self.mq = message_queue
        self.setup_consumers()
    
    def setup_consumers(self):
        """Setup message consumers"""
        self.mq.subscribe('order_created', self.process_order_created)
        self.mq.subscribe('payment_completed', self.process_payment_completed)
        self.mq.subscribe('order_shipped', self.process_order_shipped)
    
    async def process_order_created(self, message: Message) -> bool:
        """Process order creation"""
        try:
            order_data = message.payload
            print(f"Processing order creation: {order_data['order_id']}")
            
            # Business logic here
            await self._validate_order(order_data)
            await self._reserve_inventory(order_data)
            await self._process_payment(order_data)
            
            return True
        except Exception as e:
            print(f"Order processing failed: {e}")
            return False
    
    async def process_payment_completed(self, message: Message) -> bool:
        """Process payment completion"""
        try:
            payment_data = message.payload
            print(f"Processing payment completion: {payment_data['order_id']}")
            
            # Update order status
            await self._update_order_status(payment_data['order_id'], 'paid')
            
            # Trigger fulfillment
            await self._trigger_fulfillment(payment_data['order_id'])
            
            return True
        except Exception as e:
            print(f"Payment processing failed: {e}")
            return False
```

## 🎯 Distributed System Challenges

### Common Challenges and Solutions

```python
class DistributedSystemChallenges:
    """Common challenges in distributed systems and their solutions"""
    
    def network_partitions(self):
        """Handle network partition scenarios"""
        return {
            'problem': 'Services cannot communicate due to network failure',
            'solutions': [
                'Implement circuit breakers',
                'Use timeout and retry mechanisms',
                'Design for partition tolerance (CAP theorem)',
                'Implement graceful degradation'
            ],
            'example': 'Service mesh with retry policies'
        }
    
    def distributed_transactions(self):
        """Handle transactions across multiple services"""
        return {
            'problem': 'Maintaining consistency across multiple services',
            'solutions': [
                'Saga pattern for long-running transactions',
                'Two-phase commit (2PC) for strong consistency',
                'Event sourcing for audit trail',
                'Compensating actions for rollbacks'
            ],
            'trade_offs': 'Consistency vs availability vs performance'
        }
    
    def service_discovery(self):
        """Services finding and communicating with each other"""
        return {
            'problem': 'Services need to locate other services dynamically',
            'solutions': [
                'Service registry (Consul, Eureka)',
                'DNS-based discovery',
                'Load balancer with health checks',
                'Service mesh (Istio, Linkerd)'
            ],
            'considerations': 'Health checks, failover, load balancing'
        }
    
    def distributed_consensus(self):
        """Achieving agreement in distributed systems"""
        return {
            'problem': 'Multiple nodes need to agree on a value',
            'algorithms': [
                'Raft consensus algorithm',
                'PBFT (Practical Byzantine Fault Tolerance)',
                'Paxos algorithm',
                'SWIM (Scalable Weakly-consistent Infection-style Process Group Membership)'
            ],
            'use_cases': 'Leader election, configuration management'
        }
```

### Saga Pattern Implementation

```python
import asyncio
from enum import Enum
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

class SagaStepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    name: str
    action: Callable
    compensation: Callable
    status: SagaStepStatus = SagaStepStatus.PENDING

class SagaOrchestrator:
    """Saga pattern implementation for distributed transactions"""
    
    def __init__(self):
        self.sagas = {}
    
    async def execute_saga(self, saga_id: str, steps: List[SagaStep]) -> bool:
        """Execute saga with compensation on failure"""
        self.sagas[saga_id] = {
            'steps': steps,
            'completed_steps': [],
            'status': 'running'
        }
        
        try:
            # Execute steps in order
            for i, step in enumerate(steps):
                print(f"Executing step {i + 1}: {step.name}")
                
                try:
                    result = await step.action()
                    step.status = SagaStepStatus.COMPLETED
                    self.sagas[saga_id]['completed_steps'].append(step)
                    
                    print(f"Step {step.name} completed successfully")
                    
                except Exception as e:
                    print(f"Step {step.name} failed: {e}")
                    step.status = SagaStepStatus.FAILED
                    
                    # Compensate all completed steps
                    await self._compensate_saga(saga_id)
                    return False
            
            self.sagas[saga_id]['status'] = 'completed'
            print(f"Saga {saga_id} completed successfully")
            return True
            
        except Exception as e:
            print(f"Saga {saga_id} failed: {e}")
            await self._compensate_saga(saga_id)
            return False
    
    async def _compensate_saga(self, saga_id: str):
        """Execute compensation for all completed steps"""
        saga = self.sagas[saga_id]
        completed_steps = saga['completed_steps']
        
        print(f"Starting compensation for saga {saga_id}")
        
        # Compensate in reverse order
        for step in reversed(completed_steps):
            try:
                await step.compensation()
                step.status = SagaStepStatus.COMPENSATED
                print(f"Compensated step: {step.name}")
                
            except Exception as e:
                print(f"Compensation failed for step {step.name}: {e}")
                # Log and alert - compensation failure is serious
        
        saga['status'] = 'compensated'

# Example: Order processing saga
class OrderProcessingSaga:
    def __init__(self, saga_orchestrator: SagaOrchestrator):
        self.orchestrator = saga_orchestrator
        self.payment_service = PaymentService()
        self.inventory_service = InventoryService()
        self.shipping_service = ShippingService()
    
    async def process_order(self, order_data: Dict[str, Any]) -> bool:
        """Process order using saga pattern"""
        order_id = order_data['order_id']
        
        # Define saga steps
        steps = [
            SagaStep(
                name="reserve_inventory",
                action=lambda: self.inventory_service.reserve_items(order_data['items']),
                compensation=lambda: self.inventory_service.release_items(order_data['items'])
            ),
            SagaStep(
                name="process_payment",
                action=lambda: self.payment_service.charge_card(order_data['payment_info']),
                compensation=lambda: self.payment_service.refund_payment(order_data['payment_info'])
            ),
            SagaStep(
                name="create_shipment",
                action=lambda: self.shipping_service.create_shipment(order_data),
                compensation=lambda: self.shipping_service.cancel_shipment(order_data['order_id'])
            )
        ]
        
        # Execute saga
        return await self.orchestrator.execute_saga(f"order_{order_id}", steps)

# Mock services for example
class PaymentService:
    async def charge_card(self, payment_info):
        # Simulate payment processing
        await asyncio.sleep(0.1)
        if payment_info.get('card_number') == 'invalid':
            raise Exception("Invalid card")
        return {"transaction_id": "txn_123"}
    
    async def refund_payment(self, payment_info):
        await asyncio.sleep(0.1)
        return {"refund_id": "ref_123"}

class InventoryService:
    async def reserve_items(self, items):
        await asyncio.sleep(0.1)
        for item in items:
            if item.get('stock', 0) <= 0:
                raise Exception(f"Item {item['id']} out of stock")
        return {"reservation_id": "res_123"}
    
    async def release_items(self, items):
        await asyncio.sleep(0.1)
        return {"released": True}

class ShippingService:
    async def create_shipment(self, order_data):
        await asyncio.sleep(0.1)
        return {"shipment_id": "ship_123"}
    
    async def cancel_shipment(self, order_id):
        await asyncio.sleep(0.1)
        return {"cancelled": True}
```

## ✅ Distributed Systems Checklist

### Design Phase
- [ ] Define service boundaries based on business domains
- [ ] Choose appropriate communication patterns (sync vs async)
- [ ] Plan for service discovery and configuration
- [ ] Design data consistency strategy
- [ ] Plan for monitoring and observability

### Implementation Phase
- [ ] Implement circuit breakers and retries
- [ ] Add comprehensive logging and tracing
- [ ] Implement health checks for all services
- [ ] Set up service-to-service authentication
- [ ] Plan deployment and rollback strategies

### Operations Phase
- [ ] Monitor service dependencies and health
- [ ] Set up alerting for critical failures
- [ ] Implement chaos engineering practices
- [ ] Plan for capacity scaling
- [ ] Regular disaster recovery testing

## 🚀 Next Steps

Ready to master distributed systems architecture?

1. **[Microservices Design](microservices.md)** - Learn service decomposition
2. **[Service Communication](communication.md)** - Master inter-service communication
3. **[API Design](api-design.md)** - Build robust APIs
4. **[Messaging Systems](messaging.md)** - Implement async messaging
5. **[Event-Driven Architecture](event-driven.md)** - Build event-driven systems
6. **[Service Management](service-management.md)** - Manage service lifecycle

---

**Build distributed systems that scale! 🌐💪**
