# Message Patterns & Event-Driven Architecture 📬

Master advanced messaging patterns and event-driven architecture for building scalable, decoupled systems. This comprehensive guide covers messaging strategies, event sourcing, and asynchronous communication patterns.

## 🎯 Core Messaging Patterns

### **1. Producer-Consumer Pattern**

**Basic Implementation**:

```python
import asyncio
import json
from typing import Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time

class MessageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Message:
    id: str
    payload: Dict[str, Any]
    timestamp: float
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

class MessageQueue:
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.dead_letter_queue = asyncio.Queue()
        self.processing_messages = {}
    
    async def produce(self, message: Message):
        """Add message to queue"""
        await self.queue.put(message)
    
    async def consume(self) -> Message:
        """Get message from queue"""
        message = await self.queue.get()
        message.status = MessageStatus.PROCESSING
        self.processing_messages[message.id] = message
        return message
    
    async def ack(self, message_id: str):
        """Acknowledge message processing"""
        if message_id in self.processing_messages:
            message = self.processing_messages.pop(message_id)
            message.status = MessageStatus.COMPLETED
    
    async def nack(self, message_id: str):
        """Negative acknowledge - retry or dead letter"""
        if message_id in self.processing_messages:
            message = self.processing_messages.pop(message_id)
            message.retry_count += 1
            
            if message.retry_count <= message.max_retries:
                message.status = MessageStatus.PENDING
                await self.queue.put(message)
            else:
                message.status = MessageStatus.FAILED
                await self.dead_letter_queue.put(message)

class AsyncConsumer:
    def __init__(self, queue: MessageQueue, processor: Callable):
        self.queue = queue
        self.processor = processor
        self.is_running = False
    
    async def start(self):
        """Start consuming messages"""
        self.is_running = True
        while self.is_running:
            try:
                message = await self.queue.consume()
                
                # Process message
                try:
                    await self.processor(message)
                    await self.queue.ack(message.id)
                except Exception as e:
                    print(f"Failed to process message {message.id}: {e}")
                    await self.queue.nack(message.id)
                    
            except asyncio.CancelledError:
                break
    
    def stop(self):
        """Stop consuming messages"""
        self.is_running = False

# Usage example
async def order_processor(message: Message):
    """Process order message"""
    order_data = message.payload
    print(f"Processing order: {order_data}")
    
    # Simulate processing time
    await asyncio.sleep(0.1)
    
    # Simulate potential failure
    if order_data.get('amount', 0) < 0:
        raise ValueError("Invalid order amount")

async def main():
    # Create queue and consumer
    queue = MessageQueue()
    consumer = AsyncConsumer(queue, order_processor)
    
    # Start consumer
    consumer_task = asyncio.create_task(consumer.start())
    
    # Produce messages
    for i in range(10):
        message = Message(
            id=f"order_{i}",
            payload={
                'order_id': f"order_{i}",
                'amount': 100 if i % 2 == 0 else -50,  # Some invalid orders
                'customer': f"customer_{i}"
            },
            timestamp=time.time()
        )
        await queue.produce(message)
    
    # Let consumer process for a bit
    await asyncio.sleep(2)
    
    # Stop consumer
    consumer.stop()
    await consumer_task
```

### **2. Request-Reply Pattern**

**Asynchronous Request-Reply**:

```python
import asyncio
import uuid
from typing import Dict, Any, Optional
import json

class RequestReplyManager:
    def __init__(self):
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.timeout_default = 30  # seconds
    
    async def send_request(self, 
                          request_data: Dict[str, Any], 
                          timeout: float = None) -> Dict[str, Any]:
        """Send request and wait for reply"""
        request_id = str(uuid.uuid4())
        timeout = timeout or self.timeout_default
        
        # Create future for this request
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request with correlation ID
        message = {
            'id': request_id,
            'data': request_data,
            'timestamp': time.time()
        }
        
        # In real implementation, send to message broker
        await self._send_to_broker(message)
        
        try:
            # Wait for reply with timeout
            reply = await asyncio.wait_for(future, timeout=timeout)
            return reply
        except asyncio.TimeoutError:
            # Cleanup on timeout
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            raise TimeoutError(f"Request {request_id} timed out")
    
    async def handle_reply(self, reply_data: Dict[str, Any]):
        """Handle incoming reply"""
        request_id = reply_data.get('correlation_id')
        
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            if not future.done():
                future.set_result(reply_data)
    
    async def _send_to_broker(self, message: Dict[str, Any]):
        """Send message to broker (mock implementation)"""
        # In real implementation, this would send to Kafka, RabbitMQ, etc.
        print(f"Sending request: {message}")
        
        # Simulate async processing and reply
        asyncio.create_task(self._simulate_reply(message))
    
    async def _simulate_reply(self, request: Dict[str, Any]):
        """Simulate service processing and reply"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        reply = {
            'correlation_id': request['id'],
            'result': f"Processed: {request['data']}",
            'timestamp': time.time()
        }
        
        await self.handle_reply(reply)

# Usage
async def example_request_reply():
    manager = RequestReplyManager()
    
    try:
        reply = await manager.send_request({
            'operation': 'get_user',
            'user_id': 'user123'
        })
        print(f"Got reply: {reply}")
    except TimeoutError as e:
        print(f"Request timed out: {e}")
```

### **3. Publish-Subscribe Pattern**

**Topic-Based Pub/Sub**:

```python
import asyncio
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import weakref

@dataclass
class Topic:
    name: str
    subscribers: List[Callable] = None
    
    def __post_init__(self):
        if self.subscribers is None:
            self.subscribers = []

class PubSubBroker:
    def __init__(self):
        self.topics: Dict[str, Topic] = {}
        self.subscriber_registry: Dict[str, List[Callable]] = {}
    
    def create_topic(self, topic_name: str) -> Topic:
        """Create a new topic"""
        if topic_name not in self.topics:
            self.topics[topic_name] = Topic(name=topic_name)
        return self.topics[topic_name]
    
    def subscribe(self, topic_name: str, callback: Callable):
        """Subscribe to a topic"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        topic = self.topics[topic_name]
        if callback not in topic.subscribers:
            topic.subscribers.append(callback)
    
    def unsubscribe(self, topic_name: str, callback: Callable):
        """Unsubscribe from a topic"""
        if topic_name in self.topics:
            topic = self.topics[topic_name]
            if callback in topic.subscribers:
                topic.subscribers.remove(callback)
    
    async def publish(self, topic_name: str, message: Dict[str, Any]):
        """Publish message to topic"""
        if topic_name not in self.topics:
            return
        
        topic = self.topics[topic_name]
        
        # Send to all subscribers
        tasks = []
        for subscriber in topic.subscribers:
            if asyncio.iscoroutinefunction(subscriber):
                tasks.append(subscriber(message))
            else:
                # Run sync function in thread pool
                tasks.append(asyncio.get_event_loop().run_in_executor(
                    None, subscriber, message
                ))
        
        # Wait for all subscribers to process
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

class EventBus:
    """High-level event bus built on pub/sub"""
    
    def __init__(self):
        self.broker = PubSubBroker()
        self.middleware: List[Callable] = []
    
    def add_middleware(self, middleware: Callable):
        """Add middleware for processing events"""
        self.middleware.append(middleware)
    
    async def emit(self, event_name: str, data: Dict[str, Any]):
        """Emit an event"""
        event = {
            'name': event_name,
            'data': data,
            'timestamp': time.time()
        }
        
        # Apply middleware
        for middleware in self.middleware:
            event = await middleware(event) if asyncio.iscoroutinefunction(middleware) else middleware(event)
        
        # Publish to broker
        await self.broker.publish(event_name, event)
    
    def on(self, event_name: str, handler: Callable):
        """Register event handler"""
        self.broker.subscribe(event_name, handler)
    
    def off(self, event_name: str, handler: Callable):
        """Unregister event handler"""
        self.broker.unsubscribe(event_name, handler)

# Usage example
async def example_pubsub():
    bus = EventBus()
    
    # Add logging middleware
    async def logging_middleware(event):
        print(f"Event: {event['name']} at {event['timestamp']}")
        return event
    
    bus.add_middleware(logging_middleware)
    
    # Register handlers
    async def user_created_handler(event):
        print(f"User created: {event['data']}")
        # Send welcome email
        await send_welcome_email(event['data']['email'])
    
    async def audit_handler(event):
        print(f"Audit: {event}")
        # Log to audit system
    
    bus.on('user.created', user_created_handler)
    bus.on('user.created', audit_handler)
    
    # Emit events
    await bus.emit('user.created', {
        'user_id': 'user123',
        'email': 'user@example.com',
        'name': 'John Doe'
    })

async def send_welcome_email(email: str):
    """Mock email sending"""
    await asyncio.sleep(0.1)
    print(f"Welcome email sent to {email}")
```

## 🔄 Event Sourcing Pattern

### **Event Store Implementation**

```python
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

@dataclass
class Event:
    id: str
    aggregate_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    version: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'aggregate_id': self.aggregate_id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'metadata': self.metadata or {}
        }

class EventStore:
    def __init__(self):
        self.events: Dict[str, List[Event]] = {}
        self.snapshots: Dict[str, Dict[str, Any]] = {}
    
    async def append_events(self, 
                           aggregate_id: str, 
                           events: List[Event], 
                           expected_version: int) -> bool:
        """Append events to stream with optimistic concurrency control"""
        if aggregate_id not in self.events:
            self.events[aggregate_id] = []
        
        stream = self.events[aggregate_id]
        
        # Check for concurrency conflicts
        current_version = len(stream)
        if current_version != expected_version:
            raise ConcurrencyError(
                f"Expected version {expected_version}, but current version is {current_version}"
            )
        
        # Append events
        for event in events:
            stream.append(event)
        
        return True
    
    async def get_events(self, 
                        aggregate_id: str, 
                        from_version: int = 0) -> List[Event]:
        """Get events for aggregate from specific version"""
        if aggregate_id not in self.events:
            return []
        
        stream = self.events[aggregate_id]
        return stream[from_version:]
    
    async def get_all_events(self, 
                           from_timestamp: Optional[datetime] = None) -> List[Event]:
        """Get all events from all streams"""
        all_events = []
        
        for stream in self.events.values():
            for event in stream:
                if from_timestamp is None or event.timestamp >= from_timestamp:
                    all_events.append(event)
        
        # Sort by timestamp
        all_events.sort(key=lambda e: e.timestamp)
        return all_events
    
    async def save_snapshot(self, 
                           aggregate_id: str, 
                           snapshot_data: Dict[str, Any], 
                           version: int):
        """Save aggregate snapshot"""
        self.snapshots[aggregate_id] = {
            'data': snapshot_data,
            'version': version,
            'timestamp': datetime.utcnow()
        }
    
    async def get_snapshot(self, aggregate_id: str) -> Optional[Dict[str, Any]]:
        """Get latest snapshot for aggregate"""
        return self.snapshots.get(aggregate_id)

class ConcurrencyError(Exception):
    pass

class AggregateRoot:
    """Base class for event-sourced aggregates"""
    
    def __init__(self, aggregate_id: str):
        self.id = aggregate_id
        self.version = 0
        self.uncommitted_events: List[Event] = []
    
    def apply_event(self, event: Event):
        """Apply event to aggregate state"""
        method_name = f"apply_{event.event_type}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            method(event)
        self.version += 1
    
    def raise_event(self, event_type: str, data: Dict[str, Any]):
        """Raise new event"""
        event = Event(
            id=str(uuid.uuid4()),
            aggregate_id=self.id,
            event_type=event_type,
            data=data,
            timestamp=datetime.utcnow(),
            version=self.version + 1
        )
        
        self.uncommitted_events.append(event)
        self.apply_event(event)
    
    def get_uncommitted_events(self) -> List[Event]:
        """Get events that haven't been persisted"""
        return self.uncommitted_events.copy()
    
    def mark_events_as_committed(self):
        """Mark all uncommitted events as committed"""
        self.uncommitted_events.clear()

# Example: User aggregate
class User(AggregateRoot):
    def __init__(self, user_id: str):
        super().__init__(user_id)
        self.email = None
        self.name = None
        self.is_active = True
    
    def create_user(self, email: str, name: str):
        """Create new user"""
        self.raise_event('user_created', {
            'email': email,
            'name': name
        })
    
    def change_email(self, new_email: str):
        """Change user email"""
        if self.email != new_email:
            self.raise_event('email_changed', {
                'old_email': self.email,
                'new_email': new_email
            })
    
    def deactivate(self):
        """Deactivate user"""
        if self.is_active:
            self.raise_event('user_deactivated', {})
    
    # Event handlers
    def apply_user_created(self, event: Event):
        self.email = event.data['email']
        self.name = event.data['name']
        self.is_active = True
    
    def apply_email_changed(self, event: Event):
        self.email = event.data['new_email']
    
    def apply_user_deactivated(self, event: Event):
        self.is_active = False

class Repository:
    """Repository for event-sourced aggregates"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def save(self, aggregate: AggregateRoot):
        """Save aggregate to event store"""
        uncommitted_events = aggregate.get_uncommitted_events()
        
        if uncommitted_events:
            expected_version = aggregate.version - len(uncommitted_events)
            
            await self.event_store.append_events(
                aggregate.id,
                uncommitted_events,
                expected_version
            )
            
            aggregate.mark_events_as_committed()
    
    async def get_by_id(self, aggregate_id: str, aggregate_class: type) -> Optional[AggregateRoot]:
        """Load aggregate from event store"""
        # Try to load from snapshot first
        snapshot = await self.event_store.get_snapshot(aggregate_id)
        
        if snapshot:
            # Load from snapshot
            aggregate = aggregate_class(aggregate_id)
            # Restore state from snapshot
            for key, value in snapshot['data'].items():
                setattr(aggregate, key, value)
            aggregate.version = snapshot['version']
            
            # Apply events since snapshot
            events = await self.event_store.get_events(aggregate_id, snapshot['version'])
        else:
            # Load from beginning
            aggregate = aggregate_class(aggregate_id)
            events = await self.event_store.get_events(aggregate_id)
        
        # Apply events to rebuild state
        for event in events:
            aggregate.apply_event(event)
        
        return aggregate

# Usage example
async def example_event_sourcing():
    # Create event store and repository
    event_store = EventStore()
    repo = Repository(event_store)
    
    # Create user
    user = User('user123')
    user.create_user('john@example.com', 'John Doe')
    
    # Save user
    await repo.save(user)
    
    # Load user from event store
    loaded_user = await repo.get_by_id('user123', User)
    print(f"Loaded user: {loaded_user.name}, {loaded_user.email}")
    
    # Make changes
    loaded_user.change_email('john.doe@example.com')
    loaded_user.deactivate()
    
    # Save changes
    await repo.save(loaded_user)
    
    # Get all events for audit
    all_events = await event_store.get_all_events()
    for event in all_events:
        print(f"Event: {event.event_type} - {event.data}")
```

## 📊 Message Broker Integration

### **Kafka Integration**

```python
from kafka import KafkaProducer, KafkaConsumer
import json
import asyncio
from typing import Dict, Any, Callable

class KafkaEventBus:
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.handlers: Dict[str, List[Callable]] = {}
    
    async def publish(self, topic: str, message: Dict[str, Any], key: str = None):
        """Publish message to Kafka topic"""
        try:
            future = self.producer.send(topic, value=message, key=key)
            record_metadata = future.get(timeout=10)
            
            return {
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset
            }
        except Exception as e:
            raise Exception(f"Failed to publish message: {e}")
    
    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to Kafka topic"""
        if topic not in self.handlers:
            self.handlers[topic] = []
        
        self.handlers[topic].append(handler)
        
        # Create consumer if not exists
        if topic not in self.consumers:
            self.consumers[topic] = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id=f'consumer_group_{topic}'
            )
    
    async def start_consuming(self):
        """Start consuming messages from all subscribed topics"""
        tasks = []
        
        for topic, consumer in self.consumers.items():
            task = asyncio.create_task(self._consume_topic(topic, consumer))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _consume_topic(self, topic: str, consumer: KafkaConsumer):
        """Consume messages from specific topic"""
        for message in consumer:
            # Process message with all handlers
            for handler in self.handlers.get(topic, []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message.value)
                    else:
                        handler(message.value)
                except Exception as e:
                    print(f"Error processing message in {topic}: {e}")
    
    def close(self):
        """Close producer and consumers"""
        self.producer.close()
        for consumer in self.consumers.values():
            consumer.close()
```

## 🔄 Saga Pattern

### **Orchestration-Based Saga**

```python
from enum import Enum
from typing import Dict, List, Any, Optional
import asyncio
import uuid

class SagaStatus(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"

class StepStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    name: str
    action: Callable
    compensation: Callable
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = None

class Saga:
    def __init__(self, saga_id: str, name: str):
        self.id = saga_id
        self.name = name
        self.steps: List[SagaStep] = []
        self.status = SagaStatus.STARTED
        self.context: Dict[str, Any] = {}
    
    def add_step(self, name: str, action: Callable, compensation: Callable):
        """Add step to saga"""
        step = SagaStep(name, action, compensation)
        self.steps.append(step)
    
    async def execute(self) -> bool:
        """Execute saga steps"""
        try:
            # Execute all steps
            for i, step in enumerate(self.steps):
                try:
                    result = await step.action(self.context)
                    step.result = result
                    step.status = StepStatus.COMPLETED
                    
                    # Update context with result
                    self.context[f"{step.name}_result"] = result
                    
                except Exception as e:
                    step.error = str(e)
                    step.status = StepStatus.FAILED
                    
                    # Start compensation
                    await self._compensate(i - 1)
                    self.status = SagaStatus.FAILED
                    return False
            
            self.status = SagaStatus.COMPLETED
            return True
            
        except Exception as e:
            self.status = SagaStatus.FAILED
            return False
    
    async def _compensate(self, from_step: int):
        """Compensate failed saga by undoing completed steps"""
        self.status = SagaStatus.COMPENSATING
        
        # Compensate in reverse order
        for i in range(from_step, -1, -1):
            step = self.steps[i]
            
            if step.status == StepStatus.COMPLETED:
                try:
                    await step.compensation(self.context)
                    step.status = StepStatus.COMPENSATED
                except Exception as e:
                    # Compensation failed - log and continue
                    print(f"Compensation failed for step {step.name}: {e}")

# Example: Order processing saga
class OrderSaga:
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.saga = Saga(f"order_{order_id}", "Order Processing")
        self._setup_steps()
    
    def _setup_steps(self):
        """Setup saga steps"""
        self.saga.add_step(
            "validate_order",
            self._validate_order,
            self._cancel_validation
        )
        
        self.saga.add_step(
            "reserve_inventory",
            self._reserve_inventory,
            self._release_inventory
        )
        
        self.saga.add_step(
            "process_payment",
            self._process_payment,
            self._refund_payment
        )
        
        self.saga.add_step(
            "ship_order",
            self._ship_order,
            self._cancel_shipment
        )
        
        self.saga.add_step(
            "send_confirmation",
            self._send_confirmation,
            self._send_cancellation
        )
    
    async def process_order(self) -> bool:
        """Process order using saga"""
        return await self.saga.execute()
    
    # Step implementations
    async def _validate_order(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Validate order logic
        await asyncio.sleep(0.1)  # Simulate API call
        return {"valid": True, "order_id": self.order_id}
    
    async def _cancel_validation(self, context: Dict[str, Any]):
        # Cancel validation
        print(f"Canceling validation for order {self.order_id}")
    
    async def _reserve_inventory(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Reserve inventory
        await asyncio.sleep(0.1)
        return {"reserved": True, "reservation_id": f"res_{self.order_id}"}
    
    async def _release_inventory(self, context: Dict[str, Any]):
        # Release inventory
        reservation_id = context.get("reserve_inventory_result", {}).get("reservation_id")
        print(f"Releasing inventory reservation {reservation_id}")
    
    async def _process_payment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Process payment
        await asyncio.sleep(0.1)
        
        # Simulate payment failure
        if self.order_id == "failed_order":
            raise Exception("Payment failed")
        
        return {"payment_id": f"pay_{self.order_id}", "amount": 100.0}
    
    async def _refund_payment(self, context: Dict[str, Any]):
        # Refund payment
        payment_id = context.get("process_payment_result", {}).get("payment_id")
        print(f"Refunding payment {payment_id}")
    
    async def _ship_order(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Ship order
        await asyncio.sleep(0.1)
        return {"shipment_id": f"ship_{self.order_id}"}
    
    async def _cancel_shipment(self, context: Dict[str, Any]):
        # Cancel shipment
        shipment_id = context.get("ship_order_result", {}).get("shipment_id")
        print(f"Canceling shipment {shipment_id}")
    
    async def _send_confirmation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Send confirmation
        await asyncio.sleep(0.1)
        return {"confirmation_sent": True}
    
    async def _send_cancellation(self, context: Dict[str, Any]):
        # Send cancellation notice
        print(f"Sending cancellation notice for order {self.order_id}")

# Usage
async def example_saga():
    # Successful order
    order_saga = OrderSaga("order_123")
    success = await order_saga.process_order()
    print(f"Order processing success: {success}")
    
    # Failed order
    failed_saga = OrderSaga("failed_order")
    success = await failed_saga.process_order()
    print(f"Failed order processing: {success}")
```

## 🎯 Best Practices

### **Message Design**

1. **Immutable Events**: Events should be immutable once created
2. **Idempotency**: Ensure message processing is idempotent
3. **Schema Evolution**: Design for backward compatibility
4. **Metadata**: Include correlation IDs and timestamps

### **Error Handling**

1. **Dead Letter Queues**: Handle failed messages
2. **Circuit Breakers**: Prevent cascade failures
3. **Retry Policies**: Implement exponential backoff
4. **Monitoring**: Track message flow and failures

### **Performance**

1. **Batch Processing**: Process messages in batches
2. **Async Processing**: Use async/await for I/O operations
3. **Partitioning**: Distribute load across partitions
4. **Compression**: Compress large messages

### **Security**

1. **Authentication**: Secure message broker access
2. **Authorization**: Control topic access
3. **Encryption**: Encrypt sensitive data
4. **Audit Logging**: Track message access

---

*"Event-driven architecture is not just about messaging—it's about building systems that can evolve, scale, and remain resilient through the power of decoupled, asynchronous communication."*
