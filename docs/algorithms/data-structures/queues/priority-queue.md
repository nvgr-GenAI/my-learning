# Priority Queue

## 🔍 Overview

A Priority Queue is an abstract data type where each element has an associated priority, and elements are served based on their priority rather than their insertion order. Higher priority elements are dequeued before lower priority elements, regardless of when they were added.

---

## 📊 Characteristics

### Key Properties

- **Priority-Based Ordering**: Elements are ordered by priority, not insertion order
- **Abstract Data Type**: Can be implemented using various data structures
- **Duplicate Priorities**: Multiple elements can have the same priority
- **Dynamic Priorities**: Priorities can sometimes be modified after insertion
- **Two Types**: Min-priority queue (smallest first) or Max-priority queue (largest first)

### Conceptual Layout

```text
Priority Queue (Max-heap example):
Priority: 9   7   8   3   5   2   4
         [A] [B] [C] [D] [E] [F] [G]
          ↑
      Highest priority (dequeued first)

After dequeue: [B] becomes new highest priority
```

---

## ⏱️ Time Complexities

### Different Implementation Approaches

| Implementation | Insert | Extract-Max | Peek-Max | Notes |
|----------------|--------|-------------|----------|-------|
| **Unsorted Array** | O(1) | O(n) | O(n) | Simple but inefficient |
| **Sorted Array** | O(n) | O(1) | O(1) | Good for few insertions |
| **Linked List** | O(1) or O(n) | O(n) | O(n) | Depends on insertion strategy |
| **Binary Heap** | O(log n) | O(log n) | O(1) | Balanced performance |
| **Fibonacci Heap** | O(1) | O(log n) | O(1) | Advanced, decrease-key O(1) |

---

## 💻 Implementation

### Method 1: Using Built-in heapq (Min-Heap)

```python
import heapq

class PriorityQueue:
    """Priority queue using Python's heapq (min-heap)."""
    
    def __init__(self):
        """Initialize empty priority queue."""
        self._heap = []
        self._index = 0  # For tie-breaking and order preservation
    
    def enqueue(self, item, priority):
        """Add item with given priority."""
        # Use negative priority for max-heap behavior
        # Include index for stable ordering
        heapq.heappush(self._heap, (priority, self._index, item))
        self._index += 1
    
    def dequeue(self):
        """Remove and return highest priority item."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, index, item = heapq.heappop(self._heap)
        return item
    
    def peek(self):
        """Return highest priority item without removing."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, index, item = self._heap[0]
        return item
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self._heap) == 0
    
    def size(self):
        """Get number of items."""
        return len(self._heap)
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "PriorityQueue(empty)"
        
        items = [(priority, item) for priority, _, item in self._heap]
        return f"PriorityQueue({items})"

# Usage Example
pq = PriorityQueue()
pq.enqueue("Low priority task", 1)
pq.enqueue("High priority task", 10)
pq.enqueue("Medium priority task", 5)

print(pq)  # Shows internal heap structure
print(pq.dequeue())  # "Low priority task" (min-heap: lowest priority first)
```

### Method 2: Max-Priority Queue

```python
class MaxPriorityQueue:
    """Max-priority queue (highest priority first)."""
    
    def __init__(self):
        """Initialize empty max-priority queue."""
        self._heap = []
        self._index = 0
    
    def enqueue(self, item, priority):
        """Add item with given priority."""
        # Negate priority to simulate max-heap with min-heap
        heapq.heappush(self._heap, (-priority, self._index, item))
        self._index += 1
    
    def dequeue(self):
        """Remove and return highest priority item."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        neg_priority, index, item = heapq.heappop(self._heap)
        return item
    
    def peek(self):
        """Return highest priority item without removing."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        neg_priority, index, item = self._heap[0]
        return item
    
    def get_priority(self):
        """Get priority of the highest priority item."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        neg_priority, index, item = self._heap[0]
        return -neg_priority
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self._heap) == 0
    
    def size(self):
        """Get number of items."""
        return len(self._heap)

# Usage Example
max_pq = MaxPriorityQueue()
max_pq.enqueue("Low priority", 1)
max_pq.enqueue("High priority", 10)
max_pq.enqueue("Medium priority", 5)

print(max_pq.dequeue())  # "High priority" (highest priority first)
print(max_pq.dequeue())  # "Medium priority"
print(max_pq.dequeue())  # "Low priority"
```

### Method 3: Custom Object Priority

```python
class Task:
    """Task with priority and metadata."""
    
    def __init__(self, name, priority, deadline=None):
        self.name = name
        self.priority = priority
        self.deadline = deadline
    
    def __str__(self):
        return f"Task({self.name}, priority={self.priority})"
    
    def __repr__(self):
        return str(self)

class TaskPriorityQueue:
    """Priority queue for task objects."""
    
    def __init__(self, max_priority=True):
        """Initialize task priority queue."""
        self._heap = []
        self._index = 0
        self.max_priority = max_priority
    
    def add_task(self, task):
        """Add task to queue."""
        priority = task.priority
        if self.max_priority:
            priority = -priority  # Negate for max-heap behavior
        
        heapq.heappush(self._heap, (priority, self._index, task))
        self._index += 1
    
    def get_next_task(self):
        """Get next task to execute."""
        if self.is_empty():
            raise IndexError("No tasks in queue")
        
        priority, index, task = heapq.heappop(self._heap)
        return task
    
    def peek_next_task(self):
        """Peek at next task without removing."""
        if self.is_empty():
            raise IndexError("No tasks in queue")
        
        priority, index, task = self._heap[0]
        return task
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self._heap) == 0
    
    def pending_tasks(self):
        """Get number of pending tasks."""
        return len(self._heap)
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "TaskQueue(empty)"
        
        tasks = [task.name for _, _, task in sorted(self._heap)]
        return f"TaskQueue({tasks})"

# Usage Example
task_queue = TaskPriorityQueue(max_priority=True)
task_queue.add_task(Task("Send email", 2))
task_queue.add_task(Task("Fix bug", 9))
task_queue.add_task(Task("Write docs", 4))

print(task_queue)
print(f"Next task: {task_queue.get_next_task()}")  # Fix bug (priority 9)
print(f"Then: {task_queue.get_next_task()}")       # Write docs (priority 4)
```

### Method 4: Array-Based Implementation

```python
class ArrayPriorityQueue:
    """Simple array-based priority queue."""
    
    def __init__(self, capacity=10, max_priority=True):
        """Initialize with fixed capacity."""
        self.items = []
        self.capacity = capacity
        self.max_priority = max_priority
    
    def enqueue(self, item, priority):
        """Insert item maintaining priority order."""
        if len(self.items) >= self.capacity:
            raise OverflowError("Priority queue is full")
        
        # Create tuple of (priority, item)
        entry = (priority, item)
        
        # Find insertion position to maintain order
        insertion_pos = 0
        for i, (existing_priority, _) in enumerate(self.items):
            if self.max_priority:
                if priority > existing_priority:
                    insertion_pos = i
                    break
                insertion_pos = i + 1
            else:
                if priority < existing_priority:
                    insertion_pos = i
                    break
                insertion_pos = i + 1
        
        self.items.insert(insertion_pos, entry)
    
    def dequeue(self):
        """Remove highest priority item."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, item = self.items.pop(0)
        return item
    
    def peek(self):
        """Get highest priority item without removing."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, item = self.items[0]
        return item
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0
    
    def is_full(self):
        """Check if queue is full."""
        return len(self.items) >= self.capacity
    
    def size(self):
        """Get number of items."""
        return len(self.items)
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "ArrayPriorityQueue(empty)"
        
        items_str = [f"{item}(p={priority})" for priority, item in self.items]
        return f"ArrayPriorityQueue([{', '.join(items_str)}])"

# Usage Example
arr_pq = ArrayPriorityQueue(max_priority=True)
arr_pq.enqueue("Task A", 3)
arr_pq.enqueue("Task B", 7)
arr_pq.enqueue("Task C", 1)
arr_pq.enqueue("Task D", 5)

print(arr_pq)  # Shows items in priority order
print(arr_pq.dequeue())  # "Task B" (priority 7)
```

---

## 🔧 Advanced Features

### Priority Queue with Decrease-Key

```python
class AdvancedPriorityQueue:
    """Priority queue with decrease-key operation."""
    
    def __init__(self):
        """Initialize advanced priority queue."""
        self._heap = []
        self._entry_map = {}  # item -> [priority, index, item]
        self._index = 0
    
    def enqueue(self, item, priority):
        """Add item with priority."""
        if item in self._entry_map:
            self.change_priority(item, priority)
            return
        
        entry = [priority, self._index, item]
        self._entry_map[item] = entry
        heapq.heappush(self._heap, entry)
        self._index += 1
    
    def dequeue(self):
        """Remove highest priority item."""
        while self._heap:
            priority, index, item = heapq.heappop(self._heap)
            
            # Check if entry is still valid (not marked as removed)
            if item in self._entry_map:
                del self._entry_map[item]
                return item
        
        raise IndexError("Priority queue is empty")
    
    def change_priority(self, item, new_priority):
        """Change priority of existing item."""
        if item not in self._entry_map:
            raise KeyError(f"Item {item} not in queue")
        
        # Mark old entry as removed
        old_entry = self._entry_map[item]
        old_entry[2] = None  # Mark as removed
        
        # Add new entry
        entry = [new_priority, self._index, item]
        self._entry_map[item] = entry
        heapq.heappush(self._heap, entry)
        self._index += 1
    
    def contains(self, item):
        """Check if item is in queue."""
        return item in self._entry_map
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self._entry_map) == 0
    
    def size(self):
        """Get number of items."""
        return len(self._entry_map)

# Usage Example
adv_pq = AdvancedPriorityQueue()
adv_pq.enqueue("Task A", 5)
adv_pq.enqueue("Task B", 3)
adv_pq.change_priority("Task B", 8)  # Increase priority
print(adv_pq.dequeue())  # "Task B" (now has priority 8)
```

### Multi-Level Priority Queue

```python
class MultiLevelPriorityQueue:
    """Priority queue with multiple priority levels."""
    
    def __init__(self, num_levels=3):
        """Initialize with specified number of priority levels."""
        self.levels = [[] for _ in range(num_levels)]
        self.num_levels = num_levels
        self.current_level = 0
    
    def enqueue(self, item, priority_level):
        """Add item to specific priority level."""
        if not (0 <= priority_level < self.num_levels):
            raise ValueError(f"Priority level must be 0-{self.num_levels-1}")
        
        self.levels[priority_level].append(item)
    
    def dequeue(self):
        """Remove item from highest non-empty priority level."""
        # Start from highest priority level (0)
        for level in range(self.num_levels):
            if self.levels[level]:
                return self.levels[level].pop(0)
        
        raise IndexError("Priority queue is empty")
    
    def peek(self):
        """Peek at highest priority item."""
        for level in range(self.num_levels):
            if self.levels[level]:
                return self.levels[level][0]
        
        raise IndexError("Priority queue is empty")
    
    def is_empty(self):
        """Check if queue is empty."""
        return all(not level for level in self.levels)
    
    def size(self):
        """Get total number of items."""
        return sum(len(level) for level in self.levels)
    
    def level_sizes(self):
        """Get size of each priority level."""
        return [len(level) for level in self.levels]
    
    def __str__(self):
        """String representation."""
        level_strs = []
        for i, level in enumerate(self.levels):
            if level:
                level_strs.append(f"L{i}: {level}")
        
        if not level_strs:
            return "MultiLevelPQ(empty)"
        
        return f"MultiLevelPQ({', '.join(level_strs)})"

# Usage Example
ml_pq = MultiLevelPriorityQueue(3)
ml_pq.enqueue("Critical task", 0)      # Highest priority
ml_pq.enqueue("Normal task", 1)        # Medium priority
ml_pq.enqueue("Background task", 2)    # Lowest priority
ml_pq.enqueue("Another critical", 0)   # Highest priority

print(ml_pq)
print(ml_pq.dequeue())  # "Critical task" (first in level 0)
print(ml_pq.dequeue())  # "Another critical" (second in level 0)
print(ml_pq.dequeue())  # "Normal task" (from level 1)
```

---

## ⚖️ Advantages & Disadvantages

### ✅ Advantages

- **Flexible Ordering**: Elements served by importance, not insertion order
- **Efficient for Scheduling**: Perfect for task scheduling and resource allocation
- **Dynamic Priorities**: Can handle changing priorities efficiently
- **Versatile**: Many real-world applications

### ❌ Disadvantages

- **More Complex**: More complex than simple FIFO queue
- **Overhead**: Additional complexity for priority management
- **Memory Usage**: May require extra storage for priorities
- **Implementation Choice**: Performance depends on underlying data structure

---

## 🎯 When to Use Priority Queues

### ✅ Use Priority Queue When

- **Task Scheduling**: Operating system process scheduling
- **Dijkstra's Algorithm**: Shortest path algorithms
- **A* Search**: Pathfinding with heuristics
- **Event Simulation**: Discrete event simulation
- **Huffman Coding**: Building optimal prefix codes
- **Emergency Systems**: Medical triage, emergency dispatch

### ❌ Avoid Priority Queue When

- **Simple FIFO Needed**: Order of insertion is the desired order
- **All Equal Priority**: All elements have same importance
- **Performance Critical**: Simple queue operations needed
- **Memory Constrained**: Additional priority overhead not acceptable

---

## 🔄 Real-World Applications

### 1. **Operating System Scheduling**

```python
class ProcessScheduler:
    """CPU process scheduler using priority queue."""
    
    def __init__(self):
        self.ready_queue = MaxPriorityQueue()
        self.time_slice = 10
    
    def add_process(self, process_id, priority, burst_time):
        """Add process to ready queue."""
        process = {
            'id': process_id,
            'burst_time': burst_time,
            'remaining_time': burst_time,
            'arrival_time': time.time()
        }
        self.ready_queue.enqueue(process, priority)
    
    def schedule_next(self):
        """Get next process to execute."""
        if not self.ready_queue.is_empty():
            return self.ready_queue.dequeue()
        return None
```

### 2. **Hospital Emergency System**

```python
class EmergencyRoom:
    """Hospital emergency room triage system."""
    
    PRIORITY_LEVELS = {
        'critical': 1,     # Life-threatening
        'urgent': 2,       # Urgent care needed
        'standard': 3,     # Standard care
        'non-urgent': 4    # Can wait
    }
    
    def __init__(self):
        self.patient_queue = PriorityQueue()
        self.patient_counter = 0
    
    def admit_patient(self, name, condition):
        """Admit patient with medical condition."""
        priority = self.PRIORITY_LEVELS.get(condition, 4)
        patient = {
            'id': self.patient_counter,
            'name': name,
            'condition': condition,
            'arrival_time': time.time()
        }
        
        # Lower number = higher priority
        self.patient_queue.enqueue(patient, priority)
        self.patient_counter += 1
    
    def next_patient(self):
        """Get next patient to see."""
        if not self.patient_queue.is_empty():
            return self.patient_queue.dequeue()
        return None

# Usage Example
er = EmergencyRoom()
er.admit_patient("John Doe", "standard")
er.admit_patient("Jane Smith", "critical")
er.admit_patient("Bob Wilson", "urgent")

print(er.next_patient())  # Jane Smith (critical)
print(er.next_patient())  # Bob Wilson (urgent)
```

### 3. **Game AI Decision Making**

```python
class AIDecisionEngine:
    """AI decision engine using priority queue."""
    
    def __init__(self):
        self.action_queue = MaxPriorityQueue()
    
    def evaluate_actions(self, game_state):
        """Evaluate and queue possible actions."""
        # Example: evaluate different AI actions
        actions = [
            ('attack_enemy', self.calculate_attack_priority(game_state)),
            ('gather_resources', self.calculate_gather_priority(game_state)),
            ('build_defense', self.calculate_defense_priority(game_state)),
            ('explore', self.calculate_explore_priority(game_state))
        ]
        
        for action, priority in actions:
            if priority > 0:  # Only queue viable actions
                self.action_queue.enqueue(action, priority)
    
    def get_best_action(self):
        """Get highest priority action."""
        if not self.action_queue.is_empty():
            return self.action_queue.dequeue()
        return "idle"
    
    def calculate_attack_priority(self, game_state):
        """Calculate priority for attacking."""
        # Simplified example
        enemy_health = game_state.get('enemy_health', 100)
        player_health = game_state.get('player_health', 100)
        
        if enemy_health < 30 and player_health > 50:
            return 90  # High priority when enemy is weak
        elif player_health < 20:
            return 10  # Low priority when player is weak
        return 50  # Medium priority otherwise
    
    def calculate_gather_priority(self, game_state):
        """Calculate priority for resource gathering."""
        resources = game_state.get('resources', 0)
        return max(0, 100 - resources)  # Higher priority when resources are low
    
    def calculate_defense_priority(self, game_state):
        """Calculate priority for building defenses."""
        player_health = game_state.get('player_health', 100)
        enemy_attack = game_state.get('enemy_attack_power', 10)
        
        if player_health < 40 and enemy_attack > 15:
            return 85  # High priority when vulnerable
        return 30  # Low priority otherwise
    
    def calculate_explore_priority(self, game_state):
        """Calculate priority for exploration."""
        map_explored = game_state.get('map_explored_percent', 0)
        return max(0, 50 - map_explored)  # Higher when less explored
```

---

## 🚀 Next Steps

- **[Array Queue](array-queue.md)**: Compare with simple FIFO queues
- **[Medium Problems](medium-problems.md)**: Practice with priority queue problems
- **[Hard Problems](hard-problems.md)**: Master advanced priority queue algorithms
- **[Heaps](../../heaps/index.md)**: Deep dive into heap data structures

---

*Priority queues are fundamental for many algorithms and systems - master them well!*
