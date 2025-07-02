# Linked List Stack

## 🔍 Overview

Linked list-based stacks use nodes connected via pointers to implement the stack data structure. This approach provides dynamic sizing without the need for pre-allocated arrays and avoids the resize operations that can occur with array-based stacks.

---

## 📊 Characteristics

### Key Properties

- **Dynamic Size**: Grows and shrinks as needed without pre-allocation
- **Node-Based**: Each element is stored in a separate node with a pointer
- **No Resize**: Never needs to resize like dynamic arrays
- **Memory Efficient**: Uses exactly the memory needed for current elements
- **Pointer Overhead**: Each node requires additional memory for the pointer

### Memory Layout

```text
Linked List Stack:
Top → [C|•] → [B|•] → [A|•] → NULL
      data     data     data
      next     next     next
```

---

## ⏱️ Time Complexities

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Push** | O(1) | Always constant time |
| **Pop** | O(1) | Direct access to top node |
| **Peek/Top** | O(1) | No node removal |
| **isEmpty** | O(1) | Check if top is None |
| **Size** | O(1) or O(n) | Depends on whether size is tracked |

---

## 💻 Implementation

### Basic Node Structure

```python
class Node:
    """Node for linked list stack."""
    
    def __init__(self, data):
        """Initialize node with data."""
        self.data = data
        self.next = None
    
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return f"Node({self.data})"
```

### Simple Linked List Stack

```python
class LinkedListStack:
    """Stack implementation using linked list."""
    
    def __init__(self):
        """Initialize empty stack."""
        self._top = None
        self._size = 0
    
    def push(self, item):
        """Add item to top of stack."""
        new_node = Node(item)
        new_node.next = self._top
        self._top = new_node
        self._size += 1
    
    def pop(self):
        """Remove and return top item."""
        if self.is_empty():
            raise IndexError("Stack underflow: pop from empty stack")
        
        item = self._top.data
        self._top = self._top.next
        self._size -= 1
        return item
    
    def peek(self):
        """Return top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        return self._top.data
    
    def is_empty(self):
        """Check if stack is empty."""
        return self._top is None
    
    def size(self):
        """Return number of items in stack."""
        return self._size
    
    def clear(self):
        """Remove all items from stack."""
        self._top = None
        self._size = 0
    
    def __str__(self):
        """String representation with top on the left."""
        if self.is_empty():
            return "Stack([])"
        
        items = []
        current = self._top
        while current:
            items.append(str(current.data))
            current = current.next
        
        return f"Stack({' -> '.join(items)})"
    
    def __repr__(self):
        return f"LinkedListStack(size={self._size})"
    
    def __iter__(self):
        """Iterator from top to bottom."""
        current = self._top
        while current:
            yield current.data
            current = current.next

def demo_linked_list_stack():
    """Demonstrate linked list stack operations."""
    
    print("=== Linked List Stack Demo ===")
    stack = LinkedListStack()
    
    # Push operations
    items = ['A', 'B', 'C', 'D']
    for item in items:
        stack.push(item)
        print(f"Push {item}: {stack} (size: {stack.size()})")
    
    # Peek operation
    print(f"Peek: {stack.peek()}")
    print(f"Stack after peek: {stack}")
    
    # Iterate through stack
    print("Stack contents (top to bottom):")
    for item in stack:
        print(f"  {item}")
    
    # Pop operations
    while not stack.is_empty():
        popped = stack.pop()
        print(f"Pop {popped}: {stack} (size: {stack.size()})")
    
    # Try popping from empty stack
    try:
        stack.pop()
    except IndexError as e:
        print(f"Error: {e}")
    
    return stack
```

### Memory-Optimized Stack (Without Size Tracking)

```python
class LightweightLinkedStack:
    """Lightweight stack without size tracking to save memory."""
    
    def __init__(self):
        """Initialize empty stack."""
        self._top = None
    
    def push(self, item):
        """Add item to top of stack."""
        new_node = Node(item)
        new_node.next = self._top
        self._top = new_node
    
    def pop(self):
        """Remove and return top item."""
        if self.is_empty():
            raise IndexError("Stack underflow: pop from empty stack")
        
        item = self._top.data
        self._top = self._top.next
        return item
    
    def peek(self):
        """Return top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        return self._top.data
    
    def is_empty(self):
        """Check if stack is empty."""
        return self._top is None
    
    def size(self):
        """Return number of items (O(n) operation)."""
        count = 0
        current = self._top
        while current:
            count += 1
            current = current.next
        return count
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "LightweightStack([])"
        
        items = []
        current = self._top
        while current:
            items.append(str(current.data))
            current = current.next
        
        return f"LightweightStack({' -> '.join(items)})"

def demo_lightweight_stack():
    """Demonstrate lightweight stack."""
    
    print("=== Lightweight Stack Demo ===")
    stack = LightweightLinkedStack()
    
    # Push some items
    for i in range(5):
        stack.push(f"Item{i}")
        print(f"Push Item{i}: {stack}")
    
    print(f"Size (counted): {stack.size()}")
    
    # Pop some items
    for _ in range(3):
        popped = stack.pop()
        print(f"Pop {popped}: {stack}")
    
    return stack
```

### Generic Typed Stack

```python
from typing import TypeVar, Generic, Optional, Iterator

T = TypeVar('T')

class TypedLinkedStack(Generic[T]):
    """Type-safe linked list stack."""
    
    class _Node:
        """Internal node class."""
        def __init__(self, data: T, next_node: Optional['TypedLinkedStack._Node'] = None):
            self.data = data
            self.next = next_node
    
    def __init__(self):
        """Initialize empty stack."""
        self._top: Optional[TypedLinkedStack._Node] = None
        self._size = 0
    
    def push(self, item: T) -> None:
        """Add item to top of stack."""
        new_node = self._Node(item, self._top)
        self._top = new_node
        self._size += 1
    
    def pop(self) -> T:
        """Remove and return top item."""
        if self.is_empty():
            raise IndexError("Stack underflow: pop from empty stack")
        
        assert self._top is not None
        item = self._top.data
        self._top = self._top.next
        self._size -= 1
        return item
    
    def peek(self) -> T:
        """Return top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        assert self._top is not None
        return self._top.data
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return self._top is None
    
    def size(self) -> int:
        """Return number of items in stack."""
        return self._size
    
    def __iter__(self) -> Iterator[T]:
        """Iterator from top to bottom."""
        current = self._top
        while current:
            yield current.data
            current = current.next
    
    def __len__(self) -> int:
        """Return size (for len() function)."""
        return self._size

def demo_typed_stack():
    """Demonstrate typed stack."""
    
    print("=== Typed Stack Demo ===")
    
    # Integer stack
    int_stack: TypedLinkedStack[int] = TypedLinkedStack()
    for i in range(5):
        int_stack.push(i * 10)
    
    print(f"Integer stack: {list(int_stack)}")
    
    # String stack
    str_stack: TypedLinkedStack[str] = TypedLinkedStack()
    words = ["hello", "world", "stack", "example"]
    for word in words:
        str_stack.push(word)
    
    print(f"String stack: {list(str_stack)}")
    
    # Type safety (this would cause type checker warnings)
    # int_stack.push("string")  # Type error!
    
    return int_stack, str_stack
```

---

## 🎯 Advanced Use Cases

### Multi-Stack Implementation

```python
class MultiStack:
    """Multiple stacks sharing nodes efficiently."""
    
    def __init__(self, num_stacks):
        """Initialize multiple empty stacks."""
        self.num_stacks = num_stacks
        self.tops = [None] * num_stacks
        self.sizes = [0] * num_stacks
    
    def push(self, stack_num, item):
        """Push item to specified stack."""
        if not 0 <= stack_num < self.num_stacks:
            raise ValueError(f"Invalid stack number: {stack_num}")
        
        new_node = Node(item)
        new_node.next = self.tops[stack_num]
        self.tops[stack_num] = new_node
        self.sizes[stack_num] += 1
    
    def pop(self, stack_num):
        """Pop item from specified stack."""
        if not 0 <= stack_num < self.num_stacks:
            raise ValueError(f"Invalid stack number: {stack_num}")
        
        if self.is_empty(stack_num):
            raise IndexError(f"Stack {stack_num} is empty")
        
        top_node = self.tops[stack_num]
        item = top_node.data
        self.tops[stack_num] = top_node.next
        self.sizes[stack_num] -= 1
        return item
    
    def peek(self, stack_num):
        """Peek at top of specified stack."""
        if not 0 <= stack_num < self.num_stacks:
            raise ValueError(f"Invalid stack number: {stack_num}")
        
        if self.is_empty(stack_num):
            raise IndexError(f"Stack {stack_num} is empty")
        
        return self.tops[stack_num].data
    
    def is_empty(self, stack_num):
        """Check if specified stack is empty."""
        if not 0 <= stack_num < self.num_stacks:
            raise ValueError(f"Invalid stack number: {stack_num}")
        
        return self.tops[stack_num] is None
    
    def size(self, stack_num):
        """Get size of specified stack."""
        if not 0 <= stack_num < self.num_stacks:
            raise ValueError(f"Invalid stack number: {stack_num}")
        
        return self.sizes[stack_num]
    
    def __str__(self):
        """String representation of all stacks."""
        result = []
        for i in range(self.num_stacks):
            items = []
            current = self.tops[i]
            while current:
                items.append(str(current.data))
                current = current.next
            
            stack_str = ' -> '.join(items) if items else 'empty'
            result.append(f"Stack {i}: {stack_str}")
        
        return '\n'.join(result)

def demo_multi_stack():
    """Demonstrate multi-stack."""
    
    print("=== Multi-Stack Demo ===")
    multi = MultiStack(3)
    
    # Push to different stacks
    multi.push(0, 'A0')
    multi.push(1, 'B1')
    multi.push(0, 'A1')
    multi.push(2, 'C0')
    multi.push(1, 'B2')
    
    print("After pushes:")
    print(multi)
    
    # Pop from stacks
    print(f"\nPop from stack 0: {multi.pop(0)}")
    print(f"Pop from stack 1: {multi.pop(1)}")
    
    print("\nAfter pops:")
    print(multi)
    
    return multi
```

### Stack with Minimum Tracking

```python
class MinStack:
    """Stack that tracks minimum element efficiently."""
    
    def __init__(self):
        """Initialize stack with minimum tracking."""
        self._stack = LinkedListStack()
        self._min_stack = LinkedListStack()
    
    def push(self, item):
        """Push item and update minimum."""
        self._stack.push(item)
        
        # Update minimum stack
        if self._min_stack.is_empty() or item <= self._min_stack.peek():
            self._min_stack.push(item)
    
    def pop(self):
        """Pop item and update minimum."""
        if self._stack.is_empty():
            raise IndexError("Stack underflow: pop from empty stack")
        
        item = self._stack.pop()
        
        # Update minimum stack
        if item == self._min_stack.peek():
            self._min_stack.pop()
        
        return item
    
    def peek(self):
        """Return top item."""
        return self._stack.peek()
    
    def get_min(self):
        """Return minimum element in O(1)."""
        if self._min_stack.is_empty():
            raise IndexError("Stack is empty")
        
        return self._min_stack.peek()
    
    def is_empty(self):
        """Check if stack is empty."""
        return self._stack.is_empty()
    
    def size(self):
        """Return size of stack."""
        return self._stack.size()
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "MinStack([])"
        
        return f"MinStack({list(self._stack)}, min={self.get_min()})"

def demo_min_stack():
    """Demonstrate minimum tracking stack."""
    
    print("=== Min Stack Demo ===")
    stack = MinStack()
    
    # Push items
    items = [5, 2, 8, 1, 9, 1, 3]
    for item in items:
        stack.push(item)
        print(f"Push {item}: min = {stack.get_min()}")
    
    print(f"\nStack: {stack}")
    
    # Pop items
    while not stack.is_empty():
        popped = stack.pop()
        min_val = stack.get_min() if not stack.is_empty() else "N/A"
        print(f"Pop {popped}: min = {min_val}")
    
    return stack
```

---

## 🚀 Performance Analysis

### Memory Usage Comparison

```python
def memory_analysis():
    """Compare memory usage of different stack implementations."""
    
    import sys
    from array import array
    
    def measure_stack_memory(stack_type, n=1000):
        """Measure memory usage for n elements."""
        if stack_type == "array":
            stack = list(range(n))
            return sys.getsizeof(stack)
        
        elif stack_type == "linked":
            # Approximate linked list memory
            # Each node: object overhead + data + pointer
            node_size = sys.getsizeof(type('Node', (), {})) + sys.getsizeof(0) + sys.getsizeof(None)
            return n * node_size
        
        return 0
    
    n = 1000
    array_memory = measure_stack_memory("array", n)
    linked_memory = measure_stack_memory("linked", n)
    
    print(f"Memory Usage Comparison ({n} elements):")
    print(f"Array stack: {array_memory} bytes ({array_memory/n:.1f} per element)")
    print(f"Linked stack: {linked_memory} bytes ({linked_memory/n:.1f} per element)")
    print(f"Overhead ratio: {linked_memory/array_memory:.2f}x")
    
    return array_memory, linked_memory

def performance_comparison():
    """Compare performance of array vs linked list stacks."""
    
    import time
    
    def time_stack_operations(create_stack, n=50000):
        """Time push and pop operations."""
        stack = create_stack()
        
        # Time push operations
        start = time.time()
        for i in range(n):
            stack.push(i)
        push_time = time.time() - start
        
        # Time pop operations
        start = time.time()
        while not stack.is_empty():
            stack.pop()
        pop_time = time.time() - start
        
        return push_time, pop_time
    
    n = 30000
    print(f"Performance Comparison ({n} operations):")
    
    # Array stack
    from .array_stack import ArrayStack  # Assuming previous implementation
    array_push, array_pop = time_stack_operations(ArrayStack, n)
    
    # Linked list stack
    linked_push, linked_pop = time_stack_operations(LinkedListStack, n)
    
    print(f"Array Stack:")
    print(f"  Push: {array_push:.4f}s ({n/array_push:.0f} ops/sec)")
    print(f"  Pop:  {array_pop:.4f}s ({n/array_pop:.0f} ops/sec)")
    
    print(f"Linked Stack:")
    print(f"  Push: {linked_push:.4f}s ({n/linked_push:.0f} ops/sec)")
    print(f"  Pop:  {linked_pop:.4f}s ({n/linked_pop:.0f} ops/sec)")
    
    print(f"Speed Comparison:")
    print(f"  Push ratio: {linked_push/array_push:.2f}x")
    print(f"  Pop ratio:  {linked_pop/array_pop:.2f}x")
```

---

## 🎯 When to Use Linked List Stacks

### ✅ Best Use Cases

1. **Unknown Size**: When maximum size is unpredictable
2. **Memory Constraints**: Need to use exact amount of memory
3. **No Resize Penalty**: Want consistent O(1) operations
4. **Embedded Systems**: Limited memory environments
5. **Concurrent Access**: Easier to implement lock-free versions

### ❌ Limitations

1. **Memory Overhead**: Extra pointer storage per element
2. **Cache Performance**: Poor memory locality compared to arrays
3. **Implementation Complexity**: More complex than array-based
4. **Memory Fragmentation**: Nodes scattered throughout memory

### Comparison Summary

| Aspect | Array Stack | Linked List Stack |
|--------|-------------|-------------------|
| **Memory per element** | Lower | Higher (pointer overhead) |
| **Cache performance** | Better | Worse |
| **Resize operations** | Occasional O(n) | Never needed |
| **Memory usage** | May waste space | Exact fit |
| **Implementation** | Simpler | More complex |
| **Concurrent access** | Harder | Easier for lock-free |

---

## 🔗 Related Topics

- **[Array Stack](array-stack.md)**: Alternative implementation using arrays
- **[Easy Problems](easy-problems.md)**: Practice with stack problems
- **[Linked Lists](../../linked-lists/index.md)**: Understanding the underlying structure
- **[Memory Management](../../../systems/memory-management.md)**: Node allocation strategies

---

*Ready to practice? Try [Easy Problems](easy-problems.md) or compare with [Array Stack](array-stack.md)!*
