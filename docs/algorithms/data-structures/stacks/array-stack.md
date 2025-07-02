# Array-Based Stack

## 🔍 Overview

Array-based stacks use a dynamic array (like Python lists) as the underlying data structure. This is the most common and straightforward implementation of stacks, offering excellent performance and simplicity.

---

## 📊 Characteristics

### Key Properties

- **Fixed or Dynamic Size**: Can use fixed-size arrays or dynamic arrays
- **Top Pointer**: Index tracking the top element position
- **Contiguous Memory**: Elements stored in consecutive memory locations
- **Simple Implementation**: Easy to understand and implement
- **Cache Friendly**: Excellent memory locality

### Memory Layout

```text
Array-Based Stack:
Index:  0   1   2   3   4   5
Array: [A] [B] [C] [ ] [ ] [ ]
        ↑           ↑
       base        top
```

---

## ⏱️ Time Complexities

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| **Push** | O(1) amortized | May resize array occasionally |
| **Pop** | O(1) | Direct access to top element |
| **Peek/Top** | O(1) | No element removal |
| **isEmpty** | O(1) | Check if top index is valid |
| **Size** | O(1) | Track size or use top index |

---

## 💻 Implementation

### Basic Array Stack

```python
class ArrayStack:
    """Array-based stack implementation using Python list."""
    
    def __init__(self, capacity=None):
        """Initialize empty stack with optional capacity limit."""
        self._items = []
        self._capacity = capacity
    
    def push(self, item):
        """Add item to top of stack."""
        if self._capacity and len(self._items) >= self._capacity:
            raise OverflowError("Stack overflow: capacity exceeded")
        
        self._items.append(item)
    
    def pop(self):
        """Remove and return top item."""
        if self.is_empty():
            raise IndexError("Stack underflow: pop from empty stack")
        
        return self._items.pop()
    
    def peek(self):
        """Return top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        return self._items[-1]
    
    def is_empty(self):
        """Check if stack is empty."""
        return len(self._items) == 0
    
    def size(self):
        """Return number of items in stack."""
        return len(self._items)
    
    def __str__(self):
        """String representation with top on the right."""
        return f"Stack({' -> '.join(map(str, self._items))})"
    
    def __repr__(self):
        return f"ArrayStack({self._items})"

def demo_array_stack():
    """Demonstrate array stack operations."""
    
    print("=== Array Stack Demo ===")
    stack = ArrayStack()
    
    # Push operations
    items = ['A', 'B', 'C', 'D']
    for item in items:
        stack.push(item)
        print(f"Push {item}: {stack}")
    
    # Peek operation
    print(f"Peek: {stack.peek()}")
    print(f"Stack after peek: {stack}")
    
    # Pop operations
    while not stack.is_empty():
        popped = stack.pop()
        print(f"Pop {popped}: {stack}")
    
    # Try popping from empty stack
    try:
        stack.pop()
    except IndexError as e:
        print(f"Error: {e}")
    
    return stack
```

### Fixed-Size Array Stack

```python
class FixedArrayStack:
    """Fixed-size array stack implementation."""
    
    def __init__(self, capacity):
        """Initialize stack with fixed capacity."""
        self._capacity = capacity
        self._items = [None] * capacity
        self._top = -1  # Index of top element (-1 means empty)
    
    def push(self, item):
        """Add item to top of stack."""
        if self.is_full():
            raise OverflowError("Stack overflow: capacity exceeded")
        
        self._top += 1
        self._items[self._top] = item
    
    def pop(self):
        """Remove and return top item."""
        if self.is_empty():
            raise IndexError("Stack underflow: pop from empty stack")
        
        item = self._items[self._top]
        self._items[self._top] = None  # Clear reference
        self._top -= 1
        return item
    
    def peek(self):
        """Return top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        return self._items[self._top]
    
    def is_empty(self):
        """Check if stack is empty."""
        return self._top == -1
    
    def is_full(self):
        """Check if stack is full."""
        return self._top == self._capacity - 1
    
    def size(self):
        """Return number of items in stack."""
        return self._top + 1
    
    def capacity(self):
        """Return maximum capacity."""
        return self._capacity
    
    def __str__(self):
        """String representation."""
        if self.is_empty():
            return "Stack([])"
        
        items = [str(self._items[i]) for i in range(self._top + 1)]
        return f"Stack({' -> '.join(items)})"

def demo_fixed_stack():
    """Demonstrate fixed-size stack operations."""
    
    print("=== Fixed Array Stack Demo ===")
    stack = FixedArrayStack(capacity=3)
    
    # Push until full
    items = ['X', 'Y', 'Z']
    for item in items:
        stack.push(item)
        print(f"Push {item}: {stack} (size: {stack.size()}/{stack.capacity()})")
    
    # Try pushing to full stack
    try:
        stack.push('W')
    except OverflowError as e:
        print(f"Error: {e}")
    
    # Pop all items
    while not stack.is_empty():
        popped = stack.pop()
        print(f"Pop {popped}: {stack}")
    
    return stack
```

### Optimized Array Stack

```python
class OptimizedArrayStack:
    """Memory-optimized array stack with shrinking capability."""
    
    def __init__(self, initial_capacity=4):
        """Initialize with small initial capacity."""
        self._capacity = initial_capacity
        self._items = [None] * self._capacity
        self._size = 0
    
    def push(self, item):
        """Add item, resizing if necessary."""
        # Resize if full
        if self._size >= self._capacity:
            self._resize(self._capacity * 2)
        
        self._items[self._size] = item
        self._size += 1
    
    def pop(self):
        """Remove item, shrinking if necessary."""
        if self.is_empty():
            raise IndexError("Stack underflow: pop from empty stack")
        
        self._size -= 1
        item = self._items[self._size]
        self._items[self._size] = None  # Clear reference
        
        # Shrink if quarter full and capacity > 4
        if (self._size <= self._capacity // 4 and 
            self._capacity > 4):
            self._resize(self._capacity // 2)
        
        return item
    
    def peek(self):
        """Return top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        
        return self._items[self._size - 1]
    
    def is_empty(self):
        """Check if stack is empty."""
        return self._size == 0
    
    def size(self):
        """Return number of items."""
        return self._size
    
    def capacity(self):
        """Return current capacity."""
        return self._capacity
    
    def _resize(self, new_capacity):
        """Resize internal array."""
        old_items = self._items
        self._capacity = new_capacity
        self._items = [None] * new_capacity
        
        # Copy existing items
        for i in range(self._size):
            self._items[i] = old_items[i]
    
    def __str__(self):
        if self.is_empty():
            return "Stack([])"
        
        items = [str(self._items[i]) for i in range(self._size)]
        return f"Stack({' -> '.join(items)})"

def demo_optimized_stack():
    """Demonstrate optimized stack with resizing."""
    
    print("=== Optimized Array Stack Demo ===")
    stack = OptimizedArrayStack(initial_capacity=2)
    
    # Push items to trigger resizing
    items = ['1', '2', '3', '4', '5']
    for item in items:
        stack.push(item)
        print(f"Push {item}: size={stack.size()}, capacity={stack.capacity()}")
    
    print(f"Final stack: {stack}")
    
    # Pop items to trigger shrinking
    while not stack.is_empty():
        popped = stack.pop()
        print(f"Pop {popped}: size={stack.size()}, capacity={stack.capacity()}")
    
    return stack
```

---

## 🎯 Use Cases and Applications

### System Stack (Call Stack)

```python
def recursive_function_demo():
    """Demonstrate how system uses stack for function calls."""
    
    def factorial(n, depth=0):
        """Calculate factorial with call stack visualization."""
        indent = "  " * depth
        print(f"{indent}→ factorial({n}) called")
        
        if n <= 1:
            print(f"{indent}← factorial({n}) returns 1")
            return 1
        
        result = n * factorial(n - 1, depth + 1)
        print(f"{indent}← factorial({n}) returns {result}")
        return result
    
    print("Call Stack Demonstration:")
    result = factorial(4)
    print(f"Final result: {result}")
    
    return result

def expression_evaluation():
    """Use stack for expression evaluation."""
    
    def evaluate_postfix(expression):
        """Evaluate postfix expression using stack."""
        stack = ArrayStack()
        
        for token in expression.split():
            if token.isdigit():
                stack.push(int(token))
            else:
                # Operator: pop two operands
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = a // b
                
                stack.push(result)
                print(f"  {a} {token} {b} = {result}")
        
        return stack.pop()
    
    # Example: "3 4 + 2 * 1 -" = ((3 + 4) * 2) - 1 = 13
    expression = "3 4 + 2 * 1 -"
    print(f"Evaluating: {expression}")
    result = evaluate_postfix(expression)
    print(f"Result: {result}")
    
    return result
```

### Undo/Redo Functionality

```python
class UndoRedoTextEditor:
    """Text editor with undo/redo using stacks."""
    
    def __init__(self):
        self.text = ""
        self.undo_stack = ArrayStack()
        self.redo_stack = ArrayStack()
    
    def type_text(self, new_text):
        """Add text and save state for undo."""
        # Save current state before modification
        self.undo_stack.push(self.text)
        
        # Clear redo stack (new action invalidates redo history)
        self.redo_stack = ArrayStack()
        
        # Apply change
        self.text += new_text
        print(f"Typed '{new_text}': '{self.text}'")
    
    def delete_chars(self, count):
        """Delete characters and save state for undo."""
        if count > len(self.text):
            count = len(self.text)
        
        # Save current state
        self.undo_stack.push(self.text)
        self.redo_stack = ArrayStack()
        
        # Apply change
        self.text = self.text[:-count] if count > 0 else self.text
        print(f"Deleted {count} chars: '{self.text}'")
    
    def undo(self):
        """Undo last operation."""
        if self.undo_stack.is_empty():
            print("Nothing to undo")
            return
        
        # Save current state for redo
        self.redo_stack.push(self.text)
        
        # Restore previous state
        self.text = self.undo_stack.pop()
        print(f"Undo: '{self.text}'")
    
    def redo(self):
        """Redo last undone operation."""
        if self.redo_stack.is_empty():
            print("Nothing to redo")
            return
        
        # Save current state for undo
        self.undo_stack.push(self.text)
        
        # Restore redone state
        self.text = self.redo_stack.pop()
        print(f"Redo: '{self.text}'")

def demo_undo_redo():
    """Demonstrate undo/redo functionality."""
    
    print("=== Undo/Redo Text Editor Demo ===")
    editor = UndoRedoTextEditor()
    
    # Type some text
    editor.type_text("Hello")
    editor.type_text(" World")
    editor.type_text("!")
    
    # Delete some characters
    editor.delete_chars(1)
    editor.delete_chars(6)
    
    # Undo operations
    editor.undo()  # Undo delete 6
    editor.undo()  # Undo delete 1
    editor.undo()  # Undo type "!"
    
    # Redo operations
    editor.redo()   # Redo type "!"
    editor.redo()   # Redo delete 1
    
    # Type more (clears redo stack)
    editor.type_text("?")
    
    editor.redo()   # Should show "Nothing to redo"
    
    return editor
```

---

## 🚀 Performance Optimization

### Memory Efficiency

```python
def memory_efficiency_comparison():
    """Compare memory usage of different implementations."""
    
    import sys
    
    # Python list (dynamic array)
    python_list = [i for i in range(1000)]
    list_size = sys.getsizeof(python_list)
    
    # Array stack
    array_stack = ArrayStack()
    for i in range(1000):
        array_stack.push(i)
    stack_size = sys.getsizeof(array_stack._items)
    
    # Fixed array stack
    fixed_stack = FixedArrayStack(1000)
    for i in range(1000):
        fixed_stack.push(i)
    fixed_size = sys.getsizeof(fixed_stack._items)
    
    print("Memory Usage Comparison:")
    print(f"Python list: {list_size} bytes")
    print(f"Array stack: {stack_size} bytes")
    print(f"Fixed stack: {fixed_size} bytes")
    
    return list_size, stack_size, fixed_size

def performance_benchmark():
    """Benchmark different stack operations."""
    
    import time
    
    def time_operations(stack_class, n=100000):
        """Time push and pop operations."""
        stack = stack_class() if stack_class != FixedArrayStack else stack_class(n)
        
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
    
    n = 50000
    print(f"Performance Benchmark ({n} operations):")
    
    # Test different implementations
    implementations = [
        ("ArrayStack", ArrayStack),
        ("FixedArrayStack", lambda: FixedArrayStack(n)),
        ("OptimizedArrayStack", OptimizedArrayStack)
    ]
    
    for name, impl in implementations:
        push_time, pop_time = time_operations(impl, n)
        print(f"{name}:")
        print(f"  Push: {push_time:.4f}s ({n/push_time:.0f} ops/sec)")
        print(f"  Pop:  {pop_time:.4f}s ({n/pop_time:.0f} ops/sec)")
```

---

## 🎯 When to Use Array-Based Stacks

### ✅ Best Use Cases

1. **General Purpose**: Most common stack implementation
2. **Simple Requirements**: When you need basic stack operations
3. **Memory Efficiency**: Good memory locality and less overhead
4. **Integration**: Easy to integrate with existing array-based code
5. **Performance**: Excellent for most use cases

### ❌ Limitations

1. **Fixed Size**: Fixed-size arrays can overflow
2. **Memory Waste**: Dynamic arrays may have unused capacity
3. **Resizing Cost**: Occasional O(n) resize operations
4. **Memory Allocation**: Large blocks of contiguous memory needed

### Comparison with Linked List Stacks

| Aspect | Array Stack | Linked List Stack |
|--------|-------------|-------------------|
| Memory | Better locality | More fragmented |
| Overhead | Lower per element | Higher (pointers) |
| Resizing | Occasional cost | Not needed |
| Implementation | Simpler | More complex |

---

## 🔗 Related Topics

- **[Linked List Stack](linked-list-stack.md)**: Alternative implementation using nodes
- **[Easy Problems](easy-problems.md)**: Practice with stack problems
- **[Applications](../fundamentals.md#applications)**: Real-world stack usage
- **[Dynamic Arrays](../../arrays/dynamic-arrays.md)**: Understanding the underlying structure

---

*Ready to explore a different approach? Check out [Linked List Stack](linked-list-stack.md) next!*
