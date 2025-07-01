# Stacks: Fundamentals & Operations

## 📚 What is a Stack?

A **Stack** is a linear data structure that follows the **Last In, First Out (LIFO)** principle. Elements are added and removed from the same end, called the "top" of the stack. Think of it as a stack of books - you can only add or remove books from the top.

## 🔄 LIFO Principle

**Last In, First Out** means:

- The **last element** added to the stack is the **first one** to be removed
- Elements are always added and removed from the **top**
- You cannot access elements in the middle without removing elements above them

```text
Push Operations:    Pop Operations:
                   
    |   |              |   |
    | C |              |   |  <- C (popped first)
    | B |              | B |
    | A |              | A |
    +---+              +---+
   
   C was last in,     C is first out
```

## 🏗️ Basic Operations

### 1. Push (Insert)

Add an element to the top of the stack.

```python
def push(stack, element):
    """Add element to top of stack."""
    stack.append(element)
    
# Time: O(1), Space: O(1)
```

### 2. Pop (Remove)

Remove and return the top element from the stack.

```python
def pop(stack):
    """Remove and return top element."""
    if is_empty(stack):
        raise IndexError("Pop from empty stack")
    return stack.pop()

# Time: O(1), Space: O(1)
```

### 3. Peek/Top (Access)

Return the top element without removing it.

```python
def peek(stack):
    """Return top element without removing."""
    if is_empty(stack):
        raise IndexError("Peek from empty stack")
    return stack[-1]

# Time: O(1), Space: O(1)
```

### 4. isEmpty (Check)

Check if the stack is empty.

```python
def is_empty(stack):
    """Check if stack is empty."""
    return len(stack) == 0

# Time: O(1), Space: O(1)
```

### 5. Size (Query)

Get the number of elements in the stack.

```python
def size(stack):
    """Get stack size."""
    return len(stack)

# Time: O(1), Space: O(1)
```

## 💻 Implementations

### 1. Array-Based Implementation

```python
class ArrayStack:
    def __init__(self, capacity=10):
        """Initialize stack with given capacity."""
        self.items = []
        self.capacity = capacity
    
    def push(self, item):
        """Push item to top of stack."""
        if len(self.items) >= self.capacity:
            raise OverflowError("Stack overflow")
        self.items.append(item)
    
    def pop(self):
        """Pop item from top of stack."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        """Peek at top item without removing."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Get stack size."""
        return len(self.items)
    
    def __str__(self):
        """String representation."""
        return f"Stack({self.items})"

# Usage
stack = ArrayStack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack)        # Stack([1, 2, 3])
print(stack.pop())  # 3
print(stack.peek()) # 2
```

### 2. Linked List Implementation

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedListStack:
    def __init__(self):
        """Initialize empty stack."""
        self.head = None  # Top of stack
        self._size = 0
    
    def push(self, item):
        """Push item to top of stack."""
        new_node = ListNode(item)
        new_node.next = self.head
        self.head = new_node
        self._size += 1
    
    def pop(self):
        """Pop item from top of stack."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        
        item = self.head.val
        self.head = self.head.next
        self._size -= 1
        return item
    
    def peek(self):
        """Peek at top item without removing."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.head.val
    
    def is_empty(self):
        """Check if stack is empty."""
        return self.head is None
    
    def size(self):
        """Get stack size."""
        return self._size
    
    def __str__(self):
        """String representation."""
        items = []
        current = self.head
        while current:
            items.append(current.val)
            current = current.next
        return f"Stack(top -> {items})"

# Usage
stack = LinkedListStack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack)        # Stack(top -> [3, 2, 1])
print(stack.pop())  # 3
print(stack.peek()) # 2
```

## 📊 Complexity Analysis

| **Implementation** | **Operation** | **Time** | **Space** |
|-------------------|---------------|----------|-----------|
| **Array-based** | Push | O(1)* | O(1) |
| | Pop | O(1) | O(1) |
| | Peek | O(1) | O(1) |
| | Search | O(n) | O(1) |
| **Linked List** | Push | O(1) | O(1) |
| | Pop | O(1) | O(1) |
| | Peek | O(1) | O(1) |
| | Search | O(n) | O(1) |

*Note: Push can be O(n) if array needs to be resized (amortized O(1))

## ⚖️ Array vs Linked List Implementation

| **Feature** | **Array-based** | **Linked List** |
|------------|-----------------|-----------------|
| **Memory** | Contiguous | Scattered |
| **Cache Performance** | Better | Worse |
| **Memory Overhead** | Lower | Higher (pointers) |
| **Dynamic Size** | Limited by capacity | Unlimited |
| **Implementation** | Simpler | More complex |

## 🎯 When to Use Each Implementation

### ✅ Use Array-based Stack When

- **Known maximum size**: Capacity is predictable
- **Memory efficiency**: Want minimal overhead
- **Cache performance**: Need fast access patterns
- **Simple implementation**: Want straightforward code

### ✅ Use Linked List Stack When

- **Dynamic size**: Unpredictable stack size
- **Memory constraints**: Fixed array too large
- **Flexibility**: Need true dynamic allocation
- **No size limits**: Want unlimited growth

## 🔧 Advanced Operations

### 1. Multi-Push/Pop

```python
def multi_push(stack, items):
    """Push multiple items at once."""
    for item in items:
        stack.push(item)

def multi_pop(stack, count):
    """Pop multiple items at once."""
    if count > stack.size():
        raise IndexError("Not enough items to pop")
    
    items = []
    for _ in range(count):
        items.append(stack.pop())
    return items
```

### 2. Stack with Minimum

```python
class MinStack:
    def __init__(self):
        """Stack that tracks minimum element."""
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        """Push value and update minimum."""
        self.stack.append(val)
        
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        """Pop value and update minimum."""
        if not self.stack:
            raise IndexError("Pop from empty stack")
        
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        
        return val
    
    def peek(self):
        """Get top value."""
        if not self.stack:
            raise IndexError("Peek from empty stack")
        return self.stack[-1]
    
    def get_min(self):
        """Get minimum value in O(1)."""
        if not self.min_stack:
            raise IndexError("No minimum in empty stack")
        return self.min_stack[-1]
```

## 🎨 Common Patterns

### 1. Monotonic Stack

A stack that maintains elements in monotonic (increasing or decreasing) order:

```python
def next_greater_elements(nums):
    """Find next greater element for each number."""
    result = [-1] * len(nums)
    stack = []  # Stores indices
    
    for i, num in enumerate(nums):
        # While stack not empty and current num is greater
        while stack and nums[stack[-1]] < num:
            index = stack.pop()
            result[index] = num
        
        stack.append(i)
    
    return result

# Example: [2, 1, 2, 4, 3, 1] → [4, 2, 4, -1, -1, -1]
```

### 2. Parentheses Matching

```python
def is_valid_parentheses(s):
    """Check if parentheses are properly matched."""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0
```

### 3. Expression Evaluation

```python
def evaluate_postfix(expression):
    """Evaluate postfix expression using stack."""
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in expression.split():
        if token in operators:
            # Pop two operands
            b = stack.pop()
            a = stack.pop()
            
            # Perform operation
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                result = a / b
            
            stack.append(result)
        else:
            # Operand
            stack.append(float(token))
    
    return stack[0]

# Example: "3 4 + 2 *" → ((3 + 4) * 2) = 14
```

## 🚀 Applications

### 1. Function Call Management

```python
def factorial(n):
    """Factorial using recursion (uses call stack)."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Call stack for factorial(3):
# factorial(3) calls factorial(2)
# factorial(2) calls factorial(1)  
# factorial(1) returns 1
# factorial(2) returns 2 * 1 = 2
# factorial(3) returns 3 * 2 = 6
```

### 2. Undo/Redo Operations

```python
class TextEditor:
    def __init__(self):
        self.content = ""
        self.undo_stack = []
        self.redo_stack = []
    
    def type(self, text):
        """Type text and save state for undo."""
        self.undo_stack.append(self.content)
        self.content += text
        self.redo_stack.clear()  # Clear redo when new action
    
    def undo(self):
        """Undo last operation."""
        if self.undo_stack:
            self.redo_stack.append(self.content)
            self.content = self.undo_stack.pop()
    
    def redo(self):
        """Redo last undone operation."""
        if self.redo_stack:
            self.undo_stack.append(self.content)
            self.content = self.redo_stack.pop()
```

### 3. Browser History

```python
class BrowserHistory:
    def __init__(self, homepage):
        self.history = [homepage]
        self.current = 0
    
    def visit(self, url):
        """Visit new URL."""
        # Remove forward history
        self.history = self.history[:self.current + 1]
        self.history.append(url)
        self.current += 1
    
    def back(self, steps):
        """Go back in history."""
        self.current = max(0, self.current - steps)
        return self.history[self.current]
    
    def forward(self, steps):
        """Go forward in history."""
        self.current = min(len(self.history) - 1, self.current + steps)
        return self.history[self.current]
```

## 💡 Pro Tips

!!! tip "Memory Management"
    In languages like C++, remember to deallocate memory when popping from linked list stack to avoid memory leaks.

!!! warning "Common Mistakes"
    - **Stack underflow**: Always check if stack is empty before pop/peek
    - **Infinite recursion**: Can cause stack overflow in call stack
    - **Wrong order**: Remember LIFO - last pushed is first popped

!!! success "Best Practices"
    - Use stacks for problems involving nested structures
    - Consider monotonic stacks for "next greater/smaller" problems
    - Think about stack when you need to reverse processing order
    - Always handle empty stack edge cases

## 🚀 Next Steps

Now that you understand stack fundamentals, practice with:

- **[Easy Problems](easy-problems.md)** - Build confidence with basic stack operations
- **[Medium Problems](medium-problems.md)** - Learn advanced patterns like monotonic stacks
- **[Hard Problems](hard-problems.md)** - Master complex stack applications

---

*Ready to start practicing? Begin with the [Easy Problems](easy-problems.md) section!*
