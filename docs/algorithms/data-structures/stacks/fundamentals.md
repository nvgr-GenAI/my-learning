# Stacks: Fundamentals & Theory

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

## 💻 Implementation Approaches

Stacks can be implemented using different underlying data structures, each with its own trade-offs:

### 🔹 Array-Based Implementation

**Advantages:**

- Simple and intuitive
- Excellent cache performance
- Lower memory overhead per element
- Fast operations due to memory locality

**Disadvantages:**

- May need to resize when capacity is exceeded
- Fixed maximum size (for static arrays)
- Potential memory waste if over-allocated

**Best for:** When you have predictable size requirements and want maximum performance

👉 **[Learn Array-Based Stack Implementation](array-stack.md)**

### 🔹 Linked List Implementation

**Advantages:**

- Truly dynamic size
- No need for resizing operations
- Efficient memory usage (allocate only what's needed)
- No maximum size limit

**Disadvantages:**

- Higher memory overhead per element (pointers)
- Potential cache misses due to scattered memory
- Slightly more complex implementation

**Best for:** When stack size is highly unpredictable or memory is constrained

👉 **[Learn Linked List Stack Implementation](linked-list-stack.md)**

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

## ⚖️ Implementation Comparison

| **Feature** | **Array-based** | **Linked List** |
|------------|-----------------|-----------------|
| **Memory** | Contiguous | Scattered |
| **Cache Performance** | Better | Worse |
| **Memory Overhead** | Lower | Higher (pointers) |
| **Dynamic Size** | Limited by capacity | Unlimited |
| **Implementation** | Simpler | More complex |
| **Resize Cost** | O(n) occasionally | Never needed |

## 🎯 Choosing the Right Implementation

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

## 🚀 Core Applications

Stacks are fundamental in many areas of computer science:

### 1. Function Call Management

- **Call Stack**: Programming languages use stacks to manage function calls
- **Recursion**: Each recursive call is pushed onto the call stack
- **Stack Frames**: Store local variables and return addresses
- **Stack Overflow**: When recursion depth exceeds stack capacity

### 2. Expression Processing

- **Infix to Postfix**: Convert mathematical expressions
- **Expression Evaluation**: Evaluate postfix/prefix expressions
- **Operator Precedence**: Handle operator priorities
- **Parentheses Matching**: Validate balanced brackets

### 3. Undo/Redo Operations

- **Text Editors**: Track state changes for undo functionality
- **Image Editors**: Layer operations and transformations
- **Database Transactions**: Rollback operations
- **Game State**: Save/restore game states

### 4. Parsing and Compilation

- **Syntax Analysis**: Parse nested language constructs
- **Symbol Tables**: Manage variable scopes
- **Code Generation**: Generate assembly code
- **Error Recovery**: Handle syntax errors gracefully

### 5. Memory Management

- **Stack Memory**: Automatic variable allocation
- **Garbage Collection**: Mark and sweep algorithms
- **Activation Records**: Function call overhead
- **Stack-based Virtual Machines**: JVM, .NET CLR

## 🎯 Problem-Solving Patterns

Understanding these patterns will help you recognize when to use stacks:

### 1. **LIFO Processing**

- When you need to process elements in reverse order
- Backtracking algorithms
- Reversing sequences

### 2. **Nested Structures**

- Matching parentheses, brackets, braces
- HTML/XML tag validation
- Mathematical expression parsing

### 3. **Monotonic Stack**

- Next greater/smaller element problems
- Histogram problems (largest rectangle)
- Stock span problems

### 4. **State Management**

- Undo/redo functionality
- Game state management
- Browser history

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

Now that you understand stack fundamentals, choose your learning path:

### 1. **Learn Implementations**

- **[Array-Based Stack](array-stack.md)** - Simple, cache-friendly implementation
- **[Linked List Stack](linked-list-stack.md)** - Dynamic, flexible implementation

### 2. **Practice Problems**

- **[Easy Problems](easy-problems.md)** - Build confidence with basic stack operations
- **[Medium Problems](medium-problems.md)** - Learn advanced patterns like monotonic stacks
- **[Hard Problems](hard-problems.md)** - Master complex stack applications

---

*Ready to dive deeper? Start with **[Array-Based Stack](array-stack.md)** for a solid foundation!*
