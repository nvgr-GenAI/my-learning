# Parentheses Matching Using Stacks

Parentheses matching is a classic problem that demonstrates the elegance of stack-based solutions. This fundamental operation is critical in parsing various text formats including programming languages, mathematical expressions, and markup languages.

## Overview

The parentheses matching problem involves determining whether a string containing various types of brackets (parentheses `()`, square brackets `[]`, curly braces `{}`) has all its brackets properly matched and nested. A string is considered to have valid parentheses if:

1. Every opening bracket must have a corresponding closing bracket of the same type.
2. Opening brackets must be closed in the correct order.
3. Every closing bracket must have a corresponding opening bracket of the same type.

## Stack-Based Solution

A stack is the perfect data structure for this problem because it naturally tracks the expected closing brackets in the correct order:

```python
def is_balanced(expression):
    """
    Check if the parentheses in an expression are balanced.
    
    Args:
        expression: A string containing parentheses, brackets, braces
        
    Returns:
        True if all parentheses are properly matched, False otherwise
    """
    stack = []
    
    # Dictionary to map closing brackets to their opening counterparts
    brackets_map = {')': '(', ']': '[', '}': '{'}
    
    # Set of opening brackets
    opening_brackets = set(['(', '[', '{'])
    
    for char in expression:
        # If it's an opening bracket, push to stack
        if char in opening_brackets:
            stack.append(char)
        
        # If it's a closing bracket
        elif char in brackets_map:
            # If stack is empty or the top doesn't match, it's unbalanced
            if not stack or stack.pop() != brackets_map[char]:
                return False
    
    # If stack is empty, all brackets were matched
    return len(stack) == 0
```

**Time Complexity**: O(n) - We process each character exactly once
**Space Complexity**: O(n) - In the worst case, the stack may contain all characters (e.g., all opening brackets)

## Extended Applications

### Validating Nested Structures

We can extend the basic parentheses matching to validate more complex nested structures:

```python
def validate_nested_structure(code):
    stack = []
    line_number = 1
    column_number = 1
    
    # Store positions for better error reporting
    positions = []
    
    for char in code:
        if char in '({[':
            stack.append(char)
            positions.append((line_number, column_number))
        
        elif char in ')}]':
            if not stack:
                return False, f"Unexpected closing bracket at line {line_number}, column {column_number}"
            
            opening_bracket = stack.pop()
            opening_pos = positions.pop()
            
            # Check matching pairs
            if (opening_bracket == '(' and char != ')') or \
               (opening_bracket == '[' and char != ']') or \
               (opening_bracket == '{' and char != '}'):
                return False, f"Mismatched brackets at line {line_number}, column {column_number}. " \
                             f"Found '{char}' but expected matching pair for '{opening_bracket}' " \
                             f"from line {opening_pos[0]}, column {opening_pos[1]}"
        
        # Update line and column numbers
        if char == '\n':
            line_number += 1
            column_number = 1
        else:
            column_number += 1
    
    # Check if all brackets were closed
    if stack:
        opening_pos = positions[0]
        opening_bracket = stack[0]
        return False, f"Unclosed '{opening_bracket}' from line {opening_pos[0]}, column {opening_pos[1]}"
    
    return True, "Structure is valid"
```

### Checking Expression Validity

Beyond just matching parentheses, we can validate entire expressions:

```python
def validate_expression(expression):
    stack = []
    
    for i, char in enumerate(expression):
        if char in '({[':
            stack.append((char, i))
        
        elif char in ')}]':
            if not stack:
                return False, f"Unexpected closing bracket at position {i}"
            
            opening_bracket, opening_pos = stack.pop()
            
            # Check matching pairs
            if (opening_bracket == '(' and char != ')') or \
               (opening_bracket == '[' and char != ']') or \
               (opening_bracket == '{' and char != '}'):
                return False, f"Mismatched brackets: Found '{char}' at position {i} " \
                             f"but expected matching pair for '{opening_bracket}' from position {opening_pos}"
        
        # Check for operator validity (simplified example)
        elif char in '+-*/':
            if i == 0 or expression[i-1] in '(+-*/':
                return False, f"Invalid operator placement at position {i}"
    
    # Check if all brackets were closed
    if stack:
        opening_bracket, opening_pos = stack[0]
        return False, f"Unclosed '{opening_bracket}' from position {opening_pos}"
    
    return True, "Expression is valid"
```

## Real-World Applications

### Syntax Checking in Compilers and Interpreters

Programming language parsers use parentheses matching to validate code structure:

```python
def check_syntax(code):
    # Stack for tracking brackets
    bracket_stack = []
    
    # Stack for tracking blocks (if/for/while/etc.)
    block_stack = []
    
    lines = code.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        for col_num, char in enumerate(line, 1):
            if char in '({[':
                bracket_stack.append((char, line_num, col_num))
            
            elif char in ')}]':
                if not bracket_stack:
                    return f"Unexpected closing bracket at line {line_num}, column {col_num}"
                
                opening_bracket, opening_line, opening_col = bracket_stack.pop()
                
                # Check matching pairs
                if not is_matching_pair(opening_bracket, char):
                    return f"Mismatched brackets: Found '{char}' at line {line_num}, column {col_num} " \
                          f"but expected matching pair for '{opening_bracket}' from line {opening_line}, column {opening_col}"
        
        # Simplified block checking (e.g., if/endif, for/next)
        if line.strip().startswith('if '):
            block_stack.append(('if', line_num))
        elif line.strip() == 'endif':
            if not block_stack or block_stack[-1][0] != 'if':
                return f"Unexpected 'endif' at line {line_num}"
            block_stack.pop()
    
    # Check unclosed brackets
    if bracket_stack:
        opening_bracket, opening_line, opening_col = bracket_stack[0]
        return f"Unclosed '{opening_bracket}' from line {opening_line}, column {opening_col}"
    
    # Check unclosed blocks
    if block_stack:
        block_type, block_line = block_stack[0]
        return f"Unclosed '{block_type}' block from line {block_line}"
    
    return "Syntax is valid"

def is_matching_pair(opening, closing):
    return (opening == '(' and closing == ')') or \
           (opening == '[' and closing == ']') or \
           (opening == '{' and closing == '}')
```

### XML and HTML Validation

Markup languages require proper nesting of tags, which is essentially a parentheses matching problem:

```python
def validate_markup(markup):
    stack = []
    
    i = 0
    while i < len(markup):
        # Find opening tags
        if markup[i:i+1] == '<' and markup[i+1:i+2] != '/':
            # Extract tag name
            j = i + 1
            while j < len(markup) and markup[j] not in ' >':
                j += 1
            
            tag_name = markup[i+1:j]
            
            # Skip to end of tag
            while j < len(markup) and markup[j] != '>':
                j += 1
            
            # If not self-closing tag
            if markup[j-1] != '/':
                stack.append(tag_name)
            
            i = j + 1
        
        # Find closing tags
        elif markup[i:i+2] == '</':
            # Extract tag name
            j = i + 2
            while j < len(markup) and markup[j] != '>':
                j += 1
            
            tag_name = markup[i+2:j]
            
            # Check if matches top of stack
            if not stack or stack.pop() != tag_name:
                return False, f"Mismatched tag: Found closing tag '{tag_name}' without matching opening tag"
            
            i = j + 1
        
        else:
            i += 1
    
    # Check if all tags were closed
    if stack:
        return False, f"Unclosed tags: {', '.join(stack)}"
    
    return True, "Markup is valid"
```

## Handling Edge Cases

A robust parentheses matcher needs to handle various edge cases:

### Ignoring Quoted Strings

In programming contexts, brackets inside strings should be ignored:

```python
def is_balanced_with_strings(code):
    stack = []
    in_string = False
    string_delimiter = None
    
    i = 0
    while i < len(code):
        # Handle string delimiters
        if (code[i] == '"' or code[i] == "'") and (i == 0 or code[i-1] != '\\'):
            if not in_string:
                # Start of string
                in_string = True
                string_delimiter = code[i]
            elif code[i] == string_delimiter:
                # End of string
                in_string = False
        
        # Only check brackets if not in a string
        elif not in_string:
            if code[i] in '({[':
                stack.append(code[i])
            elif code[i] in ')}]':
                if not stack:
                    return False
                
                # Check for matching bracket
                top = stack.pop()
                if not is_matching_pair(top, code[i]):
                    return False
        
        i += 1
    
    # If all brackets were matched, stack should be empty
    return len(stack) == 0
```

### Handling Comments

Similarly, brackets in comments should be ignored:

```python
def is_balanced_with_comments(code):
    stack = []
    i = 0
    in_comment = False
    
    while i < len(code):
        # Check for comment start
        if i < len(code) - 1 and code[i:i+2] == '/*':
            in_comment = True
            i += 2
            continue
        
        # Check for comment end
        if in_comment and i < len(code) - 1 and code[i:i+2] == '*/':
            in_comment = False
            i += 2
            continue
        
        # Ignore characters in comments
        if in_comment:
            i += 1
            continue
        
        # Handle brackets
        if code[i] in '({[':
            stack.append(code[i])
        elif code[i] in ')}]':
            if not stack:
                return False
            
            # Check for matching bracket
            top = stack.pop()
            if not is_matching_pair(top, code[i]):
                return False
        
        i += 1
    
    # If all brackets were matched, stack should be empty
    return len(stack) == 0 and not in_comment
```

## Performance Optimization

For very long inputs, we can optimize our approach:

```python
def is_balanced_optimized(expression):
    # Early termination: If length is odd, it can't be balanced
    if len(expression) % 2 != 0:
        return False
    
    # Count of each bracket type
    counts = {'(': 0, ')': 0, '[': 0, ']': 0, '{': 0, '}': 0}
    
    # Preflight check: Count all brackets
    for char in expression:
        if char in counts:
            counts[char] += 1
    
    # Quick check: Each opening bracket should have a closing counterpart
    if counts['('] != counts[')'] or counts['['] != counts[']'] or counts['{'] != counts['}']:
        return False
    
    # If counts match, perform full validation
    stack = []
    brackets_map = {')': '(', ']': '[', '}': '{'}
    
    for char in expression:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != brackets_map[char]:
                return False
    
    return len(stack) == 0
```

## Conclusion

Parentheses matching is a quintessential example of how stacks provide elegant solutions to common programming problems. The algorithm's simplicity, efficiency, and versatility make it a fundamental tool in text processing, compiler design, and syntax validation. Understanding this pattern is essential for developers working on any kind of parser or text processor, and it serves as an excellent introduction to stack-based algorithms in general.
