# Expression Evaluation Using Stacks

Expression evaluation is one of the most practical applications of stacks. Whether parsing mathematical expressions, programming language syntax, or configuration formats, stacks provide an elegant solution for handling nested structures and operator precedence.

## Overview

Stacks are ideal for expression evaluation because they naturally handle the Last-In-First-Out (LIFO) nature of nested expressions and operator precedence. Common expression evaluation algorithms using stacks include:

1. **Infix to Postfix Conversion**
2. **Postfix Expression Evaluation**
3. **Infix Expression Evaluation**
4. **Balanced Parentheses Checking**

## Infix to Postfix Conversion

Infix notation (`a + b`) is what humans typically use, while postfix notation (`a b +`) is easier for computers to evaluate. Converting from infix to postfix is a classic stack application:

```python
def infix_to_postfix(expression):
    # Define operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    stack = []
    postfix = []
    
    for char in expression:
        # If operand, add to output
        if char.isalnum():
            postfix.append(char)
        
        # If left parenthesis, push to stack
        elif char == '(':
            stack.append(char)
        
        # If right parenthesis, pop until matching left parenthesis
        elif char == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            
            # Remove the '(' from stack
            if stack and stack[-1] == '(':
                stack.pop()
        
        # If operator, handle based on precedence
        else:
            while (stack and stack[-1] != '(' and 
                   stack[-1] in precedence and 
                   precedence.get(stack[-1], 0) >= precedence.get(char, 0)):
                postfix.append(stack.pop())
            stack.append(char)
    
    # Pop any remaining operators
    while stack:
        postfix.append(stack.pop())
    
    return ''.join(postfix)
```

**Time Complexity**: O(n) - Each character is processed once
**Space Complexity**: O(n) - In the worst case, all operators might be on the stack

## Postfix Expression Evaluation

Once we have a postfix expression, evaluating it is straightforward using a stack:

```python
def evaluate_postfix(expression):
    stack = []
    
    for char in expression:
        # If operand, push to stack
        if char.isdigit():
            stack.append(int(char))
        
        # If operator, pop operands and apply
        else:
            if len(stack) < 2:
                raise ValueError("Invalid expression")
                
            b = stack.pop()  # Second operand
            a = stack.pop()  # First operand
            
            if char == '+':
                stack.append(a + b)
            elif char == '-':
                stack.append(a - b)
            elif char == '*':
                stack.append(a * b)
            elif char == '/':
                stack.append(a // b if a * b > 0 else -(abs(a) // abs(b)))
            elif char == '^':
                stack.append(a ** b)
    
    # Result should be the only item left on the stack
    if len(stack) != 1:
        raise ValueError("Invalid expression")
    
    return stack.pop()
```

**Time Complexity**: O(n) - Each character is processed once
**Space Complexity**: O(n) - In the worst case, all operands might be on the stack

## Direct Infix Expression Evaluation

We can also evaluate infix expressions directly using two stacks (one for operators, one for operands):

```python
def evaluate_infix(expression):
    # Remove spaces
    expression = expression.replace(" ", "")
    
    # Define operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    operands = []
    operators = []
    
    i = 0
    while i < len(expression):
        # If current char is a digit
        if expression[i].isdigit():
            # Parse multi-digit numbers
            j = i
            while j < len(expression) and expression[j].isdigit():
                j += 1
            operands.append(int(expression[i:j]))
            i = j
            continue
        
        # If current char is an operator
        elif expression[i] in "+-*/^":
            while (operators and operators[-1] != '(' and 
                   operators[-1] in precedence and 
                   precedence.get(operators[-1], 0) >= precedence.get(expression[i], 0)):
                apply_operator(operands, operators.pop())
            operators.append(expression[i])
        
        # If left parenthesis
        elif expression[i] == '(':
            operators.append(expression[i])
        
        # If right parenthesis
        elif expression[i] == ')':
            while operators and operators[-1] != '(':
                apply_operator(operands, operators.pop())
            
            # Remove the '(' from stack
            if operators and operators[-1] == '(':
                operators.pop()
        
        i += 1
    
    # Apply remaining operators
    while operators:
        apply_operator(operands, operators.pop())
    
    # Result should be the only item left in operands
    if len(operands) != 1:
        raise ValueError("Invalid expression")
    
    return operands.pop()

def apply_operator(operands, operator):
    if len(operands) < 2:
        raise ValueError("Invalid expression")
    
    b = operands.pop()  # Second operand
    a = operands.pop()  # First operand
    
    if operator == '+':
        operands.append(a + b)
    elif operator == '-':
        operands.append(a - b)
    elif operator == '*':
        operands.append(a * b)
    elif operator == '/':
        operands.append(a // b if a * b > 0 else -(abs(a) // abs(b)))
    elif operator == '^':
        operands.append(a ** b)
```

**Time Complexity**: O(n) - Each character is processed once
**Space Complexity**: O(n) - The operator and operand stacks combined may hold up to n elements

## Handling Complex Expressions

Real-world expressions often require additional considerations:

### Function Calls

```python
def evaluate_with_functions(expression):
    # Similar to infix evaluation, but also handle function calls
    # e.g., "max(2, 3 + 4)" or "sin(30)"
    
    # Implementation would include function name recognition
    # and argument parsing using stacks
    
    # This is a simplified conceptual example
    pass
```

### Variable Substitution

```python
def evaluate_with_variables(expression, variables):
    # Replace variable names with their values
    for var_name, value in variables.items():
        expression = expression.replace(var_name, str(value))
    
    # Then evaluate using standard infix evaluation
    return evaluate_infix(expression)
```

### Error Handling

Robust expression evaluators need to handle various error conditions:

```python
def safe_evaluate(expression):
    try:
        # Check for balanced parentheses
        if not is_balanced(expression):
            return "Unbalanced parentheses"
        
        # Check for invalid characters
        if any(c not in "0123456789+-*/^() " for c in expression):
            return "Invalid character in expression"
        
        # Check for division by zero
        if '/0' in expression.replace(' ', ''):
            return "Division by zero"
        
        # Evaluate expression
        return evaluate_infix(expression)
    
    except Exception as e:
        return f"Error: {str(e)}"
```

## Shunting Yard Algorithm

The Shunting Yard algorithm, developed by Edsger Dijkstra, is a more formalized approach to parsing mathematical expressions:

```python
def shunting_yard(expression):
    # Define operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    associativity = {'+': 'left', '-': 'left', '*': 'left', '/': 'left', '^': 'right'}
    
    output_queue = []  # For postfix expression
    operator_stack = []
    
    tokens = tokenize(expression)  # Split expression into tokens
    
    for token in tokens:
        if token.isdigit():
            output_queue.append(token)
        
        elif token in precedence:  # It's an operator
            while (operator_stack and operator_stack[-1] != '(' and 
                  operator_stack[-1] in precedence and 
                  ((associativity[token] == 'left' and 
                    precedence[operator_stack[-1]] >= precedence[token]) or
                   (associativity[token] == 'right' and
                    precedence[operator_stack[-1]] > precedence[token]))):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        
        elif token == '(':
            operator_stack.append(token)
        
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            
            # Pop the '(' from the stack
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()
    
    # Pop any remaining operators
    while operator_stack:
        if operator_stack[-1] == '(':
            raise ValueError("Mismatched parentheses")
        output_queue.append(operator_stack.pop())
    
    return output_queue
```

**Time Complexity**: O(n) - Each token is processed once
**Space Complexity**: O(n) - In the worst case, all operators might be on the stack

## Applications of Expression Evaluation

1. **Calculators**: Scientific and programmable calculators use expression evaluation algorithms.

2. **Spreadsheets**: Formula evaluation in spreadsheets relies on expression parsing.

3. **Programming Language Compilers**: Parsing expressions is a key component of compilation.

4. **Database Query Optimizers**: SQL WHERE clauses are parsed and evaluated as expressions.

5. **Configuration Systems**: Many configuration formats support expressions for dynamic values.

## Conclusion

Expression evaluation using stacks demonstrates the power of this simple data structure in solving complex parsing problems. By understanding these algorithms, you can implement parsers for custom languages, evaluate mathematical expressions, and handle nested structures in various applications. The combination of infix-to-postfix conversion and postfix evaluation provides an elegant and efficient approach to expression handling.
