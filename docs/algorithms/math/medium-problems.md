# Mathematical Algorithms - Medium Problems

## 🎯 Learning Objectives

Master intermediate mathematical algorithms and advanced number theory:

- Modular arithmetic and properties
- Advanced combinatorics and probability
- Optimization problems with mathematical constraints
- Complex number theory algorithms
- Mathematical pattern recognition

---

## Problem 1: Pow(x, n)

**Difficulty**: 🟡 Medium  
**Pattern**: Fast Exponentiation  
**Time**: O(log n), **Space**: O(log n)

### Problem Description

Implement `pow(x, n)`, which calculates `x` raised to the power `n` (i.e., `x^n`).

**Examples:**
```
Input: x = 2.00000, n = 10
Output: 1024.00000

Input: x = 2.00000, n = -2
Output: 0.25000
```

### Solution

```python
def myPow(x, n):
    """
    Fast exponentiation using divide and conquer
    Time: O(log n), Space: O(log n) for recursion stack
    """
    if n == 0:
        return 1.0
    
    if n < 0:
        return 1.0 / myPow(x, -n)
    
    # Divide and conquer
    half = myPow(x, n // 2)
    
    if n % 2 == 0:
        return half * half
    else:
        return half * half * x

# Iterative approach (space optimized)
def myPowIterative(x, n):
    """
    Iterative fast exponentiation
    Time: O(log n), Space: O(1)
    """
    if n == 0:
        return 1.0
    
    if n < 0:
        x = 1.0 / x
        n = -n
    
    result = 1.0
    current_power = x
    
    while n > 0:
        if n & 1:  # If n is odd
            result *= current_power
        current_power *= current_power
        n >>= 1  # n = n // 2
    
    return result

# Test
print(myPow(2.0, 10))   # 1024.0
print(myPow(2.0, -2))   # 0.25
```

### 🔍 Key Insights

- **Binary representation**: Use binary representation of exponent
- **Divide and conquer**: x^n = (x^(n/2))^2 if n is even
- **Odd exponent**: x^n = x * (x^(n-1)) if n is odd
- **Negative exponent**: x^(-n) = 1 / x^n

---

## Problem 2: Sqrt(x)

**Difficulty**: 🟡 Medium  
**Pattern**: Binary Search  
**Time**: O(log x), **Space**: O(1)

### Problem Description

Given a non-negative integer `x`, return the square root of `x` rounded down to the nearest integer.

### Solution

```python
def mySqrt(x):
    """
    Binary search for square root
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Return floor of sqrt

# Newton's method (faster convergence)
def mySqrtNewton(x):
    """
    Newton's method for square root
    """
    if x < 2:
        return x
    
    # Initial guess
    guess = x
    
    while guess * guess > x:
        guess = (guess + x // guess) // 2
    
    return guess

# Test
print(mySqrt(4))   # 2
print(mySqrt(8))   # 2
```

---

## Problem 3: Divide Two Integers

**Difficulty**: 🟡 Medium  
**Pattern**: Bit Manipulation  
**Time**: O(log n), **Space**: O(1)

### Problem Description

Given two integers `dividend` and `divisor`, divide two integers without using multiplication, division, and mod operator.

### Solution

```python
def divide(dividend, divisor):
    """
    Division using bit manipulation and subtraction
    """
    # Handle overflow
    MAX_INT = 2**31 - 1
    MIN_INT = -2**31
    
    if dividend == MIN_INT and divisor == -1:
        return MAX_INT
    
    # Determine sign
    negative = (dividend < 0) ^ (divisor < 0)
    
    # Work with positive numbers
    dividend = abs(dividend)
    divisor = abs(divisor)
    
    quotient = 0
    
    while dividend >= divisor:
        # Find largest multiple of divisor that fits in dividend
        temp_divisor = divisor
        multiple = 1
        
        while dividend >= (temp_divisor << 1):
            temp_divisor <<= 1
            multiple <<= 1
        
        dividend -= temp_divisor
        quotient += multiple
    
    return -quotient if negative else quotient

# Test
print(divide(10, 3))   # 3
print(divide(7, -3))   # -2
```

---

## Problem 4: Fraction to Recurring Decimal

**Difficulty**: 🟡 Medium  
**Pattern**: Long Division Simulation  
**Time**: O(denominator), **Space**: O(denominator)

### Problem Description

Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

### Solution

```python
def fractionToDecimal(numerator, denominator):
    """
    Simulate long division to detect repeating decimals
    """
    if numerator == 0:
        return "0"
    
    result = []
    
    # Handle sign
    if (numerator < 0) ^ (denominator < 0):
        result.append("-")
    
    # Work with positive numbers
    numerator = abs(numerator)
    denominator = abs(denominator)
    
    # Integer part
    result.append(str(numerator // denominator))
    numerator %= denominator
    
    if numerator == 0:
        return "".join(result)
    
    # Fractional part
    result.append(".")
    remainder_positions = {}
    
    while numerator != 0:
        if numerator in remainder_positions:
            # Repeating decimal found
            index = remainder_positions[numerator]
            result.insert(index, "(")
            result.append(")")
            break
        
        remainder_positions[numerator] = len(result)
        numerator *= 10
        result.append(str(numerator // denominator))
        numerator %= denominator
    
    return "".join(result)

# Test
print(fractionToDecimal(1, 2))   # "0.5"
print(fractionToDecimal(2, 1))   # "2"
print(fractionToDecimal(2, 3))   # "0.(6)"
print(fractionToDecimal(4, 333)) # "0.(012)"
```

---

## Problem 5: Excel Sheet Column Title

**Difficulty**: 🟡 Medium  
**Pattern**: Base Conversion  
**Time**: O(log n), **Space**: O(log n)

### Problem Description

Given an integer `columnNumber`, return its corresponding column title as it appears in an Excel sheet.

**Examples:**
```
1 -> A
2 -> B
...
26 -> Z
27 -> AA
28 -> AB
```

### Solution

```python
def convertToTitle(columnNumber):
    """
    Convert decimal to bijective base-26 (A=1, B=2, ..., Z=26)
    """
    result = []
    
    while columnNumber > 0:
        # Adjust for 1-indexed (A=1, not A=0)
        columnNumber -= 1
        
        # Get the character
        result.append(chr(ord('A') + columnNumber % 26))
        columnNumber //= 26
    
    return ''.join(reversed(result))

# Alternative approach
def convertToTitleAlt(columnNumber):
    """
    Build string from right to left
    """
    result = ""
    
    while columnNumber > 0:
        columnNumber -= 1  # Convert to 0-indexed
        result = chr(ord('A') + columnNumber % 26) + result
        columnNumber //= 26
    
    return result

# Test
print(convertToTitle(1))    # "A"
print(convertToTitle(28))   # "AB"
print(convertToTitle(701))  # "ZY"
```

---

## Problem 6: Valid Number

**Difficulty**: 🟡 Medium  
**Pattern**: Finite State Machine  
**Time**: O(n), **Space**: O(1)

### Problem Description

Determine if a given string `s` is a valid number.

A valid number can be split into these components (in order):
1. A decimal number or an integer
2. (Optional) An 'e' or 'E', followed by an integer

### Solution

```python
def isNumber(s):
    """
    Use finite state machine to validate number format
    """
    s = s.strip()
    if not s:
        return False
    
    i = 0
    n = len(s)
    
    # Skip leading sign
    if i < n and s[i] in '+-':
        i += 1
    
    # Count digits and decimal points before 'e'/'E'
    digits = 0
    decimal_points = 0
    
    while i < n and s[i] not in 'eE':
        if s[i].isdigit():
            digits += 1
        elif s[i] == '.':
            decimal_points += 1
        else:
            return False
        i += 1
    
    # Must have digits and at most one decimal point
    if digits == 0 or decimal_points > 1:
        return False
    
    # Check exponent part if exists
    if i < n and s[i] in 'eE':
        i += 1
        
        # Skip sign after 'e'/'E'
        if i < n and s[i] in '+-':
            i += 1
        
        # Must have digits after 'e'/'E'
        if i >= n:
            return False
        
        while i < n:
            if not s[i].isdigit():
                return False
            i += 1
    
    return True

# Regex approach (more concise)
import re

def isNumberRegex(s):
    """
    Use regex to match valid number patterns
    """
    pattern = r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$'
    return bool(re.match(pattern, s.strip()))

# Test
print(isNumber("0"))      # True
print(isNumber("e"))      # False
print(isNumber("."))      # False
print(isNumber("2e10"))   # True
print(isNumber("-90e3"))  # True
```

---

## Problem 7: Integer to Roman

**Difficulty**: 🟡 Medium  
**Pattern**: Greedy Algorithm  
**Time**: O(1), **Space**: O(1)

### Problem Description

Convert an integer to a Roman numeral. Roman numerals use these symbols and values:

- I = 1, V = 5, X = 10, L = 50, C = 100, D = 500, M = 1000
- Subtraction rules: IV = 4, IX = 9, XL = 40, XC = 90, CD = 400, CM = 900

### Solution

```python
def intToRoman(num):
    """
    Greedy approach: use largest possible values first
    """
    # Values and symbols in descending order
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    
    result = []
    
    for i in range(len(values)):
        count = num // values[i]
        if count:
            result.append(symbols[i] * count)
            num -= values[i] * count
    
    return ''.join(result)

# Alternative: Dictionary-based approach
def intToRomanDict(num):
    """
    Use dictionary for cleaner code
    """
    mapping = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    
    result = ''
    for value, symbol in mapping:
        count = num // value
        result += symbol * count
        num -= value * count
    
    return result

# Test
print(intToRoman(3))      # "III"
print(intToRoman(4))      # "IV"
print(intToRoman(9))      # "IX"
print(intToRoman(58))     # "LVIII"
print(intToRoman(1994))   # "MCMXCIV"
```

---

## Problem 8: Multiply Strings

**Difficulty**: 🟡 Medium  
**Pattern**: Simulation  
**Time**: O(m×n), **Space**: O(m+n)

### Problem Description

Given two non-negative integers `num1` and `num2` represented as strings, return the product of `num1` and `num2`, also represented as a string.

### Solution

```python
def multiply(num1, num2):
    """
    Simulate multiplication algorithm
    """
    if num1 == "0" or num2 == "0":
        return "0"
    
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    
    # Reverse both strings for easier calculation
    num1 = num1[::-1]
    num2 = num2[::-1]
    
    for i in range(m):
        for j in range(n):
            # Multiply digits
            product = int(num1[i]) * int(num2[j])
            
            # Add to result array
            result[i + j] += product
            
            # Handle carry
            if result[i + j] >= 10:
                result[i + j + 1] += result[i + j] // 10
                result[i + j] %= 10
    
    # Remove leading zeros and convert to string
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return ''.join(map(str, result[::-1]))

# Test
print(multiply("2", "3"))     # "6"
print(multiply("123", "456")) # "56088"
```

---

## Problem 9: Angle Between Clock Hands

**Difficulty**: 🟡 Medium  
**Pattern**: Mathematical Calculation  
**Time**: O(1), **Space**: O(1)

### Problem Description

Given a time in 24-hour format, calculate the angle between the hour and minute hands of a clock.

### Solution

```python
def angleClock(hour, minutes):
    """
    Calculate angle between clock hands
    """
    # Convert to 12-hour format
    hour = hour % 12
    
    # Calculate angles from 12 o'clock position
    # Minute hand: 6 degrees per minute
    minute_angle = minutes * 6
    
    # Hour hand: 30 degrees per hour + 0.5 degrees per minute
    hour_angle = hour * 30 + minutes * 0.5
    
    # Calculate absolute difference
    angle = abs(hour_angle - minute_angle)
    
    # Return smaller angle
    return min(angle, 360 - angle)

# Test
print(angleClock(12, 30))  # 165.0
print(angleClock(3, 30))   # 75.0
print(angleClock(3, 15))   # 7.5
```

---

## 📝 Summary

### Advanced Mathematical Patterns

1. **Fast Exponentiation** - O(log n) power calculation using binary representation
2. **Binary Search on Answer** - Find square roots, division without operators
3. **Simulation Algorithms** - Long division, multiplication, clock calculations
4. **Base Conversion** - Handle different numbering systems efficiently
5. **State Machines** - Validate complex input formats
6. **Greedy Algorithms** - Roman numeral conversion using largest values first

### Key Algorithmic Insights

| **Problem Type** | **Key Technique** | **Time Complexity** |
|------------------|-------------------|-------------------|
| **Exponentiation** | Divide and conquer | O(log n) |
| **Division** | Bit manipulation | O(log n) |
| **Base Conversion** | Modular arithmetic | O(log n) |
| **String Multiplication** | Digit-by-digit simulation | O(m×n) |
| **Decimal Detection** | Hash map for remainder tracking | O(denominator) |
| **Validation** | Finite state machine | O(n) |

### Mathematical Properties Used

- **Binary representation**: Powers of 2 for fast exponentiation
- **Modular arithmetic**: Base conversion, cycle detection
- **Greedy choice**: Roman numerals, largest values first
- **Simulation**: Long division, multiplication algorithms
- **Geometric properties**: Clock angle calculations

### Optimization Techniques

1. **Space-time tradeoffs**: Iterative vs recursive implementations
2. **Early termination**: Stop when answer is found
3. **Mathematical shortcuts**: Use properties to avoid brute force
4. **Efficient data structures**: Hash maps for cycle detection

---

Ready for the ultimate mathematical challenge? Move on to **[Hard Math Problems](hard-problems.md)** to master advanced number theory, complex optimizations, and mathematical algorithms used in competitive programming!

### 📚 What's Next

- **[Hard Problems](hard-problems.md)** - Advanced mathematical algorithms
- **[Number Theory](number-theory.md)** - Deep mathematical concepts
- **[Geometry](geometry.md)** - Computational geometry problems
- **[Combinatorics](combinatorics.md)** - Counting and probability
