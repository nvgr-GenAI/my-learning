# Mathematical Algorithms - Medium Problems

## 🎯 Learning Objectives

Master intermediate mathematical algorithms and advanced number theory:

- Modular arithmetic and properties
- Advanced combinatorics and probability
- Optimization problems with mathematical constraints
- Complex number theory algorithms
- Mathematical pattern recognition

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Pow(x, n) | Fast Exponentiation | Medium | O(log n) | O(log n) |
    | 2 | Sqrt(x) | Binary Search/Math | Medium | O(log n) | O(1) |
    | 3 | Divide Two Integers | Bit Manipulation | Medium | O(log n) | O(1) |
    | 4 | Fraction to Recurring Decimal | Hash Map/Math | Medium | O(n) | O(n) |
    | 5 | Excel Sheet Column Title | Base Conversion | Medium | O(log n) | O(1) |
    | 6 | Valid Number | String Parsing | Medium | O(n) | O(1) |
    | 7 | Integer to Roman | Greedy/Math | Medium | O(1) | O(1) |
    | 8 | Multiply Strings | String Math | Medium | O(mn) | O(m+n) |
    | 9 | Angle Between Clock Hands | Geometry/Math | Medium | O(1) | O(1) |
    | 10 | Factorial Trailing Zeroes | Math/Number Theory | Medium | O(log n) | O(1) |
    | 11 | Gray Code | Bit Manipulation | Medium | O(2^n) | O(2^n) |
    | 12 | Unique Paths | DP/Combinatorics | Medium | O(mn) | O(mn) |
    | 13 | Combination Sum | Backtracking/Math | Medium | O(2^n) | O(n) |
    | 14 | Next Permutation | Array/Math | Medium | O(n) | O(1) |
    | 15 | Permutations | Backtracking | Medium | O(n!) | O(n) |

=== "🎯 Advanced Math Patterns"

    **🔢 Fast Exponentiation:**
    - Divide and conquer approach
    - Handle negative exponents
    - Iterative vs recursive solutions
    
    **🔍 Binary Search on Answer:**
    - Square root approximation
    - Finding optimal solutions
    - Precision handling for floating point
    
    **🧮 String Mathematics:**
    - Large number arithmetic
    - Base conversion algorithms
    - Parsing and validation

=== "📚 Algorithm Templates"

    **Fast Exponentiation Template:**
    ```python
    def fast_pow(base, exp):
        if exp == 0:
            return 1
        if exp < 0:
            return 1 / fast_pow(base, -exp)
        
        half = fast_pow(base, exp // 2)
        if exp % 2 == 0:
            return half * half
        else:
            return half * half * base
    ```
    
    **Binary Search Template:**
    ```python
    def binary_search_answer(left, right, condition):
        while left < right:
            mid = (left + right) // 2
            if condition(mid):
                right = mid
            else:
                left = mid + 1
        return left
    ```
    
    **Combinatorics Template:**
    ```python
    def combination(n, r):
        if r > n - r:
            r = n - r
        
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result
    ```

---

## Problem 1: Pow(x, n)

**Difficulty**: 🟡 Medium  
**Pattern**: Fast Exponentiation  
**Time**: O(log n), **Space**: O(log n)

=== "Problem"

    Implement `pow(x, n)`, which calculates `x` raised to the power `n` (i.e., `x^n`).

    **Examples:**
    ```
    Input: x = 2.00000, n = 10
    Output: 1024.00000

    Input: x = 2.00000, n = -2
    Output: 0.25000

    Input: x = 2.00000, n = 0
    Output: 1.00000
    ```

=== "Solution"

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

    # Iterative approach
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
            if n % 2 == 1:
                result *= current_power
            current_power *= current_power
            n //= 2
        
        return result

    # Test
    print(myPow(2.0, 10))   # 1024.0
    print(myPow(2.0, -2))   # 0.25
    ```

=== "Insights"

    **Key Insights:**
    - **Divide and conquer**: Reduce problem size by half each time
    - **Negative exponents**: Convert to positive and take reciprocal
    - **Even/odd handling**: Square the half result, multiply by base if odd
    - **Iterative optimization**: Avoid recursion stack for large n

---

## Problem 2: Sqrt(x)

**Difficulty**: 🟡 Medium  
**Pattern**: Binary Search/Math  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Given a non-negative integer `x`, compute and return the square root of `x`.

    Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

    **Examples:**
    ```
    Input: x = 4
    Output: 2

    Input: x = 8
    Output: 2 (sqrt(8) = 2.828...)
    ```

=== "Solution"

    ```python
    def mySqrt(x):
        """
        Binary search approach
        Time: O(log x), Space: O(1)
        """
        if x < 2:
            return x
        
        left, right = 2, x // 2
        
        while left <= right:
            mid = (left + right) // 2
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                left = mid + 1
            else:
                right = mid - 1
        
        return right

    # Newton's method
    def mySqrtNewton(x):
        """
        Newton's method for finding square root
        Time: O(log x), Space: O(1)
        """
        if x < 2:
            return x
        
        # Start with x/2 as initial guess
        result = x
        
        while result * result > x:
            result = (result + x // result) // 2
        
        return result

    # Test
    print(mySqrt(4))   # 2
    print(mySqrt(8))   # 2
    print(mySqrtNewton(8))   # 2
    ```

=== "Insights"

    **Key Insights:**
    - **Binary search**: Search space is from 2 to x/2
    - **Newton's method**: Faster convergence with iterative refinement
    - **Integer truncation**: Return floor of the square root
    - **Optimization**: For x >= 2, sqrt(x) <= x/2

---

## Problem 3: Divide Two Integers

**Difficulty**: 🟡 Medium  
**Pattern**: Bit Manipulation  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Given two integers `dividend` and `divisor`, divide two integers without using multiplication, division, and mod operator.

    The integer division should truncate toward zero.

    **Examples:**
    ```
    Input: dividend = 10, divisor = 3
    Output: 3

    Input: dividend = 7, divisor = -3
    Output: -2
    ```

=== "Solution"

    ```python
    def divide(dividend, divisor):
        """
        Division using bit manipulation and exponential search
        Time: O(log n), Space: O(1)
        """
        # Handle overflow
        if dividend == -2**31 and divisor == -1:
            return 2**31 - 1
        
        # Determine sign
        negative = (dividend < 0) ^ (divisor < 0)
        
        # Work with positive numbers
        dividend = abs(dividend)
        divisor = abs(divisor)
        
        result = 0
        
        while dividend >= divisor:
            # Find largest multiple of divisor that fits in dividend
            temp_divisor = divisor
            multiple = 1
            
            while dividend >= (temp_divisor << 1):
                temp_divisor <<= 1
                multiple <<= 1
            
            dividend -= temp_divisor
            result += multiple
        
        return -result if negative else result

    # Alternative: Using subtraction
    def divideSubtraction(dividend, divisor):
        """
        Simple subtraction approach (slower)
        Time: O(dividend/divisor), Space: O(1)
        """
        if dividend == -2**31 and divisor == -1:
            return 2**31 - 1
        
        negative = (dividend < 0) ^ (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        
        result = 0
        while dividend >= divisor:
            dividend -= divisor
            result += 1
        
        return -result if negative else result

    # Test
    print(divide(10, 3))   # 3
    print(divide(7, -3))   # -2
    ```

=== "Insights"

    **Key Insights:**
    - **Bit manipulation**: Use left shift for fast multiplication by 2
    - **Exponential search**: Find largest multiple that fits
    - **Overflow handling**: Check for INT_MIN / -1 case
    - **Sign handling**: XOR to determine result sign

---

## Problem 4: Fraction to Recurring Decimal

**Difficulty**: 🟡 Medium  
**Pattern**: Hash Map/Math  
**Time**: O(n), **Space**: O(n)

=== "Problem"

    Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

    If the fractional part is repeating, enclose the repeating part in parentheses.

    **Examples:**
    ```
    Input: numerator = 1, denominator = 2
    Output: "0.5"

    Input: numerator = 2, denominator = 1
    Output: "2"

    Input: numerator = 2, denominator = 3
    Output: "0.(6)"
    ```

=== "Solution"

    ```python
    def fractionToDecimal(numerator, denominator):
        """
        Long division with cycle detection using hash map
        Time: O(n), Space: O(n) where n is number of digits
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
                # Found cycle
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
    ```

=== "Insights"

    **Key Insights:**
    - **Long division**: Simulate manual division process
    - **Cycle detection**: Use hash map to track remainder positions
    - **String building**: Efficiently construct result string
    - **Edge cases**: Handle zero, negative numbers, no remainder

---

## Problem 5: Excel Sheet Column Title

**Difficulty**: 🟡 Medium  
**Pattern**: Base Conversion  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Given an integer `columnNumber`, return its corresponding column title as it appears in an Excel sheet.

    **Examples:**
    ```
    Input: columnNumber = 1
    Output: "A"

    Input: columnNumber = 28
    Output: "AB"

    Input: columnNumber = 701
    Output: "ZY"
    ```

=== "Solution"

    ```python
    def convertToTitle(columnNumber):
        """
        Convert number to Excel column title (base 26 with A=1)
        Time: O(log n), Space: O(1)
        """
        result = []
        
        while columnNumber > 0:
            # Adjust for 1-based indexing
            columnNumber -= 1
            result.append(chr(ord('A') + columnNumber % 26))
            columnNumber //= 26
        
        return "".join(reversed(result))

    # Alternative: Recursive approach
    def convertToTitleRecursive(columnNumber):
        """
        Recursive solution for Excel column title
        """
        if columnNumber == 0:
            return ""
        
        columnNumber -= 1
        return convertToTitleRecursive(columnNumber // 26) + chr(ord('A') + columnNumber % 26)

    # Test
    print(convertToTitle(1))     # "A"
    print(convertToTitle(28))    # "AB"
    print(convertToTitle(701))   # "ZY"
    ```

=== "Insights"

    **Key Insights:**
    - **1-based indexing**: Subtract 1 before division and modulo
    - **Base 26**: Similar to base conversion but 1-indexed
    - **Character mapping**: Use ASCII values for A-Z
    - **Reverse result**: Build from right to left, then reverse

---

## Problem 6: Multiply Strings

**Difficulty**: 🟡 Medium  
**Pattern**: String Math  
**Time**: O(mn), **Space**: O(m+n)

=== "Problem"

    Given two non-negative integers `num1` and `num2` represented as strings, return the product of `num1` and `num2`, also represented as a string.

    **Examples:**
    ```
    Input: num1 = "2", num2 = "3"
    Output: "6"

    Input: num1 = "123", num2 = "456"
    Output: "56088"
    ```

=== "Solution"

    ```python
    def multiply(num1, num2):
        """
        Multiply two strings representing large numbers
        Time: O(mn), Space: O(m+n)
        """
        if num1 == "0" or num2 == "0":
            return "0"
        
        m, n = len(num1), len(num2)
        result = [0] * (m + n)
        
        # Multiply each digit
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                p1, p2 = i + j, i + j + 1
                total = mul + result[p2]
                
                result[p2] = total % 10
                result[p1] += total // 10
        
        # Convert to string, skip leading zeros
        start = 0
        while start < len(result) and result[start] == 0:
            start += 1
        
        return "".join(str(digit) for digit in result[start:])

    # Alternative: Using addition
    def multiplyAddition(num1, num2):
        """
        Multiply by repeated addition (for small numbers)
        """
        if num1 == "0" or num2 == "0":
            return "0"
        
        def add_strings(num1, num2):
            result = []
            carry = 0
            i, j = len(num1) - 1, len(num2) - 1
            
            while i >= 0 or j >= 0 or carry:
                digit1 = int(num1[i]) if i >= 0 else 0
                digit2 = int(num2[j]) if j >= 0 else 0
                
                total = digit1 + digit2 + carry
                result.append(str(total % 10))
                carry = total // 10
                
                i -= 1
                j -= 1
            
            return "".join(reversed(result))
        
        result = "0"
        for _ in range(int(num2)):
            result = add_strings(result, num1)
        
        return result

    # Test
    print(multiply("2", "3"))       # "6"
    print(multiply("123", "456"))   # "56088"
    ```

=== "Insights"

    **Key Insights:**
    - **Position mapping**: Result[i+j] and result[i+j+1] for multiplication
    - **Carry handling**: Propagate carries properly
    - **Leading zeros**: Skip leading zeros in final result
    - **Edge cases**: Handle multiplication by zero

---

## 📝 Summary

### Medium Math Patterns Mastered

1. **Fast Exponentiation** - O(log n) power computation
2. **Binary Search on Answer** - Finding optimal values
3. **Bit Manipulation** - Efficient arithmetic operations
4. **String Mathematics** - Large number arithmetic
5. **Base Conversion** - Number system transformations
6. **Cycle Detection** - Finding patterns in sequences

### Key Algorithms

| **Algorithm** | **Time** | **Space** | **Use Cases** |
|---------------|----------|-----------|---------------|
| **Fast Exponentiation** | O(log n) | O(1) | Power computation |
| **Binary Search** | O(log n) | O(1) | Finding square roots |
| **Long Division** | O(n) | O(n) | Fraction conversion |
| **String Multiplication** | O(mn) | O(m+n) | Large number arithmetic |

### Problem-Solving Strategies

1. **Divide and Conquer**: Break complex problems into smaller parts
2. **Bit Manipulation**: Use shifts for efficient multiplication/division
3. **Hash Maps**: Track states for cycle detection
4. **String Processing**: Handle large numbers as strings
5. **Mathematical Properties**: Leverage number theory for optimization

---

## 🏆 Congratulations

You've mastered intermediate mathematical algorithms! These skills are essential for:

- **System Design** - Handling large scale computations
- **Financial Software** - Precise arithmetic operations
- **Scientific Computing** - Numerical analysis
- **Competitive Programming** - Mathematical contests
- **Cryptography** - Number theory applications

### 📚 What's Next

- **[Hard Problems](hard-problems.md)** - Advanced mathematical challenges
- **[Number Theory](../advanced/number-theory.md)** - Deep mathematical concepts
- **[Optimization](../advanced/optimization.md)** - Mathematical optimization
- **[Cryptography](../advanced/cryptography.md)** - Applied mathematics

*Continue building your mathematical foundation for advanced problem-solving!*
