# Mathematical Algorithms - Hard Problems

## 🎯 Learning Objectives

Master the most challenging mathematical algorithms and advanced number theory:

- Complex modular arithmetic and number theory
- Advanced combinatorics and probability theory
- Optimization problems with multiple constraints
- Mathematical pattern recognition at scale
- Algorithmic number theory and cryptographic applications

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Super Pow | Modular Exponentiation | Hard | O(n) | O(1) |
    | 2 | Nth Digit | Mathematical Pattern | Hard | O(log n) | O(1) |
    | 3 | Basic Calculator | Stack/Math | Hard | O(n) | O(n) |
    | 4 | Count of Range Sum | Merge Sort/Math | Hard | O(n log n) | O(n) |
    | 5 | Max Points on a Line | Geometry/Math | Hard | O(n²) | O(n) |
    | 6 | Largest Rectangle in Histogram | Stack/Math | Hard | O(n) | O(n) |
    | 7 | Trapping Rain Water | Two Pointers/Math | Hard | O(n) | O(1) |
    | 8 | Integer to English Words | String/Math | Hard | O(1) | O(1) |
    | 9 | Russian Doll Envelopes | LIS/Math | Hard | O(n log n) | O(n) |
    | 10 | Largest Number | Greedy/Math | Hard | O(n log n) | O(n) |
    | 11 | Self Crossing | Geometry/Math | Hard | O(n) | O(1) |
    | 12 | Cat and Mouse | Game Theory/Math | Hard | O(n³) | O(n²) |
    | 13 | Minimum Window Substring | Sliding Window/Math | Hard | O(n) | O(k) |
    | 14 | Median of Two Sorted Arrays | Binary Search/Math | Hard | O(log(min(m,n))) | O(1) |
    | 15 | Regular Expression Matching | DP/Math | Hard | O(mn) | O(mn) |

=== "🎯 Expert Math Patterns"

    **🧮 Advanced Number Theory:**
    - Modular arithmetic with large numbers
    - Prime factorization and divisibility
    - Chinese Remainder Theorem
    - Euler's totient function
    
    **🔢 Computational Geometry:**
    - Line intersections and collinearity
    - Convex hull algorithms
    - Area and perimeter calculations
    - Geometric transformations
    
    **🎲 Game Theory & Optimization:**
    - Minimax algorithms
    - Nash equilibrium
    - Dynamic programming on games
    - Probability calculations

=== "📚 Advanced Algorithm Templates"

    **Modular Exponentiation Template:**
    ```python
    def mod_pow(base, exp, mod):
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        return result
    ```
    
    **Line Intersection Template:**
    ```python
    def are_collinear(p1, p2, p3):
        # Check if three points are collinear
        return (p2[1] - p1[1]) * (p3[0] - p1[0]) == (p3[1] - p1[1]) * (p2[0] - p1[0])
    ```
    
    **Stack-based Calculator Template:**
    ```python
    def evaluate_expression(expression):
        stack = []
        num = 0
        sign = '+'
        
        for i, char in enumerate(expression):
            if char.isdigit():
                num = num * 10 + int(char)
            
            if char in '+-*/' or i == len(expression) - 1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                elif sign == '/':
                    stack.append(int(stack.pop() / num))
                
                sign = char
                num = 0
        
        return sum(stack)
    ```

---

## Problem 1: Super Pow

**Difficulty**: 🔴 Hard  
**Pattern**: Modular Exponentiation  
**Time**: O(n), **Space**: O(1)

=== "Problem"

    Your task is to calculate a^b mod 1337 where a is a positive integer and b is an extremely large positive integer given in the form of an array.

    **Examples:**
    ```
    Input: a = 2, b = [3]
    Output: 8

    Input: a = 2, b = [1,0]
    Output: 1024

    Input: a = 2147483647, b = [2,0,0]
    Output: 1198
    ```

=== "Solution"

    ```python
    def superPow(a, b):
        """
        Calculate a^b mod 1337 using modular exponentiation
        b is given as array of digits
        """
        MOD = 1337
        
        def pow_mod(base, exp, mod):
            """Fast modular exponentiation"""
            result = 1
            base = base % mod
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % mod
                exp = exp >> 1
                base = (base * base) % mod
            return result
        
        def super_pow_helper(a, b):
            if not b:
                return 1
            
            # Extract last digit
            last_digit = b.pop()
            
            # a^b = (a^(b//10))^10 * a^(b%10)
            # Using property: a^(xy) = (a^x)^y
            part1 = pow_mod(super_pow_helper(a, b), 10, MOD)
            part2 = pow_mod(a, last_digit, MOD)
            
            return (part1 * part2) % MOD
        
        return super_pow_helper(a, b[:])  # Pass copy to avoid modifying original

    # Alternative: Iterative approach
    def superPowIterative(a, b):
        """
        Iterative solution using mathematical properties
        """
        MOD = 1337
        
        def pow_mod(base, exp):
            result = 1
            base = base % MOD
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % MOD
                exp = exp >> 1
                base = (base * base) % MOD
            return result
        
        result = 1
        a = a % MOD
        
        for digit in b:
            result = pow_mod(result, 10) * pow_mod(a, digit) % MOD
        
        return result

    # Test
    print(superPow(2, [3]))      # 8
    print(superPow(2, [1,0]))    # 1024
    ```

=== "Insights"

    **Key Insights:**
    - **Modular exponentiation**: Use fast exponentiation with modulo
    - **Mathematical property**: a^(xy) = (a^x)^y
    - **Digit processing**: Handle array representation of large numbers
    - **Euler's theorem**: Can optimize using φ(1337) = 1140

---

## Problem 2: Nth Digit

**Difficulty**: 🔴 Hard  
**Pattern**: Mathematical Pattern  
**Time**: O(log n), **Space**: O(1)

=== "Problem"

    Given an integer n, return the nth digit of the infinite integer sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...

    **Examples:**
    ```
    Input: n = 3
    Output: 3

    Input: n = 11
    Output: 0 (The 11th digit is 0, which is part of the number 10)
    ```

=== "Solution"

    ```python
    def findNthDigit(n):
        """
        Find the nth digit in the sequence 1,2,3,4,5,6,7,8,9,10,11,...
        """
        # 1-digit numbers: 1-9 (9 numbers, 9 digits)
        # 2-digit numbers: 10-99 (90 numbers, 180 digits)
        # 3-digit numbers: 100-999 (900 numbers, 2700 digits)
        
        digits = 1
        start = 1
        count = 9
        
        # Find which group (1-digit, 2-digit, etc.) contains the nth digit
        while n > digits * count:
            n -= digits * count
            digits += 1
            start *= 10
            count *= 10
        
        # Find which number in the group
        number = start + (n - 1) // digits
        
        # Find which digit in the number
        digit_index = (n - 1) % digits
        
        return int(str(number)[digit_index])

    # Alternative: Direct calculation
    def findNthDigitDirect(n):
        """
        Direct calculation approach
        """
        if n <= 9:
            return n
        
        # Remove 1-digit numbers
        n -= 9
        
        # Check 2-digit numbers
        if n <= 90 * 2:
            # n is in 2-digit range
            number = 10 + (n - 1) // 2
            digit_pos = (n - 1) % 2
            return int(str(number)[digit_pos])
        
        # Remove 2-digit numbers
        n -= 90 * 2
        
        # Check 3-digit numbers
        if n <= 900 * 3:
            number = 100 + (n - 1) // 3
            digit_pos = (n - 1) % 3
            return int(str(number)[digit_pos])
        
        # Continue pattern for larger numbers...
        # General formula implementation above is more elegant
        
        return 0

    # Test
    print(findNthDigit(3))   # 3
    print(findNthDigit(11))  # 0
    print(findNthDigit(15))  # 2
    ```

=== "Insights"

    **Key Insights:**
    - **Pattern recognition**: Group numbers by digit count
    - **Mathematical ranges**: 1-digit (1-9), 2-digit (10-99), etc.
    - **Index calculation**: Find group, then number, then digit position
    - **Efficient search**: Binary search-like approach through groups

---

## Problem 3: Basic Calculator

**Difficulty**: 🔴 Hard  
**Pattern**: Stack/Math  
**Time**: O(n), **Space**: O(n)

=== "Problem"

    Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.

    **Examples:**
    ```
    Input: s = "1 + 1"
    Output: 2

    Input: s = " 2-1 + 2 "
    Output: 3

    Input: s = "(1+(4+5+2)-3)+(6+8)"
    Output: 23
    ```

=== "Solution"

    ```python
    def calculate(s):
        """
        Basic calculator with parentheses support
        """
        stack = []
        result = 0
        number = 0
        sign = 1
        
        for char in s:
            if char.isdigit():
                number = number * 10 + int(char)
            elif char == '+':
                result += sign * number
                number = 0
                sign = 1
            elif char == '-':
                result += sign * number
                number = 0
                sign = -1
            elif char == '(':
                # Push current result and sign to stack
                stack.append(result)
                stack.append(sign)
                # Reset for expression in parentheses
                result = 0
                sign = 1
            elif char == ')':
                # Complete current expression
                result += sign * number
                number = 0
                # Pop sign and previous result
                result *= stack.pop()  # sign
                result += stack.pop()  # previous result
        
        return result + sign * number

    # Calculator with multiplication and division
    def calculateAdvanced(s):
        """
        Calculator with +, -, *, / operations
        """
        stack = []
        num = 0
        operation = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                num = num * 10 + int(char)
            
            if char in '+-*/' or i == len(s) - 1:
                if operation == '+':
                    stack.append(num)
                elif operation == '-':
                    stack.append(-num)
                elif operation == '*':
                    stack.append(stack.pop() * num)
                elif operation == '/':
                    # Handle negative division
                    prev = stack.pop()
                    stack.append(int(prev / num))
                
                operation = char
                num = 0
        
        return sum(stack)

    # Test
    print(calculate("1 + 1"))                    # 2
    print(calculate(" 2-1 + 2 "))                # 3
    print(calculate("(1+(4+5+2)-3)+(6+8)"))      # 23
    print(calculateAdvanced("3+2*2"))            # 7
    ```

=== "Insights"

    **Key Insights:**
    - **Stack for parentheses**: Store previous results and signs
    - **State management**: Track current number, result, and sign
    - **Operator precedence**: Handle multiplication/division before addition/subtraction
    - **Edge cases**: Negative numbers, spaces, nested parentheses

---

## Problem 4: Max Points on a Line

**Difficulty**: 🔴 Hard  
**Pattern**: Geometry/Math  
**Time**: O(n²), **Space**: O(n)

=== "Problem"

    Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.

    **Examples:**
    ```
    Input: points = [[1,1],[2,2],[3,3]]
    Output: 3

    Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
    Output: 4
    ```

=== "Solution"

    ```python
    def maxPointsOnLine(points):
        """
        Find maximum points on same line using slope calculation
        """
        if len(points) <= 2:
            return len(points)
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def get_slope(p1, p2):
            """Get normalized slope as tuple (dy, dx)"""
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx == 0:
                return (1, 0)  # Vertical line
            if dy == 0:
                return (0, 1)  # Horizontal line
            
            # Normalize slope
            g = gcd(abs(dx), abs(dy))
            dx, dy = dx // g, dy // g
            
            # Ensure consistent representation
            if dx < 0:
                dx, dy = -dx, -dy
            
            return (dy, dx)
        
        max_points = 0
        
        for i in range(len(points)):
            slopes = {}
            duplicate = 0
            local_max = 0
            
            for j in range(i + 1, len(points)):
                if points[i] == points[j]:
                    duplicate += 1
                    continue
                
                slope = get_slope(points[i], points[j])
                slopes[slope] = slopes.get(slope, 0) + 1
                local_max = max(local_max, slopes[slope])
            
            max_points = max(max_points, local_max + duplicate + 1)
        
        return max_points

    # Alternative: Using floating point slopes (less precise)
    def maxPointsOnLineFloat(points):
        """
        Using floating point slopes (simpler but less precise)
        """
        if len(points) <= 2:
            return len(points)
        
        max_points = 0
        
        for i in range(len(points)):
            slopes = {}
            duplicate = 0
            vertical = 0
            local_max = 0
            
            for j in range(i + 1, len(points)):
                if points[i] == points[j]:
                    duplicate += 1
                elif points[i][0] == points[j][0]:
                    vertical += 1
                    local_max = max(local_max, vertical)
                else:
                    slope = (points[j][1] - points[i][1]) / (points[j][0] - points[i][0])
                    slopes[slope] = slopes.get(slope, 0) + 1
                    local_max = max(local_max, slopes[slope])
            
            max_points = max(max_points, local_max + duplicate + 1)
        
        return max_points

    # Test
    points1 = [[1,1],[2,2],[3,3]]
    print(maxPointsOnLine(points1))  # 3
    
    points2 = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
    print(maxPointsOnLine(points2))  # 4
    ```

=== "Insights"

    **Key Insights:**
    - **Slope calculation**: Use GCD to normalize slopes and avoid precision issues
    - **Duplicate points**: Handle identical points separately
    - **Vertical lines**: Special case where slope is undefined
    - **Hash map**: Group points by slope from each reference point

---

## Problem 5: Median of Two Sorted Arrays

**Difficulty**: 🔴 Hard  
**Pattern**: Binary Search/Math  
**Time**: O(log(min(m,n))), **Space**: O(1)

=== "Problem"

    Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

    **Examples:**
    ```
    Input: nums1 = [1,3], nums2 = [2]
    Output: 2.0

    Input: nums1 = [1,2], nums2 = [3,4]
    Output: 2.5
    ```

=== "Solution"

    ```python
    def findMedianSortedArrays(nums1, nums2):
        """
        Find median using binary search on the smaller array
        """
        # Ensure nums1 is the smaller array
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        
        while left <= right:
            # Partition nums1 and nums2
            partition1 = (left + right) // 2
            partition2 = (m + n + 1) // 2 - partition1
            
            # Elements on left side
            max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
            max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
            
            # Elements on right side
            min_right1 = float('inf') if partition1 == m else nums1[partition1]
            min_right2 = float('inf') if partition2 == n else nums2[partition2]
            
            # Check if partition is correct
            if max_left1 <= min_right2 and max_left2 <= min_right1:
                # Found correct partition
                if (m + n) % 2 == 0:
                    return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
                else:
                    return max(max_left1, max_left2)
            elif max_left1 > min_right2:
                # Too many elements from nums1 on left side
                right = partition1 - 1
            else:
                # Too few elements from nums1 on left side
                left = partition1 + 1
        
        raise ValueError("Input arrays are not sorted")

    # Alternative: Merge approach (simpler but O(m+n))
    def findMedianSortedArraysMerge(nums1, nums2):
        """
        Merge arrays and find median (O(m+n) time)
        """
        merged = []
        i = j = 0
        
        # Merge arrays
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                merged.append(nums1[i])
                i += 1
            else:
                merged.append(nums2[j])
                j += 1
        
        # Add remaining elements
        merged.extend(nums1[i:])
        merged.extend(nums2[j:])
        
        # Find median
        n = len(merged)
        if n % 2 == 0:
            return (merged[n//2 - 1] + merged[n//2]) / 2
        else:
            return merged[n//2]

    # Test
    print(findMedianSortedArrays([1,3], [2]))      # 2.0
    print(findMedianSortedArrays([1,2], [3,4]))    # 2.5
    ```

=== "Insights"

    **Key Insights:**
    - **Binary search**: Search for correct partition point
    - **Partition property**: Left side ≤ right side for both arrays
    - **Median calculation**: Different for even vs odd total length
    - **Optimization**: Always search on the smaller array

---

## 📝 Summary

### Hard Math Patterns Mastered

1. **Modular Exponentiation** - Efficient power computation with large numbers
2. **Mathematical Pattern Recognition** - Finding patterns in sequences
3. **Computational Geometry** - Line intersections, point relationships
4. **Advanced Calculator** - Expression parsing and evaluation
5. **Binary Search on Answer** - Finding optimal solutions in sorted search spaces
6. **Game Theory** - Optimal strategy computation

### Advanced Techniques

| **Technique** | **Time** | **Space** | **Applications** |
|---------------|----------|-----------|------------------|
| **Modular Arithmetic** | O(log n) | O(1) | Cryptography, large number computation |
| **Slope Normalization** | O(1) | O(1) | Geometric calculations, line detection |
| **Stack-based Parsing** | O(n) | O(n) | Expression evaluation, syntax analysis |
| **Binary Search** | O(log n) | O(1) | Optimization, search in sorted data |

### Problem-Solving Strategies

1. **Mathematical Properties**: Leverage number theory and geometric theorems
2. **Precision Handling**: Use GCD and integer arithmetic to avoid floating-point errors
3. **Pattern Recognition**: Identify mathematical sequences and relationships
4. **Divide and Conquer**: Break complex problems into manageable subproblems
5. **State Management**: Use stacks and careful bookkeeping for complex operations

---

## 🏆 Congratulations

You've mastered the most challenging mathematical algorithms! These advanced techniques are used in:

- **Cryptography and Security** - RSA encryption, hash functions
- **Computer Graphics** - 3D transformations, collision detection
- **Financial Technology** - High-precision calculations, risk modeling
- **Scientific Computing** - Numerical analysis, simulation
- **Competitive Programming** - Mathematical contests, algorithm competitions

### 📚 What's Next

- **[Advanced Number Theory](../advanced/number-theory.md)** - Deeper mathematical concepts
- **[Computational Geometry](../advanced/geometry.md)** - Advanced geometric algorithms
- **[Cryptography](../advanced/cryptography.md)** - Applied mathematical cryptography
- **[Optimization](../advanced/optimization.md)** - Mathematical optimization techniques

*You now possess the mathematical foundation for the most challenging algorithmic problems!*
