# Stacks: Medium Problems

## 🚀 Intermediate Stack Challenges

Master advanced stack techniques including monotonic stacks, expression evaluation, and complex stack applications.

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Daily Temperatures | Monotonic Stack | Medium | O(n) | O(n) |
    | 2 | Next Greater Element II | Monotonic Stack | Medium | O(n) | O(n) |
    | 3 | Valid Parentheses | Stack Matching | Medium | O(n) | O(n) |
    | 4 | Evaluate Reverse Polish Notation | Stack Evaluation | Medium | O(n) | O(n) |
    | 5 | Basic Calculator | Stack + Parsing | Medium | O(n) | O(n) |
    | 6 | Decode String | Stack + Parsing | Medium | O(n) | O(n) |
    | 7 | Remove Duplicate Letters | Monotonic Stack | Medium | O(n) | O(1) |
    | 8 | Asteroid Collision | Stack Simulation | Medium | O(n) | O(n) |
    | 9 | Car Fleet | Stack + Sorting | Medium | O(n log n) | O(n) |
    | 10 | Online Stock Span | Monotonic Stack | Medium | O(n) | O(n) |
    | 11 | Simplify Path | Stack + Parsing | Medium | O(n) | O(n) |
    | 12 | Remove K Digits | Monotonic Stack | Medium | O(n) | O(n) |
    | 13 | Largest Rectangle in Histogram | Monotonic Stack | Medium | O(n) | O(n) |
    | 14 | Maximal Rectangle | Stack + DP | Medium | O(mn) | O(n) |
    | 15 | Trapping Rain Water | Monotonic Stack | Medium | O(n) | O(n) |

=== "🎯 Interview Tips"

    **📝 Key Patterns:**
    
    - **Monotonic Stack**: Maintain increasing/decreasing order
    - **Expression Parsing**: Handle operators and parentheses
    - **Stack Simulation**: Model real-world stack behavior
    - **String Processing**: Build and decode strings
    - **Optimization**: Use stack for greedy choices
    
    **⚡ Problem-Solving Strategies:**
    
    - Identify when to use monotonic stacks
    - Handle edge cases in expression parsing
    - Practice stack-based string manipulation
    - Combine stacks with other data structures
    
    **🚫 Common Pitfalls:**
    
    - Not handling empty stack operations
    - Forgetting to process remaining elements
    - Incorrect operator precedence
    - Missing edge cases in parsing

=== "📚 Study Plan"

    **Week 1: Monotonic Stacks (Problems 1-5)**
    - Master next greater/smaller patterns
    - Practice expression evaluation
    
    **Week 2: String Processing (Problems 6-10)**
    - Learn stack-based parsing
    - Focus on string manipulation
    
    **Week 3: Advanced Applications (Problems 11-15)**
    - Complex stack simulations
    - Optimization problems

=== "Daily Temperatures"

    **Problem Statement:**
    Given an array of integers temperatures representing daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.

    **Example:**
    ```text
    Input: temperatures = [73,74,75,71,69,72,76,73]
    Output: [1,1,4,2,1,1,0,0]
    ```

    **Solution:**
    ```python
    def dailyTemperatures(temperatures):
        """
        Monotonic stack to find next warmer day.
        
        Time: O(n) - each element pushed/popped once
        Space: O(n) - stack storage
        """
        n = len(temperatures)
        result = [0] * n
        stack = []  # Stack of indices
        
        for i in range(n):
            # Pop elements that are cooler than current
            while stack and temperatures[stack[-1]] < temperatures[i]:
                idx = stack.pop()
                result[idx] = i - idx
            
            stack.append(i)
        
        return result
    ```

    **Key Insights:**
    - Monotonic stack maintains decreasing temperatures
    - Store indices instead of values for distance calculation
    - Each element processed exactly once
    - Stack holds indices of unresolved temperatures

=== "Next Greater Element II"

    **Problem Statement:**
    Given a circular integer array nums, return the next greater number for every element in nums. The next greater number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number.

    **Example:**
    ```text
    Input: nums = [1,2,1]
    Output: [2,-1,2]
    ```

    **Solution:**
    ```python
    def nextGreaterElements(nums):
        """
        Handle circular array by traversing twice.
        
        Time: O(n) - two passes
        Space: O(n) - stack and result
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # Traverse twice to handle circular nature
        for i in range(2 * n):
            # Pop smaller elements
            while stack and nums[stack[-1]] < nums[i % n]:
                idx = stack.pop()
                result[idx] = nums[i % n]
            
            # Only push indices in first pass
            if i < n:
                stack.append(i)
        
        return result
    ```

    **Key Insights:**
    - Circular array requires two passes
    - Use modulo to wrap around indices
    - Only push indices in first pass
    - Stack handles the circular logic naturally

=== "Valid Parentheses"

    **Problem Statement:**
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

    **Example:**
    ```text
    Input: s = "()[]{}"
    Output: true
    ```

    **Solution:**
    ```python
    def isValid(s):
        """
        Stack-based bracket matching.
        
        Time: O(n) - single pass
        Space: O(n) - stack storage
        """
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

    **Key Insights:**
    - Stack naturally handles nested structures
    - Use mapping for bracket pairs
    - Empty stack means all brackets matched
    - Order matters for proper nesting

=== "Evaluate Reverse Polish Notation"

    **Problem Statement:**
    Evaluate the value of an arithmetic expression in Reverse Polish Notation. Valid operators are +, -, *, /. Each operand may be an integer or another expression.

    **Example:**
    ```text
    Input: tokens = ["2","1","+","3","*"]
    Output: 9
    Explanation: ((2 + 1) * 3) = 9
    ```

    **Solution:**
    ```python
    def evalRPN(tokens):
        """
        Stack-based evaluation of postfix notation.
        
        Time: O(n) - single pass
        Space: O(n) - stack storage
        """
        stack = []
        operators = {'+', '-', '*', '/'}
        
        for token in tokens:
            if token in operators:
                # Pop two operands
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                else:  # token == '/'
                    result = int(a / b)  # Truncate towards zero
                
                stack.append(result)
            else:
                # Operand
                stack.append(int(token))
        
        return stack[0]
    ```

    **Key Insights:**
    - Stack naturally handles postfix evaluation
    - Pop order matters for non-commutative operations
    - Division truncates towards zero
    - Final result is single element in stack

=== "Basic Calculator"

    **Problem Statement:**
    Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.

    **Example:**
    ```text
    Input: s = "1 + 1"
    Output: 2
    ```

    **Solution:**
    ```python
    def calculate(s):
        """
        Stack-based expression evaluation.
        
        Time: O(n) - single pass
        Space: O(n) - stack storage
        """
        stack = []
        number = 0
        sign = 1
        result = 0
        
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
                # Push current result and sign
                stack.append(result)
                stack.append(sign)
                result = 0
                sign = 1
            elif char == ')':
                result += sign * number
                number = 0
                # Pop sign and previous result
                result *= stack.pop()  # sign
                result += stack.pop()  # previous result
        
        return result + sign * number
    ```

    **Key Insights:**
    - Handle parentheses by saving state
    - Process numbers digit by digit
    - Track current sign and result
    - Stack stores intermediate results

=== "Decode String"

    **Problem Statement:**
    Given an encoded string, return its decoded string. The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times.

    **Example:**
    ```text
    Input: s = "3[a]2[bc]"
    Output: "aaabcbc"
    ```

    **Solution:**
    ```python
    def decodeString(s):
        """
        Stack-based string decoding.
        
        Time: O(n) - where n is length of decoded string
        Space: O(n) - stack storage
        """
        stack = []
        current_num = 0
        current_str = ""
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                # Push current state to stack
                stack.append(current_str)
                stack.append(current_num)
                current_str = ""
                current_num = 0
            elif char == ']':
                # Pop and decode
                num = stack.pop()
                prev_str = stack.pop()
                current_str = prev_str + current_str * num
            else:
                current_str += char
        
        return current_str
    ```

    **Key Insights:**
    - Stack handles nested encoding
    - Build numbers and strings incrementally
    - Save state before entering brackets
    - Restore and combine when exiting brackets

=== "Remove Duplicate Letters"

    **Problem Statement:**
    Given a string s, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.

    **Example:**
    ```text
    Input: s = "bcabc"
    Output: "abc"
    ```

    **Solution:**
    ```python
    def removeDuplicateLetters(s):
        """
        Monotonic stack with lexicographical order.
        
        Time: O(n) - single pass
        Space: O(1) - constant space (26 letters)
        """
        # Count frequency of each character
        count = {}
        for char in s:
            count[char] = count.get(char, 0) + 1
        
        stack = []
        in_stack = set()
        
        for char in s:
            count[char] -= 1
            
            if char in in_stack:
                continue
            
            # Remove larger characters that appear later
            while (stack and stack[-1] > char and 
                   count[stack[-1]] > 0):
                removed = stack.pop()
                in_stack.remove(removed)
            
            stack.append(char)
            in_stack.add(char)
        
        return ''.join(stack)
    ```

    **Key Insights:**
    - Monotonic stack maintains lexicographical order
    - Count frequencies to know if character appears later
    - Use set to track characters in stack
    - Greedy approach for optimal result

=== "Asteroid Collision"

    **Problem Statement:**
    We are given an array asteroids of integers representing asteroids in a row. For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Find out the state after all collisions.

    **Example:**
    ```text
    Input: asteroids = [5,10,-5]
    Output: [5,10]
    ```

    **Solution:**
    ```python
    def asteroidCollision(asteroids):
        """
        Stack simulation of asteroid collisions.
        
        Time: O(n) - each asteroid processed once
        Space: O(n) - stack storage
        """
        stack = []
        
        for asteroid in asteroids:
            while stack and asteroid < 0 and stack[-1] > 0:
                # Collision between right-moving and left-moving
                if stack[-1] < -asteroid:
                    # Right-moving asteroid explodes
                    stack.pop()
                    continue
                elif stack[-1] == -asteroid:
                    # Both explode
                    stack.pop()
                
                # Left-moving asteroid explodes or both explode
                break
            else:
                # No collision or left-moving asteroid survives
                stack.append(asteroid)
        
        return stack
    ```

    **Key Insights:**
    - Stack represents surviving asteroids
    - Collisions only happen between right and left moving
    - Compare sizes to determine survivors
    - Use while loop for chain reactions

=== "Car Fleet"

    **Problem Statement:**
    There are n cars going to the same destination along a one-lane road. The destination is target miles away. You are given two integer arrays position and speed of length n, where position[i] is the position of the ith car and speed[i] is the speed of the ith car (in miles per hour).

    **Example:**
    ```text
    Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
    Output: 3
    ```

    **Solution:**
    ```python
    def carFleet(target, position, speed):
        """
        Stack to track car fleets.
        
        Time: O(n log n) - sorting
        Space: O(n) - stack storage
        """
        # Create pairs and sort by position (descending)
        cars = [(pos, spd) for pos, spd in zip(position, speed)]
        cars.sort(reverse=True)
        
        stack = []
        
        for pos, spd in cars:
            # Calculate time to reach target
            time = (target - pos) / spd
            
            # If current car is slower, it forms a new fleet
            if not stack or time > stack[-1]:
                stack.append(time)
        
        return len(stack)
    ```

    **Key Insights:**
    - Sort cars by position (closest to target first)
    - Calculate time to reach target
    - Slower cars form new fleets
    - Stack tracks distinct fleets

=== "Online Stock Span"

    **Problem Statement:**
    Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day. The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backward) for which the stock price was less than or equal to today's price.

    **Example:**
    ```text
    Input: ["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
    [[], [100], [80], [60], [70], [60], [75], [85]]
    Output: [null, 1, 1, 1, 2, 1, 4, 6]
    ```

    **Solution:**
    ```python
    class StockSpanner:
        def __init__(self):
            """
            Initialize with monotonic stack.
            """
            self.stack = []  # (price, span) pairs
        
        def next(self, price):
            """
            Calculate span for current price.
            
            Time: O(1) amortized
            Space: O(n) - stack storage
            """
            span = 1
            
            # Pop prices that are less than or equal to current
            while self.stack and self.stack[-1][0] <= price:
                span += self.stack.pop()[1]
            
            self.stack.append((price, span))
            return span
    ```

    **Key Insights:**
    - Monotonic stack maintains decreasing prices
    - Store both price and span for efficiency
    - Accumulate spans when popping
    - Amortized O(1) time per operation

=== "Simplify Path"

    **Problem Statement:**
    Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

    **Example:**
    ```text
    Input: path = "/home//foo/"
    Output: "/home/foo"
    ```

    **Solution:**
    ```python
    def simplifyPath(path):
        """
        Stack-based path simplification.
        
        Time: O(n) - single pass
        Space: O(n) - stack storage
        """
        stack = []
        components = path.split('/')
        
        for component in components:
            if component == '..':
                # Go back one directory
                if stack:
                    stack.pop()
            elif component and component != '.':
                # Valid directory name
                stack.append(component)
        
        return '/' + '/'.join(stack)
    ```

    **Key Insights:**
    - Split path by '/' separator
    - Use stack to track directory hierarchy
    - Handle '..' by popping from stack
    - Ignore empty components and '.'

=== "Remove K Digits"

    **Problem Statement:**
    Given string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.

    **Example:**
    ```text
    Input: num = "1432219", k = 3
    Output: "1219"
    ```

    **Solution:**
    ```python
    def removeKdigits(num, k):
        """
        Monotonic stack for lexicographically smallest result.
        
        Time: O(n) - single pass
        Space: O(n) - stack storage
        """
        stack = []
        
        for digit in num:
            # Remove larger digits to make smaller number
            while stack and stack[-1] > digit and k > 0:
                stack.pop()
                k -= 1
            
            stack.append(digit)
        
        # Remove remaining digits from the end
        while k > 0:
            stack.pop()
            k -= 1
        
        # Handle leading zeros and empty result
        result = ''.join(stack).lstrip('0')
        return result or '0'
    ```

    **Key Insights:**
    - Monotonic stack maintains increasing digits
    - Remove larger digits when possible
    - Handle remaining removals from end
    - Remove leading zeros from result

=== "Largest Rectangle in Histogram"

    **Problem Statement:**
    Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

    **Example:**
    ```text
    Input: heights = [2,1,5,6,2,3]
    Output: 10
    ```

    **Solution:**
    ```python
    def largestRectangleArea(heights):
        """
        Monotonic stack to find largest rectangle.
        
        Time: O(n) - each element pushed/popped once
        Space: O(n) - stack storage
        """
        stack = []
        max_area = 0
        
        for i, height in enumerate(heights):
            # Pop taller bars and calculate area
            while stack and heights[stack[-1]] > height:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            
            stack.append(i)
        
        # Process remaining bars
        while stack:
            h = heights[stack.pop()]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        return max_area
    ```

    **Key Insights:**
    - Monotonic stack maintains increasing heights
    - Calculate area when popping taller bars
    - Width determined by stack positions
    - Handle remaining bars after main loop

=== "Maximal Rectangle"

    **Problem Statement:**
    Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

    **Example:**
    ```text
    Input: matrix = [["1","0","1","0","0"],
                     ["1","0","1","1","1"],
                     ["1","1","1","1","1"],
                     ["1","0","0","1","0"]]
    Output: 6
    ```

    **Solution:**
    ```python
    def maximalRectangle(matrix):
        """
        Convert to histogram problem for each row.
        
        Time: O(mn) - process each cell
        Space: O(n) - histogram array
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        for row in matrix:
            # Update heights for current row
            for i in range(cols):
                if row[i] == '1':
                    heights[i] += 1
                else:
                    heights[i] = 0
            
            # Calculate max rectangle in current histogram
            max_area = max(max_area, largestRectangleArea(heights))
        
        return max_area
    
    def largestRectangleArea(heights):
        """Same as previous problem"""
        stack = []
        max_area = 0
        
        for i, height in enumerate(heights):
            while stack and heights[stack[-1]] > height:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        while stack:
            h = heights[stack.pop()]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        return max_area
    ```

    **Key Insights:**
    - Convert 2D problem to 1D histogram
    - Update heights row by row
    - Use largest rectangle algorithm for each row
    - Track maximum area across all rows

=== "Trapping Rain Water"

    **Problem Statement:**
    Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

    **Example:**
    ```text
    Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
    Output: 6
    ```

    **Solution:**
    ```python
    def trap(height):
        """
        Monotonic stack to calculate trapped water.
        
        Time: O(n) - single pass
        Space: O(n) - stack storage
        """
        stack = []
        water = 0
        
        for i, h in enumerate(height):
            # Pop shorter bars and calculate water
            while stack and height[stack[-1]] < h:
                bottom = stack.pop()
                
                if not stack:
                    break
                
                # Calculate water trapped
                distance = i - stack[-1] - 1
                bounded_height = min(h, height[stack[-1]]) - height[bottom]
                water += distance * bounded_height
            
            stack.append(i)
        
        return water
    ```

    **Alternative (Two Pointers):**
    ```python
    def trap(height):
        """
        Two-pointer approach for O(1) space.
        
        Time: O(n) - single pass
        Space: O(1) - constant space
        """
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        water = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        
        return water
    ```

    **Key Insights:**
    - Stack approach: calculate water when popping
    - Two-pointer approach: move from lower side
    - Water level determined by minimum of boundaries
    - Both approaches achieve O(n) time complexity

## 📝 Summary

Medium stack problems require:

- **Monotonic Stack Patterns**: Next greater/smaller elements
- **Expression Evaluation**: Parsing and calculating
- **String Manipulation**: Building and processing strings
- **Simulation**: Modeling real-world scenarios
- **Optimization**: Using stacks for greedy algorithms

These problems are essential for:

- **Technical Interviews**: Common patterns in coding interviews
- **Algorithm Design**: Understanding stack-based solutions
- **Problem Recognition**: Identifying when to use stacks
- **Performance Optimization**: Achieving optimal time complexity

Master these patterns to excel in stack-based problem solving! 🚀
