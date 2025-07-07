# Stacks - Hard Problems

## 🎯 Learning Objectives

Master advanced stack techniques and complex algorithms:

- Monotonic stack patterns for optimization
- Stack-based tree traversal and construction
- Complex expression evaluation and parsing
- Advanced sliding window with stack
- Stack-based dynamic programming

=== "Problem 1: Largest Rectangle in Histogram"

    **LeetCode 84** | **Difficulty: Hard**

    ## Problem Statement

    Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

    **Example:**
    ```
    Input: heights = [2,1,5,6,2,3]
    Output: 10
    ```

    ## Solution

    ```python
    def largestRectangleArea(heights):
        """
        Find largest rectangle area using monotonic stack.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        max_area = 0
        heights.append(0)  # Add sentinel to process remaining bars
        
        for i, height in enumerate(heights):
            # Maintain increasing stack
            while stack and heights[stack[-1]] > height:
                h = heights[stack.pop()]
                # Width = current index - previous index - 1
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            
            stack.append(i)
        
        return max_area
    ```

    ## Alternative: Divide and Conquer

    ```python
    def largestRectangleArea(heights):
        """
        Divide and conquer approach.
        
        Time: O(n log n) average, O(n²) worst case
        Space: O(log n)
        """
        def find_max_area(start, end):
            if start > end:
                return 0
            
            # Find minimum height index
            min_idx = start
            for i in range(start + 1, end + 1):
                if heights[i] < heights[min_idx]:
                    min_idx = i
            
            # Max area is maximum of:
            # 1. Area with minimum height as width
            # 2. Max area in left part
            # 3. Max area in right part
            return max(
                heights[min_idx] * (end - start + 1),
                find_max_area(start, min_idx - 1),
                find_max_area(min_idx + 1, end)
            )
        
        return find_max_area(0, len(heights) - 1)
    ```

    ## Key Insights

    - Monotonic stack efficiently finds boundaries
    - For each bar, find largest rectangle where it's the shortest
    - Sentinel value simplifies edge case handling

=== "Problem 2: Maximal Rectangle"

    **LeetCode 85** | **Difficulty: Hard**

    ## Problem Statement

    Given a binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

    **Example:**
    ```
    Input: matrix = [["1","0","1","0","0"],
                     ["1","0","1","1","1"],
                     ["1","1","1","1","1"],
                     ["1","0","0","1","0"]]
    Output: 6
    ```

    ## Solution

    ```python
    def maximalRectangle(matrix):
        """
        Find maximal rectangle using largest rectangle in histogram.
        
        Time: O(m * n)
        Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        def largest_rectangle_area(heights):
            """Helper function from previous problem."""
            stack = []
            max_area = 0
            heights.append(0)
            
            for i, height in enumerate(heights):
                while stack and heights[stack[-1]] > height:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                stack.append(i)
            
            return max_area
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        for row in matrix:
            # Update heights for current row
            for j in range(cols):
                if row[j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # Find max rectangle in current histogram
            max_area = max(max_area, largest_rectangle_area(heights[:]))
        
        return max_area
    ```

    ## Dynamic Programming Approach

    ```python
    def maximalRectangle(matrix):
        """
        DP approach tracking left, right, and height arrays.
        
        Time: O(m * n)
        Space: O(n)
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        left = [0] * cols
        right = [cols] * cols
        height = [0] * cols
        max_area = 0
        
        for i in range(rows):
            cur_left = 0
            cur_right = cols
            
            # Update height
            for j in range(cols):
                if matrix[i][j] == '1':
                    height[j] += 1
                else:
                    height[j] = 0
            
            # Update left
            for j in range(cols):
                if matrix[i][j] == '1':
                    left[j] = max(left[j], cur_left)
                else:
                    left[j] = 0
                    cur_left = j + 1
            
            # Update right
            for j in range(cols - 1, -1, -1):
                if matrix[i][j] == '1':
                    right[j] = min(right[j], cur_right)
                else:
                    right[j] = cols
                    cur_right = j
            
            # Calculate area
            for j in range(cols):
                max_area = max(max_area, (right[j] - left[j]) * height[j])
        
        return max_area
    ```

    ## Key Insights

    - Transform 2D problem to multiple 1D histogram problems
    - Build histogram incrementally for each row
    - DP approach tracks boundaries for each column

=== "Problem 3: Basic Calculator"

    **LeetCode 224** | **Difficulty: Hard**

    ## Problem Statement

    Given a string s representing a valid expression, implement a basic calculator to evaluate it.

    **Example:**
    ```
    Input: s = "(1+(4+5+2)-3)+(6+8)"
    Output: 23
    ```

    ## Solution

    ```python
    def calculate(s):
        """
        Evaluate expression using stack.
        
        Time: O(n)
        Space: O(n)
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
                # Reset for new sub-expression
                result = 0
                sign = 1
            elif char == ')':
                result += sign * number
                number = 0
                # Pop sign and previous result
                result *= stack.pop()  # sign before parentheses
                result += stack.pop()  # result before parentheses
        
        return result + sign * number
    ```

    ## Recursive Approach

    ```python
    def calculate(s):
        """
        Recursive descent parser.
        
        Time: O(n)
        Space: O(n)
        """
        index = 0
        
        def parse_expression():
            nonlocal index
            result = parse_term()
            
            while index < len(s) and s[index] in '+-':
                op = s[index]
                index += 1
                term = parse_term()
                if op == '+':
                    result += term
                else:
                    result -= term
            
            return result
        
        def parse_term():
            nonlocal index
            
            # Skip whitespace
            while index < len(s) and s[index] == ' ':
                index += 1
            
            if index < len(s) and s[index] == '(':
                index += 1  # skip '('
                result = parse_expression()
                index += 1  # skip ')'
                return result
            
            # Parse number
            start = index
            if index < len(s) and s[index] in '+-':
                index += 1
            
            while index < len(s) and s[index].isdigit():
                index += 1
            
            return int(s[start:index])
        
        return parse_expression()
    ```

    ## Key Insights

    - Stack handles nested parentheses elegantly
    - Track current result and sign separately
    - Recursive descent naturally handles operator precedence

=== "Problem 4: Basic Calculator II"

    **LeetCode 227** | **Difficulty: Hard**

    ## Problem Statement

    Given a string s which represents an expression, evaluate this expression and return its value.

    **Example:**
    ```
    Input: s = "3+2*2"
    Output: 7
    ```

    ## Solution

    ```python
    def calculate(s):
        """
        Evaluate expression with operator precedence.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        number = 0
        operation = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                number = number * 10 + int(char)
            
            if char in '+-*/' or i == len(s) - 1:
                if operation == '+':
                    stack.append(number)
                elif operation == '-':
                    stack.append(-number)
                elif operation == '*':
                    stack.append(stack.pop() * number)
                elif operation == '/':
                    # Handle negative division
                    prev = stack.pop()
                    stack.append(int(prev / number))
                
                operation = char
                number = 0
        
        return sum(stack)
    ```

    ## Without Stack (Constant Space)

    ```python
    def calculate(s):
        """
        Constant space solution.
        
        Time: O(n)
        Space: O(1)
        """
        result = 0
        current_number = 0
        last_number = 0
        operation = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                current_number = current_number * 10 + int(char)
            
            if char in '+-*/' or i == len(s) - 1:
                if operation == '+':
                    result += last_number
                    last_number = current_number
                elif operation == '-':
                    result += last_number
                    last_number = -current_number
                elif operation == '*':
                    last_number *= current_number
                elif operation == '/':
                    last_number = int(last_number / current_number)
                
                operation = char
                current_number = 0
        
        return result + last_number
    ```

    ## Key Insights

    - Stack handles operator precedence naturally
    - Process multiplication and division immediately
    - Addition and subtraction can be deferred

=== "Problem 5: Remove K Digits"

    **LeetCode 402** | **Difficulty: Hard**

    ## Problem Statement

    Given string num representing a non-negative integer, remove k digits from the number so that the new number is the smallest possible.

    **Example:**
    ```
    Input: num = "1432219", k = 3
    Output: "1219"
    ```

    ## Solution

    ```python
    def removeKdigits(num, k):
        """
        Remove k digits to get smallest number using monotonic stack.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        to_remove = k
        
        for digit in num:
            # Remove larger digits from stack
            while stack and stack[-1] > digit and to_remove > 0:
                stack.pop()
                to_remove -= 1
            
            stack.append(digit)
        
        # Remove remaining digits from end
        while to_remove > 0:
            stack.pop()
            to_remove -= 1
        
        # Convert to string and remove leading zeros
        result = ''.join(stack).lstrip('0')
        
        return result if result else '0'
    ```

    ## Alternative: Greedy Approach

    ```python
    def removeKdigits(num, k):
        """
        Greedy approach without explicit stack.
        
        Time: O(n)
        Space: O(n)
        """
        result = []
        
        for digit in num:
            # Remove digits that are larger than current digit
            while result and result[-1] > digit and k > 0:
                result.pop()
                k -= 1
            
            result.append(digit)
        
        # Remove remaining digits from end
        result = result[:-k] if k > 0 else result
        
        # Handle leading zeros and empty result
        result = ''.join(result).lstrip('0')
        return result if result else '0'
    ```

    ## Key Insights

    - Monotonic stack creates lexicographically smallest sequence
    - Greedy approach: remove larger digits first
    - Handle edge cases: leading zeros, empty result

=== "Problem 6: Trapping Rain Water"

    **LeetCode 42** | **Difficulty: Hard**

    ## Problem Statement

    Given n non-negative integers representing an elevation map, compute how much water can be trapped after raining.

    **Example:**
    ```
    Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
    Output: 6
    ```

    ## Solution

    ```python
    def trap(height):
        """
        Trap rainwater using stack.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        water = 0
        
        for i, h in enumerate(height):
            while stack and height[stack[-1]] < h:
                bottom = stack.pop()
                
                if not stack:
                    break
                
                # Calculate water level
                width = i - stack[-1] - 1
                bounded_height = min(h, height[stack[-1]]) - height[bottom]
                water += width * bounded_height
            
            stack.append(i)
        
        return water
    ```

    ## Two Pointers Approach

    ```python
    def trap(height):
        """
        Two pointers approach (optimal).
        
        Time: O(n)
        Space: O(1)
        """
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
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

    ## Key Insights

    - Stack approach calculates water layer by layer
    - Two pointers approach is more space-efficient
    - Water level determined by smaller of two boundaries

=== "Problem 7: Valid Parentheses"

    **LeetCode 20** | **Difficulty: Hard** (Extended Version)

    ## Problem Statement

    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. Extended to handle nested expressions.

    ## Solution

    ```python
    def isValid(s):
        """
        Check if parentheses are valid using stack.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        
        return len(stack) == 0
    ```

    ## Extended: Minimum Remove to Make Valid

    ```python
    def minRemoveToMakeValid(s):
        """
        Remove minimum parentheses to make string valid.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        to_remove = set()
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    stack.pop()
                else:
                    to_remove.add(i)
        
        # Mark unmatched opening parentheses
        to_remove.update(stack)
        
        # Build result
        result = []
        for i, char in enumerate(s):
            if i not in to_remove:
                result.append(char)
        
        return ''.join(result)
    ```

    ## Key Insights

    - Stack naturally handles nested structures
    - Track indices for removal problems
    - Process opening and closing brackets differently

=== "Problem 8: Score of Parentheses"

    **LeetCode 856** | **Difficulty: Hard**

    ## Problem Statement

    Given a balanced parentheses string s, return the score of the string.

    **Example:**
    ```
    Input: s = "(()(()))"
    Output: 6
    ```

    ## Solution

    ```python
    def scoreOfParentheses(s):
        """
        Calculate score using stack.
        
        Time: O(n)
        Space: O(n)
        """
        stack = [0]  # Initialize with 0 for base level
        
        for char in s:
            if char == '(':
                stack.append(0)
            else:  # char == ')'
                last = stack.pop()
                # If last is 0, this is a simple pair "()"
                # Otherwise, it's a nested structure
                stack[-1] += max(2 * last, 1)
        
        return stack[0]
    ```

    ## Constant Space Approach

    ```python
    def scoreOfParentheses(s):
        """
        Calculate score without extra space.
        
        Time: O(n)
        Space: O(1)
        """
        result = 0
        depth = 0
        
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            else:
                depth -= 1
                # If previous char was '(', we found a basic pair
                if s[i - 1] == '(':
                    result += 1 << depth  # 2^depth
        
        return result
    ```

    ## Key Insights

    - Stack tracks nested structure scores
    - Basic pairs "()" contribute 2^depth to total
    - Constant space approach leverages depth calculation

=== "Problem 9: Decode String"

    **LeetCode 394** | **Difficulty: Hard**

    ## Problem Statement

    Given an encoded string, return its decoded string. The encoding rule is: k[encoded_string].

    **Example:**
    ```
    Input: s = "3[a2[c]]"
    Output: "accaccacc"
    ```

    ## Solution

    ```python
    def decodeString(s):
        """
        Decode string using stack.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        current_string = ""
        current_number = 0
        
        for char in s:
            if char.isdigit():
                current_number = current_number * 10 + int(char)
            elif char == '[':
                # Push current state to stack
                stack.append((current_string, current_number))
                current_string = ""
                current_number = 0
            elif char == ']':
                # Pop from stack and decode
                prev_string, num = stack.pop()
                current_string = prev_string + num * current_string
            else:
                current_string += char
        
        return current_string
    ```

    ## Recursive Approach

    ```python
    def decodeString(s):
        """
        Recursive approach.
        
        Time: O(n)
        Space: O(n)
        """
        index = 0
        
        def decode():
            nonlocal index
            result = ""
            num = 0
            
            while index < len(s):
                char = s[index]
                
                if char.isdigit():
                    num = num * 10 + int(char)
                elif char == '[':
                    index += 1  # skip '['
                    substr = decode()
                    result += num * substr
                    num = 0
                elif char == ']':
                    return result
                else:
                    result += char
                
                index += 1
            
            return result
        
        return decode()
    ```

    ## Key Insights

    - Stack handles nested brackets naturally
    - Track both string and number states
    - Recursive approach mirrors stack behavior

=== "Problem 10: Asteroid Collision"

    **LeetCode 735** | **Difficulty: Hard**

    ## Problem Statement

    Given an array asteroids of integers representing asteroids in a row, find out the state after all collisions.

    **Example:**
    ```
    Input: asteroids = [5,10,-5]
    Output: [5,10]
    ```

    ## Solution

    ```python
    def asteroidCollision(asteroids):
        """
        Simulate asteroid collision using stack.
        
        Time: O(n)
        Space: O(n)
        """
        stack = []
        
        for asteroid in asteroids:
            # Process collisions
            while stack and asteroid < 0 and stack[-1] > 0:
                # Collision occurs
                if stack[-1] < -asteroid:
                    # Right-moving asteroid is destroyed
                    stack.pop()
                    continue
                elif stack[-1] == -asteroid:
                    # Both asteroids are destroyed
                    stack.pop()
                
                # Left-moving asteroid is destroyed
                break
            else:
                # No collision or collision resolved
                stack.append(asteroid)
        
        return stack
    ```

    ## Alternative: Simulation

    ```python
    def asteroidCollision(asteroids):
        """
        Direct simulation approach.
        
        Time: O(n)
        Space: O(n)
        """
        result = []
        
        for asteroid in asteroids:
            if asteroid > 0:
                result.append(asteroid)
            else:
                # Handle collisions with right-moving asteroids
                while result and result[-1] > 0 and result[-1] < -asteroid:
                    result.pop()
                
                if result and result[-1] == -asteroid:
                    result.pop()
                elif not result or result[-1] < 0:
                    result.append(asteroid)
        
        return result
    ```

    ## Key Insights

    - Stack naturally handles collision sequence
    - Only right-moving followed by left-moving asteroids collide
    - Multiple collisions can occur in sequence

=== "Problem 11: Next Greater Element II"

    **LeetCode 503** | **Difficulty: Hard**

    ## Problem Statement

    Given a circular integer array nums, return the next greater number for every element in nums.

    **Example:**
    ```
    Input: nums = [1,2,1]
    Output: [2,-1,2]
    ```

    ## Solution

    ```python
    def nextGreaterElements(nums):
        """
        Find next greater elements in circular array.
        
        Time: O(n)
        Space: O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # Process array twice to handle circular nature
        for i in range(2 * n):
            # Remove elements smaller than current
            while stack and nums[stack[-1]] < nums[i % n]:
                index = stack.pop()
                result[index] = nums[i % n]
            
            # Only push indices from first iteration
            if i < n:
                stack.append(i)
        
        return result
    ```

    ## Optimized Single Pass

    ```python
    def nextGreaterElements(nums):
        """
        Optimized approach with single conceptual pass.
        
        Time: O(n)
        Space: O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # Use doubled array concept
        for i in range(n * 2):
            while stack and nums[stack[-1]] < nums[i % n]:
                result[stack.pop()] = nums[i % n]
            
            if i < n:
                stack.append(i)
        
        return result
    ```

    ## Key Insights

    - Double the array to handle circular nature
    - Monotonic stack finds next greater element efficiently
    - Process each element at most twice

=== "Problem 12: Sliding Window Maximum"

    **LeetCode 239** | **Difficulty: Hard**

    ## Problem Statement

    Given an array nums and a sliding window of size k, return the maximum value in each window.

    **Example:**
    ```
    Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]
    ```

    ## Solution

    ```python
    from collections import deque

    def maxSlidingWindow(nums, k):
        """
        Find sliding window maximum using deque.
        
        Time: O(n)
        Space: O(k)
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        for i in range(len(nums)):
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # Remove indices with smaller values
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            
            dq.append(i)
            
            # Add maximum to result (window is complete)
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    ```

    ## Stack-Based Approach

    ```python
    def maxSlidingWindow(nums, k):
        """
        Using two stacks to simulate deque.
        
        Time: O(n)
        Space: O(k)
        """
        def push_stack(stack, val):
            stack.append((val, max(val, stack[-1][1] if stack else val)))
        
        def pop_stack(stack):
            return stack.pop()[0]
        
        def get_max(stack):
            return stack[-1][1] if stack else float('-inf')
        
        left_stack = []
        right_stack = []
        result = []
        
        for i in range(len(nums)):
            # Add current element to right stack
            push_stack(right_stack, nums[i])
            
            # Remove elements outside window
            if i >= k:
                if not left_stack:
                    # Move elements from right to left
                    while right_stack:
                        push_stack(left_stack, pop_stack(right_stack))
                
                pop_stack(left_stack)
            
            # Get maximum for current window
            if i >= k - 1:
                result.append(max(get_max(left_stack), get_max(right_stack)))
        
        return result
    ```

    ## Key Insights

    - Deque maintains elements in decreasing order
    - Each element is added and removed at most once
    - Stack-based approach provides alternative implementation

=== "Problem 13: Minimum Stack"

    **LeetCode 155** | **Difficulty: Hard** (Extended Version)

    ## Problem Statement

    Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

    ## Solution

    ```python
    class MinStack:
        """
        Stack with constant time minimum retrieval.
        
        Time: O(1) for all operations
        Space: O(n)
        """
        
        def __init__(self):
            self.stack = []
            self.min_stack = []
        
        def push(self, val):
            self.stack.append(val)
            # Push to min_stack if it's empty or val is smaller or equal
            if not self.min_stack or val <= self.min_stack[-1]:
                self.min_stack.append(val)
        
        def pop(self):
            if self.stack:
                val = self.stack.pop()
                # Pop from min_stack if it's the minimum
                if self.min_stack and val == self.min_stack[-1]:
                    self.min_stack.pop()
                return val
        
        def top(self):
            return self.stack[-1] if self.stack else None
        
        def getMin(self):
            return self.min_stack[-1] if self.min_stack else None
    ```

    ## Single Stack Approach

    ```python
    class MinStack:
        """
        Single stack approach storing differences.
        
        Time: O(1) for all operations
        Space: O(n)
        """
        
        def __init__(self):
            self.stack = []
            self.min_val = None
        
        def push(self, val):
            if not self.stack:
                self.stack.append(0)
                self.min_val = val
            else:
                # Store difference from minimum
                diff = val - self.min_val
                self.stack.append(diff)
                
                # Update minimum if necessary
                if val < self.min_val:
                    self.min_val = val
        
        def pop(self):
            if not self.stack:
                return None
            
            diff = self.stack.pop()
            
            if diff < 0:
                # Minimum was popped, restore previous minimum
                val = self.min_val
                self.min_val = self.min_val - diff
                return val
            else:
                return self.min_val + diff
        
        def top(self):
            if not self.stack:
                return None
            
            diff = self.stack[-1]
            return self.min_val if diff < 0 else self.min_val + diff
        
        def getMin(self):
            return self.min_val
    ```

    ## Key Insights

    - Auxiliary stack tracks minimum at each level
    - Single stack approach uses difference encoding
    - Both approaches achieve O(1) time complexity

=== "Problem 14: Exclusive Time of Functions"

    **LeetCode 636** | **Difficulty: Hard**

    ## Problem Statement

    Given the running logs of n functions, return the exclusive time of each function.

    **Example:**
    ```
    Input: n = 2, logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
    Output: [3,4]
    ```

    ## Solution

    ```python
    def exclusiveTime(n, logs):
        """
        Calculate exclusive time using stack.
        
        Time: O(m) where m is number of logs
        Space: O(n)
        """
        stack = []
        result = [0] * n
        prev_time = 0
        
        for log in logs:
            function_id, action, timestamp = log.split(':')
            function_id = int(function_id)
            timestamp = int(timestamp)
            
            if action == 'start':
                # Add time to current running function
                if stack:
                    result[stack[-1]] += timestamp - prev_time
                
                stack.append(function_id)
                prev_time = timestamp
            else:  # action == 'end'
                # Add time to ending function
                result[stack.pop()] += timestamp - prev_time + 1
                prev_time = timestamp + 1
        
        return result
    ```

    ## Alternative: Interval Processing

    ```python
    def exclusiveTime(n, logs):
        """
        Process intervals explicitly.
        
        Time: O(m)
        Space: O(n)
        """
        result = [0] * n
        stack = []
        
        for log in logs:
            function_id, action, timestamp = log.split(':')
            function_id = int(function_id)
            timestamp = int(timestamp)
            
            if action == 'start':
                stack.append([function_id, timestamp])
            else:
                start_time = stack.pop()[1]
                duration = timestamp - start_time + 1
                result[function_id] += duration
                
                # Subtract nested function time
                if stack:
                    result[stack[-1][0]] -= duration
        
        return result
    ```

    ## Key Insights

    - Stack tracks function call hierarchy
    - Exclusive time = total time - nested function time
    - Handle start and end timestamps carefully

=== "Problem 15: Implement Queue using Stacks"

    **LeetCode 232** | **Difficulty: Hard** (Optimized Version)

    ## Problem Statement

    Implement a first in first out (FIFO) queue using only two stacks.

    ## Solution

    ```python
    class MyQueue:
        """
        Queue implementation using two stacks.
        
        Push: O(1)
        Pop: O(1) amortized
        Peek: O(1) amortized
        Empty: O(1)
        """
        
        def __init__(self):
            self.input_stack = []
            self.output_stack = []
        
        def push(self, x):
            """Add element to queue."""
            self.input_stack.append(x)
        
        def pop(self):
            """Remove element from queue."""
            self.peek()
            return self.output_stack.pop()
        
        def peek(self):
            """Get front element."""
            if not self.output_stack:
                # Move all elements from input to output
                while self.input_stack:
                    self.output_stack.append(self.input_stack.pop())
            
            return self.output_stack[-1]
        
        def empty(self):
            """Check if queue is empty."""
            return len(self.input_stack) == 0 and len(self.output_stack) == 0
    ```

    ## Alternative: Single Stack with Recursion

    ```python
    class MyQueue:
        """
        Queue using single stack with recursion.
        
        All operations: O(n)
        """
        
        def __init__(self):
            self.stack = []
        
        def push(self, x):
            """Add element to queue."""
            self.stack.append(x)
        
        def pop(self):
            """Remove element from queue."""
            if len(self.stack) == 1:
                return self.stack.pop()
            
            # Remove bottom element using recursion
            item = self.stack.pop()
            result = self.pop()
            self.stack.append(item)
            return result
        
        def peek(self):
            """Get front element."""
            if len(self.stack) == 1:
                return self.stack[-1]
            
            item = self.stack.pop()
            result = self.peek()
            self.stack.append(item)
            return result
        
        def empty(self):
            """Check if queue is empty."""
            return len(self.stack) == 0
    ```

    ## Key Insights

    - Two stacks achieve amortized O(1) operations
    - Output stack reverses order of input stack
    - Single stack approach uses recursion for ordering
