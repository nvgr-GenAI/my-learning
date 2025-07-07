# Greedy Algorithms - Easy Problems

## Problem Categories

### 1. Basic Optimization
- Coin change (canonical systems)
- Activity selection
- Fractional knapsack

### 2. Scheduling Problems  
- Job scheduling
- Meeting rooms
- Task assignment

### 3. Array and String Problems
- Jump game
- Gas station
- Remove duplicates

---

## 1. Coin Change (Greedy Works for Canonical Systems)

**Problem**: Given coins and a target amount, find minimum number of coins needed.

**Note**: Greedy only works for canonical coin systems (like US coins: 1, 5, 10, 25).

**Example**:
```
Input: coins = [1, 5, 10, 25], amount = 30
Output: 2 (25 + 5)
```

**Solution**:
```python
def coin_change_greedy(coins, amount):
    """
    Greedy coin change for canonical systems only.
    
    Time Complexity: O(n) where n is number of coin types
    Space Complexity: O(1)
    
    WARNING: This only works for canonical coin systems!
    """
    # Sort coins in descending order
    coins.sort(reverse=True)
    
    count = 0
    result = []
    
    for coin in coins:
        # Greedy choice: use as many of current coin as possible
        while amount >= coin:
            amount -= coin
            count += 1
            result.append(coin)
    
    return count if amount == 0 else -1, result

# Example where greedy fails (non-canonical system)
def coin_change_dp(coins, amount):
    """
    Correct DP solution for any coin system.
    
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Test
print(coin_change_greedy([1, 5, 10, 25], 30))  # Output: (2, [25, 5])
print(coin_change_greedy([1, 3, 4], 6))        # Output: (3, [4, 1, 1]) - WRONG!
print(coin_change_dp([1, 3, 4], 6))            # Output: 2 - CORRECT (3 + 3)
```

**Key Points**:
- Greedy works only for canonical coin systems
- Always use largest denomination first
- For general coin systems, use dynamic programming

---

## 2. Activity Selection Problem

**Problem**: Select the maximum number of activities that don't overlap.

**Example**:
```
Input: activities = [(1,3), (2,4), (3,5), (0,6)]
Output: 2 activities: (1,3) and (3,5)
```

**Solution**:
```python
def activity_selection(activities):
    """
    Select maximum non-overlapping activities.
    
    Time Complexity: O(n log n) - due to sorting
    Space Complexity: O(1)
    """
    if not activities:
        return []
    
    # Greedy criterion: sort by finish time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish_time = activities[0][1]
    
    for i in range(1, len(activities)):
        start_time, finish_time = activities[i]
        
        # Greedy choice: select if no overlap
        if start_time >= last_finish_time:
            selected.append(activities[i])
            last_finish_time = finish_time
    
    return selected

# Variation: Activity selection with weights
def weighted_activity_selection(activities):
    """
    Select activities to maximize total weight (requires DP, not greedy).
    
    This shows where greedy fails - when we need to consider weights.
    """
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    n = len(activities)
    
    # DP approach needed here
    dp = [0] * n
    dp[0] = activities[0][2]  # weight
    
    for i in range(1, n):
        # Include current activity
        include_weight = activities[i][2]
        
        # Find latest non-overlapping activity
        latest_non_overlap = -1
        for j in range(i-1, -1, -1):
            if activities[j][1] <= activities[i][0]:
                latest_non_overlap = j
                break
        
        if latest_non_overlap != -1:
            include_weight += dp[latest_non_overlap]
        
        # Choose maximum
        dp[i] = max(dp[i-1], include_weight)
    
    return dp[n-1]

# Test
activities = [(1, 3), (2, 4), (3, 5), (0, 6)]
print(activity_selection(activities))  # Output: [(1, 3), (3, 5)]
```

**Key Points**:
- Sort by finish time (not start time or duration)
- Greedy choice: earliest finish time maximizes remaining time
- Exchange argument proves optimality

---

## 3. Jump Game

**Problem**: Determine if you can reach the last index of an array where each element represents maximum jump length.

**Example**:
```
Input: nums = [2,3,1,1,4]
Output: true (jump 1 step from index 0 to 1, then 3 steps to last index)
```

**Solution**:
```python
def can_jump(nums):
    """
    Determine if we can reach the last index.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return False
    
    max_reach = 0
    
    for i in range(len(nums)):
        # If current position is unreachable
        if i > max_reach:
            return False
        
        # Greedy choice: update maximum reachable position
        max_reach = max(max_reach, i + nums[i])
        
        # Early termination: if we can reach the end
        if max_reach >= len(nums) - 1:
            return True
    
    return True

def jump_game_minimum_jumps(nums):
    """
    Find minimum number of jumps to reach the last index.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        # Update farthest reachable position
        farthest = max(farthest, i + nums[i])
        
        # If we've reached the end of current jump
        if i == current_end:
            jumps += 1
            current_end = farthest
            
            # If we can reach the end
            if current_end >= len(nums) - 1:
                break
    
    return jumps

# Test
print(can_jump([2, 3, 1, 1, 4]))     # Output: True
print(can_jump([3, 2, 1, 0, 4]))     # Output: False
print(jump_game_minimum_jumps([2, 3, 1, 1, 4]))  # Output: 2
```

**Key Points**:
- Track maximum reachable position at each step
- Greedy choice: always try to reach as far as possible
- For minimum jumps, use level-by-level BFS approach

---

## 4. Gas Station Problem

**Problem**: There are gas stations along a circular route. Determine if you can complete the circuit.

**Example**:
```
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3 (start from station 3)
```

**Solution**:
```python
def can_complete_circuit(gas, cost):
    """
    Find starting gas station to complete circuit.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    total_gas = sum(gas)
    total_cost = sum(cost)
    
    # If total gas < total cost, impossible to complete circuit
    if total_gas < total_cost:
        return -1
    
    current_gas = 0
    start_station = 0
    
    for i in range(len(gas)):
        current_gas += gas[i] - cost[i]
        
        # If we can't proceed from current station
        if current_gas < 0:
            # Greedy choice: start from next station
            start_station = i + 1
            current_gas = 0
    
    return start_station

def can_complete_circuit_detailed(gas, cost):
    """
    More detailed explanation of the greedy approach.
    """
    n = len(gas)
    total_surplus = 0
    current_surplus = 0
    start = 0
    
    for i in range(n):
        surplus = gas[i] - cost[i]
        total_surplus += surplus
        current_surplus += surplus
        
        # If current surplus becomes negative, we can't reach here from start
        if current_surplus < 0:
            # Reset: start from next station
            start = i + 1
            current_surplus = 0
    
    # If total surplus is non-negative, circuit is possible
    return start if total_surplus >= 0 else -1

# Test
gas = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
print(can_complete_circuit(gas, cost))  # Output: 3
```

**Key Points**:
- If total gas ≥ total cost, solution exists and is unique
- When current gas becomes negative, restart from next station
- Greedy choice: if we can't reach station i from any station ≤ j, then we can't reach i from any station between j and i

---

## 5. Remove Duplicate Letters

**Problem**: Remove duplicate letters so that every letter appears exactly once and the result is lexicographically smallest.

**Example**:
```
Input: s = "bcabc"
Output: "abc"
```

**Solution**:
```python
def remove_duplicate_letters(s):
    """
    Remove duplicates to get lexicographically smallest result.
    
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 26 characters
    """
    from collections import Counter
    
    # Count frequency of each character
    count = Counter(s)
    result = []
    in_result = set()
    
    for char in s:
        # Decrease count as we process
        count[char] -= 1
        
        # Skip if already in result
        if char in in_result:
            continue
        
        # Greedy choice: remove larger characters from end if they appear later
        while (result and 
               result[-1] > char and 
               count[result[-1]] > 0):
            removed = result.pop()
            in_result.remove(removed)
        
        result.append(char)
        in_result.add(char)
    
    return ''.join(result)

def remove_k_digits(num, k):
    """
    Related problem: Remove k digits to make smallest number.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    to_remove = k
    
    for digit in num:
        # Greedy choice: remove larger digits from end if possible
        while stack and stack[-1] > digit and to_remove > 0:
            stack.pop()
            to_remove -= 1
        
        stack.append(digit)
    
    # Remove remaining digits from end
    while to_remove > 0:
        stack.pop()
        to_remove -= 1
    
    # Handle edge cases
    result = ''.join(stack).lstrip('0')
    return result if result else '0'

# Test
print(remove_duplicate_letters("bcabc"))    # Output: "abc"
print(remove_duplicate_letters("cbacdcbc")) # Output: "acdb"
print(remove_k_digits("1432219", 3))       # Output: "1219"
```

**Key Points**:
- Use stack to maintain result in increasing order
- Remove characters greedily when they can appear later
- Count remaining occurrences to decide if removal is safe

---

## 6. Assign Cookies

**Problem**: Assign cookies to children to maximize the number of content children.

**Example**:
```
Input: children = [1,2,3], cookies = [1,1]
Output: 1
```

**Solution**:
```python
def find_content_children(children, cookies):
    """
    Maximize number of content children.
    
    Time Complexity: O(n log n + m log m)
    Space Complexity: O(1)
    """
    # Sort both arrays
    children.sort()
    cookies.sort()
    
    child_idx = 0
    content_children = 0
    
    for cookie in cookies:
        # Greedy choice: give cookie to smallest unsatisfied child who can be satisfied
        if child_idx < len(children) and cookie >= children[child_idx]:
            content_children += 1
            child_idx += 1
    
    return content_children

def assign_cookies_detailed(children, cookies):
    """
    More detailed greedy approach with explanation.
    """
    children.sort()  # Sort by greed factor
    cookies.sort()   # Sort by size
    
    result = []
    child_idx = 0
    
    for cookie_idx, cookie_size in enumerate(cookies):
        # Find the first child that can be satisfied by this cookie
        while child_idx < len(children) and children[child_idx] > cookie_size:
            child_idx += 1
        
        # If found a child, assign cookie
        if child_idx < len(children):
            result.append((child_idx, cookie_idx))
            child_idx += 1
    
    return len(result), result

# Test
children = [1, 2, 3]
cookies = [1, 1]
print(find_content_children(children, cookies))  # Output: 1

children = [1, 2]
cookies = [1, 2, 3]
print(find_content_children(children, cookies))  # Output: 2
```

**Key Points**:
- Sort both children and cookies
- Assign smallest adequate cookie to smallest unsatisfied child
- Greedy choice maximizes the number of satisfied children

---

## Common Patterns in Easy Greedy Problems

### 1. Sorting + Linear Scan

```python
def greedy_sorting_pattern(items, criteria):
    """
    Common pattern: sort by greedy criteria, then scan linearly.
    """
    items.sort(key=criteria)
    
    result = []
    for item in items:
        if is_beneficial(item, result):
            result.append(item)
    
    return result
```

### 2. Maintain Running State

```python
def greedy_running_state_pattern(sequence):
    """
    Pattern: maintain running state and make greedy decisions.
    """
    current_state = initialize_state()
    
    for element in sequence:
        if should_update_state(element, current_state):
            current_state = update_state(element, current_state)
    
    return extract_result(current_state)
```

### 3. Stack-Based Greedy

```python
def greedy_stack_pattern(sequence):
    """
    Pattern: use stack to maintain optimal order.
    """
    stack = []
    
    for element in sequence:
        while stack and should_remove_top(stack[-1], element):
            stack.pop()
        
        if should_add(element, stack):
            stack.append(element)
    
    return stack
```

## Tips for Easy Greedy Problems

### 1. Identify the Greedy Choice
- What local decision leads to global optimum?
- Can you sort the input to make greedy choice obvious?
- Is there a clear "best" choice at each step?

### 2. Prove Correctness (Informally)
- Why does the greedy choice not prevent future optimal choices?
- Can you use exchange argument or stays-ahead proof?

### 3. Implementation Tips
- Sort input when beneficial
- Use appropriate data structures (heap, stack, etc.)
- Handle edge cases (empty input, single element)
- Consider early termination conditions

### 4. Common Gotchas
- Ensure greedy choice is actually optimal
- Watch out for non-canonical systems (like coin change)
- Consider whether weights/priorities affect the greedy choice
- Verify that local optimum leads to global optimum

These easy problems demonstrate fundamental greedy principles and provide building blocks for more complex greedy algorithms.
