# Greedy Algorithms - Easy Problems

## 🎯 Learning Objectives

Master fundamental greedy algorithm concepts and decision-making patterns:

- Greedy choice property and optimal substructure
- Sorting-based greedy approaches
- Scheduling and interval problems
- Stack-based greedy techniques
- Proof techniques for greedy correctness

=== "📋 Problem List"

    | # | Problem | Pattern | Difficulty | Time | Space |
    |---|---------|---------|------------|------|-------|
    | 1 | Assign Cookies | Sorting + Two Pointers | Easy | O(n log n) | O(1) |
    | 2 | Lemonade Change | Greedy State Tracking | Easy | O(n) | O(1) |
    | 3 | Best Time to Buy and Sell Stock II | Peak-Valley | Easy | O(n) | O(1) |
    | 4 | Jump Game | Greedy Reachability | Easy | O(n) | O(1) |
    | 5 | Gas Station | Circular Array Greedy | Easy | O(n) | O(1) |
    | 6 | Remove Duplicate Letters | Stack + Greedy | Easy | O(n) | O(1) |
    | 7 | Monotonic Array | Single Pass | Easy | O(n) | O(1) |
    | 8 | Two City Scheduling | Sorting by Difference | Easy | O(n log n) | O(1) |
    | 9 | Minimum Deletions for Sorted Array | Greedy LIS | Easy | O(n) | O(1) |
    | 10 | Last Stone Weight | Greedy with Heap | Easy | O(n log n) | O(n) |
    | 11 | Boats to Save People | Two Pointers | Easy | O(n log n) | O(1) |
    | 12 | Minimum Cost Climbing Stairs | DP/Greedy Choice | Easy | O(n) | O(1) |
    | 13 | Is Subsequence | Two Pointers | Easy | O(n) | O(1) |
    | 14 | Can Place Flowers | Linear Scan | Easy | O(n) | O(1) |
    | 15 | Maximize Sum After K Negations | Sorting + Greedy | Easy | O(n log n) | O(1) |

=== "🎯 Core Patterns"

    **💰 Greedy Choice Property:**
    - Make locally optimal choice at each step
    - Trust that local optimum leads to global optimum
    - Identify what makes a choice "greedy"
    
    **📊 Sorting-Based Greedy:**
    - Sort by key property (size, time, cost)
    - Process elements in optimal order
    - Two-pointer technique after sorting
    
    **🔄 State Tracking:**
    - Maintain running totals or states
    - Update state based on greedy decisions
    - Handle edge cases in state transitions
    
    **📚 Stack-Based Greedy:**
    - Use stack to maintain optimal sequence
    - Remove suboptimal elements greedily
    - Build result incrementally

=== "💡 Solutions"

    === "Assign Cookies"
        ```python
        def findContentChildren(g, s):
            """
            Assign cookies to children greedily
            Sort both arrays and use two pointers
            """
            g.sort()  # greed factors
            s.sort()  # cookie sizes
            
            child = cookie = 0
            while child < len(g) and cookie < len(s):
                if s[cookie] >= g[child]:
                    child += 1  # child is satisfied
                cookie += 1
            
            return child
        ```
    
    === "Lemonade Change"
        ```python
        def lemonadeChange(bills):
            """
            Track $5 and $10 bills for making change
            Greedy: prefer giving $10 over two $5s
            """
            five = ten = 0
            
            for bill in bills:
                if bill == 5:
                    five += 1
                elif bill == 10:
                    if five == 0:
                        return False
                    five -= 1
                    ten += 1
                else:  # bill == 20
                    # Prefer giving one $10 and one $5
                    if ten > 0 and five > 0:
                        ten -= 1
                        five -= 1
                    elif five >= 3:
                        five -= 3
                    else:
                        return False
            
            return True
        ```
    
    === "Best Time to Buy and Sell Stock II"
        ```python
        def maxProfit(prices):
            """
            Buy and sell multiple times
            Greedy: capture every profitable opportunity
            """
            profit = 0
            
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    profit += prices[i] - prices[i-1]
            
            return profit
        ```
    
    === "Jump Game"
        ```python
        def canJump(nums):
            """
            Determine if can reach last index
            Greedy: track maximum reachable position
            """
            max_reach = 0
            
            for i in range(len(nums)):
                if i > max_reach:
                    return False
                max_reach = max(max_reach, i + nums[i])
                if max_reach >= len(nums) - 1:
                    return True
            
            return True
        ```
    
    === "Gas Station"
        ```python
        def canCompleteCircuit(gas, cost):
            """
            Find starting gas station for circular trip
            Greedy: if total gas >= total cost, solution exists
            """
            total_gas = total_cost = current_gas = start = 0
            
            for i in range(len(gas)):
                total_gas += gas[i]
                total_cost += cost[i]
                current_gas += gas[i] - cost[i]
                
                # If can't reach next station, try starting from next
                if current_gas < 0:
                    start = i + 1
                    current_gas = 0
            
            return start if total_gas >= total_cost else -1
        ```
    
    === "Remove Duplicate Letters"
        ```python
        def removeDuplicateLetters(s):
            """
            Remove duplicates to get lexicographically smallest result
            Use stack with greedy removal strategy
            """
            count = {}
            in_stack = set()
            stack = []
            
            # Count occurrences
            for char in s:
                count[char] = count.get(char, 0) + 1
            
            for char in s:
                count[char] -= 1
                
                if char in in_stack:
                    continue
                
                # Remove characters that are larger and appear later
                while (stack and stack[-1] > char and 
                       count[stack[-1]] > 0):
                    removed = stack.pop()
                    in_stack.remove(removed)
                
                stack.append(char)
                in_stack.add(char)
            
            return ''.join(stack)
        ```
    
    === "Two City Scheduling"
        ```python
        def twoCitySchedCost(costs):
            """
            Send n people to city A and n to city B
            Greedy: sort by cost difference
            """
            # Sort by difference (cost_A - cost_B)
            costs.sort(key=lambda x: x[0] - x[1])
            
            n = len(costs) // 2
            total_cost = 0
            
            # Send first n to city A, rest to city B
            for i in range(n):
                total_cost += costs[i][0]  # City A
            for i in range(n, 2 * n):
                total_cost += costs[i][1]  # City B
            
            return total_cost
        ```
    
    === "Boats to Save People"
        ```python
        def numRescueBoats(people, limit):
            """
            Minimum boats needed (capacity = 2 people max)
            Greedy: pair heaviest with lightest
            """
            people.sort()
            left, right = 0, len(people) - 1
            boats = 0
            
            while left <= right:
                # Always take the heaviest person
                if people[left] + people[right] <= limit:
                    left += 1  # Take the lightest too
                right -= 1
                boats += 1
            
            return boats
        ```

=== "📊 Key Insights"

    **🔧 Greedy Strategy Selection:**
    - **Sorting First**: When order matters for optimal choice
    - **State Tracking**: For problems with running totals/counts
    - **Stack/Queue**: For maintaining optimal sequences
    - **Two Pointers**: After sorting for pairing problems
    
    **⚡ Proof Techniques:**
    - **Exchange Argument**: Show greedy is no worse than optimal
    - **Optimal Substructure**: Optimal solution contains optimal subsolutions
    - **Greedy Choice Property**: Local optimum leads to global optimum
    - **Contradiction**: Assume non-greedy is better, derive contradiction
    
    **🎯 Pattern Recognition:**
    - **Scheduling**: Sort by end time, process earliest first
    - **Resource Allocation**: Sort by efficiency ratio
    - **Interval Problems**: Sort by start/end time
    - **String Problems**: Use stack for lexicographical ordering

=== "🚀 Advanced Tips"

    **💡 Problem-Solving Approach:**
    1. **Identify Greedy Choice**: What locally optimal decision to make?
    2. **Prove Correctness**: Why does greedy work for this problem?
    3. **Handle Edge Cases**: Empty input, single element, ties
    4. **Optimize Implementation**: Use appropriate data structures
    5. **Test Thoroughly**: Verify with examples and edge cases
    
    **🔍 Common Pitfalls:**
    - **Greedy Doesn't Always Work**: Some problems need DP/backtracking
    - **Sorting Order**: Choose correct sorting criteria
    - **State Management**: Maintain consistent state throughout
    - **Edge Cases**: Handle empty arrays, single elements
    
    **🏆 Best Practices:**
    - Start with brute force, then identify greedy opportunity
    - Prove greedy correctness before implementing
    - Use sorting when order affects optimal choice
    - Maintain simple, clear state variables
    - Test with counterexamples to verify approach

## 📝 Summary

These easy greedy problems focus on:

- **Fundamental Greedy Patterns** with clear optimal choices
- **Sorting-Based Approaches** for order-dependent problems
- **State Tracking** for problems with running totals
- **Stack Techniques** for sequence optimization
- **Proof Intuition** for understanding why greedy works

Master these patterns to recognize when greedy algorithms apply and build confidence in their correctness!
