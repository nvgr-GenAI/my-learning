# Problem-Solving Patterns

Master the essential patterns that solve 95% of coding interview problems. Each pattern includes theory, implementations, variants, and practice problems.

---

## 🎯 Pattern Selection Guide

=== "Quick Reference"
    | Problem Clue | Pattern | Complexity |
    |--------------|---------|------------|
    | Sorted array + search | [Binary Search](#binary-search) | O(log n) |
    | Find pair with sum | [Two Pointers](#two-pointers) | O(n) |
    | Subarray/substring constraint | [Sliding Window](#sliding-window) | O(n) |
    | LinkedList cycle/middle | [Fast & Slow Pointers](#fast-slow-pointers) | O(n) |
    | Level-order traversal | [BFS](#breadth-first-search-bfs) | O(V+E) |
    | All paths/backtracking | [DFS](#depth-first-search-dfs) | O(V+E) |
    | Fast lookups | [Hash Map](#hash-map) | O(n) |
    | Overlapping subproblems | [Dynamic Programming](#dynamic-programming) | Varies |
    | Local optimal choices | [Greedy](#greedy-algorithm) | O(n log n) |
    | Numbers 1 to n | [Cyclic Sort](#cyclic-sort) | O(n) |
    | Merge k sorted lists | [K-way Merge](#k-way-merge) | O(N log k) |
    | Top K elements | [Heap](#top-k-elements-heap) | O(n log k) |
    | Overlapping intervals | [Merge Intervals](#merge-intervals) | O(n log n) |
    | Connected components | [Union Find](#union-find) | O(α(n)) |
    | Prefix matching | [Trie](#trie-prefix-tree) | O(m) |
    | Task dependencies | [Topological Sort](#topological-sort) | O(V+E) |
    | All combinations | [Backtracking](#backtracking) | O(2ⁿ) |
    | Next greater element | [Monotonic Stack](#monotonic-stack) | O(n) |
    | Range queries | [Prefix Sum](#prefix-sum) | O(1) |
    | Binary operations | [Bit Manipulation](#bit-manipulation) | O(1) |

=== "By Category"
    **Array/String:** [Two Pointers](#two-pointers) • [Sliding Window](#sliding-window) • [Cyclic Sort](#cyclic-sort) • [Prefix Sum](#prefix-sum) • [Monotonic Stack](#monotonic-stack) • [Merge Intervals](#merge-intervals)

    **LinkedList:** [Fast & Slow Pointers](#fast-slow-pointers)

    **Tree/Graph:** [BFS](#breadth-first-search-bfs) • [DFS](#depth-first-search-dfs) • [Union Find](#union-find) • [Topological Sort](#topological-sort)

    **Search:** [Binary Search](#binary-search) • [Trie](#trie-prefix-tree)

    **Optimization:** [Dynamic Programming](#dynamic-programming) • [Greedy](#greedy-algorithm) • [Backtracking](#backtracking)

    **Data Structure:** [Hash Map](#hash-map) • [Heap](#top-k-elements-heap) • [K-way Merge](#k-way-merge)

    **Advanced:** [Bit Manipulation](#bit-manipulation)

---

## Array & String Patterns

### Two Pointers

=== "Understanding the Pattern"

    ## What is Two Pointers?

    Imagine you're at a library bookshelf trying to find two books whose page counts add up to exactly 500 pages. The naive approach? Pick each book and check it against every other book—that's checking thousands of combinations!

    But wait—if the books are arranged by page count, you can be smarter. Start with the thinnest book (left end) and thickest book (right end):
    - Sum too small? The thin book isn't thick enough. Move to a thicker one (move left pointer right)
    - Sum too large? The thick book is too heavy. Try a thinner one (move right pointer left)
    - Just right? Found your pair!

    This is **Two Pointers**: using two references that traverse the data from different positions, making intelligent decisions at each step to eliminate impossible solutions.

    ---

    ## How It Works

    The Two Pointers pattern works by maintaining two indices (pointers) that scan through your data structure. There are two main approaches:

    **1. Opposite Direction (Converging):**
    ```
    Start:  [2, 3, 5, 7, 11, 15]
             ↑                ↑
            left            right

    Move left or right based on current values
    Eventually meet in middle
    ```

    **2. Same Direction (Chasing):**
    ```
    Start:  [2, 3, 5, 7, 11, 15]
             ↑
           slow/fast

    One pointer explores ahead
    Other follows at different pace
    ```

    ---

    ## Key Intuition

    **The Aha Moment:** Instead of checking all n×n possible pairs (O(n²)), we leverage the sorted property to eliminate entire ranges of impossible solutions with each move.

    Think of it like playing the "guess the number" game:
    - Checking every number from 1 to 100? Takes up to 100 guesses.
    - Using high/low hints? Takes at most 7 guesses!

    Two pointers gives us those "hints" through the sorted structure:
    - Sum too small → we need bigger numbers → move left pointer right
    - Sum too large → we need smaller numbers → move right pointer left

    Each move eliminates multiple impossible combinations at once!

    ---

    ## Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n) | Single pass through array, visiting each element once |
    | **Space** | O(1) | Only using two pointer variables, no extra data structures |
    | **Improvement** | From O(n²) | Reduces nested loops to single pass |

    **Why so fast?** Each pointer moves at most n times, and we never backtrack. Total operations: at most 2n steps.

=== "When to Use This Pattern"

    ## Perfect For

    | Scenario | Why Two Pointers Works | Example |
    |----------|----------------------|---------|
    | **Sorted array + find pair** | Can eliminate ranges by sum comparison | Two Sum II, 3Sum |
    | **Palindrome checking** | Compare from both ends moving inward | Valid Palindrome |
    | **Remove duplicates in-place** | One pointer explores, one tracks write position | Remove Duplicates |
    | **Partitioning arrays** | Separate elements meeting criteria | Sort Colors |
    | **Finding triplets** | Fix one element, two pointers on rest | 3Sum, 4Sum |

    **Red Flags That Suggest Two Pointers:**
    - Problem mentions "sorted array"
    - Need to find "pair" or "triplet" with certain sum
    - Words like "palindrome", "reverse", "partition"
    - In-place modifications required
    - Comparing elements from different positions

    ---

    ## When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Unsorted array** | Can't make elimination decisions | Sort first O(n log n) or use Hash Map O(n) |
    | **Need ALL pairs** | Two pointers optimizes by skipping, but you need to check everything | Nested loops O(n²) unavoidable |
    | **Order matters in output** | Two pointers doesn't preserve original order | Store indices separately |
    | **Complex conditions** | Simple comparisons work best | Consider DP or other approaches |
    | **Very small arrays (n < 10)** | Optimization overhead not worth it | Simple nested loops fine |

    ---

    ## Decision Flowchart

    ```
    Is array sorted?
    ├─ Yes → Is it about finding pairs/combinations?
    │         ├─ Yes → USE TWO POINTERS ✓
    │         └─ No → Check other criteria
    └─ No → Can you sort it first?
              ├─ Yes → Consider sorting + two pointers
              └─ No → Use Hash Map instead
    ```

=== "Implementation Templates"

    === "Template 1: Opposite Ends"

        **Use Case:** Finding pairs in sorted array, palindrome checks

        **Pattern:** Start from both ends, move toward center

        ```python
        def two_pointers_opposite(arr, target):
            """
            Two pointers converging from opposite ends.

            Perfect for: Sorted array problems, finding pairs with target sum

            Time: O(n) - Each pointer moves at most n times
            Space: O(1) - Only pointer variables

            Example: Two Sum II in sorted array
            """
            left, right = 0, len(arr) - 1

            while left < right:
                current_sum = arr[left] + arr[right]

                if current_sum == target:
                    # Found our pair!
                    return [left, right]

                elif current_sum < target:
                    # Need larger sum, move left pointer right
                    # This eliminates all pairs with arr[left]
                    left += 1

                else:  # current_sum > target
                    # Need smaller sum, move right pointer left
                    # This eliminates all pairs with arr[right]
                    right -= 1

            # No valid pair found
            return []


        # Example Usage:
        arr = [2, 7, 11, 15]
        target = 9
        result = two_pointers_opposite(arr, target)
        print(result)  # Output: [0, 1] because arr[0] + arr[1] = 9
        ```

        **Key Points:**
        - Condition: `left < right` (not <=, since we need two different elements)
        - Movement: Greedy decision based on comparison
        - Termination: Pointers meet or cross

    === "Template 2: Same Direction"

        **Use Case:** In-place modifications, partitioning, removing elements

        **Pattern:** Both pointers start at beginning, one explores ahead

        ```python
        def remove_duplicates(arr):
            """
            Two pointers moving in same direction.

            Perfect for: In-place array modifications

            Concept:
            - 'write' pointer: Next position to write unique element
            - 'read' pointer: Explores array finding unique elements

            Time: O(n) - Single pass
            Space: O(1) - In-place modification

            Example: Remove duplicates from sorted array
            """
            if not arr:
                return 0

            # write pointer: tracks where to place next unique element
            write = 1

            # read pointer: scans through array
            for read in range(1, len(arr)):
                # Found new unique element?
                if arr[read] != arr[read - 1]:
                    arr[write] = arr[read]  # Write it
                    write += 1              # Move write position

            return write  # New length (write pointer position)


        # Example Usage:
        arr = [1, 1, 2, 2, 2, 3, 4, 4]
        new_length = remove_duplicates(arr)
        print(arr[:new_length])  # Output: [1, 2, 3, 4]
        ```

        **Key Points:**
        - Fast pointer (read): explores array
        - Slow pointer (write): tracks modification position
        - Invariant: Everything before write pointer is valid result

    === "Template 3: Palindrome Check"

        **Use Case:** Symmetric structure verification

        **Pattern:** Compare elements from both ends moving inward

        ```python
        def is_palindrome(s):
            """
            Two pointers for palindrome verification.

            Perfect for: Checking symmetric structures

            Concept: If string is palindrome, characters at opposite
                    positions must match as we move inward

            Time: O(n) - Visit each character once
            Space: O(1) - No extra storage

            Example: "racecar" → True, "hello" → False
            """
            # Start from both ends
            left, right = 0, len(s) - 1

            while left < right:
                # Skip non-alphanumeric characters
                if not s[left].isalnum():
                    left += 1
                    continue
                if not s[right].isalnum():
                    right -= 1
                    continue

                # Compare characters (case-insensitive)
                if s[left].lower() != s[right].lower():
                    return False  # Mismatch found

                # Move toward center
                left += 1
                right -= 1

            return True  # All characters matched


        # Example Usage:
        print(is_palindrome("A man, a plan, a canal: Panama"))  # True
        print(is_palindrome("race a car"))  # False
        ```

        **Key Points:**
        - Characters at distance d from ends must match
        - Early termination on mismatch
        - Can handle special character filtering

    === "Visual Walkthrough"

        **Problem:** Find two numbers that sum to 9 in [2, 7, 11, 15]

        ```
        Initial State:
        [2,  7,  11,  15]
         ↑            ↑
        left        right
        Sum: 2 + 15 = 17 > 9 (too large)

        Step 1: Move right pointer left
        [2,  7,  11,  15]
         ↑        ↑
        left    right
        Sum: 2 + 11 = 13 > 9 (still too large)

        Step 2: Move right pointer left again
        [2,  7,  11,  15]
         ↑    ↑
        left right
        Sum: 2 + 7 = 9 ✓ (found it!)

        Result: indices [0, 1]
        ```

        **Why This Works:**

        After step 1, we KNOW that pairing 15 with ANY element won't work:
        - 2 + 15 = 17 (too large)
        - 7 + 15 = 22 (even larger)
        - 11 + 15 = 26 (way too large)

        So we eliminate 15 entirely! This is why we get O(n) instead of O(n²).

        ---

        **Palindrome Check:** "racecar"

        ```
        Step 1: Compare outer characters
        r a c e c a r
        ↑           ↑
        r == r ✓

        Step 2: Move inward
        r a c e c a r
          ↑       ↑
        a == a ✓

        Step 3: Move inward
        r a c e c a r
            ↑   ↑
        c == c ✓

        Step 4: Move inward
        r a c e c a r
              ↑
        Pointers met, it's a palindrome!
        ```

=== "Practice Problems"

    ## Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic mechanics of two pointers.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Valid Palindrome | Opposite ends | Basic comparison, filtering | [LeetCode 125](https://leetcode.com/problems/valid-palindrome/) |
    | Remove Duplicates | Same direction | In-place modification | [LeetCode 26](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) |
    | Move Zeroes | Same direction | Partitioning | [LeetCode 283](https://leetcode.com/problems/move-zeroes/) |
    | Reverse String | Opposite ends | Swapping | [LeetCode 344](https://leetcode.com/problems/reverse-string/) |
    | Squares of Sorted Array | Opposite ends | Merge from ends | [LeetCode 977](https://leetcode.com/problems/squares-of-a-sorted-array/) |

    **Goal:** Solve all 5 problems. Understand when to move left vs right pointer.

    ---

    ### Phase 2: Application (Medium)
    Apply pattern to more complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Two Sum II | Opposite ends | Classic sorted pair | [LeetCode 167](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) |
    | 3Sum | Opposite ends | Fix one + two pointers | [LeetCode 15](https://leetcode.com/problems/3sum/) |
    | Container With Most Water | Opposite ends | Greedy decisions | [LeetCode 11](https://leetcode.com/problems/container-with-most-water/) |
    | Sort Colors | Three pointers | Dutch National Flag | [LeetCode 75](https://leetcode.com/problems/sort-colors/) |
    | Remove Nth Node From End | Same direction | Offset pointers | [LeetCode 19](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) |

    **Goal:** Solve 3 out of 5. Learn to combine two pointers with other techniques.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex variations and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Trapping Rain Water | Opposite ends | Complex logic per move | [LeetCode 42](https://leetcode.com/problems/trapping-rain-water/) |
    | 4Sum | Multiple pointers | Nested two pointers | [LeetCode 18](https://leetcode.com/problems/4sum/) |
    | Minimum Window Substring | Same direction + hash | Sliding window variant | [LeetCode 76](https://leetcode.com/problems/minimum-window-substring/) |
    | Longest Mountain | Same direction | Pattern recognition | [LeetCode 845](https://leetcode.com/problems/longest-mountain-in-array/) |

    **Goal:** Solve 2 out of 4. Master edge cases and optimizations.

    ---

    ## Practice Strategy

    1. **Start with Easy:** Build confidence with straightforward problems
    2. **Identify the Variant:** Opposite ends or same direction?
    3. **Draw It Out:** Visualize pointer movement on paper
    4. **Code Without Looking:** Try implementing from memory
    5. **Time Yourself:** Aim for <15 minutes per easy, <30 for medium
    6. **Review After 24 Hours:** Spaced repetition solidifies learning

    ---

    ## Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Using `<=` instead of `<` | Want to check all elements | Remember: need two DIFFERENT elements |
    | Not handling duplicates | Forget array might have repeats | Add logic to skip duplicates |
    | Infinite loop | Forgetting to move pointer | Always ensure pointer moves in each iteration |
    | Off-by-one errors | Confusion with indices | Test with small example (3 elements) |
    | Not checking bounds | Assuming array not empty | Add `if not arr` check |

---

### Sliding Window

=== "Understanding the Pattern"

    ## What is Sliding Window?

    Picture yourself on a train looking out the window as it moves along a scenic route. As the train glides forward, you see new scenery entering your view on the right while the old scenery exits on the left. You don't need to re-examine everything each time—you just focus on what's new entering and what's leaving.

    This is exactly how the **Sliding Window** pattern works! Instead of recalculating everything from scratch as we examine different subarrays, we maintain a "window" that slides across the data, efficiently updating our result by:
    - Adding the new element entering on the right
    - Removing the old element leaving on the left
    - Updating our answer based on just these two changes

    This simple insight transforms an O(n×k) brute force solution into an elegant O(n) algorithm!

    ---

    ## How It Works

    There are two main flavors of sliding window:

    **1. Fixed-Size Window:**
    ```
    Array: [2, 1, 5, 1, 3, 2]  (Find max sum of size k=3)

    Step 1: [2, 1, 5] → sum = 8
             ←─────→

    Step 2: [2, 1, 5, 1] → Remove 2, Add 1 → sum = 7
                ←─────→

    Step 3: [2, 1, 5, 1, 3] → Remove 1, Add 3 → sum = 9
                   ←─────→
    ```

    **2. Variable-Size Window:**
    ```
    String: "abcabcbb"  (Longest substring without repeating)

    Expand: a b c → valid, keep expanding
            ←───→

    Contract: a b c a → 'a' repeats! Shrink from left
              ←─────→

    Continue: b c a b → valid window
                ←───→
    ```

    ---

    ## Key Intuition

    **The Aha Moment:** Why recalculate from scratch when you can just update the difference?

    Think of calculating average temperature over 7 days:
    - **Brute Force:** Every day, sum up the last 7 temperatures → 7 additions per day
    - **Sliding Window:** Subtract yesterday's leaving temperature, add today's new one → 2 operations per day!

    For an array of length n with window size k:
    - **Without sliding window:** n windows × k operations = O(n×k)
    - **With sliding window:** n windows × 2 operations = O(n)

    This is the power of reusing previous computations!

    ---

    ## Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n) | Each element enters and exits window exactly once |
    | **Space** | O(1) to O(k) | O(1) for simple tracking, O(k) with hash map for constraints |
    | **Improvement** | From O(n×k) | Eliminates redundant recalculations |

    **Why so efficient?** Every element is processed at most twice—once when it enters the window (right pointer) and once when it leaves (left pointer). Total operations: 2n = O(n).

=== "When to Use This Pattern"

    ## Perfect For

    | Scenario | Why Sliding Window Works | Example |
    |----------|------------------------|---------|
    | **Fixed-size subarray** | Window size constant, slide one step at a time | Max sum of K consecutive elements |
    | **Contiguous sequences** | Keywords: "subarray", "substring", "consecutive" | Longest substring without repeating chars |
    | **Min/max with constraints** | Track state as window moves | Smallest subarray with sum ≥ target |
    | **Anagram/permutation matching** | Fixed pattern, sliding character frequency | Find all anagrams in string |
    | **Range queries with updates** | Window bounds change dynamically | Longest subarray with at most K distinct |

    **Red Flags That Suggest Sliding Window:**
    - "Contiguous" or "consecutive" in problem description
    - "Subarray" or "substring" (not subsequence!)
    - "In a window of size K"
    - "Longest/shortest/maximum/minimum" with linear scan possible
    - Can maintain window state incrementally

    ---

    ## When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Non-contiguous elements** | Sliding window only works on adjacent elements | DP for subsequences |
    | **Need to check all subarrays** | Some problems require examining every combination | Nested loops O(n²) or better algorithm |
    | **Window can "jump"** | Can't maintain state if skipping elements | Different approach needed |
    | **Complex interdependencies** | Window state can't be updated incrementally | Try two pointers or DP |
    | **Very small input (n < 20)** | Optimization not worth the code complexity | Simple brute force fine |

    ---

    ## Decision Flowchart

    ```
    Is it a subarray/substring problem?
    ├─ Yes → Is the size fixed or variable?
    │         ├─ Fixed size K → USE FIXED WINDOW ✓
    │         └─ Variable size → Can you maintain state incrementally?
    │                            ├─ Yes → USE VARIABLE WINDOW ✓
    │                            └─ No → Try two pointers or DP
    └─ No → Not a sliding window problem
    ```

=== "Implementation Templates"

    === "Template 1: Fixed-Size Window"

        **Use Case:** Max/min/average of all K-sized subarrays

        **Pattern:** Window size stays constant, slide one element at a time

        ```python
        def fixed_window(arr, k):
            """
            Fixed-size sliding window.

            Perfect for: All subarrays of exact size K

            Concept: Calculate first window, then slide by:
            - Removing element leaving on left
            - Adding element entering on right

            Time: O(n) - Each element processed once
            Space: O(1) - Only tracking window state

            Example: Max sum of any K consecutive elements
            """
            if len(arr) < k:
                return None

            # Step 1: Calculate first window sum
            window_sum = sum(arr[:k])
            max_sum = window_sum

            # Step 2: Slide the window
            for i in range(k, len(arr)):
                # Remove leftmost element leaving window
                window_sum -= arr[i - k]
                # Add rightmost element entering window
                window_sum += arr[i]
                # Update maximum
                max_sum = max(max_sum, window_sum)

            return max_sum


        # Example Usage:
        arr = [2, 1, 5, 1, 3, 2]
        k = 3
        result = fixed_window(arr, k)
        print(result)  # Output: 9 (from subarray [5,1,3])
        ```

        **Key Points:**
        - Window size is always exactly K
        - Subtract arr[i-k], add arr[i] in each iteration
        - One pass through array after initial window

    === "Template 2: Variable-Size Window"

        **Use Case:** Longest/shortest subarray satisfying constraint

        **Pattern:** Expand right, contract left when constraint violated

        ```python
        def variable_window(s):
            """
            Variable-size sliding window.

            Perfect for: Longest/shortest with constraints

            Concept:
            - Expand: Move right pointer, add to window
            - Contract: When invalid, move left pointer until valid
            - Update: Track best window at each step

            Time: O(n) - Each element added/removed once
            Space: O(k) - Hash map for constraint tracking

            Example: Longest substring without repeating characters
            """
            left = 0
            char_set = set()  # Track characters in window
            max_length = 0

            for right in range(len(s)):
                # Expand: Add right character to window
                # If character already exists, contract from left
                while s[right] in char_set:
                    char_set.remove(s[left])
                    left += 1

                # Now window is valid, add character
                char_set.add(s[right])

                # Update result with current window size
                max_length = max(max_length, right - left + 1)

            return max_length


        # Example Usage:
        s = "abcabcbb"
        result = variable_window(s)
        print(result)  # Output: 3 (substring "abc")
        ```

        **Key Points:**
        - Window size changes dynamically
        - Right pointer always moves forward
        - Left pointer moves to maintain validity
        - Each element enters/exits at most once

    === "Template 3: With Character Frequency"

        **Use Case:** Anagram matching, permutation finding

        **Pattern:** Track character counts in window

        ```python
        def window_with_frequency(s, pattern):
            """
            Sliding window with character frequency tracking.

            Perfect for: Anagram problems, permutation matching

            Concept: Track frequency of characters in window,
                    compare with target frequency

            Time: O(n) - Single pass through string
            Space: O(k) - Frequency map of pattern

            Example: Find all anagrams of pattern in string
            """
            from collections import Counter

            if len(pattern) > len(s):
                return []

            pattern_freq = Counter(pattern)
            window_freq = Counter()
            result = []
            k = len(pattern)

            for i in range(len(s)):
                # Add character entering window
                window_freq[s[i]] += 1

                # Remove character leaving window (if window exceeds size)
                if i >= k:
                    if window_freq[s[i - k]] == 1:
                        del window_freq[s[i - k]]
                    else:
                        window_freq[s[i - k]] -= 1

                # Check if window is an anagram
                if i >= k - 1 and window_freq == pattern_freq:
                    result.append(i - k + 1)  # Starting index

            return result


        # Example Usage:
        s = "cbaebabacd"
        pattern = "abc"
        result = window_with_frequency(s, pattern)
        print(result)  # Output: [0, 6] (anagrams at indices 0 and 6)
        ```

        **Key Points:**
        - Use Counter or dict for frequency tracking
        - Compare frequencies, not actual strings
        - Efficient O(1) comparison if alphabet size small

    === "Visual Walkthrough"

        **Problem:** Max sum subarray of size k=3 in [2, 1, 5, 1, 3, 2]

        ```
        Initial Window (i=0 to i=2):
        [2, 1, 5, 1, 3, 2]
         ←─────→
        Sum = 2 + 1 + 5 = 8
        Max = 8

        Slide Right (remove 2, add 1):
        [2, 1, 5, 1, 3, 2]
            ←─────→
        Sum = 8 - 2 + 1 = 7
        Max = 8 (no update)

        Slide Right (remove 1, add 3):
        [2, 1, 5, 1, 3, 2]
               ←─────→
        Sum = 7 - 1 + 3 = 9
        Max = 9 ✓ (new maximum!)

        Slide Right (remove 5, add 2):
        [2, 1, 5, 1, 3, 2]
                  ←─────→
        Sum = 9 - 5 + 2 = 6
        Max = 9 (no update)

        Result: 9
        ```

        **Why This is O(n):**
        - Calculated first window: 3 operations
        - Each slide: 2 operations (subtract + add)
        - Total: 3 + 2×(n-k) = O(n)

        Compare to brute force: Sum each window from scratch = k×(n-k+1) = O(n×k)

        ---

        **Problem:** Longest substring without repeating in "abcabcbb"

        ```
        Step 1: Expand right
        "abcabcbb"
         ←──→
        Window: "abc" → valid (all unique)
        Max length = 3

        Step 2: Expand right, collision!
        "abcabcbb"
         ←────→
        Window: "abca" → invalid ('a' repeats)
        Contract: Remove 'a' from left

        Step 3: After contraction
        "abcabcbb"
          ←───→
        Window: "bca" → valid
        Max length = 3 (no improvement)

        Continue sliding...
        "abcabcbb"
              ←──→
        Final window examples: "abc", "cab", "b"
        Result: 3
        ```

=== "Practice Problems"

    ## Learning Path

    ### Phase 1: Foundation (Easy)
    Master fixed and simple variable windows.

    | Problem | Window Type | Key Learning | Link |
    |---------|------------|--------------|------|
    | Maximum Average Subarray I | Fixed | Basic fixed window | [LeetCode 643](https://leetcode.com/problems/maximum-average-subarray-i/) |
    | Contains Duplicate II | Fixed | Fixed window with set | [LeetCode 219](https://leetcode.com/problems/contains-duplicate-ii/) |
    | Longest Nice Substring | Variable | Basic variable window | [LeetCode 1763](https://leetcode.com/problems/longest-nice-substring/) |
    | Minimum Difference | Fixed | Fixed window with sort | [LeetCode 1984](https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/) |
    | Max Consecutive Ones III | Variable | Variable with counter | [LeetCode 1004](https://leetcode.com/problems/max-consecutive-ones-iii/) |

    **Goal:** Solve all 5 problems. Recognize when to expand vs contract window.

    ---

    ### Phase 2: Application (Medium)
    Apply to string matching and constraints.

    | Problem | Window Type | Challenge | Link |
    |---------|------------|-----------|------|
    | Longest Substring Without Repeating | Variable | Set-based constraint | [LeetCode 3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) |
    | Longest Repeating Character Replacement | Variable | Frequency + counter | [LeetCode 424](https://leetcode.com/problems/longest-repeating-character-replacement/) |
    | Permutation in String | Fixed | Anagram matching | [LeetCode 567](https://leetcode.com/problems/permutation-in-string/) |
    | Find All Anagrams | Fixed | Multiple anagram matches | [LeetCode 438](https://leetcode.com/problems/find-all-anagrams-in-a-string/) |
    | Fruit Into Baskets | Variable | At most K distinct | [LeetCode 904](https://leetcode.com/problems/fruit-into-baskets/) |

    **Goal:** Solve 3 out of 5. Master frequency tracking and constraint management.

    ---

    ### Phase 3: Mastery (Hard)
    Complex windows with multiple constraints.

    | Problem | Window Type | Advanced Concept | Link |
    |---------|------------|------------------|------|
    | Minimum Window Substring | Variable | Min window with full coverage | [LeetCode 76](https://leetcode.com/problems/minimum-window-substring/) |
    | Sliding Window Maximum | Fixed | Deque for max tracking | [LeetCode 239](https://leetcode.com/problems/sliding-window-maximum/) |
    | Substring with Concatenation | Fixed | Multiple word matching | [LeetCode 30](https://leetcode.com/problems/substring-with-concatenation-of-all-words/) |
    | Longest Substring At Most K Distinct | Variable | K-constraint optimization | [LeetCode 340](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/) |

    **Goal:** Solve 2 out of 4. Handle complex state management.

    ---

    ## Practice Strategy

    1. **Identify Window Type:** Fixed or variable size?
    2. **Define Validity:** What makes a window valid/invalid?
    3. **Track State:** What data structure tracks window contents?
    4. **Expansion Logic:** When do you move right pointer?
    5. **Contraction Logic:** When do you move left pointer?
    6. **Update Answer:** When do you record/update the result?

    ---

    ## Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Forgetting to shrink window | Focus only on expansion | Always handle both expand AND contract |
    | Off-by-one in window size | Confusion with indices | Use `right - left + 1` for size |
    | Not removing from state | Only track additions | Remove leaving elements from hash map/set |
    | Wrong inequality in while loop | Logic error in validity check | Clearly define when window is invalid |
    | Updating answer at wrong time | Unclear about when window is valid | Update only when window satisfies constraint |

---

### Binary Search

=== "Understanding the Pattern"

    ## 📖 What is Binary Search?

    Imagine you're playing a guessing game with a friend who picked a number between 1 and 100. You could start guessing from 1, 2, 3... but that's slow! Instead, you guess 50. Too high? Now you know it's between 1-49. Guess 25. Too low? It's between 26-49. Each guess eliminates half the remaining possibilities!

    This is **Binary Search**: repeatedly dividing a sorted search space in half, discarding the half that cannot contain the answer. It's like finding a word in a dictionary—you don't start at page 1, you open to the middle and decide which half to search.

    The beauty? While a linear search might need 100 comparisons, binary search needs at most 7 (since 2^7 = 128 > 100)!

    ---

    ## 🔧 How It Works

    Binary Search requires a sorted dataset and works by maintaining a search range:

    **Basic Mechanism:**
    ```
    Sorted Array: [1, 3, 5, 7, 9, 11, 13, 15]  (Find 7)

    Step 1: Check middle
    [1, 3, 5, 7, 9, 11, 13, 15]
             ↑
            mid=7? YES! Found it!

    If target was 11:
    Step 1: mid=7 < 11 → search right half
    [1, 3, 5, 7 | 9, 11, 13, 15]
                   ↑
                  mid=11? YES!
    ```

    **Three Key Decisions:**
    1. **Middle element = target**: Found it!
    2. **Middle element < target**: Search right half (left = mid + 1)
    3. **Middle element > target**: Search left half (right = mid - 1)

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Why is this so much faster? Because we're not just eliminating one element at a time—we're eliminating half the remaining elements!

    Think of it like a phone book:
    - **Linear search (bad):** Check every name from A to Z
    - **Binary search (smart):** Open to middle. Looking for "Miller"? Middle shows "Johnson"? M > J, so ignore entire first half!

    This exponential elimination is why binary search is O(log n):
    - 1,000 elements → max 10 comparisons
    - 1,000,000 elements → max 20 comparisons
    - 1,000,000,000 elements → max 30 comparisons

    **The sorted requirement:** Binary search only works because sorting guarantees all smaller elements are on the left, all larger on the right. This property lets us confidently discard half the data.

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(log n) | Search space halves with each comparison |
    | **Space** | O(1) | Iterative uses constant space |
    | **Space (recursive)** | O(log n) | Recursion stack depth equals number of halvings |
    | **Improvement** | From O(n) | Reduces linear scan by exponential factor |

    **Why O(log n)?** Because after k comparisons, search space = n / 2^k. When this equals 1, we're done: n / 2^k = 1 → k = log₂(n).

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Binary Search Works | Example |
    |----------|----------------------|---------|
    | **Sorted array search** | Can eliminate half based on comparison | Classic binary search |
    | **Find boundaries** | Locate first/last occurrence efficiently | First/last position of element |
    | **Rotated sorted array** | Modified binary search handles pivot | Search in rotated array |
    | **Answer space search** | When answer range is sortable | Koko eating bananas |
    | **Min/max optimization** | Search for threshold value | Minimum days to make m bouquets |

    **Red Flags That Suggest Binary Search:**
    - Data is sorted or partially sorted
    - "Find element in sorted array"
    - "Minimum/maximum value that satisfies condition"
    - Can eliminate half the search space with each check
    - Search space can be conceptually sorted (not just arrays!)

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Unsorted data** | Can't make elimination decisions | Sort first O(n log n) or linear search O(n) |
    | **Need all occurrences** | Binary search finds one, finding all requires extra work | Depends: if many matches, linear might be better |
    | **Data changes frequently** | Maintaining sorted property expensive | Hash table for O(1) lookup |
    | **Very small datasets (n < 20)** | Overhead not worth it | Simple linear scan fine |
    | **Complex unsortable conditions** | Can't define meaningful ordering | Try different approach |

    ---

    ## 🎯 Decision Flowchart

    ```
    Is the data sorted (or sortable)?
    ├─ Yes → Searching for a specific value?
    │         ├─ Yes → USE BINARY SEARCH (Classic) ✓
    │         └─ No → Searching for boundary/threshold?
    │                  ├─ Yes → USE BINARY SEARCH (Answer Space) ✓
    │                  └─ No → Consider other patterns
    └─ No → Can you sort it?
              ├─ Yes, and you'll search multiple times → Sort + Binary Search ✓
              └─ No, or only searching once → Hash Map or Linear Search
    ```

=== "Implementation Templates"

    === "Template 1: Classic Binary Search"

        **Use Case:** Find exact value in sorted array

        **Pattern:** Converge left and right pointers until match found or exhausted

        ```python
        def binary_search(arr, target):
            """
            Classic binary search for exact match.

            Perfect for: Finding specific element in sorted array

            Time: O(log n) - Halve search space each iteration
            Space: O(1) - Only using pointer variables

            Example: Search for 7 in [1, 3, 5, 7, 9, 11, 13]
            """
            left, right = 0, len(arr) - 1

            while left <= right:
                # Calculate middle (avoid integer overflow)
                mid = left + (right - left) // 2

                if arr[mid] == target:
                    # Found the target!
                    return mid

                elif arr[mid] < target:
                    # Target is in right half
                    # Eliminate left half including mid
                    left = mid + 1

                else:  # arr[mid] > target
                    # Target is in left half
                    # Eliminate right half including mid
                    right = mid - 1

            # Search space exhausted, target not found
            return -1


        # Example Usage:
        arr = [1, 3, 5, 7, 9, 11, 13, 15]
        target = 7
        result = binary_search(arr, target)
        print(result)  # Output: 3 (index of 7)
        ```

        **Key Points:**
        - Use `left <= right` (not `<`) to handle single-element range
        - Move `left` or `right` past `mid` since we know `mid` isn't the answer
        - Returns `-1` when element not found

    === "Template 2: Lower Bound (First Occurrence)"

        **Use Case:** Find first position where element >= target

        **Pattern:** Shrink right boundary while preserving potential answer

        ```python
        def lower_bound(arr, target):
            """
            Find first position where arr[i] >= target.

            Perfect for: Finding insert position, first occurrence, boundaries

            Concept: Keep moving right boundary left when we find candidates,
                    ensuring we find the FIRST (leftmost) valid position

            Time: O(log n) - Binary search
            Space: O(1) - Constant space

            Example: Find first occurrence of 5 in [1, 2, 5, 5, 5, 7, 9]
            """
            left, right = 0, len(arr)  # Note: right = len(arr), not len(arr)-1

            while left < right:  # Note: < not <=
                mid = left + (right - left) // 2

                if arr[mid] < target:
                    # Need larger values, eliminate left half
                    left = mid + 1
                else:  # arr[mid] >= target
                    # This could be the answer, but maybe there's something
                    # earlier, so keep searching left
                    right = mid  # Note: = mid, not mid - 1

            # left == right, pointing to first position >= target
            return left


        # Example Usage:
        arr = [1, 2, 5, 5, 5, 7, 9]
        target = 5
        result = lower_bound(arr, target)
        print(result)  # Output: 2 (first index where element >= 5)
        ```

        **Key Points:**
        - Use `left < right` (not `<=`)
        - Initialize `right = len(arr)` (not `len(arr) - 1`)
        - Set `right = mid` (not `mid - 1`) to preserve potential answer
        - Returns valid insert position even if target not found

    === "Template 3: Answer Space Search"

        **Use Case:** Find minimum/maximum value satisfying a condition

        **Pattern:** Binary search on the answer range, not the input array

        ```python
        def search_answer_space(min_val, max_val, is_valid_func):
            """
            Binary search on answer space.

            Perfect for: "Minimum capacity", "maximum minimum distance" problems

            Concept: Instead of searching in an array, search the range of
                    possible answers. Check if each candidate answer is valid.

            Time: O(log(max-min) × validation_cost)
            Space: O(1)

            Example: Minimum capacity to ship packages in D days
            """
            left, right = min_val, max_val
            result = -1

            while left <= right:
                mid = left + (right - left) // 2

                # Check if 'mid' is a valid answer
                # (e.g., can we ship all packages with capacity 'mid'?)
                if is_valid_func(mid):
                    # mid works! But maybe we can do better (find smaller)
                    result = mid
                    right = mid - 1  # Try smaller values
                else:
                    # mid doesn't work, need larger value
                    left = mid + 1

            return result


        # Example Usage: Koko Eating Bananas
        def min_eating_speed(piles, h):
            """
            Find minimum eating speed to finish all banana piles in h hours.
            """
            def can_finish(speed):
                """Check if eating at 'speed' bananas/hour finishes in time"""
                import math
                hours_needed = sum(math.ceil(pile / speed) for pile in piles)
                return hours_needed <= h

            # Answer must be between 1 and max(piles)
            return search_answer_space(1, max(piles), can_finish)


        piles = [3, 6, 7, 11]
        h = 8
        result = min_eating_speed(piles, h)
        print(result)  # Output: 4 (eating 4 bananas/hour works)
        ```

        **Key Points:**
        - Search the range of possible answers, not input array
        - Define what makes an answer "valid"
        - For "minimum": if valid, try smaller (right = mid - 1)
        - For "maximum": if valid, try larger (left = mid + 1)

    === "Visual Walkthrough"

        **Problem:** Find 11 in [1, 3, 5, 7, 9, 11, 13, 15]

        ```
        Initial State:
        [1,  3,  5,  7,  9,  11,  13,  15]
         ↑               ↑                ↑
        left            mid             right
        mid = 7, target = 11
        7 < 11 → search right half

        Step 1: Eliminate left half
        [1,  3,  5,  7 | 9,  11,  13,  15]
                         ↑   ↑         ↑
                        left mid     right
        mid = 11, target = 11
        Found it! Return index 5

        Total comparisons: 2 (vs 6 for linear search)
        ```

        **Why This Works:**

        Each comparison eliminates half:
        - Step 0: Search space = 8 elements [indices 0-7]
        - Step 1: Search space = 4 elements [indices 4-7]
        - Step 2: Found! (or would continue halving)

        Maximum comparisons for n elements = ⌈log₂(n)⌉ + 1

        ---

        **Problem:** Lower bound of 5 in [1, 2, 5, 5, 5, 7, 9]

        ```
        Finding FIRST occurrence of 5:

        Step 1: mid = 3 (value=5)
        [1,  2,  5,  5,  5,  7,  9]
         ↑           ↑           ↑
        left        mid        right
        arr[mid] >= 5 → could be answer, search left
        right = mid (keep this as potential answer)

        Step 2: mid = 1 (value=2)
        [1,  2,  5,  5,  5,  7,  9]
         ↑   ↑   ↑
        left mid right
        arr[mid] < 5 → need larger, search right
        left = mid + 1

        Step 3: left = right = 2
        [1,  2,  5,  5,  5,  7,  9]
                 ↑
            left/right
        Exit loop, return 2 (first index of 5)
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic mechanics of binary search.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Binary Search | Classic | Basic template | [LeetCode 704](https://leetcode.com/problems/binary-search/) |
    | First Bad Version | Lower bound | Finding boundary | [LeetCode 278](https://leetcode.com/problems/first-bad-version/) |
    | Sqrt(x) | Answer space | Integer square root | [LeetCode 69](https://leetcode.com/problems/sqrtx/) |
    | Search Insert Position | Lower bound | Insert position | [LeetCode 35](https://leetcode.com/problems/search-insert-position/) |
    | Valid Perfect Square | Answer space | Perfect square check | [LeetCode 367](https://leetcode.com/problems/valid-perfect-square/) |

    **Goal:** Solve all 5 problems. Understand when to use left < right vs left <= right.

    ---

    ### Phase 2: Application (Medium)
    Apply pattern to more complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Find First and Last Position | Lower/upper bound | Find range | [LeetCode 34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) |
    | Search in Rotated Sorted Array | Modified binary | Handle rotation | [LeetCode 33](https://leetcode.com/problems/search-in-rotated-sorted-array/) |
    | Find Peak Element | Modified binary | Local maximum | [LeetCode 162](https://leetcode.com/problems/find-peak-element/) |
    | Koko Eating Bananas | Answer space | Minimize maximum | [LeetCode 875](https://leetcode.com/problems/koko-eating-bananas/) |
    | Capacity To Ship Packages | Answer space | Binary search on answer | [LeetCode 1011](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) |

    **Goal:** Solve 3 out of 5. Learn to recognize answer space search.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex variations and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Median of Two Sorted Arrays | Advanced binary | Binary search on partition | [LeetCode 4](https://leetcode.com/problems/median-of-two-sorted-arrays/) |
    | Split Array Largest Sum | Answer space | Complex validation | [LeetCode 410](https://leetcode.com/problems/split-array-largest-sum/) |
    | Minimum Number of Days to Make m Bouquets | Answer space | Constraint satisfaction | [LeetCode 1482](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/) |
    | Find Minimum in Rotated Sorted Array II | Modified binary | With duplicates | [LeetCode 154](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/) |

    **Goal:** Solve 2 out of 4. Master complex answer space problems.

    ---

    ## 🎯 Practice Strategy

    1. **Understand the Template:** Start with classic binary search, then variants
    2. **Identify the Search Space:** Array elements or range of answers?
    3. **Define the Condition:** What makes left/right boundaries move?
    4. **Draw It Out:** Visualize the array and pointer movements
    5. **Edge Cases:** Single element, target at boundaries, not found
    6. **Time Yourself:** Aim for <10 minutes per easy, <25 for medium

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Integer overflow with `(left + right) / 2` | Addition can overflow | Use `left + (right - left) // 2` |
    | Wrong loop condition (`<` vs `<=`) | Different templates need different conditions | Classic: `<=`, Lower bound: `<` |
    | Off-by-one in boundary updates | Confusion about inclusive/exclusive | Classic: `mid ± 1`, Lower bound: `mid` |
    | Infinite loop | Boundary not moving forward | Ensure left or right changes each iteration |
    | Not handling empty array | Forgot edge case | Add `if not arr: return -1` |

---

### Cyclic Sort

=== "Understanding the Pattern"

    ## 📖 What is Cyclic Sort?

    Imagine you're organizing a library where books are numbered 1 to 100, and each shelf spot is also numbered 1 to 100. The books are completely scrambled—book 47 is on shelf 12, book 3 is on shelf 89, and so on. Your job: put each book on its matching shelf number.

    Here's the clever insight: instead of searching for book 1, then searching for book 2, you can be smarter! Start at shelf 1. Whatever book is there (say, book 47), take it and swap it with whatever's on shelf 47. Keep doing this—each swap places one book in its correct position. Eventually, book 1 ends up on shelf 1, and you move to shelf 2.

    This is **Cyclic Sort**: when you have numbers in a known range (like 1 to n), use their values as addresses! Each number "knows" where it should go. By repeatedly swapping numbers to their correct positions, you organize everything in a single pass.

    The beautiful part: each element moves to its correct position at most once. Even though it looks like nested operations, it's actually O(n) because each swap puts at least one element in its final place!

    ---

    ## 🔧 How It Works

    Cyclic Sort works by using the number's value to determine its correct index position.

    **Core Concept:**
    ```
    Array with numbers 1 to n:
    - Number 1 belongs at index 0
    - Number 2 belongs at index 1
    - Number n belongs at index n-1

    For each position, keep swapping until correct number arrives
    ```

    **Visual Example:**
    ```
    Array: [3, 1, 5, 4, 2]  (numbers 1-5)
    Goal:  [1, 2, 3, 4, 5]

    i=0: Position 0 should have 1, but has 3
         Put 3 where it belongs (index 2)
         [3, 1, 5, 4, 2]
          ↓     ↓
         [5, 1, 3, 4, 2]

    i=0: Position 0 now has 5, should be at index 4
         [5, 1, 3, 4, 2]
          ↓           ↓
         [2, 1, 3, 4, 5]

    i=0: Position 0 now has 2, should be at index 1
         [2, 1, 3, 4, 5]
          ↓  ↓
         [1, 2, 3, 4, 5]

    i=0: Position 0 has 1 ✓ Move to next
    i=1: Position 1 has 2 ✓ Move to next
    i=2: Position 2 has 3 ✓ Move to next
    ...
    Done!
    ```

    **Why Don't We Get Stuck in Infinite Loops?**

    Each swap places at least one number in its correct final position. Once a number is correct, we never touch it again!

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** When numbers are in range 1 to n, their values ARE their addresses! This direct mapping means we can sort in-place without extra space.

    Think of it like a cloakroom with numbered tickets:
    - Ticket 5 should go in cubby 5
    - Ticket 23 should go in cubby 23
    - If you find ticket 23 in cubby 5, swap it with whatever's in cubby 23

    **Why This Pattern is Powerful:**

    Traditional sorting needs comparisons and can't do better than O(n log n) in general. But when you know the exact range (1 to n), you can exploit the number-to-index mapping to achieve O(n) time with O(1) space—no other sorting algorithm can do this!

    **Finding Anomalies:**

    After cyclic sort, any position where `nums[i] != i+1` reveals an anomaly:
    - Missing number problem: which index doesn't have its correct number?
    - Duplicate problem: which number appears where it shouldn't?
    - Multiple missing: collect all mismatched indices

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n) | Each element swapped to correct position at most once |
    | **Space** | O(1) | In-place swapping, no extra arrays or structures |
    | **Improvement** | From O(n log n) | Beats comparison-based sorting by exploiting range constraint |

    **Why Only O(n) Despite Nested Operations?**

    The outer loop runs n times, but the inner while loop (swapping) doesn't run n times for each position. Across the entire algorithm, there are at most n swaps total because each swap places one element correctly, and we never re-swap correct elements.

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Cyclic Sort Works | Example |
    |----------|---------------------|---------|
    | **Find missing number in [1,n]** | After sorting, first position with wrong value is missing | Missing Number (LC 268) |
    | **Find duplicate in [1,n]** | Two elements try to go to same position | Find Duplicate Number (LC 287) |
    | **All missing numbers** | Collect all indices with wrong values | Find Disappeared Numbers (LC 448) |
    | **First missing positive** | Sort positives, find first gap | First Missing Positive (LC 41) |
    | **Corrupted array (missing + duplicate)** | Sort reveals both anomalies | Set Mismatch (LC 645) |

    **Red Flags That Suggest Cyclic Sort:**
    - Problem mentions "numbers from 1 to n" or "range [0, n]"
    - Array length is n but numbers are in specific range
    - Words like "missing", "duplicate" with range constraint
    - Need O(n) time and O(1) space (cyclic sort's specialty)
    - Array allowed to be modified in-place

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Numbers not in [1,n] range** | No direct index mapping exists | Use hash set for missing/dups O(n) space |
    | **Negative numbers or large gaps** | Can't use value as index | Modified cyclic sort or different approach |
    | **Array must stay unchanged** | Cyclic sort modifies original array | Copy array first, or use marking technique |
    | **Need stable sort** | Cyclic sort doesn't preserve order of equal elements | Use stable sort like merge sort |
    | **Very small arrays (n < 5)** | Overhead not worth optimization | Simple linear scan works fine |

    ---

    ## 🎯 Decision Flowchart

    ```
    Are numbers in range [1,n] or [0,n]?
    ├─ Yes → Can you modify array in-place?
    │         ├─ Yes → Looking for missing/duplicate/first positive?
    │         │         ├─ Yes → USE CYCLIC SORT ✓
    │         │         └─ No → Just need sorted array?
    │         │                 └─ USE CYCLIC SORT ✓
    │         └─ No → Copy array, then use cyclic sort
    └─ No → Numbers have large gaps or negative?
              └─ Use hash set or different pattern
    ```

=== "Implementation Templates"

    === "Template 1: Basic Cyclic Sort"

        **Use Case:** Sort array containing numbers from 1 to n

        **Pattern:** Keep swapping until each number reaches its correct position

        ```python
        def cyclic_sort(nums):
            """
            Sort array with numbers 1 to n using cyclic sort.

            Perfect for: When you need to sort 1-to-n range in O(n) time, O(1) space

            Concept:
            - Number 'k' belongs at index 'k-1'
            - Keep swapping current element to its correct position
            - Only move to next index when current position is correct

            Time: O(n) - Each element swapped to correct position once
            Space: O(1) - In-place sorting

            Example: Sort [3, 1, 5, 4, 2] → [1, 2, 3, 4, 5]
            """
            i = 0
            while i < len(nums):
                # Calculate where nums[i] should be
                correct_idx = nums[i] - 1  # For range 1 to n

                # Is nums[i] at its correct position?
                if nums[i] != nums[correct_idx]:
                    # No - swap it to correct position
                    # This places one element correctly
                    nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
                    # Don't increment i - check new element at position i
                else:
                    # Yes - move to next position
                    i += 1

            return nums


        # Example Usage:
        arr = [3, 1, 5, 4, 2]
        result = cyclic_sort(arr)
        print(result)  # Output: [1, 2, 3, 4, 5]
        ```

        **Key Points:**
        - Don't increment `i` after swap (need to check new element at position i)
        - Each swap places at least one element correctly
        - Loop ends when all elements in correct positions

    === "Template 2: Find Missing Number"

        **Use Case:** Find the missing number in range [0, n]

        **Pattern:** Sort using cyclic sort, then find first mismatch

        ```python
        def find_missing_number(nums):
            """
            Find missing number in array containing n numbers from range [0, n].

            Perfect for: Missing number problems with range [0, n]

            Concept:
            - For range [0, n], number k belongs at index k
            - After cyclic sort, first index where nums[i] != i is missing
            - Special case: if all match, missing number is n

            Time: O(n) - Sort in O(n), scan in O(n)
            Space: O(1) - In-place modifications

            Example: [4, 0, 3, 1] → missing is 2
            """
            i, n = 0, len(nums)

            # Phase 1: Cyclic sort (for range 0 to n)
            while i < n:
                correct_idx = nums[i]

                # Place nums[i] at its index (if valid and not already there)
                if correct_idx < n and nums[i] != nums[correct_idx]:
                    nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
                else:
                    i += 1

            # Phase 2: Find first mismatch
            for i in range(n):
                if nums[i] != i:
                    return i  # Found missing number

            # All positions correct, missing number is n
            return n


        # Example Usage:
        arr = [4, 0, 3, 1]  # Missing 2
        result = find_missing_number(arr)
        print(result)  # Output: 2

        arr2 = [0, 1, 3]  # Missing 2
        result2 = find_missing_number(arr2)
        print(result2)  # Output: 2

        arr3 = [0, 1, 2]  # Missing 3
        result3 = find_missing_number(arr3)
        print(result3)  # Output: 3
        ```

        **Key Points:**
        - Range [0, n] means number k goes to index k (not k-1)
        - Check `correct_idx < n` to avoid index out of bounds
        - Missing number could be n (largest in range)

    === "Template 3: Find Duplicate"

        **Use Case:** Find duplicate number in array with n+1 elements from [1, n]

        **Pattern:** During cyclic sort, duplicate tries to go to occupied position

        ```python
        def find_duplicate(nums):
            """
            Find duplicate in array with n+1 numbers from range [1, n].

            Perfect for: Finding single duplicate when range is [1, n]

            Concept:
            - Array has n+1 elements but range is only [1, n]
            - By pigeonhole principle, at least one number repeats
            - During cyclic sort, duplicate tries to swap with itself
            - When nums[i] == nums[correct_idx], found duplicate!

            Time: O(n) - At most n swaps
            Space: O(1) - In-place detection

            Example: [1, 3, 4, 2, 2] → duplicate is 2
            """
            i = 0
            while i < len(nums):
                correct_idx = nums[i] - 1  # For range [1, n]

                # Try to place nums[i] at its correct position
                if nums[i] != nums[correct_idx]:
                    # Swap to correct position
                    nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
                    # Don't increment i - check new element
                else:
                    # Either already at correct position, or duplicate found
                    if i != correct_idx:
                        # We're not at the correct index for this value
                        # But this value exists at its correct position
                        # This means it's a duplicate!
                        return nums[i]
                    i += 1

            return -1  # No duplicate found (shouldn't happen if input valid)


        # Example Usage:
        arr = [1, 3, 4, 2, 2]
        result = find_duplicate(arr)
        print(result)  # Output: 2

        arr2 = [3, 1, 3, 4, 2]
        result2 = find_duplicate(arr2)
        print(result2)  # Output: 3
        ```

        **Key Points:**
        - Check if `nums[i] == nums[correct_idx]` but `i != correct_idx`
        - This means duplicate found (two positions claim same number)
        - Works because array has n+1 elements for range [1, n]

    === "Visual Walkthrough"

        **Problem:** Sort [3, 1, 5, 4, 2] using cyclic sort

        ```
        Initial: [3, 1, 5, 4, 2]
                  ↑ i=0

        Step 1: nums[0]=3, should be at index 2
                Check: nums[0] != nums[2]? → 3 != 5? Yes, swap
                [3, 1, 5, 4, 2]
                 ↓     ↓
                [5, 1, 3, 4, 2]
                 ↑ i=0 (don't move i yet!)

        Step 2: nums[0]=5, should be at index 4
                Check: nums[0] != nums[4]? → 5 != 2? Yes, swap
                [5, 1, 3, 4, 2]
                 ↓           ↓
                [2, 1, 3, 4, 5]
                 ↑ i=0 (don't move i yet!)

        Step 3: nums[0]=2, should be at index 1
                Check: nums[0] != nums[1]? → 2 != 1? Yes, swap
                [2, 1, 3, 4, 5]
                 ↓  ↓
                [1, 2, 3, 4, 5]
                 ↑ i=0 (don't move i yet!)

        Step 4: nums[0]=1, should be at index 0
                Check: nums[0] != nums[0]? → No! Already correct
                [1, 2, 3, 4, 5]
                 ↑ i=0 → Move to i=1

        Step 5-8: i=1,2,3,4 all already correct
                  [1, 2, 3, 4, 5] ✓

        Result: Sorted in 3 swaps (O(n) total time)
        ```

        **Why This Works:**

        Each swap places one element in its final position. We made 3 swaps:
        1. Placed 3 at index 2
        2. Placed 5 at index 4
        3. Placed 2 at index 1

        After these swaps, both 1 and 4 were already at correct positions. Total time: O(n) because at most n swaps total.

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic mechanics of cyclic sort.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Missing Number | Template 2 | Basic cyclic sort with range [0,n] | [LeetCode 268](https://leetcode.com/problems/missing-number/) |
    | Find All Disappeared | Collect missing | Identify all mismatched indices | [LeetCode 448](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/) |
    | Move Zeroes | Modified sorting | Partition into two groups | [LeetCode 283](https://leetcode.com/problems/move-zeroes/) |
    | Array Partition | Sort pairs | Cyclic sort then pair logic | [LeetCode 561](https://leetcode.com/problems/array-partition/) |
    | Sort Array by Parity | Partitioning | Even/odd partitioning | [LeetCode 905](https://leetcode.com/problems/sort-array-by-parity/) |

    **Goal:** Solve all 5 problems. Understand the swap-until-correct logic. Recognize range [1,n] or [0,n] patterns.

    ---

    ### Phase 2: Application (Medium)
    Apply pattern to find duplicates and handle missing numbers.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Find Duplicate Number | Template 3 | Detect duplicate during sort | [LeetCode 287](https://leetcode.com/problems/find-the-duplicate-number/) |
    | Find All Duplicates | Collect duplicates | Find all numbers appearing twice | [LeetCode 442](https://leetcode.com/problems/find-all-duplicates-in-an-array/) |
    | Set Mismatch | Both missing and dup | Identify which was removed and added | [LeetCode 645](https://leetcode.com/problems/set-mismatch/) |
    | Missing Positive | Only positive numbers | Filter negatives, then cyclic sort | [LeetCode 41](https://leetcode.com/problems/first-missing-positive/) |
    | Couples Holding Hands | Position swapping | Swap pairs to correct positions | [LeetCode 765](https://leetcode.com/problems/couples-holding-hands/) |

    **Goal:** Master duplicate detection. Handle edge cases (negatives, out of range). Combine filtering with cyclic sort.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex variations and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | First Missing Positive | Hardest cyclic sort | Handle negatives, zeros, out of range | [LeetCode 41](https://leetcode.com/problems/first-missing-positive/) |
    | Find Duplicate Subtrees | Tree + cyclic concept | Apply position tracking to trees | [LeetCode 652](https://leetcode.com/problems/find-duplicate-subtrees/) |
    | Kth Missing Positive | Math + cyclic sort | Calculate missing without full sort | [LeetCode 1539](https://leetcode.com/problems/kth-missing-positive-number/) |
    | Maximum Gap | Bucket + cyclic concept | Use cyclic positioning in buckets | [LeetCode 164](https://leetcode.com/problems/maximum-gap/) |

    **Goal:** Solve problems that combine cyclic sort with other concepts. Optimize for edge cases. Understand when cyclic sort mindset applies beyond arrays.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Template 1:** Code basic cyclic sort from memory 5 times. Understand the "don't increment i after swap" pattern.
    2. **Identify the Variant:** For each problem, determine if it's range [0,n] or [1,n]. Adjust your template accordingly.
    3. **Draw It Out:** For first 3 problems, manually trace the swaps on paper. See how elements reach correct positions.
    4. **Handle Edge Cases:** Missing could be n (largest). Out of range values need bounds checking.
    5. **Time Yourself:** After solving once, re-solve from scratch in under 10 minutes without looking.
    6. **Review After 24 Hours:** Re-code solutions next day. This solidifies the pattern in long-term memory.

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Incrementing i after swap** | Forgetting new element at i also needs checking | Only increment i when nums[i] is correct: `nums[i] == nums[correct_idx]` |
    | **Infinite loop on duplicate** | Swapping duplicate with itself forever | Check if nums[i] == nums[correct_idx] but i != correct_idx (duplicate found) |
    | **Index out of bounds** | Not validating correct_idx < n | Always check: `if correct_idx < n and nums[i] != nums[correct_idx]` |
    | **Wrong range adjustment** | Mixing up [0,n] and [1,n] | [1,n]: correct_idx = nums[i] - 1; [0,n]: correct_idx = nums[i] |
    | **Not handling missing = n** | Forgetting n could be missing in [0,n] | After loop, check all indices. If all match, return n |

---

### Prefix Sum

=== "Understanding the Pattern"

    ## 📖 What is Prefix Sum?

    Imagine you're driving across the country, and your car's odometer shows cumulative miles from the start of your trip. At City A (mile 200), it reads 200. At City B (mile 350), it reads 350. Now, if someone asks "How far is it from City A to City B?", you don't need to drive it again—just subtract: 350 - 200 = 150 miles!

    This is the essence of **Prefix Sum**: precompute cumulative totals at each position so that any range query becomes a simple subtraction. Instead of summing elements [i..j] every time (which takes O(j-i) time), you precompute cumulative sums once, then answer each query in O(1) with: `sum[i..j] = prefix[j] - prefix[i-1]`.

    It's like having a running total at every checkpoint—once you have these totals, finding any segment's sum is instant!

    **Real-World Analogy:**
    - Bank statements showing running balance (not transaction-by-transaction calculation)
    - Speedometer showing cumulative distance (not measuring each mile separately)
    - Points leaderboard showing total scores (not recounting every game)

    ---

    ## 🔧 How It Works

    Prefix Sum builds an auxiliary array where each position stores the sum of all elements up to that index.

    **Core Formula:**
    ```
    prefix[i] = nums[0] + nums[1] + ... + nums[i]

    Range sum from i to j:
    sum[i..j] = prefix[j] - prefix[i-1]

    Why? prefix[j] includes everything up to j
         prefix[i-1] includes everything before i
         Subtracting removes the "before i" part, leaving [i..j]
    ```

    **Visual Example:**
    ```
    Array: [3, 5, 2, 8, 1]
    Index:  0  1  2  3  4

    Build prefix array (with prefix[0] = 0 for convenience):
    prefix[0] = 0
    prefix[1] = 0 + 3 = 3
    prefix[2] = 3 + 5 = 8
    prefix[3] = 8 + 2 = 10
    prefix[4] = 10 + 8 = 18
    prefix[5] = 18 + 1 = 19

    prefix = [0, 3, 8, 10, 18, 19]

    Query: What's the sum from index 1 to 3? (elements [5, 2, 8])
    Answer: prefix[4] - prefix[1] = 18 - 3 = 15 ✓

    Verification: 5 + 2 + 8 = 15 ✓

    Why it works:
    prefix[4] = 3+5+2+8 = 18 (sum up to index 3)
    prefix[1] = 3 (sum up to index 0)
    Difference = 5+2+8 (elements from index 1 to 3)
    ```

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** By storing cumulative sums, we convert repeated O(n) summations into one-time O(n) preprocessing + O(1) queries. The trick is: any range sum is just the difference between two cumulative sums!

    Think of filling a bathtub:
    - After 5 minutes: 20 liters total
    - After 10 minutes: 50 liters total
    - Water added between minute 5 and 10? 50 - 20 = 30 liters

    Prefix sum works the same way: subtract cumulative totals to find segment totals.

    **When This Pays Off:**

    If you need to answer Q queries on an array of length N:
    - Naive approach: O(Q × N) - recalculate sum each query
    - Prefix sum: O(N) preprocessing + O(Q × 1) = O(N + Q)

    For Q > 1, prefix sum is faster! For Q = 100 and N = 1000:
    - Naive: 100,000 operations
    - Prefix: 1,000 + 100 = 1,100 operations (90x faster!)

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Build Time** | O(n) | Single pass to compute cumulative sums |
    | **Query Time** | O(1) | Simple subtraction of two prefix values |
    | **Space** | O(n) | Store one prefix value per element |
    | **Improvement** | From O(n) per query | Each query now O(1) instead of O(n) |

    **Trade-off:** Use extra O(n) space to gain query speed. Worth it when many queries expected.

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Prefix Sum Works | Example |
    |----------|---------------------|---------|
    | **Multiple range sum queries** | Preprocessing once beats recalculating every time | Range Sum Query (LC 303) |
    | **Subarray sum equals K** | Use prefix sum + hash map to find complements | Subarray Sum Equals K (LC 560) |
    | **2D matrix range queries** | Extend to 2D: sum rectangles in O(1) | Range Sum Query 2D (LC 304) |
    | **Equilibrium/pivot index** | Left sum = right sum check | Find Pivot Index (LC 724) |
    | **Cumulative frequency/probability** | Running totals for distributions | Prefix application in statistics |

    **Red Flags That Suggest Prefix Sum:**
    - Problem asks for "sum of subarray [i, j]" multiple times
    - Words like "range query", "subarray sum", "cumulative"
    - Need to find subarrays with specific sum
    - Looking for equilibrium point (left sum = right sum)
    - 2D matrix region/rectangle sums

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Array frequently updated** | Each update requires O(n) rebuild | Segment Tree or Fenwick Tree (O(log n) updates) |
    | **Only one query** | Preprocessing overhead not worth it | Direct summation O(n) is fine |
    | **Need product/max/min** | Prefix sum only works for addition | Sparse table, segment tree for min/max |
    | **Space constrained** | Requires O(n) extra space | May need to use on-the-fly calculation |
    | **Need to modify specific ranges** | Prefix sum is read-only after build | Use difference array or segment tree |

    ---

    ## 🎯 Decision Flowchart

    ```
    Do you need range sums?
    ├─ Yes → How many queries?
    │         ├─ Multiple (Q > 1) → Does array change?
    │         │                     ├─ No → USE PREFIX SUM ✓
    │         │                     └─ Yes → Use Segment/Fenwick Tree
    │         └─ Single → Direct calculation O(n) is fine
    └─ No → Need different aggregation (min/max/product)?
              └─ Use specialized data structures
    ```

=== "Implementation Templates"

    === "Template 1: 1D Prefix Sum"

        **Use Case:** Answer range sum queries on 1D array efficiently

        **Pattern:** Build cumulative sum array, query via subtraction

        ```python
        class PrefixSum:
            """
            Preprocess array for O(1) range sum queries.

            Perfect for: Multiple range queries on static array

            Concept:
            - Build prefix array where prefix[i] = sum of nums[0..i-1]
            - Range sum [left, right] = prefix[right+1] - prefix[left]
            - Extra prefix[0] = 0 simplifies edge cases

            Time: O(n) build, O(1) per query
            Space: O(n)

            Example: Range Sum Query - Immutable (LC 303)
            """
            def __init__(self, nums):
                """
                Build prefix sum array.

                prefix[0] = 0 (nothing before index 0)
                prefix[i] = sum of nums[0] to nums[i-1]
                """
                self.prefix = [0]  # Start with 0 for convenience

                # Build cumulative sum
                for num in nums:
                    self.prefix.append(self.prefix[-1] + num)

                # Result: len(prefix) = len(nums) + 1

            def range_sum(self, left, right):
                """
                Return sum of elements from index left to right (inclusive).

                Formula: prefix[right+1] - prefix[left]

                Why right+1? prefix[right+1] includes nums[right]
                Why prefix[left]? Excludes everything before left
                """
                return self.prefix[right + 1] - self.prefix[left]


        # Example Usage:
        nums = [1, 2, 3, 4, 5]
        ps = PrefixSum(nums)

        # Query sum from index 1 to 3 (elements [2, 3, 4])
        result = ps.range_sum(1, 3)
        print(result)  # Output: 9 (2 + 3 + 4)

        # Query sum from index 0 to 4 (entire array)
        result2 = ps.range_sum(0, 4)
        print(result2)  # Output: 15 (1 + 2 + 3 + 4 + 5)
        ```

        **Key Points:**
        - Extra element `prefix[0] = 0` handles edge cases cleanly
        - Always use `prefix[right+1] - prefix[left]` for inclusive range
        - Length of prefix array is `len(nums) + 1`

    === "Template 2: Prefix Sum + Hash Map"

        **Use Case:** Find subarrays with specific sum (e.g., sum equals K)

        **Pattern:** Track prefix sums in hash map, check for complements

        ```python
        def subarray_sum_equals_k(nums, k):
            """
            Count number of continuous subarrays with sum equal to k.

            Perfect for: Subarray sum problems without explicit range

            Concept:
            - If prefix_sum at j is X, and prefix_sum at i is X-k,
              then subarray [i+1..j] has sum k
            - Use hash map to track: prefix_sum → frequency
            - Check if (current_prefix - k) exists in map

            Time: O(n) - Single pass with hash lookups
            Space: O(n) - Hash map of prefix sums

            Example: Subarray Sum Equals K (LC 560)
            """
            count = 0
            prefix_sum = 0
            # Map: prefix_sum → how many times we've seen it
            sum_freq = {0: 1}  # Base case: sum 0 seen once (empty prefix)

            for num in nums:
                # Update running sum
                prefix_sum += num

                # Check if (prefix_sum - k) exists
                # If yes, we found subarray(s) with sum k
                if prefix_sum - k in sum_freq:
                    count += sum_freq[prefix_sum - k]

                # Record current prefix sum
                sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1

            return count


        # Example Usage:
        nums = [1, 2, 3, 4, 5]
        k = 9
        result = subarray_sum_equals_k(nums, k)
        print(result)  # Output: 2 (subarrays [2,3,4] and [4,5])

        nums2 = [1, 1, 1]
        k2 = 2
        result2 = subarray_sum_equals_k(nums2, k2)
        print(result2)  # Output: 2 (subarrays [1,1] at positions 0-1 and 1-2)
        ```

        **Key Points:**
        - Initialize with `{0: 1}` to handle subarrays starting from index 0
        - Check for complement `prefix_sum - k` before adding current prefix
        - This technique extends to other problems: subarray with sum divisible by k, etc.

    === "Template 3: 2D Prefix Sum"

        **Use Case:** Answer rectangle/region sum queries in 2D matrix

        **Pattern:** Build 2D cumulative sum matrix, use inclusion-exclusion

        ```python
        class PrefixSum2D:
            """
            Preprocess 2D matrix for O(1) rectangle sum queries.

            Perfect for: Range queries on 2D grids/images

            Concept:
            - prefix[i][j] = sum of all elements in rectangle from (0,0) to (i-1,j-1)
            - Use inclusion-exclusion principle for building and querying
            - Add row, add column, subtract overlap (top-left)

            Time: O(m*n) build, O(1) per query
            Space: O(m*n)

            Example: Range Sum Query 2D (LC 304)
            """
            def __init__(self, matrix):
                """
                Build 2D prefix sum matrix.

                prefix[i][j] = sum of rectangle from (0,0) to (i-1,j-1)
                """
                if not matrix or not matrix[0]:
                    self.prefix = []
                    return

                m, n = len(matrix), len(matrix[0])
                # Extra row and column (all zeros) for convenience
                self.prefix = [[0] * (n + 1) for _ in range(m + 1)]

                # Build 2D prefix sum using inclusion-exclusion
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        self.prefix[i][j] = (
                            matrix[i-1][j-1] +          # Current cell
                            self.prefix[i-1][j] +        # Above
                            self.prefix[i][j-1] -        # Left
                            self.prefix[i-1][j-1]        # Subtract overlap (top-left)
                        )

            def region_sum(self, r1, c1, r2, c2):
                """
                Return sum of rectangle from (r1,c1) to (r2,c2) inclusive.

                Use inclusion-exclusion principle:
                Total = large_rect - top_rect - left_rect + overlap
                """
                return (
                    self.prefix[r2+1][c2+1] -     # Include entire rectangle
                    self.prefix[r1][c2+1] -       # Subtract top part
                    self.prefix[r2+1][c1] +       # Subtract left part
                    self.prefix[r1][c1]           # Add back overlap (subtracted twice)
                )


        # Example Usage:
        matrix = [
            [3, 0, 1, 4, 2],
            [5, 6, 3, 2, 1],
            [1, 2, 0, 1, 5],
            [4, 1, 0, 1, 7],
            [1, 0, 3, 0, 5]
        ]

        ps2d = PrefixSum2D(matrix)

        # Query sum of rectangle from (2,1) to (4,3)
        result = ps2d.region_sum(2, 1, 4, 3)
        print(result)  # Output: 8 (2+0+1 + 1+0+1 + 0+3+0)
        ```

        **Key Points:**
        - Extra row/column of zeros simplifies boundary conditions
        - Inclusion-exclusion: add two sides, subtract overlap
        - Coordinates in prefix are offset by 1 from original matrix

    === "Visual Walkthrough"

        **Problem:** Find sum of subarray from index 1 to 3 in [3, 5, 2, 8, 1]

        ```
        Array:    [3,  5,  2,  8,  1]
        Index:     0   1   2   3   4

        Step 1: Build Prefix Array
        ────────────────────────────
        prefix[0] = 0 (base case)
        prefix[1] = 0 + 3 = 3
        prefix[2] = 3 + 5 = 8
        prefix[3] = 8 + 2 = 10
        prefix[4] = 10 + 8 = 18
        prefix[5] = 18 + 1 = 19

        Prefix:   [0,  3,  8, 10, 18, 19]
        Index:     0   1   2   3   4   5

        Step 2: Query sum[1..3] (elements [5, 2, 8])
        ──────────────────────────────────────────────
        Formula: prefix[right+1] - prefix[left]
               = prefix[4] - prefix[1]
               = 18 - 3
               = 15 ✓

        Visual Explanation:
        ──────────────────────
        prefix[4] = sum[0..3] = 3+5+2+8 = 18
        prefix[1] = sum[0..0] = 3

        Subtract to remove index 0:
        18 - 3 = 15 = 5+2+8 ✓

        Think of it as:
        [3, 5, 2, 8, 1]
         └─prefix[1]
         └────prefix[4]────┘
            └──answer──┘
        ```

        **Why This Works:**

        Prefix sum at index i contains all elements from 0 to i. To get sum from index a to b:
        - Take prefix[b] (includes everything from 0 to b)
        - Subtract prefix[a-1] (removes everything from 0 to a-1)
        - What remains? Elements from a to b!

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master basic prefix sum construction and queries.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Range Sum Query - Immutable | 1D basic | Build and query prefix array | [LeetCode 303](https://leetcode.com/problems/range-sum-query-immutable/) |
    | Running Sum of 1D Array | Direct prefix | Build prefix sum (simplified) | [LeetCode 1480](https://leetcode.com/problems/running-sum-of-1d-array/) |
    | Find Pivot Index | Left vs right sum | Use prefix for equilibrium | [LeetCode 724](https://leetcode.com/problems/find-pivot-index/) |
    | Left and Right Sum Differences | Dual prefix | Compute both directions | [LeetCode 2574](https://leetcode.com/problems/left-and-right-sum-differences/) |
    | Find Middle Index | Pivot variation | Another equilibrium problem | [LeetCode 1991](https://leetcode.com/problems/find-the-middle-index-in-array/) |

    **Goal:** Solve all 5 problems. Understand how prefix sum converts O(n) queries to O(1). Practice building prefix arrays with the extra 0 at start.

    ---

    ### Phase 2: Application (Medium)
    Apply prefix sum with hash maps and 2D matrices.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Subarray Sum Equals K | Prefix + hash map | Use complement checking technique | [LeetCode 560](https://leetcode.com/problems/subarray-sum-equals-k/) |
    | Continuous Subarray Sum | Modulo prefix | Track prefix % k in hash map | [LeetCode 523](https://leetcode.com/problems/continuous-subarray-sum/) |
    | Range Sum Query 2D | 2D prefix | Build and query 2D cumulative sum | [LeetCode 304](https://leetcode.com/problems/range-sum-query-2d-immutable/) |
    | Product of Array Except Self | Prefix + suffix | Use two-pass prefix approach | [LeetCode 238](https://leetcode.com/problems/product-of-array-except-self/) |
    | Subarray Sums Divisible by K | Modulo math | Count subarrays with sum % k = 0 | [LeetCode 974](https://leetcode.com/problems/subarray-sums-divisible-by-k/) |

    **Goal:** Master the prefix sum + hash map pattern for subarray problems. Understand 2D extension with inclusion-exclusion principle.

    ---

    ### Phase 3: Mastery (Hard)
    Combine prefix sum with advanced techniques.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Max Sum of 3 Non-Overlapping Subarrays | Multi-pass prefix | Track best subarrays using prefix | [LeetCode 689](https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/) |
    | Subarrays with K Different Integers | Sliding + prefix | Combine multiple techniques | [LeetCode 992](https://leetcode.com/problems/subarrays-with-k-different-integers/) |
    | Count Subarrays with Median K | Prefix + balance | Track count balance around median | [LeetCode 2488](https://leetcode.com/problems/count-subarrays-with-median-k/) |
    | Count Number of Nice Subarrays | Odd/even prefix | Convert to prefix sum problem | [LeetCode 1248](https://leetcode.com/problems/count-number-of-nice-subarrays/) |

    **Goal:** Apply prefix sum in non-obvious scenarios. Combine with other patterns (sliding window, two pointers). Recognize when cumulative tracking helps.

    ---

    ## 🎯 Practice Strategy

    1. **Start with 1D Basic:** Build prefix sum arrays manually for first 3 problems. Trace through queries by hand.
    2. **Master the Formula:** Memorize `sum[i..j] = prefix[j+1] - prefix[i]`. Practice off-by-one errors until intuitive.
    3. **Prefix + Hash Map:** For subarray sum problems, draw the prefix sum array and trace hash map lookups.
    4. **2D Visualization:** For 2D problems, draw small matrices and manually compute prefix values using inclusion-exclusion.
    5. **Time Yourself:** After solving once, re-solve from scratch in under 10 minutes.
    6. **Review After 24 Hours:** Re-code solutions next day to reinforce pattern recognition.

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Off-by-one in query** | Confusion about inclusive/exclusive ranges | Always use `prefix[right+1] - prefix[left]` for inclusive [left, right] |
    | **Forgetting prefix[0] = 0** | Not adding extra element at start | Initialize with `prefix = [0]` then build from there |
    | **Wrong 2D formula** | Inclusion-exclusion principle tricky | Draw diagram: add two rectangles, subtract overlap |
    | **Not handling negatives** | Assuming prefix always increases | Prefix sum works fine with negatives; just track all values |
    | **Hash map initialized wrong** | Missing base case for subarray from index 0 | Always start with `{0: 1}` in subarray sum problems |

---

### Monotonic Stack

=== "Understanding the Pattern"

    ## 📖 What is Monotonic Stack?

    Imagine you're standing in a crowded concert venue, trying to see the stage. You look around and notice something interesting: you can only see the people between you and the stage who are taller than everyone in front of them. Why? Because shorter people get "blocked" by taller ones in front!

    This is exactly how a **Monotonic Stack** works. It maintains a stack of elements in either increasing or decreasing order, automatically "popping" (removing) elements that get "blocked" or "dominated" by new arrivals. It's like nature's way of filtering out irrelevant information—keeping only what matters.

    The brilliant insight: when looking for "next greater" or "next smaller" elements, most elements can be eliminated immediately because they'll never be the answer for anything!

    ---

    ## 🔧 How It Works

    A monotonic stack maintains a specific ordering property. When a new element arrives that violates this property, we pop elements until the property is restored.

    **Monotonic Decreasing Stack (for Next Greater Element):**
    ```
    Array: [4, 5, 2, 10, 6]

    Process 4: Stack: [4]
    Process 5: 5 > 4 → pop 4 (5 is next greater for 4)
               Stack: [5]
    Process 2: 2 < 5 → just push
               Stack: [5, 2]
    Process 10: 10 > 2 → pop 2 (10 is next greater for 2)
                10 > 5 → pop 5 (10 is next greater for 5)
                Stack: [10]
    Process 6: 6 < 10 → just push
               Stack: [10, 6]
    ```

    **Key Observation:** Each element is pushed and popped at most once → O(n) time!

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** If element B comes after element A and B is greater, then A can NEVER be the "next greater element" for anything that comes after B!

    Think of it like a game of dominance:
    - Taller buildings block the view of shorter ones behind them
    - A stronger competitor makes previous weaker ones irrelevant
    - A higher temperature makes previous lower temps unimportant

    **Why This Works:**

    When we process elements left to right and encounter a new element that's greater than the stack top:
    1. We found the "next greater" for the stack top
    2. The stack top becomes irrelevant for future elements (blocked by the new element)
    3. Pop it and record the answer

    The stack always contains elements that are still "candidates" for being someone's next greater element.

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n) | Each element pushed once, popped once—total 2n operations |
    | **Space** | O(n) | Stack can contain all elements in worst case (sorted descending) |
    | **Improvement** | From O(n²) | Eliminates nested loops that check every pair |

    **Why so efficient?** Even though there's a while loop inside a for loop, each element can only be popped once, so total pops = n maximum.

---

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Monotonic Stack Works | Example |
    |----------|---------------------------|---------|
    | **Next Greater/Smaller Element** | Automatically maintains candidates in order | Next Greater Element, Daily Temperatures |
    | **Histogram/Rectangle Problems** | Track boundaries where height constraints break | Largest Rectangle in Histogram |
    | **Stock Span Problems** | Count consecutive days satisfying condition | Stock Price Span |
    | **Visibility Problems** | Determine what can be "seen" from each position | Buildings with Ocean View |
    | **Range Queries with Constraints** | Find subarrays where element is min/max | Sum of Subarray Minimums |

    **Red Flags That Suggest Monotonic Stack:**
    - "Next greater/smaller element to the right/left"
    - "How many consecutive elements before current satisfy condition?"
    - "Largest rectangle/area with constraints"
    - "For each element, find the span/range it dominates"
    - Problem involves "blocking", "visibility", or "dominance"

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Need all pairs comparison** | Monotonic stack optimizes by skipping, but you need everything | Nested loops O(n²) unavoidable |
    | **No monotonic relationship** | Elements don't have greater/smaller ordering | Different data structure needed |
    | **Random access required** | Stack only gives access to top element | Array or hash map |
    | **Bidirectional queries** | Need both next and previous efficiently | Precompute both directions separately |
    | **Dynamic updates** | Elements added/removed frequently | Segment tree or other dynamic structure |

    ---

    ## 🎯 Decision Flowchart

    ```
    Does problem ask for "next" or "previous" greater/smaller?
    ├─ Yes → Can you process elements in one direction?
    │         ├─ Yes → USE MONOTONIC STACK ✓
    │         └─ No → Process both directions separately
    └─ No → Does problem involve "span" or "range domination"?
              ├─ Yes → USE MONOTONIC STACK ✓
              └─ No → Are you finding max/min in subarrays?
                       ├─ Yes → USE MONOTONIC STACK (histogram variant) ✓
                       └─ No → Different pattern needed
    ```

=== "Implementation Templates"

    === "Template 1: Next Greater Element"

        **Use Case:** Find the next element to the right that is greater than current element

        **Pattern:** Maintain decreasing stack, pop when smaller elements encountered

        ```python
        def next_greater_element(nums):
            """
            Find next greater element for each element in array.

            Perfect for: Problems asking "what's the next bigger value?"

            Concept:
            - Maintain decreasing stack of indices
            - When current > stack top: found answer for stack top
            - Pop and record, repeat until stack top >= current

            Time: O(n) - Each element pushed/popped once
            Space: O(n) - Stack storage

            Example: [4, 5, 2, 10] → [5, 10, 10, -1]
            """
            n = len(nums)
            result = [-1] * n  # Default: no greater element
            stack = []  # Stores indices (not values!)

            for i in range(n):
                # Pop all elements smaller than current
                # They found their "next greater"!
                while stack and nums[stack[-1]] < nums[i]:
                    idx = stack.pop()
                    result[idx] = nums[i]  # nums[i] is next greater for nums[idx]

                # Current element is still waiting for its next greater
                stack.append(i)

            # Elements still in stack: no next greater exists
            # (result already initialized to -1)
            return result


        # Example Usage:
        nums = [4, 5, 2, 10]
        result = next_greater_element(nums)
        print(result)  # Output: [5, 10, 10, -1]
        ```

        **Key Points:**
        - Store indices, not values (need to record position in result)
        - Stack maintains decreasing order: top is smallest
        - Pop means "found the answer for this element"

    === "Template 2: Next Smaller Element"

        **Use Case:** Find the next element to the right that is smaller than current element

        **Pattern:** Maintain increasing stack (opposite of next greater)

        ```python
        def next_smaller_element(nums):
            """
            Find next smaller element for each element in array.

            Perfect for: Finding boundaries in histogram problems

            Concept:
            - Maintain increasing stack of indices
            - When current < stack top: found answer for stack top
            - Opposite logic of next greater

            Time: O(n) - Each element pushed/popped once
            Space: O(n) - Stack storage

            Example: [10, 5, 11, 4] → [5, 4, 4, -1]
            """
            n = len(nums)
            result = [-1] * n
            stack = []  # Monotonic increasing stack

            for i in range(n):
                # Pop all elements GREATER than current
                # (opposite of next greater!)
                while stack and nums[stack[-1]] > nums[i]:
                    idx = stack.pop()
                    result[idx] = nums[i]  # nums[i] is next smaller for nums[idx]

                stack.append(i)

            return result


        # Example Usage:
        nums = [10, 5, 11, 4]
        result = next_smaller_element(nums)
        print(result)  # Output: [5, 4, 4, -1]
        ```

        **Key Points:**
        - Stack maintains increasing order (opposite of next greater)
        - Pop when current is SMALLER than top
        - Used in histogram problems to find boundaries

    === "Template 3: Largest Rectangle in Histogram"

        **Use Case:** Find the largest rectangle that can be formed in a histogram

        **Pattern:** Use next smaller element concept to find width boundaries

        ```python
        def largest_rectangle_histogram(heights):
            """
            Find largest rectangle area in histogram.

            Perfect for: Max area problems with height constraints

            Concept:
            - For each bar, find how far left and right it can extend
            - Left boundary: previous smaller element
            - Right boundary: next smaller element
            - Area = height × width

            Time: O(n) - Single pass with stack
            Space: O(n) - Stack storage

            Example: [2, 1, 5, 6, 2, 3] → 10 (bars 5,6 with height 5)
            """
            stack = []  # Stores indices
            max_area = 0

            for i, h in enumerate(heights):
                # Pop bars taller than current
                # Current bar limits how far those bars can extend right
                while stack and heights[stack[-1]] > h:
                    height_idx = stack.pop()
                    height = heights[height_idx]

                    # Width calculation:
                    # Right boundary: current position i
                    # Left boundary: element after new stack top
                    width = i if not stack else i - stack[-1] - 1

                    area = height * width
                    max_area = max(max_area, area)

                stack.append(i)

            # Process remaining bars (no right boundary, extend to end)
            while stack:
                height_idx = stack.pop()
                height = heights[height_idx]
                width = len(heights) if not stack else len(heights) - stack[-1] - 1
                max_area = max(max_area, height * width)

            return max_area


        # Example Usage:
        heights = [2, 1, 5, 6, 2, 3]
        result = largest_rectangle_histogram(heights)
        print(result)  # Output: 10
        ```

        **Key Points:**
        - Width = distance from next smaller on left to next smaller on right
        - Stack helps find both boundaries efficiently
        - Classic application of monotonic stack pattern

    === "Visual Walkthrough"

        **Problem:** Next Greater Element in [4, 5, 2, 10]

        ```
        Initial State: result = [-1, -1, -1, -1], stack = []

        Step 1: Process 4 (index 0)
        Stack is empty, push 0
        Stack: [0]    (values: [4])

        Step 2: Process 5 (index 1)
        5 > 4 (nums[1] > nums[0])
        → Pop 0, set result[0] = 5
        Push 1
        Stack: [1]    (values: [5])
        Result: [5, -1, -1, -1]

        Step 3: Process 2 (index 2)
        2 < 5 (nums[2] < nums[1])
        → Just push 2
        Stack: [1, 2]    (values: [5, 2])
        Result: [5, -1, -1, -1]

        Step 4: Process 10 (index 3)
        10 > 2 (nums[3] > nums[2])
        → Pop 2, set result[2] = 10
        10 > 5 (nums[3] > nums[1])
        → Pop 1, set result[1] = 10
        Push 3
        Stack: [3]    (values: [10])
        Result: [5, 10, 10, -1]

        Done! Element at index 3 (value 10) has no next greater.
        ```

        **Why This Works:**

        Notice how elements get popped as soon as something greater arrives:
        - When 5 arrives, 4 is eliminated (5 blocks 4)
        - When 10 arrives, both 2 and 5 are eliminated (10 blocks both)
        - Stack only keeps elements that might still be useful

        ---

        **Histogram Problem:** Heights = [2, 1, 5, 6, 2, 3]

        ```
        Goal: Find largest rectangle area

        Visualization:
        6│    ┌─┐
        5│    │ │
        4│    │ │
        3│    │ │    ┌─┐
        2│┌─┐ │ │ ┌─┐│ │
        1││ │┌┴─┴─┤ ││ │
        0└┴─┴┴───┴─┴┴─┴─
         0 1 2 3 4 5

        Best rectangle: bars 2 and 3 (heights 5 and 6)
        → Use height 5, width 2 → Area = 10

        Stack Processing:
        i=0: push 0, stack=[0]
        i=1: 1 < 2, pop 0, area=2×1=2, push 1, stack=[1]
        i=2: push 2, stack=[1,2]
        i=3: push 3, stack=[1,2,3]
        i=4: 2 < 6, pop 3, width=4-2-1=1, area=6×1=6
             2 < 5, pop 2, width=4-1-1=2, area=5×2=10 ← MAX!
             push 4, stack=[1,4]
        i=5: push 5, stack=[1,4,5]
        End: process remaining...
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic mechanics of monotonic stack.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Next Greater Element I | Decreasing stack | Basic template, understand push/pop logic | [LeetCode 496](https://leetcode.com/problems/next-greater-element-i/) |
    | Remove All Adjacent Duplicates | Stack application | Using stack for pattern matching | [LeetCode 1047](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/) |
    | Baseball Game | Stack operations | Practice stack manipulation | [LeetCode 682](https://leetcode.com/problems/baseball-game/) |
    | Backspace String Compare | Stack simulation | Stack for edit operations | [LeetCode 844](https://leetcode.com/problems/backspace-string-compare/) |
    | Min Stack | Auxiliary stack | Maintain additional monotonic property | [LeetCode 155](https://leetcode.com/problems/min-stack/) |

    **Goal:** Understand when to push, when to pop, and what to store in the stack.

    ---

    ### Phase 2: Application (Medium)
    Apply pattern to classic next greater/smaller problems.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Daily Temperatures | Next greater | Days until warmer—store distance not value | [LeetCode 739](https://leetcode.com/problems/daily-temperatures/) |
    | Next Greater Element II | Circular array | Process array twice for circular wrap | [LeetCode 503](https://leetcode.com/problems/next-greater-element-ii/) |
    | Online Stock Span | Span calculation | Count consecutive elements meeting condition | [LeetCode 901](https://leetcode.com/problems/online-stock-span/) |
    | Remove K Digits | Monotonic property | Build result maintaining increasing order | [LeetCode 402](https://leetcode.com/problems/remove-k-digits/) |
    | Sum of Subarray Minimums | Advanced | Combine with next/previous smaller element | [LeetCode 907](https://leetcode.com/problems/sum-of-subarray-minimums/) |

    **Goal:** Master the core template and understand distance vs value storage.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex histogram and rectangle problems.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Largest Rectangle in Histogram | Histogram | Find area using width boundaries | [LeetCode 84](https://leetcode.com/problems/largest-rectangle-in-histogram/) |
    | Maximal Rectangle | 2D histogram | Apply histogram technique row by row | [LeetCode 85](https://leetcode.com/problems/maximal-rectangle/) |
    | Trapping Rain Water | Water trapped | Calculate trapped water using height constraints | [LeetCode 42](https://leetcode.com/problems/trapping-rain-water/) |
    | Maximum Width Ramp | Width maximization | Find max distance with value constraint | [LeetCode 962](https://leetcode.com/problems/maximum-width-ramp/) |

    **Goal:** Apply monotonic stack to complex area/volume calculation problems.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Next Greater:** Master the basic template first
    2. **Understand Push/Pop:** Know exactly why each operation happens
    3. **Draw the Stack:** Visualize stack state after each iteration
    4. **Index vs Value:** Decide whether to store indices or values
    5. **Identify Monotonic Property:** Increasing or decreasing?
    6. **Practice Variations:** Next/previous, greater/smaller, circular

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Storing values instead of indices | Forget that you need position information | Always store indices unless problem explicitly needs only values |
    | Wrong comparison in while loop | Confuse greater with smaller, or >= with > | Next greater: pop when current > top; Next smaller: pop when current < top |
    | Forgetting to process remaining stack | Focus only on main loop | After loop, process elements still in stack (they have no next greater/smaller) |
    | Not initializing result array | Assume default values will work | Initialize result to -1 or appropriate default before starting |
    | Off-by-one in width calculation | Confuse stack top position with width | Width = current_index - stack_top_index - 1 (for histogram problems) |

---

### Merge Intervals

=== "Understanding the Pattern"

    ## 📖 What is Merge Intervals?

    Imagine you're a meeting room coordinator at a busy office. Throughout the day, different teams book the same conference room for overlapping times. Your job? Figure out which meetings can share a room and which need separate spaces. Instead of checking every pair of meetings against each other (which would take forever), you organize them by start time on a timeline and simply walk through, grouping overlapping meetings together.

    This is the **Merge Intervals** pattern! It's about organizing time ranges (or any intervals) on a line and efficiently handling overlaps. The magic? Sort once, scan once—transforming a potentially O(n²) comparison problem into an elegant O(n log n) solution.

    Real-world applications are everywhere: scheduling systems, calendar management, resource allocation, network packet handling, and even video streaming (merging buffered segments)!

    ---

    ## 🔧 How It Works

    The Merge Intervals pattern works through two key steps:

    **Step 1: Sort by Start Time**
    ```
    Input:  [[15,18], [1,3], [8,10], [2,6]]

    Sorted: [[1,3], [2,6], [8,10], [15,18]]
            ├────┤  ├────────┤  ├────┤  ├────────┤
            Timeline →
    ```

    **Step 2: Linear Scan with Merging**
    ```
    Start: [[1,3]]  (first interval)

    Check [2,6]: Does 2 <= 3? YES → Overlap!
                 Merge: [1, max(3,6)] = [1,6]

    Check [8,10]: Does 8 <= 6? NO → Gap!
                  Add new: [[1,6], [8,10]]

    Check [15,18]: Does 15 <= 10? NO → Gap!
                   Add new: [[1,6], [8,10], [15,18]]
    ```

    **The Merge Condition:**
    Two intervals [a, b] and [c, d] overlap if: `c <= b` (after sorting by start)

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Sorting guarantees that if two intervals CAN merge, they will be adjacent in the sorted list!

    Think about it: if interval A overlaps with interval C, and B is between them in sorted order, then B must also overlap with both A and C. This means we never need to look back—we can merge intervals in a single forward pass!

    **Why This Works:**
    ```
    If: start[A] <= start[B] <= start[C]  (sorted order)
    And: A overlaps C (end[A] >= start[C])
    Then: B MUST overlap both A and C

    Because: start[B] <= start[C] <= end[A]
    ```

    This property lets us maintain just ONE "current merged interval" at a time, extending it when we find overlaps and starting fresh when we find gaps.

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n log n) | Dominated by sorting; merge scan is O(n) |
    | **Space** | O(n) | Result array (worst case: no merges, all n intervals returned) |
    | **Improvement** | From O(n²) | Without sorting, need to compare each pair of intervals |

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Merge Intervals Works | Example |
    |----------|--------------------------|---------|
    | **Overlapping time ranges** | Natural timeline representation | Meeting room scheduling |
    | **Resource conflicts** | Identify when resources clash | CPU/memory allocation |
    | **Range merging** | Consolidate continuous segments | File system blocks, IP ranges |
    | **Gap finding** | Sorted order makes gaps obvious | Find free time slots |
    | **Coverage problems** | Track what's covered by intervals | Video buffering, download ranges |

    **Red Flags That Suggest Merge Intervals:**
    - Problem mentions "intervals", "ranges", or "time periods"
    - Keywords: "overlapping", "merge", "scheduling", "conflicts"
    - Need to find "free time" or "gaps" between events
    - Counting simultaneous events (meetings, tasks, etc.)
    - Asked to "insert" a new interval into existing ones

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Order must be preserved** | Sorting changes original order | Store indices with intervals |
    | **Point queries** | Not optimized for "is X in any interval?" | Segment tree, interval tree |
    | **Dynamic updates** | Sorting after each insert is expensive | Interval tree with O(log n) insert |
    | **Very few intervals** | Sorting overhead not worth it | Simple O(n²) pairwise comparison |
    | **Intervals on 2D plane** | 1D merge doesn't work | Sweep line algorithm |

    ---

    ## 🎯 Decision Flowchart

    ```
    Are you working with intervals/ranges?
    ├─ Yes → Do they potentially overlap?
    │         ├─ Yes → Need to merge overlaps?
    │         │        ├─ Yes → USE MERGE INTERVALS ✓
    │         │        └─ No → Need to count overlaps?
    │         │                └─ Yes → USE MERGE INTERVALS (variant) ✓
    │         └─ No → Need to find gaps?
    │                  └─ Yes → USE MERGE INTERVALS (sorted scan) ✓
    └─ No → Use a different pattern
    ```

=== "Implementation Templates"

    === "Template 1: Basic Merge"

        **Use Case:** Consolidate all overlapping intervals into non-overlapping ones

        **Pattern:** Sort by start time, then greedily extend current interval or start new one

        ```python
        def merge_intervals(intervals):
            """
            Merge all overlapping intervals.

            Perfect for: Calendar consolidation, resource allocation

            Concept:
            - Sort ensures overlapping intervals are adjacent
            - Extend current interval when overlap found
            - Start new interval when gap encountered

            Time: O(n log n) - sorting dominates
            Space: O(n) - result array

            Example: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]
            """
            if not intervals:
                return []

            # Sort by start time (ensures overlapping intervals are adjacent)
            intervals.sort(key=lambda x: x[0])

            # Initialize with first interval
            merged = [intervals[0]]

            for current in intervals[1:]:
                last = merged[-1]

                # Check overlap: does current start before last ends?
                if current[0] <= last[1]:
                    # OVERLAP: Merge by extending last interval's end
                    # Use max() in case current is fully contained in last
                    last[1] = max(last[1], current[1])
                else:
                    # GAP: No overlap, start a new interval
                    merged.append(current)

            return merged

        # Example Usage:
        intervals = [[1,3], [2,6], [8,10], [15,18]]
        result = merge_intervals(intervals)
        print(result)  # [[1,6], [8,10], [15,18]]
        ```

        **Key Points:**
        - Overlap condition: `current[0] <= last[1]` (not `<` because touching intervals merge)
        - Use `max(last[1], current[1])` to handle fully contained intervals
        - Modifying `last` directly modifies the list (last is a reference)

    === "Template 2: Insert Interval"

        **Use Case:** Insert a new interval into a sorted, non-overlapping list and merge if needed

        **Pattern:** Three-phase approach without needing to re-sort

        ```python
        def insert_interval(intervals, new_interval):
            """
            Insert new interval and merge overlapping ones.

            Perfect for: Dynamic scheduling, adding events to calendar

            Concept:
            - Phase 1: Add all intervals that end before new one starts
            - Phase 2: Merge all intervals overlapping with new one
            - Phase 3: Add all intervals that start after new one ends

            Time: O(n) - single pass, no sorting needed
            Space: O(n) - result array

            Example: Insert [2,5] into [[1,2],[6,9]] → [[1,5],[6,9]]
            """
            result = []
            i = 0
            n = len(intervals)

            # Phase 1: Add all intervals BEFORE new_interval
            # (intervals that end before new_interval starts)
            while i < n and intervals[i][1] < new_interval[0]:
                result.append(intervals[i])
                i += 1

            # Phase 2: MERGE all overlapping intervals
            # (intervals whose start <= new_interval's end)
            while i < n and intervals[i][0] <= new_interval[1]:
                # Expand new_interval to encompass this overlapping interval
                new_interval[0] = min(new_interval[0], intervals[i][0])
                new_interval[1] = max(new_interval[1], intervals[i][1])
                i += 1

            # Add the merged interval
            result.append(new_interval)

            # Phase 3: Add all remaining intervals AFTER new_interval
            while i < n:
                result.append(intervals[i])
                i += 1

            return result

        # Example Usage:
        intervals = [[1,2], [3,5], [6,7], [8,10], [12,16]]
        new_interval = [4,8]
        result = insert_interval(intervals, new_interval)
        print(result)  # [[1,2], [3,10], [12,16]]
        ```

        **Key Points:**
        - No sorting needed since input is already sorted
        - Use `min()` for start and `max()` for end when merging
        - Three distinct phases make logic clear and bug-free

    === "Template 3: Meeting Rooms II"

        **Use Case:** Count maximum number of overlapping intervals at any point

        **Pattern:** Separate start and end events, process chronologically

        ```python
        def min_meeting_rooms(intervals):
            """
            Minimum meeting rooms needed for all meetings.

            Perfect for: Resource allocation, concurrent task scheduling

            Concept:
            - Treat starts and ends as separate events
            - Process events chronologically
            - Track running count of active meetings

            Time: O(n log n) - sorting starts and ends
            Space: O(n) - separate arrays for starts/ends

            Example: [[0,30],[5,10],[15,20]] → 2 rooms
            (meetings at 0 and 5 overlap, need 2 rooms)
            """
            if not intervals:
                return 0

            # Separate and sort start and end times independently
            starts = sorted([interval[0] for interval in intervals])
            ends = sorted([interval[1] for interval in intervals])

            rooms_needed = 0
            max_rooms = 0
            start_ptr = 0
            end_ptr = 0

            # Process all start events
            while start_ptr < len(starts):
                if starts[start_ptr] < ends[end_ptr]:
                    # New meeting starts before earliest one ends
                    # Need an additional room
                    rooms_needed += 1
                    max_rooms = max(max_rooms, rooms_needed)
                    start_ptr += 1
                else:
                    # A meeting ends, free up a room
                    rooms_needed -= 1
                    end_ptr += 1

            return max_rooms

        # Example Usage:
        meetings = [[0, 30], [5, 10], [15, 20]]
        result = min_meeting_rooms(meetings)
        print(result)  # 2
        ```

        **Key Points:**
        - Sort starts and ends separately (key insight!)
        - Compare earliest unprocessed start vs earliest unprocessed end
        - Track maximum rooms ever needed, not just final count

    === "Visual Walkthrough"

        **Problem:** Merge [[1,3], [2,6], [8,10], [15,18]]

        ```
        Initial State (after sorting):
        [[1,3], [2,6], [8,10], [15,18]]

        Timeline:
        0    5    10   15   20
        |────┤           (1,3)
          ├────────┤     (2,6)
                ├──┤     (8,10)
                      ├────┤ (15,18)

        Step 1: Start with first interval
        merged = [[1,3]]

        Step 2: Check [2,6]
        Does 2 <= 3? YES → Overlap!
        Merge: [1, max(3,6)] = [1,6]
        merged = [[1,6]]

        Timeline after merge:
        0    5    10   15   20
        |────────────┤        (1,6) ← merged!
                ├──┤          (8,10)
                      ├────┤  (15,18)

        Step 3: Check [8,10]
        Does 8 <= 6? NO → Gap!
        Add as new interval
        merged = [[1,6], [8,10]]

        Step 4: Check [15,18]
        Does 15 <= 10? NO → Gap!
        Add as new interval
        merged = [[1,6], [8,10], [15,18]]

        Final Result: [[1,6], [8,10], [15,18]]
        Three separate time blocks!
        ```

        **Why This Works:**
        After sorting, if [2,6] overlaps with [1,3], any interval between them MUST also overlap. This is why we only need to check adjacent intervals!

        ---

        **Meeting Rooms II Example:** [[0,30], [5,10], [15,20]]

        ```
        Events timeline:
        Time:   0   5   10  15  20  30
        Starts: [0, 5, 15]
        Ends:   [10, 20, 30]

        Sorted starts: [0, 5, 15]
        Sorted ends:   [10, 20, 30]

        Time 0: Meeting starts → rooms = 1
        Time 5: Meeting starts (before time 10 end) → rooms = 2 ← MAX!
        Time 10: Meeting ends → rooms = 1
        Time 15: Meeting starts (before time 20 end) → rooms = 2
        Time 20: Meeting ends → rooms = 1
        Time 30: Meeting ends → rooms = 0

        Answer: 2 rooms needed maximum
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic mechanics of interval merging.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Meeting Rooms | Basic overlap check | Detect ANY overlaps in intervals | [LeetCode 252](https://leetcode.com/problems/meeting-rooms/) |
    | Merge Intervals | Basic merge | Core merging logic | [LeetCode 56](https://leetcode.com/problems/merge-intervals/) |
    | Can Attend Meetings | Sorting + gap check | Verify no overlaps exist | [LeetCode 252](https://leetcode.com/problems/meeting-rooms/) |
    | Summary Ranges | Continuous integers | Merge consecutive numbers | [LeetCode 228](https://leetcode.com/problems/summary-ranges/) |
    | Add Bold Tag | String intervals | Apply pattern to strings | [LeetCode 616](https://leetcode.com/problems/add-bold-tag-in-string/) |

    **Goal:** Solve all 5 problems. Understand overlap detection and basic merging.

    ---

    ### Phase 2: Application (Medium)
    Apply pattern to more complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Insert Interval | Three-phase insert | Insert without re-sorting | [LeetCode 57](https://leetcode.com/problems/insert-interval/) |
    | Meeting Rooms II | Count overlaps | Minimum resources needed | [LeetCode 253](https://leetcode.com/problems/meeting-rooms-ii/) |
    | Non-overlapping Intervals | Greedy selection | Minimum removals to eliminate overlaps | [LeetCode 435](https://leetcode.com/problems/non-overlapping-intervals/) |
    | Interval List Intersections | Two-pointer merge | Find intersection of two interval lists | [LeetCode 986](https://leetcode.com/problems/interval-list-intersections/) |
    | Merge Two Sorted Lists | Interval merging variant | Apply to linked lists | [LeetCode 21](https://leetcode.com/problems/merge-two-sorted-lists/) |

    **Goal:** Solve 4 out of 5. Learn to handle dynamic inserts and count overlaps.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex variations and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Employee Free Time | Multiple interval lists | Find gaps across multiple schedules | [LeetCode 759](https://leetcode.com/problems/employee-free-time/) |
    | My Calendar III | Dynamic booking | Max k-booking (overlapping events) | [LeetCode 732](https://leetcode.com/problems/my-calendar-iii/) |
    | Data Stream as Intervals | Dynamic intervals | Maintain intervals with streaming data | [LeetCode 352](https://leetcode.com/problems/data-stream-as-disjoint-intervals/) |
    | Remove Covered Intervals | Containment logic | Remove fully covered intervals | [LeetCode 1288](https://leetcode.com/problems/remove-covered-intervals/) |

    **Goal:** Solve 2 out of 4. Master dynamic scenarios and complex interval relationships.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Easy:** Build confidence by mastering the basic merge template
    2. **Identify the Variant:** Is it basic merge, insert, count overlaps, or find gaps?
    3. **Draw It Out:** Always sketch intervals on a timeline to visualize overlaps
    4. **Code Without Looking:** Try implementing from memory after understanding
    5. **Time Yourself:** Aim for <15 minutes per easy, <30 for medium, <45 for hard
    6. **Review After 24 Hours:** Spaced repetition solidifies pattern recognition

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Using `<` instead of `<=` for overlap | Forget that touching intervals should merge | Remember: [1,3] and [3,5] overlap at point 3 |
    | Not using `max()` when merging ends | Assume current interval always extends beyond | Current might be fully contained: [1,10] + [2,3] |
    | Forgetting to sort first | Try to merge unsorted intervals | Always sort by start time before merging |
    | Modifying input array | Sorting changes original data | Use `intervals[:]` or create copy if needed |
    | Off-by-one in insert template | Confuse phase boundaries | Use clear conditions: `end < start`, `start <= end` |

---

## LinkedList Patterns

### Fast & Slow Pointers

=== "Understanding the Pattern"

    ## 📖 What is Fast & Slow Pointers?

    Picture two runners on a track—one jogging at a steady pace, the other sprinting at double speed. If the track is a straight line, the sprinter reaches the finish first. But if the track is circular, something magical happens: eventually, the sprinter laps the jogger! This simple observation is the foundation of one of computer science's most elegant algorithms: **Floyd's Cycle Detection** (also called the "Tortoise and Hare" algorithm).

    **Fast & Slow Pointers** uses two references that traverse a linked structure at different speeds. The slow pointer moves one step at a time, while the fast pointer moves two steps. This speed difference creates powerful properties:
    - **Cycle Detection**: In a cycle, they MUST meet
    - **Middle Finding**: When fast finishes, slow is at the halfway point
    - **Offset Finding**: By starting one pointer ahead, they meet at specific positions

    This pattern turns complex linked list problems into simple, elegant O(1) space solutions!

    ---

    ## 🔧 How It Works

    The magic lies in the relative speed difference:

    **1. Cycle Detection:**
    ```
    Linear list (no cycle):
    1 → 2 → 3 → 4 → 5 → null

    Step 1: slow=1, fast=1
    Step 2: slow=2, fast=3
    Step 3: slow=3, fast=5
    Step 4: slow=4, fast=null  ← Fast reaches end, NO CYCLE

    Circular list (has cycle):
    1 → 2 → 3 → 4 → 5
        ↑           ↓
        ← ← ← ← ← ←

    Step 1: slow=1, fast=1
    Step 2: slow=2, fast=3
    Step 3: slow=3, fast=5
    Step 4: slow=4, fast=2 (wrapped around)
    Step 5: slow=5, fast=4
    Step 6: slow=2, fast=2  ← MEET! CYCLE DETECTED
    ```

    **2. Finding Middle:**
    ```
    List: 1 → 2 → 3 → 4 → 5

    Fast moves 2x: When fast reaches end, slow is at middle!

    Step 0: slow=1, fast=1
    Step 1: slow=2, fast=3
    Step 2: slow=3, fast=5
    Step 3: slow=3 ← MIDDLE, fast at end
    ```

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** In a cycle, the faster runner MUST eventually catch the slower one—it's mathematically inevitable!

    Think of it this way: each iteration, the gap between fast and slow changes by 1 (fast gains +2 positions, slow gains +1, net difference = +1 closing the gap). In a cycle of length C, after at most C iterations, the gap closes to 0 and they meet!

    **Why Two Speeds Work:**
    - **Speed 1x vs 2x**: Guarantees meeting in cycle with minimum overhead
    - **Speed 1x vs 3x**: Would work but might miss in certain cycle lengths
    - **Speed 1x vs 1x**: Would never meet (same speed)

    The 2x speed is the sweet spot: fast enough to guarantee detection, slow enough to be simple and reliable.

    **Mathematical Proof:**
    If cycle length = C, and fast gains 1 position per step, then after C steps, fast will have gained C positions = one full lap = meet!

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n) | Visit each node at most twice (slow once, fast once or twice) |
    | **Space** | O(1) | Only two pointer variables—no hash set or recursion! |
    | **Improvement** | From O(n) space | Alternative: use hash set to track visited nodes (O(n) space) |

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Fast & Slow Works | Example |
    |----------|----------------------|---------|
    | **Cycle detection** | Speed difference forces meeting in cycles | Linked list has cycle? |
    | **Find middle** | Fast reaches end when slow at middle | Middle of linked list |
    | **Offset problems** | Start fast ahead by k steps | Kth node from end |
    | **Palindrome check** | Find middle, then reverse and compare | Is linked list a palindrome? |
    | **Happy number** | Cycle in number sequence | Does number reach 1? |

    **Red Flags That Suggest Fast & Slow Pointers:**
    - "Detect cycle" in linked list
    - "Find middle" without knowing length
    - "Kth from end" without counting length first
    - "In-place" or "O(1) space" for linked list
    - Working with sequences that might loop

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Array with indices** | Can directly calculate middle/end | Use index arithmetic O(1) |
    | **Need exact cycle length** | Fast/slow only detects, doesn't measure | Hash set with counter |
    | **Doubly linked list** | Can traverse backward | Use prev pointers |
    | **Tree structures** | Multiple children, not linear | Use BFS/DFS |
    | **Multiple cycles** | Can only detect one cycle at a time | Graph algorithms |

    ---

    ## 🎯 Decision Flowchart

    ```
    Working with linked list?
    ├─ Yes → Need O(1) space?
    │         ├─ Yes → Detect cycle OR find middle OR kth from end?
    │         │        ├─ Yes → USE FAST & SLOW POINTERS ✓
    │         │        └─ No → Check other patterns
    │         └─ No → Hash set/recursion OK
    └─ No → Can model as linked list? (e.g., array with next indices)
              └─ Yes → USE FAST & SLOW POINTERS ✓
    ```

=== "Implementation Templates"

    === "Template 1: Cycle Detection"

        **Use Case:** Determine if a linked list has a cycle

        **Pattern:** Move pointers at different speeds; they meet if cycle exists

        ```python
        def has_cycle(head):
            """
            Detect if linked list has a cycle (Floyd's Algorithm).

            Perfect for: Cycle detection in any linked structure

            Concept:
            - Slow moves 1 step per iteration
            - Fast moves 2 steps per iteration
            - If cycle exists, they MUST meet
            - If no cycle, fast reaches null

            Time: O(n) - Visit each node at most once
            Space: O(1) - Only two pointers

            Example: 1→2→3→4→2 (cycle) → True
                     1→2→3→4→null → False
            """
            # Edge case: empty list or single node
            if not head or not head.next:
                return False

            # Initialize both pointers at head
            slow = head
            fast = head

            # Loop until fast reaches end (no cycle)
            while fast and fast.next:
                slow = slow.next          # Move 1 step
                fast = fast.next.next     # Move 2 steps

                if slow == fast:
                    return True  # Pointers met → cycle exists!

            # Fast reached end → no cycle
            return False

        # Example Usage:
        # list: 1 → 2 → 3 → 4 → 2 (cycle back to node 2)
        # has_cycle(list.head)  # Returns: True
        ```

        **Key Points:**
        - Check `fast.next` to avoid null pointer error on `fast.next.next`
        - Don't check `slow.next` (if fast is valid, slow is always valid)
        - Initialize both at head (some variants start slow at head, fast at head.next)

    === "Template 2: Find Cycle Start"

        **Use Case:** Find the exact node where the cycle begins

        **Pattern:** Two-phase approach using mathematical property

        ```python
        def detect_cycle(head):
            """
            Find the node where the cycle begins.

            Perfect for: Identifying cycle entry point

            Concept (Mathematical Property):
            - Phase 1: Detect cycle (fast meets slow at some point M)
            - Phase 2: Reset slow to head, move both at same speed
            - They meet at cycle start (mathematical proof below)

            Math: If cycle starts at distance k from head, and has length C,
                  then after meeting, both are k steps from cycle start

            Time: O(n) - Two passes at most
            Space: O(1) - Only two pointers

            Example: 1→2→3→4→5→3 (cycle at 3) → Returns node 3
            """
            if not head or not head.next:
                return None

            # Phase 1: Detect if cycle exists
            slow = fast = head
            has_cycle = False

            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next

                if slow == fast:
                    has_cycle = True
                    break  # Exit when they meet

            if not has_cycle:
                return None  # No cycle, return null

            # Phase 2: Find cycle start
            # Mathematical property: reset slow to head,
            # move both at same speed → they meet at cycle start
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next  # Both move at same speed now

            return slow  # This is the cycle start node

        # Example Usage:
        # list: 1 → 2 → 3 → 4 → 5 → 3 (cycle back to node 3)
        # node = detect_cycle(list.head)  # Returns: node 3
        ```

        **Key Points:**
        - After meeting in phase 1, both are equidistant from cycle start
        - Phase 2 uses 1x speed for BOTH pointers (not 2x anymore)
        - This is based on Floyd's mathematical proof (trust the algorithm!)

    === "Template 3: Find Middle"

        **Use Case:** Find the middle node of a linked list in one pass

        **Pattern:** When fast reaches end, slow is at middle

        ```python
        def find_middle(head):
            """
            Find middle node of linked list in one pass.

            Perfect for: Splitting list, palindrome check, merge sort

            Concept:
            - Fast moves 2x speed → reaches end 2x faster
            - When fast at end, slow has moved half distance
            - For even length, returns second middle

            Time: O(n) - Single pass
            Space: O(1) - Only two pointers

            Example: 1→2→3→4→5 → Returns node 3 (middle)
                     1→2→3→4 → Returns node 3 (second middle)
            """
            if not head:
                return None

            slow = fast = head

            # Fast moves 2 steps, slow moves 1 step
            while fast and fast.next:
                slow = slow.next          # Move 1 step
                fast = fast.next.next     # Move 2 steps

            # When fast reaches end, slow is at middle
            return slow

        # Example Usage:
        # list: 1 → 2 → 3 → 4 → 5
        # middle = find_middle(list.head)  # Returns: node 3
        ```

        **Key Points:**
        - For odd length: slow points to exact middle
        - For even length: slow points to second middle (use `fast.next.next` check to get first middle)
        - Perfect for splitting lists in merge sort

    === "Template 4: Kth From End"

        **Use Case:** Find the kth node from the end without knowing list length

        **Pattern:** Start fast k steps ahead, then move both together

        ```python
        def kth_from_end(head, k):
            """
            Find kth node from end without knowing length.

            Perfect for: Remove nth from end, split at position

            Concept:
            - Move fast k steps ahead of slow
            - Maintain gap of k between them
            - When fast reaches end, slow is k from end

            Time: O(n) - Single pass
            Space: O(1) - Only two pointers

            Example: 1→2→3→4→5, k=2 → Returns node 4 (2nd from end)
            """
            fast = slow = head

            # Move fast k steps ahead
            for _ in range(k):
                if not fast:
                    return None  # k is larger than list length
                fast = fast.next

            # Move both until fast reaches end
            # Maintain gap of k between them
            while fast:
                slow = slow.next
                fast = fast.next

            # Slow is now k nodes from the end
            return slow

        # Example Usage:
        # list: 1 → 2 → 3 → 4 → 5
        # node = kth_from_end(list.head, 2)  # Returns: node 4
        ```

        **Key Points:**
        - For "remove kth from end", advance fast by k+1 to get node before target
        - Handle edge case where k > list length
        - Both pointers move at same speed (not 2x like other variants)

    === "Visual Walkthrough"

        **Problem:** Detect cycle in 1 → 2 → 3 → 4 → 5 → 3 (cycles back to node 3)

        ```
        Structure:
        1 → 2 → 3 → 4 → 5
                ↑       ↓
                ← ← ← ←

        Phase 1: Detect Cycle
        ─────────────────────
        Step 0:
        1 → 2 → 3 → 4 → 5 → 3 → ...
        ↑
        slow, fast (both start at 1)

        Step 1:
        1 → 2 → 3 → 4 → 5 → 3 → ...
            ↑       ↑
          slow     fast

        Step 2:
        1 → 2 → 3 → 4 → 5 → 3 → 4 → ...
                ↑           ↑
              slow        fast

        Step 3:
        1 → 2 → 3 → 4 → 5 → 3 → 4 → 5 → ...
                    ↑               ↑
                  slow            fast

        Step 4:
        1 → 2 → 3 → 4 → 5 → 3 → 4 → ...
                        ↑   ↑
                      slow fast  ← THEY MEET! Cycle detected

        Phase 2: Find Cycle Start
        ──────────────────────────
        Reset slow to head, move both at same speed:

        Step 1:
        1 → 2 → 3 → 4 → 5 → 3 → ...
        ↑           ↑
      slow        fast

        Step 2:
        1 → 2 → 3 → 4 → 5 → 3 → ...
            ↑           ↑
          slow        fast

        Step 3:
        1 → 2 → 3 → 4 → 5 → 3 → ...
                ↑   ↑
            slow, fast  ← MEET AT NODE 3! This is cycle start
        ```

        **Why This Works (Mathematical Proof):**
        - Let k = distance from head to cycle start = 2
        - Let C = cycle length = 4
        - When they meet in phase 1, both are k steps from cycle start
        - Moving both k steps at same speed → both reach cycle start!

        ---

        **Find Middle Example:** 1 → 2 → 3 → 4 → 5

        ```
        Step 0: slow=1, fast=1
        Step 1: slow=2, fast=3
        Step 2: slow=3, fast=5 (fast reached end)

        Result: slow at node 3 (middle)

        Visual:
        1 → 2 → 3 → 4 → 5
                ↑       ↑
              slow    fast (at end)

        Distance traveled: slow = 2 steps, fast = 4 steps
        Slow traveled exactly half!
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic mechanics of fast & slow pointers.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Linked List Cycle | Cycle detection | Basic Floyd's algorithm | [LeetCode 141](https://leetcode.com/problems/linked-list-cycle/) |
    | Middle of Linked List | Find middle | Speed difference for positioning | [LeetCode 876](https://leetcode.com/problems/middle-of-the-linked-list/) |
    | Remove Nth From End | Kth from end | Offset pointer technique | [LeetCode 19](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) |
    | Happy Number | Cycle in numbers | Apply pattern to non-linked data | [LeetCode 202](https://leetcode.com/problems/happy-number/) |
    | Intersection of Two Lists | Find meeting point | Two-pointer variant | [LeetCode 160](https://leetcode.com/problems/intersection-of-two-linked-lists/) |

    **Goal:** Solve all 5 problems. Understand basic cycle detection and middle finding.

    ---

    ### Phase 2: Application (Medium)
    Apply pattern to more complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Linked List Cycle II | Find cycle start | Two-phase approach | [LeetCode 142](https://leetcode.com/problems/linked-list-cycle-ii/) |
    | Palindrome Linked List | Find middle + reverse | Combine techniques | [LeetCode 234](https://leetcode.com/problems/palindrome-linked-list/) |
    | Reorder List | Find middle + merge | Multi-step manipulation | [LeetCode 143](https://leetcode.com/problems/reorder-list/) |
    | Sort List | Find middle for merge sort | Use in sorting algorithm | [LeetCode 148](https://leetcode.com/problems/sort-list/) |
    | Delete Middle Node | Find and delete | Modify while finding | [LeetCode 2095](https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/) |

    **Goal:** Solve 4 out of 5. Learn to combine fast/slow with other operations.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex variations and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Find Duplicate Number | Array as implicit linked list | Apply to arrays using indices as "next" | [LeetCode 287](https://leetcode.com/problems/find-the-duplicate-number/) |
    | Rotate List | Find kth from end + rotation | Calculate rotation point | [LeetCode 61](https://leetcode.com/problems/rotate-list/) |
    | Circular Array Loop | Cycle in array | Handle forward/backward cycles | [LeetCode 457](https://leetcode.com/problems/circular-array-loop/) |
    | Split Linked List in Parts | Multiple middle finding | Divide into k parts | [LeetCode 725](https://leetcode.com/problems/split-linked-list-in-parts/) |

    **Goal:** Solve 2 out of 4. Master non-traditional applications of the pattern.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Easy:** Build intuition with basic cycle detection and middle finding
    2. **Identify the Variant:** Cycle detection, find middle, or offset finding?
    3. **Draw It Out:** Visualize pointer movement on paper step-by-step
    4. **Code Without Looking:** Implement from memory to build muscle memory
    5. **Time Yourself:** Aim for <15 minutes per easy, <25 for medium
    6. **Review After 24 Hours:** Revisit to reinforce the pattern recognition

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Not checking `fast.next` before `fast.next.next` | Causes null pointer exception | Always check: `while fast and fast.next` |
    | Using slow/fast for kth from end | Wrong technique for this variant | Use equal-speed pointers with offset |
    | Starting pointers at wrong positions | Confusion about initialization | Cycle: both at head; Kth from end: fast ahead by k |
    | Forgetting to reset pointer in phase 2 | Cycle start detection requires reset | Reset slow to head after detecting cycle |
    | Off-by-one in "remove kth from end" | Need node BEFORE target | Move fast k+1 steps ahead to get previous node |

---

## Tree & Graph Patterns

### Breadth-First Search (BFS)

=== "Understanding the Pattern"

    ## 📖 What is BFS?

    Imagine dropping a pebble into a still pond. Ripples expand outward in perfect circles—the closest points are touched first, then points further away, layer by layer. This is exactly how **Breadth-First Search (BFS)** explores trees and graphs!

    BFS visits all nodes at distance 1 from the start, then all nodes at distance 2, then distance 3, and so on. It's like exploring your social network: first your direct friends (level 1), then friends-of-friends (level 2), then friends-of-friends-of-friends (level 3).

    The key insight? By exploring level-by-level, BFS guarantees finding the shortest path in unweighted graphs—the first time you reach a node is the shortest route to it!

    ---

    ## 🔧 How It Works

    BFS uses a **queue** (First-In-First-Out) to maintain the exploration order:

    **Basic Mechanism:**
    ```
    Tree:           1
                   / \
                  2   3
                 / \   \
                4   5   6

    Level 0: Visit 1 → Queue: []
    Level 1: Add 2, 3 → Queue: [2, 3]
             Visit 2 → Queue: [3] → Add 4, 5
             Visit 3 → Queue: [4, 5, 6] → Add 6
    Level 2: Visit 4, 5, 6 → Queue: []

    Visit Order: 1 → 2 → 3 → 4 → 5 → 6 (level-by-level!)
    ```

    **The Queue Ensures Order:**
    - **Enqueue**: Add children/neighbors to back of queue
    - **Dequeue**: Process from front of queue (FIFO)
    - **Result**: All nodes at level k processed before any node at level k+1

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Why does BFS find the shortest path? Because it explores distances in increasing order!

    Think of it like searching for your keys:
    - **BFS approach**: Check all places 1 step away, then 2 steps, then 3 steps...
    - **DFS approach**: Check 1 path all the way to the end, backtrack, try another path...

    BFS guarantees you'll find your keys in the closest possible location because you check closer places first!

    **Why Queue?**
    - Queue = FIFO = First node added is first node processed
    - This creates the level-by-level, expanding ripple effect
    - Contrast with DFS using stack (LIFO = Last node added is first processed = depth-first)

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(V + E) | Visit each vertex once, explore each edge once |
    | **Space** | O(V) | Queue can hold entire level (worst: all nodes in widest level) |
    | **Shortest Path** | Guaranteed | First visit to node is shortest path in unweighted graph |
    | **Memory vs DFS** | Higher | Queue holds level width vs DFS stack holds height |

    **Why O(V + E)?** We visit each vertex once (V) and check each edge once when exploring neighbors (E). Total: V + E operations.

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why BFS Works | Example |
    |----------|---------------|---------|
    | **Shortest path (unweighted)** | First visit = shortest distance | Find minimum moves in game |
    | **Level-order traversal** | Natural level-by-level exploration | Print tree by levels |
    | **Nodes at distance K** | Track level while traversing | Find all nodes K distance away |
    | **Connected components** | Explore all reachable nodes from start | Count islands in grid |
    | **Minimum steps problems** | Each step = one level | Minimum knight moves, word ladder |

    **Red Flags That Suggest BFS:**
    - "Shortest path" or "minimum steps" (unweighted)
    - "Level by level" or "layer by layer"
    - "Nearest" or "closest" nodes
    - Grid problems with minimum moves
    - Tree problems asking for level information

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Weighted graphs** | BFS assumes all edges cost 1 | Dijkstra's algorithm |
    | **Need all paths** | BFS finds shortest, not all | DFS with backtracking |
    | **Deep, narrow graphs** | BFS uses more memory | DFS (uses less memory) |
    | **Topological sort** | Need to explore deeply first | DFS post-order |
    | **Detecting back edges** | BFS doesn't naturally detect | DFS tracks back edges |

    ---

    ## 🎯 Decision Flowchart

    ```
    Need to explore a graph/tree?
    ├─ Yes → Need shortest path (unweighted)?
    │         ├─ Yes → USE BFS ✓
    │         └─ No → Need level-by-level info?
    │                  ├─ Yes → USE BFS ✓
    │                  └─ No → Need to explore all paths?
    │                           ├─ Yes → DFS better
    │                           └─ No → Either works, prefer BFS for shortest
    └─ No → Different pattern needed
    ```

=== "Implementation Templates"

    === "Template 1: Tree Level-Order"

        **Use Case:** Traverse tree level by level

        **Pattern:** Process all nodes at current level before moving to next

        ```python
        from collections import deque

        def level_order(root):
            """
            Level-order traversal of binary tree.

            Perfect for: Tree problems requiring level information

            Concept: Use queue to process nodes level by level.
                    Level size captured at start of each level ensures
                    we process exactly one level at a time.

            Time: O(n) - Visit each node once
            Space: O(w) - Queue holds max level width

            Example: [[1], [2,3], [4,5]] for tree below
            """
            if not root:
                return []

            result = []
            queue = deque([root])  # Start with root

            while queue:
                # Capture current level size
                level_size = len(queue)
                current_level = []

                # Process exactly 'level_size' nodes (this level)
                for _ in range(level_size):
                    node = queue.popleft()  # Get next node in level
                    current_level.append(node.val)

                    # Add children for next level
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)

                result.append(current_level)

            return result


        # Example Usage:
        #       1
        #      / \
        #     2   3
        #    / \
        #   4   5
        result = level_order(root)
        print(result)  # Output: [[1], [2,3], [4,5]]
        ```

        **Key Points:**
        - Capture `len(queue)` at start of each level
        - Process exactly that many nodes
        - Children added for next level

    === "Template 2: Graph BFS"

        **Use Case:** Traverse graph, avoid cycles with visited set

        **Pattern:** Track visited nodes to prevent infinite loops

        ```python
        from collections import deque

        def bfs_graph(graph, start):
            """
            BFS traversal of graph.

            Perfect for: Graph exploration, connected components

            Concept: Use visited set to avoid revisiting nodes (cycles).
                    Process each node once, explore all neighbors.

            Time: O(V + E) - Visit each vertex, check each edge
            Space: O(V) - Queue and visited set

            Example: Graph {0: [1,2], 1: [3], 2: [3], 3: []}
                    → Visit order: [0, 1, 2, 3]
            """
            visited = set([start])  # Mark start as visited
            queue = deque([start])
            result = []

            while queue:
                node = queue.popleft()
                result.append(node)

                # Explore all unvisited neighbors
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)  # Mark before adding to queue!
                        queue.append(neighbor)

            return result


        # Example Usage:
        graph = {
            0: [1, 2],
            1: [3],
            2: [3],
            3: []
        }
        result = bfs_graph(graph, 0)
        print(result)  # Output: [0, 1, 2, 3]
        ```

        **Key Points:**
        - Mark node as visited **when adding to queue**, not when processing
        - This prevents adding same node multiple times
        - visited set = O(1) lookup

    === "Template 3: Shortest Path"

        **Use Case:** Find minimum distance in unweighted graph

        **Pattern:** Track distance with each node in queue

        ```python
        from collections import deque

        def shortest_path(graph, start, end):
            """
            Find shortest path length in unweighted graph.

            Perfect for: Minimum steps, shortest distance problems

            Concept: Store (node, distance) in queue. First time we
                    reach target is guaranteed shortest in unweighted graph.

            Time: O(V + E) - BFS traversal
            Space: O(V) - Queue and visited

            Example: shortest_path({0: [1,2], 1: [3], 2: [3], 3: []}, 0, 3)
                    → Returns 2 (0→1→3 or 0→2→3)
            """
            if start == end:
                return 0

            visited = set([start])
            queue = deque([(start, 0)])  # (node, distance)

            while queue:
                node, dist = queue.popleft()

                # Check each neighbor
                for neighbor in graph[node]:
                    # Found target! Return distance
                    if neighbor == end:
                        return dist + 1

                    # Continue exploring
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            return -1  # No path exists


        # Example Usage:
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        distance = shortest_path(graph, 0, 3)
        print(distance)  # Output: 2
        ```

        **Key Points:**
        - Store distance with each node
        - First arrival at node = shortest path
        - Early termination when target found

    === "Visual Walkthrough"

        **Problem:** Level-order traversal of tree

        ```
        Tree:       1
                   / \
                  2   3
                 / \
                4   5

        Initial: queue = [1]

        Level 0:
        - Process 1
        - Add children 2, 3
        - queue = [2, 3]
        - result = [[1]]

        Level 1:
        - level_size = 2 (process 2 nodes)
        - Process 2: Add children 4, 5 → queue = [3, 4, 5]
        - Process 3: No children → queue = [4, 5]
        - result = [[1], [2, 3]]

        Level 2:
        - level_size = 2
        - Process 4, 5 (no children)
        - queue = []
        - result = [[1], [2, 3], [4, 5]]

        Done!
        ```

        **Why Level-Size Trick Works:**

        The key is `level_size = len(queue)` at start of each level:
        - Before processing level: queue contains **exactly** that level's nodes
        - As we process: we add next level's nodes
        - By processing exactly `level_size` times: we process current level only

        ---

        **Problem:** Shortest path from A to D

        ```
        Graph:  A -- B
                |    |
                C -- D

        BFS from A:

        Step 0: queue = [(A, 0)], visited = {A}

        Step 1: Process A, dist=0
                Neighbors: B, C
                queue = [(B, 1), (C, 1)]
                visited = {A, B, C}

        Step 2: Process B, dist=1
                Neighbors: D
                Found D! Return 1+1 = 2

        Path: A → B → D (distance 2)
        OR:   A → C → D (distance 2)
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master basic BFS mechanics.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Binary Tree Level Order Traversal | Tree BFS | Level-by-level traversal | [LeetCode 102](https://leetcode.com/problems/binary-tree-level-order-traversal/) |
    | Average of Levels | Tree BFS | Level aggregation | [LeetCode 637](https://leetcode.com/problems/average-of-levels-in-binary-tree/) |
    | Minimum Depth of Binary Tree | Tree BFS | Shortest path to leaf | [LeetCode 111](https://leetcode.com/problems/minimum-depth-of-binary-tree/) |
    | Symmetric Tree | Tree BFS | Level comparison | [LeetCode 101](https://leetcode.com/problems/symmetric-tree/) |
    | N-ary Tree Level Order | Tree BFS | Multiple children | [LeetCode 429](https://leetcode.com/problems/n-ary-tree-level-order-traversal/) |

    **Goal:** Solve all 5 problems. Understand queue mechanics and level processing.

    ---

    ### Phase 2: Application (Medium)
    Apply BFS to graphs and complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Binary Tree Right Side View | Tree BFS | Rightmost per level | [LeetCode 199](https://leetcode.com/problems/binary-tree-right-side-view/) |
    | Rotting Oranges | Multi-source BFS | Multiple starting points | [LeetCode 994](https://leetcode.com/problems/rotting-oranges/) |
    | Word Ladder | Graph BFS | String transformation | [LeetCode 127](https://leetcode.com/problems/word-ladder/) |
    | 01 Matrix | Grid BFS | Multi-source on grid | [LeetCode 542](https://leetcode.com/problems/01-matrix/) |
    | Binary Tree Zigzag Traversal | Tree BFS | Alternating direction | [LeetCode 103](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) |

    **Goal:** Solve 3 out of 5. Master multi-source BFS and grid problems.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex BFS with multiple constraints.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Shortest Path in Binary Matrix | Grid BFS | 8-directional movement | [LeetCode 1091](https://leetcode.com/problems/shortest-path-in-binary-matrix/) |
    | Bus Routes | Graph BFS | Complex state space | [LeetCode 815](https://leetcode.com/problems/bus-routes/) |
    | Sliding Puzzle | State-space BFS | Configuration BFS | [LeetCode 773](https://leetcode.com/problems/sliding-puzzle/) |
    | Minimum Genetic Mutation | Graph BFS | String mutation paths | [LeetCode 433](https://leetcode.com/problems/minimum-genetic-mutation/) |

    **Goal:** Solve 2 out of 4. Master state-space and complex graph BFS.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Trees:** Easier to visualize than graphs
    2. **Master Level Processing:** Understand the level_size trick
    3. **Add Visited Tracking:** Essential for graphs with cycles
    4. **Track Distance:** Practice shortest path problems
    5. **Try Multi-Source:** Multiple starting points in queue
    6. **Visualize:** Draw the tree/graph and trace queue contents

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Not capturing level_size | Process queue size changes during iteration | Store `len(queue)` before loop |
    | Marking visited when dequeuing | Node added to queue multiple times | Mark visited when **enqueuing** |
    | Forgetting base case | Empty tree/graph crashes | Check `if not root` or `if not graph` |
    | Using stack instead of queue | Confusion with DFS | Always use `deque` for BFS |
    | Not handling disconnected graphs | BFS only reaches connected component | Run BFS from each unvisited node |

---

### Depth-First Search (DFS)

=== "Understanding the Pattern"

    ## What is DFS?

    Imagine you're exploring a vast cave system with many branching tunnels. You have a flashlight and a ball of string. Your strategy? Pick a tunnel, follow it as far as it goes until you reach a dead end or discover something interesting. Then, use your string to backtrack to the last fork in the path, and explore a different tunnel. Repeat until you've explored every possible path.

    This is **Depth-First Search (DFS)**! Instead of exploring level by level (like BFS), DFS dives deep into one path before backtracking to explore alternatives. It's like being an adventurer who always says "let's see where this path leads" before checking the other options.

    The beauty of DFS lies in its commitment: it explores one possibility fully before considering others, making it perfect for problems where you need to find all possible solutions or explore every branch of a decision tree.

    ---

    ## How It Works

    DFS can be implemented in two ways, but both follow the same "go deep first" philosophy:

    **1. Recursive Approach (Using Call Stack):**
    ```
    Tree:       1
               / \
              2   3
             / \
            4   5

    Exploration Order:
    Visit 1 → Go left
    Visit 2 → Go left
    Visit 4 → Dead end, backtrack
    Back to 2 → Go right
    Visit 5 → Dead end, backtrack
    Back to 1 → Go right
    Visit 3 → Dead end, done!

    Result: [1, 2, 4, 5, 3]
    ```

    **2. Iterative Approach (Using Stack):**
    ```
    Stack: [1]          → Pop 1, push children [3, 2]
    Stack: [3, 2]       → Pop 2, push children [3, 5, 4]
    Stack: [3, 5, 4]    → Pop 4 (leaf)
    Stack: [3, 5]       → Pop 5 (leaf)
    Stack: [3]          → Pop 3 (leaf)

    Same result, different mechanism!
    ```

    ---

    ## Key Intuition

    **The Aha Moment:** DFS explores like you're solving a maze with the "right-hand rule"—always follow one wall as far as you can, then backtrack when stuck.

    Think of these real-world analogies:

    **Reading a book's table of contents:**
    - Chapter 1 → Section 1.1 → Subsection 1.1.1 (go deep!)
    - Finish 1.1.1? → Back to 1.1 → Try 1.1.2
    - Finish all of 1.1? → Back to Chapter 1 → Try Section 1.2

    **Searching through folders:**
    - Desktop → Projects → ProjectA → src → main.py (dive deep!)
    - Done with ProjectA? → Back to Projects → ProjectB
    - This is how file explorers work!

    The power of DFS comes from the **call stack** (recursive) or **explicit stack** (iterative) that remembers your path, so you can always backtrack to unexplored branches.

    ---

    ## Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(V + E) | Visit each vertex once, explore each edge once (V=vertices, E=edges) |
    | **Space (Recursive)** | O(H) | Recursion stack depth = height of tree/graph (worst case O(V)) |
    | **Space (Iterative)** | O(W) | Explicit stack size = width at deepest level (worst case O(V)) |
    | **Best For** | Deep structures | Excels when you need to go deep before going wide |

    **Why these complexities?** Each node is visited exactly once, and each edge is explored exactly once. The space depends on how much we need to remember about our path—in a balanced tree that's log(n), but in a chain it could be n.

=== "When to Use This Pattern"

    ## Perfect For

    | Scenario | Why DFS Works | Example |
    |----------|---------------|---------|
    | **Find all paths** | Explores every possibility exhaustively | All root-to-leaf paths, all routes in maze |
    | **Backtracking problems** | Natural fit for trying options and undoing | N-Queens, Sudoku solver, permutations |
    | **Connected components** | Mark all reachable nodes from starting point | Number of islands, friend circles |
    | **Cycle detection** | Track visited nodes in current path | Course schedule, deadlock detection |
    | **Topological sorting** | Postorder DFS gives reverse topological order | Build dependencies, task scheduling |
    | **Tree traversals** | Any preorder/inorder/postorder need | Copy tree, evaluate expressions |

    **Red Flags That Suggest DFS:**
    - "Find all possible..." (combinations, paths, solutions)
    - "Explore connected..." (components, regions, groups)
    - Tree or graph structure is explicitly given
    - Problem involves backtracking or trying multiple options
    - Need to go "as deep as possible" before exploring alternatives
    - Keywords: "path", "route", "maze", "island", "connected"

    ---

    ## When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Shortest path needed** | DFS doesn't guarantee shortest, just finds A path | BFS for unweighted, Dijkstra for weighted |
    | **Level-order traversal** | DFS goes deep, not wide | BFS for level-by-level |
    | **Very deep graphs** | Risk of stack overflow (especially recursive) | Iterative DFS or BFS |
    | **Infinite graphs** | Could get stuck in deep path forever | BFS with depth limit |
    | **Need nodes by distance** | DFS doesn't track distance from source | BFS guarantees closest nodes first |

    ---

    ## Decision Flowchart

    ```
    Does the problem involve a tree or graph?
    ├─ Yes → What do you need?
    │         ├─ All paths/solutions? → USE DFS ✓
    │         ├─ Shortest path? → Use BFS
    │         ├─ Connected components? → USE DFS ✓
    │         ├─ Cycle detection? → USE DFS ✓
    │         └─ Level-by-level? → Use BFS
    └─ No → Does it involve trying options and backtracking?
              ├─ Yes → USE DFS ✓
              └─ No → Different pattern needed
    ```

=== "Implementation Templates"

    === "Template 1: Recursive Tree DFS"

        **Use Case:** Tree traversals, path finding in trees

        **Pattern:** Let recursion handle the call stack

        ```python
        def dfs_recursive(root):
            """
            Recursive DFS traversal (preorder).

            Perfect for: Tree problems, clean and readable code

            Concept: Visit node, recurse left, recurse right
            The call stack automatically handles backtracking

            Time: O(n) - Visit each node once
            Space: O(h) - Recursion stack height (h = height)

            Example: Preorder traversal, max depth, path sum
            """
            result = []

            def dfs(node):
                # Base case: empty node
                if not node:
                    return

                # Process current node (preorder)
                result.append(node.val)

                # Recurse on children
                dfs(node.left)   # Explore left subtree fully
                dfs(node.right)  # Then explore right subtree

            dfs(root)
            return result


        # Example Usage:
        #     1
        #    / \
        #   2   3
        #  / \
        # 4   5
        # Output: [1, 2, 4, 5, 3]
        ```

        **Key Points:**
        - Base case: `if not node: return`
        - Order matters: preorder, inorder, or postorder
        - Call stack handles backtracking automatically

    === "Template 2: Iterative DFS with Stack"

        **Use Case:** Avoid recursion limits, explicit control

        **Pattern:** Manually manage stack for node exploration

        ```python
        def dfs_iterative(root):
            """
            Iterative DFS using explicit stack.

            Perfect for: Deep trees, avoiding recursion limits

            Concept: Stack simulates recursive call stack
            Push children in reverse order for same traversal as recursive

            Time: O(n) - Visit each node once
            Space: O(h) - Stack size = height

            Example: Same traversal as recursive, but iterative
            """
            if not root:
                return []

            result = []
            stack = [root]  # Initialize with root

            while stack:
                # Pop from top (LIFO - Last In First Out)
                node = stack.pop()
                result.append(node.val)

                # Push right first, then left
                # (so left is popped first, matching preorder)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)

            return result


        # Example Usage:
        # Same tree as above
        # Stack evolution:
        # [1] → [3,2] → [3,5,4] → [3,5] → [3]
        # Output: [1, 2, 4, 5, 3]
        ```

        **Key Points:**
        - Use list as stack: `append()` to push, `pop()` to pop
        - Push right before left for preorder traversal
        - Explicitly manage what recursive does automatically

    === "Template 3: Graph DFS with Visited Set"

        **Use Case:** Graphs with cycles, connected components

        **Pattern:** Track visited nodes to avoid infinite loops

        ```python
        def dfs_graph(graph, start):
            """
            DFS traversal of graph (handles cycles).

            Perfect for: Graphs, connected components, cycle detection

            Concept: Use visited set to avoid revisiting nodes
            Essential for graphs that might have cycles

            Time: O(V + E) - Visit vertices, explore edges
            Space: O(V) - Visited set stores all vertices

            Example: Find all reachable nodes, count components
            """
            visited = set()
            result = []

            def dfs(node):
                # Already visited? Skip to avoid cycle
                if node in visited:
                    return

                # Mark as visited before exploring
                visited.add(node)
                result.append(node)

                # Explore all neighbors
                for neighbor in graph[node]:
                    dfs(neighbor)

            dfs(start)
            return result


        # Example Usage:
        # graph = {
        #     0: [1, 2],
        #     1: [0, 3],
        #     2: [0],
        #     3: [1]
        # }
        # dfs_graph(graph, 0) → [0, 1, 3, 2]
        ```

        **Key Points:**
        - Check visited BEFORE recursing prevents cycles
        - Add to visited BEFORE exploring neighbors
        - Graph representation: adjacency list (dict of lists)

    === "Template 4: Backtracking with Path"

        **Use Case:** Find all paths, collect routes, solve puzzles

        **Pattern:** Build path as you go, backtrack by removing

        ```python
        def find_all_paths(root, target):
            """
            Find all root-to-target paths using backtracking.

            Perfect for: All paths problems, combinations, permutations

            Concept: Build path as you explore, undo (pop) when backtracking
            This is the essence of backtracking!

            Time: O(n) for visiting, O(2^n) for all paths
            Space: O(h) for recursion + path storage

            Example: All root-to-leaf paths, all subsets
            """
            all_paths = []

            def dfs(node, path):
                if not node:
                    return

                # Add current node to path
                path.append(node.val)

                # Found target? Save this path
                if node.val == target:
                    all_paths.append(path[:])  # Copy path!

                # Explore children
                dfs(node.left, path)
                dfs(node.right, path)

                # BACKTRACK: Remove current node
                path.pop()

            dfs(root, [])
            return all_paths


        # Example Usage:
        #     1
        #    / \
        #   2   3
        #  /
        # 4
        # find_all_paths(root, 4) → [[1, 2, 4]]
        # The path.pop() ensures we try other routes!
        ```

        **Key Points:**
        - Build path: `path.append(node.val)`
        - Save copy: `all_paths.append(path[:])` not `path`
        - Backtrack: `path.pop()` after exploring
        - This template is THE backtracking pattern!

    === "Visual Walkthrough"

        **Problem:** DFS traversal of tree

        ```
        Tree Structure:
                1
               / \
              2   3
             / \
            4   5

        Recursive DFS (Preorder) Step-by-Step:

        Step 1: Visit 1
        Current: 1, Stack (implicit): [1]
        Output: [1]
        Action: Recurse left to 2

        Step 2: Visit 2
        Current: 2, Stack: [1→2]
        Output: [1, 2]
        Action: Recurse left to 4

        Step 3: Visit 4
        Current: 4, Stack: [1→2→4]
        Output: [1, 2, 4]
        Action: Recurse left → None, Recurse right → None
        Backtrack to 2

        Step 4: Back at 2
        Stack: [1→2]
        Action: Recurse right to 5

        Step 5: Visit 5
        Current: 5, Stack: [1→2→5]
        Output: [1, 2, 4, 5]
        Action: Recurse left → None, Recurse right → None
        Backtrack to 2, then to 1

        Step 6: Back at 1
        Stack: [1]
        Action: Recurse right to 3

        Step 7: Visit 3
        Current: 3, Stack: [1→3]
        Output: [1, 2, 4, 5, 3]
        Action: Recurse left → None, Recurse right → None
        Done!

        Final Result: [1, 2, 4, 5, 3]
        ```

        **Why This Order?**
        - Preorder: Visit node BEFORE children
        - We always go LEFT first, then RIGHT
        - The call stack remembers where to backtrack
        - Each node is visited exactly once!

        ---

        **Graph DFS with Cycle:**

        ```
        Graph:  0 ←→ 1
                ↓    ↓
                2 ←→ 3

        Step 1: Start at 0, visited={0}
        Explore neighbors [1, 2]

        Step 2: Visit 1, visited={0,1}
        Explore neighbors [0, 3]
        → 0 already visited, skip!

        Step 3: Visit 3, visited={0,1,3}
        Explore neighbors [1, 2]
        → 1 already visited, skip!

        Step 4: Visit 2, visited={0,1,3,2}
        Explore neighbors [0, 3]
        → Both already visited!

        Result: [0, 1, 3, 2]
        The visited set prevents infinite loops!
        ```

=== "Practice Problems"

    ## Learning Path

    ### Phase 1: Foundation (Easy)
    Master basic DFS mechanics in trees and simple graphs.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Maximum Depth of Binary Tree | Recursive DFS | Basic depth calculation | [LeetCode 104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) |
    | Same Tree | Recursive comparison | Comparing two trees simultaneously | [LeetCode 100](https://leetcode.com/problems/same-tree/) |
    | Path Sum | Path tracking | Root-to-leaf with accumulation | [LeetCode 112](https://leetcode.com/problems/path-sum/) |
    | Invert Binary Tree | Recursive swap | Modifying tree structure | [LeetCode 226](https://leetcode.com/problems/invert-binary-tree/) |
    | Sum of Left Leaves | Conditional processing | Identifying node type | [LeetCode 404](https://leetcode.com/problems/sum-of-left-leaves/) |

    **Goal:** Solve all 5 problems. Get comfortable with recursive DFS and base cases.

    ---

    ### Phase 2: Application (Medium)
    Apply DFS to graphs, backtracking, and complex tree problems.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Number of Islands | Grid DFS | 2D DFS, connected components | [LeetCode 200](https://leetcode.com/problems/number-of-islands/) |
    | All Paths From Source to Target | Backtracking | Collecting all paths | [LeetCode 797](https://leetcode.com/problems/all-paths-from-source-to-target/) |
    | Course Schedule | Cycle detection | Topological sort, visited states | [LeetCode 207](https://leetcode.com/problems/course-schedule/) |
    | Binary Tree Right Side View | Level tracking | Modified DFS with levels | [LeetCode 199](https://leetcode.com/problems/binary-tree-right-side-view/) |
    | Clone Graph | Graph traversal | Copying with visited map | [LeetCode 133](https://leetcode.com/problems/clone-graph/) |

    **Goal:** Solve 3 out of 5. Learn graph DFS and backtracking patterns.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex state, multiple DFS passes, and optimization.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Serialize and Deserialize Binary Tree | Encoding/decoding | Tree representation as string | [LeetCode 297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) |
    | Binary Tree Maximum Path Sum | Complex logic | Path can start/end anywhere | [LeetCode 124](https://leetcode.com/problems/binary-tree-maximum-path-sum/) |
    | Word Ladder II | Backtracking + BFS | Finding all shortest paths | [LeetCode 126](https://leetcode.com/problems/word-ladder-ii/) |
    | Sudoku Solver | Backtracking | Constraint satisfaction | [LeetCode 37](https://leetcode.com/problems/sudoku-solver/) |

    **Goal:** Solve 2 out of 4. Master complex DFS applications.

    ---

    ## Practice Strategy

    1. **Start with Trees:** Easier than graphs, no cycle handling needed
    2. **Draw the Recursion:** Sketch the call stack on paper
    3. **Identify Pattern:** Preorder/inorder/postorder? Backtracking?
    4. **Write Base Cases First:** What stops the recursion?
    5. **Test Small Examples:** 3-node tree before 100-node tree
    6. **Iterative After Recursive:** Master recursive first, then try iterative
    7. **Track Time:** <15min for easy, <30min for medium

    ---

    ## Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Forgetting base case** | Dive into recursion without exit | Always write `if not node: return` first |
    | **Modifying during iteration** | Changing structure while traversing | Copy to new structure or mark visited |
    | **Not tracking visited (graphs)** | Forgetting graphs can have cycles | Always use visited set for graphs |
    | **Wrong traversal order** | Confused about pre/in/post order | Draw small tree, trace by hand |
    | **Not copying path** | Appending same list reference | Use `path[:]` or `path.copy()` |
    | **Stack overflow** | Very deep recursion | Switch to iterative or increase limit |
    | **Not backtracking** | Forgetting to undo path.append | Always `path.pop()` after recursion |

---

## Optimization & Search Patterns

### Hash Map

=== "Understanding the Pattern"

    ## What is a Hash Map?

    Imagine you're a librarian managing thousands of books. Someone asks, "Do you have The Great Gatsby?" The old way: walk through every shelf checking each book until you find it—that could take hours! But what if books were organized by a magic formula: take the first letter and a few other characters, compute a number, and that number tells you exactly which shelf to check? You'd find any book in seconds!

    This is a **Hash Map** (also called Hash Table or Dictionary). It's like having a super-smart filing system where a "hash function" converts your search key (like a book title) into a specific location where the value is stored. Instead of searching through everything, you instantly jump to the right spot.

    The brilliance? What normally takes O(n) time becomes O(1)—constant time, no matter how much data you have! You trade space (memory) for speed, storing everything in a way that makes lookups instantaneous.

    ---

    ## How It Works

    The magic happens through a **hash function** that converts keys to array indices:

    **Basic Mechanism:**
    ```
    Key: "apple"
    Hash Function: converts "apple" → 42
    Storage: array[42] = {"apple": $1.99}

    Lookup "apple":
    1. Hash "apple" → 42
    2. Check array[42]
    3. Found it instantly!

    Without Hash Map:
    Check index 0, 1, 2, 3... until found (O(n))

    With Hash Map:
    Direct access to index 42 (O(1))
    ```

    **Real Implementation:**
    ```python
    # Python dict is a hash map
    prices = {}
    prices["apple"] = 1.99    # Insert: O(1)
    prices["banana"] = 0.59   # Insert: O(1)

    cost = prices["apple"]    # Lookup: O(1)
    # No loops needed! Direct access!
    ```

    **Collision Handling:**
    ```
    What if two keys hash to same index?
    Key "apple" → hash → 42
    Key "elppa" → hash → 42 (collision!)

    Solution: Chaining
    array[42] = [("apple", $1.99), ("elppa", $2.50)]

    Still fast! Average O(1), worst case O(n)
    ```

    ---

    ## Key Intuition

    **The Aha Moment:** Instead of asking "is this needle in the haystack?", put the needle in a special place where you know exactly where to look!

    Think of these real-world analogies:

    **Phone contacts:**
    - Without hash map: Scroll through 500 contacts to find "Mom"
    - With hash map: Type "Mom", instantly see her number
    - Your phone uses a hash map!

    **Website login:**
    - Without hash map: Check every username/password pair
    - With hash map: Hash the username, jump to the password
    - Hash "john@email.com" → Check that exact storage slot

    **Counting votes:**
    - Without hash map: For each vote, count all previous occurrences
    - With hash map: Look up candidate's count (O(1)), add 1, store back
    - Transform O(n²) to O(n)!

    The power comes from **trading space for time**. We use extra memory (the hash table) to eliminate the need to search through everything.

    ---

    ## Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time (Average)** | O(1) | Direct index access via hash function |
    | **Time (Worst)** | O(n) | All keys collide to same bucket (rare with good hash) |
    | **Space** | O(n) | Store all n key-value pairs |
    | **Insert** | O(1) | Hash key, store at index |
    | **Delete** | O(1) | Hash key, remove from index |
    | **Search** | O(1) | Hash key, check index |

    **Why O(1)?** The hash function runs in constant time, and array access is O(1). Even with collisions, a good hash function keeps bucket sizes small and bounded.

=== "When to Use This Pattern"

    ## Perfect For

    | Scenario | Why Hash Map Works | Example |
    |----------|-------------------|---------|
    | **Fast lookups** | Direct access beats linear search | Check if element exists, find index |
    | **Frequency counting** | Increment counters without searching | Count letter occurrences, vote tallies |
    | **Two Sum problems** | Store complements for O(1) lookup | Find pair summing to target |
    | **Detect duplicates** | Check existence in O(1) | Remove duplicates, unique elements |
    | **Caching/Memoization** | Store computed results | Fibonacci memo, expensive calculations |
    | **Grouping/Categorizing** | Map keys to lists of items | Group anagrams, organize by property |
    | **Mapping relationships** | Key-to-value associations | User ID to name, word to definition |

    **Red Flags That Suggest Hash Map:**
    - "Find pair that..." (two sum, complement search)
    - "Count frequency of..." (histograms, tallies)
    - "Check if exists..." (membership testing)
    - "First occurrence of..." (track indices)
    - "Group by..." (categorization)
    - Need O(1) lookups instead of O(n) loops
    - Trading space for time is acceptable

    ---

    ## When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Need sorted order** | Hash maps are unordered | TreeMap/BST (O(log n)), sort then iterate |
    | **Memory constrained** | Hash maps use O(n) extra space | Two pointers on sorted array |
    | **Range queries** | Can't find "all between X and Y" | Binary Search Tree, Segment Tree |
    | **Minimum/Maximum tracking** | Need to maintain order | Heap (O(log n)), sorted structure |
    | **Few lookups** | Overhead not worth it for 2-3 lookups | Simple loop might be faster |
    | **Sequential access only** | If only iterating once | Array/list sufficient |

    ---

    ## Decision Flowchart

    ```
    Do you need fast lookups?
    ├─ Yes → What are you looking up?
    │         ├─ Existence check? → USE HASH SET ✓
    │         ├─ Key-value pairs? → USE HASH MAP ✓
    │         ├─ Frequencies? → USE HASH MAP (Counter) ✓
    │         └─ In sorted order? → Use TreeMap instead
    └─ No → What do you need?
              ├─ Sorted access? → Use BST/TreeMap
              ├─ Min/Max? → Use Heap
              └─ Sequential? → Use Array/List
    ```

=== "Implementation Templates"

    === "Template 1: Two Sum Pattern"

        **Use Case:** Find pairs, complements, matches

        **Pattern:** Store what you need to find, check if it exists

        ```python
        def two_sum(nums, target):
            """
            Find two numbers that sum to target.

            Perfect for: Pair finding, complement search

            Concept: For each number, calculate what we need (complement).
            Check if we've seen it before using hash map.

            Time: O(n) - Single pass through array
            Space: O(n) - Store up to n numbers in hash map

            Example: [2, 7, 11, 15], target=9 → [0, 1]
            """
            seen = {}  # value: index

            for i, num in enumerate(nums):
                # What number do we need to make target?
                complement = target - num

                # Have we seen this complement before?
                if complement in seen:
                    # Found it! Return both indices
                    return [seen[complement], i]

                # Haven't found pair yet, remember this number
                seen[num] = i

            # No valid pair found
            return []


        # Example Usage:
        nums = [2, 7, 11, 15]
        target = 9
        result = two_sum(nums, target)
        print(result)  # [0, 1] because nums[0] + nums[1] = 9
        ```

        **Key Points:**
        - Store "what we've seen" as we go
        - Check "what we need" in O(1)
        - One pass is enough!

    === "Template 2: Frequency Counter"

        **Use Case:** Count occurrences, find most/least common

        **Pattern:** Map each item to its count

        ```python
        from collections import Counter

        def frequency_analysis(arr):
            """
            Count frequency of each element.

            Perfect for: Histograms, statistics, mode finding

            Concept: Hash map where key=element, value=count
            Iterate once, increment counts

            Time: O(n) - Single pass
            Space: O(k) - k unique elements

            Example: [1,2,2,3,3,3] → {1:1, 2:2, 3:3}
            """
            # Method 1: Using Counter (recommended)
            freq = Counter(arr)

            # Method 2: Manual counting
            freq_manual = {}
            for num in arr:
                freq_manual[num] = freq_manual.get(num, 0) + 1

            return freq


        # Example Usage:
        arr = ['a', 'b', 'a', 'c', 'a', 'b']
        freq = frequency_analysis(arr)
        print(freq)  # Counter({'a': 3, 'b': 2, 'c': 1})

        # Find most common
        most_common = freq.most_common(2)
        print(most_common)  # [('a', 3), ('b', 2)]
        ```

        **Key Points:**
        - `Counter` from collections is optimized hash map
        - `dict.get(key, default)` safely increments
        - Perfect for "find k most frequent" problems

    === "Template 3: Grouping/Categorization"

        **Use Case:** Group items by property, collect by category

        **Pattern:** Map category to list of items

        ```python
        from collections import defaultdict

        def group_anagrams(strs):
            """
            Group strings that are anagrams.

            Perfect for: Grouping by property, categorization

            Concept: Use sorted string as key, map to all anagrams
            defaultdict auto-creates empty lists

            Time: O(n*k log k) - n strings, k = max length
            Space: O(n*k) - Store all strings

            Example: ["eat","tea","tan","ate","nat","bat"]
                  → [["eat","tea","ate"], ["tan","nat"], ["bat"]]
            """
            groups = defaultdict(list)

            for s in strs:
                # Anagrams have same sorted characters
                key = ''.join(sorted(s))
                groups[key].append(s)

            return list(groups.values())


        # Example Usage:
        strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
        result = group_anagrams(strs)
        print(result)
        # [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
        ```

        **Key Points:**
        - `defaultdict(list)` creates empty list if key not found
        - Choose right key (sorted chars for anagrams)
        - Returns groups as list of lists

    === "Template 4: Caching/Memoization"

        **Use Case:** Avoid recomputing expensive results

        **Pattern:** Store computed results, look up before computing

        ```python
        def fibonacci_memo(n, memo=None):
            """
            Fibonacci with memoization (caching).

            Perfect for: Recursive problems, expensive calculations

            Concept: Before computing, check if already cached
            If found, return cached result (O(1))
            If not, compute, cache, and return

            Time: O(n) instead of O(2^n)!
            Space: O(n) for cache

            Example: fib(5) without memo: 15 calls
                     fib(5) with memo: 5 calls
            """
            if memo is None:
                memo = {}

            # Already computed? Return cached value
            if n in memo:
                return memo[n]

            # Base cases
            if n <= 1:
                return n

            # Compute and cache result
            memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
            return memo[n]


        # Example Usage:
        print(fibonacci_memo(10))  # 55
        # Without memo: ~177 recursive calls
        # With memo: 10 recursive calls!
        ```

        **Key Points:**
        - Check cache before computing
        - Store result after computing
        - Transforms exponential to linear!

    === "Visual Walkthrough"

        **Problem:** Two Sum with [2, 7, 11, 15], target = 9

        ```
        Initial State:
        seen = {}
        target = 9

        Iteration 1: i=0, num=2
        ┌─────────────────────────────┐
        │ complement = 9 - 2 = 7      │
        │ Is 7 in seen? NO            │
        │ Store: seen[2] = 0          │
        └─────────────────────────────┘
        seen = {2: 0}

        Iteration 2: i=1, num=7
        ┌─────────────────────────────┐
        │ complement = 9 - 7 = 2      │
        │ Is 2 in seen? YES! ✓        │
        │ Return [seen[2], 1]         │
        │        [0, 1] ✓             │
        └─────────────────────────────┘

        Result: [0, 1]
        Why it works: We remember what we need (7),
        then find it when we see it!
        ```

        ---

        **Frequency Counting:** Count letters in "hello"

        ```
        Initial: freq = {}

        Process 'h':
        freq = {'h': 1}

        Process 'e':
        freq = {'h': 1, 'e': 1}

        Process first 'l':
        freq = {'h': 1, 'e': 1, 'l': 1}

        Process second 'l':
        freq = {'h': 1, 'e': 1, 'l': 2}  ← Increment existing

        Process 'o':
        freq = {'h': 1, 'e': 1, 'l': 2, 'o': 1}

        Final: {'h': 1, 'e': 1, 'l': 2, 'o': 1}
        Each character counted in O(1)!
        ```

        ---

        **Why Hash Maps Are Fast:**

        ```
        Array Search (without hash map):
        Find 42 in [10, 23, 16, 42, 8]
        Check index 0: 10 ≠ 42
        Check index 1: 23 ≠ 42
        Check index 2: 16 ≠ 42
        Check index 3: 42 = 42 ✓
        Operations: 4 comparisons (O(n))

        Hash Map (with hash map):
        seen = {10: 0, 23: 1, 16: 2, 42: 3, 8: 4}
        Is 42 in seen? → Hash 42 → Check index → YES ✓
        Operations: 1 lookup (O(1))

        10x speedup for size 10, 1000x for size 1000!
        ```

=== "Practice Problems"

    ## Learning Path

    ### Phase 1: Foundation (Easy)
    Master basic hash map operations and patterns.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Two Sum | Complement search | Classic two sum with hash map | [LeetCode 1](https://leetcode.com/problems/two-sum/) |
    | Contains Duplicate | Existence check | Use set for O(1) membership | [LeetCode 217](https://leetcode.com/problems/contains-duplicate/) |
    | Valid Anagram | Frequency counting | Compare character frequencies | [LeetCode 242](https://leetcode.com/problems/valid-anagram/) |
    | First Unique Character | Frequency + iteration | Count then find first unique | [LeetCode 387](https://leetcode.com/problems/first-unique-character-in-a-string/) |
    | Intersection of Two Arrays | Set operations | Use sets for intersection | [LeetCode 349](https://leetcode.com/problems/intersection-of-two-arrays/) |

    **Goal:** Solve all 5 problems. Get comfortable with hash map basics.

    ---

    ### Phase 2: Application (Medium)
    Apply hash maps to complex scenarios and combinations.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Group Anagrams | Grouping with key | Map sorted string to anagrams | [LeetCode 49](https://leetcode.com/problems/group-anagrams/) |
    | Top K Frequent Elements | Frequency + sorting | Combine Counter with heap | [LeetCode 347](https://leetcode.com/problems/top-k-frequent-elements/) |
    | Longest Consecutive Sequence | Set for O(n) | Use set to find sequence starts | [LeetCode 128](https://leetcode.com/problems/longest-consecutive-sequence/) |
    | Subarray Sum Equals K | Prefix sum + hash | Map cumulative sum to count | [LeetCode 560](https://leetcode.com/problems/subarray-sum-equals-k/) |
    | 4Sum II | Multiple hash maps | Store sums in map, lookup complements | [LeetCode 454](https://leetcode.com/problems/4sum-ii/) |

    **Goal:** Solve 3 out of 5. Learn to combine hash maps with other techniques.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex caching, multiple maps, and advanced patterns.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | LRU Cache | Caching with ordering | OrderedDict or custom doubly-linked list | [LeetCode 146](https://leetcode.com/problems/lru-cache/) |
    | First Missing Positive | In-place hashing | Array as hash map | [LeetCode 41](https://leetcode.com/problems/first-missing-positive/) |
    | Substring with Concatenation | Sliding window + hash | Complex pattern matching | [LeetCode 30](https://leetcode.com/problems/substring-with-concatenation-of-all-words/) |
    | Longest Duplicate Substring | Rolling hash | Advanced hashing technique | [LeetCode 1044](https://leetcode.com/problems/longest-duplicate-substring/) |

    **Goal:** Solve 2 out of 4. Master advanced hash map applications.

    ---

    ## Practice Strategy

    1. **Start with Two Sum:** The quintessential hash map problem
    2. **Identify What to Store:** Key decision—what goes in the map?
    3. **Choose Right Structure:** Dict? Set? Counter? DefaultDict?
    4. **Draw the Hash Map State:** Visualize at each iteration
    5. **Consider Edge Cases:** Empty input, duplicates, no solution
    6. **Optimize Space:** Sometimes set is better than full map
    7. **Time Yourself:** <10min for easy, <25min for medium

    ---

    ## Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Forgetting to check existence** | Assume key exists, get KeyError | Use `if key in dict:` or `dict.get(key, default)` |
    | **Modifying during iteration** | Changing dict while looping | Create list of keys first: `for key in list(dict.keys())` |
    | **Using list as key** | Lists are unhashable | Convert to tuple: `tuple(my_list)` |
    | **Not considering collisions** | Assume perfect hashing | Python handles it, but be aware in interviews |
    | **Overusing hash maps** | Use when not needed | For 5 elements, simple loop might be clearer |
    | **Wrong data structure** | Using dict when set suffices | Set for membership, dict for key-value |
    | **Not initializing defaultdict type** | `defaultdict()` without argument | Specify type: `defaultdict(int)` or `defaultdict(list)` |

---

### Dynamic Programming

=== "Understanding the Pattern"

    ## 📖 What is Dynamic Programming?

    Imagine you're climbing a staircase to reach the 100th step. You can take either 1 step or 2 steps at a time. How many different ways can you reach the top? The naive approach would be to explore EVERY possible path—but that's millions of calculations! Here's the insight: **to reach step 100, you must have come from either step 99 or step 98**. So:

    `ways(100) = ways(99) + ways(98)`

    Instead of recalculating ways(99) thousands of times, calculate it once and remember the answer. This is **Dynamic Programming (DP)**—the art of breaking complex problems into simpler subproblems, solving each subproblem just once, and storing the results for reuse.

    **The Magic:** DP transforms problems that would take exponential time (O(2ⁿ) or O(n!) — billions of years for large inputs) into polynomial time (O(n) or O(n²) — fractions of a second). It's one of the most powerful optimization techniques in computer science!

    Real-world applications: route optimization in GPS, video compression, gene sequence alignment, stock trading strategies, and even autocorrect on your phone!

    ---

    ## 🔧 How It Works

    Dynamic Programming works through two key properties:

    **1. Overlapping Subproblems**
    ```
    Fibonacci without DP: fib(5)
                          /        \
                     fib(4)        fib(3)
                    /     \        /     \
                fib(3)  fib(2)  fib(2)  fib(1)
                /   \    /   \    /   \
            fib(2) fib(1) ...

    Notice: fib(3) calculated 2 times, fib(2) calculated 3 times!
    Total calls for fib(50): 2^50 = 1,125,899,906,842,624 (over a quadrillion!)

    With DP Memoization:
    Calculate fib(3) once → store result → reuse
    Total calls for fib(50): just 50 (calculate each value once)
    ```

    **2. Optimal Substructure**
    ```
    Problem: Minimum cost to reach last step
    stairs = [10, 15, 20]
    cost[0] = 0

    To reach step i with minimum cost:
    cost[i] = stairs[i] + min(cost[i-1], cost[i-2])

    Optimal solution uses optimal solutions of subproblems!
    ```

    **Two Approaches:**

    **Top-Down (Memoization):** Start from problem, recursively break down, cache results
    ```python
    memo = {}
    def fib(n):
        if n in memo:
            return memo[n]  # Reuse!
        if n <= 1:
            return n
        memo[n] = fib(n-1) + fib(n-2)  # Store!
        return memo[n]
    ```

    **Bottom-Up (Tabulation):** Start from base cases, build up to answer
    ```python
    def fib(n):
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]  # Build from base!
        return dp[n]
    ```

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Don't solve the same problem twice—remember and reuse!

    Think of dynamic programming like taking notes in class:
    - **Without notes (no DP):** Re-derive every formula from scratch every time you need it
    - **With notes (DP):** Write formula down once, look it up whenever needed

    **Real-World Analogy: Map Navigation**
    ```
    Finding shortest path from A to Z through cities:

    Without DP (Brute Force):
    - Try EVERY possible route: A→B→C→Z, A→B→D→Z, A→C→B→Z...
    - Recalculate distance A→B→C many times
    - Exponential: 1000 cities = impossible

    With DP:
    - Calculate shortest path A→B once, store it
    - Calculate shortest path A→C once, store it
    - Build up: shortest A→Z = min(path_A→B + path_B→Z, path_A→C + path_C→Z)
    - Polynomial: 1000 cities = easy!
    ```

    **The Core Trade-off:**
    - Sacrifice: Space (memory to store subproblem results)
    - Gain: Time (exponential → polynomial reduction)

    This trade-off is almost always worth it—memory is cheap, time is precious!

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n) to O(n³) | Depends on problem dimensions (1D, 2D, 3D arrays) |
    | **Space** | O(n) to O(n²) | Store subproblem results (often optimizable to O(1)) |
    | **Improvement** | From O(2ⁿ) or O(n!) | Eliminate redundant recalculations |

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Dynamic Programming Works | Example |
    |----------|------------------------------|---------|
    | **Counting ways** | Sum of all paths/combinations | Ways to climb stairs, decode messages |
    | **Optimization** | Find minimum/maximum | Minimum path sum, maximum profit |
    | **Overlapping subproblems** | Same calculation repeated many times | Fibonacci, factorial problems |
    | **Sequence problems** | Build solution from previous elements | Longest increasing subsequence |
    | **Grid traversal** | Combine paths from neighbors | Unique paths, dungeon game |
    | **Knapsack variants** | Include/exclude decisions | Subset sum, partition problems |

    **Red Flags That Suggest Dynamic Programming:**
    - "Count all ways to..."
    - "Find minimum/maximum..."
    - "Longest/shortest sequence..."
    - "Can you reach..." (with choices at each step)
    - Problem has recursive structure with repeated subproblems
    - Brute force is exponential but has overlapping subproblems

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Greedy works** | Local optimal = global optimal | Greedy algorithm (simpler & faster) |
    | **No overlapping subproblems** | Each subproblem unique | Regular recursion or iteration |
    | **Need all paths independently** | Can't reuse subproblem solutions | Backtracking (must explore all) |
    | **Space constraints** | Can't afford O(n) or O(n²) memory | Greedy or space-optimized approaches |
    | **Subproblems don't build on each other** | No optimal substructure | Different algorithmic paradigm |

    ---

    ## 🎯 Decision Flowchart

    ```
    Can problem be broken into subproblems?
    ├─ Yes → Are subproblems overlapping (recalculated multiple times)?
    │         ├─ Yes → Do optimal solutions use optimal subproblem solutions?
    │         │        ├─ Yes → USE DYNAMIC PROGRAMMING ✓
    │         │        └─ No → Consider backtracking
    │         └─ No → Does greedy choice work?
    │                  ├─ Yes → Use greedy
    │                  └─ No → Use divide & conquer
    └─ No → Different approach needed
    ```

=== "Implementation Templates"

    === "Template 1: Top-Down (Memoization)"

        **Use Case:** Natural recursive thinking, explore from problem to base cases

        **Pattern:** Write recursive solution, add caching to eliminate redundant calls

        ```python
        def top_down_dp(n, memo=None):
            """
            Top-down DP with memoization (recursive).

            Perfect for: Problems naturally expressed recursively

            Concept:
            - Write solution as if solving from scratch
            - Before computing, check if already cached
            - After computing, store in cache
            - Recursive calls automatically reuse cached values

            Time: O(n) - Each subproblem computed once
            Space: O(n) - Memo dict + recursion stack

            Example: Fibonacci, decode ways, climbing stairs
            """
            # Initialize memo on first call
            if memo is None:
                memo = {}

            # Base case: already computed
            if n in memo:
                return memo[n]

            # Base cases: smallest subproblems
            if n <= 1:
                return n

            # Recursive case: break into subproblems
            result = top_down_dp(n - 1, memo) + top_down_dp(n - 2, memo)

            # Cache result before returning
            memo[n] = result
            return result

        # Example Usage: Fibonacci
        print(top_down_dp(50))  # Instant! (without DP: would take forever)
        ```

        **Key Points:**
        - Always check cache before computing
        - Cache before returning (not after)
        - Pass memo through recursive calls or use global/nonlocal
        - Watch for mutable default argument issue (use `memo=None`)

    === "Template 2: Bottom-Up (Tabulation)"

        **Use Case:** Build solution iteratively from smallest subproblems to answer

        **Pattern:** Create DP table, fill base cases, iterate to build solution

        ```python
        def bottom_up_dp(n):
            """
            Bottom-up DP with tabulation (iterative).

            Perfect for: Clear iterative build-up, avoiding recursion overhead

            Concept:
            - Create array/table for all subproblems
            - Fill base cases
            - Iteratively compute each state from previous states
            - Final answer is at dp[n]

            Time: O(n) - Single pass through states
            Space: O(n) - DP table

            Example: Climbing stairs, coin change, house robber
            """
            # Edge case
            if n <= 1:
                return n

            # Create DP table
            dp = [0] * (n + 1)

            # Base cases
            dp[0] = 0
            dp[1] = 1

            # Fill table iteratively
            for i in range(2, n + 1):
                # State transition: combine previous states
                dp[i] = dp[i - 1] + dp[i - 2]

            # Answer is at final position
            return dp[n]

        # Example Usage: Fibonacci
        print(bottom_up_dp(50))  # Same result as top-down
        ```

        **Key Points:**
        - Initialize DP array with appropriate size
        - Set base cases explicitly
        - Iterate in correct order (dependencies must be computed first)
        - Often more space-efficient and faster than top-down

    === "Template 3: Space-Optimized DP"

        **Use Case:** When only need previous k states, not entire history

        **Pattern:** Use variables or sliding window instead of full array

        ```python
        def space_optimized_dp(n):
            """
            Space-optimized DP (O(1) space).

            Perfect for: When each state only depends on last few states

            Concept:
            - Only store states currently needed
            - Sliding window of previous k values
            - Update window as you progress
            - Reduces space from O(n) to O(k)

            Time: O(n) - Still need to compute all states
            Space: O(1) - Constant space (just 2 variables)

            Example: Fibonacci, house robber, climbing stairs
            """
            if n <= 1:
                return n

            # Only keep last 2 values (not entire array)
            prev2, prev1 = 0, 1

            # Build solution with sliding window
            for i in range(2, n + 1):
                current = prev1 + prev2

                # Slide window: drop oldest, keep newer
                prev2 = prev1
                prev1 = current

            return prev1

        # Example Usage: Fibonacci
        print(space_optimized_dp(1000))  # Can handle huge n!
        ```

        **Key Points:**
        - Identify how many previous states needed (usually 1-3)
        - Use descriptive names: prev1, prev2 (not a, b)
        - Update variables in correct order (don't overwrite before using)
        - Dramatic space reduction: O(n) → O(1)

    === "Template 4: 2D DP (Grid/Sequence)"

        **Use Case:** Problems with two dimensions (grids, comparing sequences)

        **Pattern:** Create 2D table where dp[i][j] combines adjacent cells

        ```python
        def two_d_dp(text1, text2):
            """
            2D DP for sequence comparison problems.

            Perfect for: Grid traversal, comparing two sequences

            Concept:
            - dp[i][j] = solution for first i elements of text1, first j of text2
            - Fill row by row or column by column
            - Each cell computed from neighbors (up, left, diagonal)

            Time: O(m * n) - Fill entire table
            Space: O(m * n) - 2D array

            Example: Longest Common Subsequence, Edit Distance, Grid paths
            """
            m, n = len(text1), len(text2)

            # Create 2D table (often with extra row/col for base case)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            # Base cases (first row and column) already 0 by initialization

            # Fill table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if text1[i - 1] == text2[j - 1]:
                        # Match: take diagonal + 1
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        # No match: take best of neighbors
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            # Answer typically at bottom-right corner
            return dp[m][n]

        # Example Usage: Longest Common Subsequence
        text1 = "abcde"
        text2 = "ace"
        print(two_d_dp(text1, text2))  # Output: 3 ("ace")
        ```

        **Key Points:**
        - Extra row/column for base cases (size (m+1) x (n+1))
        - Iterate i and j from 1 to include all elements
        - Access text with [i-1] and [j-1] due to offset
        - Can often optimize to O(n) space using rolling array

    === "Visual Walkthrough"

        **Problem:** Climbing Stairs - How many ways to reach step 5? (can take 1 or 2 steps)

        ```
        Step-by-Step Computation:

        Base Cases:
        step 0: 1 way (stay at bottom)
        step 1: 1 way (take 1 step)

        Build up:
        step 2: ways(1) + ways(0) = 1 + 1 = 2
                [1,1] or [2]

        step 3: ways(2) + ways(1) = 2 + 1 = 3
                [1,1,1] or [1,2] or [2,1]

        step 4: ways(3) + ways(2) = 3 + 2 = 5
                [1,1,1,1] or [1,1,2] or [1,2,1] or [2,1,1] or [2,2]

        step 5: ways(4) + ways(3) = 5 + 3 = 8

        DP Table:
        i:       0   1   2   3   4   5
        dp[i]:   1   1   2   3   5   8  ← Answer!

        Time saved:
        Without DP: 2^5 = 32 recursive calls (many redundant)
        With DP: 5 iterations (each step computed once)
        ```

        **Why This Works:**
        To reach step n, you MUST be at either step n-1 or n-2 just before.
        So total ways = ways_to_reach(n-1) + ways_to_reach(n-2)

        ---

        **2D DP Example:** Longest Common Subsequence of "ace" and "abcde"

        ```
        DP Table Construction:

            ""  a   b   c   d   e
        ""   0   0   0   0   0   0
        a    0   1   1   1   1   1
        c    0   1   1   2   2   2
        e    0   1   1   2   2   3  ← Answer at [3][5]

        How to fill each cell:
        - If chars match: dp[i][j] = dp[i-1][j-1] + 1
        - If no match: dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        Step-by-step for dp[2][3] (comparing "ac" vs "abc"):
        - text1[1]='c', text2[2]='c' → MATCH!
        - dp[2][3] = dp[1][2] + 1 = 1 + 1 = 2

        Common subsequence: "ace" (length 3)
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic DP mechanics with 1D problems.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Climbing Stairs | Linear 1D | Classic DP introduction | [LeetCode 70](https://leetcode.com/problems/climbing-stairs/) |
    | Fibonacci Number | Linear 1D | Memoization basics | [LeetCode 509](https://leetcode.com/problems/fibonacci-number/) |
    | Min Cost Climbing Stairs | Linear 1D with cost | State transition with weights | [LeetCode 746](https://leetcode.com/problems/min-cost-climbing-stairs/) |
    | Divisor Game | Simple decision DP | True/false states | [LeetCode 1025](https://leetcode.com/problems/divisor-game/) |
    | House Robber | Linear DP | Include/exclude pattern | [LeetCode 198](https://leetcode.com/problems/house-robber/) |

    **Goal:** Solve all 5 problems. Understand state definition and transitions.

    ---

    ### Phase 2: Application (Medium)
    Apply DP to 2D problems and more complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Unique Paths | 2D Grid | Basic grid DP | [LeetCode 62](https://leetcode.com/problems/unique-paths/) |
    | Coin Change | Unbounded knapsack | Minimize coins | [LeetCode 322](https://leetcode.com/problems/coin-change/) |
    | Longest Increasing Subsequence | 1D sequence | O(n²) to O(n log n) optimization | [LeetCode 300](https://leetcode.com/problems/longest-increasing-subsequence/) |
    | Longest Common Subsequence | 2D sequence | Comparing two strings | [LeetCode 1143](https://leetcode.com/problems/longest-common-subsequence/) |
    | Word Break | String DP | Dictionary matching | [LeetCode 139](https://leetcode.com/problems/word-break/) |

    **Goal:** Solve 4 out of 5. Master 2D DP and optimization patterns.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex state transitions and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Edit Distance | 2D sequence | Three operations (insert/delete/replace) | [LeetCode 72](https://leetcode.com/problems/edit-distance/) |
    | Regular Expression Matching | 2D pattern matching | Complex state transitions | [LeetCode 10](https://leetcode.com/problems/regular-expression-matching/) |
    | Burst Balloons | Interval DP | Divide and conquer DP | [LeetCode 312](https://leetcode.com/problems/burst-balloons/) |
    | Maximal Rectangle | DP + stack | 2D optimization | [LeetCode 85](https://leetcode.com/problems/maximal-rectangle/) |

    **Goal:** Solve 2 out of 4. Master advanced DP techniques and optimizations.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Easy:** Build confidence with classic 1D DP (Fibonacci, Climbing Stairs)
    2. **Identify the State:** What does dp[i] or dp[i][j] represent?
    3. **Find the Recurrence:** How does dp[i] relate to previous states?
    4. **Draw It Out:** Create small example (n=5) and fill DP table by hand
    5. **Code Without Looking:** Try bottom-up first, then optimize space
    6. **Time Yourself:** <20 minutes per easy, <35 for medium, <50 for hard

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Wrong state definition** | Unclear what dp[i] represents | Write clear comment: "dp[i] = minimum cost to reach step i" |
    | **Incorrect initialization** | Base cases not set properly | Manually verify dp[0], dp[1] before loop |
    | **Off-by-one errors** | Confusion with array sizes | Use dp[n+1] for problems with n elements |
    | **Wrong iteration order** | Computing before dependencies ready | Ensure i-1 computed before i (bottom-up) |
    | **Mutable default argument** | memo={} as default parameter | Use memo=None, then memo = {} inside function |
    | **Not considering all transitions** | Missing a case in recurrence | List all ways to reach state, ensure all covered |
    | **Stack overflow in recursion** | Deep recursion without memoization | Use bottom-up or increase recursion limit |

---

### Greedy Algorithm

=== "Understanding the Pattern"

    ## 📖 What is a Greedy Algorithm?

    Imagine you're hiking up a mountain in thick fog. You can only see a few feet ahead, but you need to reach the summit. Your strategy? At each step, always move in the direction that goes UP the most. Pick the locally best option right now, hoping it leads to the top. Sometimes this works perfectly! Other times... you end up on a local peak, not the summit.

    This is the essence of **Greedy Algorithms**: making the locally optimal choice at each step, hoping these choices lead to a globally optimal solution. The beauty? When it works, it's often the simplest and fastest solution—just sort, then scan! No complex recursion, no massive DP tables.

    **The Catch:** Greedy doesn't always work. You need to prove (or verify through counterexamples) that local optimal choices actually lead to global optimum. But when they do? It's elegant, efficient, and easy to code!

    Real-world applications: Huffman coding (data compression), Dijkstra's algorithm (GPS routing), activity scheduling, resource allocation, and making change with coins.

    ---

    ## 🔧 How It Works

    Greedy algorithms follow a simple pattern:

    **The Greedy Choice Property:**
    ```
    At each step, make the choice that looks best RIGHT NOW
    Never reconsider or backtrack
    Hope the sequence of local choices leads to global optimum

    Example: Activity Selection
    Activities: [(1,4), (3,5), (0,6), (5,7), (8,9), (5,9)]

    Greedy Choice: Pick activity that ENDS EARLIEST
    Why? Leaves most time for remaining activities

    Step 1: Sort by end time: [(1,4), (3,5), (0,6), (5,7), (8,9), (5,9)]
    Step 2: Pick (1,4) → blocks nothing after
    Step 3: Pick (5,7) → starts after (1,4) ends
    Step 4: Pick (8,9) → starts after (5,7) ends

    Result: 3 activities (optimal!)
    ```

    **Common Greedy Strategies:**
    1. **Sort** by some criterion (end time, value, deadline, etc.)
    2. **Scan** through sorted list once
    3. **Make choice** based on greedy rule
    4. **Update** state based on choice
    5. **Repeat** until done

    **Why So Fast?**
    - Sorting: O(n log n)
    - Single scan: O(n)
    - Total: O(n log n) typically
    - No recursion, no memoization, no backtracking!

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Sometimes the best overall strategy is to always do what's best right now!

    Think of these real-world analogies:

    **Making Change:**
    ```
    Amount: $0.67
    Coins: [25¢, 10¢, 5¢, 1¢]

    Greedy: Always use largest coin possible
    - 2 quarters (50¢) → $0.17 left
    - 1 dime (10¢) → $0.07 left
    - 1 nickel (5¢) → $0.02 left
    - 2 pennies (2¢) → Done!

    Result: 6 coins (optimal for US currency!)

    BUT... with coins [25¢, 10¢, 1¢] for amount 30¢:
    Greedy: 25¢ + 5×1¢ = 6 coins
    Optimal: 3×10¢ = 3 coins
    Greedy FAILS! (need DP instead)
    ```

    **Job Scheduling:**
    ```
    Jobs with deadlines and profits
    Greedy: Always pick highest profit job that fits deadline
    Result: Maximum profit (provably optimal!)
    ```

    **When Greedy Works:**
    1. **Greedy Choice Property**: Local optimum → Global optimum
    2. **Optimal Substructure**: Optimal solution contains optimal subsolutions
    3. **No dependencies**: Choices don't affect each other's validity

    **When Greedy Fails:**
    - Future choices constrained by current choice in bad ways
    - Need to consider multiple steps ahead
    - Optimal solution requires "worse" choices early on

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n log n) | Sorting dominates; scan is O(n) |
    | **Space** | O(1) to O(n) | Usually O(1) except sorting space |
    | **Improvement** | From O(2ⁿ) or O(n²) | Avoids trying all combinations or DP |

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Greedy Works | Example |
    |----------|-----------------|---------|
    | **Activity/Interval selection** | Earliest end time leaves most options | Meeting rooms, task scheduling |
    | **Huffman coding** | Least frequent first minimizes encoding | Data compression |
    | **Fractional knapsack** | Highest value/weight ratio maximizes | Resource allocation |
    | **Dijkstra's shortest path** | Closest unvisited node next | GPS routing, network routing |
    | **Minimum spanning tree** | Smallest edge that doesn't create cycle | Network design (Kruskal's, Prim's) |
    | **Jump game** | Track furthest reachable | Determine if goal reachable |

    **Red Flags That Suggest Greedy:**
    - "Maximize" or "minimize" something
    - "Earliest", "latest", "shortest", "longest" hints at sorting criterion
    - Interval/scheduling problems
    - Can prove local optimal → global optimal
    - Problem has natural sorting order

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Coin change (arbitrary denominations)** | Local choice blocks better global solution | Dynamic Programming |
    | **0/1 Knapsack** | Can't take fractions; greedy by value fails | Dynamic Programming |
    | **Longest increasing subsequence** | Need to track all possibilities | Dynamic Programming O(n²) or O(n log n) |
    | **Traveling salesman** | Local shortest edge doesn't give shortest path | Approximation algorithms, DP |
    | **Graph coloring** | Greedy gives suboptimal coloring | Backtracking or heuristics |

    ---

    ## 🎯 Decision Flowchart

    ```
    Is there a natural greedy choice?
    ├─ Yes → Can you prove/verify local optimal → global optimal?
    │         ├─ Yes → USE GREEDY ✓
    │         └─ No → Try DP or backtracking
    └─ No → Does problem have overlapping subproblems?
              ├─ Yes → Use Dynamic Programming
              └─ No → Use different approach
    ```

=== "Implementation Templates"

    === "Template 1: Activity Selection"

        **Use Case:** Maximize number of non-overlapping intervals

        **Pattern:** Sort by end time, greedily pick earliest ending activities

        ```python
        def max_activities(intervals):
            """
            Maximum non-overlapping activities (Interval Scheduling).

            Perfect for: Meeting rooms, task scheduling

            Concept:
            - Sort by end time (finish earliest)
            - Greedy choice: Pick activity that ends soonest
            - Why? Leaves maximum time for remaining activities

            Time: O(n log n) - Sorting dominates
            Space: O(1) - In-place selection

            Example: [[1,3], [2,4], [3,5]] → 2 activities max
            """
            if not intervals:
                return 0

            # Greedy strategy: Sort by END time
            # Why? Finishing early leaves more room for others
            intervals.sort(key=lambda x: x[1])

            count = 1  # Always pick first activity
            last_end = intervals[0][1]

            # Scan through remaining activities
            for start, end in intervals[1:]:
                # Greedy choice: Pick if non-overlapping
                if start >= last_end:
                    count += 1
                    last_end = end  # Update end time

            return count

        # Example Usage:
        activities = [[1,3], [2,5], [4,6], [6,7], [5,8], [8,9]]
        result = max_activities(activities)
        print(result)  # Output: 4 (max non-overlapping)
        ```

        **Key Points:**
        - Sort by END time (not start time!)
        - Always pick first after sorting (earliest end)
        - Check `start >= last_end` for non-overlap
        - Provably optimal (greedy exchange argument)

    === "Template 2: Jump Game"

        **Use Case:** Determine if can reach end of array with jumps

        **Pattern:** Track maximum reachable index greedily

        ```python
        def can_jump(nums):
            """
            Can reach last index with variable jumps?

            Perfect for: Reachability problems with choices

            Concept:
            - Track furthest index reachable so far
            - At each position, update max reachable
            - If current position > max reachable, stuck!

            Time: O(n) - Single pass
            Space: O(1) - Just one variable

            Example: [2,3,1,1,4] → True (0→1→4)
                     [3,2,1,0,4] → False (stuck at index 3)
            """
            max_reach = 0  # Furthest index we can reach

            for i in range(len(nums)):
                # If current position unreachable, fail
                if i > max_reach:
                    return False

                # Greedy choice: Update max reachable from here
                max_reach = max(max_reach, i + nums[i])

                # Early termination: Can reach end
                if max_reach >= len(nums) - 1:
                    return True

            return True  # Reached end of loop

        # Example Usage:
        nums = [2, 3, 1, 1, 4]
        result = can_jump(nums)
        print(result)  # True
        ```

        **Key Points:**
        - Don't need to track actual path (just reachability)
        - Greedy: always extend reach as far as possible
        - Check if stuck (i > max_reach) before updating
        - Early termination when end is reachable

    === "Template 3: Gas Station (Circular)"

        **Use Case:** Find starting point for circular journey

        **Pattern:** Track running balance, reset start when balance goes negative

        ```python
        def gas_station(gas, cost):
            """
            Find starting gas station for complete circular trip.

            Perfect for: Circular array problems with balance tracking

            Concept:
            - If total gas < total cost, impossible
            - Start at station 0, track fuel balance
            - When balance < 0, can't start from any earlier station
            - Greedy: Try starting from next station

            Time: O(n) - Single pass
            Space: O(1) - Constant space

            Example: gas=[1,2,3,4,5], cost=[3,4,5,1,2] → Start at 3
            """
            total_gas = sum(gas)
            total_cost = sum(cost)

            # Impossible if not enough total gas
            if total_gas < total_cost:
                return -1

            start = 0
            tank = 0

            # Single pass to find valid start
            for i in range(len(gas)):
                tank += gas[i] - cost[i]

                # Greedy choice: If balance negative,
                # can't start from 'start' or any station before i
                if tank < 0:
                    start = i + 1  # Try starting from next
                    tank = 0       # Reset balance

            return start  # Guaranteed to work (total_gas >= total_cost)

        # Example Usage:
        gas = [1, 2, 3, 4, 5]
        cost = [3, 4, 5, 1, 2]
        result = gas_station(gas, cost)
        print(result)  # Output: 3
        ```

        **Key Points:**
        - Check total_gas >= total_cost first (necessary condition)
        - If balance goes negative, NONE of the previous stations work as start
        - Greedy insight: reset start to next station
        - Single pass sufficient (don't need to simulate from each start)

    === "Template 4: Two-Pass Greedy (Candy)"

        **Use Case:** Satisfy constraints from both directions

        **Pattern:** Make two greedy passes (left-to-right, then right-to-left)

        ```python
        def distribute_candy(ratings):
            """
            Minimum candies to give children with rating constraints.

            Perfect for: Two-directional constraint satisfaction

            Concept:
            - Each child must have >= 1 candy
            - Higher rating than neighbor → more candy than that neighbor
            - Pass 1: Satisfy left neighbor constraint
            - Pass 2: Satisfy right neighbor constraint

            Time: O(n) - Two passes
            Space: O(n) - Candy array

            Example: ratings=[1,0,2] → candies=[2,1,2] (total=5)
            """
            n = len(ratings)
            candies = [1] * n  # Everyone gets at least 1

            # Pass 1: Left to right (satisfy left neighbor)
            for i in range(1, n):
                if ratings[i] > ratings[i - 1]:
                    candies[i] = candies[i - 1] + 1

            # Pass 2: Right to left (satisfy right neighbor)
            for i in range(n - 2, -1, -1):
                if ratings[i] > ratings[i + 1]:
                    # Take max to satisfy BOTH neighbors
                    candies[i] = max(candies[i], candies[i + 1] + 1)

            return sum(candies)

        # Example Usage:
        ratings = [1, 0, 2]
        result = distribute_candy(ratings)
        print(result)  # Output: 5 (candies=[2,1,2])
        ```

        **Key Points:**
        - Initialize all to 1 (minimum constraint)
        - First pass handles increasing sequences left-to-right
        - Second pass handles increasing sequences right-to-left
        - Use max() to satisfy both directions simultaneously

    === "Visual Walkthrough"

        **Problem:** Activity Selection - [[1,4], [3,5], [0,6], [5,7], [8,9], [5,9]]

        ```
        Step 1: Sort by END time
        [(1,4), (3,5), (0,6), (5,7), (5,9), (8,9)]

        Timeline:
        0    2    4    6    8    10
        |────────|              (0,6)
             |──|               (1,4) ← Pick first!
                  |─|           (3,5)
                      |─|       (5,7) ← Pick! (starts at 4 ≥ 4)
                      |───|     (5,9)
                            |─| (8,9) ← Pick! (starts at 8 ≥ 7)

        Step 2: Pick (1,4)
        last_end = 4, count = 1

        Step 3: Check (3,5)
        3 < 4? Yes, skip (overlaps)

        Step 4: Check (0,6)
        0 < 4? Yes, skip (overlaps)

        Step 5: Check (5,7)
        5 >= 4? Yes, pick!
        last_end = 7, count = 2

        Step 6: Check (5,9)
        5 >= 7? No, skip

        Step 7: Check (8,9)
        8 >= 7? Yes, pick!
        last_end = 9, count = 3

        Result: 3 activities maximum
        ```

        **Why Earliest End Works:**
        Picking earliest-ending activity leaves maximum time for remaining activities. Any other choice would end later, blocking more future activities.

        ---

        **Jump Game Example:** [2, 3, 1, 1, 4]

        ```
        Index:      0  1  2  3  4
        Value:      2  3  1  1  4
        Jump:      →→ →→→ → →

        Step 0: i=0, max_reach=0
        Update: max_reach = max(0, 0+2) = 2

        Step 1: i=1, max_reach=2
        1 <= 2? Yes, reachable
        Update: max_reach = max(2, 1+3) = 4 ← Can reach end!

        Early termination: 4 >= 4 (last index)
        Result: True

        Greedy insight: Don't track actual path, just furthest reach!
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic greedy mechanics.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Assign Cookies | Greedy matching | Sort both arrays, match greedily | [LeetCode 455](https://leetcode.com/problems/assign-cookies/) |
    | Lemonade Change | Sequential greedy | Make greedy choices with constraints | [LeetCode 860](https://leetcode.com/problems/lemonade-change/) |
    | Best Time to Buy/Sell Stock | Single-pass greedy | Track minimum so far | [LeetCode 121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) |
    | Maximum Subarray | Kadane's algorithm | Greedy substring selection | [LeetCode 53](https://leetcode.com/problems/maximum-subarray/) |
    | Remove K Digits | Monotonic stack greedy | Remove digits to minimize | [LeetCode 402](https://leetcode.com/problems/remove-k-digits/) |

    **Goal:** Solve all 5 problems. Understand greedy choice at each step.

    ---

    ### Phase 2: Application (Medium)
    Apply greedy to scheduling and optimization.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Jump Game | Reachability greedy | Track maximum reachable index | [LeetCode 55](https://leetcode.com/problems/jump-game/) |
    | Jump Game II | Minimum jumps | BFS-like greedy (minimum steps) | [LeetCode 45](https://leetcode.com/problems/jump-game-ii/) |
    | Gas Station | Circular array | Balance tracking, reset start | [LeetCode 134](https://leetcode.com/problems/gas-station/) |
    | Task Scheduler | Greedy scheduling | Maximize idle time usage | [LeetCode 621](https://leetcode.com/problems/task-scheduler/) |
    | Partition Labels | Interval merging | Track last occurrence greedily | [LeetCode 763](https://leetcode.com/problems/partition-labels/) |

    **Goal:** Solve 4 out of 5. Learn to identify greedy choice property.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex greedy with proofs and two-pass techniques.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Candy | Two-pass greedy | Satisfy constraints from both directions | [LeetCode 135](https://leetcode.com/problems/candy/) |
    | Minimum Refueling Stops | Greedy + heap | Maximize reach with priority queue | [LeetCode 871](https://leetcode.com/problems/minimum-number-of-refueling-stops/) |
    | Remove Duplicate Letters | Greedy + stack | Lexicographically smallest result | [LeetCode 316](https://leetcode.com/problems/remove-duplicate-letters/) |
    | Merge Triplets | Greedy selection | Include/exclude with constraints | [LeetCode 1899](https://leetcode.com/problems/merge-triplets-to-form-target-triplet/) |

    **Goal:** Solve 2 out of 4. Master proving greedy correctness.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Easy:** Build intuition with simple greedy choices
    2. **Identify the Greedy Choice:** What should we optimize at each step?
    3. **Verify Correctness:** Can you prove or check that local optimal → global optimal?
    4. **Draw It Out:** Visualize with timeline or graph
    5. **Code Without Looking:** Greedy is usually simple—one loop!
    6. **Time Yourself:** <10 minutes per easy, <25 for medium, <40 for hard

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Sorting by wrong criterion** | Not identifying correct greedy choice | Activity selection: sort by END not start |
    | **Assuming greedy always works** | Not verifying greedy choice property | Test with small counterexamples first |
    | **Not updating state** | Forget to update after greedy choice | Track last_end, max_reach, or balance |
    | **Missing edge cases** | Empty input, single element | Check n==0 or n==1 explicitly |
    | **Overcomplicating** | Adding unnecessary checks | Greedy is usually simple—resist overthinking! |

---

### Backtracking

=== "Understanding the Pattern"

    ## 📖 What is Backtracking?

    Imagine you're in a corn maze trying to find the exit. At each intersection, you pick a path. If you hit a dead end, you walk back to the last decision point and try a different path. You keep track of where you've been so you don't repeat mistakes. Eventually, by systematically trying every possibility, you either find the exit or prove it doesn't exist!

    This is **Backtracking**: an algorithmic technique for solving problems by trying to build solutions incrementally, abandoning a solution ("backtracking") as soon as you determine it cannot possibly lead to a valid complete solution. It's like playing chess several moves ahead: "If I move here, then they move there... oh that leads to checkmate against me—let me undo and try something else!"

    **The Core Idea:** Explore all possible paths through a decision tree, pruning branches that violate constraints. It's essentially DFS (Depth-First Search) with the added superpower of **undoing choices** when they lead to dead ends.

    Real-world applications: solving Sudoku, N-Queens puzzle, generating all possible passwords, pathfinding with obstacles, scheduling under constraints, and even protein folding simulations!

    ---

    ## 🔧 How It Works

    Backtracking follows a simple recursive pattern:

    **The Backtracking Template:**
    ```
    1. Make a CHOICE (pick an option)
    2. EXPLORE (recursively solve the rest)
    3. BACKTRACK (undo the choice)
    4. Repeat for all choices
    ```

    **Visual Example: Generate Subsets of [1, 2, 3]**
    ```
    Decision Tree (Include or Exclude each element):

                         []
                    /          \
                Include 1     Exclude 1
                  [1]            []
              /       \        /     \
          Include 2  Exclude 2  ...
            [1,2]      [1]
           /    \
       +3       -3
     [1,2,3]  [1,2]

    At each node:
    - CHOOSE: include or exclude current element
    - EXPLORE: recurse with new state
    - BACKTRACK: undo choice, try next option
    ```

    **The Magic of Backtracking:**
    ```python
    def backtrack(path):
        if is_solution(path):
            solutions.append(path.copy())
            return

        for choice in get_choices():
            # 1. CHOOSE
            path.append(choice)

            # 2. EXPLORE
            backtrack(path)

            # 3. BACKTRACK (undo choice)
            path.pop()
    ```

    **Key Insight:** By undoing choices (pop), we reuse the same path array across all recursive calls, saving memory and time!

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Don't waste time exploring paths that can never lead to a solution—prune them early!

    Think of backtracking like a treasure hunt with clues:
    - **Brute Force:** Dig everywhere (2ⁿ or n! holes)
    - **Backtracking:** If clue says "treasure not in this quadrant," skip entire quadrant (prune!)

    **Example: N-Queens**
    ```
    Without Pruning: Try all n^n ways to place n queens (huge!)
    With Backtracking: Place queen 1, check if valid
                       If any future position attacks it → PRUNE
                       Reduces from billions to thousands of checks!
    ```

    **The Power of Pruning:**
    ```
    Problem: Generate combinations of 3 numbers from [1,2,3,4,5]

    Without backtracking: Generate all 5^3 = 125 possibilities, filter valid
    With backtracking: Only explore valid paths, generate exactly C(5,3) = 10

    Pruning rule: Don't use number smaller than previous (avoid duplicates)
    ```

    **When Backtracking Shines:**
    - Problem has constraints that eliminate large portions of search space
    - Need ALL solutions (not just one)
    - State can be easily saved/restored

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(2ⁿ) to O(n!) | Exponential—exploring combinations or permutations |
    | **Space** | O(n) | Recursion stack depth (reuse path array) |
    | **Improvement** | From O(n^n) or worse | Pruning eliminates invalid branches early |

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Backtracking Works | Example |
    |----------|----------------------|---------|
    | **Generate all combinations** | Systematically build each one | Subsets, power set |
    | **Generate all permutations** | Try each element in each position | All arrangements |
    | **Constraint satisfaction** | Prune invalid states early | Sudoku, N-Queens |
    | **Path finding with constraints** | Explore all paths, backtrack on obstacles | Word Search, maze solving |
    | **Partition problems** | Try all ways to split | Palindrome partitioning |
    | **Combination with target** | Prune when exceeding target | Combination Sum |

    **Red Flags That Suggest Backtracking:**
    - "Find all...", "generate all...", "list all..."
    - "Permutations", "combinations", "subsets"
    - Constraint satisfaction (Sudoku, N-Queens)
    - Decision tree where each level = one choice
    - Need to explore all possibilities with pruning

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Only need one solution** | Backtracking finds all | Greedy or DP (faster) |
    | **Overlapping subproblems** | Recomputing same states | Dynamic Programming |
    | **Very large search space** | 2^100 is too many paths | Approximation, heuristics |
    | **No good pruning rules** | Can't eliminate branches early | Problem may be intractable |
    | **Optimal solution, not all** | Backtracking enumerates all | Greedy or DP for optimization |

    ---

    ## 🎯 Decision Flowchart

    ```
    Need to find ALL solutions?
    ├─ Yes → Can you build solutions incrementally?
    │         ├─ Yes → Can you prune invalid branches early?
    │         │        ├─ Yes → USE BACKTRACKING ✓
    │         │        └─ No → Brute force (may be slow)
    │         └─ No → Different approach
    └─ No → Need just ONE optimal solution?
              └─ Use DP or Greedy instead
    ```

=== "Implementation Templates"

    === "Template 1: Subsets (Power Set)"

        **Use Case:** Generate all possible subsets (include/exclude decisions)

        **Pattern:** At each position, choose to include or exclude element

        ```python
        def subsets(nums):
            """
            Generate all subsets (power set).

            Perfect for: Any include/exclude decision problem

            Concept:
            - At each position: two choices (include or exclude)
            - Build path incrementally
            - Add to result at every step (not just at end)

            Time: O(2ⁿ) - 2^n total subsets
            Space: O(n) - Recursion depth

            Example: [1,2,3] → [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
            """
            result = []

            def backtrack(start, path):
                # Every path is a valid subset (add it!)
                result.append(path[:])  # IMPORTANT: copy path

                # Try including each remaining element
                for i in range(start, len(nums)):
                    # 1. CHOOSE: include nums[i]
                    path.append(nums[i])

                    # 2. EXPLORE: recurse with next element
                    backtrack(i + 1, path)

                    # 3. BACKTRACK: undo choice
                    path.pop()

            backtrack(0, [])
            return result

        # Example Usage:
        nums = [1, 2, 3]
        result = subsets(nums)
        # [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
        ```

        **Key Points:**
        - Add result at EVERY recursive call (not just base case)
        - Use `path[:]` to copy (avoid reference issues)
        - `start` parameter prevents duplicates ([1,2] same as [2,1])

    === "Template 2: Permutations"

        **Use Case:** Generate all possible orderings of elements

        **Pattern:** Try each unused element at each position

        ```python
        def permute(nums):
            """
            Generate all permutations.

            Perfect for: All arrangements, ordering problems

            Concept:
            - At each position, try every unused element
            - Track which elements already used
            - Add to result when all positions filled

            Time: O(n!) - n! total permutations
            Space: O(n) - Recursion depth + used set

            Example: [1,2,3] → [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
            """
            result = []

            def backtrack(path, used):
                # Base case: all elements used
                if len(path) == len(nums):
                    result.append(path[:])
                    return

                # Try each unused element
                for i in range(len(nums)):
                    if i in used:
                        continue  # Skip if already used

                    # 1. CHOOSE: use nums[i]
                    path.append(nums[i])
                    used.add(i)

                    # 2. EXPLORE: fill remaining positions
                    backtrack(path, used)

                    # 3. BACKTRACK: undo choice
                    path.pop()
                    used.remove(i)

            backtrack([], set())
            return result

        # Example Usage:
        nums = [1, 2, 3]
        result = permute(nums)
        # [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
        ```

        **Key Points:**
        - Use set to track used indices (O(1) lookup)
        - Base case: `len(path) == len(nums)` (all positions filled)
        - Unlike subsets, add result only at leaves (complete permutations)

    === "Template 3: Combination Sum"

        **Use Case:** Find all combinations that sum to target (with pruning)

        **Pattern:** Build sum incrementally, prune when exceeding target

        ```python
        def combination_sum(candidates, target):
            """
            Find all combinations that sum to target.

            Perfect for: Target sum problems with choices

            Concept:
            - Try adding each candidate
            - Prune when sum exceeds target (early stop!)
            - Allow reuse of elements (start at i, not i+1)

            Time: O(2^n) worst case, much faster with pruning
            Space: O(target/min(candidates)) - Max recursion depth

            Example: candidates=[2,3,6,7], target=7 → [[2,2,3], [7]]
            """
            result = []

            def backtrack(start, path, total):
                # Success: found valid combination
                if total == target:
                    result.append(path[:])
                    return

                # PRUNE: stop if exceeding target (no point continuing)
                if total > target:
                    return

                # Try each candidate
                for i in range(start, len(candidates)):
                    # 1. CHOOSE: add candidates[i]
                    path.append(candidates[i])

                    # 2. EXPLORE: can reuse same element (start at i, not i+1)
                    backtrack(i, path, total + candidates[i])

                    # 3. BACKTRACK: undo choice
                    path.pop()

            # Optimization: sort to prune earlier
            candidates.sort()
            backtrack(0, [], 0)
            return result

        # Example Usage:
        candidates = [2, 3, 6, 7]
        target = 7
        result = combination_sum(candidates, target)
        # [[2,2,3], [7]]
        ```

        **Key Points:**
        - Pruning (`if total > target: return`) is KEY to performance
        - Sort candidates first for better pruning
        - `start=i` (not i+1) allows reusing same element
        - Track running total to avoid recalculating sum

    === "Template 4: N-Queens (Constraint Satisfaction)"

        **Use Case:** Place N queens on NxN board with no attacks

        **Pattern:** Place one queen per row, check constraints before placing

        ```python
        def solve_n_queens(n):
            """
            Solve N-Queens problem.

            Perfect for: Constraint satisfaction problems

            Concept:
            - Place one queen per row (row by row)
            - Before placing, check if position valid
            - Prune entire branch if invalid placement

            Time: O(n!) - Much less with pruning
            Space: O(n²) - Board storage

            Example: n=4 → 2 solutions
            """
            result = []
            board = [['.'] * n for _ in range(n)]

            def is_valid(row, col):
                # Check column (no queen above in same column)
                for i in range(row):
                    if board[i][col] == 'Q':
                        return False

                # Check diagonal \ (upper-left)
                i, j = row - 1, col - 1
                while i >= 0 and j >= 0:
                    if board[i][j] == 'Q':
                        return False
                    i -= 1
                    j -= 1

                # Check diagonal / (upper-right)
                i, j = row - 1, col + 1
                while i >= 0 and j < n:
                    if board[i][j] == 'Q':
                        return False
                    i -= 1
                    j += 1

                return True  # Position is safe

            def backtrack(row):
                # Base case: all queens placed
                if row == n:
                    result.append([''.join(r) for r in board])
                    return

                # Try placing queen in each column of this row
                for col in range(n):
                    # PRUNE: skip if position invalid
                    if not is_valid(row, col):
                        continue

                    # 1. CHOOSE: place queen
                    board[row][col] = 'Q'

                    # 2. EXPLORE: place queens in remaining rows
                    backtrack(row + 1)

                    # 3. BACKTRACK: remove queen
                    board[row][col] = '.'

            backtrack(0)
            return result

        # Example Usage:
        solutions = solve_n_queens(4)
        # 2 solutions for 4x4 board
        ```

        **Key Points:**
        - Check constraints BEFORE recursing (pruning!)
        - Only need to check upward (haven't placed below yet)
        - One queen per row guaranteed by recursion structure
        - Track state in 2D board, modify/restore as you backtrack

    === "Visual Walkthrough"

        **Problem:** Generate all subsets of [1, 2, 3]

        ```
        Recursion Tree (showing path state at each call):

                      backtrack(0, [])
                      Add: []
                  /           |          \
            Include 1    (continue)    Include 2
                 |                          |
          backtrack(1, [1])        backtrack(2, [2])
          Add: [1]                  Add: [2]
           /      \                      |
        +2        +3                   +3
         |         |                     |
    bt(2,[1,2])  bt(2,[1,3])      bt(3,[2,3])
    Add:[1,2]    Add:[1,3]         Add:[2,3]
      |
     +3
      |
    bt(3,[1,2,3])
    Add:[1,2,3]

        Step-by-Step Execution:

        Call backtrack(0, [])
          → Add [] to result
          → Loop i=0: path=[1]
              Call backtrack(1, [1])
                → Add [1] to result
                → Loop i=1: path=[1,2]
                    Call backtrack(2, [1,2])
                      → Add [1,2] to result
                      → Loop i=2: path=[1,2,3]
                          Call backtrack(3, [1,2,3])
                            → Add [1,2,3] to result
                            → No more elements, return
                          ← Backtrack: path=[1,2] (popped 3)
                      → End loop, return
                    ← Backtrack: path=[1] (popped 2)
                → Loop i=2: path=[1,3]
                    Call backtrack(2, [1,3])
                      → Add [1,3] to result
                      → No more elements, return
                    ← Backtrack: path=[1] (popped 3)
                → End loop, return
              ← Backtrack: path=[] (popped 1)
          → Loop i=1: path=[2]
              ... (similar for [2], [2,3], [3])

        Result: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
        ```

        **Key Observation:** Each element is either included or excluded, creating a binary decision tree. The `start` parameter ensures we don't revisit earlier elements, avoiding duplicates.

        ---

        **N-Queens Visual (n=4):**

        ```
        Try placing queen in row 0, column 0:

        Q . . .    Valid? Try row 1...
        . . . .
        . . . .
        . . . .

        Q . . .
        . . Q .    Valid? Try row 2...
        . . . .
        . . . .

        Q . . .
        . . Q .
        . . . .    Can't place anywhere! (all attacked)
        . . . .    BACKTRACK to row 1, try different column

        Eventually finds:
        . Q . .    Solution 1
        . . . Q
        Q . . .
        . . Q .

        Pruning power: Instead of checking all 4^4 = 256 placements,
        backtracking with constraint checking explores far fewer!
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master the basic backtracking template.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Subsets | Include/exclude | Basic backtracking structure | [LeetCode 78](https://leetcode.com/problems/subsets/) |
    | Subsets II | With duplicates | Handling duplicates in backtracking | [LeetCode 90](https://leetcode.com/problems/subsets-ii/) |
    | Letter Case Permutation | Binary choices | Two choices per position | [LeetCode 784](https://leetcode.com/problems/letter-case-permutation/) |
    | Binary Watch | Combinations with constraint | Fixed-size combinations | [LeetCode 401](https://leetcode.com/problems/binary-watch/) |
    | Combination Sum III | Limited elements | Fixed k elements summing to n | [LeetCode 216](https://leetcode.com/problems/combination-sum-iii/) |

    **Goal:** Solve all 5 problems. Understand choose-explore-unchoose pattern.

    ---

    ### Phase 2: Application (Medium)
    Apply backtracking to complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Permutations | All arrangements | Track used elements | [LeetCode 46](https://leetcode.com/problems/permutations/) |
    | Combinations | Choose k from n | Fixed-size subsets | [LeetCode 77](https://leetcode.com/problems/combinations/) |
    | Letter Combinations | Multiple choices per position | Phone keypad mapping | [LeetCode 17](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) |
    | Combination Sum | Target sum with reuse | Pruning when sum exceeds | [LeetCode 39](https://leetcode.com/problems/combination-sum/) |
    | Palindrome Partitioning | String partitioning | Check palindrome constraint | [LeetCode 131](https://leetcode.com/problems/palindrome-partitioning/) |
    | Word Search | 2D grid search | Backtrack on grid with visited | [LeetCode 79](https://leetcode.com/problems/word-search/) |

    **Goal:** Solve 4 out of 6. Learn pruning and constraint handling.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex constraint satisfaction and optimization.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | N-Queens | Constraint satisfaction | Complex validity checks, multiple constraints | [LeetCode 51](https://leetcode.com/problems/n-queens/) |
    | Sudoku Solver | Multi-constraint | Row, column, box constraints simultaneously | [LeetCode 37](https://leetcode.com/problems/sudoku-solver/) |
    | Word Search II | Multiple targets with Trie | Backtracking + data structure | [LeetCode 212](https://leetcode.com/problems/word-search-ii/) |
    | Expression Add Operators | String with operators | Generate expressions with evaluation | [LeetCode 282](https://leetcode.com/problems/expression-add-operators/) |

    **Goal:** Solve 2 out of 4. Master complex pruning and multiple constraints.

    ---

    ## 🎯 Practice Strategy

    1. **Start with Subsets:** Simplest template, builds foundation
    2. **Identify Decision Tree:** What choices at each step?
    3. **Draw Recursion Tree:** Visualize first few levels on paper
    4. **Write Base Case First:** When to add to result?
    5. **Add Pruning:** What branches can be skipped early?
    6. **Code Without Looking:** Backtracking is about pattern recognition
    7. **Time Yourself:** <20 minutes per easy, <35 for medium, <50 for hard

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Forgetting to copy path** | Append reference, not value | Use `path[:]` or `path.copy()` when adding to result |
    | **Not backtracking** | Forget to undo choice | Always `path.pop()` or restore state after recursion |
    | **Wrong base case** | Unclear when solution is complete | For subsets: add at every step; permutations: when len==n |
    | **Missing pruning** | Explore invalid branches | Check constraints BEFORE recursive call |
    | **Duplicate solutions** | Not handling duplicates in input | Sort input, skip duplicates: `if i>start and nums[i]==nums[i-1]: continue` |
    | **Infinite recursion** | Not advancing state | Ensure start/index parameter increases |
    | **Modifying input** | Change nums array during recursion | Use indices/flags, don't modify input directly |

---

## Advanced Graph Patterns

### Union Find

=== "Understanding the Pattern"

    ## 📖 What is Union Find?

    Imagine you're organizing a massive networking event where people form groups based on mutual connections. Initially, everyone stands alone. As people meet and connect, they join groups. Your job: quickly answer "Are these two people in the same network?" and "How many separate networks exist?"

    The naive approach? For each query, traverse all connections to see if there's a path. This could take minutes for large networks!

    **Union Find** (also called Disjoint Set Union or DSU) is the elegant solution. Think of it like a company org chart where every group has a representative (the boss). To check if two people are connected, just climb the management chain until you reach the top boss of each. Same boss? Same network!

    The magic: with two simple optimizations (path compression and union by rank), operations become nearly O(1)—so fast it's almost like magic!

    ---

    ## 🔧 How It Works

    Union Find maintains a forest of trees, where each tree represents a connected component.

    **Two Core Operations:**

    **1. Find (with Path Compression):**
    ```
    Initial tree:     After find(5):
         0                 0
        / \               /|\
       1   2             1 2 5
      / \               /
     3   4             3 4
         |
         5

    When finding 5's root, compress path:
    All nodes on path point directly to root!
    ```

    **2. Union (by Rank):**
    ```
    Two separate trees:
       0       3
       |       |
       1       4

    After union(1, 4):
       0
      / \
     1   3
         |
         4

    Attach smaller tree under larger tree's root
    → Keeps trees shallow
    ```

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Instead of storing explicit connections between all related elements, just store each element's parent in the tree. Finding connectivity = finding common ancestor!

    Think of family trees:
    - Siblings share parents (connected)
    - Cousins share grandparents (connected)
    - Unrelated people have different ancestors (not connected)

    **Why Path Compression is Brilliant:**

    After finding the root once, make everyone on the path point directly to it. Future queries on those elements become instant! It's like bookmarking the CEO instead of going through middle management every time.

    **Why Union by Rank Works:**

    Always attach the shorter tree under the taller tree. This prevents trees from becoming long chains, keeping find operations fast. It's like merging companies—the smaller one gets absorbed into the larger one's structure.

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(α(n)) per operation | α is inverse Ackermann function—grows incredibly slowly (< 5 for all practical n) |
    | **Space** | O(n) | Two arrays: parent and rank |
    | **Improvement** | From O(n) per query | Without optimizations, find could traverse entire component |

    **Why α(n) is "basically constant":** α(n) < 5 for n < 2^65536—a number larger than atoms in the universe!

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Union Find Works | Example |
    |----------|---------------------|---------|
    | **Connected Components** | Groups elements into disjoint sets efficiently | Number of Islands, Connected Components |
    | **Cycle Detection** | If union fails (already connected), cycle exists | Redundant Connection, Graph Valid Tree |
    | **Dynamic Connectivity** | Handle connect and query operations in real-time | Social Network Connections |
    | **Equivalence Classes** | Group equivalent elements together | Accounts Merge, Similar Strings |
    | **Minimum Spanning Tree** | Kruskal's algorithm uses Union Find | Network Design |

    **Red Flags That Suggest Union Find:**
    - "Connected components" in undirected graph
    - "Are X and Y in the same group/network?"
    - "Merge groups when..."
    - "Detect cycles" in undirected graph
    - "Group by transitive property" (if A~B and B~C, then A~C)

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Directed graphs** | Union Find assumes bidirectional connections | DFS/BFS with directed edges |
    | **Need to remove edges** | Union Find doesn't support "disconnect" | Maintain explicit adjacency list |
    | **Need actual paths** | Only tells if connected, not the path | BFS/DFS for path reconstruction |
    | **Weighted connectivity** | Standard UF doesn't track edge weights | Weighted Union Find variant |
    | **Frequent component listing** | Efficient at queries, not listing all members | Maintain separate component lists |

    ---

    ## 🎯 Decision Flowchart

    ```
    Working with undirected graph connectivity?
    ├─ Yes → Need to know if two nodes are connected?
    │         ├─ Yes → Need actual path between them?
    │         │        ├─ No → USE UNION FIND ✓
    │         │        └─ Yes → Use BFS/DFS
    │         └─ No → Need to detect cycles?
    │                  └─ Yes → USE UNION FIND ✓
    └─ No → Need to group elements by equivalence relation?
              ├─ Yes → USE UNION FIND ✓
              └─ No → Different pattern needed
    ```

=== "Implementation Templates"

    === "Template 1: Basic Union Find"

        **Use Case:** Standard connectivity queries and component counting

        **Pattern:** Path compression + union by rank for optimal performance

        ```python
        class UnionFind:
            """
            Union Find with path compression and union by rank.

            Perfect for: Connected component problems, cycle detection

            Optimizations:
            - Path compression: flatten tree during find
            - Union by rank: attach smaller tree to larger tree

            Time: O(α(n)) per operation (amortized, nearly O(1))
            Space: O(n) for parent and rank arrays

            Example: Social network connectivity queries
            """
            def __init__(self, n):
                # Each element starts as its own parent (own component)
                self.parent = list(range(n))
                # Rank tracks tree height (for balanced union)
                self.rank = [0] * n
                # Track total number of disjoint components
                self.components = n

            def find(self, x):
                """
                Find root of x with path compression.

                Path compression: Make all nodes on path point to root.
                This flattens the tree for future O(1) lookups.
                """
                if self.parent[x] != x:
                    # Recursively find root and compress path
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                """
                Union two sets by rank.

                Returns True if union performed, False if already connected.
                """
                root_x = self.find(x)
                root_y = self.find(y)

                # Already in same component?
                if root_x == root_y:
                    return False  # No union needed (and cycle if graph edge!)

                # Union by rank: attach shorter tree under taller tree
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                elif self.rank[root_x] > self.rank[root_y]:
                    self.parent[root_y] = root_x
                else:
                    # Equal rank: make one root and increase its rank
                    self.parent[root_y] = root_x
                    self.rank[root_x] += 1

                # Merged two components into one
                self.components -= 1
                return True

            def connected(self, x, y):
                """Check if x and y are in same component."""
                return self.find(x) == self.find(y)

            def count_components(self):
                """Return number of disjoint components."""
                return self.components


        # Example Usage:
        uf = UnionFind(5)  # 5 elements: 0,1,2,3,4
        uf.union(0, 1)     # Connect 0-1
        uf.union(1, 2)     # Connect 1-2 (transitively connects 0-2)
        print(uf.connected(0, 2))  # True
        print(uf.connected(0, 3))  # False
        print(uf.count_components())  # 3: {0,1,2}, {3}, {4}
        ```

        **Key Points:**
        - Initialize each element as its own parent
        - Path compression happens during find
        - Union by rank keeps trees balanced
        - Track component count for O(1) queries

    === "Template 2: Cycle Detection"

        **Use Case:** Detect if adding an edge creates a cycle in undirected graph

        **Pattern:** If union returns False, cycle detected

        ```python
        def find_redundant_connection(edges):
            """
            Find the edge that creates a cycle in undirected graph.

            Perfect for: Identifying which edge to remove to make tree

            Concept:
            - Process edges in order
            - If union fails (already connected), that edge creates cycle
            - Return the first edge that creates cycle

            Time: O(E × α(n)) where E = number of edges
            Space: O(n) for Union Find structure

            Example: edges=[[1,2],[1,3],[2,3]] → [2,3] (creates cycle)
            """
            n = len(edges)
            uf = UnionFind(n + 1)  # +1 because nodes might be 1-indexed

            for u, v in edges:
                # Try to union the two nodes
                if not uf.union(u, v):
                    # Union failed: already connected!
                    # This edge creates a cycle
                    return [u, v]

            return []  # No cycle found (shouldn't happen if problem guarantees one)


        # Example Usage:
        edges = [[1,2], [2,3], [3,4], [1,4], [1,5]]
        redundant = find_redundant_connection(edges)
        print(redundant)  # Output: [1,4] (creates cycle in 1-2-3-4-1)
        ```

        **Key Points:**
        - Union returns False if nodes already connected
        - First failing union indicates cycle-creating edge
        - Perfect for "graph valid tree" problems

    === "Template 3: Grouping by Equivalence"

        **Use Case:** Group items that are equivalent through transitive relations

        **Pattern:** Map items to IDs, union equivalent items, collect groups

        ```python
        def accounts_merge(accounts):
            """
            Merge accounts that share common emails.

            Perfect for: Grouping by transitive property

            Concept:
            - Each account gets an ID (index)
            - If accounts share email, they belong to same person
            - Union accounts that share any email
            - Group all emails by root account

            Time: O(n × k × α(n)) where k = emails per account
            Space: O(n × k) for email mapping

            Example: [["John","j1@","j2@"],["John","j2@","j3@"]]
                  → [["John","j1@","j2@","j3@"]] (merged via j2@)
            """
            email_to_id = {}  # Map email to account ID
            uf = UnionFind(len(accounts))

            # Build Union Find: union accounts sharing emails
            for i, account in enumerate(accounts):
                for email in account[1:]:  # Skip name at account[0]
                    if email in email_to_id:
                        # This email seen before: merge accounts
                        uf.union(i, email_to_id[email])
                    else:
                        # First time seeing this email
                        email_to_id[email] = i

            # Group emails by root account
            components = {}
            for email, acc_id in email_to_id.items():
                root = uf.find(acc_id)  # Find root account
                if root not in components:
                    components[root] = []
                components[root].append(email)

            # Format result: [name, sorted emails]
            result = []
            for root, emails in components.items():
                name = accounts[root][0]
                result.append([name] + sorted(emails))

            return result


        # Example Usage:
        accounts = [
            ["John", "john@mail.com", "john_work@mail.com"],
            ["John", "john_work@mail.com", "john_other@mail.com"],
            ["Mary", "mary@mail.com"]
        ]
        merged = accounts_merge(accounts)
        # Output: [["John", "john@mail.com", "john_other@mail.com", "john_work@mail.com"],
        #          ["Mary", "mary@mail.com"]]
        ```

        **Key Points:**
        - Map entities to integer IDs for Union Find
        - Union transitively related entities
        - Collect groups using find to get root
        - Works for any transitive equivalence relation

    === "Visual Walkthrough"

        **Problem:** Count connected components with n=5, edges=[[0,1], [1,2], [3,4]]

        ```
        Initial State:
        Nodes: 0   1   2   3   4
        Parent: [0, 1, 2, 3, 4]  (each node is its own parent)
        Components: 5

        Step 1: Union(0, 1)
        Find(0) = 0, Find(1) = 1
        Make 1's parent = 0

        0     2   3   4
        │
        1

        Parent: [0, 0, 2, 3, 4]
        Components: 4

        Step 2: Union(1, 2)
        Find(1) = 0 (follows parent chain!)
        Find(2) = 2
        Make 2's parent = 0

        0         3   4
        ├─┤
        1 2

        Parent: [0, 0, 0, 3, 4]
        Components: 3

        Step 3: Union(3, 4)
        Find(3) = 3, Find(4) = 4
        Make 4's parent = 3

        0         3
        ├─┤       │
        1 2       4

        Parent: [0, 0, 0, 3, 3]
        Components: 2

        Final: 2 components: {0,1,2} and {3,4}
        ```

        **Path Compression in Action:**

        ```
        Before path compression:
              0
              │
              1
              │
              2
              │
              3

        After Find(3):
              0
           ╱  │  ╲
          1   2   3

        All nodes now point directly to root!
        Next find(3) = O(1)
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master basic Union Find operations.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Number of Connected Components | Basic UF | Standard template, count components | [LeetCode 323](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) |
    | Find if Path Exists | Connectivity query | Simple connected() check | [LeetCode 1971](https://leetcode.com/problems/find-if-path-exists-in-graph/) |
    | The Earliest Moment | Union with sorting | Union edges by time, check when fully connected | [LeetCode 1101](https://leetcode.com/problems/the-earliest-moment-when-everyone-become-friends/) |
    | Redundant Connection | Cycle detection | Detect when union fails | [LeetCode 684](https://leetcode.com/problems/redundant-connection/) |
    | Number of Provinces | Matrix input | Build UF from adjacency matrix | [LeetCode 547](https://leetcode.com/problems/number-of-provinces/) |

    **Goal:** Implement Union Find from scratch and understand find/union operations.

    ---

    ### Phase 2: Application (Medium)
    Apply UF to various problem types.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Accounts Merge | Grouping by equivalence | Map emails to IDs, collect groups | [LeetCode 721](https://leetcode.com/problems/accounts-merge/) |
    | Graph Valid Tree | Cycle + connectivity | Must have exactly n-1 edges and be connected | [LeetCode 261](https://leetcode.com/problems/graph-valid-tree/) |
    | Sentence Similarity II | Transitive equivalence | Words equivalent if transitively related | [LeetCode 737](https://leetcode.com/problems/sentence-similarity-ii/) |
    | Satisfiability of Equality Equations | Constraint satisfaction | Union equal variables, check contradictions | [LeetCode 990](https://leetcode.com/problems/satisfiability-of-equality-equations/) |
    | Smallest String With Swaps | Equivalence classes | Group swappable indices, sort each group | [LeetCode 1202](https://leetcode.com/problems/smallest-string-with-swaps/) |

    **Goal:** Apply UF to non-obvious problems involving transitive relations.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex variations and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Swim in Rising Water | Binary search + UF | Binary search on time, check connectivity with UF | [LeetCode 778](https://leetcode.com/problems/swim-in-rising-water/) |
    | Minimize Malware Spread | Component size tracking | Union by size, find critical node | [LeetCode 924](https://leetcode.com/problems/minimize-malware-spread/) |
    | Redundant Connection II | Directed graph | Handle directed edges, multiple cases | [LeetCode 685](https://leetcode.com/problems/redundant-connection-ii/) |
    | Number of Islands II | Online queries | Dynamic island merging with multiple unions | [LeetCode 305](https://leetcode.com/problems/number-of-islands-ii/) |

    **Goal:** Combine UF with other techniques (binary search, component tracking).

    ---

    ## 🎯 Practice Strategy

    1. **Master the Template:** Implement Union Find from memory
    2. **Understand Optimizations:** Know why path compression and union by rank matter
    3. **Identify Equivalence Relations:** Spot when grouping is needed
    4. **Draw the Trees:** Visualize parent pointers and tree structure
    5. **Track Component Count:** Decrement on successful union
    6. **Map to Integers:** Convert problem entities to 0-indexed IDs

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Not using path compression | Forget to update parent during find | Always do: `self.parent[x] = self.find(self.parent[x])` |
    | Forgetting to check union result | Don't realize union returns False if already connected | Use return value for cycle detection |
    | Wrong initialization | Start parent array with zeros or wrong values | Initialize: `parent = list(range(n))` (each element is its own parent) |
    | Not decrementing component count | Forget to track components | Decrement on successful union: `self.components -= 1` |
    | Mixing up union by rank vs size | Confuse rank (tree height) with size (node count) | Rank for balancing, size for weighted problems |

---

### Topological Sort

=== "Understanding the Pattern"

    ## 📖 What is Topological Sort?

    Imagine you're getting ready for a formal event. You can't just put on clothes in random order—there are dependencies! You must wear your shirt before your tie, socks before shoes, underwear before pants. Some items are independent (watch and tie can go in any order), but others have strict requirements.

    **Topological Sort** solves exactly this problem: given a set of tasks with dependencies, find a valid order to complete them all. It's like creating a schedule that respects all "must come before" rules.

    Real-world applications are everywhere: course prerequisites (Calculus I before Calculus II), build systems (compile dependencies), project management (task ordering), package managers (install dependencies), and even spreadsheet calculations (cell formula dependencies)!

    The beautiful insight: if your dependency graph has a cycle, there's NO valid ordering (imagine: wear shirt before tie, tie before jacket, jacket before shirt—impossible!). Topological sort not only finds the order but also detects impossible situations.

    ---

    ## 🔧 How It Works

    There are two main approaches to topological sort:

    **1. Kahn's Algorithm (BFS approach):**
    ```
    Graph: 0 → 1 → 3
           0 → 2 → 3

    In-degree count: [0, 1, 1, 2]
                     Course 0 has no prerequisites!

    Step 1: Start with in-degree 0 (no dependencies)
            Queue: [0]
            Process: 0

    Step 2: Remove edges from 0, decrease in-degrees
            In-degree: [×, 0, 0, 2]
            Queue: [1, 2]

    Step 3: Process 1 and 2
            In-degree: [×, ×, ×, 0]
            Queue: [3]

    Step 4: Process 3
            Result: [0, 1, 2, 3] or [0, 2, 1, 3] (both valid!)
    ```

    **2. DFS Approach (Post-order):**
    ```
    Visit nodes in DFS, add to result in POST-order
    (after all dependencies visited)

    Reverse post-order = topological order
    ```

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** A node can only be processed after ALL its dependencies are satisfied. By tracking in-degrees (number of incoming edges), we know when a node is ready!

    Think of it like cooking a complex meal:
    - Some dishes can start immediately (in-degree = 0)
    - Others need ingredients prepared first (in-degree > 0)
    - As each dish finishes, it "unlocks" dishes waiting for it
    - If dishes circularly depend on each other, you can't start (cycle!)

    **Why Kahn's Algorithm Works:**

    Start with nodes that have no prerequisites (in-degree 0). When we "complete" a node, we remove its outgoing edges, decreasing in-degrees of dependent nodes. When a node's in-degree reaches 0, all its prerequisites are done—it's ready!

    **Why DFS Works:**

    In DFS post-order, we add a node to result AFTER visiting all its descendants (dependencies). Reversing this gives us an order where dependencies come before dependents.

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(V + E) | Visit each vertex once, traverse each edge once |
    | **Space** | O(V + E) | Graph storage + queue/stack + in-degree array |
    | **Improvement** | From O(V²) | Without smart ordering, might check all pairs repeatedly |

    **Why so efficient?** Each node is processed exactly once, each edge examined exactly once—linear in graph size!

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Topological Sort Works | Example |
    |----------|---------------------------|---------|
    | **Task Scheduling** | Order tasks respecting dependencies | Project Management, Build Systems |
    | **Course Prerequisites** | Take courses in valid order | Course Schedule, Curriculum Planning |
    | **Compilation Order** | Compile files respecting imports | Build Systems (Make, Gradle) |
    | **Dependency Resolution** | Install packages with dependencies | npm, pip, apt |
    | **Spreadsheet Calculations** | Calculate cells in dependency order | Excel formula evaluation |

    **Red Flags That Suggest Topological Sort:**
    - "Order tasks with dependencies"
    - "Course prerequisites"
    - Keywords: "directed acyclic graph", "DAG", "ordering"
    - "Can all tasks be completed?"
    - "Find valid sequence respecting constraints"
    - "Detect circular dependencies"

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Undirected graphs** | Dependencies need direction | Different graph algorithm |
    | **Cyclic graphs** | No valid ordering exists | Cycle detection first, handle separately |
    | **Not ordering problem** | Just need connectivity or paths | BFS/DFS/Union Find |
    | **Need all possible orderings** | Topo sort gives one valid order | Backtracking to generate all |
    | **Weighted dependencies** | Standard topo sort ignores weights | Critical path method |

    ---

    ## 🎯 Decision Flowchart

    ```
    Is it a directed graph?
    ├─ Yes → Does it represent dependencies/ordering?
    │         ├─ Yes → Need to find valid order?
    │         │        ├─ Yes → Check for cycles first
    │         │        │        ├─ No cycle → USE TOPOLOGICAL SORT ✓
    │         │        │        └─ Has cycle → No valid ordering possible
    │         │        └─ No → Just detect cycle?
    │         │                 └─ Yes → Use topo sort (len(result) < n)
    │         └─ No → Different problem
    └─ No → Topological sort needs directed graph
    ```

=== "Implementation Templates"

    === "Template 1: Kahn's Algorithm (BFS)"

        **Use Case:** Most intuitive, easy cycle detection, can find all nodes at same level

        **Pattern:** Track in-degrees, process nodes with in-degree 0, decrease neighbor in-degrees

        ```python
        from collections import deque, defaultdict

        def topological_sort_kahn(n, edges):
            """
            Topological sort using Kahn's algorithm (BFS approach).

            Perfect for: Course scheduling, task ordering, dependency resolution

            Concept:
            - Count in-degrees (number of prerequisites)
            - Start with nodes having no prerequisites (in-degree 0)
            - Process node: decrease in-degree of dependents
            - Add newly freed nodes (in-degree becomes 0) to queue

            Time: O(V + E) - Visit each vertex and edge once
            Space: O(V + E) - Graph + in-degree array + queue

            Example: Course prerequisites → valid course order
            """
            # Build adjacency list and in-degree count
            graph = defaultdict(list)
            in_degree = [0] * n

            for u, v in edges:
                graph[u].append(v)  # u → v (u must come before v)
                in_degree[v] += 1   # v has one more prerequisite

            # Start with all nodes that have no prerequisites
            queue = deque([i for i in range(n) if in_degree[i] == 0])
            result = []

            while queue:
                # Process node with no remaining prerequisites
                node = queue.popleft()
                result.append(node)

                # Remove this node from graph
                # Decrease in-degree of all dependents
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1

                    # If neighbor now has no prerequisites, it's ready
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # Cycle detection: if we couldn't process all nodes, there's a cycle
            if len(result) < n:
                return []  # Cycle exists, no valid topological order

            return result


        # Example Usage:
        n = 4
        edges = [[0,1], [0,2], [1,3], [2,3]]  # 0→1→3, 0→2→3
        order = topological_sort_kahn(n, edges)
        print(order)  # Output: [0, 1, 2, 3] or [0, 2, 1, 3] (both valid!)
        ```

        **Key Points:**
        - In-degree = number of incoming edges (prerequisites)
        - Queue contains nodes ready to process (no pending prerequisites)
        - If result has fewer than n nodes, cycle detected
        - Can process nodes level by level (useful for parallel scheduling)

    === "Template 2: DFS Approach"

        **Use Case:** More elegant code, easier to adapt for advanced problems

        **Pattern:** DFS post-order traversal, reverse the result

        ```python
        from collections import defaultdict

        def topological_sort_dfs(n, edges):
            """
            Topological sort using DFS approach.

            Perfect for: Problems requiring recursive exploration

            Concept:
            - DFS from each unvisited node
            - Add node to result AFTER visiting all descendants (post-order)
            - Reverse result to get topological order
            - Use 3 states: unvisited, visiting (for cycle detection), visited

            Time: O(V + E) - DFS visits each node and edge once
            Space: O(V + E) - Graph + recursion stack + visited array

            Example: Build dependency tree → compilation order
            """
            # Build adjacency list
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)

            # 3-state tracking: 0=unvisited, 1=visiting, 2=visited
            visited = [0] * n
            result = []
            has_cycle = False

            def dfs(node):
                nonlocal has_cycle

                if visited[node] == 1:
                    # Found back edge: cycle detected!
                    has_cycle = True
                    return

                if visited[node] == 2:
                    # Already processed this node
                    return

                # Mark as currently visiting (detect cycles)
                visited[node] = 1

                # Visit all descendants first (dependencies)
                for neighbor in graph[node]:
                    dfs(neighbor)
                    if has_cycle:
                        return

                # Mark as fully visited
                visited[node] = 2

                # Add in POST-order (after all descendants)
                result.append(node)

            # Start DFS from all unvisited nodes
            for i in range(n):
                if visited[i] == 0:
                    dfs(i)
                    if has_cycle:
                        return []  # Cycle detected

            # Reverse post-order gives topological order
            return result[::-1]


        # Example Usage:
        n = 4
        edges = [[0,1], [0,2], [1,3], [2,3]]
        order = topological_sort_dfs(n, edges)
        print(order)  # Output: [0, 1, 2, 3] or [0, 2, 1, 3]
        ```

        **Key Points:**
        - Three states crucial for cycle detection
        - State 1 (visiting) detects back edges (cycles)
        - Post-order means add AFTER processing all children
        - Reverse post-order gives valid topological order

    === "Template 3: Course Schedule (Practical Application)"

        **Use Case:** Can all courses be taken? Find valid course order.

        **Pattern:** Apply Kahn's algorithm to course prerequisites

        ```python
        from collections import deque, defaultdict

        def can_finish_courses(num_courses, prerequisites):
            """
            Determine if all courses can be completed given prerequisites.

            Perfect for: Cycle detection in prerequisite problems

            Concept:
            - Build graph from prerequisites
            - Use Kahn's algorithm
            - If can't process all courses, circular dependency exists

            Time: O(V + E) where V = courses, E = prerequisites
            Space: O(V + E)

            Example: prerequisites=[[1,0],[2,1]] → True (take 0→1→2)
                     prerequisites=[[1,0],[0,1]] → False (cycle!)
            """
            # Build graph: course → [courses that depend on it]
            graph = defaultdict(list)
            in_degree = [0] * num_courses

            for course, prereq in prerequisites:
                # prereq → course (must take prereq before course)
                graph[prereq].append(course)
                in_degree[course] += 1

            # Start with courses having no prerequisites
            queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
            completed = 0

            while queue:
                course = queue.popleft()
                completed += 1

                # "Complete" this course, unlock dependent courses
                for next_course in graph[course]:
                    in_degree[next_course] -= 1
                    if in_degree[next_course] == 0:
                        queue.append(next_course)

            # Could we complete all courses?
            return completed == num_courses


        def find_course_order(num_courses, prerequisites):
            """
            Find valid order to take all courses.

            Returns empty list if impossible (cycle exists).
            """
            graph = defaultdict(list)
            in_degree = [0] * num_courses

            for course, prereq in prerequisites:
                graph[prereq].append(course)
                in_degree[course] += 1

            queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
            result = []

            while queue:
                course = queue.popleft()
                result.append(course)

                for next_course in graph[course]:
                    in_degree[next_course] -= 1
                    if in_degree[next_course] == 0:
                        queue.append(next_course)

            # Return result only if all courses can be taken
            return result if len(result) == num_courses else []


        # Example Usage:
        print(can_finish_courses(2, [[1,0]]))  # True
        print(can_finish_courses(2, [[1,0],[0,1]]))  # False (cycle)
        print(find_course_order(4, [[1,0],[2,0],[3,1],[3,2]]))  # [0,1,2,3] or [0,2,1,3]
        ```

        **Key Points:**
        - Prerequisites are reversed: [course, prereq] means prereq → course
        - Can't finish all courses = cycle exists
        - Multiple valid orderings possible
        - Practical template for real interview problems

    === "Visual Walkthrough"

        **Problem:** Course Schedule with n=4, prerequisites=[[1,0], [2,0], [3,1], [3,2]]

        ```
        Graph Visualization:
        0 → 1 → 3
        ↓   ↗
        2 ──┘

        (Must take 0 before 1, 0 before 2, 1 before 3, 2 before 3)

        Initial State:
        In-degree: [0, 1, 1, 2]
        Course 0: no prerequisites (in-degree 0)
        Course 1: needs 0 (in-degree 1)
        Course 2: needs 0 (in-degree 1)
        Course 3: needs 1 and 2 (in-degree 2)

        Step 1: Process course 0 (no prerequisites)
        Queue: [0] → Process: 0
        Result: [0]
        Update in-degrees of 1 and 2:
        In-degree: [×, 0, 0, 2]
        Queue: [1, 2] (both ready now!)

        Step 2: Process course 1
        Queue: [1, 2] → Process: 1
        Result: [0, 1]
        Update in-degree of 3:
        In-degree: [×, ×, 0, 1]
        Queue: [2]

        Step 3: Process course 2
        Queue: [2] → Process: 2
        Result: [0, 1, 2]
        Update in-degree of 3:
        In-degree: [×, ×, ×, 0]
        Queue: [3] (3 is ready now!)

        Step 4: Process course 3
        Queue: [3] → Process: 3
        Result: [0, 1, 2, 3]

        Final: Valid order is [0, 1, 2, 3]
        Alternative valid order: [0, 2, 1, 3]
        ```

        **Cycle Detection Example:**

        ```
        Prerequisites: [[1,0], [0,1]] (0 needs 1, 1 needs 0)

        Graph: 0 ⟷ 1 (cycle!)

        In-degree: [1, 1] (both have prerequisites)

        Queue: [] (no course with in-degree 0!)

        Result: []
        Completed: 0 < 2 (couldn't process all courses)

        Conclusion: Circular dependency detected!
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy to Medium)
    Master basic topological sort mechanics.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Course Schedule | Cycle detection | Can all courses be taken? | [LeetCode 207](https://leetcode.com/problems/course-schedule/) |
    | Course Schedule II | Find ordering | Return valid course order | [LeetCode 210](https://leetcode.com/problems/course-schedule-ii/) |
    | Find Eventual Safe States | DFS variant | Find nodes with no outgoing cycle paths | [LeetCode 802](https://leetcode.com/problems/find-eventual-safe-states/) |
    | Sort Items by Groups | Nested topological sort | Sort with group constraints | [LeetCode 1203](https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/) |
    | Build Order | Basic topo sort | Classic dependency problem | [CTCI 4.7] |

    **Goal:** Implement both Kahn's and DFS approaches, understand cycle detection.

    ---

    ### Phase 2: Application (Medium)
    Apply topo sort to various problem types.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Alien Dictionary | Build graph from strings | Derive character ordering from sorted words | [LeetCode 269](https://leetcode.com/problems/alien-dictionary/) |
    | Minimum Height Trees | Topological + pruning | Find center nodes by removing leaves | [LeetCode 310](https://leetcode.com/problems/minimum-height-trees/) |
    | Parallel Courses | Level-by-level processing | Find minimum semesters | [LeetCode 1136](https://leetcode.com/problems/parallel-courses/) |
    | Longest Increasing Path in Matrix | DFS with memoization | Find longest path (implicit DAG) | [LeetCode 329](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) |
    | Reconstruct Itinerary | DFS Eulerian path | Find path using all edges exactly once | [LeetCode 332](https://leetcode.com/problems/reconstruct-itinerary/) |

    **Goal:** Build graphs from non-obvious inputs, handle complex constraints.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex variations and optimizations.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Sequence Reconstruction | Verify unique ordering | Check if sequences define unique topo order | [LeetCode 444](https://leetcode.com/problems/sequence-reconstruction/) |
    | Parallel Courses II | With K constraint | Take at most K courses per semester | [LeetCode 1494](https://leetcode.com/problems/parallel-courses-ii/) |
    | Strange Printer II | 2D topological sort | Layer ordering with overlap constraints | [LeetCode 1591](https://leetcode.com/problems/strange-printer-ii/) |
    | Longest Path in DAG | Dynamic programming on topo order | Find longest weighted path | [Classic DP problem] |

    **Goal:** Combine topo sort with DP, handle optimization constraints.

    ---

    ## 🎯 Practice Strategy

    1. **Master Both Approaches:** Implement Kahn's and DFS from memory
    2. **Identify Dependencies:** Draw the graph before coding
    3. **Cycle Detection:** Always check if ordering is possible
    4. **Multiple Orderings:** Understand when multiple answers exist
    5. **Build Graphs:** Practice constructing graphs from problem descriptions
    6. **Test Edge Cases:** Empty graph, single node, cycles, disconnected components

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Wrong edge direction | Confuse "depends on" with "enables" | Be clear: [A, B] means B→A or A→B? Read problem carefully |
    | Not detecting cycles | Forget to check if result has all nodes | Always check: `len(result) == n` |
    | DFS without 3 states | Use only visited/unvisited | Use 3 states: unvisited(0), visiting(1), visited(2) for cycle detection |
    | Forgetting to reverse DFS result | Post-order is reverse of topo order | Always reverse DFS result: `result[::-1]` |
    | Not handling disconnected components | Process only one component | Loop through all unvisited nodes in DFS |

---

## Data Structure Patterns

### Top K Elements (Heap)

=== "Understanding the Pattern"

    ## 📖 What is Top K Elements?

    Imagine you're running a music streaming service and need to find the "Top 10 Songs of 2024" from millions of tracks. The naive approach? Sort all millions of songs by play count and take the top 10. That works, but why sort everything when you only need 10?

    Here's a smarter strategy: maintain a "playlist" of the current top 10. As you scan through songs:
    - If the playlist isn't full (< 10 songs), add the current song
    - If playlist is full and current song has more plays than the weakest song in the playlist, replace it
    - Otherwise, skip the song

    This is the **Top K Elements** pattern using a heap! Instead of sorting all n elements (O(n log n)), we maintain a heap of size K, processing each element in O(log K) time, for a total of O(n log K). When K is much smaller than n, this is dramatically faster!

    The counterintuitive trick: to find K largest, use a min-heap (not max-heap!). Why? So you can easily identify and remove the smallest of your "top K" when something better comes along.

    ---

    ## 🔧 How It Works

    Heaps are binary trees with the heap property: parent ≤ children (min-heap) or parent ≥ children (max-heap).

    **Finding K Largest with Min-Heap:**
    ```
    Array: [3, 2, 1, 5, 6, 4], K=2 (find 2 largest)

    Process 3: heap=[3]
    Process 2: heap=[2,3]  (heap full! size=K)
    Process 1: 1 < 2 (skip, it's smaller than heap min)
    Process 5: 5 > 2 (replace 2 with 5)
               heap=[3,5]
    Process 6: 6 > 3 (replace 3 with 6)
               heap=[5,6]

    Result: [5, 6] (the 2 largest elements)

    Total operations: n insertions × O(log K) = O(n log K)
    ```

    **Key Insight:**
    - Min-heap: smallest element at top (easy to remove)
    - For K largest: keep min-heap of size K, remove smallest when better element arrives
    - For K smallest: keep max-heap of size K, remove largest when smaller element arrives

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Why sort all n elements when only K matter? Heaps let us maintain "top K" without full sorting!

    Think of it like a reality TV show with K contestant spots:
    - New contestant auditions (new element arrives)
    - If spots available, they're in (heap not full, add element)
    - If all spots taken, they must beat the weakest contestant to join (compare with heap top)
    - Weakest contestant gets eliminated (pop from heap)

    **Why Min-Heap for K Largest?**

    Counterintuitive but brilliant! With a min-heap:
    - Top of heap = smallest of our "top K"
    - Any element smaller than heap top can't be in top K
    - Quick comparison: `if new_element > heap[0]`
    - Easy removal: pop the minimum

    With a max-heap, you'd have to search through the heap to find the smallest element to replace—O(K) every time!

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n log K) | n elements, each needs O(log K) heap operation |
    | **Space** | O(K) | Heap stores at most K elements |
    | **Improvement** | From O(n log n) | No need to sort all n elements |

    **When is this better?** When K << n (K much smaller than n). Example: K=100, n=1,000,000 → massive speedup!

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Top K (Heap) Works | Example |
    |----------|----------------------|---------|
    | **K largest/smallest elements** | Maintain only K elements, not all n | Kth Largest Element, K Closest Points |
    | **K most frequent items** | Heap on frequency counts | Top K Frequent Elements, Top K Words |
    | **Streaming data** | Process elements one by one, bounded memory | Median from Data Stream |
    | **K closest to target** | Custom comparator with heap | K Closest to X |
    | **Dynamic K-th statistic** | Need K-th element with insertions | Find K-th element in updates |

    **Red Flags That Suggest Top K (Heap):**
    - "Find K largest/smallest"
    - "Top K most frequent"
    - "K closest to X"
    - "K-th largest/smallest element"
    - "Maintain K elements" in a stream
    - Need partial sorting (not full sort)

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **K is very large (K ≈ n)** | Heap overhead not worth it | Just sort: O(n log n) |
    | **Need all elements sorted** | Heap gives top K, not full ordering | Full sort or priority queue |
    | **K changes frequently** | Rebuilding heap expensive | Different data structure |
    | **Need median of all elements** | Single heap insufficient | Two heaps (max + min) |
    | **Random access to K elements** | Heap doesn't support index access | Sorted array |

    ---

    ## 🎯 Decision Flowchart

    ```
    Do you need K largest/smallest/most-frequent elements?
    ├─ Yes → Is K much smaller than n (K << n)?
    │         ├─ Yes → USE HEAP (Top K) ✓
    │         └─ No (K ≈ n) → Just sort: O(n log n)
    └─ No → Do you need K-th element?
              ├─ Yes → USE HEAP (size K) ✓
              └─ No → Need median?
                       └─ Yes → USE TWO HEAPS ✓
    ```

=== "Implementation Templates"

    === "Template 1: K Largest Elements"

        **Use Case:** Find K largest elements in unsorted array

        **Pattern:** Maintain min-heap of size K, replace minimum when better element found

        ```python
        import heapq

        def find_k_largest(nums, k):
            """
            Find K largest elements in array.

            Perfect for: Top K problems where K << n

            Concept:
            - Use min-heap (smallest element at top)
            - Keep heap size = K
            - If new element > heap top, replace top
            - Result: heap contains K largest

            Time: O(n log K) - n elements, log K per operation
            Space: O(K) - Heap stores K elements

            Example: nums=[3,2,1,5,6,4], k=2 → [5,6]
            """
            # Min-heap (Python's heapq is min-heap by default)
            heap = []

            for num in nums:
                if len(heap) < k:
                    # Heap not full: just add
                    heapq.heappush(heap, num)
                elif num > heap[0]:
                    # Found element larger than smallest in heap
                    # Replace smallest (heap[0])
                    heapq.heapreplace(heap, num)
                # Else: num too small, skip it

            return heap  # or sorted(heap) for ordered result


        # Example Usage:
        nums = [3, 2, 1, 5, 6, 4]
        k = 2
        result = find_k_largest(nums, k)
        print(result)  # Output: [5, 6] (order may vary)
        ```

        **Key Points:**
        - Use MIN-heap for K largest (not max-heap!)
        - Heap top (heap[0]) is smallest of the K largest
        - heapreplace is atomic: pop + push in one operation
        - Result is unordered; sort if order matters

    === "Template 2: K Smallest Elements"

        **Use Case:** Find K smallest elements (opposite of K largest)

        **Pattern:** Maintain max-heap of size K (negate values for Python)

        ```python
        import heapq

        def find_k_smallest(nums, k):
            """
            Find K smallest elements in array.

            Perfect for: K closest to origin, K smallest distances

            Concept:
            - Use max-heap (largest element at top)
            - Python heapq is min-heap, so negate values
            - Keep heap size = K
            - If new element < heap top, replace top

            Time: O(n log K)
            Space: O(K)

            Example: nums=[3,2,1,5,6,4], k=2 → [1,2]
            """
            # Python has min-heap only, simulate max-heap by negating
            heap = []

            for num in nums:
                if len(heap) < k:
                    heapq.heappush(heap, -num)  # Negate for max-heap
                elif num < -heap[0]:  # Compare with largest (negated)
                    heapq.heapreplace(heap, -num)

            # Negate back to get original values
            return [-x for x in heap]


        # Alternative: Use min-heap directly (simpler!)
        def find_k_smallest_simple(nums, k):
            # Just heapify and pop K times
            heapq.heapify(nums)  # O(n)
            return [heapq.heappop(nums) for _ in range(k)]  # O(k log n)


        # Example Usage:
        nums = [3, 2, 1, 5, 6, 4]
        k = 2
        result = find_k_smallest(nums, k)
        print(result)  # Output: [2, 1] or [1, 2]
        ```

        **Key Points:**
        - For K smallest: use MAX-heap (keep largest of K smallest at top)
        - Python heapq is min-heap; negate values for max-heap
        - Alternative: heapify + pop K times (simpler but modifies input)

    === "Template 3: Top K Frequent Elements"

        **Use Case:** Find K most frequent elements

        **Pattern:** Count frequencies, use min-heap on (frequency, element) pairs

        ```python
        import heapq
        from collections import Counter

        def top_k_frequent(nums, k):
            """
            Find K most frequent elements.

            Perfect for: Top K words, most common items

            Concept:
            - Count frequencies with Counter
            - Use min-heap of size K on frequencies
            - Keep K elements with highest frequencies

            Time: O(n + m log K) where m = unique elements
            Space: O(m) for frequency map + O(K) for heap

            Example: nums=[1,1,1,2,2,3], k=2 → [1,2]
            """
            # Count frequencies
            freq = Counter(nums)

            # Min-heap on (frequency, element)
            heap = []

            for num, count in freq.items():
                if len(heap) < k:
                    heapq.heappush(heap, (count, num))
                elif count > heap[0][0]:  # Compare frequencies
                    heapq.heapreplace(heap, (count, num))

            # Extract elements (ignore frequencies)
            return [num for freq, num in heap]


        # Alternative: Using Python's nlargest (cleaner)
        def top_k_frequent_clean(nums, k):
            freq = Counter(nums)
            # nlargest returns k elements with largest counts
            return heapq.nlargest(k, freq.keys(), key=freq.get)


        # Example Usage:
        nums = [1, 1, 1, 2, 2, 3]
        k = 2
        result = top_k_frequent(nums, k)
        print(result)  # Output: [1, 2] (most frequent)
        ```

        **Key Points:**
        - Count frequencies first with Counter
        - Heap stores (frequency, element) tuples
        - Min-heap keeps K elements with highest frequencies
        - Python's nlargest/nsmallest are convenient alternatives

    === "Template 4: Median from Data Stream"

        **Use Case:** Maintain median as elements are added dynamically

        **Pattern:** Two heaps—max-heap for lower half, min-heap for upper half

        ```python
        import heapq

        class MedianFinder:
            """
            Find median from data stream using two heaps.

            Perfect for: Dynamic median, running median

            Concept:
            - Max-heap (negated): stores lower half of numbers
            - Min-heap: stores upper half of numbers
            - Balance heaps: |size_diff| ≤ 1
            - Median: middle element(s) at heap tops

            Time: O(log n) per insertion, O(1) for median
            Space: O(n) for storing all elements

            Example: [1,2] → median=1.5, [1,2,3] → median=2
            """
            def __init__(self):
                # Max-heap for lower half (negate for Python min-heap)
                self.small = []  # Stores smaller half (max-heap)
                # Min-heap for upper half
                self.large = []  # Stores larger half (min-heap)

            def addNum(self, num):
                """Add number to data structure."""
                # Always add to small (max-heap) first
                heapq.heappush(self.small, -num)

                # Balance: largest in small ≤ smallest in large
                if self.small and self.large and (-self.small[0] > self.large[0]):
                    val = -heapq.heappop(self.small)
                    heapq.heappush(self.large, val)

                # Balance sizes: difference ≤ 1
                if len(self.small) > len(self.large) + 1:
                    val = -heapq.heappop(self.small)
                    heapq.heappush(self.large, val)
                if len(self.large) > len(self.small) + 1:
                    val = heapq.heappop(self.large)
                    heapq.heappush(self.small, -val)

            def findMedian(self):
                """Return current median."""
                if len(self.small) > len(self.large):
                    return -self.small[0]  # Odd total: return from larger heap
                if len(self.large) > len(self.small):
                    return self.large[0]
                # Even total: average of both tops
                return (-self.small[0] + self.large[0]) / 2.0


        # Example Usage:
        mf = MedianFinder()
        mf.addNum(1)
        mf.addNum(2)
        print(mf.findMedian())  # 1.5
        mf.addNum(3)
        print(mf.findMedian())  # 2
        ```

        **Key Points:**
        - Two heaps divide data: small (max-heap) and large (min-heap)
        - small contains lower half, large contains upper half
        - Keep heaps balanced: size difference ≤ 1
        - Median from heap tops in O(1)

    === "Visual Walkthrough"

        **Problem:** Find 3 largest elements in [3, 2, 1, 5, 6, 4]

        ```
        Using min-heap (size K=3):

        Step 1: Process 3
        Heap: [3]
        Size < K, just add

        Step 2: Process 2
        Heap: [2, 3]
        Size < K, just add

        Step 3: Process 1
        Heap: [1, 2, 3]
        Size < K, just add (heap now full!)

        Step 4: Process 5
        5 > 1 (heap top) → Replace!
        Heap: [2, 3, 5]
        (1 removed, 5 added)

        Step 5: Process 6
        6 > 2 (heap top) → Replace!
        Heap: [3, 5, 6]
        (2 removed, 6 added)

        Step 6: Process 4
        4 > 3 (heap top) → Replace!
        Heap: [4, 5, 6]
        (3 removed, 4 added)

        Final Result: [4, 5, 6] (the 3 largest)

        Key Observation: We never stored more than K=3 elements!
        ```

        **Two Heaps for Median:**

        ```
        Stream: 1, 2, 3, 4, 5

        After adding 1:
        small (max): [1]    large (min): []
        Median: 1

        After adding 2:
        small (max): [1]    large (min): [2]
        Median: (1 + 2) / 2 = 1.5

        After adding 3:
        small (max): [1,2]  large (min): [3]
        Median: 2 (middle element)

        After adding 4:
        small (max): [1,2]  large (min): [3,4]
        Median: (2 + 3) / 2 = 2.5

        After adding 5:
        small (max): [1,2]  large (min): [3,4,5]
        Median: 3 (middle element)

        Invariant: small has lower half, large has upper half
        Max of small ≤ Min of large
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy to Medium)
    Master basic heap operations and K largest/smallest.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Kth Largest Element | Quickselect or heap | Find single K-th element | [LeetCode 215](https://leetcode.com/problems/kth-largest-element-in-an-array/) |
    | Last Stone Weight | Max-heap simulation | Heap operations practice | [LeetCode 1046](https://leetcode.com/problems/last-stone-weight/) |
    | K Closest Points to Origin | Custom comparator | Distance-based heap | [LeetCode 973](https://leetcode.com/problems/k-closest-points-to-origin/) |
    | Top K Frequent Elements | Frequency heap | Count + heap combination | [LeetCode 347](https://leetcode.com/problems/top-k-frequent-elements/) |
    | Kth Largest in Stream | Maintain heap | Dynamic K-th largest | [LeetCode 703](https://leetcode.com/problems/kth-largest-element-in-a-stream/) |

    **Goal:** Understand min-heap vs max-heap for K largest vs K smallest.

    ---

    ### Phase 2: Application (Medium)
    Apply heaps to more complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Top K Frequent Words | Frequency + lexicographic | Custom comparator | [LeetCode 692](https://leetcode.com/problems/top-k-frequent-words/) |
    | K Closest to X | Distance from target | Absolute difference comparison | [LeetCode 658](https://leetcode.com/problems/find-k-closest-elements/) |
    | Sort Characters by Frequency | Frequency sorting | Count + heap | [LeetCode 451](https://leetcode.com/problems/sort-characters-by-frequency/) |
    | Reorganize String | Greedy with max-heap | Schedule characters to avoid adjacency | [LeetCode 767](https://leetcode.com/problems/reorganize-string/) |
    | Ugly Number II | Multiple heaps | Generate sequence using heaps | [LeetCode 264](https://leetcode.com/problems/ugly-number-ii/) |

    **Goal:** Master custom comparators and frequency-based heaps.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex multi-heap and streaming problems.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Find Median from Data Stream | Two heaps | Maintain median dynamically | [LeetCode 295](https://leetcode.com/problems/find-median-from-data-stream/) |
    | Sliding Window Median | Two heaps + removal | Dynamic median with window | [LeetCode 480](https://leetcode.com/problems/sliding-window-median/) |
    | IPO | Greedy + two heaps | Maximize capital with constraints | [LeetCode 502](https://leetcode.com/problems/ipo/) |
    | Employee Free Time | Merge intervals + heap | K-way merge variant | [LeetCode 759](https://leetcode.com/problems/employee-free-time/) |

    **Goal:** Combine heaps with other techniques (sliding window, greedy).

    ---

    ## 🎯 Practice Strategy

    1. **Master Heap Basics:** Understand heappush, heappop, heapify
    2. **Min vs Max:** Know when to use min-heap (K largest) vs max-heap (K smallest)
    3. **Size Maintenance:** Keep heap size exactly K (not larger!)
    4. **Custom Comparators:** Practice tuples for custom ordering
    5. **Two Heaps:** Master median technique with balanced heaps
    6. **Analyze Complexity:** Verify K << n before using heap

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Using max-heap for K largest | Intuition says "largest = max-heap" | Use MIN-heap to easily identify smallest of K largest |
    | Letting heap grow unbounded | Forgetting to limit size to K | Check `len(heap) < k` before pushing |
    | Not negating for max-heap | Python heapq only supports min-heap | Negate values: `heappush(heap, -val)` |
    | Wrong complexity analysis | Think heap solution always better | Heap only wins when K << n; otherwise sort |
    | Modifying input for heapify | heapify modifies the list | Copy list first if you need original |

---

### K-way Merge

=== "Understanding the Pattern"

    ## 📖 What is K-way Merge?

    Imagine you're a teacher grading exams from K different classes, and each class's exams are already sorted by score (worst to best). You want to create a single ranked list of all students across all classes. The naive approach? Look at all K classes every time to find the next lowest score—that's checking K positions for every single exam!

    Here's the clever insight: you only need to compare the *front* of each class (the next unprocessed exam from each). Use a min-heap to track these K "front" candidates. Pop the smallest, and when you take an exam from a class, add the next exam from that same class to the heap. You're always comparing only K elements (the frontrunners from each class), not all N total exams!

    This is **K-way Merge**: efficiently merging K sorted sequences by using a min-heap to track the smallest unprocessed element from each sequence. Instead of repeatedly scanning all K lists (O(k) per element → O(nk) total), the heap gives you the minimum in O(log k), making the total time O(n log k).

    **Real-World Analogy:**
    - Merging K sorted log files from different servers into one timeline
    - Combining search results from K different databases (each pre-sorted by relevance)
    - Streaming K sorted playlists into a single sorted playlist

    The magic: the heap size stays at most K, regardless of how many total elements (N) you're processing!

    ---

    ## 🔧 How It Works

    K-way Merge maintains a min-heap containing one candidate element from each of the K lists.

    **Core Algorithm:**
    ```
    1. Initialize heap with first element from each list (K elements)
    2. While heap not empty:
       a. Pop smallest element from heap → add to result
       b. If that element's list has more elements:
          - Add next element from same list to heap
    3. Repeat until all lists exhausted
    ```

    **Visual Example:**
    ```
    K=3 sorted lists:
    List 0: [1, 4, 7]
    List 1: [2, 5, 8]
    List 2: [3, 6, 9]

    Step 1: Initialize heap with first element from each
    Heap: [(1, list:0, idx:0), (2, list:1, idx:0), (3, list:2, idx:0)]
    Result: []

    Step 2: Pop min (1 from list 0), add next from list 0 (4)
    Heap: [(2, list:1, idx:0), (3, list:2, idx:0), (4, list:0, idx:1)]
    Result: [1]

    Step 3: Pop min (2 from list 1), add next from list 1 (5)
    Heap: [(3, list:2, idx:0), (4, list:0, idx:1), (5, list:1, idx:1)]
    Result: [1, 2]

    Step 4: Pop min (3 from list 2), add next from list 2 (6)
    Heap: [(4, list:0, idx:1), (5, list:1, idx:1), (6, list:2, idx:1)]
    Result: [1, 2, 3]

    Step 5-9: Continue until all elements processed
    Result: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ```

    **Why Use a Heap?**

    Without heap: For each of N total elements, scan K lists to find minimum → O(nk)
    With heap: For each of N elements, pop and push in heap → O(n log k)

    When K is large, `O(n log k)` is much better than `O(nk)`!

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Since each list is already sorted, you only need to compare the *next unprocessed element* from each list. A min-heap tracks these K candidates efficiently!

    Think of K checkout lines at a grocery store:
    - Each line is sorted (first person closest to cashier)
    - To find who checks out next overall, compare the first person in each line
    - After someone checks out from line 3, the next person in line 3 becomes the candidate

    **Why This Pattern is Powerful:**

    K-way merge is the generalization of 2-way merge (used in merge sort). But instead of merging pairs recursively, we merge all K lists simultaneously using a heap. This is more efficient when K is large:
    - Recursive 2-way: O(n log k) with log k merge passes
    - K-way with heap: O(n log k) in a single pass

    **Key Insight:** Heap size is only K (not N). Even with millions of total elements, if K=100, heap operations stay at O(log 100) ≈ 6-7 comparisons. That's why this scales!

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(n log k) | N elements, each requires O(log k) heap operation |
    | **Space** | O(k) | Heap stores at most K elements (one per list) |
    | **Improvement** | From O(nk) | Comparing all K lists every time vs heap lookup |

    **Breakdown:**
    - N = total number of elements across all lists
    - K = number of lists
    - Each element: O(log k) to pop from heap + O(log k) to push next element
    - Total: N × O(log k) = O(n log k)

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why K-way Merge Works | Example |
    |----------|---------------------|---------|
    | **Merge K sorted lists/arrays** | Heap efficiently tracks K frontrunners | Merge K Sorted Lists (LC 23) |
    | **Kth smallest in sorted matrix** | Each row is sorted, treat as K lists | Kth Smallest in Sorted Matrix (LC 378) |
    | **Smallest range covering K lists** | Track min/max from K current elements | Smallest Range (LC 632) |
    | **Merge sorted streams** | Real-time merging of K data sources | Log aggregation, database merge |
    | **External sorting** | Merge K sorted chunks from disk | Large dataset sorting |

    **Red Flags That Suggest K-way Merge:**
    - Problem mentions "K sorted arrays/lists"
    - Need to merge multiple sorted sequences
    - Finding Kth element in sorted structure
    - "Smallest range" or "common element" across K lists
    - Each individual sequence is sorted (critical!)

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Lists not sorted** | Can't leverage sorted property | Sort first O(n log n), or use different approach |
    | **Only 2 lists (K=2)** | Simpler two-pointer merge is better | Two pointers O(n) vs heap O(n log 2) = overhead |
    | **Need random access** | Heap only gives smallest, not arbitrary elements | Use indices or other data structure |
    | **All lists in memory** | May be faster to concatenate + sort | O(n log n) might beat O(n log k) for small K |
    | **K is very small (K < 3)** | Heap overhead not worth it | Manual comparison or two-pointer merge |

    ---

    ## 🎯 Decision Flowchart

    ```
    Do you have K sorted sequences to merge?
    ├─ Yes → Is K > 2?
    │         ├─ Yes → Are lists large (N >> K)?
    │         │         ├─ Yes → USE K-WAY MERGE ✓
    │         │         └─ No → Consider concatenate + sort
    │         └─ No (K=2) → Use two-pointer merge
    └─ No → Are lists unsorted?
              └─ Sort first, then K-way merge
    ```

=== "Implementation Templates"

    === "Template 1: Merge K Sorted Arrays"

        **Use Case:** Merge K sorted arrays into one sorted array

        **Pattern:** Min-heap tracks smallest unprocessed element from each array

        ```python
        import heapq

        def merge_k_sorted_arrays(arrays):
            """
            Merge K sorted arrays into one sorted array.

            Perfect for: Combining multiple sorted data sources

            Concept:
            - Heap stores (value, array_index, element_index)
            - Always pop smallest value
            - When popping from array i, push next element from array i

            Time: O(n log k) - n elements, log k per heap operation
            Space: O(k) - heap size is at most k

            Example: Merge K Sorted Lists variant
            """
            min_heap = []

            # Initialize heap with first element from each array
            for array_idx, array in enumerate(arrays):
                if array:  # Check array is not empty
                    # Push (value, array_index, element_index)
                    heapq.heappush(min_heap, (array[0], array_idx, 0))

            result = []

            # Process heap until empty
            while min_heap:
                # Pop smallest element
                value, array_idx, elem_idx = heapq.heappop(min_heap)
                result.append(value)

                # Add next element from same array (if exists)
                next_idx = elem_idx + 1
                if next_idx < len(arrays[array_idx]):
                    next_value = arrays[array_idx][next_idx]
                    heapq.heappush(min_heap, (next_value, array_idx, next_idx))

            return result


        # Example Usage:
        arrays = [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ]
        result = merge_k_sorted_arrays(arrays)
        print(result)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ```

        **Key Points:**
        - Tuple format: `(value, array_idx, elem_idx)` - Python heaps sort by first element
        - Check if array is non-empty before initial push
        - Only add next element from the array we just popped from

    === "Template 2: Merge K Sorted Linked Lists"

        **Use Case:** LeetCode 23 - Merge K sorted linked lists

        **Pattern:** Heap stores list nodes, custom comparison by value

        ```python
        import heapq

        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def merge_k_lists(lists):
            """
            Merge K sorted linked lists into one sorted list.

            Perfect for: Merge K Sorted Lists (LC 23)

            Concept:
            - Heap stores (node.val, unique_id, node)
            - unique_id breaks ties (prevents comparing nodes directly)
            - Pop smallest node, add its next if exists

            Time: O(n log k) - n total nodes, log k per operation
            Space: O(k) - heap size

            Example: [[1,4,5],[1,3,4],[2,6]] → [1,1,2,3,4,4,5,6]
            """
            min_heap = []

            # Initialize heap with head of each list
            for idx, node in enumerate(lists):
                if node:
                    # (value, unique_id, node)
                    # unique_id ensures no comparison of nodes
                    heapq.heappush(min_heap, (node.val, idx, node))

            # Dummy head for result list
            dummy = ListNode(0)
            current = dummy

            # Process heap
            while min_heap:
                val, idx, node = heapq.heappop(min_heap)

                # Add to result list
                current.next = node
                current = current.next

                # Add next node from same list
                if node.next:
                    heapq.heappush(min_heap, (node.next.val, idx, node.next))

            return dummy.next


        # Example Usage:
        # Create lists: [1,4,5], [1,3,4], [2,6]
        list1 = ListNode(1, ListNode(4, ListNode(5)))
        list2 = ListNode(1, ListNode(3, ListNode(4)))
        list3 = ListNode(2, ListNode(6))

        result = merge_k_lists([list1, list2, list3])

        # Print result: 1 → 1 → 2 → 3 → 4 → 4 → 5 → 6
        while result:
            print(result.val, end=" → " if result.next else "\n")
            result = result.next
        ```

        **Key Points:**
        - Use unique ID to avoid comparing nodes (Python heap compares tuples element-wise)
        - Dummy head simplifies list construction
        - Each node appears in heap at most once

    === "Template 3: Kth Smallest in Sorted Matrix"

        **Use Case:** Find Kth smallest element in row-wise and column-wise sorted matrix

        **Pattern:** Treat each row as a sorted list, use K-way merge up to K elements

        ```python
        import heapq

        def kth_smallest_in_matrix(matrix, k):
            """
            Find Kth smallest element in sorted matrix.

            Perfect for: Kth Smallest Element in Sorted Matrix (LC 378)

            Concept:
            - Each row is a sorted list
            - Use K-way merge, but stop after K elements
            - Heap tracks (value, row, col)

            Time: O(k log min(k,n)) - worst case process k elements
            Space: O(min(k, n)) - heap size

            Example: matrix = [[1,5,9],[10,11,13],[12,13,15]], k=8 → 13
            """
            if not matrix or not matrix[0]:
                return None

            n = len(matrix)  # Assume square matrix
            min_heap = []

            # Initialize heap with first element of each row
            # (but only up to k rows if matrix has more than k rows)
            for row in range(min(n, k)):
                heapq.heappush(min_heap, (matrix[row][0], row, 0))

            # Pop k-1 times, kth pop is the answer
            count = 0
            result = 0

            while min_heap and count < k:
                value, row, col = heapq.heappop(min_heap)
                count += 1
                result = value

                # If this was the kth element, we're done
                if count == k:
                    return result

                # Add next element from same row
                if col + 1 < len(matrix[row]):
                    heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))

            return result


        # Example Usage:
        matrix = [
            [1,  5,  9],
            [10, 11, 13],
            [12, 13, 15]
        ]
        k = 8
        result = kth_smallest_in_matrix(matrix, k)
        print(result)  # Output: 13

        # Elements in order: 1, 5, 9, 10, 11, 12, 13, 13
        #                                          ↑ 8th
        ```

        **Key Points:**
        - Only need to process K elements (can stop early)
        - Initialize heap with min(K, N) rows to avoid unnecessary elements
        - Optimization: for very large matrices, binary search might be faster

    === "Visual Walkthrough"

        **Problem:** Merge 3 sorted arrays: [1,4,7], [2,5,8], [3,6,9]

        ```
        Arrays:
        A: [1, 4, 7]
        B: [2, 5, 8]
        C: [3, 6, 9]

        Step 1: Initialize heap with first element from each
        ────────────────────────────────────────────────────
        Heap: [(1,A,0), (2,B,0), (3,C,0)]
        Result: []

        Step 2: Pop (1,A,0), add next from A
        ──────────────────────────────────────
        Pop: 1 from array A
        Add to result: [1]
        Push: (4,A,1) to heap
        Heap: [(2,B,0), (3,C,0), (4,A,1)]
               ↑ new min

        Step 3: Pop (2,B,0), add next from B
        ──────────────────────────────────────
        Pop: 2 from array B
        Add to result: [1, 2]
        Push: (5,B,1) to heap
        Heap: [(3,C,0), (4,A,1), (5,B,1)]
               ↑ new min

        Step 4: Pop (3,C,0), add next from C
        ──────────────────────────────────────
        Pop: 3 from array C
        Add to result: [1, 2, 3]
        Push: (6,C,1) to heap
        Heap: [(4,A,1), (5,B,1), (6,C,1)]
               ↑ new min

        Step 5: Pop (4,A,1), add next from A
        ──────────────────────────────────────
        Pop: 4 from array A
        Add to result: [1, 2, 3, 4]
        Push: (7,A,2) to heap
        Heap: [(5,B,1), (6,C,1), (7,A,2)]

        Continue until heap empty...

        Final Result: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ```

        **Why This Works:**

        At each step, heap contains exactly one element from each array (the next unprocessed one). Popping gives the smallest among these candidates. When we pop from array X, we immediately add the next element from array X, maintaining the invariant: "heap contains next candidate from each array."

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy to Medium)
    Master basic K-way merge mechanics.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Merge Two Sorted Lists | K=2 baseline | Understand merge logic before scaling to K | [LeetCode 21](https://leetcode.com/problems/merge-two-sorted-lists/) |
    | Merge Sorted Array | In-place K=2 | Merging with space constraints | [LeetCode 88](https://leetcode.com/problems/merge-sorted-array/) |
    | Kth Smallest Element in Sorted Matrix | Matrix as K lists | Treat rows as sorted lists | [LeetCode 378](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) |
    | Find K Pairs with Smallest Sums | Pair generation | Generate K candidates, merge | [LeetCode 373](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/) |
    | Ugly Number II | Generated sequences | Merge dynamically generated sequences | [LeetCode 264](https://leetcode.com/problems/ugly-number-ii/) |

    **Goal:** Solve all 5 problems. Understand heap initialization with first elements. Practice tuple ordering in heap.

    ---

    ### Phase 2: Application (Medium)
    Apply K-way merge to various scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Merge K Sorted Lists | Classic K-way | Merge K linked lists with heap | [LeetCode 23](https://leetcode.com/problems/merge-k-sorted-lists/) |
    | Smallest Range Covering K Lists | Min-max tracking | Track both min and max in heap | [LeetCode 632](https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/) |
    | Find Median from Data Stream | Dynamic K=2 merge | Maintain two heaps (max and min) | [LeetCode 295](https://leetcode.com/problems/find-median-from-data-stream/) |
    | Kth Smallest in Multiplication Table | Virtual K lists | Lists not explicit, use math | [LeetCode 668](https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/) |
    | Super Ugly Number | Multiple sequences | Merge K sequences with different multipliers | [LeetCode 313](https://leetcode.com/problems/super-ugly-number/) |

    **Goal:** Handle linked lists, track additional state (like max). Understand when lists are virtual vs explicit.

    ---

    ### Phase 3: Mastery (Hard)
    Combine K-way merge with advanced techniques.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Merge K Sorted Lists (optimal) | Divide and conquer | Compare heap vs recursive merge | [LeetCode 23](https://leetcode.com/problems/merge-k-sorted-lists/) |
    | Sliding Window Median | K-way with deletions | Maintain two heaps with removals | [LeetCode 480](https://leetcode.com/problems/sliding-window-median/) |
    | Maximum Subarray Sum with One Deletion | Dynamic programming + merge | Merge DP states | [LeetCode 1186](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/) |
    | Count of Smaller Numbers After Self | Merge sort with counts | K-way merge for divide-conquer counting | [LeetCode 315](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) |

    **Goal:** Optimize K-way merge. Handle dynamic insertions/deletions. Combine with other patterns (DP, divide-conquer).

    ---

    ## 🎯 Practice Strategy

    1. **Start with K=2:** Solve "Merge Two Sorted Lists" first. Understand basic merge logic without heap complexity.
    2. **Visualize the Heap:** For first 3 K-way problems, draw the heap at each step. Track which elements are candidates.
    3. **Tuple Ordering:** Practice creating heap entries: `(value, list_id, index)`. Understand Python compares tuples element-by-element.
    4. **Optimize Space:** Recognize that heap size is always ≤ K, even if total elements is millions.
    5. **Time Yourself:** After solving once, re-solve from scratch in under 15 minutes.
    6. **Review After 24 Hours:** Re-code solutions next day. Focus on heap initialization and next-element logic.

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Comparing nodes directly** | Python can't compare custom objects in heap | Use tuple: `(node.val, unique_id, node)` with unique ID |
    | **Not checking empty arrays** | Trying to push from empty list | Check `if array:` before initial push |
    | **Wrong heap element format** | Forgetting to track array index | Always include: `(value, array_idx, elem_idx)` |
    | **Heap too large** | Adding all elements upfront | Only add first element from each list, then add as you pop |
    | **Not handling list exhaustion** | Pushing when no next element | Check `if next_idx < len(array)` before pushing |

---

### Trie (Prefix Tree)

=== "Understanding the Pattern"

    ## 📖 What is Trie?

    Imagine you're building a phone's autocomplete feature. When someone types "app", you need to instantly suggest "apple", "application", "appointment"—all words starting with "app". The naive approach? Store all 10,000 words in a list and scan through each one checking if it starts with "app". That's slow!

    Here's a clever insight: words that share prefixes can share structure! Think of how a dictionary organizes words: "apple", "application", and "apply" are all near each other because they start with "app". We can build a tree where paths from root to leaves spell out words, and siblings share common ancestors (prefixes).

    This is a **Trie** (pronounced "try", from re**trie**val). It's a tree where:
    - Root represents empty string
    - Each edge is labeled with a character
    - Paths from root spell words
    - Nodes with common prefixes share ancestors

    The magic: checking if "apple" exists or finding all words starting with "app" takes time proportional to the length of "apple" or "app"—not the total number of words! It's like following a single path in a map rather than checking every destination.

    ---

    ## 🔧 How It Works

    A Trie stores strings by breaking them into characters and sharing common prefixes.

    **Trie Structure:**
    ```
    Words: ["app", "apple", "apply", "cat", "car"]

    Trie representation:
                    root
                  /      \
                a          c
                |          |
                p          a
                |         / \
                p        t   r
              /   \
            l      l
            |      |
            e      y
         [end]  [end]

    Paths:
    - a→p→p = "app" [end marker]
    - a→p→p→l→e = "apple" [end marker]
    - a→p→p→l→y = "apply" [end marker]
    - c→a→t = "cat" [end marker]
    - c→a→r = "car" [end marker]

    Notice: "app", "apple", "apply" share "app" prefix!
    ```

    **Operations:**

    **Insert "apple":**
    ```
    Start at root
    → Create/follow 'a' edge
    → Create/follow 'p' edge
    → Create/follow 'p' edge
    → Create/follow 'l' edge
    → Create/follow 'e' edge
    → Mark node as end of word
    ```

    **Search "app":**
    ```
    Start at root
    → Follow 'a' edge (exists? yes)
    → Follow 'p' edge (exists? yes)
    → Follow 'p' edge (exists? yes)
    → Is node marked end? yes → Found!
    ```

    **Prefix Search "ap":**
    ```
    Start at root
    → Follow 'a' edge (exists? yes)
    → Follow 'p' edge (exists? yes)
    → Reached "ap" node
    → Collect all words in subtree: ["app", "apple", "apply"]
    ```

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Instead of storing complete words separately, share the common parts (prefixes)! This makes prefix operations incredibly fast.

    Think of a family tree:
    - Siblings share parents (common ancestors)
    - Cousins share grandparents
    - The closer the relationship, the more shared ancestry

    Similarly in Trie:
    - Words with same prefix share nodes
    - Longer common prefix = more shared path
    - Each character extends the path by one node

    **Why Prefix Queries are Fast:**

    To find all words starting with "app":
    1. Navigate to the "app" node: O(3) for "app" length
    2. Collect all complete words in that subtree

    No matter if you have 10 or 10,000 words, reaching "app" node takes same time—only depends on prefix length!

    **Space Efficiency:**

    Words with common prefixes share nodes. Example:
    - Storing ["app", "apple", "apply"] separately: 14 characters
    - In Trie: 3 (root→a→p→p) + 2 (→l→e) + 2 (→l→y) = 7 nodes
    - Prefix "app" stored once, not three times!

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Insert** | O(m) | m = word length, create/follow m edges |
    | **Search** | O(m) | m = word length, follow m edges |
    | **Prefix Search** | O(m + n) | m = prefix length, n = results to return |
    | **Space** | O(ALPHABET × N × M) | Worst case: N words, length M, no shared prefixes |

    **Best case space:** O(total characters) when many shared prefixes
    **Worst case space:** O(N × M × ALPHABET) when no shared prefixes and sparse children

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Trie Works | Example |
    |----------|---------------|---------|
    | **Autocomplete** | Find all words with prefix instantly | Phone keyboards, search bars |
    | **Spell Checker** | Fast word lookup and suggestion | Word processors, browsers |
    | **IP Routing** | Longest prefix matching | Network routers |
    | **Dictionary Operations** | Insert/search/prefix in O(m) time | Scrabble validators, crossword tools |
    | **Word Games** | Find valid words quickly | Boggle, Word Search |

    **Red Flags That Suggest Trie:**
    - "Words starting with prefix..."
    - "Autocomplete" or "suggestions"
    - "Word dictionary" with many lookups
    - "Prefix matching"
    - "Find all words in a 2D grid" (Word Search II)
    - Multiple words with shared prefixes

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Few words, infrequent queries** | Trie overhead not worth it | Simple array or hash set |
    | **No prefix queries** | Just need exact match | Hash set O(m) with less space |
    | **Suffix/substring search** | Trie only helps with prefixes | Suffix tree, suffix array |
    | **Range queries** | Trie doesn't maintain order | BST or sorted array |
    | **Very long strings** | Space consumption too high | Compressed trie or hash-based |

    ---

    ## 🎯 Decision Flowchart

    ```
    Do you need to search by prefix?
    ├─ Yes → Do words share common prefixes?
    │         ├─ Yes → Many prefix queries expected?
    │         │        ├─ Yes → USE TRIE ✓
    │         │        └─ No → Simple iteration might be okay
    │         └─ No → Consider space cost vs benefit
    └─ No → Only exact word match needed?
              ├─ Yes → USE HASH SET (simpler)
              └─ No → Need suffix/substring search?
                       └─ Yes → Use suffix tree
    ```

=== "Implementation Templates"

    === "Template 1: Basic Trie"

        **Use Case:** Dictionary with insert, search, and prefix checking

        **Pattern:** Tree of nodes, each node has children map + end marker

        ```python
        class TrieNode:
            """Node in Trie: has children and end-of-word marker."""
            def __init__(self):
                self.children = {}  # Map: char → TrieNode
                self.is_end_of_word = False


        class Trie:
            """
            Prefix tree for efficient string operations.

            Perfect for: Autocomplete, spell check, word dictionary

            Operations:
            - Insert: Add word to trie
            - Search: Check if exact word exists
            - StartsWith: Check if prefix exists

            Time: O(m) for all operations (m = word/prefix length)
            Space: O(ALPHABET × N × M) worst case

            Example: Dictionary with prefix search
            """
            def __init__(self):
                self.root = TrieNode()

            def insert(self, word):
                """
                Insert word into trie.

                Time: O(m) where m = len(word)
                """
                node = self.root

                # Traverse/create path for each character
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]

                # Mark last node as end of word
                node.is_end_of_word = True

            def search(self, word):
                """
                Search for exact word in trie.

                Returns True only if word exists AND is marked as end.

                Time: O(m) where m = len(word)
                """
                node = self.root

                # Follow path for each character
                for char in word:
                    if char not in node.children:
                        return False  # Path doesn't exist
                    node = node.children[char]

                # Found path, but is it a complete word?
                return node.is_end_of_word

            def starts_with(self, prefix):
                """
                Check if any word starts with prefix.

                Time: O(m) where m = len(prefix)
                """
                node = self.root

                # Follow path for each character
                for char in prefix:
                    if char not in node.children:
                        return False  # Prefix doesn't exist
                    node = node.children[char]

                # Found complete prefix path
                return True


        # Example Usage:
        trie = Trie()
        trie.insert("apple")
        print(trie.search("apple"))      # True
        print(trie.search("app"))        # False (not marked as end)
        print(trie.starts_with("app"))   # True
        trie.insert("app")
        print(trie.search("app"))        # True (now marked as end)
        ```

        **Key Points:**
        - Each node has dictionary of children (char → node)
        - is_end_of_word distinguishes "app" from "apple" prefix
        - Search requires path exists AND end marker
        - StartsWith only needs path to exist

    === "Template 2: Autocomplete/Word Suggestions"

        **Use Case:** Find all words with given prefix

        **Pattern:** Navigate to prefix node, collect all complete words in subtree

        ```python
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end_of_word = False


        class AutocompleteTrie:
            """
            Trie with autocomplete functionality.

            Perfect for: Search suggestions, autocomplete systems

            Concept:
            - Navigate to prefix node: O(prefix_len)
            - DFS collect all complete words in subtree
            - Can limit results to top K

            Time: O(m + n) where m = prefix len, n = results
            Space: O(total chars in result words)

            Example: Type "app" → suggest ["app", "apple", "application"]
            """
            def __init__(self):
                self.root = TrieNode()

            def insert(self, word):
                """Insert word into trie."""
                node = self.root
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                node.is_end_of_word = True

            def autocomplete(self, prefix):
                """
                Find all words starting with prefix.

                Returns list of complete words.
                """
                # Step 1: Navigate to prefix node
                node = self.root
                for char in prefix:
                    if char not in node.children:
                        return []  # Prefix doesn't exist
                    node = node.children[char]

                # Step 2: Collect all words in subtree
                words = []
                self._dfs_collect(node, prefix, words)
                return words

            def _dfs_collect(self, node, current_word, words):
                """
                DFS to collect all complete words in subtree.

                current_word: word formed so far
                words: list to accumulate results
                """
                # If this node is end of word, add it
                if node.is_end_of_word:
                    words.append(current_word)

                # Explore all children
                for char, child_node in node.children.items():
                    self._dfs_collect(child_node, current_word + char, words)


        # Example Usage:
        trie = AutocompleteTrie()
        words = ["app", "apple", "application", "apply", "cat", "car"]
        for word in words:
            trie.insert(word)

        print(trie.autocomplete("app"))  # ["app", "apple", "application", "apply"]
        print(trie.autocomplete("ca"))   # ["cat", "car"]
        print(trie.autocomplete("b"))    # []
        ```

        **Key Points:**
        - Navigate to prefix: O(prefix_len)
        - DFS from prefix node to collect words
        - Can optimize by limiting depth or result count
        - Used in real autocomplete systems

    === "Template 3: Word Search with Wildcard"

        **Use Case:** Search with wildcard character (e.g., "a.p" matches "app", "amp")

        **Pattern:** DFS with branching on wildcard

        ```python
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end_of_word = False


        class WordDictionary:
            """
            Word dictionary with wildcard search support.

            Perfect for: Pattern matching, regex-like search

            Concept:
            - Regular char: follow single path
            - Wildcard '.': try all possible children (DFS)

            Time: O(m) for regular search, O(26^m) worst for wildcards
            Space: O(ALPHABET × N × M)

            Example: "b.d" matches "bad", "bed", "bid"
            """
            def __init__(self):
                self.root = TrieNode()

            def add_word(self, word):
                """Add word to dictionary."""
                node = self.root
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                node.is_end_of_word = True

            def search(self, word):
                """
                Search for word (can contain '.' wildcard).

                '.' matches any single character.

                Returns True if word exists (accounting for wildcards).
                """
                return self._search_helper(word, 0, self.root)

            def _search_helper(self, word, index, node):
                """
                Recursive helper for wildcard search.

                index: current position in word
                node: current Trie node
                """
                # Reached end of word?
                if index == len(word):
                    return node.is_end_of_word

                char = word[index]

                if char == '.':
                    # Wildcard: try ALL children
                    for child in node.children.values():
                        if self._search_helper(word, index + 1, child):
                            return True  # Found match via this child
                    return False  # No child led to match

                else:
                    # Regular character: follow single path
                    if char not in node.children:
                        return False  # Path doesn't exist
                    return self._search_helper(word, index + 1, node.children[char])


        # Example Usage:
        wd = WordDictionary()
        wd.add_word("bad")
        wd.add_word("dad")
        wd.add_word("mad")

        print(wd.search("pad"))    # False
        print(wd.search("bad"))    # True
        print(wd.search(".ad"))    # True (matches bad, dad, mad)
        print(wd.search("b.."))    # True (matches bad)
        ```

        **Key Points:**
        - Regular char: single path traversal
        - Wildcard: branch to all children (exponential worst case)
        - Recursive DFS naturally handles wildcards
        - Common in spell checkers with fuzzy matching

    === "Visual Walkthrough"

        **Problem:** Build Trie for ["app", "apple", "apply", "cat"]

        ```
        Step 1: Insert "app"
                root
                 |
                 a
                 |
                 p
                 |
                 p [END]

        Step 2: Insert "apple"
                root
                 |
                 a
                 |
                 p
                 |
                 p [END]
                 |
                 l
                 |
                 e [END]

        Notice: "apple" shares "app" prefix!

        Step 3: Insert "apply"
                root
                 |
                 a
                 |
                 p
                 |
                 p [END]
                / \
               l   l
               |   |
               e   y
            [END] [END]

        Step 4: Insert "cat"
                    root
                   /    \
                  a      c
                  |      |
                  p      a
                  |      |
                  p      t
                / \    [END]
               l   l
               |   |
               e   y
            [END] [END]

        Final Trie:
        - "app", "apple", "apply" share path a→p→p
        - "cat" has separate branch
        - [END] markers distinguish complete words
        ```

        **Search Example:** Search for "apple"

        ```
        Start at root
        → Follow 'a' edge ✓
        → Follow 'p' edge ✓
        → Follow 'p' edge ✓
        → Follow 'l' edge ✓
        → Follow 'e' edge ✓
        → Check [END] marker? YES ✓

        Result: "apple" found!

        vs. Search for "app"
        Same path to 'p'→'p' node
        → Check [END] marker? YES ✓

        vs. Search for "appl"
        Same path to 'p'→'p'→'l' node
        → Check [END] marker? NO ✗
        → "appl" NOT a complete word (just a prefix)
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy to Medium)
    Master basic Trie operations.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Implement Trie | Basic template | Insert, search, startsWith | [LeetCode 208](https://leetcode.com/problems/implement-trie-prefix-tree/) |
    | Longest Common Prefix | Trie traversal | Find shared prefix path | [LeetCode 14](https://leetcode.com/problems/longest-common-prefix/) |
    | Design Add and Search Words | Wildcard search | Handle '.' wildcard with DFS | [LeetCode 211](https://leetcode.com/problems/design-add-and-search-words-data-structure/) |
    | Maximum XOR of Two Numbers | Bit trie | Trie on binary representation | [LeetCode 421](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) |
    | Implement Magic Dictionary | Near-match search | Search with one character difference | [LeetCode 676](https://leetcode.com/problems/implement-magic-dictionary/) |

    **Goal:** Implement Trie from scratch, understand node structure and traversal.

    ---

    ### Phase 2: Application (Medium)
    Apply Trie to word search and autocomplete problems.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Search Suggestions System | Autocomplete | Return top 3 products for each prefix | [LeetCode 1268](https://leetcode.com/problems/search-suggestions-system/) |
    | Replace Words | Prefix matching | Replace with shortest root | [LeetCode 648](https://leetcode.com/problems/replace-words/) |
    | Map Sum Pairs | Trie with values | Store and sum values by prefix | [LeetCode 677](https://leetcode.com/problems/map-sum-pairs/) |
    | Prefix and Suffix Search | Trie concatenation | Match both prefix and suffix | [LeetCode 745](https://leetcode.com/problems/prefix-and-suffix-search/) |
    | Stream of Characters | Suffix trie | Check suffix against stream | [LeetCode 1032](https://leetcode.com/problems/stream-of-characters/) |

    **Goal:** Use Trie for practical applications like autocomplete and prefix matching.

    ---

    ### Phase 3: Mastery (Hard)
    Handle complex Trie applications.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Word Search II | 2D grid + Trie | Find all words in board using Trie | [LeetCode 212](https://leetcode.com/problems/word-search-ii/) |
    | Palindrome Pairs | Reverse word trie | Find word pairs forming palindrome | [LeetCode 336](https://leetcode.com/problems/palindrome-pairs/) |
    | Word Squares | Backtracking + Trie | Build word squares with prefix constraints | [LeetCode 425](https://leetcode.com/problems/word-squares/) |
    | Concatenated Words | DP + Trie | Find words made of other words | [LeetCode 472](https://leetcode.com/problems/concatenated-words/) |

    **Goal:** Combine Trie with other algorithms (backtracking, DP, DFS).

    ---

    ## 🎯 Practice Strategy

    1. **Master Basic Template:** Implement TrieNode and Trie from memory
    2. **Understand End Markers:** Know difference between prefix and complete word
    3. **Practice DFS Collection:** Collect all words with prefix
    4. **Handle Wildcards:** Implement search with '.' wildcard
    5. **Visualize Structure:** Draw Trie for small examples
    6. **Optimize Space:** Consider when Trie overhead is worth it

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | Forgetting end marker | Treat prefix as complete word | Always check `is_end_of_word` in search |
    | Not initializing children dict | Assume children exist | Initialize `self.children = {}` in `__init__` |
    | Confusing search vs startsWith | Don't distinguish exact vs prefix | search: path + end marker; startsWith: path only |
    | Wildcard infinite recursion | Wrong base case in recursive search | Check index bounds first |
    | Memory waste | Create Trie for few words | Use Trie only when many words with shared prefixes |

---

### Bit Manipulation

=== "Understanding the Pattern"

    ## 📖 What is Bit Manipulation?

    Imagine you have a magical light switch panel with 32 switches, each either ON (1) or OFF (0). Every integer is represented by a specific pattern of these switches. Now, instead of using slow arithmetic operations, you can manipulate numbers at the hardware level—flipping switches, checking patterns, combining switch states—all in a single CPU instruction!

    This is **Bit Manipulation**: working directly with the binary representation (individual bits) of numbers. It's like speaking the computer's native language instead of using a translator. Operations like AND, OR, XOR, and bit shifts execute in O(1) time and unlock elegant solutions to problems that would otherwise require complex data structures.

    **Real-World Analogy:**
    - Permissions in Unix: read(4), write(2), execute(1) → combine with OR, check with AND
    - Network subnet masks: identify network vs host portions using AND
    - Flags in system programming: pack multiple boolean states into a single integer

    The magic: mathematical properties of binary operations (like XOR's self-canceling: `a ^ a = 0`) enable tricks that seem like wizardry but are pure logic!

    ---

    ## 🔧 How It Works

    Bit manipulation uses bitwise operators to work with individual bits or bit patterns.

    **Core Operations:**
    ```
    AND (&):  Both bits must be 1
              1 & 1 = 1,  1 & 0 = 0,  0 & 0 = 0
              Use: Check if bit is set, create masks

    OR (|):   At least one bit is 1
              1 | 1 = 1,  1 | 0 = 1,  0 | 0 = 0
              Use: Set specific bits

    XOR (^):  Bits must be different
              1 ^ 1 = 0,  1 ^ 0 = 1,  0 ^ 0 = 0
              Use: Toggle bits, find unique elements

    NOT (~):  Flip all bits
              ~1 = 0,  ~0 = 1
              Use: Invert bits

    Left Shift (<<):  Move bits left, fill with 0
                      5 << 1 = 10  (binary: 101 → 1010)
                      Use: Multiply by 2^k

    Right Shift (>>): Move bits right, discard
                      5 >> 1 = 2   (binary: 101 → 10)
                      Use: Divide by 2^k
    ```

    **Visual Examples:**
    ```
    Example 1: Check if number is even
    ───────────────────────────────────
    Even numbers have last bit = 0
    Odd numbers have last bit = 1

    6 & 1:  0110 & 0001 = 0000 (0) → Even
    7 & 1:  0111 & 0001 = 0001 (1) → Odd

    Example 2: Power of Two Check
    ──────────────────────────────
    Powers of 2 have exactly one bit set:
    1 = 0001, 2 = 0010, 4 = 0100, 8 = 1000

    Trick: n & (n-1) clears rightmost set bit
    8:     1000
    8-1=7: 0111
    8&7:   0000 → Zero! Must be power of 2

    6:     0110
    6-1=5: 0101
    6&5:   0100 → Not zero, not power of 2

    Example 3: XOR Magic (Find Single Number)
    ──────────────────────────────────────────
    Array: [4, 1, 2, 1, 2]  (every element appears twice except 4)

    XOR Properties:
    - a ^ a = 0 (self-cancels)
    - a ^ 0 = a (identity)
    - XOR is commutative

    4 ^ 1 ^ 2 ^ 1 ^ 2
    = 4 ^ (1 ^ 1) ^ (2 ^ 2)
    = 4 ^ 0 ^ 0
    = 4 ✓

    All pairs cancel out, unique element remains!
    ```

    ---

    ## 💡 Key Intuition

    **The Aha Moment:** Bits follow mathematical properties that enable O(1) solutions to problems that seem to require O(n) space or complex logic!

    Think of XOR as a "toggle switch":
    - Press button A twice → back to original state (a ^ a = 0)
    - Press no button → no change (a ^ 0 = a)
    - Order doesn't matter (commutative)

    **Why This Pattern is Powerful:**

    1. **Space Efficiency:** Pack multiple boolean flags into one integer (32 flags in a 32-bit int)
    2. **Speed:** Hardware-level operations, no function calls or loops
    3. **Mathematical Properties:** XOR, AND, OR have algebraic identities that simplify problems
    4. **Constant Time:** Most bit operations are O(1)

    **Common Bit Tricks:**
    ```
    Check if bit k is set:        n & (1 << k) != 0
    Set bit k:                    n | (1 << k)
    Clear bit k:                  n & ~(1 << k)
    Toggle bit k:                 n ^ (1 << k)
    Get rightmost set bit:        n & -n
    Clear rightmost set bit:      n & (n - 1)
    Count set bits (naive):       Loop and check each bit
    ```

    ---

    ## 📊 Performance

    | Metric | Complexity | Explanation |
    |--------|------------|-------------|
    | **Time** | O(1) per operation | Single CPU instruction for basic ops (AND, OR, XOR, shift) |
    | **Space** | O(1) | No extra data structures, work in-place |
    | **Improvement** | From O(n) space | E.g., finding unique element without hash set |

    **Note:** Bit operations are O(1), but if you loop through all bits in an integer, it's O(log n) where n is the number's value (since integers have fixed bits, it's effectively O(32) = O(1) for 32-bit integers).

=== "When to Use This Pattern"

    ## ✅ Perfect For

    | Scenario | Why Bit Manipulation Works | Example |
    |----------|---------------------------|---------|
    | **Find unique element** | XOR cancels duplicates | Single Number (LC 136) |
    | **Count set bits** | Loop through bits checking each | Number of 1 Bits (LC 191) |
    | **Power of 2 check** | Power of 2 has exactly one bit set | Power of Two (LC 231) |
    | **Subsets generation** | Use bits as inclusion flags | Subsets (LC 78) |
    | **Bit masking/flags** | Pack multiple booleans into one int | State compression in DP |
    | **Fast arithmetic** | Left shift = multiply by 2, right shift = divide by 2 | Optimization tricks |

    **Red Flags That Suggest Bit Manipulation:**
    - "Every element appears twice except one"
    - "Find missing/duplicate in constant space"
    - "Generate all subsets"
    - "Check if number is power of 2"
    - "Bit representation" or "binary" in problem
    - Need O(1) space for set operations

    ---

    ## ❌ When NOT to Use

    | Situation | Why It Fails | Better Alternative |
    |-----------|-------------|-------------------|
    | **Complex logic** | Bit tricks are hard to read/maintain | Use clear conditional logic |
    | **Large number operations** | Bits limited by integer size (32 or 64) | Use big integer libraries |
    | **Non-binary problems** | Forcing bit manipulation makes code obscure | Use appropriate data structures |
    | **Floating point** | Bitwise ops are for integers only | Use float arithmetic |
    | **When clarity matters more** | Bit tricks sacrifice readability | Comment well or use simpler approach |

    ---

    ## 🎯 Decision Flowchart

    ```
    Does problem mention "binary" or "bits"?
    ├─ Yes → USE BIT MANIPULATION ✓
    └─ No → Is it about finding unique/duplicate with O(1) space?
              ├─ Yes → Consider XOR trick
              └─ No → Is it generating all subsets?
                      ├─ Yes → Use bit masking for subset generation
                      └─ No → Probably not a bit manipulation problem
    ```

=== "Implementation Templates"

    === "Template 1: XOR for Finding Unique"

        **Use Case:** Find unique element when all others appear in pairs

        **Pattern:** XOR all elements—duplicates cancel, unique remains

        ```python
        def single_number(nums):
            """
            Find the element that appears once (all others appear twice).

            Perfect for: Single Number problem (LC 136)

            Concept:
            - XOR properties: a ^ a = 0, a ^ 0 = a
            - XOR is commutative: order doesn't matter
            - All pairs cancel out, leaving the unique element

            Time: O(n) - Single pass through array
            Space: O(1) - No extra storage

            Example: [4,1,2,1,2] → 4
            """
            result = 0

            # XOR all elements together
            for num in nums:
                result ^= num

            # Pairs cancel: 1^1 = 0, 2^2 = 0
            # Only unique element remains
            return result


        # Example Usage:
        nums = [4, 1, 2, 1, 2]
        result = single_number(nums)
        print(result)  # Output: 4

        # Trace:
        # 0 ^ 4 = 4
        # 4 ^ 1 = 5
        # 5 ^ 2 = 7
        # 7 ^ 1 = 6  (1 cancels previous 1)
        # 6 ^ 2 = 4  (2 cancels previous 2)
        ```

        **Key Points:**
        - Initialize result to 0 (identity for XOR)
        - Order of XOR doesn't matter (commutative)
        - Extension: for "appears 3 times except one", use bit counting

    === "Template 2: Bit Checking and Setting"

        **Use Case:** Check if bit is set, set/clear/toggle specific bits

        **Pattern:** Use bit masks with AND, OR, XOR

        ```python
        def bit_operations_demo(n):
            """
            Common bit manipulation operations.

            Perfect for: Understanding bit masks and operations

            Concept:
            - Create mask with 1 << k (only kth bit set)
            - AND to check, OR to set, XOR to toggle

            Time: O(1) per operation
            Space: O(1)

            Example: Operate on bits of integer
            """
            k = 3  # Bit position (0-indexed from right)

            # Check if kth bit is set
            def is_bit_set(n, k):
                mask = 1 << k  # Create mask: 00001000
                return (n & mask) != 0

            # Set kth bit to 1
            def set_bit(n, k):
                mask = 1 << k
                return n | mask  # OR sets bit

            # Clear kth bit to 0
            def clear_bit(n, k):
                mask = ~(1 << k)  # NOT inverts: 11110111
                return n & mask

            # Toggle kth bit (flip it)
            def toggle_bit(n, k):
                mask = 1 << k
                return n ^ mask  # XOR flips bit

            # Get rightmost set bit
            def get_rightmost_bit(n):
                return n & -n  # Clever trick: -n is two's complement

            # Clear rightmost set bit
            def clear_rightmost_bit(n):
                return n & (n - 1)

            return {
                'is_set': is_bit_set(n, k),
                'set': set_bit(n, k),
                'clear': clear_bit(n, k),
                'toggle': toggle_bit(n, k),
                'rightmost': get_rightmost_bit(n),
                'clear_rightmost': clear_rightmost_bit(n)
            }


        # Example Usage:
        n = 10  # Binary: 1010
        ops = bit_operations_demo(n)
        print(f"Number: {n} (binary: {bin(n)})")
        print(f"Bit 3 set? {ops['is_set']}")
        print(f"After setting bit 3: {ops['set']} (binary: {bin(ops['set'])})")
        ```

        **Key Points:**
        - `1 << k` creates mask with only kth bit set
        - `n & (1 << k)` checks if bit is set
        - `n | (1 << k)` sets bit to 1
        - `n & ~(1 << k)` clears bit to 0
        - `n ^ (1 << k)` toggles bit

    === "Template 3: Power of Two Check"

        **Use Case:** Check if number is a power of 2

        **Pattern:** Power of 2 has exactly one bit set

        ```python
        def is_power_of_two(n):
            """
            Check if n is a power of 2.

            Perfect for: Power of Two (LC 231)

            Concept:
            - Power of 2: exactly one bit set (1, 2, 4, 8, 16...)
            - Binary: 1=0001, 2=0010, 4=0100, 8=1000
            - Trick: n & (n-1) clears rightmost bit
            - If only one bit was set, result is 0

            Time: O(1)
            Space: O(1)

            Example: 8 (1000) & 7 (0111) = 0 → Power of 2 ✓
            """
            # Must be positive and have exactly one bit set
            return n > 0 and (n & (n - 1)) == 0


        # Example Usage:
        print(is_power_of_two(8))   # True  (1000)
        print(is_power_of_two(6))   # False (0110)
        print(is_power_of_two(16))  # True  (10000)

        # Why n & (n-1) works:
        # 8:     1000
        # 8-1=7: 0111
        # 8&7:   0000 → Zero means power of 2!

        # 6:     0110
        # 6-1=5: 0101
        # 6&5:   0100 → Not zero, not power of 2
        ```

        **Key Points:**
        - `n & (n-1)` clears the rightmost set bit
        - If n is power of 2, only one bit is set, so result is 0
        - Must check `n > 0` (negative numbers and 0 are not powers of 2)

    === "Template 4: Count Set Bits"

        **Use Case:** Count number of 1s in binary representation

        **Pattern:** Loop clearing rightmost bit until zero

        ```python
        def count_bits(n):
            """
            Count number of 1 bits in integer.

            Perfect for: Number of 1 Bits (LC 191)

            Concept:
            - Loop: n & (n-1) clears rightmost 1 bit
            - Count how many times until n becomes 0

            Time: O(k) where k = number of set bits
            Space: O(1)

            Example: 13 = 1101 → 3 set bits
            """
            count = 0

            while n:
                n &= (n - 1)  # Clear rightmost set bit
                count += 1

            return count


        # Alternative: Check each bit position
        def count_bits_alternate(n):
            """Count by checking each bit position."""
            count = 0

            while n:
                count += n & 1  # Check if last bit is 1
                n >>= 1         # Shift right by 1

            return count


        # Example Usage:
        n = 13  # Binary: 1101
        print(count_bits(n))  # Output: 3

        # Trace of first method:
        # 13 = 1101, clear rightmost 1: 1101 & 1100 = 1100 (count=1)
        # 12 = 1100, clear rightmost 1: 1100 & 1011 = 1000 (count=2)
        # 8  = 1000, clear rightmost 1: 1000 & 0111 = 0000 (count=3)
        ```

        **Key Points:**
        - `n & (n-1)` clears rightmost set bit (Brian Kernighan's algorithm)
        - Loop runs once per set bit (not per total bit)
        - Alternative: check each bit with `n & 1`, shift right

    === "Visual Walkthrough"

        **Problem:** Find single number in [4, 1, 2, 1, 2] using XOR

        ```
        Array: [4, 1, 2, 1, 2]

        Convert to binary:
        4 = 0100
        1 = 0001
        2 = 0010
        1 = 0001
        2 = 0010

        XOR all together (bit by bit):
        ────────────────────────────────

        Step 1: 0 ^ 4
                0000 ^ 0100 = 0100

        Step 2: 0100 ^ 1
                0100 ^ 0001 = 0101

        Step 3: 0101 ^ 2
                0101 ^ 0010 = 0111

        Step 4: 0111 ^ 1 (second 1 cancels first)
                0111 ^ 0001 = 0110

        Step 5: 0110 ^ 2 (second 2 cancels first)
                0110 ^ 0010 = 0100 = 4 ✓

        Result: 4 (the unique element)

        Why It Works:
        ─────────────
        XOR is commutative, so we can rearrange:
        4 ^ 1 ^ 2 ^ 1 ^ 2
        = 4 ^ (1 ^ 1) ^ (2 ^ 2)
        = 4 ^ 0 ^ 0
        = 4

        Each pair cancels to 0!
        ```

        **Visualization: Power of Two Check**
        ```
        Check if 8 is power of 2:

        8 in binary:     1000
        8-1 = 7:         0111

        8 & 7:           1000
                       & 0111
                       ──────
                         0000  → Zero! Power of 2 ✓

        Check if 6 is power of 2:

        6 in binary:     0110
        6-1 = 5:         0101

        6 & 5:           0110
                       & 0101
                       ──────
                         0100  → Not zero, not power of 2 ✗

        The trick: Subtracting 1 flips all bits after rightmost 1
        If only one bit was set, AND gives zero
        ```

=== "Practice Problems"

    ## 📝 Learning Path

    ### Phase 1: Foundation (Easy)
    Master basic bit operations and common patterns.

    | Problem | Variant | Key Learning | Link |
    |---------|---------|--------------|------|
    | Single Number | XOR cancellation | XOR to find unique element | [LeetCode 136](https://leetcode.com/problems/single-number/) |
    | Number of 1 Bits | Bit counting | Count set bits using n & (n-1) | [LeetCode 191](https://leetcode.com/problems/number-of-1-bits/) |
    | Power of Two | Single bit check | Detect power of 2 with n & (n-1) | [LeetCode 231](https://leetcode.com/problems/power-of-two/) |
    | Reverse Bits | Bit manipulation | Reverse binary representation | [LeetCode 190](https://leetcode.com/problems/reverse-bits/) |
    | Missing Number | XOR or sum | Find missing using XOR trick | [LeetCode 268](https://leetcode.com/problems/missing-number/) |

    **Goal:** Solve all 5 problems. Master XOR properties. Understand n & (n-1) trick. Practice converting numbers to binary mentally.

    ---

    ### Phase 2: Application (Medium)
    Apply bit manipulation to more complex scenarios.

    | Problem | Variant | Challenge | Link |
    |---------|---------|-----------|------|
    | Single Number II | Appears 3 times | Count bits modulo 3 | [LeetCode 137](https://leetcode.com/problems/single-number-ii/) |
    | Counting Bits | DP + bits | Count bits for 0 to n | [LeetCode 338](https://leetcode.com/problems/counting-bits/) |
    | Bitwise AND of Range | Bit prefix | Find common prefix of range | [LeetCode 201](https://leetcode.com/problems/bitwise-and-of-numbers-range/) |
    | Sum of Two Integers | No +/- operators | Implement addition with bits | [LeetCode 371](https://leetcode.com/problems/sum-of-two-integers/) |
    | Maximum XOR of Two Numbers | Trie + bits | Build trie of binary representations | [LeetCode 421](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) |

    **Goal:** Handle variations where elements appear k times. Combine bit manipulation with DP. Implement arithmetic using only bitwise ops.

    ---

    ### Phase 3: Mastery (Hard)
    Solve advanced problems requiring bit manipulation insights.

    | Problem | Variant | Advanced Concept | Link |
    |---------|---------|------------------|------|
    | Maximum Product of Word Lengths | Bit masking | Use bits to represent character sets | [LeetCode 318](https://leetcode.com/problems/maximum-product-of-word-lengths/) |
    | Subsets | Bit generation | Use bits to generate all subsets | [LeetCode 78](https://leetcode.com/problems/subsets/) |
    | Gray Code | Bit patterns | Generate Gray code sequence | [LeetCode 89](https://leetcode.com/problems/gray-code/) |
    | Minimum XOR Sum | DP + bitmask | State compression in DP | [LeetCode 1879](https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/) |

    **Goal:** Use bits for state compression in DP. Master bit masking for sets. Recognize when bit patterns reveal solutions.

    ---

    ## 🎯 Practice Strategy

    1. **Learn the Tricks First:** Memorize common patterns (XOR cancellation, n & (n-1), power of 2 check). Practice on paper.
    2. **Binary Conversion:** For first 5 problems, manually convert numbers to binary. Trace bit operations step-by-step.
    3. **Understand XOR Magic:** XOR is the most important. Practice: `a ^ a = 0`, `a ^ 0 = a`, commutative property.
    4. **Bit Mask Patterns:** Learn to create masks: `1 << k` (only kth bit), `~(1 << k)` (all except kth).
    5. **Time Yourself:** After solving once, re-solve from scratch in under 8 minutes.
    6. **Review After 24 Hours:** Re-code solutions next day. Focus on the "aha" moment for each trick.

    ---

    ## 💡 Common Mistakes to Avoid

    | Mistake | Why It Happens | How to Fix |
    |---------|---------------|------------|
    | **Confusing & with &&** | & is bitwise AND, && is logical AND | Use & for bits, && for booleans |
    | **Not handling negative numbers** | Bit ops on negatives can be tricky | Clarify if input can be negative; handle two's complement |
    | **Forgetting operator precedence** | & has lower precedence than == | Always parenthesize: `(n & mask) != 0`, not `n & mask != 0` |
    | **Wrong shift direction** | << is left (multiply), >> is right (divide) | Visualize: 0010 << 1 = 0100 (doubled) |
    | **Integer overflow** | Shifting can overflow | For 32-bit: max shift is 31, be careful with large shifts |

---

## 📊 Pattern Complexity Comparison

| Pattern | Time | Space | Difficulty |
|---------|------|-------|------------|
| Two Pointers | O(n) | O(1) | ⭐⭐ |
| Sliding Window | O(n) | O(k) | ⭐⭐⭐ |
| Binary Search | O(log n) | O(1) | ⭐⭐ |
| Cyclic Sort | O(n) | O(1) | ⭐⭐ |
| Prefix Sum | O(1) query | O(n) | ⭐⭐ |
| Monotonic Stack | O(n) | O(n) | ⭐⭐⭐ |
| Merge Intervals | O(n log n) | O(n) | ⭐⭐⭐ |
| Fast & Slow | O(n) | O(1) | ⭐⭐ |
| BFS | O(V+E) | O(V) | ⭐⭐⭐ |
| DFS | O(V+E) | O(H) | ⭐⭐⭐ |
| Hash Map | O(n) | O(n) | ⭐⭐ |
| Dynamic Programming | O(n²) | O(n) | ⭐⭐⭐⭐⭐ |
| Greedy | O(n log n) | O(1) | ⭐⭐⭐⭐ |
| Backtracking | O(2ⁿ) | O(n) | ⭐⭐⭐⭐ |
| Union Find | O(α(n)) | O(n) | ⭐⭐⭐ |
| Topological Sort | O(V+E) | O(V) | ⭐⭐⭐ |
| Heap (Top K) | O(n log k) | O(k) | ⭐⭐⭐ |
| K-way Merge | O(n log k) | O(k) | ⭐⭐⭐⭐ |
| Trie | O(m) | O(m×n) | ⭐⭐⭐ |
| Bit Manipulation | O(1) | O(1) | ⭐⭐⭐⭐ |

---

## 💡 Final Tips

**Master These First:**
1. Two Pointers + Sliding Window
2. Hash Map + Arrays
3. BFS + DFS
4. Binary Search
5. Dynamic Programming basics

**Practice Strategy:**
- Start easy, build confidence
- Understand pattern, not memorize solutions
- Do 5-10 problems per pattern
- Review after 1 day, 1 week, 1 month

---

**Related Pages:**
- [Time & Space Complexity](time-complexity.md)
- [Learning Paths](learning-paths.md)
- [Interview Strategy](interview-strategy.md)
