# Linked Lists - Hard Problems

## 🎯 Learning Objectives

Master advanced linked list techniques and complex algorithms:

- Advanced pointer manipulation and reversal techniques
- Complex cycle detection and removal
- Merging and sorting multiple lists
- Advanced memory management
- Real-world implementation challenges

=== "Problem 1: Merge k Sorted Lists"

    **LeetCode 23** | **Difficulty: Hard**

    ## Problem Statement

    You are given an array of k linked-lists, each sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

    **Example:**
    ```
    Input: lists = [[1,4,5],[1,3,4],[2,6]]
    Output: [1,1,2,3,4,4,5,6]
    ```

    ## Solution

    ```python
    def mergeKLists(lists):
        """
        Merge k sorted lists using divide and conquer.
        
        Time: O(N log k) where N is total number of nodes
        Space: O(log k) for recursion stack
        """
        if not lists:
            return None
        
        def merge_two_lists(l1, l2):
            """Helper function to merge two sorted lists."""
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            current.next = l1 or l2
            return dummy.next
        
        def merge_lists(start, end):
            """Divide and conquer approach."""
            if start == end:
                return lists[start]
            
            if start > end:
                return None
            
            mid = (start + end) // 2
            left = merge_lists(start, mid)
            right = merge_lists(mid + 1, end)
            
            return merge_two_lists(left, right)
        
        return merge_lists(0, len(lists) - 1)
    ```

    ## Alternative: Priority Queue Approach

    ```python
    import heapq

    def mergeKLists(lists):
        """
        Using priority queue for efficient merging.
        
        Time: O(N log k)
        Space: O(k)
        """
        heap = []
        
        # Add first node of each list to heap
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(heap, (head.val, i, head))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            val, i, node = heapq.heappop(heap)
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        
        return dummy.next
    ```

    ## Key Insights

    - Divide and conquer reduces time complexity from O(kN) to O(N log k)
    - Priority queue approach is more intuitive but requires additional space
    - Both approaches maintain sorted order efficiently

=== "Problem 2: Reverse Nodes in k-Group"

    **LeetCode 25** | **Difficulty: Hard**

    ## Problem Statement

    Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.

    **Example:**
    ```
    Input: head = [1,2,3,4,5], k = 2
    Output: [2,1,4,3,5]
    ```

    ## Solution

    ```python
    def reverseKGroup(head, k):
        """
        Reverse nodes in k-group.
        
        Time: O(n)
        Space: O(1)
        """
        def reverse_linked_list(start, end):
            """Reverse linked list from start to end (exclusive)."""
            prev = None
            current = start
            
            while current != end:
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            
            return prev
        
        def get_kth_node(start, k):
            """Get the kth node from start."""
            current = start
            for _ in range(k):
                if not current:
                    return None
                current = current.next
            return current
        
        dummy = ListNode(0)
        dummy.next = head
        prev_group = dummy
        
        while True:
            # Find the start and end of current group
            start = prev_group.next
            end = get_kth_node(start, k)
            
            if not end:
                break
            
            # Store the first node of next group
            next_group = end.next
            
            # Reverse current group
            reversed_head = reverse_linked_list(start, end)
            
            # Connect with previous group
            prev_group.next = reversed_head
            start.next = next_group
            
            # Move to next group
            prev_group = start
        
        return dummy.next
    ```

    ## Key Insights

    - Use helper functions to break down the complex problem
    - Careful pointer manipulation is crucial
    - Handle edge cases when remaining nodes < k

=== "Problem 3: Copy List with Random Pointer"

    **LeetCode 138** | **Difficulty: Hard**

    ## Problem Statement

    A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null. Return a deep copy of the list.

    ## Solution

    ```python
    def copyRandomList(head):
        """
        Deep copy linked list with random pointers.
        
        Time: O(n)
        Space: O(1) excluding the result space
        """
        if not head:
            return None
        
        # Step 1: Create interleaved list
        current = head
        while current:
            new_node = Node(current.val)
            new_node.next = current.next
            current.next = new_node
            current = new_node.next
        
        # Step 2: Copy random pointers
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next
        
        # Step 3: Separate the lists
        dummy = Node(0)
        current_new = dummy
        current_old = head
        
        while current_old:
            current_new.next = current_old.next
            current_old.next = current_old.next.next
            current_new = current_new.next
            current_old = current_old.next
        
        return dummy.next
    ```

    ## Alternative: HashMap Approach

    ```python
    def copyRandomList(head):
        """
        Using HashMap for node mapping.
        
        Time: O(n)
        Space: O(n)
        """
        if not head:
            return None
        
        # Create mapping of old nodes to new nodes
        old_to_new = {}
        current = head
        
        # First pass: create all nodes
        while current:
            old_to_new[current] = Node(current.val)
            current = current.next
        
        # Second pass: set next and random pointers
        current = head
        while current:
            if current.next:
                old_to_new[current].next = old_to_new[current.next]
            if current.random:
                old_to_new[current].random = old_to_new[current.random]
            current = current.next
        
        return old_to_new[head]
    ```

    ## Key Insights

    - Interleaving approach uses O(1) extra space
    - HashMap approach is more intuitive but uses O(n) space
    - Both approaches require careful pointer manipulation

=== "Problem 4: LRU Cache"

    **LeetCode 146** | **Difficulty: Hard**

    ## Problem Statement

    Design a data structure that follows the constraints of a Least Recently Used (LRU) cache with O(1) operations.

    ## Solution

    ```python
    class LRUCache:
        """
        LRU Cache implementation using doubly linked list + hashmap.
        
        Time: O(1) for all operations
        Space: O(capacity)
        """
        
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = {}
            
            # Create dummy head and tail
            self.head = Node(0, 0)
            self.tail = Node(0, 0)
            self.head.next = self.tail
            self.tail.prev = self.head
        
        def _add_node(self, node):
            """Add node right after head."""
            node.prev = self.head
            node.next = self.head.next
            
            self.head.next.prev = node
            self.head.next = node
        
        def _remove_node(self, node):
            """Remove an existing node."""
            prev_node = node.prev
            next_node = node.next
            
            prev_node.next = next_node
            next_node.prev = prev_node
        
        def _move_to_head(self, node):
            """Move node to head (mark as recently used)."""
            self._remove_node(node)
            self._add_node(node)
        
        def _pop_tail(self):
            """Remove last node."""
            last_node = self.tail.prev
            self._remove_node(last_node)
            return last_node
        
        def get(self, key):
            node = self.cache.get(key)
            
            if node:
                # Move to head (recently used)
                self._move_to_head(node)
                return node.value
            
            return -1
        
        def put(self, key, value):
            node = self.cache.get(key)
            
            if node:
                # Update existing node
                node.value = value
                self._move_to_head(node)
            else:
                # Add new node
                new_node = Node(key, value)
                
                if len(self.cache) >= self.capacity:
                    # Remove least recently used
                    tail = self._pop_tail()
                    del self.cache[tail.key]
                
                self.cache[key] = new_node
                self._add_node(new_node)
    ```

    ## Key Insights

    - Doubly linked list enables O(1) node removal
    - HashMap provides O(1) key lookup
    - Combination achieves O(1) for all operations

=== "Problem 5: Serialize and Deserialize Binary Tree"

    **LeetCode 297** | **Difficulty: Hard**

    ## Problem Statement

    Design an algorithm to serialize and deserialize a binary tree.

    ## Solution

    ```python
    def serialize(root):
        """
        Serialize binary tree to string.
        
        Time: O(n)
        Space: O(n)
        """
        def preorder(node):
            if not node:
                return 'null'
            return str(node.val) + ',' + preorder(node.left) + ',' + preorder(node.right)
        
        return preorder(root)
    
    def deserialize(data):
        """
        Deserialize string to binary tree.
        
        Time: O(n)
        Space: O(n)
        """
        def build_tree():
            nonlocal index
            if index >= len(nodes):
                return None
            
            val = nodes[index]
            index += 1
            
            if val == 'null':
                return None
            
            node = TreeNode(int(val))
            node.left = build_tree()
            node.right = build_tree()
            return node
        
        nodes = data.split(',')
        index = 0
        return build_tree()
    ```

    ## Key Insights

    - Preorder traversal maintains structure information
    - Recursive approach simplifies tree reconstruction
    - Proper handling of null nodes is crucial

=== "Problem 6: Flatten Binary Tree to Linked List"

    **LeetCode 114** | **Difficulty: Hard**

    ## Problem Statement

    Given the root of a binary tree, flatten it to a linked list in-place.

    ## Solution

    ```python
    def flatten(root):
        """
        Flatten binary tree to linked list in-place.
        
        Time: O(n)
        Space: O(h) where h is height of tree
        """
        def flatten_tree(node):
            """Returns the tail of flattened tree."""
            if not node:
                return None
            
            # Leaf node
            if not node.left and not node.right:
                return node
            
            # Flatten left and right subtrees
            left_tail = flatten_tree(node.left)
            right_tail = flatten_tree(node.right)
            
            # If left subtree exists, rewire connections
            if left_tail:
                left_tail.right = node.right
                node.right = node.left
                node.left = None
            
            # Return the tail of current subtree
            return right_tail or left_tail
        
        flatten_tree(root)
    ```

    ## Iterative Approach

    ```python
    def flatten(root):
        """
        Iterative approach using stack.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return
        
        stack = [root]
        
        while stack:
            node = stack.pop()
            
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            
            if stack:
                node.right = stack[-1]
            node.left = None
    ```

    ## Key Insights

    - Morris-like traversal can achieve O(1) space
    - Careful rewiring of pointers is essential
    - Stack-based approach is more intuitive

=== "Problem 7: Binary Tree Maximum Path Sum"

    **LeetCode 124** | **Difficulty: Hard**

    ## Problem Statement

    Given the root of a binary tree, return the maximum path sum of any non-empty path.

    ## Solution

    ```python
    def maxPathSum(root):
        """
        Find maximum path sum in binary tree.
        
        Time: O(n)
        Space: O(h) where h is height of tree
        """
        self.max_sum = float('-inf')
        
        def max_gain(node):
            """Returns max gain from current node to leaf."""
            if not node:
                return 0
            
            # Max gain from left and right subtrees
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            # Max path sum through current node
            current_max = node.val + left_gain + right_gain
            
            # Update global maximum
            self.max_sum = max(self.max_sum, current_max)
            
            # Return max gain from current node
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return self.max_sum
    ```

    ## Key Insights

    - Path can start and end at any nodes
    - Each node can contribute to path through it or extend path from parent
    - Negative gains should be ignored

=== "Problem 8: Word Ladder"

    **LeetCode 127** | **Difficulty: Hard**

    ## Problem Statement

    Given two words, beginWord and endWord, and a word list, find the length of the shortest transformation sequence from beginWord to endWord.

    ## Solution

    ```python
    from collections import deque

    def ladderLength(beginWord, endWord, wordList):
        """
        Find shortest word ladder using BFS.
        
        Time: O(M²×N) where M is word length, N is number of words
        Space: O(M²×N)
        """
        if endWord not in wordList:
            return 0
        
        wordList = set(wordList)
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, length = queue.popleft()
            
            if word == endWord:
                return length
            
            # Try all possible transformations
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        
                        if new_word in wordList and new_word not in visited:
                            visited.add(new_word)
                            queue.append((new_word, length + 1))
        
        return 0
    ```

    ## Bidirectional BFS

    ```python
    def ladderLength(beginWord, endWord, wordList):
        """
        Bidirectional BFS for optimization.
        
        Time: O(M²×N)
        Space: O(M²×N)
        """
        if endWord not in wordList:
            return 0
        
        wordList = set(wordList)
        front = {beginWord}
        back = {endWord}
        dist = 1
        
        while front and back:
            # Always expand smaller set
            if len(front) > len(back):
                front, back = back, front
            
            next_front = set()
            
            for word in front:
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c != word[i]:
                            new_word = word[:i] + c + word[i+1:]
                            
                            if new_word in back:
                                return dist + 1
                            
                            if new_word in wordList:
                                wordList.remove(new_word)
                                next_front.add(new_word)
            
            front = next_front
            dist += 1
        
        return 0
    ```

    ## Key Insights

    - BFS guarantees shortest path
    - Bidirectional search reduces search space
    - Set operations provide O(1) lookup

=== "Problem 9: Palindrome Partitioning II"

    **LeetCode 132** | **Difficulty: Hard**

    ## Problem Statement

    Given a string s, partition s such that every substring of the partition is a palindrome. Return the minimum cuts needed.

    ## Solution

    ```python
    def minCut(s):
        """
        Find minimum cuts for palindrome partitioning.
        
        Time: O(n²)
        Space: O(n²)
        """
        n = len(s)
        
        # Precompute palindrome information
        is_palindrome = [[False] * n for _ in range(n)]
        
        # Every single character is palindrome
        for i in range(n):
            is_palindrome[i][i] = True
        
        # Check for palindromes of length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # Check for palindromes of length 3 and more
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        # DP for minimum cuts
        dp = [float('inf')] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1]
    ```

    ## Optimized Approach

    ```python
    def minCut(s):
        """
        Optimized approach with center expansion.
        
        Time: O(n²)
        Space: O(n)
        """
        n = len(s)
        cuts = list(range(n))
        
        def expand_around_center(left, right):
            while left >= 0 and right < n and s[left] == s[right]:
                if left == 0:
                    cuts[right] = 0
                else:
                    cuts[right] = min(cuts[right], cuts[left - 1] + 1)
                left -= 1
                right += 1
        
        for i in range(n):
            # Odd length palindromes
            expand_around_center(i, i)
            # Even length palindromes
            expand_around_center(i, i + 1)
        
        return cuts[n - 1]
    ```

    ## Key Insights

    - Precomputing palindrome information saves time
    - Center expansion technique is elegant
    - DP builds solution incrementally

=== "Problem 10: Regular Expression Matching"

    **LeetCode 10** | **Difficulty: Hard**

    ## Problem Statement

    Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*'.

    ## Solution

    ```python
    def isMatch(s, p):
        """
        Regular expression matching using DP.
        
        Time: O(m×n)
        Space: O(m×n)
        """
        m, n = len(s), len(p)
        
        # dp[i][j] = True if s[0:i] matches p[0:j]
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Empty pattern matches empty string
        dp[0][0] = True
        
        # Handle patterns with *
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # Can ignore the character before *
                    dp[i][j] = dp[i][j - 2]
                    
                    # Or match current character
                    if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                else:
                    # Direct character match
                    if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                        dp[i][j] = dp[i - 1][j - 1]
        
        return dp[m][n]
    ```

    ## Recursive with Memoization

    ```python
    def isMatch(s, p):
        """
        Recursive approach with memoization.
        
        Time: O(m×n)
        Space: O(m×n)
        """
        memo = {}
        
        def helper(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Base case: pattern exhausted
            if j == len(p):
                return i == len(s)
            
            # Check if current characters match
            first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')
            
            # Handle * in pattern
            if j + 1 < len(p) and p[j + 1] == '*':
                # Either skip the pattern or match current character
                result = helper(i, j + 2) or (first_match and helper(i + 1, j))
            else:
                # Direct character match
                result = first_match and helper(i + 1, j + 1)
            
            memo[(i, j)] = result
            return result
        
        return helper(0, 0)
    ```

    ## Key Insights

    - DP table captures all possible matching states
    - '*' can match zero or more characters
    - Memoization prevents redundant computations

=== "Problem 11: Edit Distance"

    **LeetCode 72** | **Difficulty: Hard**

    ## Problem Statement

    Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

    ## Solution

    ```python
    def minDistance(word1, word2):
        """
        Edit distance using dynamic programming.
        
        Time: O(m×n)
        Space: O(m×n)
        """
        m, n = len(word1), len(word2)
        
        # dp[i][j] = min operations to convert word1[0:i] to word2[0:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all characters
        
        for j in range(n + 1):
            dp[0][j] = j  # Insert all characters
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Delete
                        dp[i][j - 1],      # Insert
                        dp[i - 1][j - 1]   # Replace
                    )
        
        return dp[m][n]
    ```

    ## Space-Optimized Version

    ```python
    def minDistance(word1, word2):
        """
        Space-optimized edit distance.
        
        Time: O(m×n)
        Space: O(min(m,n))
        """
        if len(word1) < len(word2):
            word1, word2 = word2, word1
        
        m, n = len(word1), len(word2)
        prev = list(range(n + 1))
        
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            
            prev = curr
        
        return prev[n]
    ```

    ## Key Insights

    - Classical DP problem with optimal substructure
    - Three operations: insert, delete, replace
    - Space can be optimized to O(min(m,n))

=== "Problem 12: Longest Valid Parentheses"

    **LeetCode 32** | **Difficulty: Hard**

    ## Problem Statement

    Given a string containing just the characters '(' and ')', find the length of the longest valid parentheses substring.

    ## Solution

    ```python
    def longestValidParentheses(s):
        """
        Find longest valid parentheses using DP.
        
        Time: O(n)
        Space: O(n)
        """
        n = len(s)
        if n <= 1:
            return 0
        
        # dp[i] = length of longest valid parentheses ending at index i
        dp = [0] * n
        max_length = 0
        
        for i in range(1, n):
            if s[i] == ')':
                if s[i - 1] == '(':
                    # Case: ...()
                    dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
                elif dp[i - 1] > 0:
                    # Case: ...))
                    match_index = i - dp[i - 1] - 1
                    if match_index >= 0 and s[match_index] == '(':
                        dp[i] = dp[i - 1] + 2 + (dp[match_index - 1] if match_index > 0 else 0)
                
                max_length = max(max_length, dp[i])
        
        return max_length
    ```

    ## Stack-Based Approach

    ```python
    def longestValidParentheses(s):
        """
        Using stack to track indices.
        
        Time: O(n)
        Space: O(n)
        """
        stack = [-1]  # Initialize with -1 as base
        max_length = 0
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:  # char == ')'
                stack.pop()
                if not stack:
                    stack.append(i)  # No matching '('
                else:
                    max_length = max(max_length, i - stack[-1])
        
        return max_length
    ```

    ## Key Insights

    - DP approach builds solution incrementally
    - Stack tracks unmatched parentheses positions
    - Both approaches achieve O(n) time complexity

=== "Problem 13: Trapping Rain Water"

    **LeetCode 42** | **Difficulty: Hard**

    ## Problem Statement

    Given n non-negative integers representing an elevation map, compute how much water can be trapped after raining.

    ## Solution

    ```python
    def trap(height):
        """
        Trap rainwater using two pointers.
        
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

    ## Stack-Based Approach

    ```python
    def trap(height):
        """
        Using stack to find water levels.
        
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
                
                width = i - stack[-1] - 1
                bounded_height = min(h, height[stack[-1]]) - height[bottom]
                water += width * bounded_height
            
            stack.append(i)
        
        return water
    ```

    ## Key Insights

    - Two-pointer approach is most efficient
    - Water level determined by shorter boundary
    - Stack approach calculates water layer by layer

=== "Problem 14: Sliding Window Maximum"

    **LeetCode 239** | **Difficulty: Hard**

    ## Problem Statement

    Given an array and a sliding window of size k, find the maximum value in each window.

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

    ## Alternative: Segment Tree

    ```python
    class SegmentTree:
        def __init__(self, nums):
            self.n = len(nums)
            self.tree = [0] * (4 * self.n)
            self.build(nums, 0, 0, self.n - 1)
        
        def build(self, nums, node, start, end):
            if start == end:
                self.tree[node] = nums[start]
            else:
                mid = (start + end) // 2
                self.build(nums, 2 * node + 1, start, mid)
                self.build(nums, 2 * node + 2, mid + 1, end)
                self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])
        
        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return float('-inf')
            if l <= start and end <= r:
                return self.tree[node]
            
            mid = (start + end) // 2
            left_max = self.query(2 * node + 1, start, mid, l, r)
            right_max = self.query(2 * node + 2, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        def range_max(self, l, r):
            return self.query(0, 0, self.n - 1, l, r)

    def maxSlidingWindow(nums, k):
        """
        Using segment tree for range maximum queries.
        
        Time: O(n log n)
        Space: O(n)
        """
        if not nums or k == 0:
            return []
        
        seg_tree = SegmentTree(nums)
        result = []
        
        for i in range(len(nums) - k + 1):
            result.append(seg_tree.range_max(i, i + k - 1))
        
        return result
    ```

    ## Key Insights

    - Deque maintains decreasing order for O(1) maximum
    - Segment tree provides flexible range queries
    - Deque approach is optimal for sliding window

=== "Problem 15: Minimum Window Substring"

    **LeetCode 76** | **Difficulty: Hard**

    ## Problem Statement

    Given strings s and t, return the minimum window substring of s such that every character in t is included in the window.

    ## Solution

    ```python
    def minWindow(s, t):
        """
        Find minimum window substring using sliding window.
        
        Time: O(|s| + |t|)
        Space: O(|s| + |t|)
        """
        if not s or not t:
            return ""
        
        # Count characters in t
        t_count = {}
        for char in t:
            t_count[char] = t_count.get(char, 0) + 1
        
        required = len(t_count)  # Number of unique characters in t
        left = right = 0
        formed = 0  # Number of unique characters in window with desired frequency
        
        window_count = {}
        
        # Result: (window length, left, right)
        result = float('inf'), None, None
        
        while right < len(s):
            # Add character to window
            char = s[right]
            window_count[char] = window_count.get(char, 0) + 1
            
            # Check if current character's frequency matches desired frequency
            if char in t_count and window_count[char] == t_count[char]:
                formed += 1
            
            # Try to shrink window from left
            while left <= right and formed == required:
                char = s[left]
                
                # Update result if current window is smaller
                if right - left + 1 < result[0]:
                    result = (right - left + 1, left, right)
                
                # Remove character from window
                window_count[char] -= 1
                if char in t_count and window_count[char] < t_count[char]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]
    ```

    ## Optimized Version

    ```python
    def minWindow(s, t):
        """
        Optimized version with filtered string.
        
        Time: O(|s| + |t|)
        Space: O(|s| + |t|)
        """
        if not s or not t:
            return ""
        
        t_count = {}
        for char in t:
            t_count[char] = t_count.get(char, 0) + 1
        
        required = len(t_count)
        
        # Filter s to only include characters in t
        filtered_s = []
        for i, char in enumerate(s):
            if char in t_count:
                filtered_s.append((i, char))
        
        left = right = 0
        formed = 0
        window_count = {}
        
        result = float('inf'), None, None
        
        while right < len(filtered_s):
            char = filtered_s[right][1]
            window_count[char] = window_count.get(char, 0) + 1
            
            if window_count[char] == t_count[char]:
                formed += 1
            
            while left <= right and formed == required:
                char = filtered_s[left][1]
                
                start = filtered_s[left][0]
                end = filtered_s[right][0]
                
                if end - start + 1 < result[0]:
                    result = (end - start + 1, start, end)
                
                window_count[char] -= 1
                if window_count[char] < t_count[char]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]
    ```

    ## Key Insights

    - Sliding window with character frequency tracking
    - Shrink window when all characters are covered
    - Filtering optimization reduces irrelevant characters
