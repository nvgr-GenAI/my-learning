# String Problems - Hard

This section contains challenging string manipulation problems that require advanced algorithmic techniques.

## Problems

=== "1. Edit Distance"
    **Difficulty:** Hard  
    **Topics:** Dynamic Programming, String

    ### Problem
    Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`.

    You have the following three operations permitted on a word:
    - Insert a character
    - Delete a character  
    - Replace a character

    ### Example
    ```
    Input: word1 = "horse", word2 = "ros"
    Output: 3
    Explanation: 
    horse -> rorse (replace 'h' with 'r')
    rorse -> rose (remove 'r')
    rose -> ros (remove 'e')
    ```

    ### Solution
    ```python
    def minDistance(word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    ```

=== "2. Regular Expression Matching"
    **Difficulty:** Hard  
    **Topics:** Dynamic Programming, Recursion

    ### Problem
    Given an input string `s` and a pattern `p`, implement regular expression matching with support for `.` and `*` where:
    - `.` Matches any single character
    - `*` Matches zero or more of the preceding element

    ### Example
    ```
    Input: s = "aa", p = "a*"
    Output: true
    Explanation: '*' means zero or more of the preceding element, 'a'.
    ```

    ### Solution
    ```python
    def isMatch(s, p):
        dp = {}
        
        def dfs(i, j):
            if (i, j) in dp:
                return dp[(i, j)]
            
            if j == len(p):
                return i == len(s)
            
            match = i < len(s) and (s[i] == p[j] or p[j] == '.')
            
            if j + 1 < len(p) and p[j + 1] == '*':
                dp[(i, j)] = dfs(i, j + 2) or (match and dfs(i + 1, j))
            else:
                dp[(i, j)] = match and dfs(i + 1, j + 1)
            
            return dp[(i, j)]
        
        return dfs(0, 0)
    ```

=== "3. Minimum Window Substring"
    **Difficulty:** Hard  
    **Topics:** Hash Table, String, Sliding Window

    ### Problem
    Given two strings `s` and `t`, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window.

    ### Example
    ```
    Input: s = "ADOBECODEBANC", t = "ABC"
    Output: "BANC"
    Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
    ```

    ### Solution
    ```python
    def minWindow(s, t):
        if not s or not t:
            return ""
        
        need = {}
        for c in t:
            need[c] = need.get(c, 0) + 1
        
        left = right = 0
        required = len(need)
        formed = 0
        window_counts = {}
        ans = float("inf"), None, None
        
        while right < len(s):
            character = s[right]
            window_counts[character] = window_counts.get(character, 0) + 1
            
            if character in need and window_counts[character] == need[character]:
                formed += 1
            
            while left <= right and formed == required:
                character = s[left]
                
                if right - left + 1 < ans[0]:
                    ans = (right - left + 1, left, right)
                
                window_counts[character] -= 1
                if character in need and window_counts[character] < need[character]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]
    ```

=== "4. Wildcard Matching"
    **Difficulty:** Hard  
    **Topics:** Dynamic Programming, Greedy, Recursion

    ### Problem
    Given an input string `s` and a pattern `p`, implement wildcard pattern matching with support for `?` and `*` where:
    - `?` Matches any single character
    - `*` Matches any sequence of characters (including the empty sequence)

    ### Example
    ```
    Input: s = "adceb", p = "*a*b*"
    Output: true
    Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".
    ```

    ### Solution
    ```python
    def isMatch(s, p):
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-1]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif p[j-1] == '?' or s[i-1] == p[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        
        return dp[m][n]
    ```

=== "5. Palindrome Partitioning II"
    **Difficulty:** Hard  
    **Topics:** Dynamic Programming, String

    ### Problem
    Given a string `s`, partition `s` such that every substring of the partition is a palindrome. Return the minimum cuts needed for a palindrome partitioning of `s`.

    ### Example
    ```
    Input: s = "aab"
    Output: 1
    Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.
    ```

    ### Solution
    ```python
    def minCut(s):
        n = len(s)
        if n <= 1:
            return 0
        
        # Precompute palindrome check
        is_palindrome = [[False] * n for _ in range(n)]
        for i in range(n):
            is_palindrome[i][i] = True
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    if length == 2 or is_palindrome[i+1][j-1]:
                        is_palindrome[i][j] = True
        
        # DP for minimum cuts
        dp = [0] * n
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                dp[i] = i
                for j in range(i):
                    if is_palindrome[j+1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n-1]
    ```

=== "6. Text Justification"
    **Difficulty:** Hard  
    **Topics:** Array, String, Simulation

    ### Problem
    Given an array of strings `words` and a width `maxWidth`, format the text such that each line has exactly `maxWidth` characters and is fully justified.

    ### Example
    ```
    Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
    Output:
    [
       "This    is    an",
       "example  of text",
       "justification.  "
    ]
    ```

    ### Solution
    ```python
    def fullJustify(words, maxWidth):
        result, current_line, num_letters = [], [], 0
        
        for word in words:
            if num_letters + len(word) + len(current_line) > maxWidth:
                # Justify current line
                for i in range(maxWidth - num_letters):
                    current_line[i % (len(current_line) - 1 or 1)] += ' '
                result.append(''.join(current_line))
                current_line, num_letters = [], 0
            
            current_line.append(word)
            num_letters += len(word)
        
        # Last line (left justified)
        result.append(' '.join(current_line).ljust(maxWidth))
        return result
    ```

=== "7. Shortest Palindrome"
    **Difficulty:** Hard  
    **Topics:** String, Rolling Hash, KMP

    ### Problem
    You are given a string `s`. You can convert `s` to a palindrome by adding characters in front of it. Return the shortest palindrome you can find by performing this transformation.

    ### Example
    ```
    Input: s = "aacecaaa"
    Output: "aaacecaaa"
    ```

    ### Solution
    ```python
    def shortestPalindrome(s):
        def kmp(pattern):
            n = len(pattern)
            lps = [0] * n
            length = 0
            i = 1
            
            while i < n:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        rev_s = s[::-1]
        combined = s + "#" + rev_s
        lps = kmp(combined)
        
        return rev_s[:len(s) - lps[-1]] + s
    ```

=== "8. Word Ladder II"
    **Difficulty:** Hard  
    **Topics:** Hash Table, String, Backtracking, BFS

    ### Problem
    Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return all the shortest transformation sequences from `beginWord` to `endWord`.

    ### Example
    ```
    Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
    Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
    ```

    ### Solution
    ```python
    def findLadders(beginWord, endWord, wordList):
        if endWord not in wordList:
            return []
        
        wordList = set(wordList)
        queue = [[beginWord]]
        found = False
        used = set([beginWord])
        
        while queue and not found:
            local_used = set()
            for _ in range(len(queue)):
                path = queue.pop(0)
                word = path[-1]
                
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        new_word = word[:i] + c + word[i+1:]
                        
                        if new_word == endWord:
                            found = True
                            queue.append(path + [new_word])
                        elif new_word in wordList and new_word not in used:
                            local_used.add(new_word)
                            queue.append(path + [new_word])
            
            used.update(local_used)
        
        return [path for path in queue if path[-1] == endWord]
    ```

=== "9. Interleaving String"
    **Difficulty:** Hard  
    **Topics:** Dynamic Programming, String

    ### Problem
    Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an interleaving of `s1` and `s2`.

    ### Example
    ```
    Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
    Output: true
    Explanation: One way to obtain s3 is:
    Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
    Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
    ```

    ### Solution
    ```python
    def isInterleave(s1, s2, s3):
        if len(s1) + len(s2) != len(s3):
            return False
        
        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        dp[0][0] = True
        
        for i in range(1, len(s1) + 1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        
        for j in range(1, len(s2) + 1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                          (dp[i][j-1] and s2[j-1] == s3[i+j-1])
        
        return dp[len(s1)][len(s2)]
    ```

=== "10. Distinct Subsequences"
    **Difficulty:** Hard  
    **Topics:** Dynamic Programming, String

    ### Problem
    Given two strings `s` and `t`, return the number of distinct subsequences of `s` which equals `t`.

    ### Example
    ```
    Input: s = "rabbbit", t = "rabbit"
    Output: 3
    Explanation: There are 3 ways you can generate "rabbit" from s.
    ```

    ### Solution
    ```python
    def numDistinct(s, t):
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = 1
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j]
                if s[i-1] == t[j-1]:
                    dp[i][j] += dp[i-1][j-1]
        
        return dp[m][n]
    ```

=== "11. Word Break II"
    **Difficulty:** Hard  
    **Topics:** Hash Table, String, Dynamic Programming, Backtracking

    ### Problem
    Given a string `s` and a dictionary of strings `wordDict`, add spaces in `s` to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

    ### Example
    ```
    Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
    Output: ["cats and dog","cat sand dog"]
    ```

    ### Solution
    ```python
    def wordBreak(s, wordDict):
        wordSet = set(wordDict)
        memo = {}
        
        def dfs(start):
            if start in memo:
                return memo[start]
            
            if start == len(s):
                return [[]]
            
            result = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in wordSet:
                    rest = dfs(end)
                    for r in rest:
                        result.append([word] + r)
            
            memo[start] = result
            return result
        
        return [' '.join(words) for words in dfs(0)]
    ```

=== "12. Scramble String"
    **Difficulty:** Hard  
    **Topics:** String, Dynamic Programming

    ### Problem
    We can scramble a string s to get a string t using the following algorithm. Given two strings `s1` and `s2` of the same length, return `true` if `s2` is a scrambled string of `s1`, otherwise, return `false`.

    ### Example
    ```
    Input: s1 = "great", s2 = "rgeat"
    Output: true
    Explanation: One possible scenario applied on s1 is:
    "great" --> "gr/eat" // divide at index 2
    "gr/eat" --> "gr/eat" // no swapping
    "gr/eat" --> "g/r / e/at" // divide at index 1 and 3
    "g/r / e/at" --> "r/g / e/at" // swap
    "r/g / e/at" --> "r/g / e/ a/t" // divide at index 4
    "r/g / e/ a/t" --> "r/g / e/ a/t" // no swapping
    --> "rgeat"
    ```

    ### Solution
    ```python
    def isScramble(s1, s2):
        if len(s1) != len(s2):
            return False
        
        if s1 == s2:
            return True
        
        if sorted(s1) != sorted(s2):
            return False
        
        n = len(s1)
        for i in range(1, n):
            if (self.isScramble(s1[:i], s2[:i]) and 
                self.isScramble(s1[i:], s2[i:])) or \
               (self.isScramble(s1[:i], s2[n-i:]) and 
                self.isScramble(s1[i:], s2[:n-i])):
                return True
        
        return False
    ```

=== "13. Valid Number"
    **Difficulty:** Hard  
    **Topics:** String

    ### Problem
    A valid number can be split up into these components (in order):
    1. A decimal number or an integer
    2. (Optional) An 'e' or 'E', followed by an integer

    ### Example
    ```
    Input: s = "0"
    Output: true
    
    Input: s = "e"
    Output: false
    ```

    ### Solution
    ```python
    def isNumber(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        
        # Alternative finite state machine approach
        state = 0
        states = {
            0: {'digit': 1, 'sign': 2, 'dot': 3},
            1: {'digit': 1, 'dot': 4, 'exponent': 5},
            2: {'digit': 1, 'dot': 3},
            3: {'digit': 4},
            4: {'digit': 4, 'exponent': 5},
            5: {'sign': 6, 'digit': 7},
            6: {'digit': 7},
            7: {'digit': 7}
        }
        
        for c in s.strip():
            if c.isdigit():
                char_type = 'digit'
            elif c in '+-':
                char_type = 'sign'
            elif c == '.':
                char_type = 'dot'
            elif c.lower() == 'e':
                char_type = 'exponent'
            else:
                return False
            
            if char_type not in states.get(state, {}):
                return False
            
            state = states[state][char_type]
        
        return state in [1, 4, 7]
    ```

=== "14. Substring with Concatenation of All Words"
    **Difficulty:** Hard  
    **Topics:** Hash Table, String, Sliding Window

    ### Problem
    You are given a string `s` and an array of strings `words` of the same length. Return all starting indices of substring(s) in `s` that is a concatenation of each word in `words` exactly once, in any order, and without any intervening characters.

    ### Example
    ```
    Input: s = "barfoothefoobarman", words = ["foo","bar"]
    Output: [0,9]
    Explanation: 
    Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.
    ```

    ### Solution
    ```python
    def findSubstring(s, words):
        if not s or not words:
            return []
        
        word_len = len(words[0])
        word_count = len(words)
        total_len = word_len * word_count
        word_map = {}
        
        for word in words:
            word_map[word] = word_map.get(word, 0) + 1
        
        result = []
        
        for i in range(len(s) - total_len + 1):
            seen = {}
            j = 0
            
            while j < word_count:
                word = s[i + j * word_len:i + (j + 1) * word_len]
                
                if word not in word_map:
                    break
                
                seen[word] = seen.get(word, 0) + 1
                
                if seen[word] > word_map[word]:
                    break
                
                j += 1
            
            if j == word_count:
                result.append(i)
        
        return result
    ```

=== "15. Longest Valid Parentheses"
    **Difficulty:** Hard  
    **Topics:** String, Dynamic Programming, Stack

    ### Problem
    Given a string containing just the characters `'('` and `')'`, return the length of the longest valid (well-formed) parentheses substring.

    ### Example
    ```
    Input: s = "(()"
    Output: 2
    Explanation: The longest valid parentheses substring is "()".
    ```

    ### Solution
    ```python
    def longestValidParentheses(s):
        n = len(s)
        if n <= 1:
            return 0
        
        dp = [0] * n
        max_len = 0
        
        for i in range(1, n):
            if s[i] == ')':
                if s[i-1] == '(':
                    dp[i] = (dp[i-2] if i >= 2 else 0) + 2
                elif dp[i-1] > 0:
                    match_index = i - dp[i-1] - 1
                    if match_index >= 0 and s[match_index] == '(':
                        dp[i] = dp[i-1] + 2 + (dp[match_index-1] if match_index > 0 else 0)
                
                max_len = max(max_len, dp[i])
        
        return max_len
    ```
