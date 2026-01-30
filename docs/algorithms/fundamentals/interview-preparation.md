# Interview Preparation

**Complete guide to ace technical interviews** | 📋 Study Plans | 🎯 Interview Strategies | 💪 Mental Prep

## Quick Navigation

| Section | What You'll Learn | Time to Read |
|---------|------------------|--------------|
| [Interview Landscape](#interview-landscape) | Problem frequencies, formats, company types | 5 min |
| [Assess & Plan](#creating-your-plan) | Self-assessment, timeline selection | 10 min |
| [Study Strategies](#learning-strategies) | Breadth vs depth, spaced repetition, problem selection | 15 min |
| [Daily Practice](#daily-practice-routines) | Structured schedules for different time commitments | 5 min |
| [During Interview](#problem-solving-framework) | Step-by-step approach, communication techniques | 10 min |
| [Mock Interviews](#mock-interviews) | Practice techniques, platforms, improvement loop | 10 min |
| [Mental Preparation](#mental-preparation) | Anxiety management, motivation, burnout prevention | 10 min |
| [Resources](#resources) | Platforms, books, tools, communities | 5 min |

---

=== "🎯 Interview Landscape"

    ## Problem Distribution

    **Know what to expect** - understanding problem frequencies helps prioritize your study time.

    | Category | Frequency | Common Patterns | Example Problems |
    |----------|-----------|-----------------|------------------|
    | **Arrays & Strings** | 35% | Two pointers, sliding window, hash maps | Two Sum, Longest Substring, Valid Anagram |
    | **Trees & Graphs** | 25% | Tree traversal, BFS/DFS, path finding | Binary Tree Inorder, Number of Islands |
    | **Dynamic Programming** | 20% | Optimization, counting, sequences | Climbing Stairs, Coin Change, LIS |
    | **Linked Lists** | 15% | Pointer manipulation, cycle detection | Reverse List, Merge Lists, Detect Cycle |
    | **Others** | 5% | Bit manipulation, design, math | Single Number, LRU Cache |

    ---

    ## Interview Formats

    === "Phone Screen"

        **Duration:** 30-45 minutes
        **Format:** 1-2 easy to medium problems

        | Aspect | Details |
        |--------|---------|
        | **Focus** | Basic problem-solving, code fluency |
        | **Tools** | CoderPad, HackerRank, shared editor |
        | **Key Skills** | Clear thinking, verbal communication |
        | **Preparation** | Practice explaining thought process out loud |

        **Success Criteria:**
        - Working solution within time limit
        - Clean, readable code
        - Clear communication
        - Handles edge cases

    === "Technical Assessment"

        **Duration:** 60-90 minutes
        **Format:** 1-2 medium problems with deep discussion

        | Aspect | Details |
        |--------|---------|
        | **Focus** | Problem-solving approach, technical depth |
        | **Tools** | Online IDE with video/audio |
        | **Key Skills** | Optimization, tradeoff analysis |
        | **Preparation** | Practice complexity analysis, multiple solutions |

        **Success Criteria:**
        - Discusses multiple approaches
        - Analyzes time/space complexity
        - Optimizes initial solution
        - Asks clarifying questions

    === "Onsite Rounds"

        **Duration:** 4-6 hours (multiple 45-60 min sessions)
        **Format:** 4-5 interviews, 1-2 problems each

        | Round | Focus | What to Expect |
        |-------|-------|----------------|
        | **Algorithm 1-3** | Coding problems | Medium-hard difficulty, optimization focus |
        | **System Design** | Architecture | Scalability, tradeoffs (senior roles) |
        | **Behavioral** | Soft skills | Past experiences, collaboration, conflict resolution |
        | **Hiring Manager** | Fit & motivation | Career goals, team alignment, technical depth |

        **Success Criteria:**
        - Consistent performance across rounds
        - Energy maintained throughout day
        - Strong communication in all formats
        - Cultural fit demonstrated

    ---

    ## Company Types

    === "FAANG+ (Big Tech)"

        **Companies:** Google, Amazon, Apple, Microsoft, Meta, Netflix

        | Focus Area | What They Test | Preparation Strategy |
        |------------|---------------|---------------------|
        | **Algorithms** | Efficiency, optimization | LeetCode medium-hard problems |
        | **Complexity** | Big O mastery | Analyze every solution |
        | **Scale** | Large data handling | System design (senior+) |
        | **Communication** | Explaining tradeoffs | Mock interviews, peer review |

        **Problem Patterns:**
        - Google: Graph algorithms, complex DP
        - Amazon: Trees, arrays, leadership principles
        - Meta: Arrays, strings, fast implementation
        - Microsoft: Trees, linked lists, correctness

    === "Startups"

        **Companies:** Early stage, growth stage, pre-IPO

        | Focus Area | What They Test | Preparation Strategy |
        |------------|---------------|---------------------|
        | **Practical Skills** | Real-world problem solving | Build projects, system design |
        | **Adaptability** | Learning speed, flexibility | Diverse problems, new tech |
        | **Product Sense** | Business understanding | Think about user impact |
        | **Full Stack** | Breadth of knowledge | Frontend, backend, databases |

        **Interview Style:**
        - Less algorithm-heavy (still test fundamentals)
        - More emphasis on building actual features
        - Architectural decisions for ambiguous problems
        - Cultural fit and scrappy attitude

    === "Finance/Trading"

        **Companies:** Jane Street, Citadel, Two Sigma, HRT

        | Focus Area | What They Test | Preparation Strategy |
        |------------|---------------|---------------------|
        | **Optimization** | Performance-critical code | Complexity analysis, benchmarking |
        | **Math** | Probability, statistics | Brain teasers, quantitative reasoning |
        | **Concurrency** | Parallelism, race conditions | Threading, locks, atomic operations |
        | **Low-level** | Memory, cache efficiency | C/C++, systems programming |

        **Interview Style:**
        - Extremely difficult algorithms
        - On-the-spot mathematical proofs
        - Performance optimization focus
        - Competitive programming background helpful

=== "📋 Creating Your Plan"

    ## Self-Assessment

    **Know your starting point** before planning. Answer these questions:

    | Category | Questions | Your Answer |
    |----------|-----------|-------------|
    | **Knowledge** | How many data structures can you implement from scratch? | ___/10 |
    | **Experience** | How many problems have you solved? | ___ problems |
    | **Proficiency** | Can you write bug-free code in 30 minutes? | Yes / No / Sometimes |
    | **Time** | How many hours per day can you dedicate? | ___ hours |
    | **Deadline** | When do you need to be interview-ready? | ___ weeks away |

    **Interpretation:**

    | If You Have... | Choose This Plan |
    |----------------|------------------|
    | 12+ weeks, 2-3 hrs/day, beginner | [12-Week Comprehensive](#12-week-comprehensive) |
    | 4-8 weeks, 4+ hrs/day, some experience | [4-Week Intensive](#4-week-intensive) |
    | 10+ weekends, working professional | [Weekend Warrior](#weekend-warrior) |

    ---

    ## 12-Week Comprehensive

    **Best for:** Thorough preparation with steady progress
    **Time commitment:** 2-3 hours daily
    **Problems:** 200+ across all difficulties

    | Week | Topics | Theory | Practice | Mock Interview |
    |------|--------|--------|----------|----------------|
    | **1-2** | Arrays, Strings, Hash Tables | Two pointers, sliding window, hashing | 20 easy + 10 medium | ❌ |
    | **3-4** | Linked Lists, Stacks, Queues | Pointer manipulation, cycle detection | 10 easy + 15 medium | ✅ (Week 4) |
    | **5-6** | Binary Trees, BST | Traversals, construction, validation | 5 easy + 20 medium | ✅ (Week 6) |
    | **7-8** | Graphs | BFS, DFS, shortest path | 5 easy + 15 medium | ✅ (Week 8) |
    | **9-10** | Dynamic Programming | Memoization, tabulation, patterns | 10 medium + 10 hard | ✅ (Week 10) |
    | **11-12** | Review, Greedy, Advanced | Mixed topics, optimization | 15 medium + 10 hard | ✅ (2-3 per week) |

    **Weekly Structure:**
    ```
    Monday-Thursday: Learn + Practice (2-3 hrs)
    Friday: Review week's problems (2 hrs)
    Saturday: Hard problem + mock interview (3 hrs)
    Sunday: Rest or light review (1 hr)
    ```

    ---

    ## 4-Week Intensive

    **Best for:** Accelerated preparation with focused effort
    **Time commitment:** 4-5 hours daily
    **Problems:** 120+ medium/hard focus

    === "Week 1: Foundation"

        **Focus:** Arrays, Strings, Hash Tables

        | Day | Morning (2h) | Afternoon (2h) | Evening (1h) |
        |-----|--------------|----------------|--------------|
        | **Mon** | Array basics theory | 2 easy + 1 medium | Review solutions |
        | **Tue** | Two pointers pattern | 1 easy + 2 medium | Pattern recognition |
        | **Wed** | Sliding window pattern | 2 medium | Implementation practice |
        | **Thu** | Hash map techniques | 2 medium | Solution optimization |
        | **Fri** | Binary search variants | 1 medium + 1 hard | Weekly review |
        | **Sat** | Mixed practice | 3 medium | Mock interview (1h) |
        | **Sun** | Rest & reflection | Study mistakes | Plan Week 2 |

        **Target:** 20-25 problems (10 easy, 12 medium, 3 hard)

    === "Week 2: Data Structures"

        **Focus:** Linked Lists, Trees, Stacks, Queues

        | Day | Morning (2h) | Afternoon (2h) | Evening (1h) |
        |-----|--------------|----------------|--------------|
        | **Mon** | Linked list operations | 1 easy + 2 medium | Recursion practice |
        | **Tue** | Tree traversals | 3 medium | Pattern recognition |
        | **Wed** | BST operations | 2 medium | Implementation |
        | **Thu** | Stack/queue applications | 2 medium | Solution optimization |
        | **Fri** | Tree construction | 1 medium + 1 hard | Weekly review |
        | **Sat** | Mixed practice | 3 medium | Mock interview (1h) |
        | **Sun** | Rest & reflection | Study mistakes | Plan Week 3 |

        **Target:** 20-25 problems (3 easy, 18 medium, 4 hard)

    === "Week 3: Advanced Topics"

        **Focus:** Graphs, Dynamic Programming

        | Day | Morning (2h) | Afternoon (2h) | Evening (1h) |
        |-----|--------------|----------------|--------------|
        | **Mon** | Graph traversals (BFS/DFS) | 2 medium | Implementation |
        | **Tue** | Shortest path algorithms | 2 medium | Pattern recognition |
        | **Wed** | DP introduction | 1 medium + 1 hard | State definition practice |
        | **Thu** | 1D DP patterns | 2 hard | Solution optimization |
        | **Fri** | 2D DP patterns | 1 medium + 1 hard | Weekly review |
        | **Sat** | Mixed graph + DP | 2 medium + 1 hard | Mock interview (1h) |
        | **Sun** | Rest & reflection | Study mistakes | Plan Week 4 |

        **Target:** 20-25 problems (0 easy, 12 medium, 13 hard)

    === "Week 4: Integration"

        **Focus:** Mixed practice, mock interviews, weak areas

        | Day | Morning (2h) | Afternoon (2h) | Evening (1h) |
        |-----|--------------|----------------|--------------|
        | **Mon** | Mock interview | 2 medium | Review feedback |
        | **Tue** | Company-specific problems | 2 medium + 1 hard | Pattern review |
        | **Wed** | Mock interview | Weak area focus (3 problems) | Review feedback |
        | **Thu** | Mixed hard problems | 2 hard | Optimization techniques |
        | **Fri** | Final pattern review | System design basics | Cheat sheet creation |
        | **Sat** | Mock interview (full) | Rest & confidence building | - |
        | **Sun** | Rest | Mental preparation | - |

        **Target:** 15-20 problems + 3-4 mock interviews

    ---

    ## Weekend Warrior

    **Best for:** Working professionals
    **Time commitment:** 8 hours each weekend
    **Problems:** 150+ over 10 weekends

    | Weekend | Saturday (4h) | Sunday (4h) | Problems |
    |---------|---------------|-------------|----------|
    | **1** | Arrays & Two Pointers | Strings & Hash Tables | 10 (8 easy, 2 med) |
    | **2** | Linked Lists | Stacks & Queues | 10 (5 easy, 5 med) |
    | **3** | Binary Trees | BST & Heaps | 10 (3 easy, 7 med) |
    | **4** | Graph Basics & BFS | DFS & Connectivity | 10 (2 easy, 8 med) |
    | **5** | Shortest Path | Union Find & MST | 8 (all medium) |
    | **6** | DP Fundamentals | 1D DP Patterns | 8 (5 med, 3 hard) |
    | **7** | 2D DP | Greedy Algorithms | 8 (5 med, 3 hard) |
    | **8** | Binary Search | Divide & Conquer | 8 (all medium) |
    | **9** | Backtracking | Bit Manipulation | 8 (5 med, 3 hard) |
    | **10** | Mock Interviews | Review & Gaps | 6 + 2 mocks |

    **Weekday Maintenance:** 30 min daily
    - Review weekend problems
    - Watch solution videos
    - Read others' approaches

=== "📚 Learning Strategies"

    ## Breadth vs Depth

    | Approach | When to Use | Pros | Cons |
    |----------|-------------|------|------|
    | **Breadth-First** | Interviews in 2-4 weeks | Fast coverage of all topics | Shallow understanding |
    | **Depth-First** | Learning for mastery | Deep expertise in select areas | Limited topic coverage |
    | **Hybrid** ⭐ | 6+ weeks available | Balanced, retention-focused | Requires discipline |

    **Hybrid Approach (Recommended):**

    ```
    Phase 1 (30%): Learn basics of ALL topics → Build foundation
    Phase 2 (50%): Deep dive into 3-4 weak areas → Achieve mastery
    Phase 3 (20%): Mixed practice + review → Maintain & integrate
    ```

    ---

    ## Spaced Repetition

    **Science-backed technique** for long-term retention. Review problems on this schedule:

    | Review | After Initial Solve | Purpose |
    |--------|---------------------|---------|
    | **1st** | +1 day | Reinforce understanding |
    | **2nd** | +3 days | Combat forgetting curve |
    | **3rd** | +1 week | Strengthen memory |
    | **4th** | +2 weeks | Long-term retention |
    | **5th** | +1 month | Permanent mastery |

    **Implementation:**
    - Use flashcards for key patterns (Anki, physical cards)
    - Re-implement algorithms from scratch during reviews
    - Solve similar but different problems at each interval
    - Explain to someone else (or rubber duck debugging)

    ---

    ## Problem Selection

    === "By Difficulty (Beginners)"

        **Goal:** Build confidence and skills gradually

        ```
        Easy (40%) → Medium (50%) → Hard (10%)
        ```

        | Stage | Problems | Focus |
        |-------|----------|-------|
        | **Stage 1** | 15-20 easy per topic | Learn patterns, build confidence |
        | **Stage 2** | 20-30 medium per topic | Apply patterns, handle variations |
        | **Stage 3** | 5-10 hard per topic | Master edge cases, optimization |

    === "By Company (Interview Prep)"

        **Goal:** Target specific companies efficiently

        **Steps:**
        1. Filter by company on LeetCode/other platforms
        2. Sort by frequency (solve top 50 most asked)
        3. Group by pattern and solve similar problems together
        4. Practice under timed conditions

        **Company-Specific Tags:**
        - Use "Google Top 50" or "Amazon Frequently Asked"
        - Focus on recent problems (last 6 months)
        - Note: frequency data changes, stay updated

    === "By Pattern (Efficient)"

        **Goal:** Recognize patterns quickly in interviews

        **Master these 20 patterns:**

        | Pattern | Problem Count | Priority |
        |---------|---------------|----------|
        | Two Pointers | 15 | High |
        | Sliding Window | 15 | High |
        | Binary Search | 15 | High |
        | BFS/DFS | 20 | High |
        | Dynamic Programming | 25 | High |
        | Hash Map | 15 | Medium |
        | Fast & Slow Pointers | 10 | Medium |
        | Backtracking | 15 | Medium |
        | Greedy | 10 | Medium |
        | Others | 20 | Low |

    ---

    ## Tracking Progress

    === "Daily Template"

        ```markdown
        ## [Date] - Day X

        ### Focus: [Topic/Pattern]

        ### Problems Solved
        - [ ] [Problem Name] - [Difficulty] - [Time: __min] - [Status: ✅/⚠️/❌]
        - [ ] [Problem Name] - [Difficulty] - [Time: __min] - [Status: ✅/⚠️/❌]

        ### Key Learnings
        - [Pattern recognized]
        - [New technique learned]
        - [Mistake to avoid]

        ### Tomorrow's Plan
        - [Next topic/problems]
        ```

    === "Weekly Review"

        ```markdown
        ## Week X Review

        ### Stats
        | Metric | Target | Actual |
        |--------|--------|--------|
        | Problems Solved | 20 | __ |
        | Easy / Medium / Hard | 8/10/2 | __/__/__ |
        | Mock Interviews | 1 | __ |
        | Success Rate | 70%+ | __% |

        ### What Went Well
        - [Strength 1]
        - [Strength 2]

        ### What Needs Work
        - [Weakness 1] → [Action plan]
        - [Weakness 2] → [Action plan]

        ### Next Week Focus
        - [Topics to cover]
        - [Problems to revisit]
        ```

    === "Tools"

        **Digital Tracking:**
        - **Spreadsheet:** Track problems, time, difficulty, success rate
        - **Notion:** Full dashboard with notes, resources, schedule
        - **GitHub:** Repository with solutions + README stats
        - **LeetCode:** Built-in tracking for premium users

        **Physical Tracking:**
        - **Bullet Journal:** Daily logs, weekly spreads
        - **Flashcards:** Key patterns and algorithms
        - **Wall Chart:** Visual progress (100-problem grid)

=== "⏰ Daily Practice Routines"

    ## 2-3 Hours (Balanced)

    **Best for:** Most people with moderate preparation time

    | Time Block | Duration | Activity | Details |
    |------------|----------|----------|---------|
    | **Warm-up** | 15-20 min | 1 easy problem | Build momentum, recall patterns |
    | **Main Practice** | 90-120 min | 2 medium problems | Core learning, 45 min each |
    | **Review** | 20-30 min | Previous problems | Spaced repetition, improve solutions |
    | **Study** | 15-20 min | Theory/patterns | Read articles, watch videos |

    **Sample Schedule:**
    ```
    6:00 PM - Warm-up: Array easy problem
    6:20 PM - Main: Tree medium problem
    7:05 PM - Main: DP medium problem
    7:50 PM - Review: Problems from 3 days ago
    8:15 PM - Study: DP patterns article
    8:30 PM - Done
    ```

    ---

    ## 4-5 Hours (Intensive)

    **Best for:** Serious interview prep, limited timeline

    | Time Block | Duration | Activity | Details |
    |------------|----------|----------|---------|
    | **Morning** | 2 hours | 2 medium problems | Different categories |
    | **Break** | - | Rest, exercise, meals | - |
    | **Afternoon** | 2 hours | 1 hard problem | Challenge yourself |
    | **Evening** | 1 hour | Mock interview OR review | Practice or reflection |

    **Sample Schedule:**
    ```
    8:00 AM - Graph medium problem
    9:00 AM - String medium problem
    10:00 AM - Break
    2:00 PM - DP hard problem
    4:00 PM - Break
    7:00 PM - Mock interview session
    8:00 PM - Done
    ```

    ---

    ## 30-60 Min (Limited Time)

    **Best for:** Maintaining skills, busy professionals

    | Strategy | Details |
    |----------|---------|
    | **Daily Focus** | 1 problem per day, alternate categories |
    | **Quality > Quantity** | Fully understand each solution |
    | **Weekend Boost** | 2-3 hours on Sat/Sun for harder problems |
    | **Micro-learning** | 10-min theory reviews during commute |

    **Weekly Plan:**
    ```
    Mon: Array problem (30 min)
    Tue: Tree problem (45 min)
    Wed: String problem (30 min)
    Thu: Graph problem (60 min)
    Fri: Review all 4 problems (30 min)
    Sat: 2 medium problems (2 hours)
    Sun: 1 hard + mock interview (3 hours)
    ```

=== "💬 Problem-Solving Framework"

    ## The 6-Step Method

    **Use this framework EVERY problem to build muscle memory**

    ### 1. Understand (5-10 min)

    | Action | Questions to Ask |
    |--------|------------------|
    | **Read carefully** | What are the inputs? What's the expected output? |
    | **Clarify** | "Can the array be empty?" "Are there negative numbers?" |
    | **Examples** | Work through 2-3 examples manually |
    | **Constraints** | Time limit? Space limit? Input size? |

    **Example:**
    ```python
    # Problem: Find two numbers that sum to target

    # Questions to ask:
    # - Can I use the same element twice? → No
    # - Are there always exactly two numbers? → Yes, guaranteed
    # - Is the array sorted? → No
    # - Can there be negative numbers? → Yes
    ```

    ---

    ### 2. Plan (5-15 min)

    | Action | Details |
    |--------|---------|
    | **Brute force first** | Always verbalize the naive O(n²) or O(2ⁿ) approach |
    | **Think patterns** | "This looks like [two pointers / sliding window / BFS]" |
    | **Consider data structures** | Hash map? Heap? Stack? |
    | **Edge cases** | Empty input, single element, duplicates, negatives |

    **Communication Template:**
    ```
    "My first thought is [brute force approach] which would be O(__).
    But I think we can optimize using [pattern/data structure] because [reason].
    This would give us O(__) time and O(__) space.
    Let me sketch this out..."
    ```

    ---

    ### 3. Code (15-25 min)

    | Best Practice | Why |
    |---------------|-----|
    | **Talk while coding** | Shows your thought process |
    | **Use good variable names** | `left`, `right` vs `i`, `j` |
    | **Write helper functions** | Shows code organization |
    | **Comment complex parts** | Clarifies intent |

    **Coding Tips:**
    ```python
    # Good: Clear and descriptive
    def find_two_sum(nums, target):
        seen = {}  # value -> index mapping

        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i

        return []  # No solution found

    # Bad: Confusing names
    def solve(a, t):
        d = {}
        for i in range(len(a)):
            if t - a[i] in d:
                return [d[t-a[i]], i]
            d[a[i]] = i
    ```

    ---

    ### 4. Test (5-10 min)

    | Test Case | Purpose |
    |-----------|---------|
    | **Example from problem** | Verify basic functionality |
    | **Edge case: Empty** | `[]` or `""` or `None` |
    | **Edge case: Single element** | `[1]` |
    | **Edge case: Two elements** | Minimum valid input |
    | **Edge case: All same** | `[5, 5, 5, 5]` |
    | **Edge case: Large input** | Discuss, don't run (time constraint) |

    **Walkthrough Example:**
    ```
    "Let me trace through with [2, 7, 11, 15], target = 9:
    - i=0, num=2: complement=7, not in seen, add 2→0
    - i=1, num=7: complement=2, found at index 0! Return [0, 1]

    Looks correct. Now let me check edge case with [3, 3], target = 6:
    - i=0, num=3: complement=3, not in seen, add 3→0
    - i=1, num=3: complement=3, found at index 0! Return [0, 1]

    Good, handles duplicates."
    ```

    ---

    ### 5. Optimize (5-10 min)

    | Question | Analysis |
    |----------|----------|
    | **Time complexity?** | "This is O(n) because single pass" |
    | **Space complexity?** | "O(n) worst case for hash map" |
    | **Can we do better?** | "If sorted, could use O(1) space with two pointers" |
    | **Tradeoffs?** | "Sorting takes O(n log n), loses original indices" |

    **Discussion Template:**
    ```
    "My solution is O(n) time and O(n) space.

    Time: We iterate through the array once, and hash map operations are O(1).
    Space: In the worst case, we store n-1 elements in the hash map.

    Alternative: If we sort first, we can use two pointers for O(1) space,
    but sorting is O(n log n) time, which is slower.

    For this problem, I think O(n) time with O(n) space is the best approach."
    ```

    ---

    ### 6. Reflect (Post-Interview)

    **Immediately after, document:**
    - Problem statement (exact wording if possible)
    - Your approach and code
    - Interviewer's feedback
    - What went well, what to improve
    - Patterns/techniques you'll remember

    ---

    ## Communication Techniques

    === "Think Aloud"

        **Do:**
        - "I'm thinking we could use a hash map here because..."
        - "Let me consider the time complexity... this loop is O(n)..."
        - "I notice the array is sorted, so binary search might work..."

        **Don't:**
        - *Long silences while thinking*
        - "Hmm... umm... let me think..." *[silence]*
        - Writing code without explaining

    === "Clarify Requirements"

        **Before coding, always ask:**
        ```
        Input clarification:
        - "Can the input be empty?"
        - "What's the range of values?"
        - "Are there duplicates?"

        Output clarification:
        - "Should I return indices or values?"
        - "What if there's no solution?"
        - "Is there always exactly one answer?"

        Constraint clarification:
        - "What's the input size? Should I optimize for n < 100 or n < 10^6?"
        - "Is there a space constraint?"
        ```

    === "Explain Tradeoffs"

        **Template:**
        ```
        "I see two approaches:

        Approach 1 [describe]:
        - Time: O(__)
        - Space: O(__)
        - Pros: [list]
        - Cons: [list]

        Approach 2 [describe]:
        - Time: O(__)
        - Space: O(__)
        - Pros: [list]
        - Cons: [list]

        I think Approach [X] is better because [reason].
        Does that sound good to you?"
        ```

    === "Acknowledge Limitations"

        **Be honest about what you don't know:**
        - "I'm not immediately seeing an optimal solution. Let me start with the brute force and optimize from there."
        - "I haven't seen this exact pattern before, but it reminds me of [similar problem]."
        - "I'm confident this works, but I'd want to test more edge cases given more time."

=== "🎭 Mock Interviews"

    ## Types of Practice

    | Type | Setup | Benefits | Frequency |
    |------|-------|----------|-----------|
    | **Self-Mock** | Timer + problem | Build speed, identify gaps | 2-3x/week |
    | **Peer Mock** | Friend + video call | Practice communication | 1x/week |
    | **Platform Mock** | Pramp, interviewing.io | Real interview feel | 2-3x/month |
    | **Recorded Mock** | Record yourself | Review body language, speech | 1x/week |

    ---

    ## Conducting Effective Mocks

    === "Setup"

        **Environment:**
        - Quiet space, no interruptions
        - Timer visible (45-60 min)
        - Code editor or whiteboard
        - Video on (if peer/platform)

        **Problem Selection:**
        - Medium difficulty for regular practice
        - Mix of familiar and unfamiliar patterns
        - From companies you're targeting
        - Set timer: 5 min understand, 10 min plan, 25 min code, 10 min test

    === "During"

        **Act as if it's real:**
        - Introduce yourself briefly
        - Ask clarifying questions
        - Think aloud constantly
        - Write clean code with good names
        - Test your solution
        - Discuss complexity

        **If stuck:**
        - Ask for hints (in real interview, you would)
        - Think through similar problems
        - Start with brute force
        - Break problem into smaller pieces

    === "After"

        **Immediate Review (10-15 min):**
        - What went well?
        - What mistakes did you make?
        - Did you finish in time?
        - Was your communication clear?

        **Feedback (if peer/platform):**
        - Ask specific questions: "Was my explanation clear?" "Did I test enough?"
        - Take notes on all feedback
        - Watch for patterns in multiple mocks

        **Follow-up (same day):**
        - Solve the problem again from scratch
        - Research optimal solution
        - Add to spaced repetition schedule

    ---

    ## Improvement Loop

    ```mermaid
    graph LR
        A[Mock Interview] --> B[Document Experience]
        B --> C[Analyze Gaps]
        C --> D[Study Weak Areas]
        D --> E[Practice Similar Problems]
        E --> F[Next Mock]
        F --> A
    ```

    **Document Template:**
    ```markdown
    ## Mock #X - [Date]

    ### Problem
    [Brief description or LeetCode link]

    ### My Performance
    - Time to solve: __ min
    - Solution correctness: Complete / Partial / Wrong
    - Communication: Good / Ok / Poor
    - Bugs found: __

    ### What Went Well
    - [Positive 1]
    - [Positive 2]

    ### What to Improve
    - [Gap 1] → Action: [specific practice]
    - [Gap 2] → Action: [specific practice]

    ### Feedback Received
    - [Interviewer comment 1]
    - [Interviewer comment 2]

    ### Next Steps
    - [ ] Solve similar problems: [list]
    - [ ] Study technique: [topic]
    - [ ] Practice communication: [aspect]
    ```

    ---

    ## Platform Recommendations

    | Platform | Cost | Best For | Features |
    |----------|------|----------|----------|
    | **Pramp** | Free | Beginners | Peer matching, structured feedback |
    | **interviewing.io** | Free + Paid | Realistic practice | Anonymous, real engineers, detailed feedback |
    | **AlgoExpert** | $99/year | Guided prep | Video explanations, structured content |
    | **LeetCode Premium** | $35/month | Company-specific | Frequency data, mock assessments |
    | **Hired.com** | Free | Job seekers | Practice + actual job matching |

=== "🧠 Mental Preparation"

    ## Handling Interview Anxiety

    === "Before Interview"

        **1 Week Before:**
        - Reduce problem-solving intensity (avoid burnout)
        - Focus on review, not new topics
        - Get 7-8 hours of sleep nightly
        - Exercise daily (even just 20-min walk)

        **1 Day Before:**
        - Review your cheat sheet (patterns, not problems)
        - Do 1-2 easy problems (confidence boost)
        - Prepare questions to ask interviewer
        - No all-night studying (sleep > cramming)

        **1 Hour Before:**
        - Listen to pump-up music or meditate
        - Review your "wins" (problems you've solved)
        - Breathing exercises (4-7-8 technique)
        - Positive affirmations: "I am prepared. I can solve this."

    === "During Interview"

        **If Stuck:**
        - Take a deep breath
        - "Let me think through this step by step..."
        - Talk through brute force first
        - Ask for a hint: "I'm considering [approach], does that seem right?"

        **If Making Mistake:**
        - Stay calm, bugs are normal
        - "Let me trace through this again..."
        - Test with simple example
        - Interviewer will often give hints

        **If Running Out of Time:**
        - "I don't think I'll finish the full implementation, can I explain the rest?"
        - Outline remaining steps clearly
        - Discuss complexity and edge cases
        - Show you understand even if code isn't done

    === "Techniques"

        **4-7-8 Breathing:**
        ```
        1. Breathe in through nose for 4 counts
        2. Hold breath for 7 counts
        3. Exhale through mouth for 8 counts
        4. Repeat 3-4 times
        ```

        **Positive Self-Talk:**
        - Replace "I can't solve this" → "I can figure this out step by step"
        - Replace "I'm not good enough" → "I've solved 200 problems, I'm prepared"
        - Replace "What if I fail?" → "This is practice for getting better"

        **Grounding Technique (5-4-3-2-1):**
        ```
        5 things you can see
        4 things you can touch
        3 things you can hear
        2 things you can smell
        1 thing you can taste
        ```
        *(Use before interview to calm nerves)*

    ---

    ## Maintaining Motivation

    === "Set Milestones"

        **Short-term (Daily/Weekly):**
        - Solve 3 problems today ✅
        - Finish Arrays module this week ✅
        - Successfully explain solution to friend ✅

        **Medium-term (Monthly):**
        - Complete 80 problems ✅
        - Pass 3 mock interviews ✅
        - Master Dynamic Programming ✅

        **Long-term (3-6 Months):**
        - Land target company offer ✅
        - Rating 1800+ on Codeforces ✅
        - Interview-ready for senior roles ✅

    === "Accountability"

        **Methods:**
        - Study group (weekly check-ins)
        - Public commitment (social media updates)
        - GitHub repository (visible progress)
        - Accountability partner (daily messages)
        - Paid commitment (betting on yourself)

        **Example Commitment:**
        ```
        "I commit to solving 3 problems every day for the next 30 days.
        I'll post my progress daily on Twitter/LinkedIn.
        Accountability partner: @friend"
        ```

    === "Celebrate Wins"

        **Small Wins:**
        - Solved first medium problem → Treat yourself to coffee ☕
        - 7-day streak → Watch favorite show episode 📺
        - First successful mock → Share achievement with friend 🎉

        **Big Wins:**
        - 100 problems solved → Nice dinner out 🍽️
        - Passed phone screen → Buy something you've wanted 🎁
        - Got the offer → Plan celebration with loved ones 🎊

    ---

    ## Preventing Burnout

    **Warning Signs:**
    - Dreading practice sessions
    - Decreased problem-solving ability
    - Physical symptoms (headaches, fatigue)
    - Irritability, frustration
    - Avoidance behavior

    **Prevention Strategies:**

    | Strategy | Implementation |
    |----------|----------------|
    | **Scheduled breaks** | 1-2 days off per week, NO studying |
    | **Vary difficulty** | Mix easy problems when feeling overwhelmed |
    | **Switch topics** | If stuck on DP, do some array problems |
    | **Social connection** | Study groups, pair programming |
    | **Physical activity** | Exercise 20+ min daily |
    | **Remember why** | Review your motivation regularly |

    **If Burned Out:**
    1. Take 3-5 days completely off
    2. Return with only easy problems
    3. Focus on topics you enjoy
    4. Reduce daily time commitment
    5. Talk to mentor or friend about feelings

=== "📚 Resources"

    ## Practice Platforms

    | Platform | Best For | Cost | Key Features |
    |----------|----------|------|-------------|
    | **[LeetCode](https://leetcode.com/)** | Interview prep | Free + $35/mo premium | 2500+ problems, company tags, discuss forum |
    | **[HackerRank](https://www.hackerrank.com/)** | Learning fundamentals | Free | Guided learning paths, certificates |
    | **[Codeforces](https://codeforces.com/)** | Competitive programming | Free | Live contests, rating system |
    | **[AlgoExpert](https://www.algoexpert.io/)** | Structured course | $99/year | 160 curated problems, video explanations |
    | **[NeetCode](https://neetcode.io/)** | Pattern-based learning | Free + $69 course | 150 problems grouped by pattern |

    ---

    ## Books

    === "Algorithm Interview"

        **Must-Read:**

        | Book | Author | Best For |
        |------|--------|----------|
        | **Cracking the Coding Interview** | Gayle McDowell | Comprehensive interview prep, Big O, data structures |
        | **Elements of Programming Interviews** | Adnan Aziz et al. | In-depth problem analysis, detailed solutions |
        | **Programming Interviews Exposed** | John Mongan et al. | Communication skills, soft skills, junior roles |

        **Recommended:**
        - **"Daily Coding Problem"** by Alex Miller (variety of real interview problems)
        - **"Competitive Programmer's Handbook"** by Antti Laaksonen (CP focus, free PDF)

    === "System Design"

        **Must-Read:**

        | Book | Author | Best For |
        |------|--------|----------|
        | **System Design Interview Vol 1 & 2** | Alex Xu | Step-by-step framework, visual diagrams |
        | **Designing Data-Intensive Applications** | Martin Kleppmann | Deep understanding of distributed systems |

        **Supplementary:**
        - **"System Design Primer"** (GitHub) - Free, visual, example-rich
        - **"Grokking the System Design Interview"** (Educative) - Interactive course

    === "Algorithms Deep Dive"

        **For Mastery:**

        | Book | Author | Best For |
        |------|--------|----------|
        | **Algorithm Design Manual** | Steven Skiena | Practical problem-solving, war stories |
        | **Introduction to Algorithms (CLRS)** | Cormen et al. | Comprehensive reference, proofs |
        | **Algorithms** | Robert Sedgewick | Excellent visualizations, Java implementations |

    ---

    ## Video Resources

    | Channel/Course | Platform | Focus | Cost |
    |----------------|----------|-------|------|
    | **NeetCode** | YouTube | Problem explanations, patterns | Free |
    | **Abdul Bari** | YouTube | Algorithm fundamentals, visualizations | Free |
    | **William Fiset** | YouTube | Graph algorithms, DP | Free |
    | **Back To Back SWE** | YouTube | Interview problems, in-depth | Free |
    | **AlgoExpert** | algoexpert.io | Structured video course | $99/year |
    | **Grokking the Coding Interview** | Educative | Pattern-based learning | $79/year |

    ---

    ## Tools

    === "Tracking"

        - **Notion:** All-in-one workspace ([free template](https://www.notion.so/))
        - **Google Sheets:** Progress tracker, statistics
        - **GitHub:** Code repository + README stats
        - **Anki:** Flashcards for spaced repetition (free)
        - **Obsidian:** Markdown-based notes with linking (free)

    === "Coding"

        - **VS Code:** Local coding with extensions
        - **LeetCode Playground:** In-browser testing
        - **Replit:** Online IDE with collaboration
        - **Python Tutor:** Visualize code execution (free)
        - **VisuAlgo:** Algorithm visualizations (free)

    ---

    ## Communities

    | Community | Platform | Best For |
    |-----------|----------|----------|
    | **r/leetcode** | Reddit | Problem discussions, motivation, memes |
    | **r/cscareerquestions** | Reddit | Career advice, interview experiences |
    | **Blind** | App/Web | Anonymous company discussions, salary data |
    | **Discord Servers** | Discord | Real-time help, study groups (search "LeetCode Discord") |
    | **Local Meetups** | Meetup.com | In-person practice, networking |

    ---

    ## Cheat Sheets

    **Create your own, but here are starting points:**
    - [LeetCode Patterns Cheat Sheet](https://seanprashad.com/leetcode-patterns/)
    - [Big O Cheat Sheet](https://www.bigocheatsheet.com/)
    - [Data Structures Cheat Sheet](https://www.interviewcake.com/data-structures-reference)
    - [Algorithm Patterns Guide](https://algo.monster/problems/stats)

---

## Summary

**Key Takeaways:**

| Priority | Action | Why |
|----------|--------|-----|
| **1** | Choose a plan that fits your timeline | Consistency beats intensity |
| **2** | Practice communication in every problem | Interviews test process, not just answers |
| **3** | Do regular mock interviews | Builds confidence, reveals gaps |
| **4** | Track progress and adjust | What gets measured gets improved |
| **5** | Take care of mental health | Burnout kills progress |

**Final Checklist:**

- [ ] Assessed my starting point and chosen a timeline
- [ ] Created daily/weekly practice schedule
- [ ] Set up progress tracking system
- [ ] Identified 3-4 focus areas based on weaknesses
- [ ] Scheduled first mock interview
- [ ] Joined community for support
- [ ] Prepared mental preparation techniques

**Remember:** Every expert was once a beginner. Consistent daily practice with deliberate improvement will get you there. You've got this! 🚀
