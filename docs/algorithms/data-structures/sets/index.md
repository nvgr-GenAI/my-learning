# Sets

## 📚 Overview

A **Set** is a collection data structure that stores **unique elements** with no specific order (in hash-based sets) or in sorted order (in tree-based sets). Sets are fundamental for operations like deduplication, membership testing, and mathematical set operations (union, intersection, difference).

**Key characteristic:** No duplicates allowed - attempting to add a duplicate element has no effect.

---

## 🎯 What You'll Learn

### Core Concepts

- **Set Theory Basics**: Mathematical foundation of sets
- **Uniqueness Enforcement**: How sets prevent duplicates
- **Implementation Types**: Hash-based vs Tree-based
- **Set Operations**: Union, intersection, difference, symmetric difference

### Real-World Applications

- **Deduplication**: Remove duplicate entries from data
- **Membership Testing**: Quick "does this exist?" checks
- **Access Control**: Check if user has specific permissions
- **Tag Systems**: Manage unique tags or categories

---

## 📖 Learning Path

### 1. 🔧 Implementation Types

Choose your set implementation based on requirements:

#### Hash Set
**Fast operations, no ordering**

[📘 Hash Set](hash-set.md){ .md-button .md-button--primary }

- O(1) average case for add, remove, contains
- No guaranteed order of elements
- Best for: Speed, when order doesn't matter
- Uses: Deduplication, membership checks

#### Tree Set
**Sorted order, slightly slower**

[📘 Tree Set](tree-set.md){ .md-button .md-button--primary }

- O(log n) for add, remove, contains
- Elements maintained in sorted order
- Best for: Range queries, ordered iteration
- Uses: Sorted unique collections, range operations

#### Bit Set
**Space-efficient for integers**

[📘 Bit Set](bit-set.md){ .md-button .md-button--primary }

- Extremely space-efficient (1 bit per possible value)
- Very fast operations using bitwise ops
- Best for: Small integer ranges (e.g., 0-1000)
- Uses: Flags, permissions, Sieve of Eratosthenes

---

### 2. 🛠️ Common Operations

Master the essential set operations:

#### Membership Testing
[📘 Membership Operations](membership.md){ .md-button }

- Check if element exists: O(1) hash, O(log n) tree
- Add new element
- Remove element
- Use cases: User authentication, permission checks

#### Deduplication
[📘 Deduplication Techniques](deduplication.md){ .md-button }

- Remove duplicates from arrays/lists
- Find unique elements
- Count distinct elements
- Use cases: Data cleaning, analytics

#### Union & Intersection
[📘 Union & Intersection](union-intersection.md){ .md-button }

- Combine sets (union)
- Find common elements (intersection)
- Efficient algorithms for set combination
- Use cases: Feature matching, recommendation systems

#### Set Difference
[📘 Set Difference](difference.md){ .md-button }

- Elements in A but not in B
- Symmetric difference (A ∪ B) - (A ∩ B)
- Complement operations
- Use cases: Change detection, comparison

---

### 3. 💪 Practice Problems

Progress from easy to hard problems:

#### 🟢 Easy Problems
[📝 Easy Set Problems](easy-problems.md){ .md-button }

Perfect for beginners:

- Contains Duplicate
- Intersection of Two Arrays
- Happy Number
- Single Number
- Distribute Candies

**Focus:** Basic set operations, membership testing

---

#### 🟡 Medium Problems
[📝 Medium Set Problems](medium-problems.md){ .md-button }

Intermediate challenges:

- Longest Consecutive Sequence
- Set Matrix Zeroes
- Intersection of Multiple Arrays
- Group Anagrams (using sets)
- Valid Sudoku

**Focus:** Combining sets with other techniques, optimization

---

#### 🔴 Hard Problems
[📝 Hard Set Problems](hard-problems.md){ .md-button }

Advanced mastery:

- Max Points on a Line
- Substring with Concatenation of All Words
- Freedom Trail
- Minimum Window Subsequence

**Focus:** Complex set applications, performance optimization

---

## 🎓 Quick Reference

### Set vs Array vs Hash Table

| Feature | Set | Array | Hash Table |
|---------|-----|-------|------------|
| **Duplicates** | No | Yes | Keys: No, Values: Yes |
| **Order** | Hash: No, Tree: Yes | Yes | No |
| **Lookup** | O(1) or O(log n) | O(n) | O(1) |
| **Add** | O(1) or O(log n) | O(1) at end, O(n) middle | O(1) |
| **Delete** | O(1) or O(log n) | O(n) | O(1) |
| **Iteration** | All elements | By index | All keys/values |

### Time Complexity

| Operation | Hash Set | Tree Set | Bit Set |
|-----------|----------|----------|---------|
| **Add** | O(1) avg | O(log n) | O(1) |
| **Remove** | O(1) avg | O(log n) | O(1) |
| **Contains** | O(1) avg | O(log n) | O(1) |
| **Union** | O(m+n) | O(m+n) | O(max(m,n)) |
| **Intersection** | O(min(m,n)) | O(m+n) | O(max(m,n)) |
| **Space** | O(n) | O(n) | O(max_value) |

---

## When to Use Sets

### ✅ Use Sets When:

- **Need unique elements**: Automatically enforce no duplicates
- **Membership testing**: Fast "is X in the collection?" checks
- **Set operations**: Union, intersection, difference operations
- **Deduplication**: Remove duplicates from collections
- **No need for counts**: Don't care how many times element appears
- **No need for order** (hash set): Element order doesn't matter

### ❌ Avoid Sets When:

- **Need counts**: Use Hash Table/Counter instead
- **Need specific order** (except Tree Set): Use Array/List
- **Need duplicates**: Use Array/List or Multiset
- **Memory constrained + large integer range**: Hash Set uses more memory than array
- **Need index-based access**: Use Array instead

---

## Common Patterns

### 1. Deduplication Pattern
```python
# Remove duplicates from array
def remove_duplicates(arr):
    return list(set(arr))  # Order not preserved
    # OR for preserved order:
    seen = set()
    result = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
```

### 2. Two Set Pattern
```python
# Find elements in A but not in B
def difference(A, B):
    set_a = set(A)
    set_b = set(B)
    return set_a - set_b
```

### 3. Intersection Pattern
```python
# Find common elements
def intersection(A, B):
    return set(A) & set(B)
```

### 4. Seen/Visited Pattern
```python
# Track visited elements
visited = set()
for element in collection:
    if element in visited:
        # Already seen
        continue
    visited.add(element)
    # Process element
```

---

## Learning Strategy

### For Beginners
1. Start with **Hash Set** - easiest to understand
2. Practice **Deduplication** problems
3. Learn **membership testing** patterns
4. Master basic set operations: add, remove, contains

### For Intermediate
1. Learn **Tree Set** for sorted requirements
2. Practice **Union/Intersection** problems
3. Combine sets with other data structures
4. Optimize time complexity using sets

### For Advanced
1. Master **Bit Set** for specific use cases
2. Understand when NOT to use sets
3. Optimize space vs time trade-offs
4. Apply sets in complex algorithmic problems

---

## Next Steps

**New to sets?**
Start with [Hash Set](hash-set.md) to understand the basics.

**Know the basics?**
Practice with [Easy Problems](easy-problems.md) to build confidence.

**Ready for challenges?**
Move to [Medium Problems](medium-problems.md) for interview prep.

**Want specific operations?**
Jump to [Deduplication](deduplication.md) or [Union & Intersection](union-intersection.md).

---

**Remember:** Sets are your go-to data structure whenever you need to enforce uniqueness or perform fast membership checks. If you find yourself checking "if x in list" repeatedly, you probably need a set!
