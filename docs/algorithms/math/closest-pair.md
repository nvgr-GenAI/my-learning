# Closest Pair of Points

The closest pair of points problem is a fundamental computational geometry problem: given n points in a metric space, find the pair of points with the smallest distance between them.

## Problem Statement

Given a set of points in a 2D plane, find the pair of points that are closest to each other.

## Naive Approach

The simplest approach is to check all possible pairs of points:

```python
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def closest_pair_naive(points):
    n = len(points)
    min_distance = float('inf')
    closest_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(points[i], points[j])
            if dist < min_distance:
                min_distance = dist
                closest_pair = (points[i], points[j])
    
    return min_distance, closest_pair
```

**Time Complexity**: O(n²) - We compare each pair of points.
**Space Complexity**: O(1) - Only constant extra space is needed.

## Divide and Conquer Approach

The efficient solution uses a divide and conquer approach with the following steps:

1. Sort all points according to x-coordinates
2. Divide the set of points into two equal-sized subsets by a vertical line x = mid
3. Recursively find the minimum distance in both subsets
4. Find the minimum distance among the pairs where one point is in the left subset and the other is in the right subset
5. Return the minimum of all these distances

```python
def closest_pair(points):
    # Sort points by x-coordinate
    points_sorted_by_x = sorted(points, key=lambda point: point[0])
    
    # Use the divide and conquer approach
    return closest_pair_recursive(points_sorted_by_x)

def closest_pair_recursive(points_sorted_by_x):
    n = len(points_sorted_by_x)
    
    # Base cases
    if n <= 3:
        return closest_pair_naive(points_sorted_by_x)
    
    # Divide points into two halves
    mid = n // 2
    midpoint = points_sorted_by_x[mid]
    
    # Recursively find minimum distance in left and right halves
    left_min, left_pair = closest_pair_recursive(points_sorted_by_x[:mid])
    right_min, right_pair = closest_pair_recursive(points_sorted_by_x[mid:])
    
    # Find minimum of left_min and right_min
    if left_min < right_min:
        min_distance = left_min
        min_pair = left_pair
    else:
        min_distance = right_min
        min_pair = right_pair
    
    # Create a strip of points whose x-distance from midpoint is less than min_distance
    strip = [point for point in points_sorted_by_x if abs(point[0] - midpoint[0]) < min_distance]
    
    # Sort the strip by y-coordinate
    strip.sort(key=lambda point: point[1])
    
    # Find the closest pair in the strip
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and strip[j][1] - strip[i][1] < min_distance:
            dist = distance(strip[i], strip[j])
            if dist < min_distance:
                min_distance = dist
                min_pair = (strip[i], strip[j])
            j += 1
    
    return min_distance, min_pair
```

**Time Complexity**: O(n log n) - The divide-and-conquer algorithm runs in O(n log n) time.
**Space Complexity**: O(n) - We need extra space for the recursive calls and the strip.

## Key Insights

1. **Strip Consideration**: The algorithm's efficiency comes from the insight that we only need to check a limited number of points in the strip (those within a vertical distance of min_distance).

2. **Y-coordinate Optimization**: By sorting the strip by y-coordinate, we ensure that for each point, we only need to check at most 6 other points, which bounds the inner loop to constant time.

3. **Balancing the Divide**: Dividing the points into equal halves ensures logarithmic recursion depth.

## Applications

- Geographic information systems (GIS)
- Collision detection in video games
- Cluster analysis in data mining
- Image processing for feature detection

## Practice Problems

1. [SPOJ - CLOSEST](https://www.spoj.com/problems/CLOSEST/) - Closest Pair of Points
2. [CodeChef - CLOSEST](https://www.codechef.com/problems/CLOSEST) - Closest Pair
3. [UVa 10245](https://onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=1186) - The Closest Pair Problem

## Pro Tips

- For high-dimensional spaces, specialized data structures like k-d trees might be more efficient.
- The divide-and-conquer approach can be generalized to higher dimensions, but the strip checking becomes more complex.
- In practice, the naive O(n²) algorithm might be faster for small inputs due to lower constant factors.
- Be careful with floating-point precision issues when calculating distances.

## Related Topics

- [Convex Hull](convex-hull.md)
- [Line Intersection](line-intersection.md)
- [Vectors and Matrices](vectors-matrices.md)
