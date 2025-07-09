# Closest Pair of Points 🔍

Find the closest pair of points in a 2D plane using the divide-and-conquer approach.

## 🎯 Problem Statement

Given n points in a 2D plane, find the two points that are closest to each other.

**Input**: Array of n points with (x, y) coordinates
**Output**: Distance between the closest pair of points

## 🧠 Algorithm Approach

### Divide & Conquer Strategy

1. **Divide**: Split points into two halves by x-coordinate
2. **Conquer**: Recursively find closest pairs in each half
3. **Combine**: Check for closer pairs across the dividing line

## 📝 Implementation

```python
import math
from typing import List, Tuple

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"

def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def closest_pair_brute_force(points: List[Point]) -> Tuple[float, Point, Point]:
    """Brute force approach for small arrays"""
    n = len(points)
    min_dist = float('inf')
    closest_p1, closest_p2 = None, None
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                closest_p1, closest_p2 = points[i], points[j]
    
    return min_dist, closest_p1, closest_p2

def strip_closest(strip: List[Point], d: float) -> Tuple[float, Point, Point]:
    """Find closest points in strip of width 2d"""
    min_dist = d
    closest_p1, closest_p2 = None, None
    
    # Sort strip by y-coordinate
    strip.sort(key=lambda p: p.y)
    
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j].y - strip[i].y) < min_dist:
            dist = distance(strip[i], strip[j])
            if dist < min_dist:
                min_dist = dist
                closest_p1, closest_p2 = strip[i], strip[j]
            j += 1
    
    return min_dist, closest_p1, closest_p2

def closest_pair_rec(px: List[Point], py: List[Point]) -> Tuple[float, Point, Point]:
    """Recursive closest pair implementation"""
    n = len(px)
    
    # Base case: use brute force for small arrays
    if n <= 3:
        return closest_pair_brute_force(px)
    
    # Divide
    mid = n // 2
    midpoint = px[mid]
    
    pyl = [point for point in py if point.x <= midpoint.x]
    pyr = [point for point in py if point.x > midpoint.x]
    
    # Conquer
    dl, p1_l, p2_l = closest_pair_rec(px[:mid], pyl)
    dr, p1_r, p2_r = closest_pair_rec(px[mid:], pyr)
    
    # Find minimum of the two
    if dl <= dr:
        d = dl
        closest_p1, closest_p2 = p1_l, p2_l
    else:
        d = dr
        closest_p1, closest_p2 = p1_r, p2_r
    
    # Create strip array
    strip = [point for point in py if abs(point.x - midpoint.x) < d]
    
    # Find closest points in strip
    strip_dist, strip_p1, strip_p2 = strip_closest(strip, d)
    
    if strip_dist < d:
        return strip_dist, strip_p1, strip_p2
    else:
        return d, closest_p1, closest_p2

def closest_pair(points: List[Point]) -> Tuple[float, Point, Point]:
    """Main function to find closest pair"""
    if len(points) < 2:
        return float('inf'), None, None
    
    # Sort points by x and y coordinates
    px = sorted(points, key=lambda p: p.x)
    py = sorted(points, key=lambda p: p.y)
    
    return closest_pair_rec(px, py)

# Example usage
if __name__ == "__main__":
    points = [
        Point(2, 3),
        Point(12, 30),
        Point(40, 50),
        Point(5, 1),
        Point(12, 10),
        Point(3, 4)
    ]
    
    min_dist, p1, p2 = closest_pair(points)
    print(f"Closest pair: {p1} and {p2}")
    print(f"Distance: {min_dist:.2f}")
```

## ⚡ Time Complexity Analysis

- **Time Complexity**: O(n log n)
  - Sorting: O(n log n)
  - Recursive calls: T(n) = 2T(n/2) + O(n)
  - Strip processing: O(n) per level
  
- **Space Complexity**: O(n) for auxiliary arrays

## 🔄 Step-by-Step Example

```text
Points: [(2,3), (12,30), (40,50), (5,1), (12,10), (3,4)]

Step 1: Sort by x-coordinate
Sorted: [(2,3), (3,4), (5,1), (12,10), (12,30), (40,50)]

Step 2: Divide at midpoint
Left half: [(2,3), (3,4), (5,1)]
Right half: [(12,10), (12,30), (40,50)]

Step 3: Recursively find closest in each half
Left: closest is (2,3) and (3,4), distance = √2 ≈ 1.41
Right: closest is (12,10) and (12,30), distance = 20

Step 4: Check strip around dividing line
Strip width = 2 * 1.41 = 2.82
Strip points: [(5,1)] (only point within strip width)

Step 5: Return minimum distance = 1.41
```

## 🎯 Key Insights

1. **Divide Strategy**: Split by x-coordinate median
2. **Strip Optimization**: Only check points within distance d of dividing line
3. **Y-sorting in Strip**: Reduces comparisons to at most 7 per point
4. **Base Case**: Switch to brute force for small arrays (n ≤ 3)

## 📊 Performance Comparison

| Approach | Time Complexity | Space Complexity |
|----------|----------------|------------------|
| Brute Force | O(n²) | O(1) |
| Divide & Conquer | O(n log n) | O(n) |

## 🔧 Optimizations

1. **Preprocessing**: Sort points once at the beginning
2. **Strip Pruning**: Limit strip width to current minimum distance
3. **Early Termination**: Stop if distance becomes very small
4. **Memory Optimization**: Reuse arrays instead of creating new ones

## 💡 Applications

- **Computer Graphics**: Collision detection, nearest neighbor queries
- **Geographic Information Systems**: Finding closest facilities
- **Clustering**: Initial step in hierarchical clustering algorithms
- **Computational Geometry**: Building blocks for other algorithms

---

*The closest pair problem demonstrates the power of divide-and-conquer in reducing time complexity from O(n²) to O(n log n).*
