# Convex Hull

## 🎯 Overview

The convex hull of a set of points is the smallest convex polygon that contains all the points. It can be visualized as the shape formed by stretching a rubber band around all the points. This fundamental geometric construction has applications in various fields including computer graphics, pattern recognition, geographic information systems (GIS), and robotics.

## 📋 Core Concepts

### Properties of a Convex Hull

1. All points in the original set are either inside the hull or on its boundary
2. All internal angles of the hull are less than 180 degrees
3. The hull is the smallest convex polygon containing all points
4. Only a subset of the original points (those on the boundary) form the hull

### Algorithms Classification

Convex hull algorithms can be classified based on:

- **Dimension**: 2D, 3D, or higher dimensions
- **Output Sensitivity**: Whether the time complexity depends on output size
- **Execution Strategy**: Incremental, divide-and-conquer, gift wrapping, etc.

## ⚙️ Algorithm Implementations

### Graham Scan Algorithm

The Graham scan is an efficient algorithm that builds the convex hull in O(n log n) time:

```python
def graham_scan(points):
    """
    Compute the convex hull of a set of 2D points using Graham's scan algorithm.
    
    Args:
        points: List of (x, y) tuples representing points in 2D space
        
    Returns:
        List of (x, y) tuples representing vertices of the convex hull in counterclockwise order
    """
    def orientation(p, q, r):
        """
        Determine orientation of triplet (p, q, r).
        Returns:
            0: Collinear
            1: Clockwise
            2: Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2
    
    # Find the point with the lowest y-coordinate (and leftmost if tied)
    lowest = min(points, key=lambda p: (p[1], p[0]))
    
    # Sort points by polar angle with respect to the lowest point
    sorted_points = sorted(
        points, 
        key=lambda p: (
            math.atan2(p[1] - lowest[1], p[0] - lowest[0]), 
            (p[0] - lowest[0])**2 + (p[1] - lowest[1])**2
        )
    )
    
    # Remove duplicates
    i = 1
    while i < len(sorted_points):
        if sorted_points[i] == sorted_points[i-1]:
            sorted_points.pop(i)
        else:
            i += 1
    
    # Build the hull
    hull = []
    for p in sorted_points:
        # Remove points that make non-left turns
        while len(hull) > 1 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()
        hull.append(p)
    
    return hull
```

### Jarvis March (Gift Wrapping) Algorithm

Jarvis march is a simple algorithm with O(nh) time complexity where h is the number of hull vertices:

```python
def jarvis_march(points):
    """
    Compute the convex hull of a set of 2D points using the Jarvis march algorithm.
    
    Args:
        points: List of (x, y) tuples representing points in 2D space
        
    Returns:
        List of (x, y) tuples representing vertices of the convex hull in counterclockwise order
    """
    def orientation(p, q, r):
        """Calculate orientation of triplet (p, q, r)"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2
    
    n = len(points)
    if n < 3:
        return points
    
    # Find the leftmost point
    l = min(range(n), key=lambda i: points[i][0])
    
    hull = []
    p = l
    
    while True:
        hull.append(points[p])
        
        # Find the next point on hull
        q = (p + 1) % n
        for i in range(n):
            # If i is more counterclockwise than current q, update q
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        
        p = q
        
        # Break when we get back to the first point
        if p == l:
            break
    
    return hull
```

### QuickHull Algorithm

QuickHull is a divide-and-conquer algorithm with worst-case O(n²) but typically better performance:

```python
def quick_hull(points):
    """
    Compute the convex hull of a set of 2D points using the QuickHull algorithm.
    
    Args:
        points: List of (x, y) tuples representing points in 2D space
        
    Returns:
        List of (x, y) tuples representing vertices of the convex hull in counterclockwise order
    """
    def find_hull(points, a, b, hull):
        """
        Find points on hull from set of points on the right side of line ab.
        """
        if not points:
            return
        
        # Find the point with maximum distance from line ab
        max_dist = 0
        max_point = None
        for p in points:
            dist = distance_from_line(p, a, b)
            if dist > max_dist:
                max_dist = dist
                max_point = p
        
        if max_point is None:
            return
        
        # Add the point to the hull
        hull.add(max_point)
        
        # Find points on the right of lines a-max_point and max_point-b
        s1 = [p for p in points if is_right_of_line(p, a, max_point)]
        s2 = [p for p in points if is_right_of_line(p, max_point, b)]
        
        # Recursively process the two subsets
        find_hull(s1, a, max_point, hull)
        find_hull(s2, max_point, b, hull)
    
    def distance_from_line(p, a, b):
        """Calculate the perpendicular distance from point p to line ab"""
        return abs((b[1]-a[1])*p[0] - (b[0]-a[0])*p[1] + b[0]*a[1] - b[1]*a[0]) / \
               ((b[1]-a[1])**2 + (b[0]-a[0])**2)**0.5
    
    def is_right_of_line(p, a, b):
        """Check if point p is on the right side of line ab"""
        return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) > 0
    
    # Find leftmost and rightmost points
    min_x = min(points, key=lambda p: p[0])
    max_x = max(points, key=lambda p: p[0])
    
    # Initialize hull with min and max points
    hull = set([min_x, max_x])
    
    # Separate points on the right and left of line min_x-max_x
    right = [p for p in points if is_right_of_line(p, min_x, max_x)]
    left = [p for p in points if is_right_of_line(p, max_x, min_x)]
    
    # Build the hull recursively
    find_hull(right, min_x, max_x, hull)
    find_hull(left, max_x, min_x, hull)
    
    return list(hull)
```

### Chan's Algorithm

Chan's algorithm achieves O(n log h) time complexity where h is the number of hull vertices:

```python
def chans_algorithm(points):
    """
    Compute the convex hull of a set of 2D points using Chan's algorithm.
    This is an output-sensitive algorithm with O(n log h) time complexity.
    
    Args:
        points: List of (x, y) tuples representing points in 2D space
        
    Returns:
        List of (x, y) tuples representing vertices of the convex hull
    """
    def graham_scan_subset(subset):
        """Compute convex hull of a subset using Graham scan"""
        # Implementation similar to the full Graham scan above
        pass
    
    def jarvis_step(hulls, start_point, direction, m):
        """
        Find the next hull vertex using a modified Jarvis march
        that queries the precomputed hulls
        """
        best_point = None
        best_angle = -float('inf')
        
        for hull in hulls:
            # Find the point in the hull that maximizes the angle
            # from direction vector
            # This can be done in O(log n) time with binary search
            # on the precomputed hull
            pass
        
        return best_point
    
    n = len(points)
    if n <= 3:
        # Special cases for small point sets
        return points
    
    # Try increasing values of m = 2^(2^t)
    t = 1
    while True:
        m = min(2 ** (2 ** t), n)
        
        # Partition points into subsets of size m
        subsets = [points[i:i+m] for i in range(0, n, m)]
        
        # Compute convex hull of each subset
        hulls = [graham_scan_subset(subset) for subset in subsets]
        
        # Try to compute the overall convex hull using at most m Jarvis steps
        try:
            # Find the leftmost point (guaranteed to be on hull)
            start = min(points, key=lambda p: p[0])
            hull = [start]
            direction = (0, 1)  # Initial direction: upward
            
            # Perform at most m Jarvis steps
            for _ in range(m):
                next_point = jarvis_step(hulls, hull[-1], direction, m)
                if next_point == hull[0]:
                    # Completed the hull
                    return hull
                hull.append(next_point)
                direction = (next_point[0] - hull[-2][0], next_point[1] - hull[-2][1])
                
            # If we get here, m was too small
            t += 1
        except:
            # If an exception occurs (e.g., m was too small), increase t
            t += 1
```

## 🔍 3D and Higher-Dimensional Convex Hulls

### Gift Wrapping in 3D

```python
def gift_wrapping_3d_sketch():
    """
    Conceptual sketch of the Gift Wrapping algorithm for 3D convex hull.
    This is not a complete implementation but outlines the approach.
    """
    # 1. Start with a point guaranteed to be on the hull (e.g., extreme in x)
    # 2. Find two more points to form an initial triangular face
    # 3. Maintain a queue of edges to be processed (initially the edges of first face)
    # 4. For each edge:
    #    a. Find the point that makes the maximum angle with the current face
    #    b. Form a new face with this point and the current edge
    #    c. Add the new edges to the queue if they haven't been processed
    # 5. Continue until all edges have been processed
    pass
```

### Incremental Algorithm for 3D Hull

```python
def incremental_3d_hull_sketch():
    """
    Conceptual sketch of an incremental algorithm for 3D convex hull.
    This is not a complete implementation but outlines the approach.
    """
    # 1. Start with a simplex (tetrahedron) of 4 points
    # 2. For each remaining point:
    #    a. If inside the current hull, skip it
    #    b. If outside, determine which faces are visible from it
    #    c. Remove visible faces and create new faces connecting the point
    #       to the boundary of the visible region
    # 3. Continue until all points are processed
    pass
```

## ⚙️ Complexity Analysis

| Algorithm | Average Time | Worst Time | Space |
|-----------|--------------|------------|-------|
| Graham Scan | O(n log n) | O(n log n) | O(n) |
| Jarvis March | O(nh) | O(n²) | O(n) |
| QuickHull | O(n log n) | O(n²) | O(n) |
| Chan's Algorithm | O(n log h) | O(n log h) | O(n) |
| Gift Wrapping 3D | O(n²) | O(n²) | O(n) |
| Incremental 3D | O(n log n) | O(n²) | O(n) |

Where n is the number of points and h is the number of hull vertices.

## 🧩 Applications

1. **Collision Detection**: Finding the minimum bounding shapes for objects
2. **Pattern Recognition**: Shape analysis and feature extraction
3. **Geographical Information Systems**: Boundary detection and area calculations
4. **Robot Path Planning**: Avoiding obstacles defined by convex hulls
5. **Image Processing**: Finding boundaries of objects in images
6. **Data Visualization**: Creating hulls around clusters of data points
7. **Mesh Generation**: Starting point for triangulation algorithms

## 📝 Practice Problems

1. **Smallest Enclosing Circle**: Find the smallest circle containing all points
2. **Farthest Point Pairs**: Find the two points with maximum distance
3. **Convex Hull Merging**: Efficiently merge two convex hulls
4. **Dynamic Convex Hull**: Maintain a convex hull as points are added or removed
5. **Approximation**: Compute an approximate convex hull with fewer vertices

## 🌟 Pro Tips

- For 2D hulls, Graham scan is generally the most efficient for most cases
- If you expect few points on the hull, consider Jarvis march or Chan's algorithm
- For numerical stability, consider sorting points lexicographically instead of by angle
- In 3D and higher dimensions, the implementation complexity increases significantly
- Preprocessing points (e.g., removing duplicates) can improve performance
- For very large point sets, consider parallelizing the computation
- Libraries like CGAL, PCL, or scipy.spatial provide robust implementations
- When working with integer coordinates, be careful with potential overflows

## 🔗 Related Algorithms

- [Line Intersection](line-intersection.md)
- [Polygon Area](polygon-area.md)
- [Closest Pair of Points](closest-pair.md)
- [Triangulation](triangulation.md)
- [Voronoi Diagrams](voronoi.md)
