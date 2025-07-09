# Line Intersection

## 🎯 Overview

Line intersection is a fundamental problem in computational geometry that involves determining whether and where two or more lines meet. This concept has broad applications in computer graphics, robotics, geographic information systems (GIS), and many other fields that require spatial analysis and visualization.

## 📋 Core Concepts

### Types of Line Intersections

1. **Line-Line Intersection**: Determining where two infinite lines meet
2. **Line Segment Intersection**: Finding where two bounded line segments intersect
3. **Ray-Line Intersection**: Computing where a ray (semi-infinite line) meets a line
4. **Multiple Line Intersections**: Identifying all intersection points among multiple lines

### Coordinate Systems

Most line intersection algorithms work in:
- 2D Cartesian coordinates (x,y)
- 3D space (x,y,z)
- Homogeneous coordinates for projective geometry

### Representation of Lines

Lines can be represented in different ways:
- **Parametric Form**: p(t) = p₀ + t · v (point + direction)
- **Implicit Form**: ax + by + c = 0
- **Slope-Intercept Form**: y = mx + b
- **Two-Point Form**: Through points p₁ and p₂

## ⚙️ Algorithm Implementations

### 2D Line-Line Intersection

```python
def line_line_intersection(line1, line2):
    """
    Find the intersection of two lines in 2D.
    
    Args:
        line1: Tuple ((x1, y1), (x2, y2)) representing first line through two points
        line2: Tuple ((x3, y3), (x4, y4)) representing second line through two points
        
    Returns:
        Intersection point (x, y) or None if lines are parallel
    """
    # Extract points
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    # Calculate determinants
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Check if lines are parallel
    if abs(denom) < 1e-10:
        return None
    
    # Calculate intersection point
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    # ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    return (x, y)
```

### 2D Line Segment Intersection

```python
def segment_intersection(segment1, segment2):
    """
    Find the intersection of two line segments in 2D.
    
    Args:
        segment1: Tuple ((x1, y1), (x2, y2)) representing first segment
        segment2: Tuple ((x3, y3), (x4, y4)) representing second segment
        
    Returns:
        Intersection point (x, y) or None if segments don't intersect
    """
    # Extract points
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2
    
    # Calculate determinants
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    # Check if segments are parallel
    if abs(denom) < 1e-10:
        # Check for collinearity and overlap
        if abs((y1 - y3) * (x2 - x1) - (y2 - y1) * (x1 - x3)) < 1e-10:
            # The segments are collinear
            # Check for overlap (simplified check)
            if min(x1, x2) <= max(x3, x4) and min(x3, x4) <= max(x1, x2) and \
               min(y1, y2) <= max(y3, y4) and min(y3, y4) <= max(y1, y2):
                return "Segments overlap"
        return None
    
    # Calculate parameters
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    
    # Check if intersection is within both segments
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    
    return None
```

### Efficient Line Segment Intersection Check

```python
def segments_intersect(segment1, segment2):
    """
    Check if two line segments intersect (without calculating the intersection point).
    Uses orientation-based approach for efficiency.
    
    Args:
        segment1: Tuple ((x1, y1), (x2, y2)) representing first segment
        segment2: Tuple ((x3, y3), (x4, y4)) representing second segment
        
    Returns:
        True if segments intersect, False otherwise
    """
    (p1, p2), (p3, p4) = segment1, segment2
    
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
            return 0  # collinear
        return 1 if val > 0 else 2  # clockwise or counterclockwise
    
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special Cases for collinearity
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True
    
    return False

def on_segment(p, q, r):
    """Check if point q lies on line segment 'pr'"""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
```

### Intersection of a Ray with a Line Segment

```python
def ray_segment_intersection(ray_origin, ray_dir, segment):
    """
    Find the intersection of a ray with a line segment.
    
    Args:
        ray_origin: (x, y) origin point of the ray
        ray_dir: (dx, dy) direction vector of the ray
        segment: ((x1, y1), (x2, y2)) line segment
        
    Returns:
        Intersection point (x, y) and distance from ray origin, or None if no intersection
    """
    (x1, y1), (x2, y2) = segment
    (ox, oy) = ray_origin
    (dx, dy) = ray_dir
    
    # Convert to parametric form
    # Ray: p(t) = origin + t * direction
    # Segment: q(s) = p1 + s * (p2 - p1)
    
    # Calculate determinant
    denom = dx * (y2 - y1) - dy * (x2 - x1)
    
    # Check if ray and segment are parallel
    if abs(denom) < 1e-10:
        return None
    
    # Calculate parameters
    t = ((y1 - oy) * (x2 - x1) - (x1 - ox) * (y2 - y1)) / denom
    s = (ox + t * dx - x1) / (x2 - x1) if abs(x2 - x1) > 1e-10 else \
        (oy + t * dy - y1) / (y2 - y1)
    
    # Check if intersection is within the segment and in ray's direction
    if t >= 0 and 0 <= s <= 1:
        x = ox + t * dx
        y = oy + t * dy
        return ((x, y), t)
    
    return None
```

## 🔍 Computational Geometry Algorithms for Multiple Intersections

### Bentley-Ottmann Algorithm for Line Segment Intersections

The Bentley-Ottmann algorithm is an efficient sweep line algorithm for finding all intersections among a set of line segments. It has a time complexity of O((n + k) log n), where n is the number of segments and k is the number of intersections.

```python
def bentley_ottmann_outline():
    """
    Outline of the Bentley-Ottmann algorithm for finding all intersections
    among a set of line segments.
    
    Note: This is a simplified conceptual outline, not a complete implementation.
    """
    # 1. Create event queue with segment endpoints
    # 2. Initialize empty status structure (e.g., balanced binary tree)
    # 3. Process events from left to right
    
    # Pseudocode:
    """
    1. Sort all segment endpoints by x-coordinate
    2. For each event point from left to right:
       a. If it's a left endpoint:
          - Add segment to status structure
          - Check for intersections with segments above and below
       b. If it's a right endpoint:
          - Remove segment from status structure
          - Check if segments that become adjacent intersect
       c. If it's an intersection point:
          - Swap positions of intersecting segments in status
          - Check for new intersections with adjacent segments
    """
    pass
```

### Line Arrangement Data Structure

```python
class LineArrangement:
    """
    Represents an arrangement of lines in 2D.
    
    An arrangement is a partition of the plane into vertices, edges, and faces
    induced by a collection of lines.
    """
    def __init__(self, lines):
        """
        Initialize with a list of lines in ax + by + c = 0 form.
        Each line is represented as (a, b, c).
        """
        self.lines = lines
        self.vertices = []  # Intersection points
        self.edges = []     # Line segments between vertices
        self.faces = []     # Regions bounded by edges
        
        # Compute the arrangement
        self._compute_arrangement()
    
    def _compute_arrangement(self):
        """Compute the arrangement of lines"""
        # 1. Find all pairwise intersections
        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                intersection = self._line_line_intersection(self.lines[i], self.lines[j])
                if intersection:
                    self.vertices.append(intersection)
        
        # Sort vertices and remove duplicates
        # Build edges and faces
        # Note: This is just a placeholder for the algorithm outline
        # A complete implementation requires more sophisticated data structures
        
        # The DCEL (Doubly Connected Edge List) is often used for arrangements
    
    def _line_line_intersection(self, line1, line2):
        """
        Find intersection between two lines in ax + by + c = 0 form.
        
        Args:
            line1: Tuple (a1, b1, c1)
            line2: Tuple (a2, b2, c2)
            
        Returns:
            Intersection point (x, y) or None if lines are parallel
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        # Calculate determinant
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-10:
            return None  # Lines are parallel
        
        # Calculate intersection using Cramer's rule
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        
        return (x, y)
```

## ⚙️ Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Line-Line Intersection | O(1) | O(1) |
| Line Segment Intersection | O(1) | O(1) |
| Ray-Line Intersection | O(1) | O(1) |
| Bentley-Ottmann | O((n + k) log n) | O(n) |
| Naive All-Intersections | O(n²) | O(k) |
| Line Arrangement Construction | O(n²) | O(n²) |

Where n is the number of lines/segments and k is the number of intersections.

## 🧩 Applications

1. **Computer Graphics**: Clipping, ray tracing, collision detection
2. **GIS**: Map overlay operations, spatial analysis
3. **Robotics**: Path planning, obstacle avoidance
4. **Computer Vision**: Feature detection, image segmentation
5. **Computational Geometry**: Triangulation, Voronoi diagrams
6. **CAD/CAM**: Design verification, tool path generation
7. **Physics Simulations**: Light ray physics, particle interactions

## 📝 Practice Problems

1. **Polygon Intersection**: Determine if two polygons intersect
2. **Ray Casting**: Implement the point-in-polygon test using ray casting
3. **Line Segment Visibility**: Find visible segments from a viewpoint
4. **Intersection Area**: Calculate the area of intersection between two polygons
5. **Closest Intersection**: Find the closest intersection point to a given point

## 🌟 Pro Tips

- Always handle special cases (parallel lines, collinear points)
- Use robust geometric predicates to avoid numerical issues
- For many segments, use sweep line algorithms rather than checking all pairs
- Consider using integer arithmetic or fixed-point representations for better numerical stability
- When working with floating point, use appropriate epsilon values for equality tests
- Precompute and cache results when processing multiple queries
- Consider specialized libraries like CGAL for production code
- For 3D applications, consider projective geometry and homogeneous coordinates

## 🔗 Related Algorithms

- [Convex Hull](convex-hull.md)
- [Polygon Area](polygon-area.md)
- [Vectors and Matrices](vectors-matrices.md)
- [Closest Pair of Points](closest-pair.md)
- [Geometric Primitives](geometric-primitives.md)
