# Geometry

## Basic Concepts

### Points and Distance

```python
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_to(self, other):
        """Euclidean distance between two points"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def manhattan_distance(self, other):
        """Manhattan distance between two points"""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def __str__(self):
        return f"({self.x}, {self.y})"
```

### Line Equations

```python
class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def slope(self):
        """Calculate slope of the line"""
        if self.p2.x == self.p1.x:
            return float('inf')  # Vertical line
        return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
    
    def y_intercept(self):
        """Calculate y-intercept (b in y = mx + b)"""
        m = self.slope()
        if m == float('inf'):
            return None  # Vertical line has no y-intercept
        return self.p1.y - m * self.p1.x
    
    def contains_point(self, point):
        """Check if point lies on the line"""
        # Using cross product method to avoid floating point errors
        return self.cross_product(self.p1, self.p2, point) == 0
    
    @staticmethod
    def cross_product(p1, p2, p3):
        """Cross product of vectors (p2-p1) and (p3-p1)"""
        return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
```

## Area Calculations

### Triangle Area

```python
def triangle_area(p1, p2, p3):
    """Calculate area of triangle using cross product"""
    return abs(
        (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2
    )

def triangle_area_heron(a, b, c):
    """Calculate area using Heron's formula"""
    s = (a + b + c) / 2  # Semi-perimeter
    return math.sqrt(s * (s - a) * (s - b) * (s - c))
```

### Polygon Area

```python
def polygon_area(points):
    """Calculate area of polygon using Shoelace formula"""
    n = len(points)
    area = 0
    
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    
    return abs(area) / 2
```

## Geometric Algorithms

### Convex Hull (Graham Scan)

```python
def convex_hull(points):
    """Find convex hull using Graham scan"""
    def cross_product(o, a, b):
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    
    # Sort points lexicographically
    points = sorted(points, key=lambda p: (p.x, p.y))
    
    if len(points) <= 1:
        return points
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Remove last point of each half because it's repeated
    return lower[:-1] + upper[:-1]
```

### Point in Polygon

```python
def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
    x, y = point.x, point.y
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0].x, polygon[0].y
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n].x, polygon[i % n].y
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside
```

### Line Intersection

```python
def line_intersection(line1, line2):
    """Find intersection point of two lines"""
    p1, p2 = line1.p1, line1.p2
    p3, p4 = line2.p1, line2.p2
    
    denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return None
    
    t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom
    
    intersection_x = p1.x + t * (p2.x - p1.x)
    intersection_y = p1.y + t * (p2.y - p1.y)
    
    return Point(intersection_x, intersection_y)
```

## Circle Geometry

### Circle Class

```python
class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def circumference(self):
        return 2 * math.pi * self.radius
    
    def contains_point(self, point):
        return self.center.distance_to(point) <= self.radius
    
    def intersects_circle(self, other):
        distance = self.center.distance_to(other.center)
        return distance <= (self.radius + other.radius)
```

### Closest Pair of Points

```python
def closest_pair(points):
    """Find closest pair of points using divide and conquer"""
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def closest_pair_rec(px, py):
        n = len(px)
        
        if n <= 3:
            # Brute force for small arrays
            min_dist = float('inf')
            pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(px[i], px[j])
                    if d < min_dist:
                        min_dist = d
                        pair = (px[i], px[j])
            return min_dist, pair
        
        # Divide
        mid = n // 2
        midpoint = px[mid]
        
        pyl = [p for p in py if p.x <= midpoint.x]
        pyr = [p for p in py if p.x > midpoint.x]
        
        # Conquer
        dl, pair_l = closest_pair_rec(px[:mid], pyl)
        dr, pair_r = closest_pair_rec(px[mid:], pyr)
        
        # Find minimum of the two halves
        if dl <= dr:
            min_dist, pair = dl, pair_l
        else:
            min_dist, pair = dr, pair_r
        
        # Check points near the dividing line
        strip = [p for p in py if abs(p.x - midpoint.x) < min_dist]
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j].y - strip[i].y) < min_dist:
                d = distance(strip[i], strip[j])
                if d < min_dist:
                    min_dist = d
                    pair = (strip[i], strip[j])
                j += 1
        
        return min_dist, pair
    
    px = sorted(points, key=lambda p: p.x)
    py = sorted(points, key=lambda p: p.y)
    
    return closest_pair_rec(px, py)
```

## Coordinate Transformations

### Rotation

```python
def rotate_point(point, angle, center=None):
    """Rotate point around center by angle (in radians)"""
    if center is None:
        center = Point(0, 0)
    
    # Translate to origin
    x = point.x - center.x
    y = point.y - center.y
    
    # Rotate
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    new_x = x * cos_angle - y * sin_angle
    new_y = x * sin_angle + y * cos_angle
    
    # Translate back
    return Point(new_x + center.x, new_y + center.y)
```

## Practice Problems

- [ ] K Closest Points to Origin
- [ ] Valid Boomerang
- [ ] Largest Triangle Area
- [ ] Number of Boomerangs
- [ ] Convex Polygon
- [ ] Erect the Fence
- [ ] Minimum Area Rectangle
