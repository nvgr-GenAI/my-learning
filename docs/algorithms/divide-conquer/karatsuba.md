# Karatsuba Multiplication 🔢

Fast multiplication algorithm for large integers using divide-and-conquer.

## 🎯 Problem Statement

Multiply two large integers more efficiently than the standard O(n²) algorithm.

**Input**: Two n-digit integers x and y
**Output**: Product x × y

## 🧠 Algorithm Approach

### Standard Multiplication
Standard multiplication requires O(n²) single-digit multiplications.

### Karatsuba's Insight
Split numbers and use only 3 multiplications instead of 4.

## 📝 Implementation

```python
def karatsuba_multiply(x: int, y: int) -> int:
    """
    Karatsuba multiplication algorithm
    
    Args:
        x, y: Integers to multiply
        
    Returns:
        Product of x and y
    """
    # Base case for small numbers
    if x < 10 or y < 10:
        return x * y
    
    # Calculate the number of digits
    n = max(len(str(x)), len(str(y)))
    half = n // 2
    
    # Split the numbers
    high_x = x // (10 ** half)
    low_x = x % (10 ** half)
    high_y = y // (10 ** half)
    low_y = y % (10 ** half)
    
    # Three recursive calls (instead of four)
    z0 = karatsuba_multiply(low_x, low_y)           # low parts
    z1 = karatsuba_multiply((low_x + high_x), (low_y + high_y))  # sum parts
    z2 = karatsuba_multiply(high_x, high_y)         # high parts
    
    # Combine results
    return (z2 * (10 ** (2 * half))) + ((z1 - z2 - z0) * (10 ** half)) + z0

def karatsuba_multiply_optimized(x: int, y: int, threshold: int = 1000) -> int:
    """
    Optimized Karatsuba with threshold for standard multiplication
    
    Args:
        x, y: Integers to multiply
        threshold: Switch to standard multiplication below this size
        
    Returns:
        Product of x and y
    """
    # Use standard multiplication for small numbers
    if x < threshold or y < threshold:
        return x * y
    
    return karatsuba_multiply(x, y)

def standard_multiply(x: int, y: int) -> int:
    """Standard multiplication for comparison"""
    return x * y

# String-based implementation for very large numbers
def karatsuba_string(x_str: str, y_str: str) -> str:
    """
    Karatsuba multiplication for very large numbers represented as strings
    
    Args:
        x_str, y_str: String representations of integers
        
    Returns:
        String representation of the product
    """
    # Remove leading zeros and handle signs
    x_str = x_str.lstrip('0') or '0'
    y_str = y_str.lstrip('0') or '0'
    
    # Base case
    if len(x_str) == 1 and len(y_str) == 1:
        return str(int(x_str) * int(y_str))
    
    # Make lengths equal by padding with zeros
    max_len = max(len(x_str), len(y_str))
    x_str = x_str.zfill(max_len)
    y_str = y_str.zfill(max_len)
    
    # Base case for small numbers
    if max_len <= 4:
        return str(int(x_str) * int(y_str))
    
    half = max_len // 2
    
    # Split the numbers
    x_high = x_str[:len(x_str) - half]
    x_low = x_str[len(x_str) - half:]
    y_high = y_str[:len(y_str) - half]
    y_low = y_str[len(y_str) - half:]
    
    # Three recursive multiplications
    z0 = karatsuba_string(x_low, y_low)
    z1 = karatsuba_string(add_strings(x_high, x_low), add_strings(y_high, y_low))
    z2 = karatsuba_string(x_high, y_high)
    
    # Combine results: z2 * 10^(2*half) + (z1 - z2 - z0) * 10^half + z0
    result = add_strings(
        add_strings(multiply_by_power_of_10(z2, 2 * half),
                   multiply_by_power_of_10(subtract_strings(subtract_strings(z1, z2), z0), half)),
        z0
    )
    
    return result

def add_strings(num1: str, num2: str) -> str:
    """Add two number strings"""
    i, j = len(num1) - 1, len(num2) - 1
    carry = 0
    result = []
    
    while i >= 0 or j >= 0 or carry:
        digit1 = int(num1[i]) if i >= 0 else 0
        digit2 = int(num2[j]) if j >= 0 else 0
        
        total = digit1 + digit2 + carry
        result.append(str(total % 10))
        carry = total // 10
        
        i -= 1
        j -= 1
    
    return ''.join(reversed(result))

def subtract_strings(num1: str, num2: str) -> str:
    """Subtract two number strings (assuming num1 >= num2)"""
    i, j = len(num1) - 1, len(num2) - 1
    borrow = 0
    result = []
    
    while i >= 0 or j >= 0:
        digit1 = int(num1[i]) if i >= 0 else 0
        digit2 = int(num2[j]) if j >= 0 else 0
        
        digit1 -= borrow
        if digit1 < digit2:
            digit1 += 10
            borrow = 1
        else:
            borrow = 0
            
        result.append(str(digit1 - digit2))
        i -= 1
        j -= 1
    
    # Remove leading zeros
    result_str = ''.join(reversed(result)).lstrip('0')
    return result_str if result_str else '0'

def multiply_by_power_of_10(num: str, power: int) -> str:
    """Multiply number string by 10^power"""
    if num == '0':
        return '0'
    return num + '0' * power

# Example usage and benchmarking
if __name__ == "__main__":
    import time
    import random
    
    # Test with small numbers
    x, y = 1234, 5678
    
    standard_result = standard_multiply(x, y)
    karatsuba_result = karatsuba_multiply(x, y)
    
    print(f"Standard: {x} × {y} = {standard_result}")
    print(f"Karatsuba: {x} × {y} = {karatsuba_result}")
    print(f"Results match: {standard_result == karatsuba_result}")
    
    # Benchmark with large numbers
    def benchmark_multiplication(digits: int):
        """Benchmark different multiplication approaches"""
        # Generate random numbers with specified digits
        x = random.randint(10**(digits-1), 10**digits - 1)
        y = random.randint(10**(digits-1), 10**digits - 1)
        
        # Standard multiplication
        start = time.time()
        standard_result = x * y  # Python's built-in
        standard_time = time.time() - start
        
        # Karatsuba multiplication
        start = time.time()
        karatsuba_result = karatsuba_multiply(x, y)
        karatsuba_time = time.time() - start
        
        print(f"\\nDigits: {digits}")
        print(f"Standard time: {standard_time:.6f}s")
        print(f"Karatsuba time: {karatsuba_time:.6f}s")
        print(f"Results match: {standard_result == karatsuba_result}")
        
        if karatsuba_time > 0:
            print(f"Speedup: {standard_time/karatsuba_time:.2f}x")
    
    # Benchmark different sizes
    for digits in [10, 100, 1000]:
        benchmark_multiplication(digits)
```

## ⚡ Time Complexity Analysis

### Karatsuba Algorithm
- **Time Complexity**: O(n^log₂3) ≈ O(n^1.585)
- **Space Complexity**: O(log n) for recursion stack

### Comparison
| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Standard | O(n²) | O(1) |
| Karatsuba | O(n^1.585) | O(log n) |
| FFT-based | O(n log n) | O(n) |

## 🔄 Step-by-Step Example

```text
Example: 1234 × 5678

Step 1: Split numbers (n=4, half=2)
x = 1234 → high_x = 12, low_x = 34
y = 5678 → high_y = 56, low_y = 78

Step 2: Three recursive multiplications
z0 = 34 × 78 = 2652
z1 = (12+34) × (56+78) = 46 × 134 = 6164
z2 = 12 × 56 = 672

Step 3: Combine results
result = z2 × 10^4 + (z1 - z2 - z0) × 10^2 + z0
       = 672 × 10000 + (6164 - 672 - 2652) × 100 + 2652
       = 6720000 + 284000 + 2652
       = 7006652

Verification: 1234 × 5678 = 7006652 ✓
```

## 🎯 Key Insights

1. **Divide Strategy**: Split numbers into two halves
2. **Clever Algebra**: Use (a+b)(c+d) = ac + ad + bc + bd = ac + bd + (a+b)(c+d) - ac - bd
3. **Recursive Structure**: Apply same strategy to sub-problems
4. **Practical Threshold**: Use standard multiplication for small numbers

## 📊 Recursive Relation

```text
T(n) = 3T(n/2) + O(n)

Using Master Theorem:
a = 3, b = 2, f(n) = O(n)
log_b(a) = log_2(3) ≈ 1.585

Since f(n) = O(n^1.585-ε) for some ε > 0,
T(n) = O(n^log_2(3)) = O(n^1.585)
```

## 🔧 Optimizations

1. **Threshold Optimization**: Switch to standard multiplication for small numbers
2. **Memory Management**: Minimize string operations and copying
3. **Base Cases**: Optimize for single-digit and two-digit multiplication
4. **Padding Strategy**: Efficient handling of unequal length numbers

## 💡 Applications

- **Cryptography**: RSA encryption with large prime numbers
- **Arbitrary Precision Arithmetic**: Libraries like GMP, MPIR
- **Computer Algebra Systems**: Symbolic computation
- **Digital Signal Processing**: Fast convolution algorithms

## 🚀 Advanced Variants

1. **Toom-Cook Algorithm**: Generalization using more evaluation points
2. **Schönhage-Strassen**: FFT-based multiplication for very large numbers
3. **Parallel Karatsuba**: Distributed implementation
4. **Balanced Multiplication**: Handling numbers of different sizes

---

*Karatsuba multiplication demonstrates how mathematical insight can reduce the complexity of fundamental arithmetic operations.*
