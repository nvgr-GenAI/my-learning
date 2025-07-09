# Fast Fourier Transform (FFT) 🌊

Efficient algorithm for computing the Discrete Fourier Transform using divide-and-conquer.

## 🎯 Problem Statement

Compute the Discrete Fourier Transform (DFT) of a sequence in O(n log n) time instead of O(n²).

**Input**: Complex sequence x[0], x[1], ..., x[n-1]
**Output**: DFT sequence X[0], X[1], ..., X[n-1]

## 🧠 Algorithm Approach

### DFT Definition
```
X[k] = Σ(n=0 to N-1) x[n] * e^(-2πikn/N)
```

### Key Insight
- Split DFT into even and odd indexed elements
- Recursively compute smaller DFTs
- Combine results using complex arithmetic

## 📝 Implementation

```python
import numpy as np
import cmath
from typing import List, Union

def fft_recursive(x: List[complex]) -> List[complex]:
    """
    Recursive FFT implementation (Cooley-Tukey algorithm)
    
    Args:
        x: Input sequence (length must be power of 2)
        
    Returns:
        DFT of input sequence
    """
    n = len(x)
    
    # Base case
    if n <= 1:
        return x
    
    # Divide
    even = fft_recursive([x[i] for i in range(0, n, 2)])
    odd = fft_recursive([x[i] for i in range(1, n, 2)])
    
    # Combine
    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    
    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]

def ifft_recursive(X: List[complex]) -> List[complex]:
    """
    Inverse FFT using conjugate method
    
    Args:
        X: Frequency domain sequence
        
    Returns:
        Time domain sequence
    """
    n = len(X)
    
    # Conjugate input
    X_conj = [x.conjugate() for x in X]
    
    # Apply FFT
    x_conj = fft_recursive(X_conj)
    
    # Conjugate and scale output
    return [x.conjugate() / n for x in x_conj]

def fft_iterative(x: List[complex]) -> List[complex]:
    """
    Iterative FFT implementation (bit-reversal)
    
    Args:
        x: Input sequence (length must be power of 2)
        
    Returns:
        DFT of input sequence
    """
    n = len(x)
    
    # Bit-reversal permutation
    result = [0] * n
    for i in range(n):
        result[bit_reverse(i, n)] = x[i]
    
    # Iterative FFT
    length = 2
    while length <= n:
        # Twiddle factor
        w = cmath.exp(-2j * cmath.pi / length)
        
        for i in range(0, n, length):
            w_n = 1
            for j in range(length // 2):
                u = result[i + j]
                v = result[i + j + length // 2] * w_n
                result[i + j] = u + v
                result[i + j + length // 2] = u - v
                w_n *= w
        
        length *= 2
    
    return result

def bit_reverse(num: int, n: int) -> int:
    """
    Reverse bits of number for given length
    
    Args:
        num: Number to reverse
        n: Total length (must be power of 2)
        
    Returns:
        Bit-reversed number
    """
    result = 0
    bits = int(np.log2(n))
    
    for _ in range(bits):
        result = (result << 1) | (num & 1)
        num >>= 1
    
    return result

# Practical FFT class with optimizations
class FFTProcessor:
    def __init__(self):
        self.twiddle_factors = {}
    
    def _get_twiddle_factors(self, n: int) -> List[complex]:
        """Cache twiddle factors for efficiency"""
        if n not in self.twiddle_factors:
            self.twiddle_factors[n] = [
                cmath.exp(-2j * cmath.pi * k / n) for k in range(n // 2)
            ]
        return self.twiddle_factors[n]
    
    def fft(self, x: List[Union[float, complex]]) -> List[complex]:
        """
        Optimized FFT with padding and caching
        
        Args:
            x: Input sequence
            
        Returns:
            DFT of input sequence
        """
        # Convert to complex and pad to power of 2
        x_complex = [complex(val) for val in x]
        n = len(x_complex)
        
        # Pad to next power of 2
        next_power = 1
        while next_power < n:
            next_power *= 2
        
        x_padded = x_complex + [0] * (next_power - n)
        
        # Compute FFT
        return self._fft_optimized(x_padded)
    
    def _fft_optimized(self, x: List[complex]) -> List[complex]:
        """Optimized FFT implementation"""
        n = len(x)
        
        if n <= 1:
            return x
        
        # Divide
        even = self._fft_optimized([x[i] for i in range(0, n, 2)])
        odd = self._fft_optimized([x[i] for i in range(1, n, 2)])
        
        # Get twiddle factors
        twiddle = self._get_twiddle_factors(n)
        
        # Combine
        T = [twiddle[k] * odd[k] for k in range(n // 2)]
        
        return [even[k] + T[k] for k in range(n // 2)] + \
               [even[k] - T[k] for k in range(n // 2)]
    
    def ifft(self, X: List[complex]) -> List[complex]:
        """
        Inverse FFT
        
        Args:
            X: Frequency domain sequence
            
        Returns:
            Time domain sequence
        """
        n = len(X)
        
        # Conjugate input
        X_conj = [x.conjugate() for x in X]
        
        # Apply FFT
        x_conj = self._fft_optimized(X_conj)
        
        # Conjugate and scale output
        return [x.conjugate() / n for x in x_conj]

# Convolution using FFT
def convolution_fft(a: List[float], b: List[float]) -> List[float]:
    """
    Compute convolution using FFT
    
    Args:
        a, b: Input sequences
        
    Returns:
        Convolution of a and b
    """
    # Pad to avoid circular convolution
    n = len(a) + len(b) - 1
    next_power = 1
    while next_power < n:
        next_power *= 2
    
    # Pad sequences
    a_padded = a + [0] * (next_power - len(a))
    b_padded = b + [0] * (next_power - len(b))
    
    # FFT of both sequences
    processor = FFTProcessor()
    A = processor.fft(a_padded)
    B = processor.fft(b_padded)
    
    # Pointwise multiplication
    C = [A[i] * B[i] for i in range(len(A))]
    
    # Inverse FFT
    c_complex = processor.ifft(C)
    
    # Convert to real and trim
    c_real = [c.real for c in c_complex]
    return c_real[:n]

# Polynomial multiplication using FFT
def polynomial_multiply(p1: List[float], p2: List[float]) -> List[float]:
    """
    Multiply two polynomials using FFT
    
    Args:
        p1, p2: Polynomial coefficients (lowest degree first)
        
    Returns:
        Product polynomial coefficients
    """
    if not p1 or not p2:
        return []
    
    return convolution_fft(p1, p2)

# Example usage and applications
if __name__ == "__main__":
    # Test basic FFT
    x = [1, 2, 3, 4, 0, 0, 0, 0]  # Pad to power of 2
    
    # Recursive FFT
    X_recursive = fft_recursive([complex(val) for val in x])
    print("Recursive FFT result:")
    for i, val in enumerate(X_recursive):
        print(f"X[{i}] = {val:.3f}")
    
    # Iterative FFT
    X_iterative = fft_iterative([complex(val) for val in x])
    print("\\nIterative FFT result:")
    for i, val in enumerate(X_iterative):
        print(f"X[{i}] = {val:.3f}")
    
    # Verify with NumPy
    X_numpy = np.fft.fft(x)
    print("\\nNumPy FFT result:")
    for i, val in enumerate(X_numpy):
        print(f"X[{i}] = {val:.3f}")
    
    # Test inverse FFT
    x_recovered = ifft_recursive(X_recursive)
    print("\\nInverse FFT result:")
    for i, val in enumerate(x_recovered):
        print(f"x[{i}] = {val:.3f}")
    
    # Test convolution
    a = [1, 2, 3]
    b = [4, 5, 6]
    conv_result = convolution_fft(a, b)
    print(f"\\nConvolution of {a} and {b}:")
    print(f"Result: {[round(x, 3) for x in conv_result]}")
    
    # Test polynomial multiplication
    p1 = [1, 2, 3]  # 1 + 2x + 3x^2
    p2 = [4, 5]     # 4 + 5x
    poly_result = polynomial_multiply(p1, p2)
    print(f"\\nPolynomial multiplication:")
    print(f"({p1}) × ({p2}) = {[round(x, 3) for x in poly_result]}")
    
    # Benchmark comparison
    def benchmark_fft(size: int):
        """Benchmark FFT implementations"""
        import time
        
        # Generate random data
        x = [complex(np.random.random(), np.random.random()) for _ in range(size)]
        
        # Recursive FFT
        start = time.time()
        X_recursive = fft_recursive(x)
        recursive_time = time.time() - start
        
        # NumPy FFT
        start = time.time()
        X_numpy = np.fft.fft([complex(val) for val in x])
        numpy_time = time.time() - start
        
        print(f"\\nSize: {size}")
        print(f"Recursive FFT: {recursive_time:.6f}s")
        print(f"NumPy FFT: {numpy_time:.6f}s")
        print(f"Speedup: {recursive_time/numpy_time:.2f}x")
    
    # Benchmark different sizes
    for size in [64, 256, 1024]:
        benchmark_fft(size)
```

## ⚡ Time Complexity Analysis

### FFT Algorithm
- **Time Complexity**: O(n log n)
- **Space Complexity**: O(n) for recursion stack

### Comparison with DFT
| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| DFT (naive) | O(n²) | O(1) |
| FFT | O(n log n) | O(n) |

## 🔄 Step-by-Step Example

```text
Input: x = [1, 2, 3, 4] (n = 4)

Step 1: Divide
Even indices: [1, 3] (x[0], x[2])
Odd indices: [2, 4] (x[1], x[3])

Step 2: Recursive FFT
FFT([1, 3]) = [4, -2]
FFT([2, 4]) = [6, -2]

Step 3: Combine
Twiddle factors: w₀ = 1, w₁ = -i
T[0] = w₀ × 6 = 6
T[1] = w₁ × (-2) = 2i

X[0] = 4 + 6 = 10
X[1] = -2 + 2i
X[2] = 4 - 6 = -2
X[3] = -2 - 2i

Result: [10, -2+2i, -2, -2-2i]
```

## 🎯 Key Insights

1. **Divide-and-Conquer**: Split into even and odd indexed elements
2. **Twiddle Factors**: Complex exponentials for combining results
3. **Bit-Reversal**: Efficient iterative implementation
4. **Symmetry**: Exploit symmetry in complex exponentials

## 💡 Applications

- **Signal Processing**: Digital filtering, spectral analysis
- **Image Processing**: 2D FFT for frequency domain operations
- **Convolution**: Fast convolution for signal processing
- **Polynomial Multiplication**: Efficient polynomial arithmetic
- **Compression**: JPEG, MP3 compression algorithms
- **Solving PDEs**: Spectral methods for differential equations

## 🚀 Advanced Variants

1. **2D FFT**: Extend to 2D for image processing
2. **Radix-4 FFT**: Use radix-4 for better cache performance
3. **Split-Radix FFT**: Optimal number of operations
4. **Parallel FFT**: Distributed computation for large datasets
5. **Number Theoretic Transform**: FFT over finite fields

## 🔧 Optimizations

1. **Twiddle Factor Caching**: Precompute and reuse twiddle factors
2. **Bit-Reversal Optimization**: Efficient bit-reversal algorithms
3. **Mixed-Radix FFT**: Handle non-power-of-2 lengths
4. **SIMD Instructions**: Vectorized operations for performance

---

*The Fast Fourier Transform revolutionized digital signal processing by reducing DFT complexity from O(n²) to O(n log n).*
