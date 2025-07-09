# Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) is an efficient algorithm to compute the Discrete Fourier Transform (DFT) of a sequence, or its inverse (IDFT). It has numerous applications in signal processing, data compression, polynomial multiplication, and more.

## Discrete Fourier Transform

The Discrete Fourier Transform (DFT) converts a sequence of N complex numbers (x₀, x₁, ..., xₙ₋₁) into another sequence of complex numbers (X₀, X₁, ..., Xₙ₋₁) by:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i2\pi kn/N}$$

where:
- $X_k$ is the k-th DFT output
- $x_n$ is the n-th input
- $e^{-i2\pi kn/N}$ is a primitive N-th root of unity
- $i$ is the imaginary unit

## Naive DFT Implementation

The naive approach to compute the DFT has O(n²) complexity:

```python
import numpy as np

def naive_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
            
    return X
```

## Fast Fourier Transform Algorithm

The FFT algorithm uses a divide-and-conquer approach to compute the DFT in O(n log n) time. The Cooley-Tukey algorithm is the most common implementation:

```python
import numpy as np

def fft(x):
    N = len(x)
    
    # Base case
    if N == 1:
        return x
    
    # Divide: split into even and odd indices
    even = fft(x[0::2])
    odd = fft(x[1::2])
    
    # Combine
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # First half of the result
    first_half = even + factor[:N//2] * odd
    # Second half of the result
    second_half = even + factor[N//2:] * odd
    
    return np.concatenate([first_half, second_half])
```

For simplicity, the above code assumes that N is a power of 2. A more robust implementation would handle other values of N or pad the input to the nearest power of 2.

## Inverse Fast Fourier Transform

The Inverse Fast Fourier Transform (IFFT) converts frequency domain data back to the time/spatial domain:

```python
def ifft(X):
    N = len(X)
    
    # Take the complex conjugate
    x_conj = np.conj(X)
    
    # Apply the forward FFT
    x = fft(x_conj)
    
    # Take the complex conjugate again and scale
    return np.conj(x) / N
```

## Polynomial Multiplication Using FFT

One of the most common applications of FFT is fast polynomial multiplication:

```python
def polynomial_multiply(A, B):
    """
    Multiply polynomials A and B using FFT.
    A and B are coefficient lists: A = a0 + a1*x + a2*x^2 + ...
    """
    # Get sizes and pad to avoid circular convolution issues
    size_a, size_b = len(A), len(B)
    size = 1
    while size < size_a + size_b - 1:
        size *= 2
    
    # Pad inputs to 'size'
    A_padded = np.pad(A, (0, size - size_a))
    B_padded = np.pad(B, (0, size - size_b))
    
    # Transform both polynomials to frequency domain
    A_freq = fft(A_padded)
    B_freq = fft(B_padded)
    
    # Multiply in the frequency domain (convolution in time domain)
    C_freq = A_freq * B_freq
    
    # Transform back to coefficient representation
    C = ifft(C_freq)
    
    # The result should be real (small imaginary parts due to numerical errors)
    return np.round(np.real(C[:size_a + size_b - 1])).astype(int)
```

## Time and Space Complexity

- **Time Complexity**: O(n log n) for both FFT and IFFT
- **Space Complexity**: O(n) for the recursive implementation, though this can be improved to O(1) additional space with iterative implementations

## Applications

1. **Signal Processing**:
   - Filtering
   - Spectral analysis
   - Audio processing

2. **Image Processing**:
   - Image filtering
   - Feature detection
   - Compression algorithms

3. **Numerical Algorithms**:
   - Fast polynomial multiplication
   - Large integer multiplication
   - Solving partial differential equations

4. **Data Compression**:
   - JPEG, MP3, and other formats use variants of FFT

## Optimizations and Variants

1. **Iterative FFT**: An iterative implementation can reduce the overhead of recursive calls and be more efficient in practice.

2. **Real FFT**: When the input is real (not complex), specialized algorithms can be more efficient.

3. **Multi-dimensional FFT**: For applications like image processing, multi-dimensional FFTs are used.

4. **Number Theoretic Transform (NTT)**: A variant of FFT that works in a finite field, useful for exact integer computations.

## Implementation Notes

1. **Numerical Stability**: FFT algorithms can suffer from numerical stability issues due to floating-point errors. Using double-precision arithmetic helps.

2. **Library Usage**: In practice, it's usually better to use optimized FFT libraries like FFTW, cuFFT, or the implementations in NumPy/SciPy.

3. **Power of 2**: While FFT is most efficient when N is a power of 2, there are algorithms for arbitrary N (e.g., Bluestein's algorithm).

## Practice Problems

1. [SPOJ - POLYMUL](https://www.spoj.com/problems/POLYMUL/) - Polynomial Multiplication
2. [Codeforces - 993E](https://codeforces.com/problemset/problem/993/E) - Nikita and Order Statistics
3. [SPOJ - MAXMATCH](https://www.spoj.com/problems/MAXMATCH/) - Maximum Self-Matching
4. [Codeforces - 827A](https://codeforces.com/problemset/problem/827/A) - String Reconstruction

## Code Example: Iterative FFT (Radix-2)

```python
import numpy as np

def bit_reverse_copy(a):
    n = len(a)
    result = np.zeros(n, dtype=complex)
    
    # Calculate number of bits needed to represent n
    num_bits = (n-1).bit_length()
    
    for i in range(n):
        # Reverse the bits of i
        reversed_i = int(format(i, f'0{num_bits}b')[::-1], 2)
        result[reversed_i] = a[i]
        
    return result

def iterative_fft(x):
    n = len(x)
    
    # Make sure n is a power of 2
    if n & (n-1) != 0:
        raise ValueError("Size must be a power of 2")
    
    # Bit-reverse copy of the array
    a = bit_reverse_copy(x)
    
    # Main FFT computation
    log_n = (n-1).bit_length()
    for s in range(1, log_n + 1):
        m = 1 << s  # 2^s
        omega_m = np.exp(-2j * np.pi / m)
        
        for k in range(0, n, m):
            omega = 1.0
            for j in range(m//2):
                t = omega * a[k + j + m//2]
                u = a[k + j]
                a[k + j] = u + t
                a[k + j + m//2] = u - t
                omega *= omega_m
                
    return a
```

## Related Topics

- [Number Theoretic Transform (NTT)](number-theoretic-transform.md)
- [Convolution](convolution.md)
- [Binary Exponentiation](binary-exponentiation.md)
- [Signal Processing Fundamentals](signal-processing.md)
