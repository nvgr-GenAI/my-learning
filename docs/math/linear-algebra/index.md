# Part 2: Linear Algebra

Linear algebra is the single most important mathematical subject for machine learning, deep learning, and quantum computing. Every neural network layer is a matrix multiplication followed by a nonlinearity. Every dataset is a matrix. Every quantum state is a vector in a complex Hilbert space. PCA, SVD, attention mechanisms, graph neural networks, and normalizing flows are all fundamentally linear algebra.

If you learn only one part of this mathematics section, make it this one.

---

## Chapters

| Chapter | Topic | Key Results |
|---------|-------|-------------|
| 2.1 | [Vectors](vectors.md) | Vector operations, norms, inner products, geometric interpretation |
| 2.2 | [Matrices](matrices.md) | Matrix operations, types, transpose, inverse, rank |
| 2.3 | [Systems of Equations](systems-of-equations.md) | Gaussian elimination, row echelon form, solution existence |
| 2.4 | [Vector Spaces](vector-spaces.md) | Subspaces, basis, dimension, span, direct sum |
| 2.5 | [Linear Transformations](linear-transformations.md) | Kernel, image, rank-nullity theorem, change of basis |
| 2.6 | [Determinants](determinants.md) | Properties, cofactor expansion, geometric meaning, Cramer's rule |
| 2.7 | [Eigenvalues & Eigenvectors](eigenvalues.md) | Characteristic polynomial, diagonalization, spectral theorem |
| 2.8 | [Inner Product Spaces](inner-product-spaces.md) | Orthogonality, Gram-Schmidt, projections, least squares |
| 2.9 | [Matrix Decompositions](matrix-decompositions.md) | SVD, LU, QR, Cholesky, connection to PCA |
| 2.10 | [Tensors](tensors.md) | Tensor products, multilinear maps, Einstein notation |

## Prerequisites

- [Part 1: Foundations](../foundations/index.md) — sets, number systems, functions

## The Big Picture

```
                    ┌─────────────────┐
                    │     Vectors     │ ← data lives here
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Matrices     │ ← transformations live here
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐ ┌──────────────┐ ┌────────────┐
    │   Vector    │ │   Linear     │ │ Systems of │
    │   Spaces    │ │  Transforms  │ │ Equations  │
    └──────┬──────┘ └──────┬───────┘ └────────────┘
           │               │
    ┌──────▼──────┐ ┌──────▼───────┐
    │   Inner     │ │ Determinants │
    │  Products   │ └──────┬───────┘
    └──────┬──────┘        │
           │        ┌──────▼───────┐
           └───────→│ Eigenvalues  │ ← the crown jewel
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────┐
    │    Matrix    │ │  Tensors │ │ Spectral │
    │  Decomp.    │ │          │ │  Theory  │
    │ (SVD, PCA)  │ │          │ │          │
    └──────────────┘ └──────────┘ └──────────┘
```
