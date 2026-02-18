# Graph Theory

Graphs are the universal language for representing relationships. Any time you have entities and connections between them — users and friendships, neurons and synapses, words and co-occurrences, atoms and bonds — you have a graph. Graph theory provides the mathematical framework to analyze these structures, and it has become indispensable in modern ML through knowledge graphs, graph neural networks, and spectral methods. This chapter develops the theory from basic definitions through spectral graph theory to the mathematical foundations of GNNs.

---

## Prerequisites

- [Sets & Logic](../foundations/sets-and-logic.md) — set notation, proof techniques
- [Vectors](../linear-algebra/vectors.md) — norms, inner products
- [Matrices](../linear-algebra/matrices.md) — matrix operations, transpose, symmetry
- [Eigenvalues](../linear-algebra/eigenvalues.md) — eigendecomposition, spectral theorem, Rayleigh quotient
- [Linear Transformations](../linear-algebra/linear-transformations.md) — kernel, image

---

## 1. Basic Definitions

**Definition 7.6.1 (Graph).** A *graph* $G = (V, E)$ consists of a finite set $V$ of *vertices* (or nodes) and a set $E \subseteq \binom{V}{2}$ of *edges*, where $\binom{V}{2}$ denotes the set of all unordered pairs of distinct elements of $V$. We write $n = |V|$ and $m = |E|$.

**Definition 7.6.2 (Directed Graph).** A *directed graph* (digraph) $G = (V, E)$ has $E \subseteq V \times V$, where each edge $(u, v)$ is an ordered pair indicating direction from $u$ to $v$.

**Definition 7.6.3 (Weighted Graph).** A *weighted graph* is a triple $G = (V, E, w)$ where $w : E \to \mathbb{R}$ assigns a real-valued weight to each edge.

```
GRAPH TYPES

  Undirected              Directed               Weighted
  (friendship)            (web links)            (distances)

   A --- B                A ──→ B                A ─3─ B
   |   / |                ↑   ↙ |                |   / |
   |  /  |                |  /  ↓                2  1  5
   | /   |                | /   |                | /   |
   C --- D                C ──→ D                C ─4─ D
```

**Definition 7.6.4 (Degree).** The *degree* $\deg(v)$ of a vertex $v$ in an undirected graph is the number of edges incident to $v$. In a directed graph, the *in-degree* $\deg^-(v)$ counts incoming edges and the *out-degree* $\deg^+(v)$ counts outgoing edges.

**Theorem 7.6.1 (Handshaking Lemma).** In any undirected graph $G = (V, E)$:

$$\sum_{v \in V} \deg(v) = 2|E|$$

*Proof.* Each edge $\{u, v\}$ contributes exactly $1$ to $\deg(u)$ and $1$ to $\deg(v)$, so it contributes $2$ to the total sum. $\square$

**Example 7.6.1:** Consider the graph from the ASCII diagram above (undirected, vertices $\{1,2,3,4\}$, edges $\{1,2\},\{1,3\},\{2,3\},\{2,4\},\{3,4\}$). The degrees are $\deg(1)=2$, $\deg(2)=3$, $\deg(3)=3$, $\deg(4)=2$. Verify:

$$\sum_{v} \deg(v) = 2 + 3 + 3 + 2 = 10 = 2 \times 5 = 2|E| \;\checkmark$$

**Definition 7.6.5 (Walk, Path, Cycle).** A *walk* of length $k$ is a sequence $v_0, e_1, v_1, e_2, \ldots, e_k, v_k$ of alternating vertices and edges. A *path* is a walk with no repeated vertices. A *cycle* is a walk where $v_0 = v_k$ and all other vertices are distinct.

*ML connection:* **Knowledge graphs** (e.g., Freebase, Wikidata) represent entities as vertices and relations as directed, labeled edges — $(h, r, t)$ triples like (Einstein, bornIn, Ulm). Knowledge graph embeddings (TransE, RotatE) learn vector representations such that $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$. **Dependency graphs** in NLP encode syntactic structure: vertices are words, directed edges are grammatical relations.

---

## 2. Graph Representations

### 2.1 Adjacency Matrix

**Definition 7.6.6 (Adjacency Matrix).** The *adjacency matrix* $A \in \{0,1\}^{n \times n}$ of a graph $G$ with vertices $\{v_1, \ldots, v_n\}$ is defined by:

$$A_{ij} = \begin{cases} 1 & \text{if } \{v_i, v_j\} \in E \\ 0 & \text{otherwise} \end{cases}$$

For undirected graphs, $A$ is symmetric ($A = A^T$). For weighted graphs, $A_{ij} = w(v_i, v_j)$.

```
GRAPH TO ADJACENCY MATRIX

        1                          1  2  3  4
       / \                    1 [  0  1  1  0 ]
      /   \          ──→      2 [  1  0  1  1 ]
     2 --- 3                  3 [  1  1  0  1 ]
      \   /                   4 [  0  1  1  0 ]
       \ /
        4                    Symmetric: A = Aᵀ
                             Trace = 0 (no self-loops)
```

**Example 7.6.2:** For a 4-vertex path graph $P_4$: $1 - 2 - 3 - 4$ (edges $\{1,2\}, \{2,3\}, \{3,4\}$), the adjacency matrix is:

$$A = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

Note: $A$ is symmetric ($A = A^T$), has zero diagonal (no self-loops), and each row sums to the degree of that vertex: $\deg(1)=1, \deg(2)=2, \deg(3)=2, \deg(4)=1$.

**Theorem 7.6.2.** Let $A$ be the adjacency matrix of graph $G$. Then $(A^k)_{ij}$ equals the number of walks of length $k$ from $v_i$ to $v_j$.

*Proof.* By induction on $k$. Base case: $(A^1)_{ij} = A_{ij}$ counts walks of length $1$, i.e., direct edges. Inductive step: $(A^{k+1})_{ij} = \sum_\ell (A^k)_{i\ell} A_{\ell j}$. A walk of length $k+1$ from $i$ to $j$ consists of a walk of length $k$ from $i$ to some $\ell$, followed by an edge from $\ell$ to $j$. By induction, there are $(A^k)_{i\ell}$ walks of length $k$ from $i$ to $\ell$, and $A_{\ell j} = 1$ iff $\{\ell, j\}$ is an edge. Summing over all $\ell$ gives the count. $\square$

**Example 7.6.3:** Using the path graph $P_4$ from Example 7.6.2, compute the number of walks of length $2$. We need $A^2$:

$$A^2 = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}^2 = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 0 & 2 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix}$$

Reading off: $(A^2)_{14} = 0$, so there are no walks of length $2$ from vertex $1$ to $4$. But $(A^2)_{13} = 1$: the unique walk is $1 \to 2 \to 3$. The diagonal $(A^2)_{22} = 2$ counts walks $2 \to 1 \to 2$ and $2 \to 3 \to 2$ (returning to start via each neighbor).

### 2.2 Degree Matrix and Laplacian

**Definition 7.6.7 (Degree Matrix).** The *degree matrix* $D \in \mathbb{R}^{n \times n}$ is the diagonal matrix with $D_{ii} = \deg(v_i)$.

**Example 7.6.4:** For the path graph $P_4$ from Example 7.6.2 with degrees $\deg(1)=1, \deg(2)=2, \deg(3)=2, \deg(4)=1$:

$$D = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 2 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Definition 7.6.8 (Graph Laplacian).** The *(combinatorial) graph Laplacian* is:

$$L = D - A$$

The *normalized Laplacian* is:

$$\mathcal{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

**Example 7.6.5:** Continuing with $P_4$, compute $L = D - A$:

$$L = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 2 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} - \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & -1 & 0 & 0 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ 0 & 0 & -1 & 1 \end{pmatrix}$$

Verify: each row sums to $0$, so $L\mathbf{1} = \mathbf{0}$ confirming $\lambda_1 = 0$. The diagonal entries are the degrees; off-diagonal entries are $-1$ where edges exist.

**Definition 7.6.9 (Incidence Matrix).** The *incidence matrix* $B \in \mathbb{R}^{n \times m}$ has rows indexed by vertices and columns by edges. For edge $e_k = \{v_i, v_j\}$ (with $i < j$), set $B_{ik} = 1$, $B_{jk} = -1$, and all other entries to $0$.

**Theorem 7.6.3.** $L = BB^T$.

*Proof.* $(BB^T)_{ij} = \sum_k B_{ik}B_{jk}$. If $i = j$, each edge incident to $v_i$ contributes $(\pm 1)^2 = 1$, giving $\deg(v_i) = D_{ii}$. If $i \neq j$ and $\{v_i, v_j\} \in E$, the corresponding column contributes $(1)(-1) = -1$. If $\{v_i, v_j\} \notin E$, all columns contribute $0$. So $BB^T = D - A = L$. $\square$

*ML connection:* The adjacency matrix $A$ is the standard input format for **graph neural networks**. Sparse representations (adjacency lists as edge index tensors) are used in practice for scalability. The Laplacian $L$ is the foundation of **spectral methods** — its eigendecomposition defines the graph Fourier transform, which underlies spectral GNNs (ChebNet, GCN).

---

## 3. Graph Properties

**Definition 7.6.10 (Connectivity).** A graph $G$ is *connected* if there exists a path between every pair of vertices. The *connected components* of $G$ are the maximal connected subgraphs.

**Theorem 7.6.4.** The number of connected components of $G$ equals the multiplicity of eigenvalue $0$ of the Laplacian $L$.

*Proof sketch.* $L$ is positive semi-definite (since $\mathbf{x}^T L \mathbf{x} = \sum_{\{i,j\} \in E} (x_i - x_j)^2 \geq 0$). The null space of $L$ is spanned by indicator vectors of connected components: if $G$ has $k$ components, then each component's indicator vector $\mathbf{1}_{C_i}$ satisfies $L\mathbf{1}_{C_i} = \mathbf{0}$, giving $\dim(\ker L) = k$. $\square$

**Example 7.6.6:** Consider a disconnected graph on $\{1,2,3,4\}$ with edges $\{1,2\}$ and $\{3,4\}$ only (two separate edges). Its Laplacian is:

$$L = \begin{pmatrix} 1 & -1 & 0 & 0 \\ -1 & 1 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 0 & 0 & -1 & 1 \end{pmatrix}$$

The eigenvalues are $\lambda = 0, 0, 2, 2$. The eigenvalue $0$ has multiplicity **2**, confirming the graph has **2 connected components** $(\{1,2\}$ and $\{3,4\})$. Contrast with the connected path $P_4$ from Example 7.6.5, whose eigenvalues are $\lambda \approx 0, 0.586, 2, 3.414$ — the eigenvalue $0$ has multiplicity **1**, confirming connectivity.

**Definition 7.6.11 (Bipartite Graph).** A graph $G = (V, E)$ is *bipartite* if $V$ can be partitioned into two disjoint sets $U, W$ such that every edge connects a vertex in $U$ to one in $W$.

**Theorem 7.6.5.** A graph is bipartite if and only if it contains no odd-length cycles.

**Definition 7.6.12 (Tree).** A *tree* is a connected acyclic graph. A tree on $n$ vertices has exactly $n - 1$ edges.

```
GRAPH PROPERTIES

  Connected          Disconnected         Bipartite            Tree
  (one component)    (two components)     (two-colorable)      (connected, no cycles)

   ●───●              ●───●   ●──●        ○───●               ●
   │ ╲ │              │       │  │        │   │              / | \
   │  ╲│                      │  │        ○───●             ●  ●  ●
   ●───●              ●       ●──●        │   │               / \
                                          ○───●              ●   ●
```

*ML connection:* **Decision trees** are rooted trees where internal nodes are feature tests and leaves are predictions. **Bayesian networks** are directed acyclic graphs (DAGs) encoding conditional independence. Bipartite graphs model **user-item interactions** in recommendation systems — users and items form the two partitions.

---

## 4. Graph Traversal

**Definition 7.6.13 (Breadth-First Search).** BFS explores vertices in order of their distance from a source vertex $s$, using a queue. It discovers all vertices at distance $k$ before any vertex at distance $k+1$.

**Definition 7.6.14 (Depth-First Search).** DFS explores as far as possible along each branch before backtracking, using a stack (or recursion). It produces a DFS tree and classifies edges as tree, back, forward, or cross edges.

**Definition 7.6.15 (Topological Sort).** A *topological ordering* of a directed acyclic graph (DAG) is a linear ordering of vertices such that for every directed edge $(u, v)$, vertex $u$ appears before $v$. Every DAG has at least one topological ordering.

```
TOPOLOGICAL SORT OF A COMPUTATIONAL GRAPH

  Forward pass order (topological sort):

     x ──→ [×W₁] ──→ h₁ ──→ [ReLU] ──→ a₁ ──→ [×W₂] ──→ h₂ ──→ [Loss]
                                                                      ↑
                                                                      y

  Backpropagation order (reverse topological sort):

     ∂L/∂x ←── ∂h₁ ←── ∂a₁ ←── ∂h₂ ←── [Loss]
```

*ML connection:* **Computational graphs** in deep learning frameworks (PyTorch, TensorFlow) are DAGs. The forward pass evaluates nodes in topological order. **Backpropagation** traverses the graph in reverse topological order, applying the chain rule at each node. Automatic differentiation relies on this graph structure — each operation records its inputs, enabling gradient computation via reverse-mode traversal.

---

## 5. Spectral Graph Theory

This is the most important section for ML applications. Spectral graph theory studies graphs through the eigenvalues and eigenvectors of associated matrices.

### 5.1 The Graph Laplacian and Its Spectrum

**Theorem 7.6.6 (Properties of the Laplacian).** Let $L = D - A$ be the Laplacian of an undirected graph $G$ with eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$. Then:

1. $L$ is symmetric and positive semi-definite
2. $\lambda_1 = 0$ with eigenvector $\mathbf{1} = (1, 1, \ldots, 1)^T$
3. $\lambda_2 > 0$ if and only if $G$ is connected
4. $\lambda_n \leq 2\Delta$ where $\Delta = \max_v \deg(v)$

*Proof of (1) and (2).* Symmetry: $L^T = (D-A)^T = D^T - A^T = D - A = L$ since $D$ is diagonal and $A$ is symmetric for undirected graphs. PSD: for any $\mathbf{x} \in \mathbb{R}^n$,

$$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T D \mathbf{x} - \mathbf{x}^T A \mathbf{x} = \sum_i d_i x_i^2 - \sum_{\{i,j\} \in E} 2x_i x_j = \sum_{\{i,j\} \in E} (x_i - x_j)^2 \geq 0$$

For (2): $L\mathbf{1} = D\mathbf{1} - A\mathbf{1} = \mathbf{d} - \mathbf{d} = \mathbf{0}$ where $\mathbf{d}$ is the degree vector, since each row of $A$ sums to $\deg(v_i)$. $\square$

**Definition 7.6.16 (Algebraic Connectivity).** The second-smallest eigenvalue $\lambda_2$ of $L$ is called the *algebraic connectivity* or *Fiedler value* of $G$. The corresponding eigenvector $\mathbf{v}_2$ is the *Fiedler vector*.

**Example 7.6.7:** For the cycle graph $C_4$ (square: $1-2-3-4-1$), the Laplacian is:

$$L = \begin{pmatrix} 2 & -1 & 0 & -1 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ -1 & 0 & -1 & 2 \end{pmatrix}$$

The eigenvalues are $\lambda_1 = 0,\; \lambda_2 = 2,\; \lambda_3 = 2,\; \lambda_4 = 4$. The algebraic connectivity is $\lambda_2 = 2 > 0$, confirming $C_4$ is connected. The Fiedler vector $\mathbf{v}_2 = (1, -1, 1, -1)^T / 2$ alternates in sign, suggesting the bipartition $\{1,3\}$ vs. $\{2,4\}$ — which is exactly the two-coloring of this bipartite graph. Also note $\lambda_4 = 4 \leq 2\Delta = 2 \times 2 = 4$, matching the upper bound.

### 5.2 The Cheeger Inequality

The Cheeger inequality connects spectral properties to combinatorial graph structure.

**Definition 7.6.17 (Cheeger Constant).** The *Cheeger constant* (or isoperimetric number) of a graph $G$ is:

$$h(G) = \min_{\substack{S \subset V \\ 0 < |S| \leq n/2}} \frac{|\partial S|}{\min(|S|, |V \setminus S|)}$$

where $\partial S = \{\{u,v\} \in E : u \in S, v \notin S\}$ is the edge boundary. Intuitively, $h(G)$ measures the "bottleneck" — how hard it is to partition the graph.

**Theorem 7.6.7 (Cheeger Inequality).** For a $d$-regular graph:

$$\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2d \cdot \lambda_2}$$

The lower bound says: if $\lambda_2$ is small, there exists a near-partition (small cut). The upper bound says: if $\lambda_2$ is large, every cut is large (the graph is well-connected).

### 5.3 Spectral Clustering

The spectral clustering algorithm exploits the Laplacian eigenvectors:

1. Compute the Laplacian $L = D - A$ (or $\mathcal{L} = I - D^{-1/2}AD^{-1/2}$)
2. Find the $k$ smallest eigenvectors $\mathbf{u}_1, \ldots, \mathbf{u}_k$
3. Form the matrix $U \in \mathbb{R}^{n \times k}$ with these eigenvectors as columns
4. Treat each row of $U$ as a point in $\mathbb{R}^k$ and run $k$-means

```
SPECTRAL CLUSTERING — WHY IT WORKS

  Original graph                    Fiedler vector v₂
  (two clusters)                    (values on vertices)

   ●───●   ╌╌╌   ●───●            +0.5  +0.5      -0.5  -0.5
   │   │   weak   │   │             ●─────●  ╌╌╌  ●─────●
   ●───●   edge   ●───●            +0.5  +0.5      -0.5  -0.5

  Cluster A         Cluster B       Positive values    Negative values
                                    → Cluster A        → Cluster B

  The Fiedler vector is approximately constant within clusters
  and changes sign across the cut. Thresholding at 0 gives
  the bipartition.
```

**Theorem 7.6.8 (Optimality of Spectral Bipartitioning).** The Fiedler vector solves the relaxed normalized min-cut problem:

$$\mathbf{v}_2 = \arg\min_{\mathbf{x} \perp \mathbf{1}, \|\mathbf{x}\|=1} \mathbf{x}^T L \mathbf{x} = \arg\min_{\mathbf{x} \perp \mathbf{1}, \|\mathbf{x}\|=1} \sum_{\{i,j\} \in E} (x_i - x_j)^2$$

*Proof.* This follows directly from the Rayleigh quotient characterization: $\lambda_2 = \min_{\mathbf{x} \perp \mathbf{1}} R_L(\mathbf{x})$, attained at the eigenvector $\mathbf{v}_2$. The quadratic form $\sum (x_i - x_j)^2$ penalizes assigning different values to connected vertices, so the minimizer assigns similar values to vertices in the same cluster. $\square$

*ML connection:* Spectral clustering is a core technique in **community detection**, **image segmentation**, and **semi-supervised learning**. It naturally handles non-convex cluster shapes that $k$-means cannot. The normalized cut objective (Shi and Malik, 2000) uses the normalized Laplacian $\mathcal{L}$, which balances cluster sizes.

### 5.4 The Graph Fourier Transform

The classical Fourier transform decomposes signals into frequency components (eigenfunctions of the Laplacian on $\mathbb{R}^n$). The graph Fourier transform does the same for signals on graphs.

**Definition 7.6.25 (Graph Signal).** A *graph signal* is a function $f : V \to \mathbb{R}$, represented as a vector $\mathbf{f} \in \mathbb{R}^n$.

**Definition 7.6.26 (Graph Fourier Transform).** Given the eigendecomposition $L = U \Lambda U^T$, the *graph Fourier transform* of signal $\mathbf{f}$ is:

$$\hat{\mathbf{f}} = U^T \mathbf{f}$$

The inverse transform is $\mathbf{f} = U \hat{\mathbf{f}}$. The eigenvectors of $L$ serve as the "frequency basis": $\mathbf{u}_1$ (eigenvalue $0$) is the constant/DC component, and eigenvectors with larger eigenvalues oscillate more rapidly across edges.

**Definition 7.6.27 (Spectral Graph Convolution).** The *convolution* of signals $\mathbf{f}$ and $\mathbf{g}$ on a graph is defined in the spectral domain as pointwise multiplication:

$$\mathbf{f} *_G \mathbf{g} = U\left(\hat{\mathbf{f}} \odot \hat{\mathbf{g}}\right) = U\left(U^T\mathbf{f} \odot U^T\mathbf{g}\right)$$

A learnable spectral filter $g_\theta$ parameterized by $\theta$ acts as:

$$g_\theta *_G \mathbf{f} = U \, g_\theta(\Lambda) \, U^T \mathbf{f}$$

where $g_\theta(\Lambda) = \text{diag}(g_\theta(\lambda_1), \ldots, g_\theta(\lambda_n))$.

```
GRAPH FOURIER TRANSFORM — FREQUENCY INTERPRETATION

  Low frequency (λ ≈ 0)           High frequency (λ large)
  Signal varies slowly             Signal oscillates rapidly

   +1──+1──+1──+1                  +1──-1──+1──-1
   Smooth across graph              Changes sign across edges
   (e.g., cluster labels)           (e.g., noise, fine detail)

  Graph filtering: multiply spectral coefficients by g(λ)

   Low-pass filter                 Band-pass filter
   g(λ) ─┐                        g(λ)   ┌─┐
         │                               │ │
         └──────── λ               ──────┘ └──── λ
   Keeps smooth signals            Keeps specific frequencies
   (smoothing, denoising)          (community detection)
```

*ML connection:* The graph Fourier transform is computationally expensive ($O(n^2)$ per transform, $O(n^3)$ for eigendecomposition). This motivated **polynomial spectral filters** — ChebNet uses Chebyshev polynomials $g_\theta(\Lambda) = \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})$ which can be computed as $g_\theta(L)\mathbf{f} = \sum_{k=0}^K \theta_k T_k(\tilde{L})\mathbf{f}$ in $O(K \cdot m)$ time without ever computing eigenvectors. GCN simplifies further to $K=1$.

---

## 6. Random Graphs

**Definition 7.6.18 (Erdos-Renyi Model).** The Erdos-Renyi random graph $G(n, p)$ has $n$ vertices, and each pair of vertices is connected independently with probability $p$.

**Theorem 7.6.9 (Connectivity Threshold).** In $G(n, p)$:

- If $p < \frac{(1-\varepsilon)\ln n}{n}$ for any $\varepsilon > 0$, then $G$ is disconnected with high probability
- If $p > \frac{(1+\varepsilon)\ln n}{n}$, then $G$ is connected with high probability

This is a sharp *phase transition* — connectivity emerges suddenly at $p \approx \frac{\ln n}{n}$.

**Definition 7.6.19 (Small-World Graph).** The Watts-Strogatz model starts with a regular ring lattice and rewires each edge with probability $p$. For intermediate $p$, the resulting graph exhibits *small-world* properties: high clustering coefficient and short average path length.

**Definition 7.6.20 (Scale-Free Graph).** A *scale-free* network has a degree distribution following a power law: $P(\deg(v) = k) \propto k^{-\gamma}$. The Barabasi-Albert model generates scale-free networks via *preferential attachment*: new vertices connect preferentially to high-degree vertices ("the rich get richer").

```
DEGREE DISTRIBUTIONS

  Erdős–Rényi             Scale-Free (Power Law)
  (Poisson-like)          (Heavy-tailed)

  P(k)                    P(k)
   │                       │
   │   ●                   │●
   │  ● ●                  │ ●
   │ ●   ●                 │  ●
   │●     ●                │   ●
   │       ● ●             │    ● ● ● ● ●
   └────────────── k       └──────────────── k
   Most nodes ~same        Few hubs with very
   degree (concentrated)   high degree (long tail)
```

*ML connection:* Random graph models are essential for **network analysis** (social networks, citation networks, biological networks). Scale-free properties explain the robustness and vulnerability patterns of real networks — removing random nodes barely affects connectivity, but targeted removal of hubs is catastrophic. **Graph generation** models (GraphRNN, GRAN) learn to generate graphs with realistic structural properties.

**Example 7.6.9 (PageRank Iteration):** Consider a 3-page web graph: page $A$ links to $B$ and $C$; page $B$ links to $C$; page $C$ links to $A$. The transition matrix $M$ (column-stochastic) is:

$$M = \begin{pmatrix} 0 & 0 & 1 \\ 1/2 & 0 & 0 \\ 1/2 & 1 & 0 \end{pmatrix}$$

With damping factor $d = 0.85$ and uniform initialization $\mathbf{r}^{(0)} = (1/3, 1/3, 1/3)^T$, one PageRank iteration computes $\mathbf{r}^{(1)} = \frac{1-d}{n}\mathbf{1} + d \cdot M\mathbf{r}^{(0)}$:

$$M\mathbf{r}^{(0)} = \begin{pmatrix} 1/3 \\ 1/6 \\ 1/2 \end{pmatrix}, \quad \mathbf{r}^{(1)} = 0.05 \cdot \mathbf{1} + 0.85 \begin{pmatrix} 1/3 \\ 1/6 \\ 1/2 \end{pmatrix} = \begin{pmatrix} 0.333 \\ 0.192 \\ 0.475 \end{pmatrix}$$

Page $C$ has the highest rank after one step: it receives links from both $A$ and $B$.

---

## 7. Graph Neural Networks (Mathematical Foundation)

### 7.1 Message Passing Framework

**Definition 7.6.21 (Message Passing Neural Network).** A message passing neural network (MPNN) updates vertex representations over $T$ iterations:

$$\mathbf{m}_v^{(t+1)} = \bigoplus_{u \in \mathcal{N}(v)} \phi\left(\mathbf{h}_v^{(t)}, \mathbf{h}_u^{(t)}, \mathbf{e}_{uv}\right)$$

$$\mathbf{h}_v^{(t+1)} = \psi\left(\mathbf{h}_v^{(t)}, \mathbf{m}_v^{(t+1)}\right)$$

where $\mathcal{N}(v)$ is the neighborhood of $v$, $\phi$ is the *message function*, $\bigoplus$ is a permutation-invariant *aggregation* (sum, mean, max), and $\psi$ is the *update function*.

**Example 7.6.10 (One GNN Message-Passing Step):** Consider a triangle graph $1-2-3$ with initial node features $\mathbf{h}_1^{(0)} = [1, 0]$, $\mathbf{h}_2^{(0)} = [0, 1]$, $\mathbf{h}_3^{(0)} = [1, 1]$. Using a simple GCN-style rule with **mean aggregation** (no learned weights yet):

$$\mathbf{h}_v^{(1)} = \text{MEAN}\left(\{\mathbf{h}_u^{(0)} : u \in \mathcal{N}(v) \cup \{v\}\}\right)$$

For vertex $1$ ($\mathcal{N}(1) = \{2, 3\}$):

$$\mathbf{h}_1^{(1)} = \frac{1}{3}\left([1,0] + [0,1] + [1,1]\right) = \left[\frac{2}{3}, \frac{2}{3}\right]$$

For vertex $2$ ($\mathcal{N}(2) = \{1, 3\}$): $\mathbf{h}_2^{(1)} = \frac{1}{3}([0,1] + [1,0] + [1,1]) = [2/3, 2/3]$. For vertex $3$: $\mathbf{h}_3^{(1)} = \frac{1}{3}([1,1] + [1,0] + [0,1]) = [2/3, 2/3]$. After one step on a triangle, all features become identical -- information has fully diffused. This illustrates the **over-smoothing** problem in deep GNNs.

### 7.2 Key GNN Architectures as Laplacian Operations

| Architecture | Update Rule | Spectral Interpretation |
|-------------|-------------|------------------------|
| GCN (Kipf) | $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$ | 1st-order Chebyshev filter on $\mathcal{L}$ |
| GAT | $h_v = \sigma(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u)$ | Adaptive, data-dependent filter |
| GraphSAGE | $h_v = \sigma(W \cdot [\mathbf{h}_v \| \text{AGG}(\{h_u\})])$ | Sampling-based spatial filter |

Here $\tilde{A} = A + I$ (self-loops added), $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$, and $\alpha_{vu}$ are learned attention coefficients.

**Theorem 7.6.10 (GCN as Spectral Filter).** The GCN propagation rule $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ is a first-order approximation of a Chebyshev spectral filter on the normalized Laplacian. Specifically, it approximates the filter $g_\theta(\Lambda) = \theta_0 I + \theta_1 \tilde{\Lambda}$ where $\tilde{\Lambda} = \frac{2}{\lambda_{\max}}\Lambda - I$, with the constraint $\theta_0 = -\theta_1 = \theta$.

*Proof sketch.* The $K$-th order Chebyshev filter is $g_\theta(\mathcal{L}) = \sum_{k=0}^K \theta_k T_k(\tilde{\mathcal{L}})$. Setting $K = 1$ and $\theta_0 = -\theta_1 = \theta$ with $\lambda_{\max} \approx 2$ gives $g_\theta(\mathcal{L}) \approx \theta(I + D^{-1/2}AD^{-1/2})$. Renormalizing with $\tilde{A} = A + I$ yields the GCN rule. $\square$

### 7.3 Expressiveness and the WL Test

**Definition 7.6.22 (Weisfeiler-Leman Test).** The 1-dimensional Weisfeiler-Leman (1-WL) graph isomorphism test iteratively refines vertex labels:

$$c^{(t+1)}(v) = \text{HASH}\left(c^{(t)}(v), \{\!\{c^{(t)}(u) : u \in \mathcal{N}(v)\}\!\}\right)$$

where $\{\!\{\cdot\}\!\}$ denotes a multiset. The algorithm terminates when the labeling stabilizes. Two graphs are *1-WL equivalent* if they produce the same multiset of final labels.

**Theorem 7.6.11 (Xu et al., 2019).** Any message-passing GNN is at most as powerful as the 1-WL test in distinguishing non-isomorphic graphs. The Graph Isomorphism Network (GIN) achieves this upper bound with the update rule:

$$\mathbf{h}_v^{(t+1)} = \text{MLP}\left((1 + \varepsilon^{(t)}) \cdot \mathbf{h}_v^{(t)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(t)}\right)$$

where $\varepsilon$ is a learnable scalar and the MLP is injective on the input space.

```
WL TEST — DISTINGUISHING GRAPHS

  Round 0: Initial labels (all same)

   Graph 1:  ○───○───○───○     Graph 2:  ○───○
             (path P₄)                   │   │
                                         ○───○
                                         (cycle C₄)

  Round 1: Refine by neighbor multisets

   Graph 1:  A───B───B───A     Graph 2:  B───B
             (endpoints ≠                 │   │
              middle)                     B───B
                                          (all same)
  After Round 1:
    Graph 1 label multiset: {A, A, B, B}
    Graph 2 label multiset: {B, B, B, B}
    → Distinguished! (different multisets)
```

*ML connection:* The WL expressiveness result is foundational: it tells us exactly what structural patterns message-passing GNNs can and cannot detect. Standard MPNNs cannot distinguish certain non-isomorphic regular graphs (e.g., some 3-regular graphs). This has motivated **higher-order GNNs** (using $k$-WL for $k \geq 2$), **subgraph GNNs**, and **equivariant architectures** that provably exceed 1-WL power.

---

## 8. Graph Algorithms for ML

### 8.1 Shortest Paths

**Theorem 7.6.12 (Dijkstra's Algorithm Correctness).** For a graph with non-negative edge weights, Dijkstra's algorithm correctly computes the shortest path from a source $s$ to all other vertices in $O((n + m) \log n)$ time with a binary heap.

**Example 7.6.8:** Trace Dijkstra from source $s = A$ on this weighted graph:

```
    A ─3─ B ─2─ D
    |         /
    1       4
    |     /
    C ──5── D
```

Edges: $A\text{-}B{=}3,\; A\text{-}C{=}1,\; B\text{-}D{=}2,\; C\text{-}D{=}5$.

| Step | Visited          | $d(A)$ | $d(B)$    | $d(C)$    | $d(D)$          | Extract   |
| ---- | ---------------- | ------ | --------- | --------- | --------------- | --------- |
| Init | $\emptyset$      | $0$    | $\infty$  | $\infty$  | $\infty$        | $A$       |
| 1    | $\{A\}$          | $0$    | $3$       | $1$       | $\infty$        | $C$ (min) |
| 2    | $\{A,C\}$        | $0$    | $3$       | $1$       | $6$             | $B$ (min) |
| 3    | $\{A,C,B\}$      | $0$    | $3$       | $1$       | $\mathbf{5}$    | $D$       |

Shortest paths: $A{\to}C = 1$, $A{\to}B = 3$, $A{\to}B{\to}D = 5$ (beats $A{\to}C{\to}D = 6$).

| Algorithm | Complexity | Use Case |
|-----------|-----------|----------|
| BFS | $O(n + m)$ | Unweighted shortest paths |
| Dijkstra | $O((n+m) \log n)$ | Non-negative weights |
| Bellman-Ford | $O(nm)$ | Negative weights (no neg. cycles) |
| Floyd-Warshall | $O(n^3)$ | All-pairs shortest paths |

*ML connection:* Graph distance is used as a feature in **link prediction** and **node classification**. Shortest-path kernels measure graph similarity for **graph classification**. In **knowledge graph reasoning**, multi-hop paths from entity to entity represent inference chains.

### 8.2 Minimum Spanning Trees

**Definition 7.6.23 (Spanning Tree).** A *spanning tree* of a connected graph $G$ is a subgraph that is a tree and includes all vertices. A *minimum spanning tree* (MST) minimizes $\sum_{e \in T} w(e)$.

**Theorem 7.6.13 (Cut Property).** For any cut $(S, V \setminus S)$ of $G$, the minimum-weight edge crossing the cut is in every MST (assuming unique edge weights).

### 8.3 Max-Flow / Min-Cut

**Theorem 7.6.14 (Max-Flow Min-Cut Theorem, Ford-Fulkerson).** In a flow network with source $s$ and sink $t$, the maximum flow from $s$ to $t$ equals the minimum capacity of an $s$-$t$ cut:

$$\max_f \text{val}(f) = \min_{(S,T)} \text{cap}(S, T)$$

```
MAX-FLOW / MIN-CUT

  Source s                      Sink t
                ┌───┐
         10     │   │    8
    s ────────→ A ────────→ t
    │           │         ↑
    │     6     │  5      │ 7
    └────────→ B ─────→ C ┘

  Max flow = 15 (routes: s→A→t: 8, s→B→C→t: 7)
  Min cut = {(s,A)=10, (B,C)=5} = 15  ✓
```

*ML connection:* Max-flow/min-cut has direct applications in **image segmentation** (graph cuts method), where pixels are nodes, edges encode similarity, and the min-cut separates foreground from background. The **normalized cut** used in spectral clustering is a relaxation of the min-cut objective. Network flow also arises in **optimal transport** for comparing distributions.

### 8.4 Graph Coloring

**Definition 7.6.24 (Chromatic Number).** The *chromatic number* $\chi(G)$ is the minimum number of colors needed to color the vertices of $G$ such that no two adjacent vertices share the same color.

**Theorem 7.6.15.** For any graph $G$: $\chi(G) \leq \Delta(G) + 1$ where $\Delta(G)$ is the maximum degree.

*ML connection:* Graph coloring appears in **scheduling problems** (exam scheduling, register allocation) and in **parallel computation** — coloring a dependency graph determines which operations can run simultaneously. In GNN implementations, vertex coloring identifies independent sets for **parallel aggregation**.

---

## 9. Summary of Key Results

| Result | Statement | ML Relevance |
| ------ | --------- | ------------ |
| Handshaking lemma | $\sum \deg(v) = 2\lvert E\rvert$ | Degree statistics in network analysis |
| Walk counting | $(A^k)_{ij}$ = walks of length $k$ | Multi-hop message passing |
| Laplacian PSD | $\mathbf{x}^TL\mathbf{x} = \sum_{(i,j)\in E}(x_i - x_j)^2 \geq 0$ | Graph smoothness, regularization |
| Connectivity and spectrum | Components $= \dim(\ker L)$ | Graph structure from eigenvalues |
| Fiedler vector | $\mathbf{v}_2$ solves relaxed min-cut | Spectral clustering |
| Cheeger inequality | $\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2d\lambda_2}$ | Cluster quality guarantees |
| GCN = spectral filter | $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ is 1st-order Chebyshev | GNN design principles |
| WL expressiveness | MPNNs $\leq$ 1-WL in power | GNN limitations, architecture design |
| Max-flow min-cut | $\max \text{flow} = \min \text{cut}$ | Image segmentation, graph cuts |

---

## Exercises

**★ Basic**

1. Draw the adjacency matrix for the following graph: a triangle on vertices $\{1, 2, 3\}$ with an additional edge from $3$ to $4$. Verify the handshaking lemma.

2. Compute the Laplacian $L = D - A$ for the path graph $P_4$ (vertices $1$-$2$-$3$-$4$). Verify that $L\mathbf{1} = \mathbf{0}$.

3. How many walks of length $2$ exist from vertex $1$ to vertex $3$ in the graph from Exercise 1? Verify using $A^2$.

4. Is the following graph bipartite? Justify.
```
    1 --- 2
    |     |
    3 --- 4
    |     |
    5 --- 6
```

**★★ Intermediate**

5. For the cycle graph $C_4$ (a square), compute the Laplacian eigenvalues and eigenvectors. Verify that $\lambda_2 > 0$ (the graph is connected) and identify the Fiedler vector. What bipartition does it suggest?

6. Prove that for any graph $G$, the number of triangles equals $\frac{1}{6}\text{tr}(A^3)$ where $A$ is the adjacency matrix.

7. Show that the eigenvalues of the normalized Laplacian $\mathcal{L}$ lie in $[0, 2]$. When does $\lambda = 2$ occur?

8. In the GCN update rule $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$, explain the role of the self-loop ($\tilde{A} = A + I$) and the symmetric normalization ($\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$). What goes wrong without them?

**★★★ Challenging**

9. **(Cheeger inequality)** Consider the "barbell graph": two complete graphs $K_k$ connected by a single edge. Compute or bound $\lambda_2$ and $h(G)$. Verify that the Cheeger inequality holds.

10. **(WL expressiveness)** Construct two non-isomorphic 3-regular graphs on 6 vertices that the 1-WL test cannot distinguish. Explain why standard message-passing GNNs also fail to distinguish them.

11. **(Spectral clustering)** Given the weighted adjacency matrix:

$$W = \begin{pmatrix} 0 & 5 & 5 & 1 \\ 5 & 0 & 5 & 1 \\ 5 & 5 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{pmatrix}$$

Compute the normalized Laplacian $\mathcal{L}$, find its eigenvalues, and determine the optimal 2-way partition using the Fiedler vector.

12. **(Graph filter design)** Prove that any polynomial filter $g(\mathcal{L}) = \sum_{k=0}^K a_k \mathcal{L}^k$ on the graph is a $K$-localized operation: the output at vertex $v$ depends only on vertices within $K$ hops of $v$. Relate this to the concept of *receptive field* in GNNs.

---

## Related Topics

- [Eigenvalues](../linear-algebra/eigenvalues.md) — spectral theory underlying the graph Laplacian
- [Matrix Decompositions](../linear-algebra/matrix-decompositions.md) — SVD for low-rank graph approximations
- [Matrices](../linear-algebra/matrices.md) — adjacency and Laplacian matrices
- Abstract Algebra — graph automorphisms, symmetry, equivariant networks
- Topology — simplicial complexes generalize graphs to higher dimensions
- [Optimization](../optimization/index.md) — graph-based optimization, network flows
- [Probability Foundations](../probability/probability-foundations.md) — random graphs, probabilistic graph models
