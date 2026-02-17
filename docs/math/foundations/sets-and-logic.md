# Sets and Logic

Every mathematical statement in machine learning — from "this data point belongs to class A" to "the probability that $x$ falls in region $R$" — is built on set theory and logic. Sets define the collections we reason about; logic defines how we reason about them. Probability theory, the backbone of ML, is formalized entirely in terms of sets ($\sigma$-algebras) and set functions (measures). Understanding sets and logic is not optional — it is the grammar of mathematics.

---

## Prerequisites

None. This is the starting point.

---

## 1. Naive Set Theory

### 1.1 Sets and Elements

**Definition 1.1 (Set).** A *set* is an unordered collection of distinct objects called *elements* (or *members*). We write $x \in A$ to mean "$x$ is an element of $A$" and $x \notin A$ to mean "$x$ is not an element of $A$."

Sets can be specified by:

- **Roster notation:** $A = \{1, 2, 3\}$
- **Set-builder notation:** $A = \{x \in \mathbb{Z} \mid x > 0 \text{ and } x \leq 3\}$

**Important sets:**

| Symbol | Name | Elements | ML Context |
|--------|------|----------|------------|
| $\emptyset$ | Empty set | None | Empty prediction set |
| $\mathbb{N}$ | Natural numbers | $\{0, 1, 2, \ldots\}$ | Counts, indices |
| $\mathbb{Z}$ | Integers | $\{\ldots, -1, 0, 1, \ldots\}$ | Quantized weights |
| $\mathbb{Q}$ | Rationals | $\{p/q : p, q \in \mathbb{Z}, q \neq 0\}$ | — |
| $\mathbb{R}$ | Real numbers | The continuum | Model parameters, features |
| $\mathbb{C}$ | Complex numbers | $\{a + bi : a, b \in \mathbb{R}\}$ | Quantum amplitudes, Fourier |
| $\mathbb{R}^n$ | $n$-dimensional real space | $n$-tuples of reals | Feature vectors, weight spaces |

### 1.2 Subsets and Equality

**Definition 1.2 (Subset).** $A \subseteq B$ (A is a subset of B) if every element of $A$ is also an element of $B$:

$$A \subseteq B \iff (\forall x)(x \in A \Rightarrow x \in B)$$

**Definition 1.3 (Set Equality).** $A = B$ if and only if $A \subseteq B$ and $B \subseteq A$.

This gives us the standard proof technique for showing two sets are equal: show each is a subset of the other.

**Definition 1.4 (Proper Subset).** $A \subsetneq B$ if $A \subseteq B$ and $A \neq B$.

**Example 1.2:** Let $A = \{1, 2\}$, $B = \{1, 2, 3, 4\}$, $C = \{1, 2\}$.

- $A \subseteq B$ ✓ (every element of $A$ is in $B$)
- $A \subsetneq B$ ✓ (proper subset: $A \subseteq B$ but $A \neq B$ since $3 \in B$ but $3 \notin A$)
- $A \subseteq C$ and $C \subseteq A$, so $A = C$ by Definition 1.3
- $B \subseteq A$? ✗ ($3 \in B$ but $3 \notin A$)

**Definition 1.5 (Power Set).** The *power set* $\mathcal{P}(A)$ is the set of all subsets of $A$. If $|A| = n$, then $|\mathcal{P}(A)| = 2^n$.

*ML connection:* In feature selection, choosing a subset of $n$ features means selecting an element of $\mathcal{P}(\{f_1, \ldots, f_n\})$. The search space has size $2^n$ — exponential in the number of features.

**Example 1.5:** Let $A = \{a, b, c\}$. Then $|A| = 3$ and $|\mathcal{P}(A)| = 2^3 = 8$:

$$\mathcal{P}(A) = \{\emptyset,\; \{a\},\; \{b\},\; \{c\},\; \{a,b\},\; \{a,c\},\; \{b,c\},\; \{a,b,c\}\}$$

Note that $\emptyset \in \mathcal{P}(A)$ and $A \in \mathcal{P}(A)$ always hold. With 10 features, the power set has $2^{10} = 1024$ subsets — this is why exhaustive feature selection becomes infeasible for large $n$.

### 1.3 Set Operations

**Definition 1.6 (Union).** $A \cup B = \{x : x \in A \text{ or } x \in B\}$

**Definition 1.7 (Intersection).** $A \cap B = \{x : x \in A \text{ and } x \in B\}$

**Definition 1.8 (Difference).** $A \setminus B = \{x : x \in A \text{ and } x \notin B\}$

**Definition 1.9 (Complement).** $A^c = \{x \in U : x \notin A\}$ where $U$ is the universal set.

**Example 1.6–1.9:** Let $U = \{1,2,3,4,5,6,7\}$, $A = \{1,2,3,4\}$, $B = \{3,4,5,6\}$.

- **Union:** $A \cup B = \{1,2,3,4,5,6\}$ — everything in either set
- **Intersection:** $A \cap B = \{3,4\}$ — only elements in both
- **Difference:** $A \setminus B = \{1,2\}$ — in $A$ but not in $B$; $B \setminus A = \{5,6\}$
- **Complement:** $A^c = \{5,6,7\}$ — everything in $U$ not in $A$

Notice that $A \setminus B \neq B \setminus A$ in general — set difference is not commutative.

**Definition 1.10 (Cartesian Product).** $A \times B = \{(a, b) : a \in A, b \in B\}$

*ML connection:* The feature space of an ML model is a Cartesian product. If feature 1 takes values in $\mathbb{R}$ and feature 2 takes values in $\{0, 1\}$, the input space is $\mathbb{R} \times \{0, 1\}$.

**Example 1.10:** Let $A = \{1, 2, 3\}$ and $B = \{x, y\}$. Then:

$$A \times B = \{(1,x),\; (1,y),\; (2,x),\; (2,y),\; (3,x),\; (3,y)\}$$

Note $|A \times B| = |A| \cdot |B| = 3 \cdot 2 = 6$. Also, $A \times B \neq B \times A$ — the pair $(1, x)$ is not the same as $(x, 1)$. In ML, if you have 3 possible values for "color" and 2 for "size," the joint feature space has $3 \times 2 = 6$ combinations.

```
SET OPERATIONS VISUALIZED

  A ∪ B (Union)         A ∩ B (Intersection)     A \ B (Difference)
┌─────────────────┐   ┌─────────────────┐      ┌─────────────────┐
│ ████████████████ │   │      ████       │      │ █████           │
│ ████████████████ │   │      ████       │      │ █████           │
│ ████████████████ │   │      ████       │      │ █████           │
└─────────────────┘   └─────────────────┘      └─────────────────┘
  Everything in        Only the overlap          A without B
  either set
```

### 1.4 Laws of Set Algebra

**Theorem 1.1 (De Morgan's Laws).**

$$
(A \cup B)^c = A^c \cap B^c \qquad \text{and} \qquad (A \cap B)^c = A^c \cup B^c
$$

*Proof.* We prove the first identity. Let $x \in (A \cup B)^c$. Then $x \notin A \cup B$, so $x \notin A$ and $x \notin B$. Thus $x \in A^c$ and $x \in B^c$, so $x \in A^c \cap B^c$. Conversely, let $x \in A^c \cap B^c$. Then $x \notin A$ and $x \notin B$, so $x \notin A \cup B$, giving $x \in (A \cup B)^c$. $\square$

*ML connection:* De Morgan's laws appear in Boolean logic classifiers and in computing the complement of decision regions. If a classifier predicts "class A or class B," the complement of that region is "not class A *and* not class B."

**Example 1.1 (Theorem):** Verify De Morgan's first law with $U = \{1,2,3,4,5\}$, $A = \{1,2,3\}$, $B = \{3,4\}$.

- Left side: $A \cup B = \{1,2,3,4\}$, so $(A \cup B)^c = \{5\}$
- Right side: $A^c = \{4,5\}$, $B^c = \{1,2,5\}$, so $A^c \cap B^c = \{5\}$
- Both sides equal $\{5\}$ ✓

For the second law: $(A \cap B)^c = \{3\}^c = \{1,2,4,5\}$ and $A^c \cup B^c = \{4,5\} \cup \{1,2,5\} = \{1,2,4,5\}$ ✓

**Other fundamental laws:**

| Law | Formula |
|-----|---------|
| Commutativity | $A \cup B = B \cup A$, $A \cap B = B \cap A$ |
| Associativity | $(A \cup B) \cup C = A \cup (B \cup C)$ |
| Distributivity | $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ |
| Identity | $A \cup \emptyset = A$, $A \cap U = A$ |
| Complement | $A \cup A^c = U$, $A \cap A^c = \emptyset$ |
| Idempotent | $A \cup A = A$, $A \cap A = A$ |

### 1.5 Indexed Families and Partitions

**Definition 1.11 (Indexed Union/Intersection).**

$$\bigcup_{i \in I} A_i = \{x : \exists\, i \in I,\; x \in A_i\} \qquad \bigcap_{i \in I} A_i = \{x : \forall\, i \in I,\; x \in A_i\}$$

**Example 1.11:** Let $A_1 = \{1,2,3\}$, $A_2 = \{2,3,4\}$, $A_3 = \{3,4,5\}$ with index set $I = \{1,2,3\}$.

- $\bigcup_{i=1}^{3} A_i = \{1,2,3,4,5\}$ — an element needs to be in *at least one* $A_i$
- $\bigcap_{i=1}^{3} A_i = \{3\}$ — only $3$ appears in *every* $A_i$

In ensemble learning, if each model $i$ predicts a set $A_i$ of likely classes, the intersection $\bigcap A_i$ gives the classes all models agree on (high-confidence predictions).

**Definition 1.12 (Partition).** A collection $\{A_1, A_2, \ldots, A_k\}$ is a *partition* of $S$ if:

1. $A_i \neq \emptyset$ for all $i$
2. $A_i \cap A_j = \emptyset$ for $i \neq j$ (pairwise disjoint)
3. $\bigcup_{i=1}^k A_i = S$

*ML connection:* A $k$-class classifier partitions the input space $\mathbb{R}^n$ into $k$ disjoint decision regions. Clustering algorithms (e.g., $k$-means) compute a partition of the dataset.

**Example 1.12:** Let $S = \{1,2,3,4,5,6\}$. Then $\{A_1, A_2, A_3\} = \{\{1,2\}, \{3,4\}, \{5,6\}\}$ is a partition of $S$ because:

1. Each $A_i \neq \emptyset$ ✓
2. Pairwise disjoint: $A_1 \cap A_2 = \emptyset$, $A_1 \cap A_3 = \emptyset$, $A_2 \cap A_3 = \emptyset$ ✓
3. $A_1 \cup A_2 \cup A_3 = \{1,2,3,4,5,6\} = S$ ✓

However, $\{\{1,2,3\}, \{3,4,5,6\}\}$ is *not* a partition because $3$ appears in both sets (violates condition 2). Think of this as a clustering where each data point must belong to exactly one cluster.

### 1.6 Cardinality

**Definition 1.13 (Cardinality).** The *cardinality* $|A|$ is the "size" of a set. For finite sets, it is the number of elements. For infinite sets, we compare cardinalities using bijections.

**Example 1.13:** Let $A = \{a, b, c\}$ and $B = \{1, 2, 3\}$. We can define a bijection $f: A \to B$ by $f(a)=1, f(b)=2, f(c)=3$, so $|A| = |B| = 3$. But there is no bijection between $\{a,b,c\}$ and $\{1,2\}$ — they have different cardinalities. For infinite sets, the same idea applies: $|\mathbb{N}| = |\mathbb{Z}|$ because the function $0 \mapsto 0, 1 \mapsto 1, 2 \mapsto -1, 3 \mapsto 2, 4 \mapsto -2, \ldots$ is a bijection.

**Theorem 1.2 (Cantor).** $|\mathbb{R}| > |\mathbb{N}|$. The real numbers are *uncountable*.

*Proof sketch (diagonal argument).* Suppose $f: \mathbb{N} \to [0,1)$ is a bijection. List the decimal expansions: $f(n) = 0.d_{n1}d_{n2}d_{n3}\ldots$ Construct $r = 0.r_1r_2r_3\ldots$ where $r_n \neq d_{nn}$ (e.g., $r_n = 5$ if $d_{nn} \neq 5$, else $r_n = 6$). Then $r \neq f(n)$ for any $n$ since they differ in the $n$-th digit. Contradiction. $\square$

*ML connection:* The set of all functions $f: \mathbb{R}^n \to \mathbb{R}$ is uncountable, but a neural network with fixed architecture can represent only countably many functions (parameterized by rational weights). The expressiveness gap between "all functions" and "learnable functions" is fundamental to approximation theory.

---

## 2. Propositional Logic

### 2.1 Propositions and Connectives

**Definition 1.14 (Proposition).** A *proposition* is a declarative sentence that is either true (T) or false (F), but not both.

**Logical connectives:**

| Symbol | Name | English | True When |
|--------|------|---------|-----------|
| $\neg P$ | Negation | "not $P$" | $P$ is false |
| $P \land Q$ | Conjunction | "$P$ and $Q$" | Both true |
| $P \lor Q$ | Disjunction | "$P$ or $Q$" | At least one true |
| $P \Rightarrow Q$ | Implication | "if $P$ then $Q$" | $P$ false or $Q$ true |
| $P \Leftrightarrow Q$ | Biconditional | "$P$ iff $Q$" | Both same truth value |

**Example 2.1:** Let $P$ = "the model accuracy is above 90%" (True) and $Q$ = "we deploy to production" (False).

| Expression | Meaning | Value |
| ---------- | ------- | ----- |
| $\neg P$ | "accuracy is not above 90%" | F |
| $P \land Q$ | "accuracy > 90% and we deploy" | F (both must be true) |
| $P \lor Q$ | "accuracy > 90% or we deploy" | T (at least one true) |
| $P \Rightarrow Q$ | "if accuracy > 90% then we deploy" | F ($P$ true but $Q$ false) |
| $Q \Rightarrow P$ | "if we deploy then accuracy > 90%" | T ($Q$ is false, so vacuously true) |

**The implication trap:** $P \Rightarrow Q$ is true whenever $P$ is false, regardless of $Q$. "If pigs fly, then 2+2=5" is a true statement. This *vacuous truth* matters when reasoning about edge cases in ML (e.g., "if the training set is empty, the model has zero error" is vacuously true).

### 2.2 Truth Tables and Logical Equivalence

**Definition 1.15.** Two propositions are *logically equivalent* ($P \equiv Q$) if they have the same truth value in every possible assignment.

**Key equivalences:**

- **Contrapositive:** $(P \Rightarrow Q) \equiv (\neg Q \Rightarrow \neg P)$
- **De Morgan:** $\neg(P \land Q) \equiv (\neg P \lor \neg Q)$
- **Double negation:** $\neg(\neg P) \equiv P$
- **Material implication:** $(P \Rightarrow Q) \equiv (\neg P \lor Q)$

**Example 2.2:** The contrapositive in action. Consider the statement:

- $P \Rightarrow Q$: "If a model overfits ($P$), then training error < test error ($Q$)."
- Contrapositive $\neg Q \Rightarrow \neg P$: "If training error $\geq$ test error ($\neg Q$), then the model does not overfit ($\neg P$)."

Both statements are logically identical. The contrapositive is often easier to verify: instead of checking all overfitting models, check cases where training error $\geq$ test error and confirm there is no overfitting.

*ML connection:* Logic gates are the building blocks of neural network hardware (AND, OR, NOT). The XOR problem — showing that a single perceptron cannot compute $P \oplus Q$ — was a foundational result that motivated multi-layer networks.

---

## 3. Predicate Logic and Quantifiers

### 3.1 Predicates

**Definition 1.16 (Predicate).** A *predicate* is a proposition-valued function of one or more variables. $P(x)$ is a predicate; it becomes a proposition when $x$ is given a specific value.

**Example:** $P(x) = $ "$x$ is a support vector" is a predicate over the training set. For a specific data point $x_i$, $P(x_i)$ is either true or false.

### 3.2 Quantifiers

**Universal quantifier:** $\forall x \in S,\; P(x)$ — "for all $x$ in $S$, $P(x)$ holds."

**Existential quantifier:** $\exists x \in S,\; P(x)$ — "there exists an $x$ in $S$ such that $P(x)$ holds."

**Example 3.2a:** Let $S = \{2, 4, 6\}$ and $P(x)$ = "$x$ is even."

- $\forall x \in S,\; P(x)$ is **True** — every element (2, 4, 6) is even.
- $\exists x \in S,\; (x > 5)$ is **True** — $6 > 5$ satisfies it.
- $\forall x \in S,\; (x > 3)$ is **False** — $2 \leq 3$ is a counterexample.

**Negation of quantifiers:**

$$\neg(\forall x,\; P(x)) \equiv \exists x,\; \neg P(x)$$
$$\neg(\exists x,\; P(x)) \equiv \forall x,\; \neg P(x)$$

**Example 3.2b:** Negate "every model in the ensemble has accuracy above 80%":

- Original: $\forall m \in \text{Ensemble},\; \text{acc}(m) > 0.8$
- Negation: $\exists m \in \text{Ensemble},\; \text{acc}(m) \leq 0.8$
- In words: "there exists at least one model in the ensemble with accuracy $\leq$ 80%."

To disprove a universal claim, you only need one counterexample.

*ML connection:* The statement "there exists a hypothesis $h$ in class $\mathcal{H}$ with zero training error" is $\exists h \in \mathcal{H},\; \hat{R}(h) = 0$. PAC learning theory makes precise statements about when such hypotheses exist and how they generalize, using exactly this formal language.

---

## 4. Proof Techniques

### 4.1 Direct Proof

To prove $P \Rightarrow Q$: assume $P$ is true and derive $Q$.

**Example.** *Claim:* If $n$ is even, then $n^2$ is even.
*Proof.* Assume $n$ is even. Then $n = 2k$ for some integer $k$. So $n^2 = (2k)^2 = 4k^2 = 2(2k^2)$, which is even. $\square$

### 4.2 Proof by Contradiction

To prove $P$: assume $\neg P$ and derive a contradiction.

**Example.** *Claim:* $\sqrt{2}$ is irrational.
*Proof.* Suppose $\sqrt{2} = p/q$ with $\gcd(p,q)=1$. Then $2q^2 = p^2$, so $p^2$ is even, hence $p$ is even. Write $p = 2k$. Then $2q^2 = 4k^2$, so $q^2 = 2k^2$, hence $q$ is even. But then $\gcd(p,q) \geq 2$, contradicting our assumption. $\square$

### 4.3 Proof by Contrapositive

To prove $P \Rightarrow Q$: prove $\neg Q \Rightarrow \neg P$.

### 4.4 Proof by Induction

To prove $P(n)$ for all $n \geq n_0$:

1. **Base case:** Prove $P(n_0)$.
2. **Inductive step:** Prove $P(k) \Rightarrow P(k+1)$ for arbitrary $k \geq n_0$.

**Example (ML application).** *Claim:* A decision tree of depth $d$ with binary splits can represent at most $2^d$ distinct regions.

*Proof.* Base case: depth 0 is a single leaf → $2^0 = 1$ region. ✓

Inductive step: Assume a tree of depth $k$ has at most $2^k$ regions. A tree of depth $k+1$ has a root split producing two subtrees, each of depth at most $k$. Each subtree has at most $2^k$ regions. Total: $2 \cdot 2^k = 2^{k+1}$. $\square$

---

## 5. Applications in ML/AI

### 5.1 Sets in Probability Theory

All of probability is built on sets. A *probability space* $(\Omega, \mathcal{F}, P)$ consists of:

- $\Omega$: sample space (a set of outcomes)
- $\mathcal{F}$: a $\sigma$-algebra (a collection of subsets of $\Omega$ closed under complement and countable union)
- $P$: a set function $P: \mathcal{F} \to [0,1]$

The axioms of probability are axioms about set functions: $P(\Omega) = 1$, $P(A \cup B) = P(A) + P(B)$ for disjoint $A, B$.

**Example 5.1:** Roll a fair six-sided die. The probability space is:

- $\Omega = \{1, 2, 3, 4, 5, 6\}$ (all outcomes)
- $\mathcal{F} = \mathcal{P}(\Omega)$ (all $2^6 = 64$ subsets are events)
- $P(\{k\}) = 1/6$ for each outcome $k$

Let $A = \{2, 4, 6\}$ ("roll even") and $B = \{1, 2, 3\}$ ("roll $\leq 3$"). Then:

- $P(A) = 3/6 = 1/2$, $P(B) = 3/6 = 1/2$
- $A \cap B = \{2\}$, so $P(A \cap B) = 1/6$
- $P(A \cup B) = P(A) + P(B) - P(A \cap B) = 1/2 + 1/2 - 1/6 = 5/6$ ✓ (matches $|\{1,2,3,4,6\}|/6$)

Every probability calculation reduces to set operations on $\Omega$.

### 5.2 Logic in Neural Network Expressiveness

A single perceptron computes a linear threshold function, which can represent AND, OR, and NOT, but not XOR. This is because XOR requires a non-linearly-separable decision boundary.

```
AND (linearly separable)     XOR (not linearly separable)
    1 ───●───                    1 ───○───●───
         │                            │
    0 ───○───○───                0 ───●───○───
         0   1                        0   1

● = output 1, ○ = output 0
```

Minsky and Papert (1969) proved this limitation formally, leading to the "AI winter." The resolution: multi-layer networks with nonlinear activations can represent any Boolean function.

### 5.3 Partitions in Clustering

$k$-means clustering computes a partition of $\mathbb{R}^n$ into $k$ Voronoi regions. Each region $V_i = \{x \in \mathbb{R}^n : \|x - \mu_i\| \leq \|x - \mu_j\| \text{ for all } j\}$ is a convex polytope.

---

## Exercises

**★ Basic**

1. Let $A = \{1,2,3,4,5\}$ and $B = \{3,4,5,6,7\}$. Compute $A \cup B$, $A \cap B$, $A \setminus B$, and $B \setminus A$.

2. Prove that $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ using element-chasing.

3. Write the negation of: "For every training example, there exists a hypothesis with zero loss."

**★★ Intermediate**

4. Prove that $|A \cup B| = |A| + |B| - |A \cap B|$ for finite sets (inclusion-exclusion for two sets).

5. Show that the set of all binary strings of length $n$ has the same cardinality as $\mathcal{P}(\{1,\ldots,n\})$.

6. A binary classifier on $n$ features partitions $\{0,1\}^n$ into two subsets. How many distinct binary classifiers exist? What does this say about the hypothesis space?

**★★★ Challenging**

7. Prove that $|\mathcal{P}(\mathbb{N})| = |\mathbb{R}|$ (hint: use the Cantor-Bernstein theorem after constructing injections in both directions).

8. Define a $\sigma$-algebra $\mathcal{F}$ over $\Omega = \{H, T\}$ (a coin flip). List all possible $\sigma$-algebras on $\Omega$ and explain which ones are useful for defining probabilities.

---

## Related Topics

- [Number Systems](number-systems.md) — the specific sets $\mathbb{N}, \mathbb{Z}, \mathbb{Q}, \mathbb{R}, \mathbb{C}$
- [Functions & Relations](functions-and-relations.md) — maps between sets
- [Probability Foundations](../probability/probability-foundations.md) — probability as a set function
