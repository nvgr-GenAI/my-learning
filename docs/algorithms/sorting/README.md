# Sorting Algorithms Documentation

This directory contains documentation for various sorting algorithms, organized in a structured format for easy learning and reference.

## Directory Structure

```plaintext
algorithms/sorting/
├── index.md                # Overview of all sorting algorithms
├── fundamentals.md         # Core sorting concepts and properties
├── Basic Sorting/
│   ├── bubble-sort.md      # Bubble Sort implementation and analysis
│   ├── selection-sort.md   # Selection Sort implementation and analysis
│   ├── insertion-sort.md   # Insertion Sort implementation and analysis
├── Advanced Sorting/
│   ├── quick-sort.md       # Quick Sort implementation and analysis
│   ├── merge-sort.md       # Merge Sort implementation and analysis
│   ├── heap-sort.md        # Heap Sort implementation and analysis
├── Linear-time Sorting/
│   ├── counting-sort.md    # Counting Sort implementation and analysis
│   ├── radix-sort.md       # Radix Sort implementation and analysis
│   ├── bucket-sort.md      # Bucket Sort implementation and analysis
├── Hybrid Algorithms/
│   ├── timsort.md          # Timsort implementation and analysis
│   ├── introsort.md        # Introsort implementation and analysis
├── Practice Problems/
│   ├── easy-problems.md    # Easy sorting problems with solutions
│   ├── medium-problems.md  # Medium sorting problems with solutions
│   ├── hard-problems.md    # Hard sorting problems with solutions
```

## Documentation Style

Each algorithm documentation follows a consistent tabbed structure for better organization:

1. **📋 Algorithm Overview** - Key characteristics, when to use, advantages/disadvantages
2. **🔄 How It Works** - Step-by-step explanation of the algorithm with visual examples
3. **💻 Implementation** - Code samples in multiple languages or variations
4. **📊 Performance Analysis** - Time/space complexity analysis and comparison with other algorithms
5. **💡 Applications & Variations** - Real-world uses and algorithm variations
6. **🎯 Practice Problems** - Problems to solve using the algorithm

## Contribution Guidelines

When adding or editing documentation for sorting algorithms:

1. **Maintain the tabbed structure** for consistency across all algorithm pages.
2. **Include visual examples** to help readers understand the algorithm steps.
3. **Provide performance analysis** with best, average, and worst-case scenarios.
4. **Add practical code implementations** that are well-commented and efficient.
5. **Include practice problems** with varying difficulty levels.
6. **Keep the navigation structure** in `mkdocs.yml` updated when adding new algorithms.

## To-Do List

- [ ] Add visualizations to all sorting algorithm pages
- [ ] Complete the analysis sections for all algorithms
- [ ] Standardize code examples across all pages
- [ ] Add more practice problems with solutions
- [ ] Create comparison charts between different algorithms
- [ ] Add interactive examples if possible

## Reference Style Guide

### Code Blocks

Use language-specific code blocks with proper syntax highlighting:

```python
def example_sort(arr):
    # Implementation
    return sorted_array
```

### Complexity Tables

Use consistent formatting for complexity tables:

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable | In-Place |
|-----------|-----------|--------------|------------|-------|--------|----------|
| Example Sort | O(n) | O(n log n) | O(n²) | O(n) | Yes | No |

### Cross-References

Link to other relevant algorithms or concepts:

- See also: [Merge Sort](merge-sort.md) for a comparison with another divide-and-conquer approach.
