# Fractional Knapsack

## Overview

The Fractional Knapsack problem is a classic optimization problem in which we need to fill a knapsack with items to maximize the total value, where we can take fractions of items instead of having to take the complete items.

## Problem Statement

Given a set of items, each with a weight and a value, determine the number of each item to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible. In the Fractional Knapsack problem, we can take fractions of items, meaning we can break items for maximizing the total value of the knapsack.

## Algorithm

1. Calculate the value-to-weight ratio for each item
2. Sort items based on this ratio in non-increasing order
3. Initialize current weight and value to 0
4. Iterate through the sorted items:
   - If adding the entire current item doesn't exceed the capacity, add it completely
   - Otherwise, add as much of the item as possible to fill the knapsack
5. Return the maximum value

## Implementation

### Python Implementation

```python
def fractional_knapsack(values, weights, capacity):
    """
    Solves the Fractional Knapsack problem.
    
    Args:
        values: List of values of the items
        weights: List of weights of the items
        capacity: Maximum capacity of the knapsack
        
    Returns:
        Maximum value that can be obtained
    """
    # Create a list of (value, weight, index) tuples
    items = [(values[i], weights[i], i) for i in range(len(values))]
    
    # Sort by value-to-weight ratio in non-increasing order
    items.sort(key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    
    selected_items = []  # To keep track of selected items and their fractions
    
    for value, weight, index in items:
        if remaining_capacity >= weight:
            # Take the whole item
            selected_items.append((index, 1.0))  # Index and fraction
            total_value += value
            remaining_capacity -= weight
        else:
            # Take a fraction of the item
            fraction = remaining_capacity / weight
            selected_items.append((index, fraction))
            total_value += value * fraction
            break  # The knapsack is full
    
    return total_value, selected_items

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

max_value, selected = fractional_knapsack(values, weights, capacity)
print(f"Maximum value: {max_value}")
print("Selected items (index, fraction):")
for index, fraction in selected:
    print(f"Item {index}: {fraction * 100:.2f}%")
```

### Java Implementation

```java
import java.util.Arrays;
import java.util.Comparator;

public class FractionalKnapsack {
    
    static class Item {
        int value;
        int weight;
        int index;
        
        public Item(int value, int weight, int index) {
            this.value = value;
            this.weight = weight;
            this.index = index;
        }
    }
    
    static class ItemSelection {
        int index;
        double fraction;
        
        public ItemSelection(int index, double fraction) {
            this.index = index;
            this.fraction = fraction;
        }
    }
    
    public static double fractionalKnapsack(int[] values, int[] weights, int capacity) {
        int n = values.length;
        Item[] items = new Item[n];
        
        // Create items
        for (int i = 0; i < n; i++) {
            items[i] = new Item(values[i], weights[i], i);
        }
        
        // Sort by value-to-weight ratio (non-increasing)
        Arrays.sort(items, (a, b) -> Double.compare(
            (double) b.value / b.weight, 
            (double) a.value / a.weight)
        );
        
        double totalValue = 0;
        int remainingCapacity = capacity;
        ItemSelection[] selection = new ItemSelection[n];
        int selectionCount = 0;
        
        for (Item item : items) {
            if (remainingCapacity >= item.weight) {
                // Take the whole item
                selection[selectionCount++] = new ItemSelection(item.index, 1.0);
                totalValue += item.value;
                remainingCapacity -= item.weight;
            } else {
                // Take a fraction of the item
                double fraction = (double) remainingCapacity / item.weight;
                selection[selectionCount++] = new ItemSelection(item.index, fraction);
                totalValue += item.value * fraction;
                break;  // The knapsack is full
            }
        }
        
        // Print the selected items (just for demonstration)
        System.out.println("Selected items (index, fraction):");
        for (int i = 0; i < selectionCount; i++) {
            System.out.printf("Item %d: %.2f%%\n", 
                selection[i].index, selection[i].fraction * 100);
        }
        
        return totalValue;
    }
    
    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int capacity = 50;
        
        double maxValue = fractionalKnapsack(values, weights, capacity);
        System.out.println("Maximum value: " + maxValue);
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(n log n), where n is the number of items (due to sorting)
- **Space Complexity**: O(n) to store the items

## Proof of Correctness

The greedy strategy works for Fractional Knapsack because:

1. By always selecting items with the highest value-to-weight ratio first, we maximize the total value per unit of weight used
2. Since we can take fractions of items, we can always use the full capacity of the knapsack
3. This approach guarantees the optimal solution because we're always making the locally optimal choice at each step

## Difference from 0/1 Knapsack

The key difference from the 0/1 Knapsack problem is:

- In Fractional Knapsack, we can take a fraction of an item, resulting in a greedy solution
- In 0/1 Knapsack, we can either take an item completely or not take it at all, requiring dynamic programming

## Applications

1. **Resource Allocation**: Distributing limited resources to maximize utility
2. **Portfolio Optimization**: Allocating investments to maximize returns
3. **Cargo Loading**: Loading items into a vehicle with limited capacity
4. **Budget Allocation**: Distributing budget among different projects

## Variations

1. **Multiple Knapsack Problem**: Multiple knapsacks with different capacities
2. **Bounded Knapsack Problem**: Multiple instances of each item, but with limits
3. **Multi-objective Knapsack Problem**: Optimize for multiple objectives
4. **Online Knapsack Problem**: Items arrive one by one, and decisions must be made immediately

## Practice Problems

1. [Fractional Knapsack](https://practice.geeksforgeeks.org/problems/fractional-knapsack-1587115620/1) - Standard problem implementation
2. [Maximum Units on a Truck](https://leetcode.com/problems/maximum-units-on-a-truck/) - A variation of the Fractional Knapsack problem
3. [IPL Match Tickets](https://www.codingninjas.com/codestudio/problems/ipl-match-tickets_8230728) - A practical application
4. [Minimum Cost to Hire K Workers](https://leetcode.com/problems/minimum-cost-to-hire-k-workers/) - A related problem using similar concepts

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
2. Kleinberg, J., & Tardos, É. (2005). Algorithm Design. Addison-Wesley.
3. Dasgupta, S., Papadimitriou, C. H., & Vazirani, U. V. (2006). Algorithms. McGraw-Hill Education.
