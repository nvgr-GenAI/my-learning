# Activity Selection Problem

## Overview

The Activity Selection Problem is a fundamental greedy algorithm that involves selecting the maximum number of activities that can be performed by a single person, given the start and finish time of each activity.

## Problem Statement

Given a set of activities with their start time and finish time, select the maximum number of activities that can be performed by a single person, assuming the person can only work on a single activity at a time.

## Algorithm

1. Sort the activities according to their finish time
2. Select the first activity (the one that finishes earliest)
3. For the remaining activities, select an activity if its start time is greater than or equal to the finish time of the previously selected activity

## Implementation

### Python Implementation

```python
def activity_selection(start, finish):
    """
    Returns the maximum number of activities that can be performed
    by a single person.
    
    Args:
        start: List of start times
        finish: List of finish times
        
    Returns:
        List of indices of selected activities
    """
    # Create a list of activities with (start time, finish time, original index)
    activities = [(start[i], finish[i], i) for i in range(len(start))]
    
    # Sort activities based on finish time
    activities.sort(key=lambda x: x[1])
    
    # Select the first activity
    selected = [activities[0][2]]
    last_finish_time = activities[0][1]
    
    # Consider all remaining activities
    for i in range(1, len(activities)):
        # If this activity has start time greater than or equal to the finish
        # time of previously selected activity, then select it
        if activities[i][0] >= last_finish_time:
            selected.append(activities[i][2])
            last_finish_time = activities[i][1]
            
    return selected

# Example usage
start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
selected_activities = activity_selection(start, finish)
print(f"Selected activities: {selected_activities}")
print(f"Maximum number of activities: {len(selected_activities)}")
```

### Java Implementation

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class ActivitySelection {
    
    static class Activity {
        int start;
        int finish;
        int index;
        
        public Activity(int start, int finish, int index) {
            this.start = start;
            this.finish = finish;
            this.index = index;
        }
    }
    
    public static List<Integer> activitySelection(int[] start, int[] finish) {
        int n = start.length;
        Activity[] activities = new Activity[n];
        
        // Create activity objects
        for (int i = 0; i < n; i++) {
            activities[i] = new Activity(start[i], finish[i], i);
        }
        
        // Sort activities based on finish time
        Arrays.sort(activities, Comparator.comparingInt(a -> a.finish));
        
        List<Integer> selected = new ArrayList<>();
        
        // Select first activity
        selected.add(activities[0].index);
        int lastFinishTime = activities[0].finish;
        
        // Consider all remaining activities
        for (int i = 1; i < n; i++) {
            // If this activity has start time greater than or equal to the finish
            // time of previously selected activity, then select it
            if (activities[i].start >= lastFinishTime) {
                selected.add(activities[i].index);
                lastFinishTime = activities[i].finish;
            }
        }
        
        return selected;
    }
    
    public static void main(String[] args) {
        int[] start = {1, 3, 0, 5, 8, 5};
        int[] finish = {2, 4, 6, 7, 9, 9};
        
        List<Integer> selectedActivities = activitySelection(start, finish);
        
        System.out.println("Selected activities: " + selectedActivities);
        System.out.println("Maximum number of activities: " + selectedActivities.size());
    }
}
```

## Complexity Analysis

- **Time Complexity**: O(n log n), where n is the number of activities (due to sorting)
- **Space Complexity**: O(n) to store the activities

## Proof of Correctness

The greedy choice property holds for the Activity Selection Problem:
1. Sort activities by finish time
2. Always select the activity that finishes earliest and doesn't conflict with previously selected activities

This approach guarantees the optimal solution because:
- By always selecting the activity that finishes earliest, we maximize the remaining time for other activities
- This ensures we can fit in the maximum possible number of activities

## Applications

1. **Resource Allocation**: Scheduling tasks on a single resource
2. **Meeting Room Scheduling**: Maximizing the number of meetings in a single room
3. **Job Scheduling**: When jobs have deadlines
4. **Transportation and Logistics**: Scheduling delivery vehicles

## Variations

1. **Weighted Activity Selection**: Each activity has a value/profit, and we want to maximize the total value
2. **Multiple Resources**: Schedule activities across multiple resources
3. **Activity Selection with Deadlines**: Activities must be completed before their deadlines
4. **Interval Scheduling with Variable Start Times**: Activities can start at different times

## Related Problems

1. **Interval Scheduling**: Schedule a set of intervals on minimum number of machines
2. **Interval Partitioning**: Partition a set of intervals into minimum number of groups such that no two intervals in a group overlap
3. **Minimum Number of Platforms**: Find minimum number of railway platforms needed at a station

## Practice Problems

1. [Activity Selection](https://practice.geeksforgeeks.org/problems/n-meetings-in-one-room-1587115620/1) - Schedule maximum number of meetings in a room
2. [Job Sequencing Problem](https://practice.geeksforgeeks.org/problems/job-sequencing-problem-1587115620/1) - Schedule jobs to maximize profit
3. [Maximum Meetings in One Room](https://www.codingninjas.com/codestudio/problems/maximum-meetings_1062658) - Find maximum number of meetings that can be accommodated in a room
4. [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) - Find the minimum number of intervals to remove to make the rest non-overlapping

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
2. Kleinberg, J., & Tardos, É. (2005). Algorithm Design. Addison-Wesley.
