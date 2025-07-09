# De-duplication with Sets

## Introduction to De-duplication

**De-duplication** is the process of identifying and removing duplicate entries from a collection of data. Sets are the ideal data structure for de-duplication because they inherently store only unique elements. De-duplication is a fundamental operation in many computing scenarios, including:

- Data cleaning and preprocessing
- File storage systems
- Network packet processing
- Database operations
- Text processing
- Image processing

## Basic De-duplication Techniques

### Using HashSet for Simple De-duplication

The most straightforward approach to de-duplication is to leverage the set data structure:

```java
import java.util.HashSet;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class BasicDeduplication {
    
    public static <T> List<T> removeDuplicates(List<T> list) {
        // Convert to HashSet (removes duplicates) then back to List
        Set<T> set = new HashSet<>(list);
        return new ArrayList<>(set);
    }
    
    public static void main(String[] args) {
        List<Integer> numbersWithDuplicates = Arrays.asList(1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8);
        List<Integer> uniqueNumbers = removeDuplicates(numbersWithDuplicates);
        
        System.out.println("Original list: " + numbersWithDuplicates);
        System.out.println("After removing duplicates: " + uniqueNumbers);
        // Output:
        // Original list: [1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8]
        // After removing duplicates: [1, 2, 3, 4, 5, 8, 9, 7]
    }
}
```

### Python Implementation

Python provides a concise syntax for de-duplication:

```python
def remove_duplicates(input_list):
    return list(set(input_list))

# Example usage
numbers_with_duplicates = [1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8]
unique_numbers = remove_duplicates(numbers_with_duplicates)

print(f"Original list: {numbers_with_duplicates}")
print(f"After removing duplicates: {unique_numbers}")
# Output:
# Original list: [1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8]
# After removing duplicates: [1, 2, 3, 4, 5, 7, 8, 9]  # Note: order not preserved
```

### Order-Preserving De-duplication

Sometimes you need to maintain the original order of elements while removing duplicates:

```java
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class OrderPreservingDeduplication {
    
    public static <T> List<T> removeDuplicatesPreserveOrder(List<T> list) {
        // LinkedHashSet preserves insertion order
        Set<T> set = new LinkedHashSet<>(list);
        return new ArrayList<>(set);
    }
    
    public static void main(String[] args) {
        List<Integer> numbersWithDuplicates = Arrays.asList(1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8);
        List<Integer> uniqueNumbersOrdered = removeDuplicatesPreserveOrder(numbersWithDuplicates);
        
        System.out.println("Original list: " + numbersWithDuplicates);
        System.out.println("After removing duplicates (order preserved): " + uniqueNumbersOrdered);
        // Output:
        // Original list: [1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8]
        // After removing duplicates (order preserved): [1, 2, 3, 4, 5, 8, 9, 7]
    }
}
```

Python's version using `dict.fromkeys()`:

```python
def remove_duplicates_preserve_order(input_list):
    return list(dict.fromkeys(input_list))

# Example usage
numbers_with_duplicates = [1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8]
unique_numbers_ordered = remove_duplicates_preserve_order(numbers_with_duplicates)

print(f"Original list: {numbers_with_duplicates}")
print(f"After removing duplicates (order preserved): {unique_numbers_ordered}")
# Output:
# Original list: [1, 2, 3, 2, 1, 4, 5, 4, 8, 9, 7, 8]
# After removing duplicates (order preserved): [1, 2, 3, 4, 5, 8, 9, 7]
```

## Advanced De-duplication Techniques

### De-duplication with Custom Objects

When working with custom objects, proper `equals()` and `hashCode()` implementations are crucial:

```java
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

public class CustomObjectDeduplication {
    
    static class Person {
        private String name;
        private int age;
        
        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Person person = (Person) o;
            return age == person.age && Objects.equals(name, person.name);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(name, age);
        }
        
        @Override
        public String toString() {
            return "Person{name='" + name + "', age=" + age + "}";
        }
    }
    
    public static void main(String[] args) {
        List<Person> personList = new ArrayList<>();
        personList.add(new Person("Alice", 30));
        personList.add(new Person("Bob", 25));
        personList.add(new Person("Alice", 30));  // Duplicate
        personList.add(new Person("Charlie", 35));
        personList.add(new Person("Bob", 25));    // Duplicate
        
        // Remove duplicates
        List<Person> uniquePersons = new ArrayList<>(new HashSet<>(personList));
        
        System.out.println("Original list size: " + personList.size());        // 5
        System.out.println("After removing duplicates: " + uniquePersons.size()); // 3
        System.out.println("Unique persons: " + uniquePersons);
    }
}
```

### De-duplication by Specific Fields

Sometimes, you want to de-duplicate based on specific fields rather than the entire object:

```java
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class FieldBasedDeduplication {
    
    static class Employee {
        private String id;
        private String name;
        private String department;
        private double salary;
        
        // Constructor, getters, setters omitted for brevity
        
        @Override
        public String toString() {
            return "Employee{id='" + id + "', name='" + name + 
                   "', department='" + department + "', salary=" + salary + "}";
        }
    }
    
    // De-duplicate by a key extractor function
    public static <T, K> List<T> deduplicateBy(List<T> list, Function<T, K> keyExtractor) {
        return new ArrayList<>(
            list.stream()
                .collect(Collectors.toMap(
                    keyExtractor,
                    Function.identity(),
                    (existing, replacement) -> existing
                ))
                .values()
        );
    }
    
    public static void main(String[] args) {
        List<Employee> employees = new ArrayList<>();
        // Add employees with some duplicates by ID
        
        // De-duplicate by employee ID
        List<Employee> uniqueByIdEmployees = deduplicateBy(employees, Employee::getId);
        
        // De-duplicate by department (keeps one employee per department)
        List<Employee> uniqueByDeptEmployees = deduplicateBy(employees, Employee::getDepartment);
    }
}
```

### Near-Duplicate Detection

Real-world data often contains near-duplicates that require more sophisticated techniques:

```java
import java.util.*;

public class NearDuplicateDetection {
    
    // Calculate Levenshtein distance between two strings
    public static int levenshteinDistance(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i == 0) {
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;
                } else {
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + (s1.charAt(i - 1) == s2.charAt(j - 1) ? 0 : 1),
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                    );
                }
            }
        }
        
        return dp[s1.length()][s2.length()];
    }
    
    private static int min(int a, int b, int c) {
        return Math.min(Math.min(a, b), c);
    }
    
    // Group similar strings (near-duplicates)
    public static Map<String, List<String>> findNearDuplicates(List<String> strings, int maxDistance) {
        Map<String, List<String>> groups = new HashMap<>();
        Set<String> processed = new HashSet<>();
        
        for (int i = 0; i < strings.size(); i++) {
            String s1 = strings.get(i);
            if (processed.contains(s1)) continue;
            
            List<String> group = new ArrayList<>();
            group.add(s1);
            processed.add(s1);
            
            for (int j = i + 1; j < strings.size(); j++) {
                String s2 = strings.get(j);
                if (!processed.contains(s2) && levenshteinDistance(s1, s2) <= maxDistance) {
                    group.add(s2);
                    processed.add(s2);
                }
            }
            
            if (group.size() > 1) {
                groups.put(s1, group);
            }
        }
        
        return groups;
    }
    
    public static void main(String[] args) {
        List<String> addresses = Arrays.asList(
            "123 Main St, Anytown, CA",
            "123 Main Street, Anytown, CA",
            "123 Main St., Anytown, California",
            "456 Oak Ave, Somecity, TX",
            "456 Oak Avenue, Somecity, TX",
            "789 Pine Rd, Otherville, NY"
        );
        
        Map<String, List<String>> similarAddresses = findNearDuplicates(addresses, 5);
        System.out.println("Near-duplicate address groups:");
        similarAddresses.forEach((key, group) -> {
            System.out.println("Group with representative: " + key);
            group.forEach(addr -> System.out.println("  " + addr));
        });
    }
}
```

## Real-World Applications of De-duplication

### File System De-duplication

File systems use de-duplication to save storage space:

```java
import java.io.IOException;
import java.nio.file.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;

public class FileSystemDeduplication {
    
    public static Map<String, Path> findDuplicateFiles(Path directory) throws IOException, NoSuchAlgorithmException {
        Map<String, Path> uniqueFiles = new HashMap<>();
        Map<String, Path> duplicates = new HashMap<>();
        
        Files.walk(directory)
             .filter(Files::isRegularFile)
             .forEach(file -> {
                 try {
                     byte[] fileContent = Files.readAllBytes(file);
                     String hash = calculateSHA256(fileContent);
                     
                     if (uniqueFiles.containsKey(hash)) {
                         duplicates.put(file.toString(), uniqueFiles.get(hash));
                     } else {
                         uniqueFiles.put(hash, file);
                     }
                 } catch (IOException | NoSuchAlgorithmException e) {
                     e.printStackTrace();
                 }
             });
        
        return duplicates;
    }
    
    private static String calculateSHA256(byte[] data) throws NoSuchAlgorithmException {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(data);
        
        StringBuilder hexString = new StringBuilder();
        for (byte b : hash) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) hexString.append('0');
            hexString.append(hex);
        }
        return hexString.toString();
    }
    
    public static void main(String[] args) {
        try {
            Path directory = Paths.get("/path/to/directory");
            Map<String, Path> duplicates = findDuplicateFiles(directory);
            
            System.out.println("Found " + duplicates.size() + " duplicate files:");
            duplicates.forEach((duplicate, original) -> {
                System.out.println(duplicate + " is duplicate of " + original);
            });
            
        } catch (IOException | NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
    }
}
```

### Database De-duplication

De-duplicating database records:

```java
import java.sql.*;
import java.util.*;

public class DatabaseDeduplication {
    
    public static void findAndMarkDuplicates(Connection conn, String tableName, String[] keyColumns) 
            throws SQLException {
        
        // Build SQL to find duplicates
        StringBuilder sql = new StringBuilder();
        sql.append("SELECT ").append(String.join(", ", keyColumns))
           .append(", COUNT(*) as duplicate_count, MIN(id) as keep_id ")
           .append("FROM ").append(tableName)
           .append(" GROUP BY ").append(String.join(", ", keyColumns))
           .append(" HAVING COUNT(*) > 1");
        
        try (Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql.toString())) {
            
            while (rs.next()) {
                int keepId = rs.getInt("keep_id");
                int duplicateCount = rs.getInt("duplicate_count");
                
                // Build WHERE clause for the duplicate set
                StringBuilder whereClause = new StringBuilder();
                for (String column : keyColumns) {
                    if (whereClause.length() > 0) {
                        whereClause.append(" AND ");
                    }
                    
                    Object value = rs.getObject(column);
                    if (value == null) {
                        whereClause.append(column).append(" IS NULL");
                    } else if (value instanceof String) {
                        whereClause.append(column).append(" = '").append(value).append("'");
                    } else {
                        whereClause.append(column).append(" = ").append(value);
                    }
                }
                
                // Mark duplicates (e.g., set is_duplicate flag)
                String updateSql = "UPDATE " + tableName +
                                   " SET is_duplicate = true" +
                                   " WHERE id != " + keepId +
                                   " AND " + whereClause;
                
                try (Statement updateStmt = conn.createStatement()) {
                    int updated = updateStmt.executeUpdate(updateSql);
                    System.out.println("Marked " + updated + " duplicates for record with ID " + keepId);
                }
            }
        }
    }
    
    public static void main(String[] args) {
        try (Connection conn = DriverManager.getConnection("jdbc:postgresql://localhost/mydb", "user", "password")) {
            
            // Find duplicates in customers table based on email and phone
            String[] keyColumns = {"email", "phone"};
            findAndMarkDuplicates(conn, "customers", keyColumns);
            
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```

### Text Processing De-duplication

Removing duplicate words from text:

```java
import java.util.*;

public class TextDeduplication {
    
    public static String removeDuplicateWords(String text) {
        String[] words = text.split("\\s+");
        Set<String> uniqueWords = new LinkedHashSet<>(Arrays.asList(words));
        return String.join(" ", uniqueWords);
    }
    
    public static List<String> findDuplicateWords(String text) {
        String[] words = text.split("\\s+");
        Set<String> uniqueWords = new HashSet<>();
        Set<String> duplicates = new HashSet<>();
        
        for (String word : words) {
            if (!uniqueWords.add(word)) {
                duplicates.add(word);
            }
        }
        
        return new ArrayList<>(duplicates);
    }
    
    public static void main(String[] args) {
        String text = "the quick brown fox jumps over the lazy dog the fox was quick and the dog was lazy";
        
        System.out.println("Original text: " + text);
        System.out.println("Text with duplicates removed: " + removeDuplicateWords(text));
        System.out.println("Duplicate words: " + findDuplicateWords(text));
    }
}
```

## Time and Space Complexity Analysis

### Time Complexity

- **Basic De-duplication**: O(n) where n is the number of elements
- **Order-Preserving De-duplication**: O(n)
- **Custom Object De-duplication**: O(n), assuming good hash function distribution
- **Field-Based De-duplication**: O(n)
- **Near-Duplicate Detection**: O(n²) for pairwise comparison

### Space Complexity

- **Basic De-duplication**: O(k) where k is the number of unique elements
- **Order-Preserving De-duplication**: O(k)
- **Custom Object De-duplication**: O(k)
- **Field-Based De-duplication**: O(k)
- **Near-Duplicate Detection**: O(n) for storing results

## Best Practices for Efficient De-duplication

1. **Choose the Right Set Implementation**:
   - `HashSet`: Fastest lookup for most cases
   - `LinkedHashSet`: When order needs to be preserved
   - `TreeSet`: When sorted order is required

2. **Proper Hash Function Implementation**:
   - Ensure good distribution
   - Override `equals()` and `hashCode()` correctly
   - Consider all relevant fields for equality comparison

3. **Memory Efficiency**:
   - For large datasets, consider streaming approaches
   - Process data in chunks when possible
   - Use specialized algorithms for near-duplicate detection

4. **Performance Optimizations**:
   - Pre-size collections when the input size is known
   - Filter obvious non-duplicates first
   - Use parallel processing for large datasets

## Code Example: Streaming De-duplication for Large Datasets

```java
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

public class LargeDatasetDeduplication {
    
    public static void deduplicateLargeFile(Path inputFile, Path outputFile) throws IOException {
        Set<String> seenLines = new HashSet<>();
        
        try (BufferedReader reader = Files.newBufferedReader(inputFile);
             BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            
            String line;
            while ((line = reader.readLine()) != null) {
                if (seenLines.add(line)) {
                    // This is a new line, add it to the output
                    writer.write(line);
                    writer.newLine();
                }
            }
        }
    }
    
    // Stream-based approach for Java 8+
    public static void streamDeduplicateLargeFile(Path inputFile, Path outputFile) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            Files.lines(inputFile)
                 .distinct()
                 .forEach(line -> {
                     try {
                         writer.write(line);
                         writer.newLine();
                     } catch (IOException e) {
                         throw new UncheckedIOException(e);
                     }
                 });
        }
    }
    
    // Chunked processing for very large files
    public static void chunkedDeduplication(Path inputFile, Path outputFile, int chunkSize) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(inputFile);
             BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            
            Set<String> uniqueLines = new HashSet<>(chunkSize);
            String line;
            int count = 0;
            
            while ((line = reader.readLine()) != null) {
                if (uniqueLines.add(line)) {
                    writer.write(line);
                    writer.newLine();
                }
                
                count++;
                if (count % chunkSize == 0) {
                    // Process in chunks to avoid OutOfMemoryError
                    System.out.println("Processed " + count + " lines");
                    // Clear the set if necessary (if memory is a concern)
                    // uniqueLines.clear();
                }
            }
        }
    }
    
    public static void main(String[] args) {
        try {
            Path inputFile = Paths.get("large_input.txt");
            Path outputFile = Paths.get("deduplicated_output.txt");
            
            // Choose one method based on your needs:
            // deduplicateLargeFile(inputFile, outputFile);
            // streamDeduplicateLargeFile(inputFile, outputFile);
            chunkedDeduplication(inputFile, outputFile, 1000000);
            
            System.out.println("De-duplication completed!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

## Conclusion

De-duplication is a powerful application of sets that finds use across many domains in computer science. By removing redundant data, de-duplication can:

1. **Improve Data Quality**: Eliminate inconsistencies and duplicates
2. **Save Storage Space**: Reduce the footprint of redundant data
3. **Enhance Performance**: Speed up processing by reducing data volume
4. **Improve Analytics**: Ensure accurate counting and analysis

The choice of de-duplication technique depends on the specific requirements of your application, including:

- Whether order preservation is necessary
- Memory constraints
- Performance requirements
- Need for exact vs. approximate de-duplication

By understanding the principles and techniques of de-duplication, you can implement efficient solutions for handling duplicate data in your applications.
