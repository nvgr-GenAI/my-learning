# String Parsing

## Overview

String parsing is the process of analyzing a string of text to extract meaningful information according to a set of rules or grammar. It is a fundamental operation in programming, used in everything from processing user inputs to analyzing complex data formats like JSON, XML, or CSV.

Effective string parsing combines understanding of data formats, efficient algorithms, and error handling strategies to reliably extract structured data from unstructured or semi-structured text.

## Common Parsing Techniques

### 1. Character-by-Character Parsing

The most basic approach involves iterating through the string one character at a time, building up parsed components.

```python
def parse_csv_line(line):
    result = []
    current_field = ""
    in_quotes = False
    
    for char in line:
        if char == ',' and not in_quotes:
            result.append(current_field)
            current_field = ""
        elif char == '"':
            in_quotes = not in_quotes
        else:
            current_field += char
            
    result.append(current_field)  # Add the last field
    return result
```

**Time Complexity**: O(n) where n is the length of the string
**Space Complexity**: O(n) for storing the parsed result

### 2. Split and Join

Many languages provide built-in functions to split strings by delimiters and join them back together.

```python
# Simple CSV parsing (doesn't handle quoted fields correctly)
def simple_csv_parse(line):
    return line.split(',')

# Joining elements with a delimiter
def join_elements(elements, delimiter):
    return delimiter.join(elements)
```

**Time Complexity**: O(n) where n is the length of the string
**Space Complexity**: O(n) for storing the results

### 3. Regular Expressions

Regular expressions provide powerful pattern matching capabilities for more complex parsing needs.

```python
import re

def extract_email_addresses(text):
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

def parse_date_formats(text):
    # Match dates in format YYYY-MM-DD
    pattern = r'(\d{4})-(\d{2})-(\d{2})'
    matches = re.findall(pattern, text)
    return [(int(year), int(month), int(day)) for year, month, day in matches]
```

**Time Complexity**: Varies but generally O(n) for simple patterns
**Space Complexity**: O(n) for storing matches

### 4. State Machines

For more complex parsing, state machines provide a structured approach by defining states and transitions.

```python
def parse_json_string(s):
    # A simplified example - real JSON parsing is more complex
    result = ""
    state = 'START'
    i = 0
    
    while i < len(s):
        char = s[i]
        
        if state == 'START':
            if char == '"':
                state = 'IN_STRING'
            i += 1
        
        elif state == 'IN_STRING':
            if char == '\\':
                state = 'ESCAPE'
            elif char == '"':
                state = 'END'
            else:
                result += char
            i += 1
        
        elif state == 'ESCAPE':
            if char in '"\\bfnrt':
                # Handle escape sequences
                if char == 'b': result += '\b'
                elif char == 'f': result += '\f'
                elif char == 'n': result += '\n'
                elif char == 'r': result += '\r'
                elif char == 't': result += '\t'
                else: result += char  # For " and \
            state = 'IN_STRING'
            i += 1
        
        elif state == 'END':
            break
    
    return result
```

**Time Complexity**: O(n) where n is the length of the string
**Space Complexity**: O(n) for storing the result

### 5. Recursive Descent Parsing

For structured formats with nested elements (like JSON or programming languages), recursive descent parsing is often used.

```python
class SimpleJSONParser:
    def __init__(self, text):
        self.text = text
        self.pos = 0
    
    def parse(self):
        result = self.parse_value()
        return result
    
    def parse_value(self):
        self.skip_whitespace()
        
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of input")
            
        current = self.text[self.pos]
        
        if current == '{':
            return self.parse_object()
        elif current == '[':
            return self.parse_array()
        elif current == '"':
            return self.parse_string()
        elif current in '0123456789-':
            return self.parse_number()
        elif current == 't' and self.text[self.pos:self.pos+4] == 'true':
            self.pos += 4
            return True
        elif current == 'f' and self.text[self.pos:self.pos+5] == 'false':
            self.pos += 5
            return False
        elif current == 'n' and self.text[self.pos:self.pos+4] == 'null':
            self.pos += 4
            return None
        else:
            raise ValueError(f"Unexpected character at position {self.pos}")
    
    # Additional methods for parse_object, parse_array, parse_string, parse_number...
    # and utility methods like skip_whitespace would be defined here
```

**Time Complexity**: O(n) where n is the length of the string
**Space Complexity**: O(d) where d is the maximum depth of the nested structure

## Specialized Parsing

### 1. Tokenization

Breaking text into meaningful units (tokens) is often the first step in parsing.

```python
def tokenize_expression(expression):
    tokens = []
    i = 0
    
    while i < len(expression):
        char = expression[i]
        
        if char.isdigit():
            # Parse number
            j = i
            while j < len(expression) and expression[j].isdigit():
                j += 1
            tokens.append(("NUMBER", int(expression[i:j])))
            i = j
        
        elif char in "+-*/()":
            tokens.append(("OPERATOR", char))
            i += 1
        
        elif char.isspace():
            i += 1
        
        else:
            raise ValueError(f"Invalid character: {char}")
    
    return tokens
```

### 2. Parsing Structured Data

Handling common data formats like JSON, XML, CSV requires specialized parsing approaches.

For JSON:
```python
import json

def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
```

For XML:
```python
import xml.etree.ElementTree as ET

def parse_xml(text):
    try:
        root = ET.fromstring(text)
        return root
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None
```

For CSV:
```python
import csv
from io import StringIO

def parse_csv(text):
    result = []
    csv_reader = csv.reader(StringIO(text))
    for row in csv_reader:
        result.append(row)
    return result
```

## Common Parsing Challenges

### 1. Error Handling

Robust parsers need to handle invalid inputs gracefully:

```python
def safe_parse_int(s):
    try:
        return int(s)
    except ValueError:
        return None
```

### 2. Performance Optimization

For large strings, performance considerations become important:

- **Avoid string concatenation** in loops (use join on a list instead)
- **Buffer management** for reading large files
- **Lazy parsing** for on-demand processing

### 3. Encoding Issues

Different character encodings can cause parsing problems:

```python
def parse_file_with_encoding(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"Failed to decode with {encoding}, trying fallback encodings")
        for fallback in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=fallback) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode file with any encoding")
```

## Applications

1. **Compilers and Interpreters**: Parsing source code into abstract syntax trees
2. **Data Processing**: Extracting structured data from text files
3. **Natural Language Processing**: Analyzing grammatical structure of text
4. **Configuration Files**: Reading settings from structured formats
5. **Web Scraping**: Extracting information from HTML pages
6. **Command Line Interfaces**: Parsing user commands and arguments

## Best Practices

1. **Use Existing Libraries** when possible for standard formats
2. **Validate Input** before parsing to catch obvious errors early
3. **Include Robust Error Handling** with descriptive messages
4. **Test with Edge Cases** including empty strings and invalid formats
5. **Balance Performance and Readability** based on your application's needs

## Related Topics

- [Regular Expressions](../advanced/regex.md)
- [Lexical Analysis and Parsing](../advanced/lexical-analysis.md)
- [String Pattern Matching](pattern-matching.md)
- [String Comparison](comparison.md)
