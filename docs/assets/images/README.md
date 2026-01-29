# Images Directory

This directory contains all images used throughout the documentation site.

## Directory Structure

```
docs/assets/images/
├── genai/           # GenAI-related images
├── system-design/   # System design diagrams and images
├── algorithms/      # Algorithm visualizations and diagrams
├── ml/             # Machine learning diagrams and images
└── common/         # Shared/common images used across sections
```

## Image Usage Guidelines

### File Naming Convention
- Use descriptive, kebab-case names: `transformer-architecture.png`
- Include the concept being illustrated: `attention-mechanism-diagram.svg`
- For sequential images: `step-1-tokenization.png`, `step-2-embedding.png`

### Supported Formats
- **PNG**: For screenshots, detailed diagrams with text
- **SVG**: For scalable vector graphics, simple diagrams
- **JPG**: For photographs or complex images where file size matters
- **WebP**: For optimized web images (preferred when supported)

### File Size Guidelines
- Keep images under 1MB when possible
- Use appropriate compression for the content
- Consider using SVG for diagrams that can be vectorized

## How to Reference Images in Markdown

### Basic Image Reference
```markdown
![Alt text](../assets/images/genai/transformer-architecture.png)
```

### Image with Caption
```markdown
<figure markdown>
  ![Transformer Architecture](../assets/images/genai/transformer-architecture.png)
  <figcaption>The transformer architecture showing encoder and decoder stacks</figcaption>
</figure>
```

### Responsive Image with Size Control
```markdown
<img src="../assets/images/genai/attention-mechanism.png" alt="Attention Mechanism" style="width: 80%; max-width: 600px;">
```

### Image with Link
```markdown
[![Architecture Diagram](../assets/images/system-design/microservices.png)](../assets/images/system-design/microservices.png)
```

## Path Examples from Different Locations

### From Root Documents (docs/*.md)
```markdown
![Image](assets/images/genai/example.png)
```

### From GenAI Section (docs/genai/*.md)
```markdown
![Image](../assets/images/genai/example.png)
```

### From Deep Nested Files (docs/genai/transformers/*.md)
```markdown
![Image](../../assets/images/genai/example.png)
```

### From System Design Section (docs/system-design/*.md)
```markdown
![Image](../assets/images/system-design/example.png)
```

## Best Practices

1. **Alt Text**: Always provide meaningful alt text for accessibility
2. **File Organization**: Keep images organized by topic in appropriate subdirectories
3. **Optimization**: Optimize images for web delivery
4. **Backup**: Consider the images part of your documentation source code
5. **Attribution**: Include attribution for images from external sources
6. **Consistency**: Use consistent styling and formatting across similar images

## Image Creation Tools

### Recommended Tools for Diagrams
- **Draw.io (diagrams.net)**: Free online diagramming tool
- **Lucidchart**: Professional diagramming with collaboration
- **Figma**: For UI mockups and modern diagrams
- **Canva**: For quick, polished graphics
- **Mermaid**: Text-based diagramming (already integrated in MkDocs)

### For Screenshots
- **Snagit**: Professional screenshot tool
- **CleanShot X** (macOS): Advanced screenshot features
- **Greenshot** (Windows): Open-source screenshot tool
- **Built-in tools**: macOS Screenshot, Windows Snipping Tool

### For Vector Graphics
- **Inkscape**: Free vector graphics editor
- **Adobe Illustrator**: Professional vector graphics
- **Sketch** (macOS): UI/UX design tool

## Examples of Good Image Usage

### Architecture Diagrams
```markdown
![System Architecture](../assets/images/system-design/microservices-architecture.png)
*Figure 1: Microservices architecture showing service boundaries and communication patterns*
```

### Algorithm Visualizations
```markdown
<div class="grid" markdown>
![Before Sorting](../assets/images/algorithms/before-sort.png){ width="45%" }
![After Sorting](../assets/images/algorithms/after-sort.png){ width="45%" }
</div>
*Visualization of array before and after sorting*
```

### Step-by-Step Processes
```markdown
=== "Step 1: Tokenization"
    ![Tokenization](../assets/images/genai/step-1-tokenization.png)
    Text is broken down into individual tokens.

=== "Step 2: Embedding"
    ![Embedding](../assets/images/genai/step-2-embedding.png)
    Tokens are converted to vector representations.

=== "Step 3: Attention"
    ![Attention](../assets/images/genai/step-3-attention.png)
    Attention weights are calculated between tokens.
```
