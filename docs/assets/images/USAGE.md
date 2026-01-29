# Image Reference Quick Guide

This is a quick reference for how to use images in your documentation from different locations.

## Folder Structure
```
docs/
├── assets/
│   └── images/
│       ├── genai/
│       ├── system-design/
│       ├── algorithms/
│       ├── ml/
│       └── common/
├── genai/
│   └── transformers/
└── system-design/
```

## Reference Patterns

### From any file in `docs/genai/`
```markdown
![Alt text](../assets/images/genai/your-image.png)
```

### From any file in `docs/genai/transformers/`
```markdown
![Alt text](../../assets/images/genai/your-image.png)
```

### From any file in `docs/system-design/`
```markdown
![Alt text](../assets/images/system-design/your-image.png)
```

### From root-level files in `docs/`
```markdown
![Alt text](assets/images/common/your-image.png)
```

## Advanced Image Usage

### Responsive Images with Material Theme
```markdown
<figure markdown>
  ![Architecture Overview](../assets/images/genai/transformer-architecture.png){ width="800" }
  <figcaption>Transformer architecture showing encoder-decoder structure</figcaption>
</figure>
```

### Side-by-side Images
```markdown
<div class="grid" markdown>

<div markdown>
![Before](../assets/images/algorithms/unsorted-array.png)
**Before Sorting**
</div>

<div markdown>
![After](../assets/images/algorithms/sorted-array.png)
**After Sorting**
</div>

</div>
```

### Images in Tabs
```markdown
=== "Concept"
    ![Concept Diagram](../assets/images/genai/attention-concept.png)
    
=== "Implementation"
    ![Code Structure](../assets/images/genai/attention-code.png)
    
=== "Visualization"
    ![Attention Heatmap](../assets/images/genai/attention-viz.png)
```

## Tips

1. **Always use relative paths** starting with `../` or `./`
2. **Test your image links** by building the site locally
3. **Use descriptive filenames** that make sense in context
4. **Keep file sizes reasonable** for web loading
5. **Provide meaningful alt text** for accessibility
