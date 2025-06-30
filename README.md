# My Learning Journey 🚀

A comprehensive documentation site for my learning in Machine Learning, GenAI, System Design, Algorithms, and more.

## 🌟 Features

- **Beautiful Material Design**: Modern, responsive design with dark/light mode
- **Comprehensive Content**: Structured learning materials across multiple domains
- **Interactive Elements**: Code examples, diagrams, and interactive features
- **Search & Navigation**: Fast search and intuitive navigation
- **Mobile Optimized**: Works perfectly on all devices
- **Git Integration**: Shows last updated dates and contributors
- **Math Support**: LaTeX mathematical expressions
- **Syntax Highlighting**: Beautiful code highlighting for multiple languages

## 🏗️ Built With

- [MkDocs](https://www.mkdocs.org/) - Static site generator
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) - Beautiful theme
- [Python](https://www.python.org/) - Backend language
- [GitHub Pages](https://pages.github.com/) - Hosting platform

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nvgr/my-learning.git
   cd my-learning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Serve locally**
   ```bash
   mkdocs serve
   ```

4. **Open in browser**
   Navigate to `http://localhost:8000`

### Building for Production

```bash
mkdocs build
```

The site will be built in the `site/` directory.

## 📁 Project Structure

```
my-learning/
├── docs/                   # Documentation source files
│   ├── index.md           # Homepage
│   ├── ml/                # Machine Learning section
│   ├── genai/             # Generative AI section
│   ├── system-design/     # System Design section
│   ├── algorithms/        # Algorithms section
│   ├── blog/              # Blog posts
│   ├── stylesheets/       # Custom CSS
│   └── javascripts/       # Custom JavaScript
├── mkdocs.yml             # MkDocs configuration
├── requirements.txt       # Python dependencies
└── .github/workflows/     # GitHub Actions for deployment
```

## 🎨 Customization

### Colors and Theme
The site uses a custom color palette defined in `mkdocs.yml`. You can modify:
- Primary color: Currently set to Indigo
- Accent color: Currently set to Purple
- Custom CSS in `docs/stylesheets/extra.css`

### Adding Content
1. Create new markdown files in the appropriate section directory
2. Update the navigation in `mkdocs.yml`
3. Use MkDocs Material features like admonitions, code blocks, and grids

### Custom Styling
- Add custom CSS in `docs/stylesheets/extra.css`
- Add custom JavaScript in `docs/javascripts/`
- Modify theme colors in `mkdocs.yml`

## 🔧 Configuration

Key configuration options in `mkdocs.yml`:

- **Theme**: Material with custom colors and features
- **Extensions**: Python-Markdown extensions for enhanced functionality
- **Plugins**: Git revision dates, minification, search
- **Navigation**: Structured navigation across all sections
- **Social Links**: GitHub, LinkedIn, Twitter integration

## 📝 Writing Guidelines

### Markdown Features
- Use admonitions for tips, warnings, and info boxes
- Include code examples with syntax highlighting
- Add diagrams using Mermaid
- Use grids for card layouts
- Include mathematical expressions with LaTeX

### Content Structure
- Start each section with an overview
- Include practical examples and code snippets
- Add learning resources and references
- Use consistent formatting and structure

## 🚀 Deployment

### GitHub Pages (Recommended)
The site automatically deploys to GitHub Pages when you push to the main branch.

1. Enable GitHub Pages in repository settings
2. Set source to "GitHub Actions"
3. Push to main branch
4. Site will be available at `https://username.github.io/repository-name`

### Manual Deployment
```bash
mkdocs gh-deploy
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `mkdocs serve`
5. Submit a pull request

## 📚 Learning Resources

### MkDocs Material
- [Getting Started](https://squidfunk.github.io/mkdocs-material/getting-started/)
- [Reference](https://squidfunk.github.io/mkdocs-material/reference/)
- [Customization](https://squidfunk.github.io/mkdocs-material/customization/)

### Markdown
- [Markdown Guide](https://www.markdownguide.org/)
- [Python-Markdown Extensions](https://python-markdown.github.io/extensions/)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [MkDocs](https://www.mkdocs.org/) for the excellent static site generator
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for the beautiful theme
- [GitHub](https://github.com/) for hosting and CI/CD

---

**Happy Learning!** 🎓✨

*Built with ❤️ using MkDocs Material*
