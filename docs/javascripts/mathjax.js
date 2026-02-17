// MathJax configuration for mathematical expressions
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset math on page navigation (required for MkDocs Material instant loading)
document$.subscribe(() => {
  MathJax.typesetPromise()
})
