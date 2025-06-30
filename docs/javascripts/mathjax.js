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

// Custom JavaScript for enhanced functionality
document.addEventListener('DOMContentLoaded', function() {
  // Add smooth scrolling to internal links
  const links = document.querySelectorAll('a[href^="#"]');
  links.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add copy functionality to code blocks
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach(code => {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'Copy';
    button.addEventListener('click', () => {
      navigator.clipboard.writeText(code.textContent).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => {
          button.textContent = 'Copy';
        }, 2000);
      });
    });
    code.parentNode.appendChild(button);
  });

  // Progress tracking for learning paths
  const progressBars = document.querySelectorAll('.progress-bar');
  progressBars.forEach(bar => {
    const fill = bar.querySelector('.progress-fill');
    const percentage = fill.getAttribute('data-progress') || '0';
    setTimeout(() => {
      fill.style.width = percentage + '%';
    }, 500);
  });

  // Add reading time estimation
  const articles = document.querySelectorAll('article, .md-content__inner');
  articles.forEach(article => {
    const text = article.textContent || article.innerText || '';
    const words = text.trim().split(/\s+/).length;
    const readingTime = Math.ceil(words / 200); // Average reading speed
    
    if (words > 100) { // Only show for longer content
      const readingTimeElement = document.createElement('div');
      readingTimeElement.className = 'reading-time';
      readingTimeElement.innerHTML = `<i class="material-icons">schedule</i> ${readingTime} min read`;
      
      const firstHeading = article.querySelector('h1, h2');
      if (firstHeading && firstHeading.nextSibling) {
        firstHeading.parentNode.insertBefore(readingTimeElement, firstHeading.nextSibling);
      }
    }
  });
});

// Analytics tracking for learning progress (optional)
function trackLearningProgress(section, topic) {
  if (typeof gtag !== 'undefined') {
    gtag('event', 'learning_progress', {
      'section': section,
      'topic': topic,
      'timestamp': new Date().toISOString()
    });
  }
  
  // Store in localStorage for offline tracking
  const progress = JSON.parse(localStorage.getItem('learning_progress') || '{}');
  if (!progress[section]) progress[section] = [];
  if (!progress[section].includes(topic)) {
    progress[section].push(topic);
    localStorage.setItem('learning_progress', JSON.stringify(progress));
  }
}
