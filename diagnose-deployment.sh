#!/bin/bash

# Diagnostic script to check MkDocs build for GitHub Pages deployment issues

echo "=== MkDocs GitHub Pages Deployment Diagnostics ==="
echo ""

echo "1. Checking site structure..."
if [ -d "site" ]; then
    echo "✓ Site directory exists"
    echo "   Site contents:"
    ls -la site/ | head -10
else
    echo "✗ Site directory missing - run 'mkdocs build' first"
    exit 1
fi

echo ""
echo "2. Checking for CSS files..."
css_files=$(find site/ -name "*.css" | wc -l)
if [ $css_files -gt 0 ]; then
    echo "✓ Found $css_files CSS files"
    find site/ -name "*.css" | head -5
else
    echo "✗ No CSS files found"
fi

echo ""
echo "3. Checking for JavaScript files..."
js_files=$(find site/ -name "*.js" | wc -l)
if [ $js_files -gt 0 ]; then
    echo "✓ Found $js_files JavaScript files"
    find site/ -name "*.js" | head -5
else
    echo "✗ No JavaScript files found"
fi

echo ""
echo "4. Checking index.html for asset paths..."
if [ -f "site/index.html" ]; then
    echo "✓ index.html exists"
    echo "   CSS links found:"
    grep -o '<link[^>]*\.css[^>]*>' site/index.html | head -3
    echo "   JS script tags found:"
    grep -o '<script[^>]*\.js[^>]*>' site/index.html | head -3
else
    echo "✗ index.html not found"
fi

echo ""
echo "5. Checking base URL configuration..."
if grep -q "site_url:" mkdocs.yml; then
    site_url=$(grep "site_url:" mkdocs.yml)
    echo "✓ Site URL configured: $site_url"
else
    echo "✗ Site URL not configured in mkdocs.yml"
fi

echo ""
echo "6. Checking for GitHub Pages specific issues..."
echo "   Repository info:"
if grep -q "repo_name:" mkdocs.yml; then
    repo_name=$(grep "repo_name:" mkdocs.yml)
    echo "   $repo_name"
else
    echo "   ✗ repo_name not found"
fi

echo ""
echo "=== Recommendations ==="
echo "If CSS/JS are not loading on GitHub Pages:"
echo "1. Verify the site_url matches your GitHub Pages URL exactly"
echo "2. Check that the repository name in mkdocs.yml matches your actual repo"
echo "3. Ensure GitHub Pages is enabled for your repository"
echo "4. Try using the GitHub Actions workflow to deploy"
echo "5. Check browser console for specific asset loading errors"
