#!/bin/bash
# Cleanup Script: Delete 19 redundant files
# Generated: 2026-01-28
# Safe to run - only removes old/legacy/new variants

echo "🧹 Starting cleanup of redundant files..."
echo ""

# Navigate to docs directory
cd /Users/venugopala.nakka/projects/nvgr/my-learning/docs

# Count files before
BEFORE=$(find . -type f -name "*.md" | wc -l)
echo "📊 Files before cleanup: $BEFORE"
echo ""

# Linked Lists (4 files)
echo "Cleaning: Linked Lists..."
rm -v ./algorithms/data-structures/linked-lists/hard-problems-old.md
rm -v ./algorithms/data-structures/linked-lists/medium-problems-legacy.md
rm -v ./algorithms/data-structures/linked-lists/medium-problems-new.md
rm -v ./algorithms/data-structures/linked-lists/medium-problems-old.md

# Queues (2 files)
echo "Cleaning: Queues..."
rm -v ./algorithms/data-structures/queues/hard-problems-old.md
rm -v ./algorithms/data-structures/queues/medium-problems-old.md

# Stacks (2 files)
echo "Cleaning: Stacks..."
rm -v ./algorithms/data-structures/stacks/hard-problems-old.md
rm -v ./algorithms/data-structures/stacks/medium-problems-old.md

# Greedy (3 files)
echo "Cleaning: Greedy..."
rm -v ./algorithms/greedy/easy-problems-old.md
rm -v ./algorithms/greedy/hard-problems-old.md
rm -v ./algorithms/greedy/medium-problems-old.md

# Math (6 files)
echo "Cleaning: Math..."
rm -v ./algorithms/math/easy-problems-new.md
rm -v ./algorithms/math/easy-problems-old.md
rm -v ./algorithms/math/hard-problems-new.md
rm -v ./algorithms/math/hard-problems-old.md
rm -v ./algorithms/math/medium-problems-new.md
rm -v ./algorithms/math/medium-problems-old.md

# Searching (1 file)
echo "Cleaning: Searching..."
rm -v ./algorithms/searching/search-problems-legacy.md

# Trees (1 file)
echo "Cleaning: Trees..."
rm -v ./algorithms/trees/tree-problems-legacy.md

# Count files after
AFTER=$(find . -type f -name "*.md" | wc -l)
DELETED=$((BEFORE - AFTER))

echo ""
echo "✅ Cleanup complete!"
echo "📊 Files after cleanup: $AFTER"
echo "🗑️  Files deleted: $DELETED"
echo ""
echo "Next steps:"
echo "1. Run: git status"
echo "2. Verify expected files were deleted"
echo "3. Test navigation: python -m mkdocs serve"
echo "4. Commit: git add -A && git commit -m 'Clean up redundant file variants'"
