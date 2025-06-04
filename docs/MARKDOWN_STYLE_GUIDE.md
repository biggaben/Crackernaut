# Markdown Style Guide for Crackernaut

This guide covers proper Markdown formatting standards for the Crackernaut project to ensure consistent, clean, and properly rendered documentation.

## Basic Formatting Rules

### Headers

Use consistent header hierarchy:

```markdown
# Main Title (H1) - Use only once per document
## Major Section (H2)
### Subsection (H3)
#### Sub-subsection (H4)
##### Minor section (H5)
###### Smallest section (H6)
```

**✅ Correct:**

```markdown
# Crackernaut Documentation
## Installation Guide
### Prerequisites
#### Python Version Requirements
```

**❌ Incorrect:**

```markdown
#Crackernaut Documentation (missing space)
## Installation Guide
#### Prerequisites (skipping H3)
```

### Lists

#### Unordered Lists

Use consistent bullet characters and proper spacing:

```markdown
- First item
- Second item
  - Nested item (2 spaces for indentation)
  - Another nested item
- Third item
```

#### Ordered Lists

Use consistent numbering:

```markdown
1. First step
2. Second step
   1. Sub-step (3 spaces for indentation)
   2. Another sub-step
3. Third step
```

### Emphasis and Strong Text

```markdown
*italic text* or _italic text_
**bold text** or __bold text__
***bold and italic*** or ___bold and italic___
```

### Code Formatting

#### Inline Code

Use backticks for inline code:

```markdown
Use the `uv run python` command to execute scripts.
The `torch.cuda.is_available()` function checks GPU availability.
```

#### Code Blocks

Use triple backticks with language specification:

````markdown
```python
def example_function():
    """This is a Python code block."""
    return "Hello, World!"
```

```bash
# This is a bash command
uv sync --extra dev
```

```json
{
  "config": "example",
  "value": 42
}
```
````

### Links

#### External Links

```markdown
[Link text](https://example.com)
[Link with title](https://example.com "Optional title")
```

#### Internal Links

```markdown
[See installation guide](#installation-guide)
[Check the config file](./config.json)
```

### Images

```markdown
![Alt text](path/to/image.png)
![Alt text with title](path/to/image.png "Image title")
```

## Advanced Formatting

### Tables

Use proper alignment and spacing:

```markdown
| Header 1    | Header 2    | Header 3    |
|-------------|-------------|-------------|
| Cell 1      | Cell 2      | Cell 3      |
| Longer cell | Short       | Medium cell |
```

With alignment:

```markdown
| Left Aligned | Center Aligned | Right Aligned |
|:-------------|:--------------:|--------------:|
| Left         |     Center     |         Right |
| Text         |     Text       |          Text |
```

### Blockquotes

```markdown
> This is a blockquote.
> It can span multiple lines.
>
> It can also contain other elements:
> - Lists
> - **Bold text**
> - `Code`
```

### Task Lists

```markdown
- [x] Completed task
- [ ] Incomplete task
- [ ] Another incomplete task
  - [x] Nested completed task
  - [ ] Nested incomplete task
```

## Common Formatting Issues to Avoid

### 1. Missing Blank Lines

**❌ Incorrect:**

```text
## Section Header
This paragraph is too close to the header.
### Subsection
Another paragraph without proper spacing.
```

**✅ Correct:**

```text
## Section Header

This paragraph has proper spacing.

### Subsection

Another paragraph with proper spacing.
```

### 2. Inconsistent List Formatting

**❌ Incorrect:**

```text
- Item 1
* Item 2 (different bullet style)
+ Item 3 (another different style)
```

**✅ Correct:**

```text
- Item 1
- Item 2
- Item 3
```

### 3. Improper Code Block Formatting

**❌ Incorrect:**

```text
(code block without language specification)
def function():
    pass
```

**✅ Correct:**

````text
```python
def function():
    pass
```
````

### 4. Missing Language in Code Blocks

**❌ Incorrect:**

````text
```
uv run python crackernaut.py
```
````

**✅ Correct:**

````text
```bash
uv run python crackernaut.py
```
````

### 5. Inconsistent Header Spacing

**❌ Incorrect:**

```text
#Header (no space after #)
##Another Header (no space after ##)
```

**✅ Correct:**

```text
# Header (space after #)
## Another Header (space after ##)
```

## Project-Specific Guidelines

### Documentation Structure

For Crackernaut documentation, follow this structure:

```markdown
# Document Title

Brief description of the document's purpose.

## Table of Contents (for longer documents)

- [Section 1](#section-1)
- [Section 2](#section-2)

## Main Content Sections

### Prerequisites

List any requirements.

### Installation

Step-by-step installation instructions.

### Usage

Examples and usage patterns.

### Configuration

Configuration options and examples.

## Examples

Provide practical examples.

## Troubleshooting

Common issues and solutions.

## Contributing

Guidelines for contributors.
```

### Code Examples

When showing Crackernaut-specific code:

````markdown
```python
# Good: Include context and comments
import torch
from src.utils.config import load_config

def check_gpu_availability():
    """Check if CUDA is available for training."""
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        return True
    return False
```

```bash
# Good: Show actual commands users will run
uv run python crackernaut.py --password "example123" --config config.json
```
````

### File References

Use consistent formatting for file references:

```markdown
- Configuration: `config.json`
- Main script: `crackernaut.py`
- Training script: `crackernaut_train.py`
- Project structure: `src/models/`, `src/utils/`
```

## Linting and Validation

### Common Markdown Lint Rules

These are the key rules to follow:

- **MD022**: Headings should be surrounded by blank lines
- **MD031**: Fenced code blocks should be surrounded by blank lines
- **MD040**: Fenced code blocks should have a language specified
- **MD013**: Line length should be reasonable (usually 80-100 characters)
- **MD033**: Limit HTML usage in Markdown
- **MD041**: First line should be a top-level heading

### VS Code Configuration

Add to your VS Code settings for better Markdown editing:

```json
{
  "markdownlint.config": {
    "MD013": false,
    "MD033": false,
    "MD041": false
  },
  "editor.wordWrap": "on",
  "editor.rulers": [80, 100]
}
```

## Best Practices Summary

1. **Consistency**: Use the same formatting patterns throughout
2. **Spacing**: Always include blank lines around headers, code blocks, and lists
3. **Language tags**: Always specify language for code blocks
4. **Alt text**: Always provide meaningful alt text for images
5. **Link validation**: Regularly check that all links work
6. **Table formatting**: Keep tables readable with proper alignment
7. **Header hierarchy**: Use logical header progression (don't skip levels)
8. **Line length**: Keep lines reasonable (under 80-100 characters when possible)
9. **Special characters**: Escape special Markdown characters when needed
10. **File paths**: Use backticks for file names and paths

## Quick Reference

| Element | Syntax |
|---------|--------|
| Header | `# H1`, `## H2`, `### H3` |
| Bold | `**bold**` or `__bold__` |
| Italic | `*italic*` or `_italic_` |
| Code | `` `code` `` |
| Link | `[text](url)` |
| Image | `![alt](url)` |
| List | `- item` or `1. item` |
| Quote | `> quote` |
| Table | `\| col1 \| col2 \|` |

## Markdown Linting Tools

### Recommended Extensions

- **markdownlint** (David Anson) - Real-time linting in VS Code
- **Markdown All in One** - Enhanced Markdown support
- **Markdown Preview Enhanced** - Better preview experience

### Command Line Tools

```bash
# Install markdownlint-cli
npm install -g markdownlint-cli

# Lint all markdown files
markdownlint "**/*.md"

# Fix auto-fixable issues
markdownlint "**/*.md" --fix
```

Following these guidelines will ensure all Markdown files in the Crackernaut project are consistently formatted and properly rendered across different platforms and tools.
