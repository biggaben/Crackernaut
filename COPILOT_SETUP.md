# Crackernaut Workspace - Advanced Copilot Configuration with Agentic Reasoning

## Overview
This workspace implements a sophisticated GitHub Copilot configuration optimized for password security research and machine learning development, following advanced agentic reasoning methodology. The configuration utilizes hierarchical settings architecture with comprehensive context optimization specifically designed for Sonnet 4 Agent Mode capabilities.

**🆕 NEW: The project has fully migrated to [uv](https://astral.sh/uv/) for dependency management, replacing all pip/venv workflows. This implements 2025 Copilot advanced features including temporal context, next edit suggestions, and enhanced agent reasoning.**

## Agentic Reasoning Framework Applied

This configuration follows the Claude Sonnet 4 optimization methodology with four distinct phases:

### Phase 1: Deep Context Analysis
The setup recognizes Crackernaut's unique characteristics:
- **Security Research Focus**: Password analysis requiring ethical considerations and secure data handling
- **ML-Intensive Workloads**: PyTorch, CUDA, and large dataset processing requirements
- **Performance Constraints**: GPU memory optimization and async I/O for large password datasets
- **Modular Architecture**: Clear separation between models/, training data, and processing utilities

### Phase 2: Problem Space Exploration
Key optimization areas identified:
1. **Context Scope Management**: Protecting sensitive training data while maximizing code context
2. **Performance Optimization**: Excluding large datasets and binaries from indexing
3. **Advanced 2025 Features**: Leveraging temporal context and next edit suggestions
4. **Security Boundaries**: Maintaining ethical research standards with comprehensive filtering

### Phase 3: Solution Architecture Design
**Hierarchical configuration with clear boundaries**:

```
Workspace Level (Crackernaut.code-workspace):
├── Advanced 2025 Copilot features (Agent Mode, temporal context)
├── Performance-critical settings and file watchers
├── Cross-module development preferences
└── ML development optimizations

Folder Level (.vscode/settings.json):
├── Project-specific instruction file references
├── Context filtering patterns for security research
├── Local formatting and Python development settings
└── Advanced context features configuration

Repository Level (.github/):
├── Comprehensive instruction files for domain expertise
├── Advanced prompt file for enhanced reasoning
└── Security research context and ethical guidelines
```

### Phase 4: Implementation with Advanced Features

#### Workspace Configuration Strategy
```json
{
  "github.copilot.advanced": {
    "debug.temporalContext": true,        // Enhanced code evolution understanding
    "debug.nextEditSuggestions": true,   // Predictive editing for ML workflows
    "debug.agentMode": true              // Advanced reasoning for security research
  }
}
```

#### Context Optimization Logic
The configuration implements surgical precision filtering:
- **Security Protection**: Excludes sensitive password training data
- **Performance Impact**: Removes large model checkpoints and cluster data
- **Relevance Filtering**: Maintains focus on custom code while excluding third-party artifacts
- **Research Boundaries**: Protects ethical research context while enabling development assistance

## Quick Start with UV

### 1. Prerequisites
- **uv** (modern Python package manager - replaces pip/venv)
- **Python 3.8+**
- **CUDA** (optional, for GPU acceleration)

### 2. Setup

```bash
# Install uv (if not already installed)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows with WSL:
wsl curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repository-url>
cd crackernaut

# Automated migration from old pip setup (if applicable)
uv run python migrate_to_uv.py

# Or manual setup - install all dependencies including CUDA, dev tools, and tests
uv sync --all-extras
```

### 3. VS Code Integration
Open the project in VS Code and:
1. **Install recommended extensions** (prompted automatically)
2. **Use integrated tasks** via Command Palette (`Ctrl+Shift+P`):
   - `Tasks: Run Task` → `Install Dependencies`
   - `Tasks: Run Task` → `Run Crackernaut`
   - `Tasks: Run Task` → `Train Model`

### 4. Development Workflow

```bash
# Run the main application
uv run python crackernaut.py --password "test123" --model transformer

# Train models
uv run python crackernaut_train.py --config config.json

# Run tests
uv run python -m pytest

# Format code
uv run black *.py

# Check GPU status
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## File Structure

### Core Configuration Files
```
.vscode/
├── settings.json              # Workspace settings with Copilot configuration
├── launch.json               # Debug configurations for all project components
├── tasks.json                # Automated tasks for development workflow
├── extensions.json           # Recommended extensions
├── copilot-instructions.md   # Detailed project documentation (legacy)
└── python.code-snippets      # Custom Python snippets

.github/
├── copilot-instructions.md   # Repository-level Copilot instructions
└── copilot-chat.md          # Advanced prompt file for Copilot Chat

.copilotignore               # Files excluded from Copilot indexing
```

## Configuration Features

### 1. Repository-Level Instructions (`.github/copilot-instructions.md`)
- **Automatically Applied**: To all Copilot conversations in this repository
- **Project Context**: Password security research, ML models, cybersecurity
- **Technology Stack**: PyTorch, CUDA, transformers, async processing
- **Security Guidelines**: Password handling, secure coding practices
- **Performance Patterns**: GPU optimization, memory management

### 2. Workspace-Level Settings (`.vscode/settings.json`)
- **Agent Mode**: Enabled for Sonnet 4 optimization
- **Custom Instructions**: File-based and inline instructions
- **Context Enhancement**: Codebase awareness, temporal context, multi-turn conversations
- **Python Integration**: Type checking, formatting, linting configuration
- **Performance**: Excludes large training data from indexing

### 3. Advanced Prompt File (`.github/copilot-chat.md`)
- **System Role**: Expert AI assistant for security research
- **Domain Expertise**: Password security, ML, cybersecurity
- **Response Guidelines**: Security-first, performance-oriented suggestions

### 4. Privacy Protection (`.copilotignore`)
- **Sensitive Data**: Excludes password training data from Copilot indexing
- **Performance**: Excludes large binary files and temporary data
- **Security**: Protects environment variables and credentials

## Key Settings Enabled

### Advanced 2025 Copilot Features
- `github.copilot.advanced.debug.temporalContext`: true
- `github.copilot.advanced.debug.nextEditSuggestions`: true  
- `github.copilot.advanced.debug.agentMode`: true

### Enhanced Chat Capabilities
- `github.copilot.chat.useInstructionFiles`: true
- `github.copilot.chat.experimental.usePromptFiles`: true
- `github.copilot.chat.experimental.agent.enabled`: true
- `github.copilot.chat.experimental.contextAwareChat.enabled`: true
- `github.copilot.chat.experimental.workspaceContext.enabled`: true
- `github.copilot.chat.experimental.multiTurn.enabled`: true

### Code Generation Instructions
- File-based instructions from `.github/copilot-instructions.md`
- Inline security and performance guidelines
- ML-specific error handling patterns
- Async I/O best practices

### Python Development
- Pylance language server with type checking
- Black formatter with 88-character line limit
- Flake8 linting with project-specific rules
- Path configuration for models/ subdirectories

## Usage Instructions

### 1. Verify Setup
Open VS Code and check:
- **Copilot Settings**: Go to Settings → Extensions → GitHub Copilot
- **Instruction Files**: Verify "Use Instruction Files" is enabled
- **Repository Context**: Confirm Copilot recognizes the project type

### 2. Test Custom Instructions
Start a Copilot Chat and ask:
- "How should I handle password data in this project?"
- "What's the best way to implement a PyTorch model here?"
- "How do I process large datasets asynchronously?"

### 3. Development Workflow
- **F5**: Debug with pre-configured launch configurations
- **Ctrl+Shift+P**: Access custom tasks (training, testing, etc.)
- **Copilot Chat**: Get context-aware suggestions for ML and security code

### 4. Code Generation
- Use custom snippets: `torch-model`, `async-file`, `train-loop`
- Copilot will automatically apply security and performance patterns
- Type hints and error handling will be included by default

## Security Considerations

### What's Protected
- Password training data excluded from Copilot indexing
- No sensitive credentials or API keys in configuration
- Secure logging patterns enforced in code suggestions

### What's Shared with Copilot
- Project structure and code patterns
- Configuration files and documentation
- Non-sensitive code and algorithms

## Customization

### Adding New Instructions
1. **Repository-level**: Edit `.github/copilot-instructions.md`
2. **Workspace-level**: Add to `github.copilot.chat.experimental.codeGeneration.instructions` in `settings.json`
3. **Prompt customization**: Edit `.github/copilot-chat.md`

### Modifying Context
- Update file patterns in `github.copilot.chat.experimental.context.files`
- Add exclusions to `github.copilot.chat.experimental.context.excludeFiles`
- Modify `.copilotignore` for indexing exclusions

## Troubleshooting

### Common Issues
1. **Instructions not applied**: Check if "Use Instruction Files" is enabled
2. **Context not loaded**: Verify workspace is opened as folder, not individual files
3. **Performance issues**: Ensure large datasets are excluded via `.copilotignore`

### Verification Commands
```bash
# Check Copilot status
code --list-extensions | grep copilot

# Verify file structure
ls -la .github/
ls -la .vscode/
```

## Best Practices

### For Developers
- Always test security-related code suggestions carefully
- Verify GPU/CUDA code on appropriate hardware
- Review async patterns for proper error handling
- Validate ML model suggestions against project architecture

### For Team Collaboration
- Keep instructions updated as project evolves
- Document any security constraints in repository instructions
- Share workspace configuration for consistent development experience
- Regular review of excluded files in `.copilotignore`

## Advanced 2025 Features Enabled

### Copilot Advanced Configuration
- **Temporal Context**: `debug.temporalContext: true` - Enhanced understanding of code evolution patterns
- **Next Edit Suggestions**: `debug.nextEditSuggestions: true` - Predictive editing for ML development workflows  
- **Agent Mode**: `debug.agentMode: true` - Sophisticated reasoning for complex security research logic

### Enhanced Context Management
- **Workspace Context**: Comprehensive codebase awareness across all modules
- **Prompt Files**: Advanced reasoning templates for security research scenarios
- **Multi-Turn Conversations**: Context retention across extended development sessions
- **File Pattern Filtering**: Surgical inclusion/exclusion of relevant development context

### Security Research Optimizations
- **Sensitive Data Protection**: Training data and credentials excluded from all context
- **Performance Filtering**: Large binary files and model artifacts excluded for speed
- **Ethical Research Context**: Built-in understanding of responsible disclosure practices
- **Domain Expertise**: Password security, ML, and cybersecurity specialized knowledge

This configuration provides a comprehensive, secure, and performance-optimized development environment for the Crackernaut password security research project.

## Agentic Reasoning Benefits

This advanced configuration methodology provides several key advantages over basic Copilot setup:

### Enhanced Code Understanding
- **Temporal Context**: Copilot understands code evolution patterns across the development lifecycle
- **Next Edit Prediction**: Anticipates likely next steps in ML model development and security research
- **Agent Mode Reasoning**: Sophisticated analysis of complex password security algorithms and ML architectures

### Optimized Performance
- **Surgical Context Filtering**: Includes only relevant code while excluding large datasets and binaries
- **Memory Efficiency**: Reduces indexing overhead for better VS Code performance
- **GPU-Aware Development**: Optimized for CUDA development workflows and memory management

### Security Research Specific
- **Ethical Research Context**: Built-in understanding of responsible disclosure and security research ethics
- **Sensitive Data Protection**: Comprehensive exclusion of password training data from all contexts
- **Domain Expertise**: Specialized knowledge for password analysis, ML security, and cybersecurity tools

### Development Workflow Integration
- **UV Ecosystem**: Seamless integration with modern Python package management
- **Task Automation**: Enhanced understanding of VS Code tasks and development workflows
- **Multi-Modal Context**: Combines workspace, repository, and prompt-based instruction sources

This configuration transforms Crackernaut development from basic code completion to sophisticated AI-assisted security research, following proven methodologies from advanced development environments.
