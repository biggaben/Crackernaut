# Crackernaut Agents & AI Collaboration Guide

This document outlines the roles, responsibilities, and interaction guidelines for AI agents (like GitHub Copilot) and human developers working on the Crackernaut project.

## Guiding Principles

1. **Clarity & Precision**: AI-generated code and documentation must be clear, precise, and easy for humans to understand.
2. **Adherence to Standards**: All contributions must strictly follow project-defined coding standards, security practices, and architectural patterns.
3. **Iterative Refinement**: AI suggestions are starting points. Human oversight is crucial for validation, testing, and refinement.
4. **Security First**: Given the nature of password research, security is paramount. AI should assist in identifying and mitigating vulnerabilities.
5. **Efficiency & Performance**: AI should help optimize code for performance, especially in data processing and ML model training.
6. **Knowledge Sharing**: AI interactions and learnings should be documented to build a shared knowledge base.

## Agent Roles & Responsibilities

### 1. Code Generation & Refactoring Agent (GitHub Copilot)

- **Responsibilities**:
  - Generate boilerplate code, utility functions, and data structures.
  - Assist in refactoring complex code blocks to improve readability and maintainability.
  - Suggest optimizations for performance and memory usage.
  - Help implement new features based on detailed specifications.
  - Ensure adherence to PEP 8, type hinting, and project-specific style guides.
  - **NEW (June 2025)**: Actively manage and suggest refactoring to keep cognitive complexity of all functions below 15.
  - **NEW (June 2025)**: Propose and use helper functions following established patterns (`_load_*`, `_setup_*`, `_process_*`, `_handle_*`) for complex workflows.
- **Interaction Guidelines**:
  - Provide clear, concise prompts with sufficient context.
  - Use `@workspace` or specific file references (`#file`) for context.
  - Review and test all generated code thoroughly.
  - Iterate on suggestions, providing feedback for improvement.
  - Use Copilot Chat for architectural discussions and complex problem-solving.

### 2. ML Model Development & Training Agent (Conceptual)

**Note**: This agent is more conceptual but guides how we leverage AI in ML tasks

- **Responsibilities**:
  - Assist in designing PyTorch model architectures (Transformers, RNNs, MLPs).
  - Help implement training loops, loss functions, and optimizers.
  - Suggest data augmentation techniques for password datasets.
  - Aid in hyperparameter tuning and model evaluation strategies.
  - Ensure proper CUDA device management and memory handling.
  - **NEW (June 2025)**: Guide the implementation of training workflows using the newly refactored helper function structure, ensuring modularity and testability.
- **Interaction Guidelines**:
  - Define clear model requirements, input/output specifications, and performance metrics.
  - Provide context on existing model architectures and training pipelines.
  - Use AI to explore research papers and novel techniques relevant to password modeling.
  - Validate AI-suggested model components with theoretical understanding and empirical testing.

### 3. Documentation & Knowledge Management Agent

- **Responsibilities**:
  - Generate initial drafts of README files, code comments, and API documentation.
  - Help maintain consistency across all project documentation.
  - Summarize complex code sections or architectural decisions.
  - Assist in creating tutorials or usage guides.
  - **NEW (June 2025)**: Ensure all documentation reflects the latest code structure, including refactored components and helper functions.
- **Interaction Guidelines**:
  - Provide context on the specific documentation needed (e.g., "document this function," "create a README section for X").
  - Review and edit AI-generated documentation for accuracy, clarity, and completeness.
  - Use AI to identify areas where documentation is lacking or outdated.

## Best Practices for AI Collaboration

1. **Context is Key**: Always provide sufficient context to the AI. This includes relevant code snippets, file paths, error messages, and desired outcomes.
2. **Be Specific**: Vague prompts lead to generic or incorrect suggestions. Clearly define what you need.
3. **Iterate and Refine**: Don't expect perfect results on the first try. Use AI suggestions as a starting point and iterate with more specific feedback.
4. **Human Oversight is Non-Negotiable**: AI is a tool, not a replacement for human expertise. Always review, test, and validate AI-generated content.
5. **Understand Limitations**: Be aware of the AI's knowledge cutoff and potential biases. Cross-verify critical information.
6. **Security Review**: For a security-focused project like Crackernaut, all AI suggestions, especially those related to data handling or cryptographic operations (if any), must undergo rigorous security review.
7. **Version Control**: Commit AI-assisted changes in small, logical units with clear commit messages indicating AI involvement (e.g., "Refactor X with Copilot assistance").
8. **Feedback Loop**: Utilize features like "thumbs up/down" in Copilot to improve its future suggestions.

## Recent Improvements (June 2025)

The Crackernaut project has undergone significant refactoring to improve code quality, maintainability, and testability. AI agents should be aware of and leverage these improvements:

- **Cognitive Complexity Reduction**: Key training functions (`bulk_train_on_wordlist`, `interactive_training`) have been refactored to keep cognitive complexity below 15. This standard now applies to all new and modified code.
- **Modular Helper Functions**: Complex workflows have been broken down into smaller, single-responsibility helper functions (e.g., `_load_wordlist`, `_setup_training_components`, `_process_user_input`). AI should adopt and suggest similar patterns.
- **Improved Error Handling**: Helper functions now incorporate more granular error handling.
- **Enhanced Testability**: The modular design makes individual components easier to test.

By following these guidelines, we can effectively leverage AI to accelerate development, improve code quality, and enhance the capabilities of the Crackernaut project.
