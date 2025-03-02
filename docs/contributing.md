# Contributing Guidelines

Thank you for your interest in contributing to the EEG-Enhanced Language Model with Mixture-of-Experts project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
  - [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Messages](#commit-messages)
  - [Pull Requests](#pull-requests)
- [Coding Standards](#coding-standards)
  - [Python Code Style](#python-code-style)
  - [JavaScript Code Style](#javascript-code-style)
  - [Documentation Style](#documentation-style)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Writing Tests](#writing-tests)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an open and welcoming environment.

## Getting Started

### Setting Up the Development Environment

1. **Fork the repository**:
   - Visit the [GitHub repository](https://github.com/yourusername/hackathon-mercor-etched-cognition)
   - Click the "Fork" button in the top-right corner

2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/hackathon-mercor-etched-cognition.git
   cd hackathon-mercor-etched-cognition
   ```

3. **Set up the upstream remote**:
   ```bash
   git remote add upstream https://github.com/yourusername/hackathon-mercor-etched-cognition.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the package in development mode
   ```

6. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Project Structure

The project is organized as follows:

```
hackathon-mercor-etched-cognition/
├── backend/                # FastAPI backend for token streaming
│   └── main.py             # Main FastAPI application
├── frontend/               # React frontend for visualization
│   ├── public/             # Public assets
│   ├── src/                # React source code
│   └── package.json        # Frontend dependencies
├── eeg/                    # EEG processing scripts
│   ├── eeg.ipynb           # Jupyter notebook for EEG processing
│   └── eeg.py              # Python script for EEG processing
├── kevin-moe-demo/         # Kevin-MOE demo implementation
│   ├── main_eeg.py         # Core EEG-enhanced LLM implementation
│   ├── eeg_processor.py    # EEG signal processing module
│   ├── moe_control.py      # Mixture-of-Experts controller
│   └── demo.py             # Interactive demonstration script
├── tests/                  # Test suite
│   ├── backend/            # Backend tests
│   ├── eeg/                # EEG processing tests
│   ├── frontend/           # Frontend tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
├── LICENSE                 # MIT License
├── README.md               # Project overview
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
├── Makefile                # Common tasks
└── docker-compose.yml      # Docker Compose configuration
```

## Development Workflow

### Branching Strategy

We use a feature branch workflow:

1. **Create a new branch for your feature or bugfix**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. **Keep your branch updated with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Commit Messages

Please follow these guidelines for commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
  - 🎨 `:art:` when improving the format/structure of the code
  - 🐛 `:bug:` when fixing a bug
  - 📚 `:books:` when adding or updating documentation
  - ✨ `:sparkles:` when adding a new feature
  - 🔧 `:wrench:` when updating configuration files
  - 🧪 `:test_tube:` when adding tests

Example:
```
✨ Add real-time EEG visualization component

- Implement WebGL-based EEG signal visualization
- Add frequency band power display
- Connect to WebSocket for real-time updates

Fixes #42
```

### Pull Requests

1. **Push your changes to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request**:
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your branch and the main repository's main branch
   - Fill out the PR template with details about your changes

3. **PR Review Process**:
   - All PRs require at least one review from a maintainer
   - Address any requested changes and push updates to your branch
   - Once approved, a maintainer will merge your PR

## Coding Standards

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style with the following additions:

- Use 4 spaces for indentation (no tabs)
- Maximum line length is 88 characters (following Black's default)
- Use docstrings for all public modules, functions, classes, and methods
- Use type hints where appropriate

We use the following tools for code quality:
- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting
- [mypy](https://mypy.readthedocs.io/) for static type checking

You can run these tools with:
```bash
# Format code
black .
isort .

# Check code quality
flake8
mypy .
```

### JavaScript Code Style

For JavaScript/React code, we follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript) with the following additions:

- Use 2 spaces for indentation
- Use semicolons
- Use single quotes for strings
- Use ES6 features where appropriate
- Use JSDoc comments for documentation

We use the following tools for code quality:
- [ESLint](https://eslint.org/) for linting
- [Prettier](https://prettier.io/) for code formatting

You can run these tools with:
```bash
cd frontend
npm run lint
npm run format
```

### Documentation Style

- Use Markdown for documentation
- Follow the [Google Developer Documentation Style Guide](https://developers.google.com/style)
- Include code examples where appropriate
- Keep documentation up-to-date with code changes

## Testing

### Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/backend/
pytest tests/eeg/
pytest tests/integration/

# Run with coverage report
pytest --cov=.
```

For frontend tests:

```bash
cd frontend
npm test
```

### Writing Tests

- Write unit tests for all new features and bug fixes
- Aim for at least 80% code coverage
- Use pytest fixtures for test setup
- Mock external dependencies
- Write integration tests for critical paths

Example test structure:

```python
def test_compute_focus_index():
    # Arrange
    theta = [1.0, 2.0, 3.0]
    alpha = [2.0, 3.0, 4.0]
    beta = [6.0, 9.0, 12.0]
    
    # Act
    result = compute_focus_index(theta, alpha, beta)
    
    # Assert
    expected = [2.0, 1.8, 1.714]
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert round(r, 3) == e
```

## Issue Reporting

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the [GitHub Issues](https://github.com/yourusername/hackathon-mercor-etched-cognition/issues)
2. If not, create a new issue using the appropriate template
3. Include as much detail as possible:
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Screenshots or logs if applicable
   - Environment information (OS, Python version, browser, etc.)

## Feature Requests

We welcome feature requests! To submit a feature request:

1. Check if the feature has already been requested in the [GitHub Issues](https://github.com/yourusername/hackathon-mercor-etched-cognition/issues)
2. If not, create a new issue using the feature request template
3. Clearly describe the feature and its benefits
4. If possible, outline how the feature might be implemented

## Documentation

Good documentation is crucial for the project's success. When contributing:

- Update the README.md if you change project setup or usage
- Update or add docstrings for all public functions, classes, and methods
- Update the API documentation if you change or add endpoints
- Add examples for new features
- Update installation and usage guides as needed

## Community

- Join our [Discord server](https://discord.gg/your-discord-invite) for discussions
- Follow us on [Twitter](https://twitter.com/your-twitter-handle) for updates
- Attend our monthly community calls (details in the Discord server)

Thank you for contributing to the EEG-Enhanced Language Model with Mixture-of-Experts project!
