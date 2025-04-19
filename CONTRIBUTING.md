# Contributing to PandasCleaner

First off, thank you for considering contributing to PandasCleaner! It's people like you that make PandasCleaner such a great tool.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. Please report unacceptable behavior.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if possible

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A step-by-step description of the suggested enhancement
* Any possible drawbacks
* Screenshots or animated GIFs if applicable

### Adding Rust Integration

We're particularly interested in contributions that help optimize performance using Rust. Here's how you can help:

1. **Identify Bottlenecks**
   - Profile the current Python implementation
   - Look for operations that could benefit from Rust's performance
   - Focus on computationally intensive tasks

2. **Implementing Rust Modules**
   ```rust
   // Example Rust module structure
   #[derive(Debug)]
   pub struct DataFrameOps {
       // Your implementation here
   }

   impl DataFrameOps {
       pub fn new() -> Self {
           // Initialize
       }

       pub fn process_data(&self, data: Vec<f64>) -> Vec<f64> {
           // Your fast implementation here
       }
   }
   ```

3. **Python Bindings**
   - Use PyO3 for creating Python bindings
   - Ensure seamless integration with existing pandas operations
   - Maintain type safety and error handling

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the existing style.

## Development Process

1. **Setting up your development environment**
   ```bash
   # Clone your fork
   git clone git@github.com:your-username/pandas-cleaner.git
   cd pandas-cleaner

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows

   # Install development dependencies
   uv pip install -e ".[dev]"
   ```

2. **Running Tests**
   ```bash
   pytest tests/
   ```

3. **Code Style**
   - Follow PEP 8 for Python code
   - Use Rustfmt for Rust code
   - Keep functions focused and modular
   - Add comprehensive docstrings

## Project Structure

```
pandas-cleaner/
├── src/
│   └── pandas_cleaner/
│       ├── __init__.py
│       ├── app.py          # Main Streamlit application
│       └── cli.py          # CLI interface
├── rust/                   # Future Rust integration
│   └── src/
│       └── lib.rs
├── tests/                  # Test suite
├── docs/                   # Documentation
└── examples/              # Example scripts
```

## Future Rust Integration Plans

We plan to integrate Rust in the following areas:

1. **Data Processing Operations**
   - Batch operations on numerical data
   - String operations and pattern matching
   - Custom aggregation functions

2. **Performance-Critical Components**
   - Missing value imputation
   - Data transformation pipelines
   - Complex filtering operations

3. **Memory Management**
   - Efficient handling of large datasets
   - Memory-mapped file operations
   - Custom data structures

## Documentation

- Add docstrings to all public functions
- Update the README.md for significant changes
- Include examples in the docs directory
- Create or update API documentation

## Questions?

Feel free to open an issue with your question or reach out to the maintainers directly.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
