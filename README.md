# 🧹 PandasCleaner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A high-performance interactive data cleaning tool with a Streamlit UI that combines the power of Pandas with Rust for optimal performance. Clean your data with an intuitive interface while generating production-ready Python code.

## ✨ Features

- 📊 Interactive DataFrame Visualization
  - Upload and view CSV files in a responsive table interface
  - Sort columns with a single click
  - Real-time data preview

- 🧰 Comprehensive Data Cleaning Tools
  - Handle missing values intelligently (drop, fill with mean/median/mode, or custom value)
  - Advanced filtering using Pandas query syntax
  - Flexible group by operations with various aggregation functions
  - Future support for Rust-powered operations for enhanced performance

- 💾 Smart Session Management
  - Automatic session persistence using SQLite
  - Resume your work exactly where you left off
  - Multiple session support

- 📝 Code Generation
  - Export your cleaning steps as production-ready Python code
  - Direct integration with existing data pipelines
  - Generated code follows best practices

## 🚀 Quick Start

### Installation

Choose your preferred installation method:

```bash
# Using pip
pip install git+https://github.com/SANTHOSH-SACHIN/pandas-cleaner.git

# Using uv (recommended)
uv pip install git+https://github.com/SANTHOSH-SACHIN/pandas-cleaner.git

# Development installation
git clone https://github.com/SANTHOSH-SACHIN/pandas-cleaner.git
cd pandas-cleaner
uv pip install -e .
```

## 🎯 Usage

### Starting the Application

Launch the application with a single command:

```bash
pandas-cleaner start  # Starts on default port 8501
pandas-cleaner start --port 8000  # Use custom port
```

### Data Cleaning Workflow

1. 📤 **Upload Your Data**
   - Use the sidebar file uploader to select your CSV file
   - Preview your data in the interactive table

2. 🧹 **Clean Your Data**
   - **Missing Values**
     ```python
     # Example of generated code for handling missing values
     df = df.fillna(df.mean(numeric_only=True))
     ```
   - **Filtering**
     ```python
     # Example of filtering data
     df = df.query("age > 25 and category == 'A'")
     ```
   - **Aggregation**
     ```python
     # Example of group by operation
     df = df.groupby(['category']).agg('mean').reset_index()
     ```

3. 💻 **Export Your Work**
   - Click "Generate Python Code" to get production-ready code
   - Copy-paste into your data pipeline
   - All operations are automatically logged

4. ⏱ **Session Management**
   - Work is automatically saved
   - Resume from where you left off
   - Switch between multiple sessions

### Advanced Features

- **Custom Value Imputation**: Fill missing values with domain-specific defaults
- **Complex Queries**: Use the full power of Pandas query syntax
- **Aggregation Functions**: Support for sum, mean, count, min, max, and more

## 🤝 Contributing

We welcome contributions! Check out our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Setting up your development environment
- Coding standards
- Submitting pull requests
- Adding Rust optimizations
- Reporting bugs

Join our community of contributors and help make data cleaning more efficient!

## 🛠 Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SANTHOSH-SACHIN/pandas-cleaner.git
   cd pandas-cleaner
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate  # Windows
   ```

3. **Install Development Dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

## 📋 Requirements

- Python ≥ 3.13
- Core Dependencies (auto-installed):
  - streamlit ≥ 1.44.1
  - pandas ≥ 2.2.3
  - numpy ≥ 2.2.4

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Pandas](https://pandas.pydata.org/)
- Future optimizations with [Rust](https://www.rust-lang.org/)

## 🚀 Roadmap

- [ ] Add Rust integration for faster data processing
- [ ] Implement machine learning-based data cleaning suggestions
- [ ] Add support for more file formats (Excel, Parquet, etc.)
- [ ] Create a plugin system for custom cleaning operations
