# 🧹 PandasCleaner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/SANTHOSH-SACHIN/pandas-cleaner)

A high-performance interactive data cleaning tool with a Streamlit UI that combines the power of Polars and Pandas for optimal performance. Clean your data from multiple formats (CSV, Excel, Parquet) with an intuitive interface while generating production-ready Python code. Choose between lightning-fast Polars operations or familiar Pandas syntax for your data cleaning needs.

## ✨ Features

- 📊 Interactive DataFrame Visualization
  - Support for multiple file formats (CSV, Excel, Parquet)
  - Upload and view data in a responsive table interface
  - Sort columns with a single click
  - Real-time data preview

- 🧰 High-Performance Data Cleaning Tools
  - Powered by Polars for lightning-fast data operations
  - Fallback to Pandas for compatibility
  - Handle missing values intelligently (drop, fill with mean/median/mode, or custom value)
  - Advanced filtering using Polars/Pandas query syntax
  - Flexible group by operations with various aggregation functions
  - Future support for Rust-powered operations for even better performance

- 💾 Smart Session Management
  - Automatic session persistence using SQLite
  - Resume your work exactly where you left off
  - Multiple session support
  - Improved state management between operations

- 📝 Code Generation
  - Export your cleaning steps as production-ready Python code
  - Direct integration with existing data pipelines
  - Generated code follows best practices
  - Proper state tracking for all operations

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
   - Use the sidebar file uploader to select your data file
   - Supported formats: CSV, Excel (.xlsx, .xls), Parquet
   - Preview your data in the interactive table
   - Automatic format detection and optimized loading

2. 🧹 **Clean Your Data**
   - **Missing Values**
     ```python
     # Using Polars (faster)
     if use_polars:
         df = df.drop_nulls()  # or
         df = df.fill_null(df.mean())
     # Using Pandas (compatible)
     else:
         df = df.dropna()  # or
         df = df.fillna(df.mean())
     ```
   - **Filtering**
     ```python
     # Using Polars (faster)
     if use_polars:
         df = df.filter(pl.col("age") > 25 & pl.col("category") == "A")
     # Using Pandas (compatible)
     else:
         df = df.query("age > 25 and category == 'A'")
     ```
   - **Aggregation**
     ```python
     # Using Polars (faster)
     if use_polars:
         df = (df.group_by("category")
               .agg([pl.col("value").mean()])
               .collect())
     # Using Pandas (compatible)
     else:
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
- **State Management**: Improved persistence between operations

## 🔄 What's New in v0.3.0

### Major Features
1. Added Polars Integration:
   - High-performance data operations using Polars
   - Seamless fallback to Pandas when needed
   - Generated code supports both Polars and Pandas syntax
   - Optimized memory usage and processing speed

2. Added Multi-Format Support:
   - CSV file support with optimized loading
   - Excel (.xlsx, .xls) file support
   - Parquet file support
   - Automatic format detection
   - Smart memory management for large files

### Bug Fixes and Improvements
1. Fixed issues with state management:
   - Added proper session initialization check
   - Added cleaning steps persistence between reruns
   - Added file-based state management
   - Fixed state reset issues

2. Fixed Filter functionality:
   - Added proper query validation
   - Added error handling for invalid queries
   - Fixed state management for filter operations
   - Added empty result handling

3. Fixed Group By operations:
   - Improved handling of numeric columns
   - Added proper aggregation dictionary creation
   - Added warning for non-numeric columns
   - Fixed state persistence for group operations

4. Fixed Code Generation:
   - Fixed cleaning steps tracking
   - Added proper state checks
   - Added proper string escaping
   - Added reset functionality in generated code

5. Improved Session Management:
   - Added proper state persistence
   - Added file change detection
   - Added state backup and restore
   - Added debug tracking

6. Added Debug Information:
   - Added state tracking throughout the application
   - Added operation tracking
   - Added clear feedback messages
   - Added detailed error messages

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

- Python ≥ 3.9
- Core Dependencies (auto-installed):
  - streamlit ≥ 1.44.1
  - pandas ≥ 2.2.3
  - polars ≥ 0.20.15
  - numpy ≥ 2.2.4
  - openpyxl ≥ 3.1.2
  - pyarrow ≥ 15.0.0

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- High-performance operations by [Polars](https://www.pola.rs/)
- Data analysis with [Pandas](https://pandas.pydata.org/)
- Future optimizations with [Rust](https://www.rust-lang.org/)

## 🚀 Roadmap

- [ ] Add Rust integration for faster data processing
- [ ] Implement machine learning-based data cleaning suggestions
- [ ] Add support for more data formats (JSON, SQL, etc.)
- [ ] Create a plugin system for custom cleaning operations
- [ ] Add data visualization features
- [ ] Implement automated data quality checks
