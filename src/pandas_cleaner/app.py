import streamlit as st
import pandas as pd
import polars as pl
import sqlite3
import json
from datetime import datetime
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from pandas_cleaner.visualizations import PLOT_TYPES, COLOR_SCHEMES, create_correlation_heatmap

# File type constants
SUPPORTED_FILE_TYPES = {
    "csv": "CSV",
    "xlsx": "Excel",
    "xls": "Excel",
    "parquet": "Parquet"
}

def load_dataframe(uploaded_file) -> Tuple[pl.DataFrame, str]:
    """Load a dataframe from various file formats using Polars."""
    file_extension = Path(uploaded_file.name).suffix[1:].lower()

    if file_extension not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file format: {file_extension}")

    try:
        if file_extension == "csv":
            df = pl.read_csv(uploaded_file)
        elif file_extension in ["xlsx", "xls"]:
            # Convert Excel to pandas first, then to Polars
            pandas_df = pd.read_excel(uploaded_file)
            df = pl.from_pandas(pandas_df)
        elif file_extension == "parquet":
            df = pl.read_parquet(uploaded_file)
        return df, file_extension
    except Exception as e:
        raise ValueError(f"Error loading {SUPPORTED_FILE_TYPES[file_extension]} file: {str(e)}")

def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars DataFrame to Pandas DataFrame for compatibility."""
    return df.to_pandas()

def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert Pandas DataFrame to Polars DataFrame."""
    return pl.from_pandas(df)

def init_session_state():
    """Initialize session state variables."""
    # Only initialize if not already initialized
    if not st.session_state.get('initialized', False):
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'original_df' not in st.session_state:
            st.session_state.original_df = None
        if 'cleaning_steps' not in st.session_state:
            st.session_state.cleaning_steps = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.initialized = True
    # st.write("Debug: Initial session state:", {
    #     "has_df": st.session_state.df is not None,
    #     "has_original_df": st.session_state.original_df is not None,
    #     "cleaning_steps": st.session_state.cleaning_steps,
    #     "session_id": st.session_state.session_id
    # })

def save_session(conn: sqlite3.Connection, session_id: str, df: Optional[pd.DataFrame], cleaning_steps: List[Dict[str, Any]]) -> None:
    """Save current session to SQLite database."""
    if df is not None:
        df_json = df.to_json()
        steps_json = json.dumps(cleaning_steps)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sessions (session_id, data, cleaning_steps, last_modified)
            VALUES (?, ?, ?, datetime('now'))
        ''', (session_id, df_json, steps_json))
        conn.commit()

def load_session(conn: sqlite3.Connection, session_id: str) -> tuple[Optional[pd.DataFrame], List[Dict[str, Any]]]:
    """Load session from SQLite database."""
    cursor = conn.cursor()
    cursor.execute('SELECT data, cleaning_steps FROM sessions WHERE session_id = ?', (session_id,))
    result = cursor.fetchone()
    if result:
        df = pd.read_json(result[0])
        cleaning_steps = json.loads(result[1])
        return df, cleaning_steps
    return None, []

def setup_database() -> sqlite3.Connection:
    """Set up SQLite database and required tables."""
    conn = sqlite3.connect('pandas_cleaner_sessions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            data TEXT,
            cleaning_steps TEXT,
            last_modified DATETIME
        )
    ''')
    conn.commit()
    return conn

def handle_missing_values(df: pl.DataFrame, strategy: str, fill_value: Optional[Union[int, float, str]] = None) -> pl.DataFrame:
    """Handle missing values in the DataFrame using Polars."""
    if strategy == 'drop':
        return df.drop_nulls()

    if strategy == 'fill_value' and fill_value is not None:
        return df.fill_null(fill_value)

    # Get numeric and non-numeric columns
    numeric_dtypes = [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    numeric_cols = [col for col in df.columns if df[col].dtype in numeric_dtypes]
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    if strategy == 'fill_mean':
        # Fill numeric columns with mean
        for col in numeric_cols:
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).mean()))
    elif strategy == 'fill_median':
        # Fill numeric columns with median
        for col in numeric_cols:
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).median()))
    elif strategy == 'fill_mode':
        # Handle numeric and non-numeric columns separately
        for col in numeric_cols:
            mode_val = df.select(pl.col(col).mode().first()).item()
            df = df.with_columns(pl.col(col).fill_null(mode_val if mode_val is not None else 0))
        for col in non_numeric_cols:
            mode_val = df.select(pl.col(col).mode().first()).item()
            df = df.with_columns(pl.col(col).fill_null(mode_val if mode_val is not None else ''))

    return df

def validate_and_apply_filter(df: pl.DataFrame, query: str) -> tuple[bool, str, Optional[pl.DataFrame]]:
    """Validate and apply a filter query using Polars."""
    try:
        # Check if query is empty
        if not query.strip():
            return False, "Query cannot be empty", None

        # Convert Pandas-style query to Polars expression
        # Replace common pandas operations with Polars equivalents
        query = query.replace('==', '=')  # Polars uses single = for equality

        # Try to apply the filter
        filtered_df = df.filter(pl.expr(query))

        if filtered_df.height == 0:
            return True, "Filter resulted in empty DataFrame", None

        return True, "", filtered_df
    except Exception as e:
        return False, str(e), None

def convert_data_types(df: pl.DataFrame, type_conversions: Dict[str, str]) -> pl.DataFrame:
    """Convert data types of specified columns in the DataFrame using Polars."""
    for column, target_type in type_conversions.items():
        try:
            if target_type == 'int':
                df = df.with_columns(pl.col(column).cast(pl.Int64, strict=False))
            elif target_type == 'float':
                df = df.with_columns(pl.col(column).cast(pl.Float64, strict=False))
            elif target_type == 'datetime':
                df = df.with_columns(pl.col(column).str.strptime(pl.Datetime, strict=False))
            elif target_type == 'string':
                df = df.with_columns(pl.col(column).cast(pl.Utf8))
            elif target_type == 'boolean':
                df = df.with_columns(pl.col(column).cast(pl.Boolean))
        except Exception as e:
            st.error(f"Error converting column '{column}' to {target_type}: {str(e)}")
            return df  # Return unchanged DataFrame on error

    return df

def apply_group_by(df: pl.DataFrame, group_cols: List[str], agg_func: str) -> pl.DataFrame:
    """Apply groupby operation to the DataFrame using Polars."""
    if not group_cols:
        return df

    # Get numeric columns for aggregation
    numeric_dtypes = [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    numeric_cols = [col for col in df.columns if df[col].dtype in numeric_dtypes and col not in group_cols]

    if not numeric_cols:
        st.warning("No numeric columns available for aggregation")
        return df

    try:
        # Map pandas-style aggregation functions to Polars
        agg_map = {
            'mean': 'mean',
            'sum': 'sum',
            'count': 'count',
            'min': 'min',
            'max': 'max'
        }

        # Create aggregation expressions
        agg_exprs = [
            pl.col(col).agg_groups(agg_map[agg_func]).alias(col)
            for col in numeric_cols
        ]

        # Apply group by with aggregation
        return df.group_by(group_cols).agg(agg_exprs)
    except Exception as e:
        st.error(f"Error in group by operation: {str(e)}")
        return df

def export_cleaning_code(cleaning_steps: List[Dict[str, Any]]) -> str:
    """Generate Python code for the cleaning steps."""
    # Helper function to escape string values in queries
    def escape_string(s: str) -> str:
        return s.replace("'", "\\'").replace('"', '\\"')

    code = [
        "import pandas as pd",
        "import polars as pl",
        "import numpy as np",
        "import plotly.express as px",
        "import plotly.graph_objects as go",
        "\n# Choose your preferred library",
        "use_polars = True  # Set to False to use pandas\n",
        "if use_polars:",
        "    # Load your data file with Polars",
        "    df = pl.read_csv('your_file.csv')  # or read_parquet, etc.",
        "    original_df = df.clone()",
        "else:",
        "    # Load your data file with Pandas",
        "    df = pd.read_csv('your_file.csv')",
        "    original_df = df.copy()",
        "\n# Helper functions for Polars/Pandas compatibility",
        "def to_polars(df_pandas):",
        "    return pl.from_pandas(df_pandas)",
        "def to_pandas(df_polars):",
        "    return df_polars.to_pandas()\n"
    ]

    for step in cleaning_steps:
        if step['type'] == 'missing_values':
            if step['strategy'] == 'drop':
                code.append("# Drop missing values")
                code.append("if use_polars:")
                code.append("    df = df.drop_nulls()")
                code.append("else:")
                code.append("    df = df.dropna()")
            else:
                code.append(f"# Fill missing values using {step['strategy']}")
                if step['strategy'] == 'fill_value':
                    code.append("if use_polars:")
                    code.append(f"    df = df.fill_null({step['value']})")
                    code.append("else:")
                    code.append(f"    df = df.fillna({step['value']})")
                elif step['strategy'] in ['fill_mean', 'fill_median']:
                    method = step['strategy'].split('_')[1]
                    code.append("if use_polars:")
                    code.append("    # Get numeric columns and fill with " + method)
                    code.append("    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]")
                    code.append("    for col in numeric_cols:")
                    code.append(f"        df = df.with_columns(pl.col(col).fill_null(pl.col(col).{method}()))")
                    code.append("else:")
                    code.append("    # Handle numeric columns only")
                    code.append("    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns")
                    code.append(f"    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].{method}())")
                elif step['strategy'] == 'fill_mode':
                    code.append("if use_polars:")
                    code.append("    # Handle numeric and non-numeric columns separately")
                    code.append("    numeric_dtypes = [pl.Float32, pl.Float64, pl.Int32, pl.Int64]")
                    code.append("    numeric_cols = [col for col in df.columns if df[col].dtype in numeric_dtypes]")
                    code.append("    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]")
                    code.append("    for col in numeric_cols:")
                    code.append("        mode_val = df.select(pl.col(col).mode().first()).item()")
                    code.append("        df = df.with_columns(pl.col(col).fill_null(mode_val if mode_val is not None else 0))")
                    code.append("    for col in non_numeric_cols:")
                    code.append("        mode_val = df.select(pl.col(col).mode().first()).item()")
                    code.append("        df = df.with_columns(pl.col(col).fill_null(mode_val if mode_val is not None else ''))")
                    code.append("else:")
                    code.append("    # Handle numeric and non-numeric columns separately")
                    code.append("    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns")
                    code.append("    non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns")
                    code.append("    for col in numeric_cols:")
                    code.append("        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)")
                    code.append("    for col in non_numeric_cols:")
                    code.append("        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '')")
        elif step['type'] == 'group_by':
            cols = ", ".join([f"'{col}'" for col in step['columns']])
            code.append("# Group by operation")
            code.append("if use_polars:")
            code.append("    # Get numeric columns for aggregation")
            code.append("    numeric_dtypes = [pl.Float32, pl.Float64, pl.Int32, pl.Int64]")
            code.append("    numeric_cols = [col for col in df.columns if df[col].dtype in numeric_dtypes")
            code.append(f"                   and col not in [{cols}]]")
            code.append("    if numeric_cols:")
            code.append("        # Create aggregation expressions")
            code.append(f"        agg_exprs = [pl.col(col).agg_groups('{step['agg_func']}').alias(col)")
            code.append("                      for col in numeric_cols]")
            code.append(f"        df = df.group_by([{cols}]).agg(agg_exprs)")
            code.append("else:")
            code.append("    # Get numeric columns for aggregation")
            code.append("    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns")
            code.append("    # Filter out group columns from numeric columns")
            code.append(f"    agg_cols = [col for col in numeric_cols if col not in [{cols}]]")
            code.append("    # Create aggregation dictionary")
            code.append(f"    agg_dict = {{col: '{step['agg_func']}' for col in agg_cols}}")
            code.append(f"    df = df.groupby([{cols}]).agg(agg_dict).reset_index()")
        elif step['type'] == 'filter':
            code.append("# Apply filter")
            escaped_query = escape_string(step['query'])
            code.append("if use_polars:")
            code.append(f"    query = '{escaped_query}'.replace('==', '=')")  # Convert pandas query to polars
            code.append("    df = df.filter(pl.expr(query))")
            code.append("else:")
            code.append(f"    df = df.query('{escaped_query}')")
        elif step['type'] == 'visualization':
            code.append(f"# Create {step['plot_type']}")
            if step['plot_type'] == 'Correlation Heatmap':
                code.append(f"heatmap_fig = create_correlation_heatmap(df, {step['columns']})")
                code.append("heatmap_fig.show()")
            elif step['plot_type'] in ['Histogram', 'Box Plot']:
                if 'group_by' in step and step['group_by']:
                    code.append(f"fig = create_{step['plot_type'].lower().replace(' ', '_')}(df, '{step['column']}', '{step['group_by']}', '{step['color_scheme']}')")
                else:
                    if 'nbins' in step:
                        code.append(f"fig = create_{step['plot_type'].lower().replace(' ', '_')}(df, '{step['column']}', '{step['color_scheme']}', nbins={step['nbins']})")
                    else:
                        code.append(f"fig = create_{step['plot_type'].lower().replace(' ', '_')}(df, '{step['column']}', color_scheme='{step['color_scheme']}')")
                code.append("fig.show()")
            elif step['plot_type'] in ['Scatter Plot', 'Line Plot', 'Bar Plot']:
                if step['plot_type'] == 'Scatter Plot' and 'size_column' in step:
                    code.append(f"fig = create_{step['plot_type'].lower().replace(' ', '_')}(")
                    code.append(f"    df, '{step['x_column']}', '{step['y_column']}',")
                    code.append(f"    color_column='{step['color_column']}' if '{step['color_column']}' else None,")
                    code.append(f"    size_column='{step['size_column']}' if '{step['size_column']}' else None,")
                    code.append(f"    color_scheme='{step['color_scheme']}'")
                    code.append(")")
                else:
                    code.append(f"fig = create_{step['plot_type'].lower().replace(' ', '_')}(")
                    code.append(f"    df, '{step['x_column']}', '{step['y_column']}',")
                    code.append(f"    color_column='{step['color_column']}' if '{step['color_column']}' else None,")
                    code.append(f"    color_scheme='{step['color_scheme']}'")
                    code.append(")")
                code.append("fig.show()")

        elif step['type'] == 'data_type_conversion':
            code.append(f"# Convert '{step['column']}' to {step['target_type']}")
            code.append("if use_polars:")
            if step['target_type'] == 'int':
                code.append(f"    df = df.with_columns(pl.col('{step['column']}').cast(pl.Int64, strict=False))")
            elif step['target_type'] == 'float':
                code.append(f"    df = df.with_columns(pl.col('{step['column']}').cast(pl.Float64, strict=False))")
            elif step['target_type'] == 'datetime':
                code.append(f"    df = df.with_columns(pl.col('{step['column']}').str.strptime(pl.Datetime, strict=False))")
            elif step['target_type'] == 'string':
                code.append(f"    df = df.with_columns(pl.col('{step['column']}').cast(pl.Utf8))")
            elif step['target_type'] == 'boolean':
                code.append(f"    df = df.with_columns(pl.col('{step['column']}').cast(pl.Boolean))")
            code.append("else:")
            if step['target_type'] == 'int':
                code.append(f"    df['{step['column']}'] = pd.to_numeric(df['{step['column']}'], errors='coerce').astype('Int64')")
            elif step['target_type'] == 'float':
                code.append(f"    df['{step['column']}'] = pd.to_numeric(df['{step['column']}'], errors='coerce')")
            elif step['target_type'] == 'datetime':
                code.append(f"    df['{step['column']}'] = pd.to_datetime(df['{step['column']}'], errors='coerce')")
            elif step['target_type'] == 'string':
                code.append(f"    df['{step['column']}'] = df['{step['column']}'].astype(str)")
            elif step['target_type'] == 'boolean':
                code.append(f"    df['{step['column']}'] = df['{step['column']}'].astype(bool)")

    # Add visualization helper functions
    code.extend([
        "\n# Visualization helper functions",
        "def create_histogram(df, column, color_scheme='YlOrRd', nbins=30):",
        "    colors = getattr(px.colors.sequential, color_scheme)",
        "    fig = px.histogram(df, x=column, nbins=nbins, color_discrete_sequence=colors)",
        "    fig.update_layout(",
        "        title=f'Histogram of {column}',",
        "        xaxis_title=column,",
        "        yaxis_title='Count',",
        "        hovermode='x',",
        "        template='plotly_white'",
        "    )",
        "    return fig",
        "",
        "def create_scatter_plot(df, x_column, y_column, color_column=None, size_column=None, color_scheme='YlOrRd'):",
        "    fig = px.scatter(",
        "        df, x=x_column, y=y_column,",
        "        color=color_column,",
        "        size=size_column,",
        "        color_discrete_sequence=getattr(px.colors.sequential, color_scheme)",
        "    )",
        "    fig.update_layout(",
        "        title=f'{y_column} vs {x_column}',",
        "        xaxis_title=x_column,",
        "        yaxis_title=y_column,",
        "        template='plotly_white'",
        "    )",
        "    return fig",
        "",
        "def create_correlation_heatmap(df, numeric_columns):",
        "    corr_matrix = df[numeric_columns].corr()",
        "    fig = px.imshow(",
        "        corr_matrix,",
        "        color_continuous_scale='RdBu',",
        "        aspect='auto'",
        "    )",
        "    fig.update_layout(",
        "        title='Correlation Heatmap',",
        "        template='plotly_white'",
        "    )",
        "    return fig"
    ])

    # Example visualization usage
    code.extend([
        "\n# Example visualizations",
        "# Create and display a histogram",
        "hist_fig = create_histogram(df, 'your_column_name', nbins=30)",
        "# hist_fig.show()  # Uncomment to display",
        "",
        "# Create and display a scatter plot",
        "scatter_fig = create_scatter_plot(df, 'x_column', 'y_column', color_column='category_column')",
        "# scatter_fig.show()  # Uncomment to display",
        "",
        "# Create and display a correlation heatmap",
        "numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()",
        "heatmap_fig = create_correlation_heatmap(df, numeric_cols)",
        "# heatmap_fig.show()  # Uncomment to display"
    ])

    # Add notes about resetting data
    code.extend([
        "\n# To reset to original data:",
        "# df = original_df.copy()",
        "",
        "# Note: When using in a notebook or script, remove the '#' from .show()",
        "# calls to display the visualizations"
    ])

    return "\n".join(code)

def main():
    """Main Streamlit application."""
    st.title("PandasCleaner")

    # Initialize session state just once and properly handle session persistence
    if 'initialized' not in st.session_state:
        init_session_state()
    else:
        # Ensure cleaning_steps exists even after rerun
        if 'cleaning_steps' not in st.session_state:
            st.session_state.cleaning_steps = []

    # Setup database connection
    conn = setup_database()

    # Sidebar
    with st.sidebar:
        st.header("Data Cleaning Controls")

        # File upload with multiple file types
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=list(SUPPORTED_FILE_TYPES.keys())
        )

        # Handle file upload and maintain session state
        if uploaded_file is not None:
            current_file = getattr(st.session_state, 'current_file', None)

            # Only reset if a new file is uploaded
            if current_file != uploaded_file.name:
                try:
                    # Load data using Polars
                    polars_df, file_type = load_dataframe(uploaded_file)

                    # Convert to pandas for compatibility with existing functions
                    pandas_df = polars_to_pandas(polars_df)

                    # Update session state
                    st.session_state.df = pandas_df
                    st.session_state.original_df = pandas_df.copy()
                    st.session_state.polars_df = polars_df  # Keep Polars DataFrame for optimized operations
                    st.session_state.cleaning_steps = []
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.file_type = file_type

                    st.success(f"Successfully loaded {SUPPORTED_FILE_TYPES[file_type]} file")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

        if st.session_state.df is not None:
            # Missing values handling
            st.subheader("Handle Missing Values")
            missing_strategy = st.selectbox(
                "Strategy",
                ["none", "drop", "fill_mean", "fill_median", "fill_mode", "fill_value"]
            )

            fill_value = None
            if missing_strategy == "fill_value":
                fill_value = st.number_input("Fill value", value=0)

            missing_values_button = st.button("Apply Missing Values Strategy")
            if missing_strategy != "none" and missing_values_button:
                # Store current steps before operation
                current_steps = st.session_state.get('cleaning_steps', []).copy()

                try:
                    # Convert to Polars, apply missing values strategy, and convert back
                    polars_df = pandas_to_polars(st.session_state.df)
                    processed_df = handle_missing_values(
                        polars_df,
                        missing_strategy,
                        fill_value
                    )
                    st.session_state.df = polars_to_pandas(processed_df)

                    # Add the operation to cleaning steps
                    cleaning_step = {
                        'type': 'missing_values',
                        'strategy': missing_strategy,
                        'value': fill_value
                    }

                    # Update cleaning steps while preserving previous steps
                    if 'cleaning_steps' not in st.session_state:
                        st.session_state.cleaning_steps = []
                    st.session_state.cleaning_steps = current_steps + [cleaning_step]
                    st.success(f"Missing values handled successfully using {missing_strategy}")
                except Exception as e:
                    st.error(f"Error handling missing values: {str(e)}")

            # Data type conversion
            st.subheader("Convert Data Types")
            conversion_col = st.selectbox(
                "Select column to convert",
                options=st.session_state.df.columns,
                key="conversion_col"
            )
            target_type = st.selectbox(
                "Target data type",
                options=["int", "float", "string", "datetime", "boolean"],
                key="target_type"
            )
            convert_type_button = st.button("Convert Data Type")
            if convert_type_button:
                # Store current steps before operation
                current_steps = st.session_state.get('cleaning_steps', []).copy()

                try:
                    # Convert to Polars, apply type conversion, and convert back
                    polars_df = pandas_to_polars(st.session_state.df)
                    processed_df = convert_data_types(
                        polars_df,
                        {conversion_col: target_type}
                    )
                    st.session_state.df = polars_to_pandas(processed_df)

                    # Add the operation to cleaning steps
                    cleaning_step = {
                        'type': 'data_type_conversion',
                        'column': conversion_col,
                        'target_type': target_type
                    }

                    # Update cleaning steps while preserving previous steps
                    if 'cleaning_steps' not in st.session_state:
                        st.session_state.cleaning_steps = []
                    st.session_state.cleaning_steps = current_steps + [cleaning_step]
                    st.success(f"Converted '{conversion_col}' to {target_type}")
                except Exception as e:
                    st.error(f"Error converting data type: {str(e)}")

            # Group by operations
            st.subheader("Group By")
            group_cols = st.multiselect(
                "Group by columns",
                options=st.session_state.df.columns
            )
            if group_cols:
                agg_func = st.selectbox(
                    "Aggregation function",
                    ["mean", "sum", "count", "min", "max"]
                )
                group_by_button = st.button("Apply Group By")
                if group_by_button:
                    # Store current steps before operation
                    current_steps = st.session_state.get('cleaning_steps', []).copy()

                    try:
                        # Convert to Polars, apply group by, and convert back
                        polars_df = pandas_to_polars(st.session_state.df)
                        processed_df = apply_group_by(
                            polars_df,
                            group_cols,
                            agg_func
                        )
                        st.session_state.df = polars_to_pandas(processed_df)

                        # Add the operation to cleaning steps
                        cleaning_step = {
                            'type': 'group_by',
                            'columns': group_cols,
                            'agg_func': agg_func
                        }

                        # Update cleaning steps while preserving previous steps
                        if 'cleaning_steps' not in st.session_state:
                            st.session_state.cleaning_steps = []
                        st.session_state.cleaning_steps = current_steps + [cleaning_step]
                        st.success(f"Group by applied successfully on {', '.join(group_cols)} with {agg_func}")
                    except Exception as e:
                        st.error(f"Error in group by operation: {str(e)}")

    # Main panel
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df)

        # Reset button
        if st.button("Reset to Original Data"):
            st.session_state.df = st.session_state.original_df.copy()
            if 'cleaning_steps' in st.session_state:
                st.session_state.cleaning_steps = []
            # st.write("Debug: Cleaning steps after reset:", st.session_state.cleaning_steps)
            st.success("Data reset to original state")

        # Filtering
        st.subheader("Filter Data")
        filter_query = st.text_input("Enter filter query (e.g., 'column > 5')")
        if filter_query:
            is_valid, error_msg, filtered_df = validate_and_apply_filter(pandas_to_polars(st.session_state.df), filter_query)
            if is_valid and filtered_df is not None:
                # Store current steps before operation
                current_steps = st.session_state.get('cleaning_steps', []).copy()

                # Update DataFrame in session state
                st.session_state.df = polars_to_pandas(filtered_df)

                # Add the operation to cleaning steps
                cleaning_step = {
                    'type': 'filter',
                    'query': filter_query
                }

                # Update cleaning steps while preserving previous steps
                if 'cleaning_steps' not in st.session_state:
                    st.session_state.cleaning_steps = []
                st.session_state.cleaning_steps = current_steps + [cleaning_step]
                st.success(f"Filter applied successfully: {filter_query}")
            elif is_valid:
                st.warning("Filter resulted in empty DataFrame")
            else:
                st.error(f"Invalid query: {error_msg}")

        # Data Visualization Section
        st.subheader("Data Visualization")

        # Plot type selection
        plot_type = st.selectbox("Select Plot Type", list(PLOT_TYPES.keys()))

        # Column selection based on plot type
        if plot_type == 'Correlation Heatmap':
            numeric_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            selected_cols = st.multiselect('Select numeric columns for correlation', numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
            if selected_cols:
                # Store current steps before visualization
                current_steps = st.session_state.get('cleaning_steps', []).copy()

                fig = create_correlation_heatmap(st.session_state.df, selected_cols)
                st.plotly_chart(fig, use_container_width=True)

                # Add visualization step
                viz_step = {
                    'type': 'visualization',
                    'plot_type': 'Correlation Heatmap',
                    'columns': selected_cols
                }
                st.session_state.cleaning_steps = current_steps + [viz_step]
        else:
            # Color scheme selection
            color_scheme = st.selectbox('Select Color Scheme', COLOR_SCHEMES)

            # Column selections based on plot type
            if plot_type in ['Histogram', 'Box Plot']:
                column = st.selectbox('Select Column', st.session_state.df.columns)
                if plot_type == 'Box Plot':
                    group_by = st.selectbox('Group By (Optional)', ['None'] + list(st.session_state.df.columns))
                    if group_by != 'None':
                        current_steps = st.session_state.get('cleaning_steps', []).copy()
                        fig = PLOT_TYPES[plot_type](st.session_state.df, column, group_by, color_scheme)

                        # Add visualization step
                        viz_step = {
                            'type': 'visualization',
                            'plot_type': plot_type,
                            'column': column,
                            'group_by': group_by,
                            'color_scheme': color_scheme
                        }
                        st.session_state.cleaning_steps = current_steps + [viz_step]
                    else:
                        current_steps = st.session_state.get('cleaning_steps', []).copy()
                        fig = PLOT_TYPES[plot_type](st.session_state.df, column, color_scheme=color_scheme)

                        # Add visualization step
                        viz_step = {
                            'type': 'visualization',
                            'plot_type': plot_type,
                            'column': column,
                            'color_scheme': color_scheme
                        }
                        st.session_state.cleaning_steps = current_steps + [viz_step]
                else:
                    nbins = st.slider('Number of bins', 5, 100, 30)
                    current_steps = st.session_state.get('cleaning_steps', []).copy()
                    fig = PLOT_TYPES[plot_type](st.session_state.df, column, color_scheme, nbins)

                    # Add visualization step
                    viz_step = {
                        'type': 'visualization',
                        'plot_type': plot_type,
                        'column': column,
                        'color_scheme': color_scheme,
                        'nbins': nbins
                    }
                    st.session_state.cleaning_steps = current_steps + [viz_step]

            elif plot_type in ['Scatter Plot', 'Line Plot', 'Bar Plot']:
                x_column = st.selectbox('Select X Column', st.session_state.df.columns)
                y_column = st.selectbox('Select Y Column', st.session_state.df.columns)
                color_column = st.selectbox('Color By (Optional)', ['None'] + list(st.session_state.df.columns))

                if plot_type == 'Scatter Plot':
                    size_column = st.selectbox('Size By (Optional)', ['None'] + list(st.session_state.df.columns))
                    current_steps = st.session_state.get('cleaning_steps', []).copy()
                    fig = PLOT_TYPES[plot_type](
                        st.session_state.df, x_column, y_column,
                        color_column if color_column != 'None' else None,
                        size_column if size_column != 'None' else None,
                        color_scheme
                    )

                    # Add visualization step
                    viz_step = {
                        'type': 'visualization',
                        'plot_type': plot_type,
                        'x_column': x_column,
                        'y_column': y_column,
                        'color_column': color_column if color_column != 'None' else None,
                        'size_column': size_column if size_column != 'None' else None,
                        'color_scheme': color_scheme
                    }
                    st.session_state.cleaning_steps = current_steps + [viz_step]
                else:
                    current_steps = st.session_state.get('cleaning_steps', []).copy()
                    fig = PLOT_TYPES[plot_type](
                        st.session_state.df, x_column, y_column,
                        color_column if color_column != 'None' else None,
                        color_scheme
                    )

                    # Add visualization step
                    viz_step = {
                        'type': 'visualization',
                        'plot_type': plot_type,
                        'x_column': x_column,
                        'y_column': y_column,
                        'color_column': color_column if color_column != 'None' else None,
                        'color_scheme': color_scheme
                    }
                    st.session_state.cleaning_steps = current_steps + [viz_step]

            # Display the plot
            if 'fig' in locals():
                st.plotly_chart(fig, use_container_width=True)

        # Export cleaning code
        st.subheader("Export Cleaning Code")
        if st.button("Generate Python Code"):
            # st.write("Debug: Cleaning steps before generating code:", st.session_state.cleaning_steps)
            if not st.session_state.cleaning_steps:
                st.warning("No cleaning steps have been applied yet.")
            else:
                code = export_cleaning_code(st.session_state.cleaning_steps)
                st.code(code, language="python")

        # Save session and show debug info
        # st.write("Debug: Final state before saving:", {
        #     "has_df": st.session_state.df is not None,
        #     "cleaning_steps_count": len(st.session_state.cleaning_steps),
        #     "cleaning_steps": st.session_state.cleaning_steps
        # })
        save_session(
            conn,
            st.session_state.session_id,
            st.session_state.df,
            st.session_state.cleaning_steps
        )

    # Close database connection
    conn.close()

if __name__ == "__main__":
    main()
