import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime
import os
from typing import List, Dict, Any, Union, Optional

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

def handle_missing_values(df: pd.DataFrame, strategy: str, fill_value: Optional[Union[int, float, str]] = None) -> pd.DataFrame:
    """Handle missing values in the DataFrame."""
    if strategy == 'drop':
        return df.dropna()

    if strategy == 'fill_value' and fill_value is not None:
        return df.fillna(fill_value)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    result_df = df.copy()

    if strategy == 'fill_mean':
        result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].mean())
    elif strategy == 'fill_median':
        result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].median())
    elif strategy == 'fill_mode':
        # Handle numeric and non-numeric columns separately
        for col in numeric_cols:
            result_df[col] = result_df[col].fillna(result_df[col].mode().iloc[0] if not result_df[col].mode().empty else 0)
        for col in non_numeric_cols:
            result_df[col] = result_df[col].fillna(result_df[col].mode().iloc[0] if not result_df[col].mode().empty else '')

    return result_df

def validate_query(df: pd.DataFrame, query: str) -> tuple[bool, str]:
    """Validate a query string before applying it to the DataFrame."""
    try:
        # Check if query is empty
        if not query.strip():
            return False, "Query cannot be empty"

        # Try to execute the query on a copy of the DataFrame
        test_df = df.copy()
        test_df.query(query)
        return True, ""
    except Exception as e:
        return False, str(e)

def apply_group_by(df: pd.DataFrame, group_cols: List[str], agg_func: str) -> pd.DataFrame:
    """Apply groupby operation to the DataFrame."""
    if not group_cols:
        return df

    # Get numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Filter out group columns from numeric columns
    agg_cols = [col for col in numeric_cols if col not in group_cols]

    if not agg_cols:
        st.warning("No numeric columns available for aggregation")
        return df

    # Create aggregation dictionary
    agg_dict = {col: agg_func for col in agg_cols}

    try:
        return df.groupby(group_cols).agg(agg_dict).reset_index()
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
        "import numpy as np\n",
        "# Load your CSV file",
        "df = pd.read_csv('your_file.csv')\n",
        "# Track the original data",
        "original_df = df.copy()\n"
    ]

    for step in cleaning_steps:
        if step['type'] == 'missing_values':
            if step['strategy'] == 'drop':
                code.append("# Drop missing values")
                code.append("df = df.dropna()")
            else:
                code.append(f"# Fill missing values using {step['strategy']}")
                if step['strategy'] == 'fill_value':
                    code.append(f"df = df.fillna({step['value']})")
                elif step['strategy'] in ['fill_mean', 'fill_median']:
                    method = step['strategy'].split('_')[1]
                    code.append("# Handle numeric columns only")
                    code.append(f"numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns")
                    code.append(f"df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].{method}())")
                elif step['strategy'] == 'fill_mode':
                    code.append("# Handle numeric and non-numeric columns separately")
                    code.append("numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns")
                    code.append("non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns")
                    code.append("for col in numeric_cols:")
                    code.append("    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)")
                    code.append("for col in non_numeric_cols:")
                    code.append("    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else '')")
        elif step['type'] == 'group_by':
            code.append("# Group by operation")
            cols = ", ".join([f"'{col}'" for col in step['columns']])
            code.append("# Get numeric columns for aggregation")
            code.append("numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns")
            code.append("# Filter out group columns from numeric columns")
            code.append(f"agg_cols = [col for col in numeric_cols if col not in [{cols}]]")
            code.append("# Create aggregation dictionary")
            code.append(f"agg_dict = {{col: '{step['agg_func']}' for col in agg_cols}}")
            code.append(f"df = df.groupby([{cols}]).agg(agg_dict).reset_index()")
        elif step['type'] == 'filter':
            code.append("# Apply filter")
            escaped_query = escape_string(step['query'])
            code.append(f"df = df.query('{escaped_query}')")

    # Add a comment about resetting data if needed
    code.append("\n# To reset to original data:")
    code.append("# df = original_df.copy()")

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

        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        # Handle file upload and maintain session state
        if uploaded_file is not None:
            current_file = getattr(st.session_state, 'current_file', None)

            # Only reset if a new file is uploaded
            if current_file != uploaded_file.name:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.original_df = st.session_state.df.copy()
                st.session_state.cleaning_steps = []
                st.session_state.current_file = uploaded_file.name
                # st.write("Debug: New file uploaded, reset cleaning steps")

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

                st.session_state.df = handle_missing_values(
                    st.session_state.df,
                    missing_strategy,
                    fill_value
                )

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
                # st.write("Debug: Current cleaning steps:", st.session_state.cleaning_steps)
                st.success(f"Missing values handled successfully using {missing_strategy}")

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

                    st.session_state.df = apply_group_by(
                        st.session_state.df,
                        group_cols,
                        agg_func
                    )

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
                    # st.write("Debug: Current cleaning steps:", st.session_state.cleaning_steps)
                    st.success(f"Group by applied successfully on {', '.join(group_cols)} with {agg_func}")

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
            is_valid, error_msg = validate_query(st.session_state.df, filter_query)
            if is_valid:
                try:
                    filtered_df = st.session_state.df.query(filter_query)
                    if len(filtered_df) > 0:
                        st.session_state.df = filtered_df
                        # Store current steps before operation
                        current_steps = st.session_state.get('cleaning_steps', []).copy()

                        # Add the operation to cleaning steps
                        cleaning_step = {
                            'type': 'filter',
                            'query': filter_query
                        }

                        # Update cleaning steps while preserving previous steps
                        if 'cleaning_steps' not in st.session_state:
                            st.session_state.cleaning_steps = []
                        st.session_state.cleaning_steps = current_steps + [cleaning_step]
                        # st.write("Debug: Current cleaning steps:", st.session_state.cleaning_steps)
                        st.success(f"Filter applied successfully: {filter_query}")
                    else:
                        st.warning("Filter resulted in empty DataFrame")
                except Exception as e:
                    st.error(f"Error applying filter: {str(e)}")
            else:
                st.error(f"Invalid query: {error_msg}")

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
