import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime
import os

def init_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'cleaning_steps' not in st.session_state:
        st.session_state.cleaning_steps = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

def save_session(conn, session_id, df, cleaning_steps):
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

def load_session(conn, session_id):
    """Load session from SQLite database."""
    cursor = conn.cursor()
    cursor.execute('SELECT data, cleaning_steps FROM sessions WHERE session_id = ?', (session_id,))
    result = cursor.fetchone()
    if result:
        df = pd.read_json(result[0])
        cleaning_steps = json.loads(result[1])
        return df, cleaning_steps
    return None, []

def setup_database():
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

def handle_missing_values(df, strategy, fill_value=None):
    """Handle missing values in the DataFrame."""
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill_mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == 'fill_median':
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == 'fill_mode':
        df = df.fillna(df.mode().iloc[0])
    elif strategy == 'fill_value' and fill_value is not None:
        df = df.fillna(fill_value)
    return df

def apply_group_by(df, group_cols, agg_func):
    """Apply groupby operation to the DataFrame."""
    if not group_cols:
        return df
    return df.groupby(group_cols).agg(agg_func).reset_index()

def export_cleaning_code(cleaning_steps):
    """Generate Python code for the cleaning steps."""
    code = [
        "import pandas as pd",
        "import numpy as np\n",
        "# Load your CSV file",
        "df = pd.read_csv('your_file.csv')\n"
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
                else:
                    method = step['strategy'].split('_')[1]
                    code.append(f"df = df.fillna(df.{method}(numeric_only=True))")
        elif step['type'] == 'group_by':
            code.append("# Group by operation")
            cols = ", ".join([f"'{col}'" for col in step['columns']])
            code.append(f"df = df.groupby([{cols}]).agg('{step['agg_func']}').reset_index()")
        elif step['type'] == 'filter':
            code.append("# Apply filter")
            code.append(f"df = df.query('{step['query']}')")

    return "\n".join(code)

def main():
    """Main Streamlit application."""
    st.title("PandasCleaner")

    # Initialize session state and database
    init_session_state()
    conn = setup_database()

    # Sidebar
    with st.sidebar:
        st.header("Data Cleaning Controls")

        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.cleaning_steps = []

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

            if missing_strategy != "none":
                if st.button("Apply Missing Values Strategy"):
                    st.session_state.df = handle_missing_values(
                        st.session_state.df,
                        missing_strategy,
                        fill_value
                    )
                    st.session_state.cleaning_steps.append({
                        'type': 'missing_values',
                        'strategy': missing_strategy,
                        'value': fill_value
                    })

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
                if st.button("Apply Group By"):
                    st.session_state.df = apply_group_by(
                        st.session_state.df,
                        group_cols,
                        agg_func
                    )
                    st.session_state.cleaning_steps.append({
                        'type': 'group_by',
                        'columns': group_cols,
                        'agg_func': agg_func
                    })

    # Main panel
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df)

        # Filtering
        st.subheader("Filter Data")
        filter_query = st.text_input("Enter filter query (e.g., 'column > 5')")
        if filter_query:
            try:
                filtered_df = st.session_state.df.query(filter_query)
                st.session_state.df = filtered_df
                st.session_state.cleaning_steps.append({
                    'type': 'filter',
                    'query': filter_query
                })
                st.success("Filter applied successfully")
            except Exception as e:
                st.error(f"Error applying filter: {str(e)}")

        # Export cleaning code
        st.subheader("Export Cleaning Code")
        if st.button("Generate Python Code"):
            code = export_cleaning_code(st.session_state.cleaning_steps)
            st.code(code, language="python")

        # Save session
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
