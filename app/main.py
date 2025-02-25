# In main.py:

# At the top of main.py with other imports
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import time  # Add this line

from config import *
from data.loader import DataLoader
from models.evaluation import ModelEvaluator, plot_model_comparison, plot_predictions
from models.arima import ARIMAModel, SARIMAModel
from models.rf_xgb import RandomForestModel, XGBoostModel
from models.rnn_lstm import SimpleRNNModel, LSTMModel, StackedModel


def initialize_model(model_name, config):
    """Initialize model with given configuration"""
    if model_name == 'LSTM':
        model = LSTMModel(
            sequence_length=config['sequence_length'],
            n_features=1
        )
        model.build(units=config['units'])
    elif model_name == 'Simple RNN':
        model = SimpleRNNModel(
            sequence_length=config['sequence_length'],
            n_features=1
        )
        model.build(units=config['units'])
    elif model_name == 'Stacked LSTM+RNN':
        model = StackedModel(
            sequence_length=config['sequence_length'],
            n_features=1
        )
        model.build(lstm_units=config['units'])
    elif model_name == 'ARIMA':
        model = ARIMAModel(order=(config['p'], config['d'], config['q']))
    elif model_name == 'SARIMA':
        model = SARIMAModel(
            order=(config['p'], config['d'], config['q']),
            seasonal_order=(config['p'], config['d'], config['q'], config['s'])
        )
    elif model_name == 'Random Forest':
        model = RandomForestModel(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth']
        )
    else:  # XGBoost
        model = XGBoostModel(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth']
        )
    return model

def get_model_preprocessing_options(self, model_name):
    """Get model-specific preprocessing options"""
    options = {}
    if model_name in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']:
        options.update({
            'sequence_scaler': st.selectbox(
                f"{model_name} Sequence Scaler",
                ['StandardScaler', 'MinMaxScaler', 'None'],
                key=f"seq_scaler_{model_name}"
            ),
            'create_lags': st.checkbox(
                f"{model_name} Create Lagged Features",
                value=True,
                key=f"create_lags_{model_name}"
            ),
            'lag_length': st.number_input(
                f"{model_name} Lag Length",
                min_value=1, max_value=20, value=12,
                key=f"lag_length_{model_name}"
            )
        })
    return options


def initialize_app_state():
    """Initialize application state and session variables."""
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
        st.session_state.session_id = str(int(time.time()))

    if 'data_state' not in st.session_state:
        st.session_state.data_state = {
            'raw_data': None,
            'processed_data': None,
            'file_hash': None,
            'index_col': None,
            'target_col': None
        }


def create_sidebar():
    """Create and handle sidebar elements."""
    with st.sidebar:
        selected_metrics = st.multiselect(
            "Comparison Metrics",
            options=AVAILABLE_METRICS,
            default=DEFAULT_METRICS,
            key=f"metrics_select_{st.session_state.session_id}"
        )

        selected_models = st.multiselect(
            "Select Models (max 3)",
            options=ALL_MODELS,
            max_selections=3,
            key=f"models_select_{st.session_state.session_id}"
        )

        return selected_metrics, selected_models


def handle_data_upload():
    """Handle data upload and initial processing."""
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=['csv'],
        key=f"file_uploader_{st.session_state.session_id}"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data_state['raw_data'] = df
            st.session_state.data_state['file_hash'] = hash(uploaded_file.name)
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    return False


def main():
    st.set_page_config(layout="wide", page_title=APP_TITLE)
    st.title(APP_TITLE)

    # Initialize app state
    initialize_app_state()

    # Create sidebar and get selections
    selected_metrics, selected_models = create_sidebar()

    # Create main tabs
    main_tabs = st.tabs([
        "Data & Analysis",
        "Model Configuration",
        "Training Monitor",
        "Results & Comparison"
    ])

    # Initialize DataLoader
    data_loader = DataLoader()

    # Handle Data & Analysis Tab
    with main_tabs[0]:
        if handle_data_upload():
            df = st.session_state.data_state['raw_data']
            file_hash = st.session_state.data_state['file_hash']

            # Handle unnamed columns
            df = handle_unnamed_columns(df, file_hash)

            # Configure time index
            df, index_col = configure_time_index(df, file_hash)
            if index_col:
                st.session_state.data_state['index_col'] = index_col

            # Handle column management
            df = handle_column_management(df, file_hash)
            
            # Store processed data
            st.session_state.data_state['processed_data'] = df

            # Show data preview
            st.subheader("Processed Data Preview")
            st.markdown(df.head().to_html(escape=False), unsafe_allow_html=True)

            # Basic Statistics
            st.subheader("Basic Statistics")
            st.write(df.describe())

            if len(df.columns) > 0:

                # Allow target column selection
                target_col = st.selectbox(
                    "Select Target Column for Analysis",
                    df.columns,
                    key=f"target_select_{file_hash}"
                )

                st.session_state.data_state['target_col'] = target_col

                if target_col:

                    st.subheader("Data Visualization")

                    # Determine if index is a time-based index
                    is_time_index = isinstance(df.index, (pd.PeriodIndex, pd.DatetimeIndex)) or any(term in str(df.index.name).lower() for term in ['date', 'time', 'month', 'year'])

                    # Create visualization options
                    viz_options = st.expander("Visualization Options", expanded=True)
                    with viz_options:
                        # Plot type selection
                        plot_type = st.selectbox(
                            "Select Plot Type",
                            ["Line Plot", "Bar Chart", "Scatter Plot"],
                            key=f"plot_type_{file_hash}"
                        )
                        
                        # X-axis selection (index or column)
                        use_column_for_x = st.checkbox(
                            "Use Column for X-Axis Instead of Index", 
                            value=False,
                            key=f"use_col_x_{file_hash}"
                        )
                        
                        if use_column_for_x:
                            x_column = st.selectbox(
                                "Select X-Axis Column",
                                df.columns.tolist(),
                                key=f"x_column_{file_hash}"
                            )
                        
                        # Time formatting options
                        use_time_formatting = st.checkbox(
                            "Use Time-Based Formatting", 
                            value=is_time_index and not use_column_for_x,
                            key=f"time_format_{file_hash}",
                            disabled=use_column_for_x and not any(isinstance(df[x_column], (pd.DatetimeIndex, pd.PeriodIndex)))
                        )
                        
                        if use_time_formatting:
                            col1, col2 = st.columns(2)
                            with col1:
                                tick_count = st.slider(
                                    "Number of Ticks", 
                                    min_value=5, 
                                    max_value=50, 
                                    value=20,
                                    key=f"ticks_{file_hash}"
                                )
                            with col2:
                                tick_angle = st.slider(
                                    "Tick Angle", 
                                    min_value=0, 
                                    max_value=90, 
                                    value=45,
                                    key=f"angle_{file_hash}"
                                )
                        
                        # Data sampling for large datasets
                        if len(df) > 1000:
                            use_sampling = st.checkbox(
                                f"Sample Data (Dataset has {len(df)} rows)", 
                                value=True,
                                key=f"sample_{file_hash}"
                            )
                            if use_sampling:
                                sample_size = st.slider(
                                    "Sample Size", 
                                    min_value=100, 
                                    max_value=min(1000, len(df)), 
                                    value=500,
                                    key=f"sample_size_{file_hash}"
                                )
                                df_plot = df.sample(sample_size) if not is_time_index else df.iloc[::max(1, len(df)//sample_size)]
                            else:
                                df_plot = df
                        else:
                            df_plot = df

                    # Create the plot
                    fig = go.Figure()

                    # Prepare x values based on configuration
                    if use_column_for_x:
                        x_values = df_plot[x_column]
                        x_title = x_column
                    elif use_time_formatting and isinstance(df_plot.index, pd.PeriodIndex):
                        x_values = df_plot.index.strftime('%b %Y')  # Format as 'Jan 2023', 'Feb 2023', etc.
                        x_title = "Time"
                    else:
                        x_values = df_plot.index
                        x_title = "Index" if not is_time_index else "Time"

                    # Add appropriate trace based on plot type
                    if plot_type == "Line Plot":
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=df_plot[target_col],
                            mode='lines',
                            name=target_col
                        ))
                    elif plot_type == "Bar Chart":
                        fig.add_trace(go.Bar(
                            x=x_values,
                            y=df_plot[target_col],
                            name=target_col
                        ))
                    else:  # Scatter Plot
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=df_plot[target_col],
                            mode='markers',
                            name=target_col
                        ))

                    # Configure layout based on selected options
                    layout_args = {
                        "title": f"{target_col} - {plot_type}",
                        "xaxis_title": x_title,
                        "yaxis_title": target_col
                    }

                    # Add time-based formatting if selected
                    if use_time_formatting:
                        layout_args["xaxis"] = dict(
                            type='category',
                            tickmode='auto',
                            nticks=tick_count,
                            tickangle=tick_angle
                        )

                    fig.update_layout(**layout_args)
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.data_state['target_col'] = target_col

                    if target_col:
                        # Plot time series
                        st.subheader("Time Series Plot")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df[target_col],
                            mode='lines',
                            name=target_col
                        ))
                        fig.update_layout(
                            title=f"{target_col} Over Time",
                            xaxis_title="Time",
                            yaxis_title=target_col
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # Handle Model Configuration Tab
    with main_tabs[1]:
        if st.session_state.data_state['processed_data'] is not None:
            st.subheader("Model Configuration")
            
            for model in selected_models:
                st.write(f"### {model} Configuration")
                config = {}
                
                if model in ['ARIMA', 'SARIMA']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        config['p'] = st.number_input(f"{model} p", 0, 10, 1)
                    with col2:
                        config['d'] = st.number_input(f"{model} d", 0, 10, 1)
                    with col3:
                        config['q'] = st.number_input(f"{model} q", 0, 10, 1)
                    if model == 'SARIMA':
                        config['s'] = st.number_input(f"{model} Seasonal Period", 0, 52, 12)
                
                elif model in ['Random Forest', 'XGBoost']:
                    col1, col2 = st.columns(2)
                    with col1:
                        config['n_estimators'] = st.number_input(
                            f"{model} Number of Estimators",
                            10, 1000, 100
                        )
                    with col2:
                        config['max_depth'] = st.number_input(
                            f"{model} Max Depth",
                            1, 50, 10
                        )
                
                elif model in ['Simple RNN', 'LSTM', 'Stacked LSTM+RNN']:
                    col1, col2 = st.columns(2)
                    with col1:
                        config['sequence_length'] = st.number_input(
                            f"{model} Sequence Length",
                            1, 50, 10
                        )
                    with col2:
                        config['units'] = st.number_input(
                            f"{model} Units",
                            1, 200, 64
                        )
                
                st.session_state[f"{model}_config"] = config

    # Handle Training Monitor Tab
    with main_tabs[2]:
        if st.session_state.data_state['processed_data'] is not None and selected_models:
            st.subheader("Model Training")
            
            if st.button("Train Selected Models"):
                for model_name in selected_models:
                    config = st.session_state.get(f"{model_name}_config", {})
                    model = initialize_model(model_name, config)
                    
                    # Get the data
                    data = st.session_state.data_state['processed_data']
                    target_col = st.session_state.data_state['target_col']
                    
                    with st.spinner(f"Training {model_name}..."):
                        # Train the model
                        if model_name in ['ARIMA', 'SARIMA']:
                            model.train(data[target_col])
                        else:
                            # Prepare data for ML models
                            X, y = model.prepare_data(data[target_col], config.get('sequence_length', 10))
                            model.train(X, y)
                        
                        st.session_state.trained_models[model_name] = model
                        st.success(f"{model_name} trained successfully!")

    # Handle Results & Comparison Tab
    with main_tabs[3]:
        if st.session_state.trained_models:
            st.subheader("Model Comparison")
            
            evaluator = ModelEvaluator(metrics=selected_metrics)
            
            # Get predictions from all models
            predictions = {}
            data = st.session_state.data_state['processed_data']
            target_col = st.session_state.data_state['target_col']
            
            for model_name, model in st.session_state.trained_models.items():
                if model_name in ['ARIMA', 'SARIMA']:
                    pred = model.predict(steps=len(data))
                else:
                    X, _ = model.prepare_data(data[target_col], 
                                           st.session_state[f"{model_name}_config"]['sequence_length'])
                    pred = model.predict(X)
                predictions[model_name] = pred
            
            # Display results
            evaluator.display_results(
                data[target_col],
                predictions,
                display_type="Both"
            )


def handle_unnamed_columns(df, file_hash):
    """Handle unnamed columns in the dataset."""
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        st.warning(f"Found {len(unnamed_cols)} unnamed columns")

        handle_unnamed = st.radio(
            "How to handle unnamed columns?",
            ["Rename", "Drop"],
            key=f"unnamed_action_{file_hash}"
        )

        if handle_unnamed == "Rename":
            col_renames = {}
            for i, col in enumerate(unnamed_cols):
                new_name = st.text_input(
                    f"New name for {col}",
                    value=f"col_{i}",
                    key=f"rename_unnamed_{i}_{file_hash}"
                )
                col_renames[col] = new_name

            if st.button("Apply Column Renaming", key=f"apply_rename_{file_hash}"):
                df = df.rename(columns=col_renames)
                st.success("Columns renamed successfully!")
        else:
            if st.button("Drop Unnamed Columns", key=f"drop_unnamed_{file_hash}"):
                df = df.drop(columns=unnamed_cols)
                st.success("Unnamed columns dropped!")

    return df


def configure_time_index(df, file_hash):
    """Configure time index for the dataset."""
    st.write("### Time Index Configuration")

    date_cols = [col for col in df.columns
                 if any(term in str(col).lower()
                        for term in ['date', 'time', 'month', 'year'])]

    if not date_cols:
        date_cols = df.columns.tolist()
        st.warning("No date/time columns automatically detected")

    index_col = st.selectbox(
        "Select Time Index Column",
        date_cols,
        key=f"time_index_select_{file_hash}"
    )

    if index_col:
        freq_options = {
            'B': 'Business Day',
            'D': 'Calendar Day',
            'W': 'Weekly',
            'M': 'Monthly',
            'Q': 'Quarterly',
            'Y': 'Yearly'
        }

        col1, col2 = st.columns(2)
        with col1:
            freq = st.selectbox(
                "Select Frequency",
                options=list(freq_options.keys()),
                format_func=lambda x: freq_options[x],
                key=f"freq_select_{file_hash}"
            )

        with col2:
            period_anchor = None
            if freq in ['W', 'M', 'Q']:
                period_anchor = st.selectbox(
                    "Period Anchor",
                    ['Start', 'End'],
                    key=f"anchor_select_{file_hash}"
                )

        if st.button("Apply Period Index", key=f"apply_period_{file_hash}"):
            try:
                df = convert_to_period_index(df, index_col, freq, period_anchor)
                st.success("Period Index configured successfully!")
                return df, index_col
            except Exception as e:
                st.error(f"Error configuring period index: {str(e)}")

    return df, None


def convert_to_period_index(df, index_col, freq, period_anchor=None):
    """Convert dataframe index to period index."""
    df = df.copy()
    df.index = pd.to_datetime(df[index_col])

    if freq in ['W', 'M', 'Q'] and period_anchor == 'End':
        offset_map = {'W': 'W-SAT', 'M': 'M', 'Q': 'Q'}
        df.index = df.index + pd.offsets.to_offset(offset_map[freq])

    df.index = df.index.to_period(freq)
    df = df.drop(columns=[index_col])

    return df


def handle_column_management(df, file_hash):
    """Handle column selection and renaming."""
    if st.checkbox("Show Column Management", key=f"show_col_mgmt_{file_hash}"):
        st.write("### Column Management")

        # Add the date creation feature here, before the column selection
        if "year_sold" in df.columns and "month_sold" in df.columns:
            if st.checkbox("Create Date Column from Year and Month", key=f"create_date_{file_hash}"):
                # Create a proper date column by combining year and month
                # Using day=1 as a placeholder since we only care about month-level data
                df['sale_date'] = pd.to_datetime(df['year_sold'].astype(str) + '-' + 
                                               df['month_sold'].astype(str) + '-01')
                st.success("Created 'sale_date' column")

        # Original code continues below
        selected_cols = st.multiselect(
            "Select and Reorder Columns",
            df.columns.tolist(),
            default=df.columns.tolist(),
            key=f"col_select_{file_hash}"
        )

        if selected_cols:
            df = df[selected_cols]

        if st.checkbox("Rename Columns", key=f"show_rename_{file_hash}"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                col_to_rename = st.selectbox(
                    "Select Column",
                    df.columns.tolist(),
                    key=f"rename_select_{file_hash}"
                )
            with col2:
                new_name = st.text_input(
                    "New Name",
                    value=col_to_rename,
                    key=f"new_name_{file_hash}"
                )
            with col3:
                if st.button("Rename", key=f"rename_btn_{file_hash}"):
                    df = df.rename(columns={col_to_rename: new_name})
                    st.success(f"Renamed {col_to_rename} to {new_name}")

    return df

if __name__ == "__main__":
    main()