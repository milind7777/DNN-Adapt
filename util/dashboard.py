import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import datetime
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CSV_DATA_DIR = '/home/cching1/DNNAdapt/DNN-Adapt/util/csv_data_cache' # Read CSVs from here
REFRESH_INTERVAL_MS = 1000 # How often to refresh the dashboard

# CSV Filenames (must match those in log_csv_processor.py)
OVERALL_METRICS_CSV = os.path.join(CSV_DATA_DIR, "overall_metrics_live.csv")
MODEL_METRICS_CSV = os.path.join(CSV_DATA_DIR, "model_metrics_live.csv")
REQUEST_RATE_HISTORY_CSV = os.path.join(CSV_DATA_DIR, "request_rate_history.csv")
PERFORMANCE_HISTORY_CSV = os.path.join(CSV_DATA_DIR, "performance_history.csv")
RECENT_REQUESTS_DETAILS_CSV = os.path.join(CSV_DATA_DIR, "recent_requests_details.csv")


def read_csv_safely(file_path, default_df_cols=None):
    """Reads a CSV file safely, returning an empty DataFrame or default if error/not found."""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if df.empty and default_df_cols:
                 return pd.DataFrame(columns=default_df_cols)
            return df
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file {file_path} is empty.")
            return pd.DataFrame(columns=default_df_cols) if default_df_cols else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {e}")
            return pd.DataFrame(columns=default_df_cols) if default_df_cols else pd.DataFrame()
    else:
        logger.warning(f"CSV file {file_path} not found.")
        return pd.DataFrame(columns=default_df_cols) if default_df_cols else pd.DataFrame()


class CSVDashboard:
    def __init__(self):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="DNN-Adapt Monitor (CSV)"
        )
        self.app.layout = self._create_layout()
        self._setup_callbacks()

    def _create_layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("DNN-Adapt Real-Time Monitor ", className="mt-3 mb-2"),
                    html.Div(id="activity-indicator", className="mb-3") 
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader(html.H4("System Metrics", className="m-0")), dbc.CardBody(html.Div(id="metrics-display"))]), width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader(html.H4("Request Rate Patterns", className="m-0")), dbc.CardBody(dcc.Graph(id="request-rate-graph", style={"height": "350px"}))]), width=6),
                dbc.Col(dbc.Card([dbc.CardHeader(html.H4("Throughput & Performance", className="m-0")), dbc.CardBody(dcc.Graph(id="throughput-graph"))]), width=6) # throughput-graph already has height=350px set in its callback
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader(html.H4("SLO Violations by Model", className="m-0")), dbc.CardBody(dcc.Graph(id="slo-violation-bar", style={"height": "350px"}))]), width=6),
                dbc.Col(dbc.Card([dbc.CardHeader(html.H4("Request Processing Timeline (Recent)", className="m-0")), dbc.CardBody(dcc.Graph(id="request-timeline", style={"height": "350px"}))]), width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader(html.H4("Recent Requests Details", className="m-0")), dbc.CardBody(html.Div(id="recent-requests-table"))]), width=12)
            ]),
            dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL_MS, n_intervals=0)
        ], fluid=True)

    def _setup_callbacks(self):
        @self.app.callback(Output("activity-indicator", "children"), Input("interval-component", "n_intervals"))
        def update_activity_indicator(n):
            overall_metrics_df = read_csv_safely(OVERALL_METRICS_CSV)
            model_metrics_df = read_csv_safely(MODEL_METRICS_CSV)
            is_active = False
            last_update_str = "N/A"

            if not overall_metrics_df.empty and 'timestamp' in overall_metrics_df.columns:
                try:
                    last_timestamp_str = overall_metrics_df['timestamp'].iloc[-1]
                    last_update_dt = datetime.datetime.fromisoformat(last_timestamp_str)
                    if (datetime.datetime.now() - last_update_dt).total_seconds() < 5: # Active if CSV updated in last 5s
                        is_active = True
                    last_update_str = last_update_dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    logger.warning(f"Could not parse timestamp for activity: {e}")


            status_color = "#28a745" if is_active else "#dc3545"
            status_text = "Active" if is_active else "Inactive/Stale"
            
            models = []
            if not model_metrics_df.empty and 'model_name' in model_metrics_df.columns:
                models = model_metrics_df['model_name'].unique().tolist()
            
            model_text = f"Monitoring models: {', '.join(models)}" if models else "No models detected in CSV"
            
            return html.Div([
                html.Span("●", style={"color": status_color, "marginRight": "8px", "fontSize": "16px", "fontWeight": "bold"}),
                html.Span(f"CSV Monitor Status: {status_text} (Last CSV update: {last_update_str}) • ", style={"color": status_color, "fontWeight": "bold"}),
                html.Span(model_text)
            ])

        @self.app.callback(Output("metrics-display", "children"), Input("interval-component", "n_intervals"))
        def update_metrics_display(n):
            overall_df = read_csv_safely(OVERALL_METRICS_CSV, default_df_cols=['total_requests', 'processed_requests', 'avg_processing_time_us', 'recent_throughput_rps', 'slo_met_count', 'slo_violated_count'])
            model_df = read_csv_safely(MODEL_METRICS_CSV, default_df_cols=['model_name', 'request_count', 'processed_count', 'avg_processing_time_us', 'target_rate', 'slo_met_count', 'slo_violated_count'])

            cards = []
            # Overall system metrics
            if not overall_df.empty:
                metrics = overall_df.iloc[-1] # Get the last (and likely only) row
                cards.append(dbc.Row([
                    dbc.Col([html.H2(f"{metrics.get('total_requests', 0):,}", className="text-primary mb-0"), html.P("Total Requests", className="text-muted")], width=2),
                    dbc.Col([html.H2(f"{metrics.get('processed_requests', 0):,}", className="text-success mb-0"), html.P("Processed", className="text-muted")], width=2),
                    dbc.Col([html.H2(f"{metrics.get('avg_processing_time_us', 0):.2f}", className="text-info mb-0"), html.P("Avg. Time (µs)", className="text-muted")], width=2),
                    dbc.Col([html.H2(f"{metrics.get('recent_throughput_rps', 0):.2f}", className="text-warning mb-0"), html.P("Reqs/Sec", className="text-muted")], width=2),
                    dbc.Col([html.H2(f"{metrics.get('slo_met_count', 0):,}", className="text-success mb-0"), html.P("SLO Met", className="text-muted")], width=2),
                    dbc.Col([html.H2(f"{metrics.get('slo_violated_count', 0):,}", className="text-danger mb-0"), html.P("SLO Violated", className="text-muted")], width=2),
                ]))
            else:
                cards.append(html.P("Overall metrics data not available."))

            if not model_df.empty:
                cards.append(html.Hr(className="my-3"))
                model_rows = []
                for _, model_metrics in model_df.iterrows():
                    model_rows.append(dbc.Row([
                        dbc.Col([html.H5(model_metrics.get('model_name', 'N/A'), className="text-secondary mb-2")], width=12)
                    ])) # Keep mb-2 for model name heading for separation
                    model_rows.append(dbc.Row([
                        dbc.Col([html.H4(f"{model_metrics.get('request_count', 0):,}", className="text-primary mb-0"), html.P("Total Reqs", className="text-muted small")], width=2),
                        dbc.Col([html.H4(f"{model_metrics.get('processed_count', 0):,}", className="text-success mb-0"), html.P("Processed", className="text-muted small")], width=2),
                        dbc.Col([html.H4(f"{model_metrics.get('avg_processing_time_us', 0):.2f}", className="text-info mb-0"), html.P("Avg. Time (µs)", className="text-muted small")], width=2),
                        dbc.Col([html.H4(f"{model_metrics.get('target_rate', 0):.0f}", className="text-warning mb-0"), html.P("Target Rate", className="text-muted small")], width=2),
                        dbc.Col([html.H4(f"{model_metrics.get('slo_met_count', 0):,}", className="text-success mb-0"), html.P("SLO Met", className="text-muted small")], width=2),
                        dbc.Col([html.H4(f"{model_metrics.get('slo_violated_count', 0):,}", className="text-danger mb-0"), html.P("SLO Violated", className="text-muted small")], width=2),
                    ], className="mb-0")) # Reduced from mb-1 to mb-0 for even less space after each model's metrics
                cards.append(html.Div(model_rows))
            else:
                 cards.append(html.P("Model-specific metrics data not available."))
            return html.Div(cards)

        @self.app.callback(Output("request-rate-graph", "figure"), Input("interval-component", "n_intervals"))
        def update_request_rate_graph(n):
            df = read_csv_safely(REQUEST_RATE_HISTORY_CSV, default_df_cols=['timestamp', 'model_name', 'rate'])
            if df.empty or len(df) <= 1: return go.Figure().update_layout(title="Request Rate Over Time (No data)")
            
            fig = go.Figure()
            models = df['model_name'].unique()
            colors = px.colors.qualitative.Plotly
            
            df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure datetime for proper sorting/plotting
            df = df.sort_values(by='timestamp')


            for i, model in enumerate(models):
                model_df = df[df['model_name'] == model]
                if not model_df.empty:
                    fig.add_trace(go.Scatter(
                        x=model_df['timestamp'], y=model_df['rate'], mode='lines', name=f'{model} Rate',
                        line=dict(width=3, color=colors[i % len(colors)]), fill='tozeroy'
                    ))
            fig.update_layout(title="Request Rate Over Time", xaxis_title="Time", yaxis_title="Requests per Second", hovermode="x unified", legend=dict(x=0.01, y=0.99), margin=dict(l=50,r=50,t=50,b=50), transition_duration=300) # Removed height here, set in layout
            return fig

        @self.app.callback(Output("throughput-graph", "figure"), Input("interval-component", "n_intervals"))
        def update_throughput_graph(n):
            df = read_csv_safely(PERFORMANCE_HISTORY_CSV, default_df_cols=['timestamp', 'throughput_rps', 'avg_processing_time_us_overall', 'slo_met_count_cumulative', 'slo_violated_count_cumulative'])
            
            required_cols = ['timestamp', 'throughput_rps', 'avg_processing_time_us_overall'] # Keep primary required cols for plotting main lines
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"In 'update_throughput_graph': PERFORMANCE_HISTORY_CSV is missing columns: {missing_cols}. Check CSV generation or clear old CSVs.")
                return go.Figure().update_layout(title=f"System Performance Over Time (Error: Missing columns {missing_cols} in CSV)")

            if df.empty:
                 return go.Figure().update_layout(title="System Performance Over Time (No data)")

            # Attempt to convert to appropriate types, coercing errors to NaT/NaN
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['throughput_rps'] = pd.to_numeric(df['throughput_rps'], errors='coerce')
            df['avg_processing_time_us_overall'] = pd.to_numeric(df['avg_processing_time_us_overall'], errors='coerce')

            # Drop rows where any of the required columns became NaT/NaN after conversion, or were already NaN
            df_plottable = df.dropna(subset=required_cols)

            if df_plottable.empty or len(df_plottable) <= 1:
                return go.Figure().update_layout(title="System Performance Over Time (Insufficient valid data)")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            df_plottable = df_plottable.sort_values(by='timestamp')


            fig.add_trace(go.Scatter(x=df_plottable['timestamp'], y=df_plottable['throughput_rps'], mode='lines+markers', name='Throughput (req/s)', line=dict(color='#28a745')), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_plottable['timestamp'], y=df_plottable['avg_processing_time_us_overall'], mode='lines+markers', name='Avg. Processing Time (µs)', line=dict(color='#17a2b8')), secondary_y=True)
            
            fig.update_layout(title="System Performance Over Time", xaxis_title="Time", hovermode="x unified", legend=dict(orientation="h", y=1.1), margin=dict(l=50,r=50,t=50,b=50), transition_duration=300, height=350)
            fig.update_yaxes(title_text="Requests per Second", secondary_y=False)
            fig.update_yaxes(title_text="Avg. Processing Time (µs)", secondary_y=True)
            return fig

        @self.app.callback(Output("slo-violation-bar", "figure"), Input("interval-component", "n_intervals"))
        def update_slo_violation_bar(n):
            # For a line graph showing violations over time, we need to use the performance history data
            df = read_csv_safely(PERFORMANCE_HISTORY_CSV, default_df_cols=['timestamp', 'slo_met_count_cumulative', 'slo_violated_count_cumulative'])
            
            required_cols = ['timestamp', 'slo_met_count_cumulative', 'slo_violated_count_cumulative']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"In 'update_slo_violation_bar': PERFORMANCE_HISTORY_CSV is missing columns: {missing_cols}. Check CSV generation or clear old CSVs.")
                return go.Figure().update_layout(title=f"SLO Compliance Over Time (Error: Missing columns {missing_cols} in CSV)")

            if df.empty: 
                return go.Figure().update_layout(title="SLO Compliance Over Time (No data)")

            # Convert timestamp to datetime for proper plotting
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Convert to numeric, with NaN handling
            df['slo_met_count_cumulative'] = pd.to_numeric(df['slo_met_count_cumulative'], errors='coerce').fillna(0)
            df['slo_violated_count_cumulative'] = pd.to_numeric(df['slo_violated_count_cumulative'], errors='coerce').fillna(0)
            
            # Sort by timestamp to ensure proper line plotting
            df_plottable = df.dropna(subset=['timestamp']).sort_values('timestamp')

            if df_plottable.empty:
                return go.Figure().update_layout(title="SLO Compliance Over Time (Insufficient valid data)")

            # We'll also get model-specific data for the detailed view
            model_df = read_csv_safely(MODEL_METRICS_CSV, default_df_cols=['model_name', 'slo_met_count', 'slo_violated_count'])
            if not model_df.empty:
                model_df['slo_met_count'] = pd.to_numeric(model_df['slo_met_count'], errors='coerce').fillna(0)
                model_df['slo_violated_count'] = pd.to_numeric(model_df['slo_violated_count'], errors='coerce').fillna(0)
                model_df['total_count'] = model_df['slo_met_count'] + model_df['slo_violated_count']
                # Calculate violation percentage if there are any requests
                model_df['violation_pct'] = 0.0
                mask = model_df['total_count'] > 0
                if mask.any():
                    model_df.loc[mask, 'violation_pct'] = model_df.loc[mask, 'slo_violated_count'] / model_df.loc[mask, 'total_count'] * 100
            
            # Create figure with primary y-axis for cumulative counts
            fig = go.Figure()
            
            # Add traces for SLO met and violated counts
            fig.add_trace(go.Scatter(
                x=df_plottable['timestamp'], 
                y=df_plottable['slo_met_count_cumulative'], 
                mode='lines+markers', 
                name='SLO Met', 
                line=dict(color='#28a745', width=2),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=df_plottable['timestamp'], 
                y=df_plottable['slo_violated_count_cumulative'], 
                mode='lines+markers', 
                name='SLO Violated', 
                line=dict(color='#dc3545', width=2),
                marker=dict(size=6)
            ))
            
            # If we have model-specific data, add an annotation showing current state
            if not model_df.empty:
                annotation_text = "<br>".join([
                    f"{row['model_name']}: {row['violation_pct']:.1f}% violations ({int(row['slo_violated_count'])}/{int(row['total_count'])})"
                    for _, row in model_df.iterrows() if row['total_count'] > 0
                ])
                
                if annotation_text:
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=0.01, y=0.99,
                        text=f"<b>Current SLO Status:</b><br>{annotation_text}",
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="darkgrey",
                        borderwidth=1,
                        borderpad=4,
                        align="left"
                    )
            
            fig.update_layout(
                title="SLO Compliance Over Time",
                xaxis_title="Time",
                yaxis_title="Cumulative Count",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=50,r=50,t=50,b=50),
                transition_duration=300
            )
            
            return fig

        @self.app.callback(Output("request-timeline", "figure"), Input("interval-component", "n_intervals"))
        def update_request_timeline(n):
            df = read_csv_safely(RECENT_REQUESTS_DETAILS_CSV, default_df_cols=['arrival_time_readable', 'processing_time_us', 'model_name', 'status', 'gpu_id', 'batch_id', 'completion_time_readable', 'slo_status'])
            if df.empty: return go.Figure().update_layout(title="Request Processing Timeline (No data)")
            
            df_processed = df[df['status'] == 'Processed'].copy() # Make a copy
            if df_processed.empty: return go.Figure().update_layout(title="Request Processing Timeline (No processed requests in CSV)")

            # Ensure 'processing_time_us' column exists.
            # If not, check for 'processing_time_ms' and convert.
            if 'processing_time_us' not in df_processed.columns:
                if 'processing_time_ms' in df_processed.columns:
                    logger.warning("Found 'processing_time_ms' in recent_requests_details.csv for timeline. Converting to 'processing_time_us'.")
                    df_processed['processing_time_us'] = pd.to_numeric(df_processed['processing_time_ms'], errors='coerce') * 1000.0
                else:
                    logger.warning("Neither 'processing_time_us' nor 'processing_time_ms' found in recent_requests_details.csv. Processing times for timeline will be missing.")
                    df_processed['processing_time_us'] = pd.NA # Assign pd.NA to the column
            else: # Ensure it's numeric if it exists
                df_processed['processing_time_us'] = pd.to_numeric(df_processed['processing_time_us'], errors='coerce')


            df_processed['arrival_time_readable'] = pd.to_datetime(df_processed['arrival_time_readable'], errors='coerce')
            
            # Ensure model_name is also present for coloring, convert to string to be safe
            if 'model_name' not in df_processed.columns:
                df_processed['model_name'] = "Unknown" # Add a default if missing
            else:
                # Ensure model_name is string and fill NaNs if any, to prevent issues with color mapping
                df_processed['model_name'] = df_processed['model_name'].astype(str).fillna("Unknown")


            # Filter out rows where essential data for plotting might be NA
            df_plot_ready = df_processed.dropna(subset=['arrival_time_readable', 'processing_time_us', 'model_name'])
            
            # Also ensure processing_time_us is strictly positive for the 'size' aesthetic
            if not df_plot_ready.empty:
                df_plot_ready = df_plot_ready[df_plot_ready['processing_time_us'] > 0]


            if df_plot_ready.empty:
                 logger.info("Request Processing Timeline: df_plot_ready is empty after all filtering (processed, valid times, positive processing_time_us).")
                 return go.Figure().update_layout(title="Request Processing Timeline (No valid data for plotting)")

            df_plot_ready = df_plot_ready.sort_values('arrival_time_readable')
            
            hover_data_cols = ['model_name', 'gpu_id', 'batch_id', 'arrival_time_readable', 'completion_time_readable', 'processing_time_us', 'slo_status']
            # Ensure all hover_data_cols exist in df_plot_ready. read_csv_safely with default_df_cols should handle this.
            # If any are critical and might be missing, add them with default values here.
            for col in hover_data_cols:
                if col not in df_plot_ready.columns:
                    df_plot_ready[col] = pd.NA # Or an appropriate default like "N/A" for strings

            try:
                fig = px.scatter(df_plot_ready, 
                                 x='arrival_time_readable', 
                                 y='processing_time_us', 
                                 color='model_name', 
                                 size='processing_time_us', 
                                 size_max=15, 
                                 opacity=0.7,
                                 hover_data=hover_data_cols, # Pass the list of column names
                                 title="Request Processing Timeline (Recent Requests)",)
            except Exception as e:
                logger.error(f"Error creating scatter plot for request timeline: {e}")
                if not df_plot_ready.empty:
                    logger.error(f"df_plot_ready columns: {df_plot_ready.columns.tolist()}")
                    logger.error(f"df_plot_ready dtypes:\n{df_plot_ready.dtypes}")
                    logger.error(f"df_plot_ready head before error:\n{df_plot_ready.head().to_string()}")
                else:
                    logger.error("df_plot_ready was empty before px.scatter call.")
                return go.Figure().update_layout(title="Request Processing Timeline (Error during plot generation)")
            
            # Add average lines (optional, can be intensive if CSV is large)
            # Use df_plot_ready here as well
            # for model in df_plot_ready['model_name'].unique():
            #     model_data = df_plot_ready[df_plot_ready['model_name'] == model]
            #     if not model_data.empty:
            #         avg_time = model_data['processing_time_us'].mean() # Already uses _us
            #         if pd.notna(avg_time):
            #             fig.add_hline(y=avg_time, line_dash="dot", annotation_text=f"{model} avg: {avg_time:.2f} µs", annotation_position="bottom right")

            fig.update_layout(xaxis_title="Arrival Time", yaxis_title="Processing Time (µs)", hovermode="closest", margin=dict(l=50,r=50,t=50,b=50), legend=dict(orientation="h", y=1.1), transition_duration=300) # Removed height here, set in layout
            return fig

        @self.app.callback(Output("recent-requests-table", "children"), Input("interval-component", "n_intervals"))
        def update_recent_requests_table(n):
            df = read_csv_safely(RECENT_REQUESTS_DETAILS_CSV)
            if df.empty: return html.P("No recent requests data")

            # Handle potential old CSV format with 'processing_time_ms'
            if 'processing_time_us' not in df.columns and 'processing_time_ms' in df.columns:
                logger.warning("Found 'processing_time_ms' in recent_requests_details.csv for table. Converting to 'processing_time_us'.")
                df['processing_time_us'] = pd.to_numeric(df['processing_time_ms'], errors='coerce') * 1000.0
            elif 'processing_time_us' in df.columns: # Ensure it's numeric if it exists
                df['processing_time_us'] = pd.to_numeric(df['processing_time_us'], errors='coerce')


            display_cols = ['model_name', 'gpu_id', 'batch_id', 'arrival_time_readable', 'completion_time_readable', 'processing_time_us', 'status', 'slo_status']
            # Ensure all display_cols exist in df, adding them with None if not, to prevent KeyError
            for col in display_cols:
                if col not in df.columns:
                    df[col] = None
            
            df_display = df.head(15)[display_cols].copy() # Show top 15 from the CSV (which should be recent)
            
            df_display.columns = ['Model', 'GPU', 'Batch ID', 'Arrival', 'Completion', 'Proc. Time (µs)', 'Status', 'SLO Status']
            
            # Format processing time
            if 'Proc. Time (µs)' in df_display.columns:
                 df_display['Proc. Time (µs)'] = df_display['Proc. Time (µs)'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")


            table = dbc.Table.from_dataframe(df_display, striped=True, bordered=True, hover=True, className="table-sm")
            # Simple table for now, styling can be added as before if needed
            return table

    def run(self, debug=False, port=8051): # Changed port to avoid conflict if old monitor runs
        logger.info(f"Starting CSV Dashboard on http://127.0.0.1:{port}")
        self.app.run(debug=debug, port=port, host='0.0.0.0')

if __name__ == "__main__":
    dashboard = CSVDashboard()
    dashboard.run(debug=True)