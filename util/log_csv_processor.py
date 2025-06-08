import re
import time
import os
import threading
import pandas as pd
import datetime
from collections import defaultdict, deque
import logging
import shutil
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#  Configuration 
LOG_FILE_PATH = '/home/mpenumat/DNN-Adapt/logs/cpp_log/dnn_adapt.log'
CSV_OUTPUT_DIR = '/home/mpenumat/DNN-Adapt/util/csv_data_cache' # Ensure this directory exists
MAX_HISTORY_CSV_ROWS = 5000 # For history CSVs to prevent them from growing indefinitely
RECENT_REQUESTS_LIMIT = 200 # For recent_requests_details.csv

# SLO Configuration (in microseconds)
try:
    with open('/home/mpenumat/DNN-Adapt/util/slo_config.json', 'r') as f:
        slo_config = json.load(f)
    MODEL_SLOS_US = slo_config.get('model_slos_us', {})
    DEFAULT_SLO_US = slo_config.get('default_slo_us', 1000)
    logger.info(f"Loaded SLO configuration: {MODEL_SLOS_US}, default: {DEFAULT_SLO_US}")
except Exception as e:
    logger.warning(f"Failed to load SLO configuration: {e}")
    # Fallback to hardcoded values
    MODEL_SLOS_US = {
        "resnet18": 700,
        "vit16": 1000,
        "efficientnetb0": 700
    }
    DEFAULT_SLO_US = 1000
    logger.info(f"Using fallback SLO configuration: {MODEL_SLOS_US}, default: {DEFAULT_SLO_US}")

# CSV Filenames
OVERALL_METRICS_CSV = os.path.join(CSV_OUTPUT_DIR, "overall_metrics_live.csv")
MODEL_METRICS_CSV = os.path.join(CSV_OUTPUT_DIR, "model_metrics_live.csv")
REQUEST_RATE_HISTORY_CSV = os.path.join(CSV_OUTPUT_DIR, "request_rate_history.csv")
PERFORMANCE_HISTORY_CSV = os.path.join(CSV_OUTPUT_DIR, "performance_history.csv")
RECENT_REQUESTS_DETAILS_CSV = os.path.join(CSV_OUTPUT_DIR, "recent_requests_details.csv")
SECOND_VIOLATIONS_CSV = os.path.join(CSV_OUTPUT_DIR, "second_violations.csv")
SECOND_REQUESTS_CSV = os.path.join(CSV_OUTPUT_DIR, "second_requests.csv")

class LogToCsvProcessor:
    def __init__(self, log_path, csv_dir, max_history_rows=MAX_HISTORY_CSV_ROWS, recent_requests_limit=RECENT_REQUESTS_LIMIT):
        self.log_path = log_path
        self.csv_dir = csv_dir
        self.max_history_rows = max_history_rows
        self.recent_requests_limit = recent_requests_limit

        

        os.makedirs(self.csv_dir, exist_ok=True)

        self.requests = {}
        self.batches = {}
        self.pending_requests = defaultdict(list)
        self.processed_requests_details = deque(maxlen=self.recent_requests_limit)

        self.last_log_position = 0
        self.last_update_time = time.time()

        self.request_rate_history_data = deque() # Store tuples (timestamp, model_name, rate)
        self.performance_history_data = deque() # Store tuples (timestamp, total_req, proc_req, avg_time_ns, throughput)

        self.metrics = {
            'total_requests': 0,
            'processed_requests': 0,
            'avg_processing_time_ns': 0, # Store in nanoseconds for precision (?)
            'recent_throughput': 0,
            'slo_met_count': 0,
            'slo_violated_count': 0,
            'models': defaultdict(lambda: {
                'request_count': 0,
                'processed_count': 0,
                'avg_processing_time_ns': 0, # Store in nanoseconds for precision(?)
                'target_rate': 0,
                'slo_met_count': 0,
                'slo_violated_count': 0
            })
        }
        self.models_seen = set()

        self.re_request = re.compile(r'\[RequestProcessor\] \[info\] (.+?): request_count: (\d+), arrival_time: (\d+)')
        self.re_batch_formed = re.compile(r'\[RequestProcessor\] \[info\] BATCH FORMED: id:(.+?) size: (\d+)')
        self.re_batch_processed = re.compile(r'\[CudaCallback\] \[info\] BATCH PROCESSED: (.+?) @ (\d+)')
        self.re_simulate_rate = re.compile(r'\[Simulator\] \[trace\] Simulating rate: (\d+) for model (.+)')
        # Add new regex to match SLO violation warnings 
        self.re_slo_violation = re.compile(r'\[RequestProcessor\] \[warning\] SLO VIOLATED: model_name:(.+?) request_count:(\d+) time_now:(\d+)')
        # Add regex to match new REQUEST RECEIVED logs
        self.re_request_received = re.compile(r'\[RequestProcessor\] \[info\] REQUEST RECEIVED: model_name:(.+?) request_count:(\d+) time_now:(\d+)')
        
        self.running = False
        self.monitor_thread = None
        self.csv_writer_thread = None
        
        self.processed_in_last_second_count = 0
        self.last_throughput_calc_time = time.time()

        # Add data structure to track per-second SLO violations
        self.second_violations = defaultdict(lambda: defaultdict(int))  # {timestamp: {model: violation_count}}
        self.current_second = None
        
        # Add data structure to track per-second requests
        self.second_requests = defaultdict(lambda: defaultdict(int))  # {timestamp: {model: request_count}}
        
        # Add flag for second violations CSV header
        self._second_violations_header_written = False
        # Add flag for second requests CSV header
        self._second_requests_header_written = False


    def _atomic_write_csv(self, df, final_path):
        """Writes a DataFrame to a CSV file atomically."""
        temp_path = final_path + ".tmp"
        try:
            df.to_csv(temp_path, index=False)
            shutil.move(temp_path, final_path)
        except Exception as e:
            logger.error(f"Error writing CSV {final_path}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _append_to_csv(self, data_list, file_path, columns, is_header_written_func, set_header_written_func):
        """Appends a list of dictionaries to a CSV file, managing history length."""
        if not data_list:
            return

        df_new_data = pd.DataFrame(data_list)
        
        # Pre-aggregate data by timestamp and model name if those columns exist
        if 'timestamp' in df_new_data.columns and 'model_name' in df_new_data.columns:
            if 'rate' in df_new_data.columns:  # For request rate data
                df_new_data = df_new_data.groupby(['timestamp', 'model_name'])['rate'].last().reset_index()
            elif 'request_count' in df_new_data.columns:  # For request count data
                df_new_data = df_new_data.groupby(['timestamp', 'model_name'])['request_count'].sum().reset_index()
            elif 'violation_count' in df_new_data.columns:  # For violation count data
                df_new_data = df_new_data.groupby(['timestamp', 'model_name'])['violation_count'].sum().reset_index()
    
        # Check if file exists and needs truncation or header
        if os.path.exists(file_path):
            try:
                df_existing = pd.read_csv(file_path)
                if len(df_existing) + len(df_new_data) > self.max_history_rows:
                    # Keep only the most recent rows if limit is exceeded
                    rows_to_keep = self.max_history_rows - len(df_new_data)
                    df_combined = pd.concat([df_existing.tail(max(0, rows_to_keep)), df_new_data], ignore_index=True)
                else:
                    df_combined = pd.concat([df_existing, df_new_data], ignore_index=True)
                
                self._atomic_write_csv(df_combined, file_path)

            except pd.errors.EmptyDataError: # File exists but is empty
                 self._atomic_write_csv(df_new_data, file_path)
                 set_header_written_func()
            except Exception as e:
                logger.error(f"Error reading or appending to {file_path}: {e}")
                # Fallback to writing new data only
                self._atomic_write_csv(df_new_data, file_path)
                set_header_written_func()

        else: # File does not exist
            self._atomic_write_csv(df_new_data, file_path)
            set_header_written_func()
        
        data_list.clear() # Clear the list after writing

    _request_rate_header_written = False
    _performance_header_written = False

    def _is_request_rate_header_written(self): return self._request_rate_header_written
    def _set_request_rate_header_written(self): self._request_rate_header_written = True
    def _is_performance_header_written(self): return self._performance_header_written
    def _set_performance_header_written(self): self._performance_header_written = True

    # Add helper methods for second violations CSV
    def _is_second_violations_header_written(self): return self._second_violations_header_written
    def _set_second_violations_header_written(self): self._second_violations_header_written = True

    # Add helper methods for second requests CSV
    def _is_second_requests_header_written(self): return self._second_requests_header_written
    def _set_second_requests_header_written(self): self._second_requests_header_written = True

    def _write_csv_files(self):
        """Periodically writes current data to CSV files."""
        # Overall Metrics
        overall_df = pd.DataFrame([{
            'timestamp': datetime.datetime.now().isoformat(),
            'total_requests': self.metrics['total_requests'],
            'processed_requests': self.metrics['processed_requests'],
            'avg_processing_time_us': self.metrics['avg_processing_time_ns'] /1000 if self.metrics['processed_requests'] > 0 else 0,
            'recent_throughput_rps': self.metrics['recent_throughput'],
            'slo_met_count': self.metrics['slo_met_count'],
            'slo_violated_count': self.metrics['slo_violated_count']
        }])
        self._atomic_write_csv(overall_df, OVERALL_METRICS_CSV)

        # Model Metrics
        model_data = []
        for model_name, m_metrics in self.metrics['models'].items():
            model_data.append({
                'model_name': model_name,
                'request_count': m_metrics['request_count'],
                'processed_count': m_metrics['processed_count'],
                'avg_processing_time_us': m_metrics['avg_processing_time_ns'] /1000 if m_metrics['processed_count'] > 0 else 0,
                'target_rate': m_metrics['target_rate'],
                'slo_met_count': m_metrics['slo_met_count'],
                'slo_violated_count': m_metrics['slo_violated_count']
            })
        if model_data:
            model_df = pd.DataFrame(model_data)
            self._atomic_write_csv(model_df, MODEL_METRICS_CSV)

        # Recent Requests Details
        if self.processed_requests_details:
            recent_req_df = pd.DataFrame(list(self.processed_requests_details))
            self._atomic_write_csv(recent_req_df, RECENT_REQUESTS_DETAILS_CSV)
        
        # Append to history CSVs
        if self.request_rate_history_data:
            # Pre-process request rate data to ensure one entry per second per model
            request_rate_df = pd.DataFrame(list(self.request_rate_history_data))
            if not request_rate_df.empty:
                # Group by timestamp (at second precision) and model_name, taking the last rate value
                request_rate_df = request_rate_df.groupby(['timestamp', 'model_name'])['rate'].last().reset_index()
                
                # If the file exists, read it and ensure we don't duplicate seconds
                if os.path.exists(REQUEST_RATE_HISTORY_CSV):
                    try:
                        existing_df = pd.read_csv(REQUEST_RATE_HISTORY_CSV)
                        
                        # Convert timestamps if they're strings
                        if not pd.api.types.is_datetime64_any_dtype(existing_df['timestamp']):
                            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Combine and re-aggregate to ensure we truly have only one entry per second per model
                        combined_df = pd.concat([existing_df, request_rate_df], ignore_index=True)
                        request_rate_df = combined_df.groupby(['timestamp', 'model_name'])['rate'].last().reset_index()
                        
                        # Sort by timestamp and model_name for readability
                        request_rate_df = request_rate_df.sort_values(['timestamp', 'model_name'])
                        
                        # Limit the file size if needed
                        if len(request_rate_df) > self.max_history_rows:
                            request_rate_df = request_rate_df.tail(self.max_history_rows)
                        
                        # Write directly (not using _append_to_csv since we already did the aggregation)
                        self._atomic_write_csv(request_rate_df, REQUEST_RATE_HISTORY_CSV)
                    except Exception as e:
                        logger.error(f"Error handling existing request rate CSV: {e}")
                        # Fallback: Use _append_to_csv
                        self._append_to_csv(
                            request_rate_df.to_dict('records'),
                            REQUEST_RATE_HISTORY_CSV,
                            ['timestamp', 'model_name', 'rate'],
                            self._is_request_rate_header_written,
                            self._set_request_rate_header_written
                        )
                else:
                    # If no file exists, just write this aggregated data
                    self._atomic_write_csv(request_rate_df, REQUEST_RATE_HISTORY_CSV)
                    self._set_request_rate_header_written()
            self.request_rate_history_data.clear()


        if self.performance_history_data:
            self._append_to_csv(
                list(self.performance_history_data),
                PERFORMANCE_HISTORY_CSV,
                ['timestamp', 'total_requests_cumulative', 'processed_requests_cumulative', 'avg_processing_time_us_overall', 'throughput_rps', 'slo_met_count_cumulative', 'slo_violated_count_cumulative'],
                self._is_performance_header_written,
                self._set_performance_header_written
            )
            self.performance_history_data.clear()

        # Write Second Violations CSV - modified to aggregate by second and model
        second_violations_data = []
        for timestamp, model_counts in self.second_violations.items():
            for model, count in model_counts.items():
                if count > 0:  # Only store if there were violations in this second
                    second_violations_data.append({
                        'timestamp': timestamp,
                        'model_name': model,
                        'violation_count': count
                    })
        
        if second_violations_data:
            # Print summary of what we're writing
            logger.info(f"Writing {len(second_violations_data)} violation records to {SECOND_VIOLATIONS_CSV}")
            
            # Sort by timestamp for readability
            second_violations_data.sort(key=lambda x: x['timestamp'])
            
            try:
                # Convert to DataFrame
                violations_df = pd.DataFrame(second_violations_data)
                
                # Make sure directory exists
                os.makedirs(os.path.dirname(SECOND_VIOLATIONS_CSV), exist_ok=True)
                
                # NEW: Pre-aggregate violations by timestamp and model
                # This ensures we only have one entry per second per model
                if not violations_df.empty:
                    violations_df = violations_df.groupby(['timestamp', 'model_name'])['violation_count'].sum().reset_index()
                    logger.info(f"After aggregation: {len(violations_df)} violation records")
                
                # If the file exists, read it and combine with new aggregated data
                if os.path.exists(SECOND_VIOLATIONS_CSV):
                    try:
                        # Read existing data
                        existing_df = pd.read_csv(SECOND_VIOLATIONS_CSV)
                        
                        # Combine with new data
                        combined_df = pd.concat([existing_df, violations_df], ignore_index=True)
                        
                        # Re-aggregate to ensure no duplicates across current and existing data
                        combined_df = combined_df.groupby(['timestamp', 'model_name'])['violation_count'].sum().reset_index()
                        
                        # Sort by timestamp for readability
                        combined_df = combined_df.sort_values('timestamp')
                        
                        # Limit the file size if needed (keep most recent 10000 entries)
                        if len(combined_df) > 10000:
                            combined_df = combined_df.tail(10000)
                        
                        # Write atomically
                        self._atomic_write_csv(combined_df, SECOND_VIOLATIONS_CSV)
                        logger.info(f"Updated violations CSV with {len(violations_df)} new aggregated records")
                    except Exception as e:
                        logger.error(f"Error appending to existing violations CSV: {e}")
                        # Fall back to just writing new data
                        self._atomic_write_csv(violations_df, SECOND_VIOLATIONS_CSV)
                else:
                    # No existing file, write directly
                    self._atomic_write_csv(violations_df, SECOND_VIOLATIONS_CSV)
                
                # Clear the current violations after successful write
                self.second_violations.clear()
            except Exception as e:
                logger.error(f"Failed to write second_violations.csv: {e}", exc_info=True)

        # Write Second Requests CSV - similar to second violations
        second_requests_data = []
        for timestamp, model_counts in self.second_requests.items():
            for model, count in model_counts.items():
                if count > 0:  # Only store if there were requests in this second
                    second_requests_data.append({
                        'timestamp': timestamp,
                        'model_name': model,
                        'request_count': count
                    })
        
        if second_requests_data:
            # Print summary of what we're writing
            logger.info(f"Writing {len(second_requests_data)} request records to {SECOND_REQUESTS_CSV}")
            
            # Sort by timestamp for readability
            second_requests_data.sort(key=lambda x: x['timestamp'])
            
            try:
                # Convert to DataFrame
                requests_df = pd.DataFrame(second_requests_data)
                
                # Make sure directory exists
                os.makedirs(os.path.dirname(SECOND_REQUESTS_CSV), exist_ok=True)
                
                # Pre-aggregate requests by timestamp and model
                # This ensures we only have one entry per second per model
                if not requests_df.empty:
                    requests_df = requests_df.groupby(['timestamp', 'model_name'])['request_count'].sum().reset_index()
                    logger.info(f"After aggregation: {len(requests_df)} request records")
                
                # If the file exists, read it and combine with new aggregated data
                if os.path.exists(SECOND_REQUESTS_CSV):
                    try:
                        # Read existing data
                        existing_df = pd.read_csv(SECOND_REQUESTS_CSV)
                        
                        # Combine with new data
                        combined_df = pd.concat([existing_df, requests_df], ignore_index=True)
                        
                        # Re-aggregate to ensure no duplicates across current and existing data
                        combined_df = combined_df.groupby(['timestamp', 'model_name'])['request_count'].sum().reset_index()
                        
                        # Sort by timestamp for readability
                        combined_df = combined_df.sort_values('timestamp')
                        
                        # Limit the file size if needed
                        if len(combined_df) > 10000:
                            combined_df = combined_df.tail(10000)
                        
                        # Write atomically
                        self._atomic_write_csv(combined_df, SECOND_REQUESTS_CSV)
                        logger.info(f"Updated requests CSV with {len(requests_df)} new aggregated records")
                    except Exception as e:
                        logger.error(f"Error appending to existing requests CSV: {e}")
                        # Fall back to just writing new data
                        self._atomic_write_csv(requests_df, SECOND_REQUESTS_CSV)
                else:
                    # No existing file, write directly
                    self._atomic_write_csv(requests_df, SECOND_REQUESTS_CSV)
                
                # Clear the current requests after successful write
                self.second_requests.clear()
            except Exception as e:
                logger.error(f"Failed to write second_requests.csv: {e}", exc_info=True)

    def _periodic_csv_writer_loop(self):
        while self.running:
            try:
                self._write_csv_files()
                self._update_history_data_capture() # Capture data for history CSVs
            except Exception as e:
                logger.error(f"Error in CSV writer loop: {e}")
            time.sleep(0.5) # Write CSVs every 0.5 seconds

    def _update_history_data_capture(self):
        """Captures current metrics for historical CSVs."""
        now_iso = datetime.datetime.now().isoformat()
        now_second = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Only up to seconds
        
        # Capture for performance_history.csv
        self.performance_history_data.append({
            'timestamp': now_iso,
            'total_requests_cumulative': self.metrics['total_requests'],
            'processed_requests_cumulative': self.metrics['processed_requests'],
            'avg_processing_time_us_overall': self.metrics['avg_processing_time_ns'] /1000 if self.metrics['processed_requests'] > 0 else 0,
            'throughput_rps': self.metrics['recent_throughput'],
            'slo_met_count_cumulative': self.metrics['slo_met_count'],
            'slo_violated_count_cumulative': self.metrics['slo_violated_count']
        })
        
        # Create a dict to track the latest rate for each model for this second
        latest_rates_by_model = {}
        for model_name, metrics in self.metrics['models'].items():
            latest_rates_by_model[model_name] = metrics['target_rate']
            
        # Add to request_rate_history_data with second-level precision timestamp
        for model_name, rate in latest_rates_by_model.items():
            self.request_rate_history_data.append({
                'timestamp': now_second,  # Use second precision timestamp
                'model_name': model_name,
                'rate': rate
            })

    def start_monitoring(self, process_existing=True):
        if self.running:
            return
        self.running = True

        if process_existing and os.path.exists(self.log_path):
            logger.info(f"Processing existing log file: {self.log_path}")
            try:
                with open(self.log_path, 'r') as f:
                    for line in f:
                        self._process_log_line(line)
                self.last_log_position = os.path.getsize(self.log_path)
            except Exception as e:
                logger.error(f"Error processing existing log: {e}")
            logger.info(f"Finished processing existing log. Total requests found: {self.metrics['total_requests']}")

        self.monitor_thread = threading.Thread(target=self._monitor_log_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.csv_writer_thread = threading.Thread(target=self._periodic_csv_writer_loop)
        self.csv_writer_thread.daemon = True
        self.csv_writer_thread.start()
        logger.info(f"Started monitoring log file: {self.log_path} and writing to CSVs in {self.csv_dir}")

    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        if self.csv_writer_thread:
            self.csv_writer_thread.join(timeout=2.0)
        self._write_csv_files() # Final write
        logger.info("Stopped monitoring and CSV writing.")

    def _monitor_log_loop(self):
        if not os.path.exists(self.log_path):
            logger.error(f"Log file not found at startup: {self.log_path}")
            # Don't exit early - we should keep trying to find the log file
            # self.running = False # Optionally stop if log file isn't there
            # return

        while self.running:
            try:
                if not os.path.exists(self.log_path):
                    logger.warning(f"Log file {self.log_path} not found. Waiting for it to appear.")
                    time.sleep(1.0) # Wait for log file to appear
                    continue

                current_size = os.path.getsize(self.log_path)
                if current_size > self.last_log_position:
                    with open(self.log_path, 'r') as f:
                        f.seek(self.last_log_position)
                        for line in f:
                            self._process_log_line(line)
                        self.last_log_position = f.tell()
                    self.last_update_time = time.time()
                elif current_size < self.last_log_position: # Log truncated
                    logger.warning(f"Log file {self.log_path} truncated. Resetting position.")
                    self.last_log_position = 0
                
                # Calculate throughput with safeguards against division by zero
                now = time.time()
                time_diff = now - self.last_throughput_calc_time
                if time_diff >= 1.0:
                    if time_diff > 0:  # Prevent division by zero
                        self.metrics['recent_throughput'] = self.processed_in_last_second_count / time_diff
                    else:
                        self.metrics['recent_throughput'] = 0  # Avoid division by zero
                    self.processed_in_last_second_count = 0
                    self.last_throughput_calc_time = now

            except FileNotFoundError:
                logger.warning(f"Log file {self.log_path} not found during monitoring. Will retry.")
                self.last_log_position = 0 # Reset position in case it reappears
            except ZeroDivisionError:
                # Explicitly handle division by zero
                logger.warning("Division by zero in throughput calculation, setting throughput to 0")
                self.metrics['recent_throughput'] = 0
                self.processed_in_last_second_count = 0
                self.last_throughput_calc_time = time.time()
            except Exception as e:
                # Don't crash the monitoring loop - log and continue
                logger.error(f"Error in log monitoring loop: {e}", exc_info=True)
        
            time.sleep(0.05) # Poll frequency

    def _format_timestamp_readable(self, timestamp_ns):
        """
        Convert nanosecond timestamp to readable format using current system time.
        This ensures we use real wall clock time instead of monotonic time.
        """
        # Instead of converting directly from the monotonic clock time,
        # use the current system time as the base reference
        dt = datetime.datetime.now()
        return dt.strftime('%Y-%m-%d %H:%M:%S')  # Format to second precision for aggregation

    def _process_log_line(self, line):
        try:
            # Get current timestamp at second precision for consistency
            now = datetime.datetime.now()
            current_second = now.strftime('%Y-%m-%d %H:%M:%S')  # Format to second precision
            now_iso = now.isoformat()
            
            # Check for REQUEST RECEIVED logs
            match_request_received = self.re_request_received.search(line)
            if match_request_received:
                model_name, request_count_str, time_now_str = match_request_received.groups()
                time_now_ns = int(time_now_str)
                request_count = int(request_count_str)
                
                # Get the timestamp at second precision for bucketing
                request_timestamp = self._format_timestamp_readable(time_now_ns)
                request_second = request_timestamp.split('.')[0]  # Just get the second part
                
                # Record the request in second_requests
                self.second_requests[request_second][model_name] += request_count
                
                # IMPORTANT FIX: Update total_requests with the request_count, not just incrementing by 1
                # This ensures we count the actual number of requests correctly
                self.metrics['total_requests'] += request_count
                self.metrics['models'][model_name]['request_count'] += request_count
                
                # Debug log
                logger.debug(f"Request Received: {model_name}, count: {request_count}, time: {request_second}")
                
                return  # Return early after processing REQUEST RECEIVED

            # Check for SLO violation warnings from RequestProcessor
            match_slo_violation = self.re_slo_violation.search(line)
            if match_slo_violation:
                model_name, request_count_str, time_now_str = match_slo_violation.groups()
                time_now_ns = int(time_now_str)
                request_count = int(request_count_str)
                
                # Get the timestamp in readable format for bucketing by second
                violation_timestamp = self._format_timestamp_readable(time_now_ns)
                violation_second = violation_timestamp.split('.')[0]  # Just get the second part
                
                # Record the violation in second_violations
                self.second_violations[violation_second][model_name] += request_count
                
                # Also update the cumulative metrics
                self.metrics['slo_violated_count'] += request_count
                self.metrics['models'][model_name]['slo_violated_count'] += request_count
                
                # Debug log
                logger.debug(f"SLO Violation: {model_name}, count: {request_count}, time: {violation_timestamp}")
                
                return  # Return early after processing SLO violation
            
            # Rest of the existing code for processing other log patterns
            match_simulate = self.re_simulate_rate.search(line)
            if match_simulate:
                rate, model_name = match_simulate.groups()
                self.models_seen.add(model_name)
                self.metrics['models'][model_name]['target_rate'] = int(rate)
                return

            match_formed = self.re_batch_formed.search(line)
            if match_formed:
                batch_id, batch_size_str = match_formed.groups()
                model_name = batch_id.split('_')[0]
                self.models_seen.add(model_name)
                self.batches[batch_id] = {
                    'model_name': model_name,
                    'size': int(batch_size_str),
                    'requests_in_batch': [] # Store request_ids
                }
                return

            match_request = self.re_request.search(line)
            if match_request:
                batch_id, request_count_str, arrival_time_ns_str = match_request.groups()
                model_name = batch_id.split('_')[0]
                self.models_seen.add(model_name)
                
                request_id = f"{batch_id}_{arrival_time_ns_str}" # Simpler unique ID
                request_count = int(request_count_str)
                self.requests[request_id] = {
                    'model_name': model_name,
                    'batch_id': batch_id,
                    'arrival_time_ns': int(arrival_time_ns_str),
                    'request_count': request_count
                }
                if batch_id in self.batches:
                     self.batches[batch_id]['requests_in_batch'].append(request_id)

                # IMPORTANT: We now track requests from REQUEST RECEIVED logs instead of here
                # to avoid double counting
                self.pending_requests[batch_id].append(request_id)
                return

            match_processed = self.re_batch_processed.search(line)
            if match_processed:
                batch_id, completion_time_ns_str = match_processed.groups()
                completion_time_ns = int(completion_time_ns_str)
                
                if batch_id not in self.batches:
                    logger.warning(f"BatchProcessed for {batch_id} but no BATCH FORMED info. Assuming it's for pending requests.")
                
                model_name = batch_id.split('_')[0] # Infer model name
                self.models_seen.add(model_name)

                num_processed_in_this_batch = 0
                total_processing_time_us_this_batch = 0

                # Process requests associated with this batch_id
                if batch_id in self.pending_requests:
                    for req_id in list(self.pending_requests[batch_id]):
                        if req_id in self.requests:
                            req_data = self.requests[req_id]
                            model_name = req_data['model_name']
                            processing_time_ns = completion_time_ns - req_data['arrival_time_ns']
                            processing_time_us = processing_time_ns / 1000.0
                            
                            # Get the second for this request's completion for violation tracking
                            completion_time_readable = self._format_timestamp_readable(completion_time_ns)
                            completion_second = completion_time_readable.split('.')[0]  # Just get the second part
                            
                            # SLO Check using the correct unit (microseconds)
                            model_slo_us = MODEL_SLOS_US.get(model_name, DEFAULT_SLO_US)
                            
                            logger.debug(f"SLO check: {model_name} - time:{processing_time_us:.2f}μs vs threshold:{model_slo_us}μs")
                            
                            # IMPORTANT: Get the request count for proper accounting
                            request_count = req_data['request_count']
                            
                            if processing_time_us > model_slo_us:
                                # Record violation with correct timestamp for the exact second
                                # Use the actual request count
                                self.second_violations[completion_second][model_name] += request_count
                                
                                # Rest of violation handling - use request count
                                slo_status = "Violated"
                                self.metrics['slo_violated_count'] += request_count
                                self.metrics['models'][model_name]['slo_violated_count'] += request_count
                                
                                logger.debug(f"SLO Violation at {completion_second} for {model_name}, count: {request_count}")
                            else:
                                slo_status = "Met"
                                self.metrics['slo_met_count'] += request_count
                                self.metrics['models'][model_name]['slo_met_count'] += request_count
                            
                            self.processed_requests_details.append({
                                'request_id': req_id,
                                'model_name': req_data['model_name'],
                                'gpu_id': batch_id.split('_')[1] if '_' in batch_id else 'N/A',
                                'batch_id': batch_id,
                                'arrival_time_readable': self._format_timestamp_readable(req_data['arrival_time_ns']),
                                'completion_time_readable': self._format_timestamp_readable(completion_time_ns),
                                'processing_time_us': processing_time_us, # Convert ns to µs
                                'status': 'Processed',
                                'slo_status': slo_status,
                                'request_count': request_count  # Add request_count to details
                            })
                            
                            # IMPORTANT: Update processed_requests with the actual request count
                            self.metrics['processed_requests'] += request_count
                            self.metrics['models'][model_name]['processed_count'] += request_count
                            num_processed_in_this_batch += request_count
                            total_processing_time_us_this_batch += (processing_time_ns) / 1000
                            
                            # Update overall avg processing time (cumulative moving average, in nanoseconds)
                            self.metrics['avg_processing_time_ns'] += (processing_time_ns - self.metrics['avg_processing_time_ns']) / self.metrics['processed_requests']
                            
                            # Update model avg processing time (in nanoseconds)
                            model_proc_count = self.metrics['models'][model_name]['processed_count']
                            self.metrics['models'][model_name]['avg_processing_time_ns'] += \
                                (processing_time_ns - self.metrics['models'][model_name]['avg_processing_time_ns']) / model_proc_count
                            
                            del self.requests[req_id] # Clean up processed request
                        
                if batch_id in self.pending_requests:
                    del self.pending_requests[batch_id] # Clear pending for this batch
            
                if batch_id in self.batches: # Clean up batch info
                    del self.batches[batch_id]

                self.processed_in_last_second_count += num_processed_in_this_batch

                return
        except Exception as e:
            # Don't crash on processing errors - log and continue
            logger.error(f"Error processing log line: {e}, {line}", exc_info=True)

if __name__ == "__main__":
    try:
        processor = LogToCsvProcessor(LOG_FILE_PATH, CSV_OUTPUT_DIR)
        processor.start_monitoring(process_existing=True)
        logger.info(f"Processor started. Monitoring log file: {LOG_FILE_PATH}")
        
        while True:
            time.sleep(10) # Keep main thread alive
            logger.info(f"Processor alive. Total Requests: {processor.metrics['total_requests']}, Processed: {processor.metrics['processed_requests']}")
    except KeyboardInterrupt:
        logger.info("Shutting down processor due to keyboard interrupt...")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Attempting to stop processor...")
        try:
            processor.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping processor: {e}")
        logger.info("Processor shut down.")
