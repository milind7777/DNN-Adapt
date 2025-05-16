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
LOG_FILE_PATH = '/home/cching1/DNNAdapt/DNN-Adapt/logs/experiment1/dnn_adapt.log'
CSV_OUTPUT_DIR = '/home/cching1/DNNAdapt/DNN-Adapt/util/csv_data_cache' # Ensure this directory exists
MAX_HISTORY_CSV_ROWS = 5000 # For history CSVs to prevent them from growing indefinitely
RECENT_REQUESTS_LIMIT = 200 # For recent_requests_details.csv

# SLO Configuration (in microseconds)
try:
    with open('/home/cching1/DNNAdapt/DNN-Adapt/util/slo_config.json', 'r') as f:
        slo_config = json.load(f)
    MODEL_SLOS_US = slo_config.get('model_slos_us', {})
    DEFAULT_SLO_US = slo_config.get('default_slo_us', 1000)
    logger.info(f"Loaded SLO configuration: {MODEL_SLOS_US}, default: {DEFAULT_SLO_US}")
except Exception as e:
    logger.warning(f"Failed to load SLO configuration: {e}")
    # Fallback to hardcoded values
    MODEL_SLOS_US = {
        "resnet18": 500,
        "vit16": 1000,
    }
    DEFAULT_SLO_US = 1000
    logger.info(f"Using fallback SLO configuration: {MODEL_SLOS_US}, default: {DEFAULT_SLO_US}")

# CSV Filenames
OVERALL_METRICS_CSV = os.path.join(CSV_OUTPUT_DIR, "overall_metrics_live.csv")
MODEL_METRICS_CSV = os.path.join(CSV_OUTPUT_DIR, "model_metrics_live.csv")
REQUEST_RATE_HISTORY_CSV = os.path.join(CSV_OUTPUT_DIR, "request_rate_history.csv")
PERFORMANCE_HISTORY_CSV = os.path.join(CSV_OUTPUT_DIR, "performance_history.csv")
RECENT_REQUESTS_DETAILS_CSV = os.path.join(CSV_OUTPUT_DIR, "recent_requests_details.csv")

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

        self.running = False
        self.monitor_thread = None
        self.csv_writer_thread = None
        
        self.processed_in_last_second_count = 0
        self.last_throughput_calc_time = time.time()


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


    def _write_csv_files(self):
        """Periodically writes current data to CSV files."""
        # Overall Metrics
        overall_df = pd.DataFrame([{
            'timestamp': datetime.datetime.now().isoformat(),
            'total_requests': self.metrics['total_requests'],
            'processed_requests': self.metrics['processed_requests'],
            'avg_processing_time_us': self.metrics['avg_processing_time_ns'] if self.metrics['processed_requests'] > 0 else 0,
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
                'avg_processing_time_us': m_metrics['avg_processing_time_ns'] if m_metrics['processed_count'] > 0 else 0,
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
            self._append_to_csv(
                list(self.request_rate_history_data), 
                REQUEST_RATE_HISTORY_CSV,
                ['timestamp', 'model_name', 'rate'],
                self._is_request_rate_header_written,
                self._set_request_rate_header_written
            )
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
        
        # Capture for performance_history.csv
        self.performance_history_data.append({
            'timestamp': now_iso,
            'total_requests_cumulative': self.metrics['total_requests'],
            'processed_requests_cumulative': self.metrics['processed_requests'],
            'avg_processing_time_us_overall': self.metrics['avg_processing_time_ns'] if self.metrics['processed_requests'] > 0 else 0,
            'throughput_rps': self.metrics['recent_throughput'],
            'slo_met_count_cumulative': self.metrics['slo_met_count'],
            'slo_violated_count_cumulative': self.metrics['slo_violated_count']
        })
        
        # request_rate_history_data is populated directly in _process_log_line for 'Simulating rate'

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
            # self.running = False # Optionally stop if log file isn't there
            # return

        while self.running:
            try:
                if not os.path.exists(self.log_path):
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
                
                # Calculate throughput
                now = time.time()
                if now - self.last_throughput_calc_time >= 1.0:
                    self.metrics['recent_throughput'] = self.processed_in_last_second_count / (now - self.last_throughput_calc_time)
                    self.processed_in_last_second_count = 0
                    self.last_throughput_calc_time = now

            except FileNotFoundError:
                 logger.warning(f"Log file {self.log_path} not found during monitoring. Will retry.")
                 self.last_log_position = 0 # Reset position in case it reappears
            except Exception as e:
                logger.error(f"Error in log monitoring loop: {e}")
            time.sleep(0.05) # Poll frequency

    def _format_timestamp_readable(self, timestamp_ns):
        # Convert nanoseconds to seconds for datetime.fromtimestamp
        dt = datetime.datetime.fromtimestamp(timestamp_ns / 1_000_000_000.0)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Format to millisecond precision for readability


    def _process_log_line(self, line):
        now_iso = datetime.datetime.now().isoformat()
        match_simulate = self.re_simulate_rate.search(line)
        if match_simulate:
            rate, model_name = match_simulate.groups()
            self.models_seen.add(model_name)
            self.metrics['models'][model_name]['target_rate'] = int(rate)
            self.request_rate_history_data.append({ # For request_rate_history.csv
                'timestamp': now_iso,
                'model_name': model_name,
                'rate': int(rate)
            })
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
            self.requests[request_id] = {
                'model_name': model_name,
                'batch_id': batch_id,
                'arrival_time_ns': int(arrival_time_ns_str),
                'request_count': int(request_count_str) # Assuming this is part of the request, not batch size
            }
            if batch_id in self.batches:
                 self.batches[batch_id]['requests_in_batch'].append(request_id)

            self.metrics['total_requests'] += 1 # Or use request_count if it means individual sub-requests
            self.metrics['models'][model_name]['request_count'] += 1
            self.pending_requests[batch_id].append(request_id)
            return

        match_processed = self.re_batch_processed.search(line)
        if match_processed:
            batch_id, completion_time_ns_str = match_processed.groups()
            completion_time_ns = int(completion_time_ns_str)
            
            if batch_id not in self.batches:
                logger.warning(f"BatchProcessed for {batch_id} but no BATCH FORMED info. Assuming it's for pending requests.")
                # Try to find requests by batch_id prefix if it's a common pattern
                # This part needs careful handling if BATCH FORMED is missed.
                # For now, we'll process requests that were put in pending_requests[batch_id]
            
            model_name = batch_id.split('_')[0] # Infer model name
            self.models_seen.add(model_name)

            num_processed_in_this_batch = 0
            total_processing_time_us_this_batch = 0

            # Process requests associated with this batch_id
            if batch_id in self.pending_requests:
                for req_id in list(self.pending_requests[batch_id]): # Iterate copy for safe removal
                    if req_id in self.requests:
                        req_data = self.requests[req_id]
                        processing_time_ns = completion_time_ns - req_data['arrival_time_ns']
                        processing_time_us = processing_time_ns

                        # SLO Check - ensure we're using the current SLO thresholds from config
                        model_slo_us = MODEL_SLOS_US.get(req_data['model_name'], DEFAULT_SLO_US)
                        processing_time_us = processing_time_ns  # Already in microseconds in your updated code
                        
                        # Add detailed SLO logging for debugging
                        if processing_time_us > model_slo_us:
                            logger.warning(f"SLO VIOLATION: Model {req_data['model_name']} processing time {processing_time_us}μs > threshold {model_slo_us}μs")
                            slo_status = "Violated"
                            self.metrics['slo_violated_count'] += 1
                            self.metrics['models'][req_data['model_name']]['slo_violated_count'] += 1
                        else:
                            logger.debug(f"SLO MET: Model {req_data['model_name']} processing time {processing_time_us}μs <= threshold {model_slo_us}μs") 
                            slo_status = "Met"
                            self.metrics['slo_met_count'] += 1
                            self.metrics['models'][req_data['model_name']]['slo_met_count'] += 1
                        
                        self.processed_requests_details.append({
                            'request_id': req_id,
                            'model_name': req_data['model_name'],
                            'gpu_id': batch_id.split('_')[1] if '_' in batch_id else 'N/A',
                            'batch_id': batch_id,
                            'arrival_time_readable': self._format_timestamp_readable(req_data['arrival_time_ns']),
                            'completion_time_readable': self._format_timestamp_readable(completion_time_ns),
                            'processing_time_us': processing_time_us, # Convert ns to µs
                            'status': 'Processed',
                            'slo_status': slo_status
                        })
                        
                        self.metrics['processed_requests'] += 1
                        self.metrics['models'][model_name]['processed_count'] += 1
                        num_processed_in_this_batch +=1
                        total_processing_time_us_this_batch += (processing_time_ns) 
                        
                        # Update overall avg processing time (cumulative moving average, in nanoseconds)
                        self.metrics['avg_processing_time_ns'] += (processing_time_ns - self.metrics['avg_processing_time_ns']) / self.metrics['processed_requests']
                        
                        # Update model avg processing time (in nanoseconds)
                        model_proc_count = self.metrics['models'][model_name]['processed_count']
                        self.metrics['models'][model_name]['avg_processing_time_ns'] += \
                            (processing_time_ns - self.metrics['models'][model_name]['avg_processing_time_ns']) / model_proc_count
                        
                        del self.requests[req_id] # Clean up processed request
                    
                del self.pending_requests[batch_id] # Clear pending for this batch
            
            if batch_id in self.batches: # Clean up batch info
                del self.batches[batch_id]

            self.processed_in_last_second_count += num_processed_in_this_batch
            return

if __name__ == "__main__":
    processor = LogToCsvProcessor(LOG_FILE_PATH, CSV_OUTPUT_DIR)
    processor.start_monitoring(process_existing=True)
    try:
        while True:
            time.sleep(10) # Keep main thread alive
            logger.info(f"Processor alive. Total Requests: {processor.metrics['total_requests']}, Processed: {processor.metrics['processed_requests']}")
    except KeyboardInterrupt:
        logger.info("Shutting down processor...")
    finally:
        processor.stop_monitoring()
        logger.info("Processor shut down.")