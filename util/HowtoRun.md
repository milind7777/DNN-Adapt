**How to Run:**


1.  **Create the CSV data directory:**
    ```bash
    mkdir -p */DNN-Adapt/util/csv_data_cache
    ```
2.  **Run the Log Processor:**
    Open a terminal, navigate to `*/DNN-Adapt/util/` and run:
    ```bash
    python log_csv_processor.py
    ```
    This script will start monitoring your `dnn_adapt.log` and creating/updating CSV files in the `csv_data_cache` directory.
3.  **Run the Dashboard Application:**
    Open another terminal, navigate to the same directory (`*/DNN-Adapt/util/`) and run:
    ```bash
    python dashboard_app.py
    ```
    This will start the Dash web server (likely on `http://127.0.0.1:8051`). Open this URL in your browser.



