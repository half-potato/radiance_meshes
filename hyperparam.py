import csv
import subprocess
import uuid
import json
import os
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run nerf_test.py experiments from a CSV queue.")
    parser.add_argument("--queue_csv", type=str, default="tests_queue.csv",
                        help="CSV file containing the test parameters (each row is one test).")
    parser.add_argument("--output_csv", type=str, default="aggregated_results.csv",
                        help="CSV file where aggregated query and result outputs are written after each job completion.")
    parser.add_argument("--gpus", type=str, default="1,2,3",
                        help="Comma-separated list of GPU IDs to use for concurrent jobs.")
    parser.add_argument("--suffix", type=str, default="")
    return parser.parse_args()

def generate_folder_name(test_params, args, base_dir="output"):
    """
    Generate a folder name based on test parameters.
    For each parameter (except output_path), the key is split by '_' and its first letters are
    taken and concatenated with the value. All such pieces are joined with underscores.
    """
    parts = []
    # Iterate in the order provided by test_params (csv.DictReader preserves header order)
    for key, value in test_params.items():
        if key == "output_path":
            continue
        if key == "dataset_path":
            initials = key
            parts.append(f"{Path(value).name}")
            continue
        # Split the key by '_' and take the first letter of each non-empty piece
        initials = ''.join(piece[0] for piece in key.split('_') if piece)
        parts.append(f"{initials}{value}")
    folder_name = "_".join(parts) + args.suffix
    # Construct the full path inside the base directory.
    full_path = os.path.join(base_dir, folder_name)
    return full_path

def run_test(test_params, gpu_id, args):
    """
    Run a single test on a given GPU.
    - test_params: dictionary mapping argument names to values from the CSV.
    - gpu_id: the GPU to assign for this run.
    
    Returns a merged dictionary of the original test parameters and the JSON output
    (or error information) from the run.
    """
    # Generate a unique folder name based on the test parameters.
    output_folder = generate_folder_name(test_params, args)
    os.makedirs(output_folder, exist_ok=True)
    
    # Base command with GPU assignment and script name.
    base_command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py --eval "
    
    # Build command-line arguments. Override any CSV-specified output_path with our unique folder.
    cmd_args = []
    for arg, value in test_params.items():
        if arg == "output_path":
            continue
        cmd_args.append(f"--{arg} {value}")
    cmd_args.append(f"--output_path {output_folder}")
    
    # Assemble the full command string.
    command = f"{base_command} " + " ".join(cmd_args)
    print(f"Running on GPU {gpu_id}: {command}")
    
    json_file = os.path.join(output_folder, "results.json")
    # Attempt to run the command. If it fails, still try to read the JSON output.
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Test on GPU {gpu_id} failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    
    # Try to read the JSON output regardless of subprocess exit status.
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception as e_read:
            data = {"error": "JSON file exists but could not be read", "read_exception": str(e_read)}
    else:
        data = {"error": "Process failed and no JSON output file found"}
    
    # Add run metadata.
    data["output_folder"] = output_folder
    data["gpu_id"] = gpu_id
    
    # Merge the original test parameters with the result so that the final row has both.
    merged_result = {}
    merged_result.update(data)          # JSON output and additional metadata.
    merged_result.update(test_params)   # Query parameters from the CSV.
    return merged_result

def worker_task(test_params, gpu_queue, args):
    """
    Worker function for the thread pool. Acquires a GPU from the queue,
    runs the test, and ensures the GPU is released back to the queue.
    """
    gpu_id = gpu_queue.get() # This will block until a GPU is available.
    print(f"Acquired GPU {gpu_id} for test...")
    try:
        # Pass the acquired GPU and other params to the original run function.
        return run_test(test_params, gpu_id, args)
    finally:
        # This block ensures the GPU is always returned to the queue,
        # even if run_test crashes.
        print(f"Releasing GPU {gpu_id}...")
        gpu_queue.put(gpu_id)


def write_csv(aggregated_results, output_csv, csv_lock):
    """
    Writes the current aggregated_results list to a CSV file.
    This function is protected by a lock to avoid concurrent writes.
    """
    with csv_lock:
        if not aggregated_results:
            return # Nothing to write
            
        # Determine all keys present in any of the results to form a complete header.
        all_keys = set()
        for res in aggregated_results:
            all_keys.update(res.keys())
        
        # Define a preferred order, with dynamic keys added at the end.
        header = sorted(list(all_keys))

        with open(output_csv, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for res in aggregated_results:
                writer.writerow(res)
        print(f"Updated aggregated CSV: {output_csv}")

def main():
    args = parse_args()
    # Parse the list of GPU IDs.
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        print("Error: No GPU IDs provided. Exiting.")
        return

    # Use a thread-safe queue to manage the pool of available GPUs.
    gpu_queue = Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    
    # Read the tests from the CSV file.
    tests = []
    try:
        with open(args.queue_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tests.append(row)
    except FileNotFoundError:
        print(f"Error: Queue CSV file not found at {args.queue_csv}")
        return
    
    aggregated_results = []
    csv_lock = threading.Lock()  # Protects aggregated_results and CSV writing.
    
    # Use a thread pool to run tests concurrently. Number of workers equals number of GPUs.
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        future_to_test = {executor.submit(worker_task, test, gpu_queue, args): test for test in tests}
        
        # As each test finishes, collect the results and update the CSV file.
        for future in as_completed(future_to_test):
            original_test = future_to_test[future]
            try:
                result = future.result()
            except Exception as e:
                # This catches exceptions from the worker_task itself, e.g., if it can't run.
                print(f"An unexpected error occurred for test {original_test}: {e}")
                result = {"error": "Future failed unexpectedly", "exception": str(e)}
                result.update(original_test) # Add original params for context

            with csv_lock:
                aggregated_results.append(result)
            
            # This is a bit inefficient as it rewrites the file every time,
            # but it is robust and simple.
            write_csv(aggregated_results, args.output_csv, csv_lock)

if __name__ == '__main__':
    main()

