import csv
import subprocess
import uuid
import json
import os
import argparse
import threading
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Run nerf_test.py experiments from a CSV queue.")
    parser.add_argument("--queue_csv", type=str, default="tests_queue.csv",
                        help="CSV file containing the test parameters (each row is one test).")
    parser.add_argument("--output_csv", type=str, default="aggregated_results.csv",
                        help="CSV file where aggregated query and result outputs are written after each job completion.")
    parser.add_argument("--gpus", type=str, default="1,2,3",
                        help="Comma-separated list of GPU IDs to use for concurrent jobs.")
    return parser.parse_args()

def generate_folder_name(test_params, base_dir="output"):
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
        # Split the key by '_' and take the first letter of each non-empty piece
        initials = ''.join(piece[0] for piece in key.split('_') if piece)
        parts.append(f"{initials}{value}")
    folder_name = "_".join(parts)
    # Construct the full path inside the base directory.
    full_path = os.path.join(base_dir, folder_name)
    # If the folder already exists, append a short random suffix to ensure uniqueness.
    # if os.path.exists(full_path):
    #     suffix = uuid.uuid4().hex[:6]
    #     folder_name = folder_name + "_" + suffix
    #     full_path = os.path.join(base_dir, folder_name)
    return full_path

def run_test(test_params, gpu_id):
    """
    Run a single test on a given GPU.
    - test_params: dictionary mapping argument names to values from the CSV.
    - gpu_id: the GPU to assign for this run.
    
    Returns a merged dictionary of the original test parameters and the JSON output
    (or error information) from the run.
    """
    # Generate a unique folder name based on the test parameters.
    output_folder = generate_folder_name(test_params)
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
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Test on GPU {gpu_id} failed with error: {e}")
    
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
    merged_result.update(test_params)  # Query parameters from the CSV.
    merged_result.update(data)         # JSON output and additional metadata.
    return merged_result

def write_csv(aggregated_results, output_csv, csv_lock):
    """
    Writes the current aggregated_results list to a CSV file.
    This function is protected by a lock to avoid concurrent writes.
    """
    with csv_lock:
        # Determine all keys present in any of the results.
        all_keys = set()
        for res in aggregated_results:
            all_keys.update(res.keys())
        all_keys = list(all_keys)
        with open(output_csv, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for res in aggregated_results:
                writer.writerow(res)
        print(f"Updated aggregated CSV: {output_csv}")

def main():
    args = parse_args()
    # Parse the list of GPU IDs.
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    
    # Read the tests from the CSV file.
    tests = []
    with open(args.queue_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tests.append(row)
    
    aggregated_results = []
    csv_lock = threading.Lock()  # Protect aggregated_results and CSV writing.
    
    # Use a thread pool to run tests concurrently. Number of workers equals number of GPUs.
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        gpu_cycle = cycle(gpu_ids)
        future_to_test = {}
        for test in tests:
            gpu_id = next(gpu_cycle)
            future = executor.submit(run_test, test, gpu_id)
            future_to_test[future] = test
        
        # As each test finishes, collect the results and update the CSV file.
        for future in as_completed(future_to_test):
            try:
                result = future.result()
            except Exception as e:
                result = {"error": "Unexpected failure", "exception": str(e)}
            with csv_lock:
                aggregated_results.append(result)
            write_csv(aggregated_results, args.output_csv, csv_lock)

if __name__ == '__main__':
    main()
