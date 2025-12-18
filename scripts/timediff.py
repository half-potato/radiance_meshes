#!/usr/bin/env python3

import os
import argparse
import datetime

def calculate_time_difference(file1_path, file2_path):
    """
    Calculates the time difference in hours between the last modification
    times of two files.

    Args:
        file1_path (str): The path to the first file.
        file2_path (str): The path to the second file.
    """
    # Get the last modification time of each file in seconds since the epoch
    mod_time1 = os.path.getmtime(file1_path)
    mod_time2 = os.path.getmtime(file2_path)

    # Convert the timestamps to datetime objects for easier calculation
    datetime1 = datetime.datetime.fromtimestamp(mod_time1)
    datetime2 = datetime.datetime.fromtimestamp(mod_time2)

    # Calculate the absolute difference between the two times
    time_difference = abs(datetime2 - datetime1)

    # Convert the time difference to hours
    hours_difference = time_difference.total_seconds() / 3600

    print(hours_difference)

if __name__ == "__main__":
    # Set up the argument parser to accept command-line arguments
    parser = argparse.ArgumentParser(
        description="Measure the time difference in hours between the last modification times of two files."
    )
    parser.add_argument("file1", help="The path to the first file.")
    parser.add_argument("file2", help="The path to the second file.")

    args = parser.parse_args()

    # Call the function with the provided file paths
    calculate_time_difference(args.file1, args.file2)

