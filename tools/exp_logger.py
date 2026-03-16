#!/usr/bin/env python3
"""
Experiment Logging Utility
==========================

A utility module for initializing and managing experiment logs with automatic
folder structure creation, timestamping, and metadata integration.

Usage (as module):
    from tools.exp_logger import init_exp_log

    log_dir, log_path, outcome_path = init_exp_log(
        method_info="ConvTok_MHSA",
        data_info="NGAFID_Subset",
        task_type="Diagnosis"
    )

Usage (command line):
    python tools/exp_logger.py --method ConvTok_MHSA --data NGAFID_Subset --task Diagnosis

Parameters:
    method_info (str): Method identifier. Corresponds to description file in
                       MethodInfoDB/{method_info}_Info.txt
    data_info (str):   Dataset identifier. Corresponds to description file in
                       DataInfoDB/{data_info}_Info.txt
    task_type (str):   Task category folder name (default: "Detection").
                       Determines parent folder structure: EXP_Logs/{task_type}/...

Outputs:
    Creates directory structure:
    EXP_Logs/
    └── {task_type}/
        └── {data_info}/
            └── {method_info}/
                └── Log_EXP_YYYYMMDD_HHMMSS/
                    ├── log.txt      # Full experiment log with metadata
                    └── outcome.txt  # Results summary file

Returns:
    tuple: (log_directory_path, log_file_path, outcome_file_path)
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

# Get project root (parent directory of tools/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TOOLS_DIR = Path(__file__).parent.resolve()


def create_folder_if_not_exists(folder_path):
    """
    Create directory if it does not exist.

    Args:
        folder_path (str or Path): Directory path to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def read_file_to_string(file_path):
    """
    Read entire file content into a string.

    Args:
        file_path (str or Path): Path to text file.

    Returns:
        str: File content, or empty string if file not found or error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""


def append_text_to_file(text, file_path, end='\n'):
    """
    Append text to the end of specified file.

    Args:
        text (str): Content to append.
        file_path (str or Path): Target file path.
        end (str): Line ending suffix (default: newline).
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text + end)
    except Exception as e:
        print(f"Error writing to file: {e}")


def print_aptxt(text, file_path, end='\n'):
    """
    Print to console and simultaneously append to file.

    Args:
        text (str): Content to print and log.
        file_path (str or Path): Log file path.
        end (str): Line ending suffix.
    """
    print(text, end=end)
    append_text_to_file(text, file_path, end)


def get_current_datetime_string():
    """
    Generate formatted timestamp strings for display and filename use.

    Returns:
        tuple: (datetime_display_str, datetime_filename_str)
               - Display: "YYYY-MM-DD HH:MM:SS"
               - Filename: "YYYYMMDD_HHMMSS"
    """
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")
    datetime_fileid = now.strftime("%Y%m%d_%H%M%S")
    return datetime_string, datetime_fileid


def get_divider(length=32):
    """
    Generate a separator line of specified length.

    Args:
        length (int): Number of '=' characters (default: 32).

    Returns:
        str: Separator string with newline suffix.
    """
    return '=' * length + '\n'


def init_exp_log(method_info, data_info, task_type="Detection"):
    """
    Initialize experiment log directory and metadata files.

    Creates timestamped log folder, copies method and data description files
    into the log header, and prepares outcome file placeholder.

    Args:
        method_info (str): Method identifier (e.g., "ConvTok_MHSA").
                          Looks for MethodInfoDB/{method_info}_Info.txt
        data_info (str):   Data identifier (e.g., "NGAFID_Subset").
                          Looks for DataInfoDB/{data_info}_Info.txt
        task_type (str):   Task category folder name (default: "Detection").
                          Used for organizing logs under EXP_Logs/{task_type}/...

    Returns:
        tuple: (log_directory_path, log_file_path, outcome_file_path)
    """
    current_datetime, currentime_fileid = get_current_datetime_string()

    # Construct log directory path: EXP_Logs/{Task}/{Data}/{Method}/Log_EXP_{timestamp}/
    log_dir = PROJECT_ROOT / "EXP_Logs" / task_type / data_info / method_info / f"Log_EXP_{currentime_fileid}"
    create_folder_if_not_exists(log_dir)

    log_path = log_dir / "log.txt"
    outcome_path = log_dir / "outcome.txt"

    # Path to metadata description files
    method_desc = PROJECT_ROOT / "MethodInfoDB" / f"{method_info}_Info.txt"
    data_desc = PROJECT_ROOT / "DataInfoDB" / f"{data_info}_Info.txt"

    # Initialize log with timestamp
    print_aptxt(f"Experiment Timestamp: {current_datetime}", log_path)

    # Append method description if exists
    if method_desc.exists():
        append_text_to_file(read_file_to_string(method_desc), log_path)

    # Append data description if exists
    if data_desc.exists():
        append_text_to_file(read_file_to_string(data_desc), log_path)

    return str(log_dir), str(log_path), str(outcome_path)


def main():
    """Command line interface for standalone log initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize experiment log structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--method', required=True,
                        help='Method identifier (e.g., ConvTok_MHSA)')
    parser.add_argument('--data', required=True,
                        help='Dataset identifier (e.g., NGAFID_Subset)')
    parser.add_argument('--task', default='Detection',
                        help='Task type folder name (default: Detection)')

    args = parser.parse_args()

    log_dir, log_path, outcome_path = init_exp_log(
        method_info=args.method,
        data_info=args.data,
        task_type=args.task
    )

    print(f"\nLog directory created: {log_dir}")
    print(f"Log file: {log_path}")
    print(f"Outcome file: {outcome_path}")


if __name__ == "__main__":
    main()