#!/usr/bin/env python3
"""
SLURM Job Status Checker

This script parses SLURM job IDs from a log file and checks their status using sacct.
It provides flexible filtering and formatting options.

Usage:
    python slurm_checker.py logfile.txt [options]
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Optional


def parse_job_ids(filepath: str) -> list[str]:
    """
    Parse SLURM job IDs from a log file using regex.

    Args:
        filepath: Path to the log file containing SLURM job submissions

    Returns:
        list of job IDs found in the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    job_ids = []

    # Regex pattern to match "Submitted batch job [jobid]"
    pattern = r"Submitted batch job (\d+)"

    try:
        with open(filepath) as file:
            content = file.read()
            matches = re.findall(pattern, content)
            job_ids.extend(matches)
    except FileNotFoundError:
        if filepath == "sbatch.log":
            print(f"Error: Default file '{filepath}' not found in current directory.")
            print("Please specify a log file or ensure sbatch.log exists.")
        else:
            print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except OSError as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

    # Remove duplicates while preserving order
    job_ids = list(dict.fromkeys(job_ids))

    print(f"Found {len(job_ids)} unique job IDs in {filepath}")
    return job_ids


def calculate_duration(start_time: str, end_time: str) -> str:
    """
    Calculate duration between start and end times.

    Args:
        start_time: Start time string from sacct
        end_time: End time string from sacct

    Returns:
        Duration string in format HH:MM:SS or "Unknown" if calculation fails
    """
    try:
        if start_time == "Unknown" or end_time == "Unknown" or end_time == "":
            return "Unknown"

        # Parse the datetime strings (sacct format: YYYY-MM-DDTHH:MM:SS)
        start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
        end_dt = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")

        duration = end_dt - start_dt
        total_seconds = int(duration.total_seconds())

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "Unknown"


def get_available_fields() -> dict[str, str]:
    """
    Get available sacct format fields with descriptions.

    Returns:
        dictionary mapping field names to descriptions
    """
    fields = {
        # Job identification
        "jobid": "Job ID",
        "jobidraw": "Raw Job ID",
        "jobname": "Job name",
        "cluster": "Cluster name",
        # Job status and timing
        "state": "Job state",
        "exitcode": "Exit code",
        "start": "Start time",
        "end": "End time",
        "elapsed": "Elapsed time",
        "submit": "Submit time",
        "eligible": "Eligible time",
        "timelimit": "Time limit",
        "timelimitraw": "Time limit (raw)",
        # Resources
        "partition": "Partition",
        "account": "Account",
        "user": "User",
        "group": "Group",
        "nnodes": "Number of nodes",
        "ncpus": "Number of CPUs",
        "ntasks": "Number of tasks",
        "reqcpus": "Requested CPUs",
        "reqmem": "Requested memory",
        "maxrss": "Maximum RSS memory",
        "maxvmsize": "Maximum VM size",
        "avgcpu": "Average CPU time",
        "avgrss": "Average RSS memory",
        "avgvmsize": "Average VM size",
        # System info
        "nodelist": "Node list",
        "priority": "Priority",
        "qos": "Quality of Service",
        "reason": "Reason (for pending jobs)",
        "reservation": "Reservation",
        "wckey": "WC Key",
        "workdir": "Working directory",
        # Advanced
        "admincomment": "Admin comment",
        "allocnodes": "Allocated nodes",
        "alloctres": "Allocated TRES",
        "blockid": "Block ID",
        "derivedexitcode": "Derived exit code",
        "layout": "Layout",
        "maxdiskread": "Max disk read",
        "maxdiskwrite": "Max disk write",
        "mincpu": "Min CPU time",
        "mincpunode": "Min CPU node",
        "mincputask": "Min CPU task",
        "reqtres": "Requested TRES",
        "systemcpu": "System CPU time",
        "usercpu": "User CPU time",
    }
    return fields


def get_field_presets() -> dict[str, list[str]]:
    """
    Get predefined format field presets for common use cases.

    Returns:
        dictionary mapping preset names to field lists
    """
    presets = {
        "basic": ["jobid", "jobname", "state", "elapsed", "exitcode"],
        "timing": ["jobid", "jobname", "state", "start", "end", "elapsed", "submit"],
        "resources": ["jobid", "jobname", "partition", "nnodes", "ncpus", "reqmem", "state"],
        "detailed": [
            "jobid",
            "jobname",
            "partition",
            "state",
            "start",
            "end",
            "elapsed",
            "exitcode",
            "nnodes",
            "ncpus",
        ],
        "debug": ["jobid", "jobname", "state", "exitcode", "reason", "nodelist"],
        "memory": ["jobid", "jobname", "state", "reqmem", "maxrss", "avgrss", "maxvmsize"],
        "default": [
            "jobid",
            "jobidraw",
            "jobname",
            "partition",
            "state",
            "start",
            "end",
            "elapsed",
            "exitcode",
            "nnodes",
            "ncpus",
        ],
    }
    return presets


def parse_format_fields(format_arg: str) -> list[str]:
    """
    Parse format field argument into a list of fields.

    Args:
        format_arg: Comma-separated list of fields or preset name

    Returns:
        list of format field names
    """
    presets = get_field_presets()

    # Check if it's a preset
    if format_arg in presets:
        return presets[format_arg]

    # Otherwise, split by comma and clean up
    fields = [field.strip() for field in format_arg.split(",")]
    return [field for field in fields if field]  # Remove empty strings


def list_available_options():
    """Print available format fields and presets."""
    print("Available format fields:")
    print("=" * 50)
    fields = get_available_fields()

    # Group fields by category for better readability
    categories = {
        "Job Info": ["jobid", "jobidraw", "jobname", "cluster"],
        "Status & Timing": ["state", "exitcode", "start", "end", "elapsed", "submit", "eligible", "timelimit"],
        "Resources": ["partition", "account", "user", "nnodes", "ncpus", "ntasks", "reqcpus", "reqmem"],
        "Memory": ["maxrss", "maxvmsize", "avgcpu", "avgrss", "avgvmsize"],
        "System": ["nodelist", "priority", "qos", "reason", "reservation", "workdir"],
    }

    for category, field_list in categories.items():
        print(f"\n{category}:")
        for field in field_list:
            if field in fields:
                print(f"  {field:<15} - {fields[field]}")

    print(f"\nPresets:")
    print("=" * 50)
    presets = get_field_presets()
    for name, field_list in presets.items():
        print(f"  {name:<10} - {', '.join(field_list)}")

    print(f"\nNote: Use 'sacct --helpformat' to see all available fields")


def get_job_status(
    job_ids: list[str],
    format_fields: list[str],
    show_allocations: bool = False,
    job_state_filter: Optional[str] = None,
    add_duration: bool = False,
    exclude_completed: bool = False,
) -> str:
    """
    Get job status information using sacct command.

    Args:
        job_ids: list of job IDs to query
        format_fields: list of format fields to display
        show_allocations: Whether to show allocation steps (like -X flag in sacct)
        job_state_filter: Filter jobs by state (e.g., 'COMPLETED', 'FAILED', 'RUNNING')
        add_duration: Whether to calculate and add duration time
        exclude_completed: Whether to exclude completed jobs from results

    Returns:
        Formatted table string with job information
    """
    if not job_ids:
        return "No job IDs to query."

    # Join job IDs with commas for sacct
    job_list = ",".join(job_ids)

    format_string = ",".join(format_fields)

    # Build sacct command
    sacct_cmd = [
        "sacct",
        "--format",
        format_string,
        "--parsable2",  # Use parsable format for easier processing
        "--noheader",  # We'll add our own header
        "-j",
        job_list,
    ]

    # Add allocations flag if requested
    if show_allocations:
        sacct_cmd.append("-X")

    # Handle state filtering
    if job_state_filter and exclude_completed:
        # If both are specified, we need to be careful about conflicts
        if job_state_filter.upper() == "COMPLETED":
            print("Warning: --state COMPLETED conflicts with --exclude-completed. Ignoring --state.")
            # We'll filter completed jobs in post-processing
        else:
            # Use the specific state filter (exclude_completed will be applied in post-processing)
            sacct_cmd.extend(["--state", job_state_filter])
            print(f"Note: Using --state {job_state_filter}. Completed jobs will still be excluded if present.")
    elif job_state_filter:
        # Only state filter specified
        sacct_cmd.extend(["--state", job_state_filter])
    # exclude_completed will be handled in post-processing

    try:
        # Run sacct command
        result = subprocess.run(sacct_cmd, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split("\n")

        if not output_lines or output_lines == [""]:
            return "No jobs found matching the criteria."

        # Process the output and add duration calculation if requested
        processed_lines = []

        # Prepare header
        header_fields = format_fields.copy()
        if add_duration:
            header_fields.append("durationtime")

        header_line = "|".join(f"{field:<15}" for field in header_fields)
        separator_line = "|".join("-" * 15 for _ in header_fields)

        processed_lines.append(header_line)
        processed_lines.append(separator_line)

        # Ensure state field is included if we need to exclude completed jobs
        original_format_fields = format_fields.copy()
        state_idx = None

        if exclude_completed and "state" not in format_fields:
            print("Info: Adding 'state' field to format for filtering completed jobs.")
            format_fields.append("state")
            # Re-run sacct with updated format
            format_string = ",".join(format_fields)
            sacct_cmd[2] = format_string
            result = subprocess.run(sacct_cmd, capture_output=True, text=True, check=True)
            output_lines = result.stdout.strip().split("\n")

        # Find field indices
        start_idx = None
        end_idx = None

        if add_duration:
            try:
                start_idx = format_fields.index("start")
            except ValueError:
                pass
            try:
                end_idx = format_fields.index("end")
            except ValueError:
                pass

        if exclude_completed:
            try:
                state_idx = format_fields.index("state")
            except ValueError:
                pass

        # Prepare header (use original format fields for display)
        display_fields = original_format_fields.copy()
        if add_duration:
            display_fields.append("durationtime")

        header_line = "|".join(f"{field:<15}" for field in display_fields)
        separator_line = "|".join("-" * 15 for _ in display_fields)

        # Process each job line
        for line in output_lines:
            if line.strip():
                fields = line.split("|")
                if len(fields) >= len(format_fields):
                    # Check if we should exclude this job (completed)
                    if exclude_completed and state_idx is not None:
                        if state_idx < len(fields) and "COMPLETED" in fields[state_idx].upper():
                            continue  # Skip completed jobs

                    # Format the fields for display (only original format fields)
                    display_field_count = len(original_format_fields)
                    formatted_fields = [f"{field:<15}" for field in fields[:display_field_count]]

                    # Add duration calculation if requested and possible
                    if add_duration:
                        if start_idx is not None and end_idx is not None:
                            start_time = fields[start_idx] if start_idx < len(fields) else "Unknown"
                            end_time = fields[end_idx] if end_idx < len(fields) else "Unknown"
                            duration = calculate_duration(start_time, end_time)
                        else:
                            duration = "N/A (missing start/end)"
                        formatted_fields.append(f"{duration:<15}")

                    processed_lines.append("|".join(formatted_fields))

        return "\n".join(processed_lines)

    except subprocess.CalledProcessError as e:
        return f"Error running sacct command: {e}\nStderr: {e.stderr}"
    except Exception as e:
        return f"Unexpected error: {e}"


def main():
    """Main function to handle command line arguments and orchestrate the script."""
    parser = argparse.ArgumentParser(
        description="Parse SLURM job IDs from a log file and check their status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python slurm_checker.py                    # Uses sbatch.log in current directory
    python slurm_checker.py logfile.txt        # Uses specified file
    python slurm_checker.py --format basic     # Uses sbatch.log with basic format
    python slurm_checker.py -ec                # Show only active/failed jobs (short flag)
    python slurm_checker.py --exclude-completed # Show only active/failed jobs (long flag)
    python slurm_checker.py logfile.txt --format "jobid,jobname,state,elapsed"
    python slurm_checker.py --state RUNNING --duration
    python slurm_checker.py -ec --format debug # Debug non-completed jobs
    python slurm_checker.py -X --format timing --state RUNNING
    python slurm_checker.py --list-fields
        """,
    )

    parser.add_argument(
        "logfile",
        nargs="?",  # Make optional
        help="Path to the log file containing SLURM job submissions (default: sbatch.log in current directory)",
    )

    parser.add_argument(
        "-X", "--allocations", action="store_true", help="Show only allocation records (equivalent to sacct -X)"
    )

    parser.add_argument("--state", type=str, help="Filter jobs by state (e.g., COMPLETED, FAILED, RUNNING, PENDING)")

    parser.add_argument(
        "-ec",
        "--exclude-completed",
        action="store_true",
        help="Exclude completed jobs from results (useful for monitoring active/failed jobs)",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="default",
        help="Format fields to display. Use preset name (basic, timing, resources, detailed, debug, memory, default) or comma-separated list of fields",
    )

    parser.add_argument(
        "--duration",
        action="store_true",
        help="Calculate and display duration time (requires 'start' and 'end' fields)",
    )

    parser.add_argument("--job-ids", action="store_true", help="Only print the parsed job IDs (useful for debugging)")

    parser.add_argument(
        "--list-fields", action="store_true", help="list available format fields and presets, then exit"
    )

    args = parser.parse_args()

    # Handle --list-fields option
    if args.list_fields:
        list_available_options()
        return

    # Handle default logfile
    if not args.logfile:
        args.logfile = "sbatch.log"
        print(f"No logfile specified, using default: {args.logfile}")

        # Check if default file exists and provide helpful message
        if not os.path.exists(args.logfile):
            print(f"Warning: Default file '{args.logfile}' does not exist in current directory.")
            print("Please ensure the file exists or specify a different log file.")

            # list .log files in current directory as suggestions
            log_files = [f for f in os.listdir(".") if f.endswith(".log")]
            if log_files:
                print(f"Found these .log files in current directory: {', '.join(log_files)}")

    # Parse format fields
    try:
        format_fields = parse_format_fields(args.format)
    except Exception as e:
        print(f"Error parsing format fields: {e}")
        sys.exit(1)

    if not format_fields:
        print("Error: No valid format fields specified")
        sys.exit(1)

    print(f"Using format fields: {', '.join(format_fields)}")

    # Parse job IDs from the log file
    job_ids = parse_job_ids(args.logfile)

    if not job_ids:
        print("No job IDs found in the file.")
        sys.exit(1)

    # If user just wants to see the job IDs
    if args.job_ids:
        print(f"Parsed job IDs: {', '.join(job_ids)}")
        return

    # Validate duration option
    if args.duration and not ("start" in format_fields and "end" in format_fields):
        print("Warning: Duration calculation requires both 'start' and 'end' fields.")
        print("Adding 'start' and 'end' to format fields...")
        if "start" not in format_fields:
            format_fields.append("start")
        if "end" not in format_fields:
            format_fields.append("end")

    # Get job status information
    filter_info = []
    if args.state:
        filter_info.append(f"state={args.state}")
    if args.exclude_completed:
        if not args.state or args.state.upper() != "COMPLETED":
            filter_info.append("excluding completed jobs")

    filter_text = f" (filtering: {', '.join(filter_info)})" if filter_info else ""
    print(f"\nQuerying job status with sacct for {len(job_ids)} jobs{filter_text}...\n")

    job_status = get_job_status(
        job_ids, format_fields, args.allocations, args.state, args.duration, args.exclude_completed
    )
    print(job_status)


if __name__ == "__main__":
    main()
