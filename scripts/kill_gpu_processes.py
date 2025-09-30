#!/usr/bin/env python3
"""
Force kill all GPU processes without confirmation.
This script identifies and terminates all processes using NVIDIA GPUs.
"""

import subprocess
import os
import sys
import signal
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Whitelist configuration - processes containing these keywords will not be killed
WHITELIST_KEYWORDS = ['tensorboard', 'app']


def is_whitelisted(command):
    """Check if a process command contains any whitelisted keywords."""
    command_lower = command.lower()
    for keyword in WHITELIST_KEYWORDS:
        if keyword.lower() in command_lower:
            return True
    return False


def get_gpu_processes():
    """Get list of processes using GPU."""
    try:
        # Run nvidia-smi to get process information
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.strip().split(', ')
                if len(parts) >= 2:
                    pid = int(parts[0])
                    name = parts[1]
                    processes.append((pid, name))
        
        return processes
    except subprocess.CalledProcessError:
        logger.error("Failed to run nvidia-smi. Make sure NVIDIA drivers are installed.")
        return []
    except Exception as e:
        logger.error(f"Error getting GPU processes: {e}")
        return []


def kill_process(pid):
    """Force kill a process by PID."""
    try:
        # Try SIGTERM first
        os.kill(pid, signal.SIGTERM)
        logger.info(f"Sent SIGTERM to process {pid}")
        
        # Give it a moment to terminate gracefully
        import time
        time.sleep(0.5)
        
        # Check if process still exists and force kill if needed
        try:
            os.kill(pid, 0)  # Check if process exists
            # Process still exists, force kill
            os.kill(pid, signal.SIGKILL)
            logger.info(f"Force killed process {pid} with SIGKILL")
        except ProcessLookupError:
            # Process already terminated
            pass
            
    except ProcessLookupError:
        logger.warning(f"Process {pid} not found (may have already terminated)")
    except PermissionError:
        logger.error(f"Permission denied to kill process {pid}. Try running with sudo.")
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")


def get_python_processes():
    """Get list of python, python3, and torchrun processes."""
    try:
        # Get all processes
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            check=True
        )
        
        processes = []
        whitelisted = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split(None, 10)  # Split into max 11 parts
            if len(parts) >= 11:
                pid = int(parts[1])
                command = parts[10]
                
                # Check if it's a python-related process
                if any(proc in command for proc in ['python', 'python3', 'torchrun']):
                    # Skip this script itself
                    if 'kill_gpu_processes.py' not in command:
                        # Check if process is whitelisted
                        if is_whitelisted(command):
                            whitelisted.append((pid, command[:100]))
                        else:
                            processes.append((pid, command[:100]))  # Truncate long commands
        
        # Log whitelisted processes
        if whitelisted:
            logger.info(f"Found {len(whitelisted)} whitelisted Python process(es):")
            for pid, cmd in whitelisted:
                logger.info(f"  [WHITELISTED] PID: {pid}, Command: {cmd}")
        
        return processes
    except subprocess.CalledProcessError:
        logger.error("Failed to run ps command.")
        return []
    except Exception as e:
        logger.error(f"Error getting Python processes: {e}")
        return []


def force_kill_process(pid):
    """Force kill a process using SIGKILL (kill -9)."""
    try:
        os.kill(pid, signal.SIGKILL)
        logger.info(f"Force killed process {pid} with SIGKILL (kill -9)")
    except ProcessLookupError:
        logger.warning(f"Process {pid} not found (may have already terminated)")
    except PermissionError:
        logger.error(f"Permission denied to kill process {pid}. Try running with sudo.")
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")


def main():
    """Main function to kill all GPU and Python processes."""
    logger.info("Starting GPU and Python process termination...")
    
    # Kill Python processes first
    logger.info("\n=== Killing Python processes ===")
    python_processes = get_python_processes()
    
    if python_processes:
        logger.info(f"Found {len(python_processes)} Python process(es):")
        for pid, command in python_processes:
            logger.info(f"  PID: {pid}, Command: {command}")
        
        logger.info("\nKilling all Python processes with kill -9...")
        for pid, command in python_processes:
            logger.info(f"Force killing PID {pid}...")
            force_kill_process(pid)
    else:
        logger.info("No Python processes found.")
    
    # Then kill GPU processes
    logger.info("\n=== Killing GPU processes ===")
    gpu_processes = get_gpu_processes()
    
    if not gpu_processes:
        logger.info("No GPU processes found.")
    else:
        # Separate whitelisted and non-whitelisted GPU processes
        gpu_to_kill = []
        gpu_whitelisted = []
        
        for pid, name in gpu_processes:
            if is_whitelisted(name):
                gpu_whitelisted.append((pid, name))
            else:
                gpu_to_kill.append((pid, name))
        
        # Log all GPU processes
        logger.info(f"Found {len(gpu_processes)} GPU process(es) total:")
        if gpu_whitelisted:
            for pid, name in gpu_whitelisted:
                logger.info(f"  [WHITELISTED] PID: {pid}, Name: {name}")
        for pid, name in gpu_to_kill:
            logger.info(f"  PID: {pid}, Name: {name}")
        
        # Kill non-whitelisted GPU processes
        if gpu_to_kill:
            logger.info(f"\nKilling {len(gpu_to_kill)} non-whitelisted GPU process(es)...")
            killed_count = 0
            
            for pid, name in gpu_to_kill:
                logger.info(f"Killing {name} (PID: {pid})...")
                kill_process(pid)
                killed_count += 1
            
            logger.info(f"\nCompleted. Attempted to kill {killed_count} GPU process(es).")
        else:
            logger.info("\nNo GPU processes to kill (all are whitelisted).")
    
    # Verify all processes are gone
    logger.info("\n=== Verification ===")
    remaining_gpu = get_gpu_processes()
    remaining_python = get_python_processes()
    
    if remaining_gpu:
        logger.warning(f"Warning: {len(remaining_gpu)} GPU process(es) still running:")
        for pid, name in remaining_gpu:
            logger.warning(f"  PID: {pid}, Name: {name}")
    
    if remaining_python:
        logger.warning(f"Warning: {len(remaining_python)} Python process(es) still running:")
        for pid, command in remaining_python:
            logger.warning(f"  PID: {pid}, Command: {command}")
    
    if remaining_gpu or remaining_python:
        logger.warning("You may need to run this script with sudo or manually kill these processes.")
    else:
        logger.info("All GPU and Python processes successfully terminated.")


if __name__ == "__main__":
    main()
