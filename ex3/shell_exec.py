import subprocess
import os

def run_shell(command: str, shell: str = None):
    # Default to bash on Linux/Mac, and powershell on Windows to support commands like 'ls'
    if shell is None:
        shell = "/bin/bash" if os.name != "nt" else "powershell.exe"
        
    kwargs = {
        "shell": True,
        "text": True,
        "capture_output": True,
        "timeout": 20,
    }
    
    # On Windows, passing executable with shell=True can sometimes be tricky,
    # but powershell.exe usually works if it's in the PATH. 
    # Alternatively, we can just run it without 'executable' to use cmd.exe, 
    # but then 'ls' wouldn't work. We'll specify the executable.
    if os.name != "nt":
        kwargs["executable"] = shell
    else:
        # For Windows, running powershell via list format is safer than shell=True + executable
        return _run_windows_powershell(command)

    result = subprocess.run(command, **kwargs)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }

def _run_windows_powershell(command: str):
    result = subprocess.run(
        ["powershell.exe", "-Command", command],
        text=True,
        capture_output=True,
        timeout=20,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }