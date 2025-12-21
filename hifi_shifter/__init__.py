import sys
import os

# Fix for "Unable to find the current directory"
try:
    os.getcwd()
except (FileNotFoundError, OSError):
    # Fallback to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Fix for PyQt6 DLL load failed on Windows Conda environments
if os.name == 'nt':
    # Add Library/bin to PATH so DLLs can be found
    conda_lib_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.exists(conda_lib_bin):
        os.environ['PATH'] = conda_lib_bin + os.pathsep + os.environ['PATH']

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
