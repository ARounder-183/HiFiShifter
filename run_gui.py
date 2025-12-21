import sys
import os

# Ensure the current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hifi_shifter.main import main

if __name__ == '__main__':
    main()
