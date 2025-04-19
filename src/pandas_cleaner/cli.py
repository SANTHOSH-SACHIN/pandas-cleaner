import argparse
import sys
import os
import subprocess
import streamlit
from . import app

def main():
    """Command line interface for pandas-cleaner."""
    parser = argparse.ArgumentParser(description="PandasCleaner - Interactive data cleaning tool")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the pandas-cleaner application')
    start_parser.add_argument('--port', type=int, default=8501, help='Port to run the application on')

    args = parser.parse_args()

    if args.command == 'start':
        try:
            cmd = [
                'streamlit',
                'run',
                app.__file__,
                '--server.port',
                str(args.port)
            ]
            subprocess.run(cmd)
            return 0
        except KeyboardInterrupt:
            print("\nStopping pandas-cleaner application...")
            return 0
        except Exception as e:
            print(f"Error starting pandas-cleaner: {str(e)}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
