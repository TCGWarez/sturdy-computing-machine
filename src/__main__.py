"""
src/__main__.py: Entry point for running CLI as module
Allows: python -m src.cli.main <command>
"""

from src.cli.main import cli

if __name__ == '__main__':
    cli()

