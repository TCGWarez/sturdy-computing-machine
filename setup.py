"""
setup.py: Setup script for MTG Card Recognition System
"""

from setuptools import setup, find_packages

setup(
    name="mtg-image-recog",
    version="0.1.0",
    description="Two-stage MTG card recognition system",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "imagehash>=4.3.1",
        "pytesseract>=0.3.10",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "faiss-cpu>=1.7.4",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "sqlalchemy>=2.0.0",
        "pydantic>=2.5.0",
        "click>=8.1.7",
        "python-dotenv>=1.0.0",
        "python-Levenshtein>=0.21.1",
        "fuzzywuzzy>=0.18.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        'console_scripts': [
            'mtg-recog=src.cli.main:cli',
        ],
    },
)

