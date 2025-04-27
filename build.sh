#!/bin/bash

echo "Starting build process..."
echo "Python version:"
python --version

echo "Current directory:"
pwd

echo "Directory contents:"
ls -la

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Build completed" 