#!/usr/bin/env python3
"""
Launcher script for the MCP server with proper environment setup.
"""

import os
import sys
import subprocess

# Set the Python path to include our dependencies
os.environ['PYTHONPATH'] = '/Users/taimoorawan/.pyenv/versions/3.12.3/lib/python3.12/site-packages'

# Change to the server directory
os.chdir('/Users/taimoorawan/Documents/Liquid_intelligence/mcp-server')

# Import and run the server
from server import mcp

if __name__ == "__main__":
    mcp.run() 