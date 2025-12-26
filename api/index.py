#!/usr/bin/env python3
"""
Vercel Serverless Entry Point
This file serves as the entry point for Vercel Functions
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Import your Flask app from app.py
    from app import app
    
    # Vercel expects a WSGI application named 'app'
    # This is the entry point that Vercel will call
    print("✅ City Explorer API loaded successfully")
    
    # Export the Flask app for Vercel
    # Vercel will look for a variable named 'app'
    app = app
    
except Exception as e:
    print(f"❌ Failed to import Flask app: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in api/: {os.listdir(os.path.dirname(__file__))}")
    raise