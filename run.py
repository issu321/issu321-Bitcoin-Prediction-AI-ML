#!/usr/bin/env python3
"""
Bitcoin Price Prediction - Run Script
=====================================
This script initializes the database and starts the Flask application.

Usage:
    python run.py

The application will be available at http://localhost:5000

Default admin credentials:
    Username: admin
    Password: admin123
"""

import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, init_db

if __name__ == '__main__':
    print("=" * 60)
    print("  Bitcoin Price Prediction - Machine Learning Platform")
    print("=" * 60)
    print()
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
    print()
    print("Starting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print()
    print("Default admin login:")
    print("  Username: admin")
    print("  Password: admin123")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
