"""
api/index.py — Vercel Serverless Entry Point
Wraps server.py logic for Vercel's @vercel/python runtime.
The artifact (artifacts/can_ids_model.pkl) is bundled with the deployment.
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app from server.py (Vercel will call `app`)
from server import app

# Vercel expects the WSGI app to be named `app`
