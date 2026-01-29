#!/usr/bin/env python3
"""
Simple HTTP server for local testing of the CDT Online experiment.

Usage:
    python run_local_server.py [port]
    
Default port is 8000. Access at http://localhost:8000
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

# Default port
PORT = 8000

# Parse command line argument for port
if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except ValueError:
        print(f"Invalid port: {sys.argv[1]}. Using default port {PORT}")

# Change to CDT_Online directory
script_dir = Path(__file__).parent
cdt_online_dir = script_dir.parent
os.chdir(cdt_online_dir)

print(f"=" * 60)
print(f"CDT Online - Local Test Server")
print(f"=" * 60)
print(f"")
print(f"Serving from: {cdt_online_dir}")
print(f"Port: {PORT}")
print(f"")
print(f"Access the experiment at:")
print(f"  Main experiment: http://localhost:{PORT}/")
print(f"  Browser test:    http://localhost:{PORT}/test/browser_test.html")
print(f"")
print(f"Press Ctrl+C to stop the server")
print(f"=" * 60)

# Custom handler to set correct MIME types
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        '': 'application/octet-stream',
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.csv': 'text/csv',
        '.psyexp': 'application/xml',
    }
    
    def end_headers(self):
        # Add CORS headers for local testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

# Create server
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    # Open browser automatically
    try:
        webbrowser.open(f"http://localhost:{PORT}/test/browser_test.html")
    except:
        pass
    
    # Serve forever
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
