#!/usr/bin/env python3
"""
Simple HTTP server to serve the Sphinx documentation.
VS Code will automatically port-forward this for viewing in the browser.
"""
import http.server
import socketserver
import os
import sys
from pathlib import Path

def serve_docs(port=8000):
    """Serve the documentation on localhost."""
    
    # Get the documentation directory
    docs_dir = Path(__file__).parent / "build" / "html"
    
    if not docs_dir.exists():
        print(f"❌ Documentation not found at: {docs_dir}")
        print("Please run 'make html' first to build the documentation.")
        return
    
    # Change to the documentation directory
    os.chdir(docs_dir)
    
    # Create the server
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"🚀 Serving documentation at http://localhost:{port}")
            print(f"📁 Serving from: {docs_dir.absolute()}")
            print("📝 VS Code should auto port-forward this URL")
            print("🛑 Press Ctrl+C to stop the server")
            print("-" * 50)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            serve_docs(port + 1)
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    # Allow custom port via command line argument
    port = 8888
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number. Using default port 8000.")
    
    serve_docs(port)
