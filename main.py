import os
import sys

# Add the flask directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'flask'))

# Import the Flask app instance from flask/app.py
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
