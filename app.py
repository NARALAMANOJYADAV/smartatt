import os
import sys
import importlib.util

# Add the project root to sys.path so nested modules can resolve imports from root (like 'engine')
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Dynamically load the 'flask_app/app.py' module to completely avoid circular import / self-import conflicts
flask_app_path = os.path.abspath(os.path.join(project_root, 'flask_app', 'app.py'))
spec = importlib.util.spec_from_file_location("flask_app_module", flask_app_path)
flask_app_module = importlib.util.module_from_spec(spec)
sys.modules["flask_app_module"] = flask_app_module
spec.loader.exec_module(flask_app_module)

# Expose the Flask app instance so Gunicorn can locate it via 'app:app'
app = flask_app_module.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
