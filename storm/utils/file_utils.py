import os
import joblib
from typing import Any

def save_joblib(obj: Any, path: str):
    assert obj is not None, 'Object to save should not be None'
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)

def load_joblib(path: str) -> Any:
    if path and os.path.exists(path):
        return joblib.load(path)
    else:
        return None

def assemble_project_path(path):
    """Assemble a path relative to the project root directory"""
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path

def gen_relative_project_path(path):

    root = get_project_root()

    if root not in path:
        raise ValueError('Path to convert should be within the project root.')
    
    path = path.replace(root, '.').replace('.\\', '')
    return path

def exists_in_project_path(path):
    return os.path.exists(assemble_project_path(path))

def get_project_root():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.dirname(path) # get to parent, outside of project code path"
    return path

def read_resource_file(path):

    assert path.startswith("./res/") or path.startswith("res/") , 'Path should start with ./res/ or res/'

    with open(assemble_project_path(path), "r", encoding="utf-8") as fd:
        return fd.read()

def save_html(html: str, path: str):
    with open(path, "w") as f:
        f.write(html)

def init_path(path):
    os.makedirs(path, exist_ok=True)
    return path