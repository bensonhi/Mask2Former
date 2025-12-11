"""
Utility functions for detecting and loading the Mask2Former project.
"""

import os
import sys


__all__ = ['find_mask2former_project', 'ensure_mask2former_loaded']


# Global variable to track if Mask2Former is loaded
_project_dir = None


def find_mask2former_project(explicit_path=None):
    """
    Find Mask2Former project directory using multiple methods.

    Args:
        explicit_path: Explicitly provided path (highest priority)

    Returns:
        str: Path to Mask2Former project directory
    """

    # Method 1: Explicit path parameter (highest priority)
    if explicit_path:
        if os.path.exists(os.path.join(explicit_path, 'demo', 'predictor.py')):
            print(f"📁 Using Mask2Former from parameter: {explicit_path}")
            return explicit_path
        else:
            print(f"⚠️  Warning: Provided path '{explicit_path}' does not contain Mask2Former")
            print("   Falling back to other detection methods...")

    # Method 2: Environment variable
    project_path = os.environ.get('MASK2FORMER_PATH')
    if project_path and os.path.exists(os.path.join(project_path, 'demo', 'predictor.py')):
        print(f"📁 Using Mask2Former from environment: {project_path}")
        return project_path

    # Method 3: Config file in same directory as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: utils/ -> fiji_integration/
    fiji_integration_dir = os.path.dirname(script_dir)
    config_file = os.path.join(fiji_integration_dir, 'mask2former_config.txt')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            project_path = f.read().strip()
        if project_path and os.path.exists(os.path.join(project_path, 'demo', 'predictor.py')):
            print(f"📁 Using Mask2Former from config file: {project_path}")
            return project_path

    # Method 4: Auto-detection in common locations
    possible_paths = [
        os.path.dirname(fiji_integration_dir),  # Parent of fiji_integration directory
        '/fs04/scratch2/tf41/ben/Mask2Former',  # Your current path
        '/Users/wangbingsheng/PycharmProjects/CSIRO-UROP/Mask2Former',
        '/home/bwang/ar85_scratch2/ben/download/Mask2Former',
        os.path.expanduser('~/Mask2Former'),
        os.path.expanduser('~/CSIRO-UROP/Mask2Former'),
        os.getcwd()  # Current working directory
    ]

    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'demo', 'predictor.py')):
            print(f"📁 Auto-detected Mask2Former at: {path}")
            return path

    # Method 5: Error with helpful instructions
    print("❌ Could not find Mask2Former project!")
    print("🔧 To fix this, choose one of these options:")
    print("   1. Use --mask2former-path argument: --mask2former-path /path/to/Mask2Former")
    print("   2. Set environment variable: export MASK2FORMER_PATH='/fs04/scratch2/tf41/ben/Mask2Former'")
    print(f"   3. Create config file: echo '/fs04/scratch2/tf41/ben/Mask2Former' > {config_file}")
    print("   4. Make sure the project is in one of these locations:")
    for path in possible_paths:
        print(f"      - {path}")

    raise ImportError("Mask2Former project not found. See instructions above.")


def ensure_mask2former_loaded(explicit_path=None):
    """
    Ensure Mask2Former project is loaded into Python path.

    Args:
        explicit_path: Explicitly provided path to Mask2Former project (optional)

    Returns:
        str: Path to Mask2Former project directory
    """
    global _project_dir
    if _project_dir is None:
        _project_dir = find_mask2former_project(explicit_path=explicit_path)
        if _project_dir not in sys.path:
            sys.path.insert(0, _project_dir)
    return _project_dir
