import importlib
import os


def registry(path: str, name: str) -> None:
    """Register entire folder.

    Args:
        path: A path to __init__ file.
        name: A module name.
    """
    module_dir = path
    for file in os.listdir(module_dir):
        if os.path.isdir(os.path.join(module_dir, file)) and file != "__pycache__":
            for subfile in os.listdir(os.path.join(module_dir, file)):
                _ = os.path.join(module_dir, file, subfile)
                if subfile.endswith(".py"):
                    class_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                    _ = importlib.import_module(f"{name}.{file}.{class_name}")
            continue


__all__ = ("registry",)
