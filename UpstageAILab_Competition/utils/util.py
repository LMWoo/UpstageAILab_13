
import importlib

def dynamic_preprocessing_import(root_module: str, sub_path: str):
    *submodules, class_name = sub_path.split(".")
    full_module_path = ".".join([root_module] + submodules)
    module = importlib.import_module(full_module_path)
    return getattr(module, class_name)