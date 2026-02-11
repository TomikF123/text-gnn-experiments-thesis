"""
A subpackage for data loaders. 
There are two types of modules,a loader modules (ends with loader) and a helper module (does not end with loader).
Each loader module helps the load_data module to load data for a specific model type.
In each loader module, the following functions are expected to be implemented:
    - create_file_name: creates a unique file name for the dataset based on the dataset configuration and model type.
    - get_dataset_class: returns the dataset class for the specific model type.
    - get_dataset_object_func: returns a function that creates and returns the dataset object.

To extend this framework with a model type that is not compatibile with the 
"""

__all__ = ["lstmLoader", "fastTextLoader"]