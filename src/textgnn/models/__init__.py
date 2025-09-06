"""Package that contains models as subpackages. Each subpackage should contain these modules: model.py, train.py 
model.py should inlude:
    - create_model function that returns an instance of the model (named create_<model_name>_model)
    - model class definition
    - model should also include a class attribute 'train_func' that points to the training function in train.py
train.py should include:
    - train function object - either a user defined function or a reference to a common training function in the textgnn.train module
"""

__all__ = ["lstm", "mlp", "fastText", "text_gcn"]