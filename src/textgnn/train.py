TRAINING_LOOPS = {
    "lstm": "models.lstm.train.train_lstm",
    "text_gcn": "models.text_gcn.train.train_text_gcn",  # TODO
}

from .utils import get_function_from_path

def basic_inductive_training_loop(dataloader, model, config): #TODO?
    ...

def basic_transductive_training_loop(dataloader, model, config): #TODO?
    ...

def train_model(model, dataloaders, config):
    model_type = config.get("model_type",None)
    assert(model_type is not None)
    if model_type not in TRAINING_LOOPS:
        raise ValueError(f"Unsupported model type: {model_type}")

    train_fn = model.train_func
    # return train_fn(data=dataloaders, model=model) # TODO Bruh.... even model specific training funcs should have the same args right?
    return train_fn(dataloader=dataloaders, model=model, config=config)
