dataset_config = {
    "name": "mr",
    "tvt_split": [0.9, 0, 0.1],
    "random_seed": 42,
    "vocab_size": None,
    "preprocess": {"remove_stopwords": False, "remove_rare_words": 0},
    "encoding": {
        "embedding_dim": 300,
        "encode_token_type": "glove",
        "tokens_trained_on": 6,
    },
}

model_config = {
    "model_type": "lstm",
    "embedding_dim": 300,
    "common_params": {
        "batch_size": 64,
        "num_epochs": 20,
    },
    "model_specific_params": {
        "output_size": 2,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": True,
        "output_size": 2,
        "learning_rate": 0.001,
        "freeze_embeddings": True,
    },
}

run_config = {
    "dataset_config": dataset_config,
    "model_config": model_config,
}


def func(**config):
    for key, value in config.items():
        if type(value) is dict:
            func(**value)
        else:
            print(f"{key}: {value}, {type(value)}\n")


func(**run_config, idk=1)

# print(**run_config, idk=1)
