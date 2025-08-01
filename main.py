import json
import os
import argparse
import utils
from loadData import load_data
import torch
from torch.utils.data import DataLoader


def parse_json(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The file '{config_path}' does not exist.")

    with open(config_path, "r") as file:
        data = json.load(file)

    return data


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run model from JSON config")
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     required=True,
    #     help="Path to JSON config file (e.g. runConfigTextGCN.json)"
    # )

    # args = parser.parse_args()
    # config = args.config
    # path_to_configs = utils.get_configs_path()
    # parsed = parse_json(path_to_configs +"/"+ config)
    # print("Model type:", parsed["model_type"])  # e.g., "TextGCN"

    # model_type = parsed["model_type"]
    # dataset = parsed["dataset"]
    # common_params = parsed["common_params"]
    # print(type(model_type), type(dataset), type(common_params))
    # print(dataset["tvt_split"][0])
    dataset_config = {
        "name": "mr",
        "tvt_split": [0.9, 0, 0.1],
        "random_seed": 42,
        "vocab_size": 10000,
        "preprocess": {"remove_stopwords": False, "remove_rare_words": 0},
        "encoding": {
            "embedding_dim": 300,
            "encode_token_type": "glove",
            "tokens_trained_on": 6,
        },
    }

    model_config = {
        "model_type": "fastText",
        "embedding_dim": 60,
        "common_params": {
            "batch_size": 128,
            "num_epochs": 25,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "model_specific_params": {
            "output_size": 2,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.95,
            "bidirectional": True,
            "output_size": 2,
            "learning_rate": 0.001,
            "freeze_embeddings": False,
        },
    }
    train_data_set = load_data(
        dataset_config=dataset_config,
        model_type=model_config["model_type"],
        split="train",
    )

    test_data_set = load_data(
        dataset_config=dataset_config,
        model_type=model_config["model_type"],
        split="test",
    )

    print("Train dataset loaded:", train_data_set)
    print("Test dataset loaded:", test_data_set)
    print("Vocabulary size:", len(train_data_set.vocab))
    print(train_data_set.encode_token_type)
    print(train_data_set)
    print(len(train_data_set), "train dataset length")
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def lstm_collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = zip(*batch)

        # Determine if inputs are 1D (index) or 2D (glove vectors)
        if inputs[0].dim() == 1:
            # Pad 1D sequences (index encoding)
            padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        elif inputs[0].dim() == 2:
            # Pad 2D sequences (glove: [seq_len, embedding_dim])
            max_len = max(seq.shape[0] for seq in inputs)
            embedding_dim = inputs[0].shape[1]
            padded_inputs = torch.zeros(len(inputs), max_len, embedding_dim)
            for i, seq in enumerate(inputs):
                padded_inputs[i, : seq.shape[0], :] = seq
        else:
            raise ValueError("Unsupported tensor shape")

        labels = torch.stack(labels)
        return padded_inputs, labels

    train_data_loader = DataLoader(
        train_data_set,
        batch_size=model_config["common_params"]["batch_size"],
        shuffle=True,
        collate_fn=lstm_collate_fn,
    )
    test_data_loader = DataLoader(
        test_data_set,
        batch_size=model_config["common_params"]["batch_size"],
        shuffle=False,
        collate_fn=lstm_collate_fn,
    )

    # test the datraloader
    for batch in train_data_loader:
        inputs, labels = batch
        print("Batch inputs shape:", inputs.shape)
        print("Batch labels shape:", labels.shape)
        break  # Remove this to iterate through the entire dataset

    from modelFactory import create_model

    model = create_model(
        dataset_config=dataset_config,
        model_config=model_config,
    )
    print("Model created:", model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    import train

    trained_model = train.train_model(
        model=model,
        dataloaders={"train": train_data_loader, "test": test_data_loader},
        config=model_config,
    )
    # torch.save(trained_model.state_dict(), "trained_lstm_model.pth")
    # print("Model trained and saved as 'trained_lstm_model.pth'")
    import eval

    eval.evaluate(
        model=trained_model,
        data_loader=test_data_loader,
        device=model_config["common_params"]["device"],
    )


# train_dataset, val_dataset, test_dataset = load_data(model_type, dataset, common_params)

# train_loader = DataLoader(train_dataset, batch_size=common_params["batch_size"], shuffle=True)

# from loaders import lstmLoader
