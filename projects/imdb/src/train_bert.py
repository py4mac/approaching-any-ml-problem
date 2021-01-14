import io
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics, model_selection
import projects.imdb.src.config_bert as config
import projects.imdb.src.dataset_bert as dataset
from projects.imdb.src.model import BERTBaseUncased
import projects.imdb.src.engine_bert as engine

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def train():
    dfx = pd.read_csv(f"{config.TRAINING_FILE}").fillna("none")

    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset= dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.sentiment.values
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=3
    )

    valid_dataset= dataset.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.sentiment.values
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device("cpu")

    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["biais", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n,p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001
        },
        {
            "params": [
                p for n,p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS
    )

    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    print("Training Model")
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(
            valid_data_loader, model, device
        )
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(
            f" ACCURACY SCORE {accuracy}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
if __name__ == "__main__":

    train()
    