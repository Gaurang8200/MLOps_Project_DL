import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.models import *
from plotly import express as px
from collections import Counter
import numpy as np
import random
import os
import multiprocessing

from modules.dataset import IntelImageClassificationDataset
from modules.utility import NotebookPlotter, InferenceSession, Evaluator, ISO_time, apply_pruning
from modules.trainer import Trainer
from modules.optuna_monashara import run_optuna
from modules.BufferDataset import ShuffleBufferDataset


torch.manual_seed(1)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False


set_seed(1)


def main():
    print("Using device:", DEVICE)   
    # same variables as in your sandbox
    choice = 1  # 1,2,3
    freezeLayer = True
    pretrained_Weights = True
    prune_model = False
    OPTUNA_MO = False
    Multi_B = True

    if choice != 5:
        dataset = IntelImageClassificationDataset(resize=(150, 150))
    else:
        dataset = IntelImageClassificationDataset(resize=(384, 384))

    # 80% train, 20% validation for training Optuna
    train_size = int(0.8 * len(dataset.train_dataset))
    val_size = len(dataset.train_dataset) - train_size
    train_subset, val_subset = random_split(
        dataset.train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1),
    )

    def build_model():
        # SqueezeNet 1.1
        if choice == 1:
            if pretrained_Weights:
                model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
            else:
                model = models.squeezenet1_1()

            num_features = model.classifier[1].in_channels
            kernel_size = model.classifier[1].kernel_size
            if freezeLayer:
                for param in model.parameters():
                    param.requires_grad = False
            model.classifier[1] = nn.Conv2d(num_features, 6, kernel_size)

        # MobileNetV3 Small
        elif choice == 2:
            if pretrained_Weights:
                model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            else:
                model = models.mobilenet_v3_small()
            num_features = model.classifier[3].in_features
            if freezeLayer:
                for param in model.parameters():
                    param.requires_grad = False
            model.classifier[3] = nn.Linear(num_features, 6)

        else:
            raise ValueError("choice must be 1 (SqueezeNet) or 2 (MobileNetV3 Small)")

        if prune_model:
            model = apply_pruning(model, amount=0.3)

        return model
    
   # ---- Hyperparameter Tuning ---- #

    if OPTUNA_MO:
        model = build_model()

        best_params, best_model_state, study = run_optuna(
            model=model,
            train_subset=train_subset,
            val_subset=val_subset,
            TrainerClass=Trainer,
            n_trials=12,
            seed=1,
        )

        print("▶ Per-epoch validation accuracy (best trial):")
        best_trial = study.best_trial
        for epoch, acc in sorted(best_trial.intermediate_values.items()):
            print(f"   Epoch {epoch:2d}: {acc * 100:.2f}%")

        print(f"\n▶ Best hyperparameters: {best_params}")
        print(f"▶ Best overall accuracy: {study.best_value * 100:.2f}%")

        model.load_state_dict(best_model_state)

        dataloader = DataLoader(
            dataset.train_dataset,
            batch_size=best_params["BS_SUGGEST"],
            shuffle=True,
        )
        trainer = Trainer(model=model, lr=best_params["LR_SUGGEST"], device=DEVICE)
        epochs = best_params["EPOCHS"]

        # comment stays as in your notebook
        ''' BS_SUGGEST': 32, 'LR_SUGGEST': 8.841926348917726e-05, 'EPOCHS': 25 suggested from the OPTUNA
            and achieve the accuracy of 86.7 % on Testdata.'''

    else:
        model = build_model()
        dataloader = DataLoader(dataset.train_dataset, batch_size=32, shuffle=False)
        trainer = Trainer(model=model, lr=8.841926348917726e-05, device=DEVICE)
        epochs = 25

    # ---- Multi_B ---- #
    if Multi_B:
        #workers = max(1, multiprocessing.cpu_count() // 2)
        workers = multiprocessing.cpu_count() // 2
        print(f"[INFO] Enabling DataLoader multiprocessing with workers={workers}")

        dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )

    # ---- Training ----
    print("[INFO] Starting training ...")
    trainer.train(dataloader, epochs=epochs, silent=False)

    # ---- Evaluation ----
    print("[INFO] Running evaluation on test dataset ...")
    session = InferenceSession(model.to(DEVICE))
    test_images = torch.stack(tuple(item[0] for item in dataset.test_dataset)).to(DEVICE)
    test_labels = torch.tensor(tuple(item[1] for item in dataset.test_dataset)).to(DEVICE)

    with torch.no_grad():
        output = session(test_images)

    acc = Evaluator.acc(output, test_labels).item()
    print(f"[RESULT] Test accuracy: {acc * 100:.2f} %")
    
    # ------------------------------------------------------------------
    #                 SAVE MODEL FOR DEPLOYMENT
    # ------------------------------------------------------------------
    os.makedirs("saved_model", exist_ok=True)

    # Save PyTorch model weights (.pth)
    torch.save(model.state_dict(), "saved_model/model_weights.pth")
    print("[INFO] Saved model weights → saved_model/model_weights.pth")

    # Save TorchScript model (Vertex AI-ready)
    scripted = torch.jit.script(model.cpu())
    scripted.save("saved_model/model_scripted.pt")
    print("[INFO] Saved TorchScript model → saved_model/model_scripted.pt")


if __name__ == "__main__":
    main()