import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.models import *
from plotly import express as px
from google.cloud import storage
from collections import Counter
import numpy as np
import random
import os
import multiprocessing
import subprocess 

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
    choice = 1  # 1,2,3
    freezeLayer = True
    pretrained_Weights = True
    prune_model = False
    OPTUNA_MO = False
    Multi_B = False

    if choice != 2:
        dataset = IntelImageClassificationDataset(
            resize=(150, 150),
            bucket="mlops-project-intel-data",
            train_prefix="seg_pred/seg_train",
            test_prefix="seg_pred/seg_test",
            eval_prefix="seg_pred/seg_pred",
        )
    else:
        dataset = IntelImageClassificationDataset(
            resize=(384, 384),
            bucket="mlops-project-intel-data",
            train_prefix="seg_pred/seg_train",
            test_prefix="seg_pred/seg_test",
            eval_prefix="seg_pred/seg_pred",
        )

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
    history = trainer.train(dataloader, epochs=epochs, silent=False)

    print("Epoch losses:", history["epoch_loss"])

    # ---- Plot loss curve with Plotly ----
    os.makedirs("plots", exist_ok=True)  #### ensure local folder exists

    epochs_range = range(1, len(history["epoch_loss"]) + 1) 
    plt.figure()                                            
    plt.plot(epochs_range, history["epoch_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss per epoch")

    local_png = "plots/training_loss_curve.png"            
    local_jpg = "plots/training_loss_curve.jpg"

    plt.savefig(local_png)                                  
    plt.savefig(local_jpg)                                   
    plt.close()                                             
    print("[INFO] Saved loss curve →", local_png, "and", local_jpg)

    # 2) Upload to GCS bucket
    bucket_name = "mlops-project-intel-data"
    dest_prefix = "plots"  

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for fname in [local_png, local_jpg]:
            if os.path.exists(fname):
                blob = bucket.blob(f"{dest_prefix}/{os.path.basename(fname)}")
                blob.upload_from_filename(fname)
                print(f"[INFO] Uploaded {fname} → gs://{bucket_name}/{dest_prefix}/{os.path.basename(fname)}")
            else:
                print(f"[WARN] File not found, cannot upload: {fname}")
    except Exception as e:
        print("[WARN] Could not upload plots to GCS:", e)

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

    weights_path = "saved_model/model_weights.pth" 
    ts_path = "saved_model/model_scripted.pt"          

    torch.save(model.state_dict(), weights_path)
    print(f"[INFO] Saved model weights → {weights_path}")

    scripted = torch.jit.script(model.cpu())
    scripted.save(ts_path)
    print(f"[INFO] Saved TorchScript model → {ts_path}")
    
    # ------------------------------------------------------------------
    #                 UPLOAD MODEL TO GCS FOR DEPLOYMENT
    # ------------------------------------------------------------------
    model_prefix = "models/intel_squeezenet"
    mar_prefix = "models/intel_squeezenet/torchserve"
    mar_name = "model.mar" 


    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # upload model files
        blob = bucket.blob(f"{model_prefix}/model_scripted.pt")
        blob.upload_from_filename(ts_path)
        print(f"[GCS] Uploaded TorchScript → gs://{bucket_name}/{model_prefix}/model_scripted.pt")

        blob = bucket.blob(f"{model_prefix}/model_weights.pth")
        blob.upload_from_filename(weights_path)
        print(f"[GCS] Uploaded weights → gs://{bucket_name}/{model_prefix}/model_weights.pth")
        
       # #### CREATED TORCHSERVE .mar ARCHIVE AND UPLOAD TO GCS ####

        mar_dir = "model_store"

        mar_local_path = os.path.join(mar_dir, mar_name)

        cmd = [
            "torch-model-archiver",
            "--model-name", "intel_squeezenet",
            "--version", "1.0",
            "--serialized-file", ts_path,              # uses our TorchScript
            "--handler", "squeezenet_handler.py",      # your custom handler
            "--export-path", mar_dir,
            "--force",
        ]

        print("[INFO] Creating TorchServe MAR archive ...")
        subprocess.run(cmd, check=True)
        print(f"[INFO] Created TorchServe archive → {mar_local_path}")

        # Upload the .mar file to GCS so Vertex can use it
        blob = bucket.blob(f"{mar_prefix}/{mar_name}")
        blob.upload_from_filename(mar_local_path)
        print(f"[GCS] Uploaded MAR → gs://{bucket_name}/{mar_prefix}/{mar_name}")


    except Exception as e:
        print("[GCS ERROR]", e)


if __name__ == "__main__":
    main()