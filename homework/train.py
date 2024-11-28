import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.optim as optim
from .models import MLPPlanner, load_model, save_model, TransformerPlanner
from homework.datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 70,
    lr: float = 1e-6,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend")
    else:
        print("MPS and CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    if(model_name == "MLPPlanner"):
        model = MLPPlanner()
    elif(model_name == "TransformerPlanner"):
        model = TransformerPlanner()
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)  # Define optimizer

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    for epoch in range(20):
        for key in metrics:
            metrics[key].clear()

        model.train()

        for sample in train_data:
            img = sample["image"].to(device)
            track_right = sample["track_right"].to(device)
            track_left = sample["track_left"].to(device)
            waypoints = sample["waypoints"].to(device)
            waypoints_mask = sample["waypoints_mask"].to(device)

            predicted_waypoints = model(track_left, track_right)

            # Apply the mask to the predicted waypoints and the target waypoints
            masked_predicted_waypoints = predicted_waypoints * waypoints_mask
            masked_waypoints = waypoints * waypoints_mask

            # Compute the L1 loss with the masked waypoints
            loss = torch.nn.functional.l1_loss(masked_predicted_waypoints, masked_waypoints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["train_acc"].append(loss.item())

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for sample in val_data:
                img = sample["image"].to(device)
                track_right = sample["track_right"].to(device)
                track_left = sample["track_left"].to(device)
                waypoints = sample["waypoints"].to(device)
                waypoints_mask = sample["waypoints_mask"].to(device)

                predicted_waypoints = model(track_left, track_right)

                # Apply the mask to the predicted waypoints and the target waypoints
                masked_predicted_waypoints = predicted_waypoints * waypoints_mask
                masked_waypoints = waypoints * waypoints_mask

                # Compute the L1 loss with the masked waypoints
                val_loss = torch.nn.functional.l1_loss(masked_predicted_waypoints, masked_waypoints)
                metrics["val_acc"].append(val_loss)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        logger.add_scalar('Accuracy/Val', epoch_val_acc, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))