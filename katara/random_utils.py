import torch
import wandb
import os
import gc
from torch import nn
from torch.amp.grad_scaler import GradScaler
from tqdm.auto import tqdm
from safetensors.torch import save_model
from .train_utils import wandb_logger


# Model parameter count
def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count


def clearmem():
    torch.cuda.empty_cache()
    gc.collect()


# basic training loop
def classifier_training_loop(
    model, train_loader, epochs, config, optimizer, criterion, train_dtype
):
    scaler = GradScaler()
    model.train()
    train_loss = 0.0
    clearmem()

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        print(f"Training epoch {epoch+1}")

        for x, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(config.device)
            label = label.to(config.device)

            # every iteration
            clearmem()

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=train_dtype):
                output = model(image)
                train_loss = criterion(output, label.long())

                wandb.log({"loss": train_loss})
                train_loss = train_loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place

                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

        print(f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}")

        print(f"Epoch @ {epoch} complete!")

    print(f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}")

    return model


def save_checkpoint(model, file_name, epoch=None):
    safe_tensorfile = save_model(model, f"{file_name}.safetensors")
    torch.save(model.state_dict(), f"{file_name}.pth")


# training_loop()
clearmem()
