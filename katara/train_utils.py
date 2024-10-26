import torch
import gc
import wandb
from torch import GradScaler
from safetensors.torch import save_model
from tqdm.auto import tqdm


def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {p_count} params")
    return p_count


def clearmem():
    torch.cuda.empty_cache()
    gc.collect()


def wandb_logger(key: str, model, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)
    wandb.watch(model)


def save_checkpoint(model, file_name, epoch=None):
    save_model(model, f"{file_name}.safetensors")
    torch.save(model.state_dict(), f"{file_name}.pth")


# basic training loop
def trainer(model, train_loader, epochs, config, optimizer):
    scaler = GradScaler(device="cuda")
    device = config.device

    model.train()

    train_loss = 0.0

    for epoch in tqdm(range(epochs)):
        clearmem()
        optimizer.zero_grad()

        print(f"Training epoch {epoch+1}")

        for x, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(device)
            label = label.to(device)

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(image)

                train_loss = func_nn.cross_entropy(output, label.long())
                print(f"step {x}: loss {train_loss.item():.4f}")

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

    safe_tensorfile = save_model(model, config.safetensor_file)
