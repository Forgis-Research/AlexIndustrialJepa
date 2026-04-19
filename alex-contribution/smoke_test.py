"""Tiny PyTorch + W&B smoke test.

Goal: prove GPU + logging + env vars all work end-to-end on the SageMaker VM.
"""
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
assert device.type == "cuda", "No GPU — check instance type is ml.g5.xlarge"

wandb.init(
    project="alex-industrial-jepa",
    name="smoke-test-001",
    config={"lr": 1e-2, "epochs": 50, "hidden": 32},
)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

X = torch.randn(512, 10, device=device)
y = X.sum(dim=1, keepdim=True)

for epoch in range(50):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    wandb.log({"loss": loss.item(), "epoch": epoch})
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

wandb.finish()
print("Done.")
