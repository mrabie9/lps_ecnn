import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configure these ---
models = {
    "ELC": "logs/radar/exp-radar-elc",   # e.g., "logs/radar/exp-radar-mixed-4nodyn-modelA"
    "BLC": "../LPSforECNN/logs/radar/radar-bresnet-tasks",
    "LPS": "logs/radar/exp-radar-mixed-4nodyn",
}
task = "task1"   # set your task name (prefix used in filenames like "task0-train.npz")
# -----------------------

def load_train_curves(folder, task_prefix):
    """Load training-phase losses and val acc from {task}-train.npz."""
    path = os.path.join(folder, f"{task_prefix}-train.npz")
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None, None
    data = np.load(path)
    losses = np.asarray(data["losses"]).squeeze()
    acc = np.asarray(data["epoch_acc"]).squeeze()
    if losses.ndim > 1:
        losses = losses.mean(axis=tuple(range(1, losses.ndim)))
    if acc.ndim > 1:
        acc = acc.mean(axis=tuple(range(1, acc.ndim)))
    n = min(len(losses), len(acc))
    return losses[:n], acc[:n]

curves = {}
for name, folder in models.items():
    L, A = load_train_curves(folder, task)
    if L is not None and A is not None:
        curves[name] = (L, A)

if not curves:
    raise RuntimeError("No model curves loaded. Check paths and task name.")

# Epoch axis per model (supports unequal lengths)
fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
ax2 = ax1.twinx()

loss_handles, acc_handles = [], []

for name, (losses, acc) in curves.items():
    eL = np.arange(1, len(losses) + 1)
    eA = np.arange(1, len(acc) + 1)
    # Loss on left y-axis (solid), Acc on right y-axis (dashed)
    h1, = ax1.plot(eL, losses, label=f"{name} — Loss", linestyle='-')
    h2, = ax2.plot(eA, acc, label=f"{name} — Val Acc", linestyle='--')
    loss_handles.append(h1)
    acc_handles.append(h2)

ax1.set_title(f"Training Loss (left) & Validation Accuracy (right) — {task}")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Val Accuracy")

ax1.grid(True, alpha=0.3)

# Combined legend
handles = loss_handles + acc_handles
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc="right", fontsize=9, ncols=2)

plt.tight_layout()
plt.show()
