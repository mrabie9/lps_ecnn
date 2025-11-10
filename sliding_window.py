import numpy as np
from pathlib import Path

def augment_adjacent_halves_sorted_npz(in_path, out_path=None, axis=-1):
    """
    1) Sort (xtr,ytr) and (xte,yte) by label (scalar or one-hot -> argmax).
    2) For each consecutive pair with same label, create new = tail512(x[i]) + head512(x[i+1]) along `axis`.
    3) Append new samples; save to <stem>_sorted_adj512cat.npz by default.
    Assumes length 1024 along `axis`.
    """
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_sorted_adj512cat.npz")
    else:
        out_path = Path(out_path)

    data = np.load(in_path, allow_pickle=False)
    for k in ("xtr","ytr","xte","yte"):
        if k not in data.files:
            raise KeyError(f"Missing key '{k}' in {in_path}")

    def label_keys(y):
        y = np.asarray(y)
        if y.ndim == 1: return y
        if y.ndim >= 2 and y.shape[-1] > 1: return np.argmax(y, axis=-1)
        return y.squeeze(-1)

    def sort_by_label(X, Y):
        keys = label_keys(Y)
        order = np.argsort(keys, kind="stable")
        return X[order], Y[order], keys[order]

    def augment_split(X, Y):
        X = np.asarray(X); Y = np.asarray(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must share first dimension.")
        Xs, Ys, keys = sort_by_label(X, Y)

        ax = axis if axis >= 0 else Xs.ndim + axis
        L = Xs.shape[ax]
        if L != 1024:
            raise ValueError(f"Expected length 1024 along axis {ax}, got {L}.")

        new_X, new_Y = [], []

        for i in range(Xs.shape[0] - 1):
            if keys[i] != keys[i+1]:
                continue

            # Build tail/head while KEEPING the sample axis (i:i+1)
            tail_idx = [slice(None)] * Xs.ndim
            tail_idx[0] = slice(i, i+1)
            tail_idx[ax] = slice(512, 1024)
            tail_keep = Xs[tuple(tail_idx)]              # shape: (1, ..., 512)

            head_idx = [slice(None)] * Xs.ndim
            head_idx[0] = slice(i+1, i+2)
            head_idx[ax] = slice(0, 512)
            head_keep = Xs[tuple(head_idx)]              # shape: (1, ..., 512)

            new_keep = np.concatenate([tail_keep, head_keep], axis=ax)  # (1, ..., 1024)
            new = np.squeeze(new_keep, axis=0)            # drop the kept sample axis -> (..., 1024)

            new_X.append(new)
            new_Y.append(Ys[i])

        if new_X:
            new_X = np.stack(new_X, axis=0)               # (M, ..., 1024)
            new_Y = (np.stack(new_Y, axis=0) if np.ndim(Ys[0]) > 0
                     else np.asarray(new_Y))
            X_out = np.concatenate([Xs, new_X], axis=0)
            Y_out = np.concatenate([Ys, new_Y], axis=0)
        else:
            X_out, Y_out = Xs, Ys

        return X_out, Y_out

    xtr_aug, ytr_aug = augment_split(data["xtr"], data["ytr"])
    xte_aug, yte_aug = augment_split(data["xte"], data["yte"])

    np.savez_compressed(out_path, xtr=xtr_aug, ytr=ytr_aug, xte=xte_aug, yte=yte_aug)
    return xtr_aug, ytr_aug, xte_aug, yte_aug


# Example:
xtr2, ytr2, xte2, yte2 = augment_adjacent_halves_sorted_npz("radar_dataset.npz")

# radnist_npz = "radar/tasks-mixed/task0/radar_dataset.npz"
# radchar_npz = "radar/tasks-mixed/task1/radar_dataset.npz"

# radchar_npz = np.load(radchar_npz)
print(xtr2.shape)
# radnist_npz = np.load(radnist_npz)