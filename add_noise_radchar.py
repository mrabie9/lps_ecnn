import numpy as np
from sklearn.model_selection import train_test_split
# "/home/lunet/wsmr11/repos/radar/snr_splits_radnist", "/home/lunet/wsmr11/repos/radar/snr_splits_radchar"
radnist_npz = "radar/tasks-mixed/task0/radar_dataset.npz"
radchar_npz = "radar_dataset.npz"

radchar_npz = np.load(radchar_npz)
radnist_npz = np.load(radnist_npz)

radchar_xtr = radchar_npz['xtr']
radchar_xte = radchar_npz['xte']
radchar_ytr = radchar_npz['ytr']
radchar_yte = radchar_npz['yte']

radnist_xtr = radnist_npz['xtr']
radnist_xte = radnist_npz['xte']
radnist_ytr = radnist_npz['ytr']
radnist_yte = radnist_npz['yte']

print("Radchar", radchar_xtr.shape)
print("RadNIST", radnist_xtr.shape)

## Mask for class 0 radnist

noise_mask_tr = radnist_ytr==0
noise_mask_te = radnist_yte==0

noise_tr = radnist_xtr[noise_mask_tr]
noise_te = radnist_xte[noise_mask_te]
noise_tr_y = radnist_ytr[noise_mask_tr]
noise_te_y = radnist_yte[noise_mask_te]

noise_x = np.concat([noise_tr, noise_te])
noise_y = np.concat([noise_tr_y, noise_te_y])
noise_xtr, noise_xte, noise_ytr, noise_yte = train_test_split(noise_x, noise_y, test_size=0.3, random_state=42, stratify=noise_y)

noise_r = noise_tr.shape[0]//(radnist_xtr.shape[0]-noise_tr.shape[0])
print("noise_r", noise_tr.shape[0], radnist_xtr.shape[0], noise_r)

radchar_xtr, _, radchar_ytr, _ = train_test_split(radchar_xtr, radchar_ytr, test_size=0.18, random_state=42, stratify=radchar_ytr)
radchar_xte, _, radchar_yte, _ = train_test_split(radchar_xte, radchar_yte, test_size=0.18, random_state=42, stratify=radchar_yte)
print(radchar_xtr.shape[0], noise_xtr.shape[0])
print(radchar_xte.shape[0], noise_xte.shape[0])
print(np.unique_counts(radchar_ytr))

radchar_xtr = np.concatenate([radchar_xtr, noise_xtr])
radchar_ytr = np.concatenate([radchar_ytr, noise_ytr])
radchar_xte = np.concatenate([radchar_xte, noise_xte])
radchar_yte = np.concatenate([radchar_yte, noise_yte])

print(radchar_xtr.shape)
print(np.unique_counts(radchar_ytr))
print(radchar_xte.shape)
print(np.unique_counts(radchar_yte))

np.savez("radar_dataset.npz", xtr = radchar_xtr, ytr = radchar_ytr, xte=radchar_xte, yte=radchar_yte)