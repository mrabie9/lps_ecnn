from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # optional, for prettier heatmaps
import torch
from models.masknet import ResNet50_1d, ResNet18_1d
from TrainValTest import CVTrainValTest
import pickle
import numpy as np
import pandas as pd
import os
from TrainValTest import accuracy

from plots import plot_hexbin, plot_kde2d, plot_multiclass_roc_auc, get_auc, plot_acc_cov, plot_risk_cov
from sweep_eu import sweep_thresholds, choose_tau, selective_with_set_fallback

import warnings
warnings.filterwarnings("ignore")

def to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value

def save_test_outputs(output_dir, all_labels, all_preds, unique_labels, eu, h_pred, f_eu, inv_utils,filename="test_outputs.pkl"):
    """Persist the key tensors/lists returned by test_model so they can be reused later."""
    os.makedirs(output_dir, exist_ok=True)

    def _to_serializable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value
# mega, pred_class, util, inv_util
    payload = {
        "all_labels": _to_serializable(all_labels),
        "all_preds": _to_serializable(all_preds),
        "unique_labels": _to_serializable(unique_labels),
        "omega": _to_serializable(omega),
        "pred_class": _to_serializable(pred_class),
        "util": _to_serializable(util),
        "inv_utils": _to_serializable(inv_utils),
    }

    with open(os.path.join(output_dir, filename), "wb") as handle:
        pickle.dump(payload, handle)

    return os.path.join(output_dir, filename)

def load_test_outputs(file_path):
    """Restore the cached outputs produced by save_test_outputs."""
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(f"No saved outputs found at {file_path}")

    with open(file_path, "rb") as handle:
        payload = pickle.load(handle)

    return (
        payload["all_labels"],
        payload["all_preds"],
        payload["unique_labels"],
        payload["omega"],
        payload["pred_class"],
        payload["util"],
        payload["inv_utils"],
    )

class args_class:
    def __init__(self, base_path=None):
        self.base_path = base_path

file = "task0/cumu_model.pt"
task = "task0"
offset = 0
dataset = "Radar"
save_figs = True
datasets = {
    "DRC" : {"models" : "dronerc/exp-drc-1024sl-5/", "tasks" : "dronerc/tasks-1024sl/", "num_classes":17},
    "USRP"    : {"models" : "usrp/exp-usrp-DS-highlr-2/", "tasks" : "usrp/tasks - 1t-1024slices-norm-3tasks/", "num_classes":18},
    "LoRa"    : {"models" : "rfmls/exp-lora-2/", "tasks" : "rfmls/tasks_lp_downsampled/", "num_classes":10},
    "mixed-comms"    : {"models" : "mixed/exp-mixed-sm/", "tasks" : "mixed/tasks-sm/", "num_classes":15, "offset":[0]},
    "Radar"    : {"models" : "radar/exp-radar-mixed-3-nodyn/", "tasks" : "radar/tasks-mixed/", "num_classes":11, "offset": [0,6,11]}
}
args = args_class("")
args.multi_head = False
# args.base_path = "radar"
# args.current_task = 0
model_path = datasets[dataset]['models']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_configs = [
    {"file": "task0/cumu_model.pt", "task": "task0", "offset": 0},
    {"file": "task1/cumu_model.pt", "task": "task1", "offset": 5},
    # {"file": "task2/rfnet16.pt", "task": "task2", "offset": 11},
]

dataset_paths = ["/home/lunet/wsmr11/repos/radar/snr_grouped_processed", "/home/lunet/wsmr11/repos/radar/snr_splits"]
snr_range = list(range(-20, 20, 2))

fig, axs = plt.subplots(2, 2, dpi=200, figsize=(12, 10))  # 3 tasks × (CM + uncertainty)
plt.figure(figsize=(10, 6), dpi=200)
j=0
colors = ['#1f77b4', 'maroon']
labels = ["RadNIST", "RadChar"]
for i, config in enumerate(task_configs):
    args.current_task = i
    file = config["file"]
    task = config["task"]
    offset = datasets[dataset]["offset"][i]
    
    # Load model and mask
    model = ResNet18_1d(slice_size=1024, num_classes=datasets[dataset]['num_classes'])#, classes_per_task=[2,3,3])
    model.load_state_dict(torch.load(model_path + file))
    
    if os.path.exists(datasets[dataset]['models'] + f"{task}/cumu_mask.pkl"):
        trained_mask = pickle.load(open(datasets[dataset]['models'] + f"{task}/cumu_mask.pkl", 'rb'))
    else:
        trained_mask = None
    trained_mask = None
    
    model.eval()
    model.to(device)

    # Load data
    base_path = save_path = datasets[dataset]['tasks'] + task
    pipeline = CVTrainValTest(base_path=base_path, save_path=save_path)
    train_loader = pipeline.load_data_dronerc(256, offset=offset)
    
    eval_snr = True
    if eval_snr:
        folder = dataset_paths[i]
        accs, uncerts, results, aucs, covs = [], [], [], [], []
        n_rejected = 0
        rejected_top2_acc = []
        for snr in snr_range:
            filename = f"{snr}db.npz" if snr < 0 else f"{snr}db.npz"
            path = os.path.join(folder, filename)
            
            # data = np.load(path)
            # x = data['xte']
            # y = data['yte']
            print(datasets[dataset]['models'])
            save_file = datasets[dataset]['models'] + f"{task}/snr{snr}_outputs.pkl"
            train_loader = pipeline.load_data_dronerc(256, offset=offset, data = path, mixed_snrs=True)
            # loader = pipeline.convert_to_loader(x, y, batch_size=256)
            if False: #os.path.exists(save_file):
                all_labels, all_preds, unique_labels, omega, pred_class, util, inv_util = load_test_outputs(save_file)
                all_labels = np.asarray(all_labels) - offset
                all_preds = np.asarray(all_preds) - offset
            else:
                all_labels, all_preds, unique_labels, omega, pred_class, util = pipeline.test_model(
                args, model, trained_mask, cm=True, enable_diagnostics=False, mixed_snrs=True
                )
                all_labels = np.asarray(all_labels) - offset
                all_preds = np.asarray(all_preds) - offset
                correct = (np.array(all_preds) == np.array(all_labels))
                util_max, _ = torch.max(util, dim=1)
                inv_util = (1 - util_max) * 100
                # save_test_outputs(os.path.join(model_path, task), all_labels, all_preds, unique_labels, omega, pred_class, util, inv_util, filename=f"snr{snr}_outputs_95c.pkl")
            
            # plot_multiclass_roc_auc(all_labels, 
            #                         np.eye(len(unique_labels))[all_preds],
            #                         class_names=[str(lbl) for lbl in unique_labels],
            #                         title=f"ROC AUC - {dataset} - {task}"
            #                     )
            m_single = util
            m_omega  = omega
            m_single_norm = m_single / (1 - m_omega.unsqueeze(1) + 1e-8)
            aleatoric = -(m_single_norm * (m_single_norm + 1e-8).log()).sum(dim=1)

            sweep_results = sweep_thresholds(all_labels, all_preds, u=list(aleatoric+inv_util))
            best = choose_tau(sweep_results, target_coverage=0.95)  # accept ~80% most-certain

            accepted, y_hat_acc, sets_rej, metrics = selective_with_set_fallback(
                all_labels, all_preds, inv_util, tau=best["tau"],
                m_single=util, m_omega=omega, rule="betp", alpha=0.1, k=2
            )

            accept_mask = inv_util <= best["tau"]

            all_labels_accepted = np.array(all_labels)[accept_mask]
            all_preds_accepted = np.array(all_preds)[accept_mask]
            inv_util_accepted = np.array(inv_util)[accept_mask]

            print("Chosen τ:", best["tau"])
            print("Coverage:", best["coverage"])
            print("Selective accuracy:", best["sel_acc"])
            print("Selective risk:", best["sel_risk"])
            print("Mean uncertainty: ", np.mean(inv_util_accepted))
            print("top2_accuracy_rejected", metrics['contains_true_rate_rejected'])

            
            auc = get_auc(all_labels_accepted, 
                                    np.eye(len(unique_labels))[all_preds_accepted],
                                    class_names=[str(lbl) for lbl in unique_labels])
            # # continue
            # plot_multiclass_roc_auc(all_labels_accepted, 
            #                         np.eye(len(unique_labels))[all_preds_accepted],
            #                         class_names=[str(lbl) for lbl in unique_labels],
            #                         title=f"ROC AUC - {dataset} - {task}"
            #                     )
            accs.append(best["sel_acc"]*100)
            aucs.append(auc)
            uncerts.append(inv_util_accepted)
            covs.append(best['coverage'])
            if metrics['n_rejected'] != 0: rejected_top2_acc.append(metrics['contains_true_rate_rejected'])
            n_rejected += metrics['n_rejected']
                # --- Evaluate ---

        results.append((accs, uncerts))
        print("========= Stats =========")
        print("Acc", np.mean(accs))
        print("Covs", np.mean(covs))
        print("Top2 Acc", np.mean(rejected_top2_acc))
        print("n_rejected", n_rejected)

        for i, (accs, uncerts) in enumerate(results):
            plt.plot(snr_range, accs, label=f"{labels[j]} Accuracy", color=colors[j])
            # plt.plot(snr_range, uncerts, linestyle='-.', label=f"{labels[j]} Uncertainty", color=colors[j])
        j+=1
        payload = {
            "results": to_serializable(results),
            "AUC": to_serializable(aucs)
        }
        with open(os.path.join(datasets[dataset]['models'] + f"{task}/snr_results_selective.pkl"), "wb") as handle:
            pickle.dump(payload, handle)
        
    else:
        if False:#os.path.exists(model_path + task + "/test_outputs.pkl"):
            all_labels, all_preds, unique_labels, omega, pred_class, util, inv_util = load_test_outputs(model_path + task + "/test_outputs.pkl")
            all_labels = np.asarray(all_labels) - offset
            all_preds = np.asarray(all_preds) - offset
        else:
            # Evaluate model
            all_labels, all_preds, unique_labels, omega, pred_class, util = pipeline.test_model(
                args, model, trained_mask, cm=True, enable_diagnostics=False
            )
            correct = (np.array(all_preds) == np.array(all_labels))
            util_max, _ = torch.max(util, dim=1)
            inv_util = (1 - util_max) * 100
            save_test_outputs(os.path.join(model_path, task), all_labels, all_preds, unique_labels, omega, pred_class, util, inv_util)
        print(accuracy(all_labels, all_preds))
        eu = inv_util
        results_epi = sweep_thresholds(all_labels, all_preds, u=eu)
        best_epi = choose_tau(results_epi, target_coverage=0.95)
        accept = inv_util <= best_epi['tau']

        print("Chosen τ:", best_epi["tau"])
        print("Coverage:", best_epi["coverage"])
        print("Selective accuracy:", best_epi["sel_acc"])
        print("Selective risk:", best_epi["sel_risk"])
        # print(accept.sum())
        # print(len(all_labels))
        plot_acc_cov([results_epi],
                     labels=["Epistemic"],
                     mark_idx=[np.argmin(np.abs(results_epi["coverage"]-best_epi["coverage"])),
                               None, None])
        # plot_risk_cov([results_epi],
        #               labels=["Epistemic"])
        payload = {
            "results": to_serializable(results_epi),
        }
        print(best_epi['sel_acc'])
        with open(f"eu_comparison/ELC/coverage/{task}coverage_data.pkl", "wb") as handle:
            pickle.dump(payload, handle)
        # plot_multiclass_roc_auc(all_labels, 
        #                             np.eye(len(unique_labels))[all_preds],
        #                             class_names=[str(lbl) for lbl in unique_labels],
        #                             title=f"ROC AUC - {dataset} - {task}"
        #                         )
        # results = sweep_thresholds(all_labels, all_preds, u=list(inv_util))
        # best = choose_tau(results, target_coverage=0.95)  # accept ~80% most-certain

        # accept_mask = inv_util <= best["tau"]

        # all_labels_accepted = np.array(all_labels)[accept_mask]
        # all_preds_accepted = np.array(all_preds)[accept_mask]
        # inv_util_accepted = np.array(inv_util)[accept_mask]

        # print("Chosen τ:", best["tau"])
        # print("Coverage:", best["coverage"])
        # print("Selective accuracy:", best["sel_acc"])
        # print("Selective risk:", best["sel_risk"])
        
        # # continue
        # plot_multiclass_roc_auc(all_labels_accepted, 
        #                         np.eye(len(unique_labels))[all_preds_accepted],
        #                         class_names=[str(lbl) for lbl in unique_labels],
        #                         title=f"ROC AUC - {dataset} - {task}")
    
    # plt.legend(["RadNIST Accuracy","RadNIST Uncertainty","RadChar Accuracy","RadChar Uncertainty"])
   
    continue

    # # Evaluate model
    # all_labels, all_preds, unique_labels, omega, pred_class, util = pipeline.test_model(
    #     args, model, trained_mask, cm=True, enable_diagnostics=False
    # )

    # correct = (np.array(all_preds) == np.array(all_labels))
    # util_max, _ = torch.max(util, dim=1)
    # inv_util = (1 - util_max) * 100

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels, normalize="true") * 100
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=np.array(unique_labels),
                yticklabels=np.array(unique_labels),
                cbar=False, ax=axs[i, 0])
    axs[i, 0].set_xlabel("Predicted Label")
    axs[i, 0].set_ylabel("True Label")
    if i == 0:
        axs[i, 0].set_title(f"Confusion Matrices ({dataset})")

    # Uncertainty boxplot
    df = pd.DataFrame({
        "Predicted Label": all_preds,
        "Uncertainty": inv_util,
        "Correctness": np.where(np.array(all_preds) == np.array(all_labels), "Correct", "Incorrect")
    })
    hue_order = ["Correct", "Incorrect"]

    sns.stripplot(x="Predicted Label", y="Uncertainty", hue="Correctness", data=df,
                  color='black', alpha=0.1, size=2, dodge=True, ax=axs[i, 1], legend="")
    axs[i, 1].grid(True)
    axs[i, 1].set_ylabel("Uncertainty")
    if i == 0:
        axs[i, 1].set_title(f"Uncertainty per Prediction ({dataset})")
        sns.boxplot(x="Predicted Label", y="Uncertainty", hue="Correctness", hue_order=hue_order, data=df,
                whis=1.5, width=0.6, palette="Blues", fliersize=0, ax=axs[i, 1])
        # plt.legend(loc="upper right", title=None, prop={'size': 4})
    else:
        sns.boxplot(x="Predicted Label", y="Uncertainty", hue="Correctness", hue_order=hue_order, data=df,
                whis=1.5, width=0.6, palette="Blues", fliersize=0, ax=axs[i, 1], legend="")
    # plt.legend(title=None)
    
    # Clean up duplicate legends from stripplot
    handles, labels = axs[i, 1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[i, 1].legend(by_label.values(), by_label.keys(), title="", prop={'size': 8}, loc="upper left") # loc="upper right"
    
    axs[i, 0].tick_params(axis='x', labelsize=8)  # x-ticks
    axs[i, 0].tick_params(axis='y', labelsize=8)  # y-ticks
    axs[i, 1].tick_params(axis='x', labelsize=8)  # x-ticks
    axs[i, 1].tick_params(axis='y', labelsize=8)  # y-ticks
    # Find the highest top whisker line

    whisker_tops = [
        line.get_ydata()[1] for line in axs[i, 1].lines
        if len(line.get_ydata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]  # vertical line
    ]

    # Set dynamic y-limit slightly above the max whisker
    if whisker_tops:
        max_whisker = max(whisker_tops)
        axs[i, 1].set_ylim(-2, max_whisker * 1.1)  # add 5% headroom

plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
# plt.title("Model Accuracy and Uncertainty vs SNR")
plt.grid(True)
plt.legend(fontsize=7)
plt.tight_layout()
# plt.show()
# os.makedirs(model_path + "figs/", exist_ok=True)
plt.savefig("auc_snr.png", bbox_inches='tight')
# fig.tight_layout()
fig.subplots_adjust(
    left=0.352,   # space from the left edge
    bottom=0.09, # space from the bottom edge
    right=0.9,  # space from the right edge
    top=0.938,    # space from the top edge
    wspace=0.27,  # width spacing between columns
    hspace=0.474   # height spacing between rows
)

# plt.rcParams.update({
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'axes.labelsize': 12,
#     'axes.titlesize': 14
# })

if save_figs:
    os.makedirs(model_path + "figs/", exist_ok=True)
    fig.savefig(datasets[dataset]['models'] + f"figs/cm-pred_uncertainty.png", bbox_inches='tight')
plt.show()
