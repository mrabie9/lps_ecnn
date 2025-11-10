import numpy as np
import torch
from TrainValTest import accuracy

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def maxprob_uncertainty(p):
    # Lower is better; convert to "uncertainty" where higher means worse:
    return 1.0 - p.max(axis=1)

def selective_metrics(y_true, y_pred, u, tau):
    accept = u <= tau
    cov = accept.mean()
    if cov == 0:
        return {"coverage": 0.0, "sel_acc": np.nan, "sel_risk": np.nan}
    sel_acc = (y_pred[accept] == y_true[accept]).mean()
    sel_risk = 1.0 - sel_acc
    return {"coverage": cov, "sel_acc": sel_acc, "sel_risk": sel_risk}

def sweep_thresholds(y_true, y_proba, u=None, taus=None, utility=None):
    """
    y_true: (N,) integer labels
    y_proba: (N,C) probabilities
    u: (N,) uncertainty score (higher = more uncertain). If None, uses entropy.
    taus: thresholds to try. If None, uses quantiles of u.
    utility: dict with costs/rewards, e.g. {"correct": +1, "error": -5, "abstain": -0.5}
    """
    y_true = np.asarray(y_true)
    y_pred = y_proba#.argmax(axis=1)
    if u is None:
        u = entropy(y_proba)
    if taus is None:
        taus = np.quantile(u, np.linspace(0, 1, 1001))

    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)

    coverages, sel_accs, sel_risks, utils = [], [], [], []
    for t in taus:
        accept = u <= t
        cov = accept.float().mean().item() if isinstance(accept, torch.Tensor) else accept.mean()
        if cov > 0:
            sel_acc = accuracy(y_true[accept], y_pred[accept])
            sel_risk = 1.0 - sel_acc
        else:
            sel_acc = np.nan
            sel_risk = np.nan

        coverages.append(cov); sel_accs.append(sel_acc); sel_risks.append(sel_risk)

        if utility is not None:
            correct = (accept & (y_pred == y_true)).sum()
            error   = (accept & (y_pred != y_true)).sum()
            abstain = (~accept).sum()
            total = len(y_true)
            util = (utility["correct"]*correct +
                    utility["error"]*error +
                    utility["abstain"]*abstain) / total
            utils.append(util)

    out = {
        "tau": np.array(taus),
        "coverage": np.array(coverages),
        "sel_acc": np.array(sel_accs),
        "sel_risk": np.array(sel_risks),
    }
    if utility is not None:
        out["utility"] = np.array(utils)
    return out

# def choose_tau(results, target_coverage=None, max_risk=None, maximize_utility=False): 
#     cov, risk = results["coverage"], results["sel_risk"] 
#     if cov.size == 0: 
#         return {k: np.nan for k in results} 
#     idx = None 
#     if maximize_utility and "utility" in results: 
#         idx = np.nanargmax(results["utility"]) 
#     elif target_coverage is not None: 
#         # Smallest risk among τ with coverage ≥ target 
#         mask = cov >= target_coverage 
#         if mask.any(): 
#             idx = np.nanargmin(np.where(mask, risk, np.inf)) 
#         elif max_risk is not None: # Largest coverage among τ with risk ≤ max_risk 
#             mask = risk <= max_risk 
#         if mask.any(): 
#             idx = np.nanargmax(np.where(mask, cov, -np.inf)) 
#         else: # Fallback: maximize accuracy*coverage (balanced) 
#             score = (1.0 - risk) * cov 
#             idx = np.nanargmax(score) 
#     return {k: v[idx] for k, v in results.items()}

def choose_tau(results, target_coverage=None, max_risk=None, maximize_utility=False):
    cov, risk = results["coverage"], results["sel_risk"]
    if cov.size == 0:
        return {k: np.nan for k in results}

    idx = None

    if maximize_utility and "utility" in results:
        idx = np.nanargmax(results["utility"])

    elif target_coverage is not None:
        # Match a specific coverage (float-safe). If none match, choose closest.
        close = np.isclose(cov, target_coverage, rtol=1e-5, atol=1e-8)

        if np.any(close):
            # Among exact matches, pick the one with smallest risk
            candidates = np.where(close)[0]
            idx = candidates[np.nanargmin(risk[candidates])]
        else:
            # No exact match: choose coverage closest to target; break ties by lowest risk
            diffs = np.abs(cov - target_coverage)
            min_diff = np.nanmin(diffs)
            candidates = np.where(np.isclose(diffs, min_diff, rtol=0, atol=1e-12))[0]
            idx = candidates[np.nanargmin(risk[candidates])]

    elif max_risk is not None:
        # Keep this branch unchanged: largest coverage with risk ≤ max_risk
        mask = risk <= max_risk
        if mask.any():
            idx = np.nanargmax(np.where(mask, cov, -np.inf))

    if idx is None:
        # Fallback: maximize accuracy*coverage (balanced)
        score = (1.0 - risk) * cov
        idx = np.nanargmax(score)

    return {k: v[idx] for k, v in results.items()}

# ---------- Set-valued rules (DST) ----------

def betp_from_singleton_ignorance(m_single, m_omega):
    # m_single: (N,C) singleton masses; m_omega: (N,) ignorance
    N, C = m_single.shape
    return m_single + (m_omega[:, None] / C)

def set_from_betp(betp_row, alpha=0.1, k=None):
    """Smallest set with cumulative BetP >= 1 - alpha. Optionally cap size at k."""
    order = np.argsort(-betp_row)
    cum = np.cumsum(betp_row[order])
    r = np.searchsorted(cum, 1 - alpha) + 1
    S = order[:r]
    if k is not None and len(S) > k:
        S = order[:k]
    return S

def set_from_plausibility(m_single_row, m_omega_val, tau_pl=0.8, k=None):
    """Plausibility cutoff: Pl(y)=m({y})+m(Ω) >= tau_pl, with optional top-k."""
    Pl = m_single_row + m_omega_val
    S = np.flatnonzero(Pl >= tau_pl)
    if k is not None and len(S) > k:
        S = S[np.argsort(-Pl[S])[:k]]
    return S

# ---------- Selective prediction + set-valued fallback ----------

def selective_with_set_fallback(
    y_true,               # (N,) ints
    y_proba,              # (N,C) probabilities for accuracy on accepted
    u,                    # (N,) uncertainty (higher = worse)
    tau,                  # scalar uncertainty threshold
    m_single=None,        # (N,C) DST singleton masses (required for set-valued)
    m_omega=None,         # (N,) DST ignorance masses
    rule="betp",          # 'betp' or 'plausibility'
    alpha=0.1,            # for betp rule (1 - coverage mass)
    tau_pl=0.8,           # for plausibility rule
    k=2                   # cap for set size (practical)
):
    """
    Returns:
      accepted_mask: (N,) bool
      y_hat_accepted: predictions for accepted (ints)
      sets_rejected: list of arrays of class indices for each rejected sample
      metrics: dict with coverage, selective_acc, avg_set_size_rej, contains_true_rate_rej, etc.
    """
    y_true = np.asarray(y_true)
    y_pred = y_proba# np.argmax(y_proba, axis=1)
    accepted = u <= tau

    # Metrics on accepted
    cov = accepted.float().mean().item() 
    sel_acc = np.nan
    if cov > 0:
        sel_acc = (y_pred[accepted] == y_true[accepted]).mean()

    # Build sets for rejected
    rejected_idx = np.flatnonzero(~accepted)
    sets_rejected = [None] * len(rejected_idx)

    if len(rejected_idx) > 0:
        if rule == "betp":
            if m_single is None or m_omega is None:
                raise ValueError("betp rule requires m_single and m_omega.")
            BetP = betp_from_singleton_ignorance(m_single, m_omega)
            for j, i in enumerate(rejected_idx):
                sets_rejected[j] = set_from_betp(BetP[i], alpha=alpha, k=k)

        elif rule == "plausibility":
            if m_single is None or m_omega is None:
                raise ValueError("plausibility rule requires m_single and m_omega.")
            for j, i in enumerate(rejected_idx):
                sets_rejected[j] = set_from_plausibility(m_single[i], m_omega[i], tau_pl=tau_pl, k=k)
        else:
            raise ValueError("Unknown rule. Use 'betp' or 'plausibility'.")

    # Rejected-set metrics
    set_sizes = np.array([len(S) for S in sets_rejected]) if len(rejected_idx) else np.array([])
    avg_set_size = float(set_sizes.mean()) if set_sizes.size else np.nan
    contains_true = np.array([y_true[idx] in sets_rejected[j] for j, idx in enumerate(rejected_idx)]) if len(rejected_idx) else np.array([])
    contains_true_rate = float(contains_true.mean()) if contains_true.size else np.nan

    metrics = {
        "coverage": cov,
        "selective_accuracy": sel_acc,
        "n_rejected": int((~accepted).sum()),
        "avg_set_size_rejected": avg_set_size,
        "contains_true_rate_rejected": contains_true_rate,
    }

    return accepted, y_pred[accepted], sets_rejected, metrics

# ---------- Minimal example (comment out in library use) ----------
# N, C = 1000, 5
# rng = np.random.default_rng(0)
# y_true = rng.integers(0, C, size=N)
# logits = rng.normal(0, 1, size=(N, C)); logits[np.arange(N), y_true] += 1.0
# y_proba = np.exp(logits - logits.max(axis=1, keepdims=True)); y_proba /= y_proba.sum(axis=1, keepdims=True)
# u = -(y_proba * np.log(y_proba + 1e-12)).sum(axis=1)  # entropy as uncertainty
# # Fake DST masses consistent with probabilities (for illustration only)
# m_omega = np.clip(0.2 + 0.6 * (u - u.min()) / (u.ptp() + 1e-9), 0, 1)
# m_single = (1 - m_omega)[:, None] * y_proba
# accepted, y_hat_acc, sets_rej, metrics = selective_with_set_fallback(
#     y_true, y_proba, u, tau=np.quantile(u, 0.8),
#     m_single=m_single, m_omega=m_omega, rule="betp", alpha=0.1, k=2
# )
# print(metrics)
