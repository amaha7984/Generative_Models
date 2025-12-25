# ot_coupling.py
import torch

# -------------------------
# OT solvers
# -------------------------
@torch.no_grad()
def sinkhorn_log_domain(cost, eps=0.05, iters=50):
    """
    cost: (B,B) nonnegative
    returns P: (B,B) approximately doubly-stochastic OT plan
    Uniform marginals assumed.
    """
    B = cost.shape[0]
    device = cost.device
    dtype = cost.dtype

    logK = -cost / eps

    log_u = torch.zeros(B, device=device, dtype=dtype)
    log_v = torch.zeros(B, device=device, dtype=dtype)

    log_a = torch.full(
        (B,),
        -torch.log(torch.tensor(float(B), device=device, dtype=dtype)),
        device=device,
        dtype=dtype,
    )
    log_b = log_a.clone()

    for _ in range(iters):
        log_u = log_a - torch.logsumexp(logK + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(logK.transpose(0, 1) + log_u.unsqueeze(0), dim=1)

    logP = logK + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return torch.exp(logP)  # (B,B)


@torch.no_grad()
def compute_ot_plan(cost, method="exact", eps=0.05, iters=50, num_threads=1):
    """
    cost: (B,B) nonnegative
    returns P: (B,B) OT plan with uniform marginals
      - exact: POT emd (CPU)
      - sinkhorn: entropic OT (GPU)
    """
    if method == "sinkhorn":
        return sinkhorn_log_domain(cost, eps=eps, iters=iters)

    if method == "exact":
        # Exact EMD via POT (CPU)
        import ot as pot  # pip install pot

        B = cost.shape[0]
        a = pot.unif(B)
        b = pot.unif(B)
        C = cost.detach().cpu().numpy()
        P = pot.emd(a, b, C, numThreads=num_threads)
        return torch.tensor(P, device=cost.device, dtype=cost.dtype)

    raise ValueError(f"Unknown OT method: {method}")


# -------------------------
# Cost matrix
# -------------------------
@torch.no_grad()
def _cost_matrix_mse(z0, z1):
    """
    z0, z1: (B,C,H,W) or (B,D)
    cost: (B,B) mean squared L2 distance in flattened space
    """
    B = z0.shape[0]
    z0f = z0.view(B, -1)
    z1f = z1.view(B, -1)
    cost = ((z0f[:, None, :] - z1f[None, :, :]) ** 2).mean(dim=2)
    return cost


# -------------------------
# Pairing strategies
# -------------------------
@torch.no_grad()
def _sample_pairs_from_plan(P, num_pairs, replace=True):
    """
    P: (B,B) nonnegative
    returns i, j indices length num_pairs sampled from flattened plan
    """
    B = P.shape[0]
    p = (P.reshape(-1) + 1e-12)
    p = p / p.sum()
    idx = torch.multinomial(p, num_samples=num_pairs, replacement=replace)
    i = idx // B
    j = idx % B
    return i, j


@torch.no_grad()
def _pairing_ot(cost, ot_method="exact", eps=0.05, iters=50, replace=True, num_threads=1):
    P = compute_ot_plan(cost, method=ot_method, eps=eps, iters=iters, num_threads=num_threads)
    i, j = _sample_pairs_from_plan(P, num_pairs=cost.shape[0], replace=replace)
    return i, j


@torch.no_grad()
def _pairing_mnn(cost, min_mutual_frac=0.25, replace=True):
    """
    Mutual Nearest Neighbor (MNN) selection.
    Returns i=0..B-1 and j indices length B.
    """
    B = cost.shape[0]
    device = cost.device

    i_to_j = torch.argmin(cost, dim=1)  # (B,)
    j_to_i = torch.argmin(cost, dim=0)  # (B,)

    ar = torch.arange(B, device=device)
    mutual_mask = (j_to_i[i_to_j] == ar)
    mutual_i = torch.nonzero(mutual_mask, as_tuple=False).squeeze(1)
    mutual_j = i_to_j[mutual_i]

    j = i_to_j.clone()

    if mutual_i.numel() >= int(B * float(min_mutual_frac)) and mutual_i.numel() > 0:
        non_mutual_i = torch.nonzero(~mutual_mask, as_tuple=False).squeeze(1)
        if non_mutual_i.numel() > 0:
            if replace:
                pick = torch.randint(low=0, high=mutual_j.numel(),
                                     size=(non_mutual_i.numel(),), device=device)
            else:
                perm = torch.randperm(mutual_j.numel(), device=device)
                pick = perm[: min(non_mutual_i.numel(), mutual_j.numel())]
                if pick.numel() < non_mutual_i.numel():
                    extra = torch.randint(low=0, high=mutual_j.numel(),
                                          size=(non_mutual_i.numel() - pick.numel(),), device=device)
                    pick = torch.cat([pick, extra], dim=0)

            j[non_mutual_i] = mutual_j[pick]

    i = torch.arange(B, device=device)
    return i, j


@torch.no_grad()
def _pairing_hungarian(cost):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise RuntimeError(
            "Hungarian pairing requires scipy. Install with: pip install scipy"
        ) from e

    C = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(C)
    i = torch.tensor(row_ind, device=cost.device, dtype=torch.long)
    j = torch.tensor(col_ind, device=cost.device, dtype=torch.long)
    return i, j


@torch.no_grad()
def _pairing_softmax(cost, tau=0.1, replace=True):
    """
    p(j|i) ∝ exp(-C_ij / tau). No marginal constraints.
    """
    B = cost.shape[0]
    logits = -cost / max(1e-8, float(tau))
    probs = torch.softmax(logits, dim=1)  # (B,B)
    j = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)
    i = torch.arange(B, device=cost.device)
    return i, j


# -------------------------
# Unified API
# -------------------------
@torch.no_grad()
def minibatch_pair_indices_from_cost(
    cost: torch.Tensor,
    pairing="ot",                 # {"ot","mnn","hungarian","softmax"}
    ot_method="exact",
    eps=0.05,
    iters=50,
    replace=True,
    num_threads=1,
    mnn_min_mutual_frac=0.25,
    softmax_tau=0.1,
):
    """
    cost: (B,B)
    Returns: i, j indices (both length B).
    """
    if pairing == "ot":
        i, j = _pairing_ot(cost, ot_method=ot_method, eps=eps, iters=iters,
                           replace=replace, num_threads=num_threads)
    elif pairing == "mnn":
        i, j = _pairing_mnn(cost, min_mutual_frac=mnn_min_mutual_frac, replace=replace)
    elif pairing == "hungarian":
        i, j = _pairing_hungarian(cost)
    elif pairing == "softmax":
        i, j = _pairing_softmax(cost, tau=softmax_tau, replace=replace)
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing}")
    return i, j


@torch.no_grad()
def minibatch_pair_sample_plan(
    z0,
    z1,
    pairing="ot",                 # {"ot","mnn","hungarian","softmax"}
    ot_method="exact",
    eps=0.05,
    iters=50,
    replace=True,
    num_threads=1,
    mnn_min_mutual_frac=0.25,
    softmax_tau=0.1,
):
    """
    z0, z1: (B,C,H,W) or (B,D)
    Returns:
      z0_pi, z1_pi: both (B,...) selected by chosen pairing strategy.
    """
    cost = _cost_matrix_mse(z0, z1)
    i, j = minibatch_pair_indices_from_cost(
        cost,
        pairing=pairing,
        ot_method=ot_method,
        eps=eps,
        iters=iters,
        replace=replace,
        num_threads=num_threads,
        mnn_min_mutual_frac=mnn_min_mutual_frac,
        softmax_tau=softmax_tau,
    )
    return z0[i], z1[j]


# Backward-compatible name (OT-only)
@torch.no_grad()
def minibatch_ot_sample_plan(
    z0,
    z1,
    ot_method="exact",
    eps=0.05,
    iters=50,
    replace=True,
    num_threads=1,
):
    return minibatch_pair_sample_plan(
        z0, z1,
        pairing="ot",
        ot_method=ot_method,
        eps=eps,
        iters=iters,
        replace=replace,
        num_threads=num_threads,
    )
