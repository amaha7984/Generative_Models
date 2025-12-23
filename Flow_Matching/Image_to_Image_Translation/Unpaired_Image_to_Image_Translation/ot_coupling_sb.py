# ot_coupling_sb.py
# Supports Schrödinger bridge CFM
import torch


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

    logK = -cost / eps  # (B,B)

    log_u = torch.zeros(B, device=device, dtype=dtype)
    log_v = torch.zeros(B, device=device, dtype=dtype)

    # uniform marginals: log(1/B)
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
    P = torch.exp(logP)
    return P


@torch.no_grad()
def compute_ot_plan(cost, method="exact", eps=0.05, iters=50, num_threads=1, warn=True):
    """
    cost: (B,B) nonnegative
    returns P: (B,B) OT plan with uniform marginals

    - exact: POT emd (CPU) -> plan on GPU
    - sinkhorn: entropic OT (GPU) using log-domain Sinkhorn
    """
    if method == "sinkhorn":
        P = sinkhorn_log_domain(cost, eps=eps, iters=iters)
    elif method == "exact":
        import ot as pot 

        B = cost.shape[0]
        a = pot.unif(B)
        b = pot.unif(B)
        C = cost.detach().cpu().numpy()
        P = pot.emd(a, b, C, numThreads=num_threads)
        P = torch.tensor(P, device=cost.device, dtype=cost.dtype)
    else:
        raise ValueError(f"Unknown OT method: {method}")

    if not torch.isfinite(P).all() or P.abs().sum().item() < 1e-8:
        if warn:
            print("[WARN] Numerical issues in OT plan; reverting to uniform plan.")
        P = torch.ones_like(P) / P.numel()

    return P


@torch.no_grad()
def minibatch_ot_sample_plan(
    x0,
    x1,
    ot_method="exact",
    eps=0.05,
    iters=50,
    replace=True,
    num_threads=1,
    normalize_cost=False,
    warn=True,
):
    """
    x0, x1: (B,C,H,W) or (B,D) or (B,*) tensors
    Returns:
      x0_pi, x1_pi: both (B,...) sampled from the OT plan pi over the minibatch.

    Matches torchcfm behavior:
      - flatten if dim>2
      - squared Euclidean cost
      - uniform marginals
      - sample indices from flattened plan
    """
    B = x0.shape[0]

    x0f = x0.reshape(B, -1) if x0.dim() > 2 else x0
    x1f = x1.reshape(B, -1) if x1.dim() > 2 else x1

    # Squared Euclidean cost (torchcfm uses torch.cdist(...)**2)
    cost = torch.cdist(x0f, x1f) ** 2  # (B,B)

    if normalize_cost:
        cost = cost / (cost.max() + 1e-12)

    P = compute_ot_plan(cost, method=ot_method, eps=eps, iters=iters, num_threads=num_threads, warn=warn)

    # sample (i,j) from flattened P
    p = (P.reshape(-1) + 1e-12)
    p = p / p.sum()
    idx = torch.multinomial(p, num_samples=B, replacement=replace)
    i = idx // B
    j = idx % B

    return x0[i], x1[j]
