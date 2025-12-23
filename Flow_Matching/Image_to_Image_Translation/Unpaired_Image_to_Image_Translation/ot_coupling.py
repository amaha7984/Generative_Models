# ot_coupling.py
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

    # logK = -C/eps
    logK = -cost / eps

    # log u, log v for stability
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
    P = torch.exp(logP)  # (B,B)
    return P


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
        import ot as pot 

        B = cost.shape[0]
        a = pot.unif(B)
        b = pot.unif(B)
        C = cost.detach().cpu().numpy()
        P = pot.emd(a, b, C, numThreads=num_threads)
        P = torch.tensor(P, device=cost.device, dtype=cost.dtype)
        return P

    raise ValueError(f"Unknown OT method: {method}")


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
    """
    z0, z1: (B,C,H,W) or (B,D)
    Returns:
      z0_pi, z1_pi: both (B,...) sampled from the OT plan pi over the minibatch.
      (i, j) are sampled from flattened pi (closest to official sample_plan behavior).
    """
    B = z0.shape[0]
    z0f = z0.view(B, -1)
    z1f = z1.view(B, -1)

    # cost matrix (B,B): squared L2 normalized by dim for scale stability
    cost = ((z0f[:, None, :] - z1f[None, :, :]) ** 2).mean(dim=2)

    P = compute_ot_plan(cost, method=ot_method, eps=eps, iters=iters, num_threads=num_threads)

    # sample (i,j) from flattened P
    p = (P.reshape(-1) + 1e-12)
    p = p / p.sum()

    idx = torch.multinomial(p, num_samples=B, replacement=replace)
    i = idx // B
    j = idx % B

    return z0[i], z1[j]
