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

    # uniform marginals
    log_a = torch.full((B,), -torch.log(torch.tensor(float(B), device=device, dtype=dtype)), device=device, dtype=dtype)
    log_b = log_a.clone()

    for _ in range(iters):
        # log_u = log_a - logsumexp(logK + log_v[None,:], dim=1)
        log_u = log_a - torch.logsumexp(logK + log_v.unsqueeze(0), dim=1)
        # log_v = log_b - logsumexp(logK^T + log_u[None,:], dim=1)
        log_v = log_b - torch.logsumexp(logK.transpose(0, 1) + log_u.unsqueeze(0), dim=1)

    logP = logK + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    P = torch.exp(logP)  # (B,B)
    return P


@torch.no_grad()
def minibatch_ot_pairing(z0, z1, eps=0.05, iters=50, method="argmax"):
    """
    z0, z1: (B,C,H,W) or (B,D)
    Returns:
      perm: LongTensor of shape (B,) such that z1_matched = z1[perm]
    """
    B = z0.shape[0]
    z0f = z0.view(B, -1)
    z1f = z1.view(B, -1)

    # cost matrix (B,B): squared L2
    # normalized by dim for scale stability
    dim = z0f.shape[1]
    cost = ((z0f[:, None, :] - z1f[None, :, :]) ** 2).mean(dim=2)

    P = sinkhorn_log_domain(cost, eps=eps, iters=iters)

    if method == "argmax":
        perm = torch.argmax(P, dim=1)
        return perm

    if method == "sample":
        # sample j ~ P[i,:]
        probs = (P + 1e-12) / (P.sum(dim=1, keepdim=True) + 1e-12)
        perm = torch.multinomial(probs, num_samples=1).squeeze(1)
        return perm

    raise ValueError(f"Unknown pairing method: {method}")
