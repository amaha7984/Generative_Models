import torch
import numpy as np
import ot as pot 

class OTPlanSampler:
    """
    Simple minibatch OT coupler: 'exact' (BatchOT) or 'sinkhorn' (BatchEOT).
    Works on tensors of shape [B, C, H, W] or [B, D].
    """
    def __init__(self, method="exact", reg=0.05):
        assert method in [None, "exact", "sinkhorn"]
        self.method = method
        self.reg = float(reg)

    @torch.no_grad()
    def sample_plan(self, x0, x1):
        """
        Returns a re-ordered x1 so that (x0[i], x1_matched[i]) follows the OT coupling.
        x0, x1: tensors with same batch size.
        """
        if self.method is None:
            return x0, x1

        B = x0.size(0)
        X0 = x0.reshape(B, -1).detach().cpu().numpy().astype(np.float64)
        X1 = x1.reshape(B, -1).detach().cpu().numpy().astype(np.float64)

        # cost = squared Euclidean
        M = pot.dist(X0, X1, metric="sqeuclidean")

        a = np.ones(B, dtype=np.float64) / B
        b = np.ones(B, dtype=np.float64) / B

        if self.method == "exact":
            Pi = pot.emd(a, b, M)                           # almost permutation
        else:  # 'sinkhorn'
            Pi = pot.sinkhorn(a, b, M, reg=self.reg)

        # greedy readout per row (good enough; Pi ~ permutation matrix)
        match_idx = Pi.argmax(axis=1)
        match_idx = torch.from_numpy(match_idx).to(x1.device, dtype=torch.long)

        x1_matched = x1.index_select(0, match_idx)
        return x0, x1_matched
