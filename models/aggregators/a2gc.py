import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Utilities: OT solvers ==========

def log_asymmetric_ot_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0):
    """
    Asymmetric OT solver in log-domain.
    This version allows asymmetric transport, meaning the source and target
    marginals are handled differently in terms of calibration.

    Args:
        log_a: [B, m_plus] source log-weights
        log_b: [B, n] target log-weights
        M    : [B, m_plus, n] log-affinity (not cost) matrix
        reg  : temperature regularization factor
    Returns:
        log_transport: [B, m_plus, n] log of asymmetric transport matrix
    """
    M = M / reg  # regularization

    for _ in range(num_iters):
        # Row normalization (log-softmax for each row)
        row_log = Z - torch.logsumexp(Z, dim=2, keepdim=True)  # [B, m_plus, n]
        # Column normalization (log-softmax for each column)
        col_log = Z - torch.logsumexp(Z, dim=1, keepdim=True)  # [B, m_plus, n]

        # Averaging the row and column softmax to get the final log values
        Z = 0.5 * (row_log + col_log)

    # Asymmetric Marginal Calibration (separate for rows and columns)
    row_sum = torch.logsumexp(Z, dim=2)  # [B, m_plus]
    u = log_a - row_sum  # Source marginal calibration
    Z = Z + u.unsqueeze(2)

    col_sum = torch.logsumexp(Z, dim=1)  # [B, n]
    v = log_b - col_sum  # Target marginal calibration
    Z = Z + v.unsqueeze(1)

    return Z  # logP


def get_matching_probs(
        S,  # [B, m, n] raw scores
        dustbin_score=1.0,
        num_iters=3,
        reg=1.0,
        temperature=1.0,
        mode="asymmetric_ot",  "asymmetric_ot"
        gumbel_tau=1.0,
        gumbel_noise=True,
        mask=None  # Optional [B, n] boolean mask for valid tokens
):
    """
    Compute (log) assignment with dustbin using selected mode.
    Returns:
        logP_aug: [B, m+1, n]
    """
    B, m, n = S.size()

    # Optional mask invalid target positions (columns). Very useful with padding.
    if mask is not None:
        # mask: True=valid, False=invalid
        # For invalid cols, set scores to large negative.
        invalid = (~mask).unsqueeze(1)  # [B,1,n]
        S = S.masked_fill(invalid, float('-inf'))

    # Temperature scaling (sharper distributions when temperature<1)
    if temperature != 1.0:
        S = S / temperature

    # ----- Build augmented matrix with dustbin row -----
    S_aug = torch.full((B, m + 1, n), fill_value=float('-inf'), dtype=S.dtype, device=S.device)
    S_aug[:, :m, :] = S
    S_aug[:, m, :] = dustbin_score

    # ----- Build log-marginals for Contrastive OT -----
    total_mass = n + m
    norm = -math.log(total_mass)
    log_a = S.new_full((B, m + 1), norm)
    log_b = S.new_full((B, n), norm)

    dust_mass = max(n - m, 1)
    log_a[:, -1] = math.log(dust_mass) - math.log(total_mass)


    # Use Asymmetric OT solver if the mode is set to "asymmetric_ot"
    if mode == "asymmetric_ot":
        logP = log_asymmetric_ot_solver(
            log_a=log_a, log_b=log_b, M=S_aug, num_iters=num_iters, reg=reg
        )
        return logP

    # Default Contrastive OT solver
    logP = log_contrastive_ot_solver(
        log_a=log_a, log_b=log_b, M=S_aug, num_iters=num_iters, reg=reg
    )
    return logP


# ========== The SALAD module (enhanced) ==========

class SALAD(nn.Module):
    """
    Asymmetric Optimal Transport Algorithm for Locally Aggregated Descriptors (enhanced).
    Backward compatible with your original interface; new features are opt-in.

    Args:
        num_channels: d (backbone channel dim)
        num_clusters: m
        cluster_dim : l (reduced dim for local features)
        token_dim   : g (global token dim)
        dropout     : dropout rate
        assignment_mode: "contrastive_ot" (default), "gumbel_contrastive_ot", "dual_softmax", "asymmetric_ot"
        contrastive_ot_iters, contrastive_ot_reg, contrastive_ot_temp: OT hyperparams
        gumbel_tau, gumbel_noise: Gumbel-Constrastive hyperparams
        use_geometric_ot: whether to add geometric compatibility S_geo
        geo_dim, geo_lambda: dimension and weight for geometric term
    """

    def __init__(
            self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
            assignment_mode="asymmetric_ot",  # default: contrastive_ot
            contrastive_ot_iters=3,
            contrastive_ot_reg=1.0,
            contrastive_ot_temp=1.0,
            gumbel_tau=1.0,

            use_geometric_ot=True,
            geo_dim=16,
            geo_lambda=0.15,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        self.assignment_mode = assignment_mode
        self.contrastive_ot_iters = contrastive_ot_iters
        self.contrastive_ot_reg = contrastive_ot_reg
        self.contrastive_ot_temp = contrastive_ot_temp
        self.gumbel_tau = gumbel_tau


        self.use_geometric_ot = use_geometric_ot
        self.geo_dim = geo_dim
        self.geo_lambda = geo_lambda

        drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.token_dim)
        )

        # MLP for local features f_i (reduce dim from d to l)
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            drop,
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.cluster_dim, 1)
        )

        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            drop,
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        # Dustbin parameter z (learnable scalar)
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        # ----- Geometric OT branch -----
        if self.use_geometric_ot:
            # project 2D coords (x,y) to geo_dim via 1x1 conv
            self.geo_proj = nn.Conv2d(2, self.geo_dim, kernel_size=1)
            self.cluster_geo = nn.Parameter(torch.randn(self.num_clusters, self.geo_dim) * 0.02)

    @staticmethod
    def _make_coord_grid(B, H, W, device, dtype):
        ys = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        return coords

    def _add_geometric_scores(self, B, H, W, device, dtype):
        coords = self._make_coord_grid(B, H, W, device, dtype)
        geo = self.geo_proj(coords)
        geo = geo.flatten(2).transpose(1, 2)
        cluster_geo = self.cluster_geo.unsqueeze(0)
        S_geo = torch.einsum('bng,bmg->bmn', geo, cluster_geo)
        return S_geo

    def forward(self, x, mask=None):
        feat_map, t = x
        B, C, H, W = feat_map.shape
        assert C == self.num_channels

        f = self.cluster_features(feat_map).flatten(2)
        S = self.score(feat_map).flatten(2)
        if self.use_geometric_ot:
            S_geo = self._add_geometric_scores(B, H, W, feat_map.device, feat_map.dtype)
            S = S + self.geo_lambda * S_geo

        t = self.token_features(t)

        logP_aug = get_matching_probs(
            S,
            dustbin_score=self.dust_bin,
            num_iters=self.contrastive_ot_iters,
            reg=self.contrastive_ot_reg,
            temperature=self.contrastive_ot_temp,
            mode=self.assignment_mode,
            gumbel_tau=self.gumbel_tau,

            mask=mask
        )
        P_aug = torch.exp(logP_aug)
        P = P_aug[:, :-1, :]

        N = f.size(-1)
        V = torch.einsum('bln,bmn->blm', f, P)
        V = F.normalize(V, p=2, dim=1).flatten(1)

        g = F.normalize(t, p=2, dim=-1)
        out = torch.cat([g, V], dim=-1)
        return F.normalize(out, p=2, dim=-1)
