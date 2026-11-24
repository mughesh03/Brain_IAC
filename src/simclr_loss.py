"""SimCLR NT-Xent loss.

Single-device PyTorch form of ``add_contrastive_loss`` from
google-research/simclr ``tf2/objective.py``.

The two augmented views are stacked on the batch axis, view 1 in the first
half and view 2 in the second half. Embeddings are L2-normalized, similarities
are temperature-scaled, and each sample's positive is its matching view.
Self-similarity on the same view is removed from the negatives.
"""

import torch
import torch.nn.functional as F

LARGE_NUM = 1e9


def add_contrastive_loss(hidden, hidden_norm=True, temperature=0.1):
    """NT-Xent over a stacked pair of views.

    Args:
        hidden: Float tensor ``(2 * batch, dim)``.
        hidden_norm: L2-normalize embeddings before the similarity.
        temperature: Logit scale. BrainIAC was trained at 0.1.

    Returns:
        Scalar loss, the mean of the two directed cross-entropies.
    """
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be (2 * batch, dim), got {tuple(hidden.shape)}")
    if hidden.shape[0] % 2 != 0:
        raise ValueError("hidden batch must be even: view 1 stacked with view 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if hidden_norm:
        hidden = F.normalize(hidden, p=2, dim=-1)

    hidden1, hidden2 = torch.chunk(hidden, 2, dim=0)
    batch_size = hidden1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent needs at least two images so each view has a negative")

    masks = torch.eye(batch_size, device=hidden.device, dtype=hidden.dtype)
    logits_aa = (hidden1 @ hidden1.T) / temperature - masks * LARGE_NUM
    logits_bb = (hidden2 @ hidden2.T) / temperature - masks * LARGE_NUM
    logits_ab = (hidden1 @ hidden2.T) / temperature
    logits_ba = (hidden2 @ hidden1.T) / temperature

    labels = torch.arange(batch_size, device=hidden.device)
    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
    return loss_a + loss_b
