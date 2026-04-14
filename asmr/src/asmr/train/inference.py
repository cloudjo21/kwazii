"""Apply trained aggregation to numpy shortlist scores (inference path)."""

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from asmr.train.aggregation import AggregationHead, MFARFieldAdapter
from asmr.train.features import build_aux_features


def _to_torch(a: npt.NDArray[np.float32], device: torch.device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def apply_aggregation_head(
    doc_ids: list[str] | list[int],
    scores: npt.NDArray[np.float32],
    field_mask: npt.NDArray[np.bool_],
    query_emb: npt.NDArray[np.float32],
    head: torch.nn.Module,
    device: torch.device | None = None,
) -> tuple[list[str] | list[int], npt.NDArray[np.float32]]:
    """Run aggregation module on one query shortlist.

    Args:
        doc_ids: Length D candidate ids (order matches score columns).
        scores: Shape [F, M, D] or [F, D]. If 3D without M, use [F, 1, D].
        field_mask: [F, D] True if document has field.
        query_emb: [H] single query embedding.
        head: MFARFieldAdapter or AggregationHead.
        device: Torch device; default cpu.

    Returns:
        Sorted doc_ids (same element type as input) and logits (descending).
    """
    dev = device or torch.device("cpu")
    if scores.ndim == 2:
        scores_t = _to_torch(scores, dev).unsqueeze(1)
    elif scores.ndim == 3:
        scores_t = _to_torch(scores, dev)
    else:
        raise ValueError("scores must be [F,D] or [F,M,D]")

    f_num, m_num, d_num = scores_t.shape
    if len(doc_ids) != d_num:
        msg = "doc_ids length must match scores last dim"
        raise ValueError(msg)

    mask_t = _to_torch(field_mask.astype(np.bool_), dev).bool()
    if mask_t.shape != (f_num, d_num):
        raise ValueError("field_mask must be [F,D]")

    q = _to_torch(query_emb, dev).unsqueeze(0)
    scores_b = scores_t.unsqueeze(0)
    mask_b = mask_t.unsqueeze(0)

    head.eval()
    with torch.no_grad():
        if isinstance(head, MFARFieldAdapter):
            logits = head(q, scores_b).squeeze(0)
        elif isinstance(head, AggregationHead):
            aux = build_aux_features(scores_b, mask_b)
            logits = head(q, scores_b, aux).squeeze(0)
        else:
            msg = f"Unsupported head type: {type(head)}"
            raise TypeError(msg)

    order = torch.argsort(logits, descending=True).cpu().numpy()
    sorted_ids = [doc_ids[i] for i in order.tolist()]
    return sorted_ids, logits.cpu().numpy()
