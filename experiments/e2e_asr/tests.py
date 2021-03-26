import torch
from torch import Tensor
from typing import Dict, Optional

from extension import SRUppTransducerCell


if __name__ == "__main__":
    cell1 = SRUppTransducerCell(12, 12, 4, has_attention=True, highway_bias=0, right_window=1)
    cell1.transform_module.alpha.data[:] = 10.0  # type: ignore
    print(cell1)

    inputs = torch.randn(4, 1, 12)
    state: Dict[str, Optional[Tensor]] = {}
    prev_c = None
    list_h = []
    for t in range(4):
        ht, ct, state = cell1(inputs[t].unsqueeze(0), c0=prev_c, incremental_state=state)
        prev_c = ct
        list_h.append(ht)

    # attention mask of num_query * num_key
    # each tok can look 1 more to the right
    # 0, 0, x, x
    # 0, 0, 0, x
    # 0, 0, 0, 0
    # 0, 0, 0, 0
    attn_mask = torch.zeros(4, 4)
    attn_mask[0, 2] = -100000.0
    attn_mask[0, 3] = -100000.0
    attn_mask[1, 3] = -100000.0
    state = {}
    new_ht, new_ct, state = cell1(inputs, c0=None, attn_mask=attn_mask, incremental_state=state)
    print((new_ct - ct).abs().max())
    print((new_ht - torch.cat(list_h, dim=0)).abs().max())
