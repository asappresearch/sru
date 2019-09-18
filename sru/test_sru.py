from sru import SRU
import torch
from torch import nn
import numpy as np

def test_packed():
    N = 5
    max_len = 7
    V = 32
    K = 8
    K_out = 11

    print('N', N, 'max_len', max_len, 'K', K, 'K_out', K_out)

    torch.manual_seed(123)
    np.random.seed(123)
    lengths = torch.from_numpy(np.random.choice(max_len, N)) + 1
    tensors = [torch.from_numpy(np.random.choice(V, l, replace=True)) for l in lengths.tolist()]
    embedder = nn.Embedding(V, K)
    tensors = nn.utils.rnn.pad_sequence(tensors)
    print('tensors.size()', tensors.size())
    embedded = embedder(tensors)
    print('embedded.size()', embedded.size())
    packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=False, enforce_sorted=False)
    print(isinstance(packed, nn.utils.rnn.PackedSequence))

    sru = SRU(K, K_out)
    out1, state = sru(packed)
    out1, lengths1 = nn.utils.rnn.pad_packed_sequence(out1)
    print('out1.size()', out1.size())
    assert (lengths != lengths1).sum().item() == 0
    print('out1.sum()', out1.sum().item())

    # change one of the indexes taht should not be masked out
    tensors[6, 1] = 3
    embedded = embedder(tensors)
    packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=False, enforce_sorted=False)
    out2, state = sru(packed)
    out2, lengths2 = nn.utils.rnn.pad_packed_sequence(out2)
    assert (lengths != lengths2).sum().item() == 0
    print('out2.sum()', out2.sum().item())
    assert out2.sum().item() == out1.sum().item()

    # change one of the indexes taht should be masked out
    tensors[1, 1] = 3
    embedded = embedder(tensors)
    packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=False, enforce_sorted=False)
    out3, state = sru(packed)
    out3, lengths3 = nn.utils.rnn.pad_packed_sequence(out3)
    assert (lengths != lengths3).sum().item() == 0
    print('out3.sum()', out3.sum().item())
    assert out3.sum().item() != out1.sum().item()
