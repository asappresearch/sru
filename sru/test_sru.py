from sru import SRU
import torch
from torch import nn
import numpy as np

def test_gru_compatible_state_return():
    N = 5
    max_len = 7
    V = 32
    K = 8
    K_out = 11
    num_layers = 3
    bidirectional = True

    print('N', N, 'max_len', max_len, 'num_layers', num_layers, 'bidirectional', bidirectional, 'K', K, 'K_out', K_out)

    torch.manual_seed(123)
    np.random.seed(123)
    lengths = torch.from_numpy(np.random.choice(max_len, N)) + 1
    tensors = [torch.from_numpy(np.random.choice(V, l, replace=True)) for l in lengths.tolist()]
    embedder = nn.Embedding(V, K)
    tensors = nn.utils.rnn.pad_sequence(tensors)
    embedded = embedder(tensors)

    sru = SRU(K, K_out, nn_rnn_compatible_return=True, bidirectional=bidirectional, num_layers=num_layers)
    out, state = sru(embedded)
    print('out.size()', out.size())
    print('state.size()', state.size())

    gru = nn.GRU(K, K_out, bidirectional=bidirectional, num_layers=num_layers)
    gru_out, gru_state = gru(embedded)
    print('gru_state.size()', gru_state.size())
