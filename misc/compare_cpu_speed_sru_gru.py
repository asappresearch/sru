import time

import ninja
import torch
import torch.nn as nn
from sru import SRU, SRUCell

batch_size = 1
seq_len = 32
input_size = 256
hidden_size = 512
n_directions = 1
n_layers = 1

import multiprocessing
#torch.set_num_threads(multiprocessing.cpu_count())
print('N_CPUS:', multiprocessing.cpu_count())
print('N_TORCH_THREADS', torch.get_num_threads())

def benchmark_gru_cpu():
    print('-'*60)
    print('GRU CPU benchmark:')

    rnn = nn.GRU(input_size=input_size,
                 hidden_size=hidden_size,
                 batch_first=False,
                 bidirectional=(n_directions == 2),
                 num_layers=1)

    input = torch.randn(seq_len, batch_size, input_size)
    h0 = torch.randn(n_layers * n_directions, batch_size, hidden_size)
    print('input.shape', input.shape)
    print('h0.shape', h0.shape)
    output, hn = rnn(input, h0)
    print('output.shape', output.shape)
    print('hn.shape', hn.shape)

    n_iter = 1000
    start = time.time()
    with torch.no_grad():
        rnn.eval()
        for i in range(n_iter):
            rnn.forward(input)
    print('Time:', round((time.time() - start), 2), 'sec')


def benchmark_sru_cpu():
    print('-' * 60)
    print('SRU CPU benchmark:')

    rnn = SRU(input_size=input_size,
                 hidden_size=hidden_size,
                 bidirectional=(n_directions == 2),
                 num_layers=1)

    input = torch.randn(seq_len, batch_size, input_size)
    h0 = torch.randn(n_layers, batch_size, hidden_size * n_directions)
    print('input.shape', input.shape)
    print('h0.shape', h0.shape)
    with torch.no_grad():
        rnn.eval()
        output, hn = rnn(input, h0)
    print('output.shape', output.shape)
    print('hn.shape', hn.shape)

    n_iter = 1000
    start = time.time()
    with torch.no_grad():
        rnn.eval()
        for i in range(n_iter):
            rnn.forward(input)
    print('Time:', round((time.time() - start), 2), 'sec')

if __name__ == '__main__':
    benchmark_gru_cpu()
    benchmark_sru_cpu()
