# Tested with:
# torch==1.1.0
# sru==1.6.2
# cupy==6.4.0
#
# CUDA Version: 10.1
# Author: Ivan Itzcovich (iitzcovich@asapp.com)

from sru import SRU
import torch

torch.manual_seed(10)

input_t = torch.rand(50, 10, 200)
output_t = torch.empty(10, dtype=torch.long).random_(5)
loss = torch.nn.CrossEntropyLoss()

rnn = SRU(
    input_size=200,
    hidden_size=10,
    num_layers=2,
    dropout=0.0,
    bidirectional=False,
    layer_norm=False,
    highway_bias=0,
    rescale=True,
)


def profile(rnn, input_t, output_t, device):
    rnn.zero_grad()
    input_t, output_t, rnn = input_t.to(device), output_t.to(device), rnn.to(device)
    preds, state = rnn(input_t)
    output = loss(preds[-1, :, :], output_t)
    print(f"Loss: {output.item()}")
    output.backward()
    grads = [p.grad.data.sum() for p in rnn.parameters() if p.requires_grad]
    print(f"Sum of Gradients: {torch.stack(grads).sum()}")

def profile_with_transpose(rnn, input_t, output_t, device):
    rnn.zero_grad()
    input_t, output_t, rnn = input_t.to(device), output_t.to(device), rnn.to(device)
    preds, state = rnn(input_t)
    preds = preds.transpose(0, 1)
    output = loss(preds[:, -1, :], output_t)
    print(f"Loss: {output.item()}")
    output.backward()
    grads = [p.grad.data.sum() for p in rnn.parameters() if p.requires_grad]
    print(f"sum of Gradients: {torch.stack(grads).sum()}")


# CPU
print("CPU mode:")
profile(rnn, input_t, output_t, 'cpu')
print()

# GPU
print("GPU mode:")
profile(rnn, input_t, output_t, 'cuda')
print()

# CPU
print("CPU + Transposing mode:")
profile_with_transpose(rnn, input_t, output_t, 'cpu')
print()

# GPU
print("GPU + Transposing mode:")
profile_with_transpose(rnn, input_t, output_t, 'cuda')
