import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from cuda_functional import SRU

def run(x, model, N):
    start = time.time()
    for _ in range(N):
        h = model(x)
    torch.cuda.synchronize()
    print ("{:.4}".format(time.time()-start))

N = 1000
input_size, hidden_size = 2048, 2048
num_layers = 2
batch_size = 256
length = 32

x = Variable(torch.randn(length, batch_size, input_size).float())
x = x.cuda()

# single gpu
rnn = SRU(input_size, hidden_size, num_layers)
rnn.cuda()
rnn(x)
print ("Single gpu:")
run(x, rnn, N)

# multiple gpu
rnn_2 = SRU(input_size, hidden_size, num_layers)
rnn_2 = nn.DataParallel(rnn_2, dim=1)
rnn_2.cuda()
rnn_2(x)
print ("Multi gpu:")
run(x, rnn_2, N)

