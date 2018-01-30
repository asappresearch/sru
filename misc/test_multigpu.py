import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from cuda_functional import SRU

class Model(nn.Module):
    def __init__(self, rnn):
        super(Model, self).__init__()
        self.rnn = rnn

    def forward(self, x):
        out, state = self.rnn(x)
        #return out      # synchronize more data
        return out[-1:]  # synchronize less data (only the last state)

def run(x, model, N):
    start = time.time()
    for _ in range(N):
        h = model(x)
    torch.cuda.synchronize()
    print ("{:.4} sec\n".format(time.time()-start))

N = 100
input_size, hidden_size = 512, 1024
num_layers = 2
batch_size = 128*4
length = 64

x = Variable(torch.randn(length, batch_size, input_size).float(), volatile=True)
x = x.cuda()

print ("")

# single gpu
rnn = Model(SRU(input_size, hidden_size, num_layers))
rnn.cuda()
rnn(x)
print ("\nSingle gpu:")
run(x, rnn, N)

# multiple gpu
rnn_2 = Model(SRU(input_size, hidden_size, num_layers))
rnn_2 = nn.DataParallel(rnn_2, dim=1)
rnn_2.cuda()
rnn_2(x)
print ("\nMulti gpu:")
run(x, rnn_2, N)
