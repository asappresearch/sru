import time
import torch
import torch.nn as nn

N = 200

def run_mm(X, weight):
    torch.cuda.synchronize()
    start = time.time()
    for i in range(N):
        x = X[i%10]
        x_2d = x.contiguous().view(-1, x.size(-1))
        U = x_2d.mm(weight)
        loss = U.mean()
        loss.backward()
    torch.cuda.synchronize()
    print ("mm() takes: {}".format(time.time()-start))

def run_matmul(X, weight):
    torch.cuda.synchronize()
    start = time.time()
    for i in range(N):
        x = X[i%10]
        U = x.matmul(weight)
        loss = U.mean()
        loss.backward()
    torch.cuda.synchronize()
    print ("matmul() takes: {}".format(time.time()-start))

def run_addmm(X, weight, bias):
    torch.cuda.synchronize()
    start = time.time()
    for i in range(N):
        x = X[i%10]
        x_2d = x.contiguous().view(-1, x.size(-1))
        U = torch.addmm(bias, x_2d, weight)
        loss = U.mean()
        loss.backward()
    torch.cuda.synchronize()
    print ("addmm() takes: {}".format(time.time()-start))

def run_test(L, B, D, D2):
    print ("")
    print ("===========================")
    print ("L={} B={} D={} D2={}".format(L, B, D, D2))
    print ("---------------------------")
    weight = nn.Parameter(torch.Tensor(D, D2*3)).cuda()
    weight.data.uniform_(-(3.0/D)**0.5, (3.0/D)**0.5)
    weight_t = nn.Parameter(torch.Tensor(D2*3, D)).cuda()
    weight_t.data.copy_(weight.data.t())
    bias = nn.Parameter(torch.zeros(D2*3))
    bias.requires_grad = False
    bias = bias.cuda()
    x = [ torch.randn(L, B, D, requires_grad=True).cuda() for i in range(10) ]
    #print (type(weight), weight.type(), weight.requires_grad)
    #print (type(bias), bias.type(), bias.requires_grad)
    #print (type(x[0]), x[0].type(), x[0].requires_grad)
    run_mm(x, weight)
    run_matmul(x, weight)
    run_addmm(x, weight, bias)
    print("Transposed:")
    run_mm(x, weight_t.t())
    run_matmul(x, weight_t.t())
    run_addmm(x, weight_t.t(), bias)
    print ("===========================")

def run_small():
    L = 32
    B = 32
    D = D2 = 300
    run_test(L, B, D, D2)

def run_large():
    L = 64
    B = 64
    D = 512
    D2 = 2048
    run_test(L, B, D, D2)

run_small()
run_large()
