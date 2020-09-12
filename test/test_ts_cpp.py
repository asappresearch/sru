import torch
import sru

D = 4
model = sru.SRU(D, D, num_layers=2)
model.eval()

ts_model = torch.jit.script(model)
ts_model.save('sru_ts.pt')

with torch.no_grad():
    x = torch.ones(3, 2, D)
    h, c = model(x)
    h, c = h.view(-1), c.view(-1)
    print(''.join(["{:.4f} ".format(x.item()) for x in h]))
    print(''.join(["{:.4f} ".format(x.item()) for x in c]))
