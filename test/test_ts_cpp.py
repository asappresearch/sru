import torch
import sru
import argparse


def run(args):
    D = 4
    model = sru.SRU(D, D, num_layers=2, normalize_after=args.normalize_after)
    model.eval()

    ts_model = torch.jit.script(model)
    ts_model.save('sru_ts.pt')

    with torch.no_grad():
        x = torch.ones(3, 2, D)
        h, c = model(x)
        h, c = h.view(-1), c.view(-1)
        print(''.join(["{:.4f} ".format(x.item()) for x in h]))
        print(''.join(["{:.4f} ".format(x.item()) for x in c]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize-after', action='store_true')
    args = parser.parse_args()
    run(args)
