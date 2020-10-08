"""
test_regression_1.py and test_regression_2.py together
test:
- running some numbers through two versions of sru, checking they come out the sam
- save sru in older version, and loading in new version

these scripts are adapted from those by Sean Adler
"""
import torch
import sru
import argparse


def run(args):
    torch.manual_seed(1)

    batch_size = 200
    input_size = 300
    hidden_size = 250
    seq_len = 150

    inputs = torch.rand(seq_len, batch_size, input_size)
    model = sru.SRU(
        input_size,
        hidden_size,
        num_layers=4,
        bidirectional=True,
        dropout=0.1,
        rescale=False
    ).eval()

    with torch.no_grad():
        outputs = model(inputs)
    torch.save(inputs, args.out_inputs)
    torch.save(outputs, args.out_outputs)
    torch.save(model.state_dict(), args.out_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-inputs', type=str, required=True,
                        help='filepath to store inputs')
    parser.add_argument('--out-outputs', type=str, required=True,
                        help='filepath to store outputs')
    parser.add_argument('--out-model', type=str, required=True,
                        help='filepath to store model')
    args = parser.parse_args()
    run(args)
