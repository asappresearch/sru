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
    input_size = 304
    hidden_size = 304
    seq_len = 304

    inputs = torch.rand(seq_len, batch_size, input_size)
    model = sru.SRU(
        input_size,
        hidden_size,
        num_layers=4,
        bidirectional=True,
        dropout=0.1,
        rescale=False
    ).eval()

    output_sru_235 = torch.load(args.in_outputs)
    model.load_state_dict(torch.load(args.in_model))

    with torch.no_grad():
        output_sru_250 = model(inputs)
        assert len(output_sru_235) == len(output_sru_250) == 2
        assert torch.allclose(output_sru_235[0], output_sru_250[0])
        assert torch.allclose(output_sru_235[1], output_sru_250[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-outputs', type=str, required=True,
                        help='filepath to load outputs from')
    parser.add_argument('--in-model', type=str, required=True,
                        help='filepath to load model from')
    args = parser.parse_args()
    run(args)
