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


EPSILON = 1e-6


def run(args):
    torch.manual_seed(2)  # so the model is initialized differently than first stage

    input_size = 300
    hidden_size = 250

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
    inputs = torch.load(args.in_inputs)

    with torch.no_grad():
        output_sru_250 = model(inputs)
    assert len(output_sru_235) == len(output_sru_250) == 2
    max_diff0 = (output_sru_235[0] - output_sru_250[0]).abs().max().item()
    max_diff1 = (output_sru_235[1] - output_sru_250[1]).abs().max().item()
    assert max_diff0 <= EPSILON
    assert max_diff1 <= EPSILON


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-inputs', type=str, required=True,
                        help='filepath to load inputs from')
    parser.add_argument('--in-outputs', type=str, required=True,
                        help='filepath to load outputs from')
    parser.add_argument('--in-model', type=str, required=True,
                        help='filepath to load model from')
    args = parser.parse_args()
    run(args)
