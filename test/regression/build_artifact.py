"""
- instantiates an SRU object
- creates some dummy input
- passes input through sru to get outpus
- saves in artifact file:
    - model weights
    - inputs
    - outputs
    - sru constructor args
    - sru version
"""
import torch
import sru
import argparse


def run(args):
    torch.manual_seed(1)

    batch_size = 3
    input_size = 5
    hidden_size = 7
    seq_len = 4
    num_layers = 2

    sru_kwargs = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'bidirectional': True,
        'dropout': 0.1,
        'rescale': False
    }

    inputs = torch.rand(seq_len, batch_size, input_size)
    model = sru.SRU(**sru_kwargs).eval()

    with torch.no_grad():
        outputs = model(inputs)

    artifact_dict = {
        'outputs': outputs,
        'inputs': inputs,
        'model_state': model.state_dict(),
        'sru_kwargs': sru_kwargs,
        'sru.__version__': sru.__version__
    }
    torch.save(artifact_dict, args.out_artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-artifact', type=str, required=True,
                        help='filepath to store artifact')
    args = parser.parse_args()
    run(args)
