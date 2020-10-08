"""
test:
- running some numbers through two versions of sru, checking they come out the sam
- save sru in older version, and loading in new version

IMPORTANT:

    You need to run test/regression/build_artifact.sh [SRU VERSION]

for each version you want to test, first
"""
import torch
import sru
import pytest


EPSILON = 1e-6


ARTIFACT_DIR = 'test/regression/artifacts'


@pytest.mark.parametrize(
    "sru_prev_version",
    ["2.3.5"]
)
def test_regression(sru_prev_version):
    torch.manual_seed(2)  # so the model is initialized differently than first stage

    artifact_path = f'{ARTIFACT_DIR}/{sru_prev_version}.pt'
    artifact_dict = torch.load(artifact_path)
    assert artifact_dict['sru.__version__'] == sru_prev_version

    model = sru.SRU(**artifact_dict['sru_kwargs']).eval()

    output_artifact = torch.load(artifact_dict['outputs'])
    model.load_state_dict(torch.load(artifact_dict['model']))
    with torch.no_grad():
        output_current = model(artifact_dict['inputs'])

    assert len(output_artifact) == len(output_current) == 2
    max_diff0 = (output_artifact[0] - output_current[0]).abs().max().item()
    max_diff1 = (output_artifact[1] - output_current[1]).abs().max().item()
    assert max_diff0 <= EPSILON
    assert max_diff1 <= EPSILON
