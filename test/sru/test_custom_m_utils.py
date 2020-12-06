import copy
import torch
from torch import nn
from sru import custom_m_utils, SRU, SRUCell
import pytest


EPS = 1e-6


@pytest.mark.parametrize(
    "d_in,d_out", [
        (4, 3),
        (3, 4),
        (4, 4),
    ]
)
def test_convert_sru_cell_to_linear(d_in, d_out):
    cell = SRUCell(d_in, d_out)
    cell2 = copy.deepcopy(cell)

    custom_m_utils.convert_sru_cell_to_linears(cell2)
    assert cell2.weight is None
    assert cell2.weight_proj is None
    assert cell2.custom_m is not None
    assert isinstance(cell2.custom_m, nn.Linear)
    inputs = torch.rand(3, d_in)
    outputs1, _ = cell(inputs)
    outputs2, _ = cell2(inputs)
    assert (outputs1 - outputs2).abs().max() < EPS


@pytest.mark.parametrize(
    "d_in,d_proj,d_out", [
        (3, 2, 4),
        (4, 3, 4),
        (4, 2, 3),
    ]
)
def test_convert_sru_cell_with_proj_to_custom(d_in, d_proj, d_out):
    cell = SRUCell(d_in, d_out, n_proj=d_proj)
    cell2 = copy.deepcopy(cell)
    custom_m_utils.convert_sru_cell_to_linears(cell2)
    assert cell2.weight is None
    assert cell2.weight_proj is None
    assert cell2.custom_m is not None
    assert isinstance(cell2.custom_m, nn.Sequential)
    assert len(cell2.custom_m) == 2
    assert isinstance(cell2.custom_m[0], nn.Linear)
    assert isinstance(cell2.custom_m[1], nn.Linear)
    inputs = torch.rand(3, d_in)
    outputs1, _ = cell(inputs)
    outputs2, _ = cell2(inputs)
    assert (outputs1 - outputs2).abs().max() < EPS


@pytest.mark.parametrize(
    "d_in,d_out", [
        (4, 3),
        (3, 4),
        (4, 4),
    ]
)
def test_convert_sru_to_custom(d_in, d_out):
    sru = SRU(d_in, d_out, num_layers=4, bidirectional=False)
    sru2 = copy.deepcopy(sru)
    custom_m_utils.convert_sru_to_linears(sru2)
    for cell2 in sru2.rnn_lst:
        assert cell2.weight is None
        assert cell2.weight_proj is None
        assert cell2.custom_m is not None
        assert isinstance(cell2.custom_m, nn.Linear)
    inputs = torch.rand(5, 3, d_in)
    outputs1, _ = sru(inputs)
    outputs2, _ = sru2(inputs)
    assert (outputs1 - outputs2).abs().max() < EPS


@pytest.mark.parametrize(
    "d_in,d_proj,d_out", [
        (3, 2, 4),
        (4, 3, 4),
        (4, 2, 3),
    ]
)
def test_convert_sru_with_proj_to_custom(d_in, d_proj, d_out):
    sru = SRU(d_in, d_out, num_layers=4, bidirectional=False, projection_size=d_proj)
    sru2 = copy.deepcopy(sru)
    custom_m_utils.convert_sru_to_linears(sru2)
    for cell2 in sru2.rnn_lst:
        assert cell2.weight is None
        assert cell2.weight_proj is None
        assert cell2.custom_m is not None
        assert isinstance(cell2.custom_m, nn.Sequential)
        assert len(cell2.custom_m) == 2
        assert isinstance(cell2.custom_m[0], nn.Linear)
        assert isinstance(cell2.custom_m[1], nn.Linear)
    inputs = torch.rand(5, 3, d_in)
    outputs1, _ = sru(inputs)
    outputs2, _ = sru2(inputs)
    assert (outputs1 - outputs2).abs().max() < EPS
