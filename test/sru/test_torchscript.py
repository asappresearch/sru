import pytest
import torch
import sru

@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("rescale", [False, True])
def test_all(bidirectional, rescale):
    eps = 1e-4
    torch.manual_seed(1234)
    L = 16
    B = 8
    D = 32
    x = torch.randn(L, B, D)
    model = sru.SRU(D, D, bidirectional=bidirectional,
                    rescale=rescale)
    model.eval()

    h, c = model(x)
    h, c = h.detach(), c.detach()

    with torch.no_grad():
        h_, c_ = model(x)
        assert (h - h_).abs().max() <= eps
        assert (c - c_).abs().max() <= eps

    ts_model = torch.jit.script(model)
    h_, c_ = ts_model(x)
    assert (h - h_).abs().max() <= eps
    assert (c - c_).abs().max() <= eps

