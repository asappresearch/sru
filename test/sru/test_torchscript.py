import pytest
import torch
import sru


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("proj", [0, 4])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_all(bidirectional, rescale, proj, layer_norm):
    eps = 1e-4
    torch.manual_seed(1234)
    L = 16
    B = 8
    D = 32
    x = torch.randn(L, B, D)
    model = sru.SRU(D, D, bidirectional=bidirectional,
                    projection_size=proj,
                    layer_norm=layer_norm,
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


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("proj", [2, 4])
@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("normalize_after", [False, True])
@pytest.mark.parametrize("attn_every_n_layers", [1, 2])
def test_srupp(bidirectional, proj, layer_norm, normalize_after, attn_every_n_layers):
    eps = 1e-4
    torch.manual_seed(1234)
    L = 16
    B = 8
    D = 32
    x = torch.randn(L, B, D)
    model = sru.SRUpp(D, D, proj,
                      bidirectional=bidirectional,
                      layer_norm=layer_norm,
                      normalize_after=normalize_after,
                      attention_every_n_layers=attn_every_n_layers)
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
