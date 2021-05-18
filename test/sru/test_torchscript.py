import pytest
import torch
import sru


@pytest.mark.parametrize(
    "cuda",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="no cuda available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("proj", [0, 4])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_all(cuda, bidirectional, rescale, proj, layer_norm):
    eps = 1e-4
    torch.manual_seed(1234)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    L = 16
    B = 8
    D = 32
    x = torch.randn(L, B, D)
    model = sru.SRU(D, D, bidirectional=bidirectional,
                    projection_size=proj,
                    layer_norm=layer_norm,
                    rescale=rescale)
    if cuda:
        model = model.cuda()
        x = x.cuda()
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
