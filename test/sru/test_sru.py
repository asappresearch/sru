import pytest

import torch
from torch import nn

import sru
from sru import SRU, SRUpp
from sru.modules import SRUppAttention, SRUppProjectedLinear


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
@pytest.mark.parametrize("with_grad", [False, True])
@pytest.mark.parametrize("compat", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_sru(cuda, with_grad, compat, bidirectional, layer_norm):
    torch.manual_seed(123)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run():
        eps = 1e-4
        num_sentences = 3
        embedding_size = 7
        rnn_hidden = 4
        max_len = 4
        layers = 5
        encoder = sru.SRU(
            embedding_size,
            rnn_hidden,
            layers,
            bidirectional=bidirectional,
            layer_norm=layer_norm,
            nn_rnn_compatible_return=compat,
        )
        words_embeddings = torch.rand(
            (max_len, num_sentences, embedding_size), dtype=torch.float32
        )
        if cuda:
            words_embeddings = words_embeddings.to("cuda")
            encoder.cuda()
        encoder.eval()
        hidden, cell = encoder(words_embeddings)

        def cell_to_emb(cell, batch_size):
            if compat:
                # should arrive as:
                # (num_layers * num_directions, batch, hidden_size)
                cell = cell.view(
                    layers, 2 if bidirectional else 1, batch_size, rnn_hidden
                )
                cell = cell[-1].transpose(0, 1)
                # (batch, num_directions, hidden_size)
                cell = cell.contiguous().view(batch_size, -1)
            else:
                # should arrive as:
                # (num_layers, batch_size, num_directions * hidden_size)
                cell = cell[-1].view(batch_size, -1)
            return cell

        scores = cell_to_emb(cell, num_sentences)
        for i in range(num_sentences):
            hidden, cell = encoder(words_embeddings[:, i : i + 1])
            score = cell_to_emb(cell, 1)
            assert (score.detach() - scores[i].detach()).abs().max() <= eps

    if with_grad:
        run()
    else:
        with torch.no_grad():
            run()


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
@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("normalize_after", [False, True])
@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("has_skip_term", [False, True])
def test_sru_backward_simple(cuda, bidirectional, layer_norm, normalize_after, rescale, has_skip_term):
    torch.manual_seed(123)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    input_length = 3
    batch_size = 5
    input_size = 4
    hidden_size = 2
    encoder = sru.SRU(input_size, hidden_size,
                      bidirectional=bidirectional,
                      layer_norm=layer_norm,
                      normalize_after=normalize_after,
                      rescale=rescale,
                      has_skip_term=has_skip_term)
    if cuda:
        encoder = encoder.cuda()

    def run(x):
        if cuda:
            x = x.cuda()
        output, state = encoder(x)
        output.mean().backward()

    # test batch size > 1
    input_data = torch.rand(input_length, batch_size, input_size)
    run(input_data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("normalize_after", [False, True])
def test_sru_backward(bidirectional, layer_norm, normalize_after):
    eps = 1e-4
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_length = 3
    batch_size = 5
    input_size = 4
    hidden_size = 2
    encoder = sru.SRU(input_size, hidden_size,
                      bidirectional=bidirectional,
                      layer_norm=layer_norm,
                      normalize_after=normalize_after)
    x = torch.randn(input_length, batch_size, input_size)

    # backward in CPU mode
    h, c = encoder(x)
    h.sum().backward()
    grads = [p.grad.clone() for p in encoder.parameters() if p.requires_grad]

    # backward in GPU mode
    encoder.zero_grad()
    encoder, x = encoder.cuda(), x.cuda()
    h_, c_ = encoder(x)
    h_.sum().backward()
    grads_ = [p.grad.cpu().clone() for p in encoder.parameters() if p.requires_grad]

    assert len(grads) == len(grads_)
    for g1, g2 in zip(grads, grads_):
        assert (g1 - g2).abs().max() <= eps


@pytest.mark.parametrize(
    "projection_size,expected_transform_module",
    [
        (0, (nn.Linear, nn.Linear, nn.Linear)),
        (2, (nn.Sequential, nn.Sequential, nn.Sequential)),
        (3, (nn.Sequential, nn.Sequential, nn.Sequential)),
        ((0, 2, 3), (nn.Linear, nn.Sequential, nn.Sequential)),
    ]
)
def test_projection(projection_size, expected_transform_module):
    num_layers = 3

    sru = SRU(2, 3, num_layers=num_layers, projection_size=projection_size)
    assert len(sru.rnn_lst) == 3
    for i in range(num_layers):
        assert isinstance(sru.rnn_lst[i].transform_module, expected_transform_module[i])


@pytest.mark.parametrize(
    "attn_every_n_layers,expected_transform_module",
    [
        (1, (SRUppAttention, SRUppAttention)),
        (2, (SRUppProjectedLinear, SRUppAttention)),
    ]
)
def test_srupp_creation(attn_every_n_layers, expected_transform_module):
    num_layers = 2

    srupp = SRUpp(2, 2, 3, num_layers=num_layers,
                  attention_every_n_layers=attn_every_n_layers)
    assert len(srupp.rnn_lst) == num_layers
    for i in range(num_layers):
        assert isinstance(srupp.rnn_lst[i].transform_module, expected_transform_module[i])


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
@pytest.mark.parametrize("with_grad", [False, True])
@pytest.mark.parametrize("compat", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_srupp(cuda, with_grad, compat, bidirectional, layer_norm):
    torch.manual_seed(123)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run():
        eps = 1e-4
        num_sentences = 3
        embedding_size = 7
        rnn_proj = 2
        rnn_hidden = 4
        max_len = 4
        layers = 5
        encoder = SRUpp(
            embedding_size,
            rnn_hidden,
            rnn_proj,
            layers,
            bidirectional=bidirectional,
            layer_norm=layer_norm,
            nn_rnn_compatible_return=compat,
        )
        words_embeddings = torch.rand(
            (max_len, num_sentences, embedding_size), dtype=torch.float32
        )
        if cuda:
            words_embeddings = words_embeddings.to("cuda")
            encoder.cuda()
        encoder.eval()
        hidden, cell, _ = encoder(words_embeddings)

        def cell_to_emb(cell, batch_size):
            if compat:
                # should arrive as:
                # (num_layers * num_directions, batch, hidden_size)
                cell = cell.view(
                    layers, 2 if bidirectional else 1, batch_size, rnn_hidden
                )
                cell = cell[-1].transpose(0, 1)
                # (batch, num_directions, hidden_size)
                cell = cell.contiguous().view(batch_size, -1)
            else:
                # should arrive as:
                # (num_layers, batch_size, num_directions * hidden_size)
                cell = cell[-1].view(batch_size, -1)
            return cell

        scores = cell_to_emb(cell, num_sentences)
        for i in range(num_sentences):
            hidden, cell, _ = encoder(words_embeddings[:, i : i + 1])
            score = cell_to_emb(cell, 1)
            assert (score.detach() - scores[i].detach()).abs().max() <= eps

    if with_grad:
        run()
    else:
        with torch.no_grad():
            run()


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
@pytest.mark.parametrize("layer_norm", [False, True])
def test_srupp_backward_simple(cuda, bidirectional, layer_norm):
    torch.manual_seed(123)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    input_length = 3
    batch_size = 5
    input_size = 4
    hidden_size = 2
    encoder = sru.SRUpp(input_size, hidden_size,
                        bidirectional=bidirectional,
                        layer_norm=layer_norm)
    if cuda:
        encoder = encoder.cuda()

    def run(x):
        if cuda:
            x = x.cuda()
        output, state = encoder(x)
        output.mean().backward()

    # test batch size > 1
    input_data = torch.rand(input_length, batch_size, input_size)
    run(input_data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_srupp_backward(bidirectional, layer_norm):
    eps = 1e-4
    torch.manual_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_length = 3
    batch_size = 5
    input_size = 4
    hidden_size = 2
    encoder = sru.SRUpp(input_size, hidden_size,
                        bidirectional=bidirectional,
                        layer_norm=layer_norm)
    x = torch.randn(input_length, batch_size, input_size)

    # backward in CPU mode
    h, c = encoder(x)
    h.sum().backward()
    grads = [p.grad.clone() for p in encoder.parameters() if p.requires_grad]

    # backward in GPU mode
    encoder.zero_grad()
    encoder, x = encoder.cuda(), x.cuda()
    h_, c_ = encoder(x)
    h_.sum().backward()
    grads_ = [p.grad.cpu().clone() for p in encoder.parameters() if p.requires_grad]

    assert len(grads) == len(grads_)
    for g1, g2 in zip(grads, grads_):
        assert (g1 - g2).abs().max() <= eps