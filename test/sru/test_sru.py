import pytest
from sru import SRU
import torch
from torch import nn


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
def test_cell(cuda, with_grad, compat):
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
        bidirectional = True
        encoder = SRU(
            embedding_size,
            rnn_hidden,
            layers,
            bidirectional=bidirectional,
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
    "projection_size,expected_custom_m",
    [
        (0, (nn.Linear, nn.Linear, nn.Linear)),
        (2, (nn.Sequential, nn.Sequential, nn.Sequential)),
        (3, (nn.Sequential, nn.Sequential, nn.Sequential)),
        ((0, 2, 3), (nn.Linear, nn.Sequential, nn.Sequential)),
    ]
)
def test_projection(projection_size, expected_custom_m):
    num_layers = 3

    sru = SRU(2, 3, num_layers=num_layers, projection_size=projection_size)
    assert len(sru.rnn_lst) == 3
    for i in range(num_layers):
        assert isinstance(sru.rnn_lst[i].custom_m, expected_custom_m[i])
