"""
test amp

these tests are designed to run on a gpu
"""
import pytest
import sru as sru_lib
from typing import Any
import time
import torch
from torch import optim
import contextlib


class NullGradScalar(object):
    def scale(self, loss):
        return loss

    def step(self, opt):
        opt.step()

    def update(self):
        pass


class NullAmp(object):
    @contextlib.contextmanager
    def autocast(self):
        yield

    def GradScaler(self):
        return NullGradScalar()


def sync(device: str) -> None:
    if device == 'cuda':
        torch.cuda.synchronize()


has_amp = torch.cuda.is_available() and 'amp' in torch.cuda.__dict__


@pytest.mark.skipif(not has_amp, reason='AMP not available')
@pytest.mark.parametrize(
    'use_amp,fp16_recurrence', [
        [False, False],
        [False, False],
        [True, False],
        [True, True]]
)
def test_amp(use_amp: bool, fp16_recurrence: bool):
    its = 20
    warmup = 3
    batch_size = 200
    seq_len = 304
    hidden_size = 304

    input_size = hidden_size
    device = 'cuda'

    inputs = torch.rand(seq_len, batch_size, input_size, device=device)
    cell = sru_lib.SRUCell(
        input_size, hidden_size, amp_recurrence_fp16=fp16_recurrence).to(device)
    opt = optim.Adam(lr=0.02, params=cell.parameters())

    if use_amp:
        amp : Any = torch.cuda.amp
    else:
        amp = NullAmp()
    scaler = amp.GradScaler()

    sync(device)
    for it in range(its + warmup):
        if it == warmup:
            sync(device)
            start_time = time.time()
        with amp.autocast():
            out = cell(inputs)
            s = out[0].sum()

        opt.zero_grad()
        scaler.scale(s).backward()
        scaler.step(opt)
        scaler.update()
    sync(device)
    elapsed = time.time() - start_time
    print('elapsed %.3f' % elapsed)
