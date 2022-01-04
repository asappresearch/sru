import torch


# Run at the beginning of the test suites to ensure no previous use of SRUCells
def test_no_eager_cuda_init():
    # Notice the test is expected to pass both with GPU available and without it
    import sru
    assert not torch.cuda.is_initialized()