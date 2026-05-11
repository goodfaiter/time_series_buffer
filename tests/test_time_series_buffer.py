import os
import sys
import torch

# Ensure `src` is on sys.path so the package can be imported when running tests from
# the repository root or the package directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from time_series_buffer.time_series_buffer import TimeSeriesBuffer


def test_add_and_get_basic():
    num_envs = 2
    dim = 3
    max_size = 4
    stride = 1
    buf = TimeSeriesBuffer(num_envs, dim, max_size, stride, device='cpu')

    # Add a first value and verify get() returns that value in the most-recent slot
    v0 = torch.arange(num_envs * dim, dtype=torch.float32).view(num_envs, dim)
    buf.add(v0)

    out = buf.get()
    assert out.shape == (num_envs, max_size // stride if stride > 0 else max_size, dim)

    # After a single add the newest entry should appear first for each env
    # and the rest should be zeros
    first_frame = out[:, 0, :]
    assert torch.allclose(first_frame, v0)


def test_wrap_around_behavior():
    num_envs = 1
    dim = 2
    max_size = 3
    stride = 1
    buf = TimeSeriesBuffer(num_envs, dim, max_size, stride, device='cpu')

    # Add values to fill and then wrap
    buf.add(torch.tensor([[1.0, 1.0]]))  # write index -> 1
    buf.add(torch.tensor([[2.0, 2.0]]))  # write index -> 2
    buf.add(torch.tensor([[3.0, 3.0]]))  # write index -> 0 (wrap)
    buf.add(torch.tensor([[4.0, 4.0]]))  # overwrites position 0

    out = buf.get()
    # Expect the most recent entries in order (newest first)
    newest = out[0, 0, :]
    assert torch.allclose(newest, torch.tensor([4.0, 4.0]))


def test_reset_clears_buffer_and_indices():
    num_envs = 2
    dim = 2
    max_size = 4
    stride = 1
    buf = TimeSeriesBuffer(num_envs, dim, max_size, stride, device='cpu')

    buf.add(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    buf.add(torch.tensor([[3.0, 3.0], [4.0, 4.0]]))

    # Reset the second environment
    reset_mask = torch.tensor([False, True])
    buf.reset(reset_mask)

    # The buffer rows for the second env should be zeros and its write index reset
    out = buf.get()
    assert torch.allclose(out[1], torch.zeros_like(out[1]))
    assert int(buf.write_indices[1].item()) == 0


def test_partial_reset_three_envs_indices_mismatch():
    num_envs = 3
    dim = 2
    max_size = 4
    stride = 1
    buf = TimeSeriesBuffer(num_envs, dim, max_size, stride, device='cpu')

    # Add two frames so write_indices advance
    v0 = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    v1 = torch.tensor([[1.0, 1.0], [11.0, 11.0], [21.0, 21.0]])
    buf.add(v0)
    buf.add(v1)

    # Reset the middle environment (index 1)
    reset_mask = torch.tensor([False, True, False])
    buf.reset(reset_mask)

    # Add another frame after reset; this will make write_indices differ
    v2 = torch.tensor([[2.0, 2.0], [12.0, 12.0], [22.0, 22.0]])
    buf.add(v2)

    out = buf.get()

    # For the reset env (1): newest entry should be v2 and older slots should be zeros
    assert torch.allclose(out[1, 0, :], torch.tensor([12.0, 12.0]))
    assert torch.allclose(out[1, 1:, :], torch.zeros_like(out[1, 1:, :]))

    # For a non-reset env (0): newest should be v2 and previous entries include v1 and v0
    assert torch.allclose(out[0, 0, :], torch.tensor([2.0, 2.0]))
    # Ensure at least one older entry matches a previous add
    older_matches = (out[0, 1:, :] == torch.tensor([1.0, 1.0])).any()
    assert older_matches
