import unittest
import numpy as np

from nets import ResnetBlock
from tinygrad import Tensor


class TestResnetBlock(unittest.TestCase):
  def test_forward(self):
    breakpoint()
    res_block = ResnetBlock(in_channels=32, out_channels=32, temb_channels=0, dropout=0.)
    x = Tensor(np.load("test_files/resblock_in.npy"))
    y = res_block(x, None)
    y_expected = np.load("test_files/resblock_out_same_channels.npy")
    np.testing.assert_allclose(y.numpy(), y_expected, atol=1e-5)


if __name__ == '__main__':
  unittest.main()
