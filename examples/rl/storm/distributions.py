from typing import Tuple
from tinygrad import Tensor

class Categorical:
  def __init__(self, logits: Tensor) -> None:
    self.logits = logits
    self.probs = self.logits.exp() / self.logits.exp().sum(-1, keepdim=True)

  def onehot_sample(self) -> Tensor:
    s: Tensor = (self.probs.cumsum(-1) > Tensor.rand(*self.probs.shape[:-1], 1)).argmax(-1)
    return s.one_hot(s.shape[-1])

  def sample(self) -> Tensor:
    return (self.probs.cumsum(-1) > Tensor.rand(*self.probs.shape[:-1], 1)).argmax(-1)


def kl_categorical_with_free_bits(p: Tensor, q: Tensor, free_bits: float) -> Tuple[Tensor, Tensor]:
  real_kl_div = (p.softmax() * (p - q)).sum((-1, -2)).mean() #TODO: check for probability 0
  kl_div = Tensor.max(real_kl_div, Tensor.ones_like(real_kl_div) * free_bits)
  return kl_div, real_kl_div
