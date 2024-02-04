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


def kl_categorical(p: Categorical, q: Categorical) -> Tensor:
  # TODO: check for probability 0
  return (p.probs * (p.logits - q.logits)).sum(-1)

class CategoricalKLDivLossWithFreeBits:
  def __init__(self, free_bits) -> None:
    self.free_bits = free_bits

  def __call__(self, p_logits: Tensor, q_logits: Tensor) -> Tuple[Tensor, Tensor]:
    p_dist = Categorical(p_logits)
    q_dist = Categorical(q_logits)
    kl_div = kl_categorical(p_dist, q_dist).sum(-1).mean()
    real_kl_div = kl_div
    kl_div = Tensor.max(kl_div, Tensor.ones_like(kl_div) * self.free_bits)
    return kl_div, real_kl_div
