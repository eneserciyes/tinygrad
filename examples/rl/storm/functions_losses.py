from tinygrad import Tensor

def symlog(x:Tensor):
  x.requires_grad = False
  return x.sign() * (1 + x.abs()).log()

def symexp(x:Tensor):
  x.requires_grad = False
  return x.sign() * (x.abs().exp() - 1)

def symlog_loss(output: Tensor, target: Tensor):
  target = symlog(target)
  return 0.5 * (output - target).pow(2).mean()


def bucketize(x, boundaries):
  """
  Bucketize the input tensor x based on the boundaries
  Boundaries must be sorted in ascending order!
  """
  # Create a tensor of the same shape as x with the boundaries repeated
  b = boundaries.repeat((*x.shape, 1))
  d = x.unsqueeze(-1) - b
  # count the number of distances that are greater than 0
  return (d > 0).sum(-1)


class SymLogTwoHotLoss:
  def __init__(self, num_classes:int, lower_bound: float, upper_bound: float):
    self.num_classes = num_classes
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.bin_length = (upper_bound - lower_bound) / (num_classes-1)
    self.bins = Tensor.full((num_classes,), -20) + Tensor.arange(num_classes) * (40 / (num_classes-1))
    self.bins.requires_grad = False

  def __call__(self, output: Tensor, target: Tensor):
    target = symlog(target)
    assert target.min() > self.lower_bound and target.max() <= self.upper_bound

    index = bucketize(target, self.bins)
    diff = target - self.bins[index-1] # get lower bound
    weight = diff / self.bin_length
    weight = weight.clip(0, 1).unsqueeze(-1)

    target_prob = (1-weight) * (index-1).one_hot(self.num_classes) + weight * index.one_hot(self.num_classes)
    return (-target_prob.detach() * output.log_softmax(-1)).sum(-1).mean()

  def decode(self, output: Tensor):
    return symexp(output.softmax(-1) @ self.bins)


