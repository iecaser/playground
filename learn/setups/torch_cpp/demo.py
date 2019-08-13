from test import Test
from loguru import logger
import torch

test = Test()
x = torch.autograd.Variable(torch.Tensor([1, 2, 3]), requires_grad=True)
y = torch.autograd.Variable(torch.Tensor([4, 5, 6]), requires_grad=True)
z = test(x, y).double()
z.sum().backward()
logger.info(x)
logger.info(y)
logger.info(z)
logger.info(x.grad)
logger.info(y.grad)
