import torch
import test_cpp


class TestFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return test_cpp.forward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_y = test_cpp.backward(grad_output)
        return grad_x, grad_y


class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return TestFunction.apply(x, y)
