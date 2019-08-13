#include <torch/extension.h>
#include <vector>

torch::Tensor test_forward_cpu(const torch::Tensor& inputA,
                               const torch::Tensor& inputB);
std::vector<torch::Tensor> test_backward_cpu(const torch::Tensor& grad_output);

torch::Tensor test_forward_cpu(const torch::Tensor& x,
                               const torch::Tensor& y){
  AT_ASSERTM(x.sizes()==y.sizes(), "x must be the same size with y.");
  torch::Tensor z = torch::zeros(x.sizes());
  z = 2*x +y;
  return z;
}

std::vector<torch::Tensor> test_backward_cpu(const torch::Tensor& grad_output){
  torch::Tensor grad_x = 2* torch::ones(grad_output.sizes());
  torch::Tensor  grad_y = torch::ones(grad_output.sizes());
  return {grad_x, grad_y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("forward", &test_forward_cpu,"This is a forward test.");
  m.def("backward", &test_backward_cpu, "That is a backward test.");
}
