
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module = torch::jit::load(argv[1]);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({3, 2, 4}));

  auto outputs = module.forward(std::move(inputs));
  auto h = outputs.toTuple()->elements()[0].toTensor();
  auto c = outputs.toTuple()->elements()[1].toTensor();
  std::cout << h << std::endl;
  std::cout << c << std::endl;
}
