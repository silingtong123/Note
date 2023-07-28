#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <chrono>
int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
//  inputs.push_back(torch::ones({1, 3, 224, 224}).to(at::kCUDA));

   // Execute the model and turn its output into a tensor.
//   module.to(at::kCUDA);

   inputs.push_back(torch::ones({1, 3, 224, 224}));
   at::Tensor output;
   for(int i =0; i < 2; ++i)//warm up
          output = module.forward(inputs).toTensor();
   auto start = std::chrono::high_resolution_clock::now();
   for(int i=0; i <100; ++i)
           output = module.forward(inputs).toTensor();
   auto end = std::chrono::high_resolution_clock::now();
   auto total_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
   std::cout<<"cost time ------"<<total_elapsed_time<<std::endl;
   std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
/*
GPU infer:
input移到到GPU: inputs.push_back(torch::ones({1, 3, 224, 224}).to(at::kCUDA));
model移到到GPU: module.to(at::kCUDA);

A100机器 单卡：cost time ------116ms
A100机器 CPU： cost time ------715ms

run shell:
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
./example-app /path/traced_resnet_model.pt
*/