#pragma once

#include <torch/torch.h>

struct ConvNetImpl : public torch::nn::Module 
{
    ConvNetImpl(int64_t channels, int64_t height, int64_t width) 
        : conv1(torch::nn::Conv2dOptions(3 /*input channels*/, 8 /*output channels*/, 5 /*kernel size*/).stride(2)),
          conv2(torch::nn::Conv2dOptions(8, 16, 5).stride(2)),
          conv3(torch::nn::Conv2dOptions(16, 32, 3).stride(2)),
          
          lin1(GetConvOutput(channels, height, width), 64),
          lin2(64, 32),
          lin3(32, 1) {

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);

        register_module("lin1", lin1);
        register_module("lin2", lin2);
        register_module("lin3", lin3);
    };

    // Implement the forward method.
    torch::Tensor forward(torch::Tensor x) {

        x = torch::relu(conv1(x));
        x = torch::relu(conv2(x));
        x = torch::relu(conv3(x));

        x = torch::relu(lin1(x));
        x = torch::relu(lin2(x));
        x = torch::relu(lin3(x));

        return std::get<1>(x.max(1)).view({1,1});
    };

    // Get number of elements of output.
    int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {

        torch::Tensor x = torch::zeros({1, channels, height, width});
        x = conv1(x);
        x = conv2(x);
        x = conv3(x);

        return x.numel();
    }

    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::Linear lin1, lin2, lin3;
};

TORCH_MODULE(ConvNet);