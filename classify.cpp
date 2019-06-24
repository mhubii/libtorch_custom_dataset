#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "model.h"

int main(int arc, char** argv)
{
    std::string loc = argv[1];

    // Load image with OpenCV.
    cv::Mat img = cv::imread(loc);

    // Convert the image and label to a tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW
    img_tensor = img_tensor.to(torch::kF32);

    // Load the model.
    ConvNet model(3/*channel*/, 64/*height*/, 64/*width*/);
    torch::load(model, "../best_model.pt");

    // Predict the probabilities for the classes.
    torch::Tensor prob = model(img_tensor);
    std::cout << prob << std::endl;

    return 0;
}