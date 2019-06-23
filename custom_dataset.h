#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"

namespace torch {
namespace data {
namespace datasets {

class CustomDataset : public Dataset<CustomDataset>
{
    private:
        std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;
int i = 0;
    public:
        explicit CustomDataset(std::string& file_names_csv)
            // Load csv file with file locations and labels.
            : csv_(ReadCsv(file_names_csv)) {

        };

        // Override the get method to load custom data.
        Example<> get(size_t index) override {
std::cout << i << std::endl;
i++;
std::cout << "index: " << index << std::endl;
std::cout << std::get<0>(csv_[index]) << std::endl;
            std::string file_location = std::get<0>(csv_[index]);
            int64_t label = std::get<1>(csv_[index]);

            // Load image with OpenCV.
            cv::Mat img = cv::imread(file_location);

            // Convert the image and label to a tensor.
            Tensor img_tensor = from_blob(img.data, {1, img.rows, img.cols, 3});
            img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW

            Tensor label_tensor = full({1, 1}, label);

            return {img_tensor, label_tensor};
        };

        // Override the size method to infer the size of the data set.
        optional<size_t> size() const override {

            return csv_.size();
        };
};

} // end of namesapce datasets
} // end of namespace data
} // end of namespace torch
