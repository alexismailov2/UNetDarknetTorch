#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include <torch/torch.h>

#include <string>
#include <vector>

class UNetDataset
  : public torch::data::datasets::Dataset<UNetDataset>
{
public:
    UNetDataset(std::vector<std::tuple<std::string, std::string>> const& datasetDirsPath,
                std::vector<cv::Scalar> const& classes,
                cv::Size size = {256, 256},
                bool grayscale = false);
    auto get(size_t index) -> torch::data::Example<> override;
    auto size() const -> c10::optional<size_t> override;

private:
    std::vector<std::tuple<std::string, std::string>> _datasetDirsPath;
    cv::Size                                          _size;
    std::vector<std::pair<std::string, std::string>>  _imagesAndMasks;
    std::vector<cv::Scalar>                           _classColors;
    std::vector<cv::Mat>                              _image;
    std::vector<cv::Mat>                              _mask;
    std::vector<torch::data::Example<>>               _data;
};

auto TestDatasetCreate(cv::Size const& size, std::vector<cv::Scalar> const& classes) -> std::pair<cv::Mat, cv::Mat>;