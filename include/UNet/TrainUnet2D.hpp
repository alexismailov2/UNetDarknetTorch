#pragma once

#include <opencv2/opencv.hpp>

#include <map>
#include <string>
#include <vector>

bool CountingLabeledObjects(std::map<std::string, uint32_t>& map, std::string const& polygonInfo, bool forImage = false);
auto ConvertPolygonsToMask(std::string const& polygonInfo, std::map<std::string, cv::Scalar> colorToClass) -> cv::Mat;
void ConvertPolygonsToMaskInMultipleDatasetFolders(std::vector<std::pair<std::string, std::string>> const& datasetFolderPathes,
                                                   cv::Rect const& roi = {},
                                                   std::map<std::string, cv::Scalar> const& colorToClass = {},
                                                   std::function<bool(cv::Mat const&)>&& skipPredicat = [](cv::Mat const&) -> bool { return false; });
void ConvertImagesInMultipleDatasetFolders(std::vector<std::pair<std::string, std::string>> const& datasetFolderPathes,
                                           cv::Rect const& roi = {}, bool isClahe = false, cv::Size const& newSize = {},
                                           std::function<bool(cv::Mat const&)>&& skipPredicat = [](cv::Mat const&) -> bool { return false; });
void LabelMeDeleteImage(std::string const& polygonInfo);

void runOpts(std::map<std::string, std::vector<std::string>> params);
auto ParseOptions(int argc, char *argv[]) -> std::map<std::string, std::vector<std::string>>;
