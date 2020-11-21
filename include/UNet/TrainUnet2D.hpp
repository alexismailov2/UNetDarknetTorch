#pragma once

#include <opencv2/opencv.hpp>

#include <map>
#include <string>
#include <vector>

bool CountingLabeledObjects(std::map<std::string, uint32_t>& map, std::string const& polygonInfo, bool forImage = false);
auto ConvertPolygonsToMask(std::string const& polygonInfo, std::map<std::string, cv::Scalar> colorToClass) -> cv::Mat;

void runOpts(std::map<std::string, std::vector<std::string>> params);
auto ParseOptions(int argc, char *argv[]) -> std::map<std::string, std::vector<std::string>>;