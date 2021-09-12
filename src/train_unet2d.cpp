#include <UNet/UNetDataset.hpp>
#include <UNet/DarknetParser.hpp>

#include <torch/script.h>

#ifdef _MSC_VER
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <boost/algorithm/string/split.hpp>

auto diceLoss(torch::Tensor predict, torch::Tensor masks, double smooth = 1.0)
{
    predict = predict.contiguous();
    masks   = masks.contiguous();

    auto intersection = (predict * masks).sum(2).sum(2);
    auto loss = (1. - ((2. * intersection + smooth) / (predict.sum(2).sum(2) + masks.sum(2).sum(2) + smooth)));

    return loss.mean();
}

auto calcLoss(torch::Tensor const& predict, torch::Tensor const& masks, std::map<std::string, float>& metrics,
              double bce_weight = 0.5) -> torch::Tensor
{
    using namespace torch::nn::functional;
    auto bce  = torch::binary_cross_entropy_with_logits(predict, masks);
    auto dice = diceLoss(predict, masks);
    auto loss = bce * bce_weight + dice * (1 - bce_weight);

    metrics["bce"]  += bce.cpu().template item<float>();
    metrics["dice"] += dice.cpu().template item<float>();
    metrics["loss"] += loss.cpu().template item<float>();

    return loss;
}

void printMetrics(std::map<std::string, float>& metrics, size_t datasetSize)
{
   std::cout << "bce: " << metrics["bce"]/datasetSize << ", dice: " << metrics["dice"]/datasetSize << ", loss: " << metrics["loss"]/datasetSize << std::endl;
}

auto toClassesMapsThreshold(cv::Mat const& score,
                            cv::Size const& inputSize,
                            std::vector<float> threshold) -> std::vector<cv::Mat>
{
   auto const rows = score.size[3];
   auto const cols = score.size[2];
   auto const channels = score.size[1];
   auto const batches = score.size[0];
   auto step0 = score.step.p[0];
   auto classesMaps = std::vector<cv::Mat>(static_cast<size_t>(channels) * batches);
   for (auto bt = 0; bt < batches; ++bt)
   {
       for (auto ch = 0; ch < channels; ++ch)
       {
           cv::Mat channelScore = cv::Mat(rows, cols, CV_32FC1, const_cast<float *>(score.ptr<float>(bt, ch, 0)));
           cv::inRange(channelScore, threshold[ch], 1000000.0, classesMaps[ch + (bt*channels)]);
           if ((inputSize.width != 0) && (inputSize.height != 0))
           {
               cv::resize(classesMaps[ch], classesMaps[ch], inputSize);
           }
       }
   }
   return classesMaps;
}

template <typename DataLoader>
void train(size_t epoch, Darknet& model, torch::Device device, DataLoader& data_loader, size_t size, torch::optim::Optimizer& optimizer, std::function<void(int32_t, int32_t)>&& step = [](int32_t, int32_t){})
{
   std::cout << "=======================" << std::endl;
   std::cout << "Epoch: " << epoch << std::endl;
   std::map<std::string, float> metrics;

   std::cout << "device:" << device << std::endl;
   model->train();
   // TODO: std::distance(data_loader.begin(), data_loader.end()); - Very very slow
   auto const count = size; //std::distance(data_loader.begin(), data_loader.end());
   auto current = 0;
   for (auto& batch : data_loader)
   {
      torch::Tensor data = batch.data.to(device);
      auto targets = batch.target.to(device);

      optimizer.zero_grad();
      auto output = model->forward(data);
      auto loss = calcLoss(output, targets, metrics);
      loss.backward();
      optimizer.step();
      step(count, current++);
   }
   printMetrics(metrics, size/*std::distance(data_loader.begin(), data_loader.end())*/);
}

template <typename DataLoader>
void valid(Darknet& model,
           torch::Device device,
           DataLoader& data_loader,
           size_t size,
           std::function<void(torch::Tensor&, torch::Tensor&)>&& handler = [](torch::Tensor&, torch::Tensor&){},
           std::function<void(std::map<std::string, float>)>&& metricsHandler = [](auto&&){})
{
   std::cout << "-----------------------" << std::endl;
   std::cout << "Validation: " << std::endl;
   std::map<std::string, float> metrics;

   torch::NoGradGuard no_grad;
   model->eval();
   for (const auto& batch : data_loader)
   {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);
      auto output = model->forward(data);

      calcLoss(output, targets, metrics);
      handler(output, targets);
   }
   // TODO: std::distance(data_loader.begin(), data_loader.end()); - very slow
   auto datasetSize = size; //std::distance(data_loader.begin(), data_loader.end());
   printMetrics(metrics, datasetSize);
   metrics["bce"]/=datasetSize;
   metrics["dice"]/=datasetSize;
   metrics["loss"]/=datasetSize;
   metricsHandler(metrics);
   std::cout << "=======================" << std::endl;
}

auto toColorMask(std::vector<cv::Mat> const& masks, std::vector<cv::Scalar> const& colors) -> cv::Mat
{
   cv::Mat coloredMasks(masks[0].rows, masks[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));
   for (size_t i = 0; i < masks.size(); ++i)
   {
      coloredMasks.setTo(colors[i], masks[i]);
   }
   return coloredMasks;
}

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <random>

bool ConvertingXmlLabelmeToJsonLabelme(std::string const& polygonInfo, std::string const& prefixPath)
{
    auto file = fs::path(polygonInfo).filename().string();
    std::cout << file << std::endl;
    auto dir = fs::path(polygonInfo).remove_filename().string() + "../Data/";
    fs::create_directories(dir);
    boost::property_tree::ptree pt;
    try
    {
        boost::property_tree::read_xml(polygonInfo, pt);
    }
    catch (boost::property_tree::xml_parser::xml_parser_error& ex)
    {
        std::cout << ex.message() << std::endl;
        return false;
    }
    boost::property_tree::ptree out;
    out.put("version", "4.5.6");
    out.put("flags", "{}");
    out.put("imagePath", "");
    out.put("imageData", "null");
    out.put("imageHeight", "0");
    out.put("imageWidth", "0");

    std::string imagePath;
    std::string imageDir;
    std::string imageFile;
    boost::property_tree::ptree ptShapes;
    for (auto const& annotation : pt.get_child("annotation"))
    {
        if (annotation.first == "object")
        {
#if 0
            //    {
//        "shapes": [
//        {
//            "label": "0_cable",
//            "points": [
//            [
//                "0.0",
//                "280.73394495412845"
//            ],
//            [
//                "748.6238532110091",
//                "285.3211009174312"
//            ],
//            [
//                "1446.788990825688",
//                "318.348623853211"
//            ],
//            [
//                "1445.8715596330273",
//                "556.880733944954"
//            ],
//            [
//                "0.0",
//                "504.58715596330273"
//            ]
//            ],
//            "group_id": "null",
//            "shape_type": "polygon",
//            "flags": ""
//        }
//        ],
//    }
#endif
            boost::property_tree::ptree ptShape;
            for (auto const& object : annotation.second.get_child(""))
            {
                if (object.first == "polygon")
                {
                    boost::property_tree::ptree ptPolygon;
                    for (auto const& polygon : object.second.get_child(""))
                    {
                        if (polygon.first == "pt")
                        {
                            boost::property_tree::ptree ptPoints;
                            boost::property_tree::ptree ptItemX;
                            boost::property_tree::ptree ptItemY;
                            auto x = polygon.second.get<float>("x");
                            auto y = polygon.second.get<float>("y");
                            ptItemX.put<float>("", x);
                            ptItemY.put<float>("", y);
                            ptPoints.push_back(std::make_pair("", ptItemX));
                            ptPoints.push_back(std::make_pair("", ptItemY));
                            ptPolygon.push_back(std::make_pair("", ptPoints));
                            //std::cout << polygon.first << ": " << x << ", " << y << std::endl;
                        } else if (polygon.first == "username") {
                           // std::cout <<  object.first << ": " << "<skip>" << std::endl;
                        }
                    }
                    ptShape.add_child("points", ptPolygon);
                } else if (object.first == "name") {
                    auto name = object.second.get_value<std::string>();
                    ptShape.put<std::string>("label", name);
                    //std::cout << object.first << ": " << name << std::endl;
                } else if (object.first == "deleted") {
                    auto deleted = object.second.get_value<bool>();
                    //std::cout << object.first << ": " << deleted << std::endl;
                } else if (object.first == "verified") {
                    auto verified = object.second.get_value<bool>();
                    //std::cout << object.first << ": " << verified << std::endl;
                } else if (object.first == "occluded") {
                    auto occluded = object.second.get_value<std::string>();
                    //std::cout << object.first << ": " << occluded << std::endl;
                } else if (object.first == "id") {
                    auto id = object.second.get_value<uint32_t>();
                    //std::cout << object.first << ": " << id << std::endl;
                } else {
                   // std::cout <<  object.first << ": " << "<skip>" << std::endl;
                }
            }
            ptShape.put<std::string>("group_id", "null");
            ptShape.put<std::string>("shape_type", "polygon");
            ptShape.put<std::string>("flags", "{}");
            ptShapes.push_back(std::make_pair("", ptShape));
        } else if (annotation.first == "folder") {
            imageDir = annotation.second.get_value<std::string>();
            imagePath = prefixPath + ((!prefixPath.empty()) ? "\/" : "") + imageDir + ((!imageDir.empty()) ? "\/" : "") + imageFile;
            out.put("imagePath", imagePath);
            //std::cout << annotation.first << ": " << imageDir << std::endl;
        } else if (annotation.first == "filename") {
            imageFile = annotation.second.get_value<std::string>();
            //std::cout << annotation.first << ": " << imageFile << std::endl;
        } else if (annotation.first == "imagesize") {
            auto imageHeight = annotation.second.get<uint32_t>("nrows");
            auto imageWidth = annotation.second.get<uint32_t>("ncols");
            out.put("imageHeight", imageHeight);
            out.put("imageWidth", imageWidth);
            //std::cout << annotation.first << ": " << imageWidth << ", " << imageHeight << std::endl;
        } else {
           // std::cout << annotation.first << ": " << " <skiped>" << std::endl;
        }
    }
    auto dataFile = fs::path(imageFile).replace_extension(".json").string();
    out.add_child("shapes",ptShapes);
    boost::property_tree::write_json(dir + "/" + dataFile, out);
    return true;
}

void ConvertingXmlLabelmeToJsonLabelmeInFolder(std::string const& directory, std::string const& prefixPath)
{
    for (auto const& file : fs::directory_iterator(directory))
    {
        ConvertingXmlLabelmeToJsonLabelme(file.path().string(), prefixPath);
    }
}

bool CountingLabeledObjects(std::map<std::string, uint32_t>& map, std::string const& polygonInfo, bool forImage = false)
{
   boost::property_tree::ptree pt;
   try
   {
      boost::property_tree::read_json(polygonInfo, pt);
   }
   catch (boost::property_tree::json_parser::json_parser_error& ex)
   {
      return false;
   }

   std::set<std::string> classesSetForImage;
   auto shapes = pt.get_child("shapes");
   for (auto const& shape : shapes)
   {
      auto label = shape.second.get<std::string>("label");
      if (forImage)
      {
         classesSetForImage.insert(label);
      }
      else
      {
         map[label]++;
      }
   }
   for (auto const& item : classesSetForImage)
   {
      map[item]++;
   }
   return true;
}

void LabelMeDeleteImage(std::string const& polygonInfo)
{
  boost::property_tree::ptree pt;
  try
  {
    boost::property_tree::read_json(polygonInfo, pt);
    pt.put<std::string>("imageData", {});
    boost::property_tree::write_json(polygonInfo, pt);
  }
  catch (boost::property_tree::json_parser::json_parser_error& ex)
  {
    return;
  }
}

auto ConvertPolygonsToMask(std::string const& polygonInfo, std::map<std::string, cv::Scalar> colorToClass) -> cv::Mat
{
   boost::property_tree::ptree pt;
   try
   {
     boost::property_tree::read_json(polygonInfo, pt);
   }
   catch (boost::property_tree::json_parser::json_parser_error& ex)
   {
     return {};
   }

   cv::Mat mask = cv::Mat::zeros(pt.get<int>("imageHeight"), pt.get<int>("imageWidth"), CV_8UC3);
   auto shapes = pt.get_child("shapes");
   for (auto const& shape : shapes)
   {
      auto points = shape.second.get_child("points");
      std::vector<cv::Point> cv_points;
      for (auto const& point : points)
      {
         std::vector<float> vecxy;
         for (const auto& pointxy : point.second) {
            vecxy.emplace_back(pointxy.second.get_value<float>());
         }
         cv_points.emplace_back(cv::Point{static_cast<int>(vecxy[0]), static_cast<int>(vecxy[1])});
      }
      auto label = shape.second.get<std::string>("label");
      auto color = colorToClass[label];
      cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{cv_points}, color);
   }
   return mask.clone();
}

auto ConvertPolygonsToRects(std::string const& polygonInfo, bool isUnionRectNeededByClass = false) -> std::pair<cv::Size, std::map<std::string, std::vector<cv::Rect>>>
{
    std::map<std::string, std::vector<cv::Rect>> rectsMap;
    boost::property_tree::ptree pt;
    try
    {
        boost::property_tree::read_json(polygonInfo, pt);
    }
    catch (boost::property_tree::json_parser::json_parser_error& ex)
    {
        return std::make_pair(cv::Size{}, rectsMap);
    }
    auto shapes = pt.get_child("shapes");
    for (auto const& shape : shapes)
    {
        auto points = shape.second.get_child("points");
        std::vector<cv::Point> cv_points;
        for (auto const& point : points)
        {
            std::vector<float> vecxy;
            for (const auto& pointxy : point.second) {
                vecxy.emplace_back(pointxy.second.get_value<float>());
            }
            cv_points.emplace_back(cv::Point{static_cast<int>(vecxy[0]), static_cast<int>(vecxy[1])});
        }
        auto const label = shape.second.get<std::string>("label");
        if (isUnionRectNeededByClass)
        {
            if (rectsMap[label].empty())
            {
                rectsMap[label].emplace_back(cv::boundingRect(std::vector<cv::Point>{cv_points}));
            }
            else
            {
                rectsMap[label].front() |= cv::boundingRect(std::vector<cv::Point>{cv_points});
            }
        }
        else
        {
            rectsMap[label].emplace_back(cv::boundingRect(std::vector<cv::Point>{cv_points}));
        }
    }
    return std::make_pair(cv::Size(pt.get<int>("imageWidth"), pt.get<int>("imageHeight")), rectsMap);
}

void ConvertPolygonsToBoundingBoxesYoloDarknet(std::vector<std::pair<std::string, std::string>> const& datasetFolderPathes,
                                               std::vector<std::string> const& labels,
                                               std::string const& outputFolder = "G:/Datasets/FOLDER_WITHOUT_CRAZY_SPACEBARS/Biggest_Dataset")
{
    auto trainList = std::vector<std::string>{};
    for (auto datasetFolderPath : datasetFolderPathes)
    {
        fs::create_directories(datasetFolderPath.second + "/labels");
        for (auto file : fs::directory_iterator(datasetFolderPath.first))
        {
            if (fs::is_directory(file))
            {
                continue;
            }
            auto imageFileName = file.path().filename();
            imageFileName.replace_extension(".bmp");
            trainList.emplace_back(datasetFolderPath.second + "/labels/" + imageFileName.string());
            auto labelFileName = file.path().filename();
            labelFileName.replace_extension(".txt");
            auto yoloDarknetFile = std::ofstream(datasetFolderPath.second + "/labels/" + labelFileName.string());
            auto rects = ConvertPolygonsToRects(file.path().string());
            for (auto i = 0; i < labels.size(); ++i)
            {
                for (auto const& item : rects.second[labels[i]])
                {
                    yoloDarknetFile << i << " "
                                    << (static_cast<float>(item.x + item.width/2))/rects.first.width << " "
                                    << (static_cast<float>(item.y + item.height/2))/rects.first.height << " "
                                    << (static_cast<float>(item.width)/rects.first.width) << " "
                                    << (static_cast<float>(item.height)/rects.first.height) << std::endl;
                }
            }
        }
        //darknet.exe detector calc_anchors "G:/Datasets/FOLDER_WITHOUT_CRAZY_SPACEBARS/all_dataset_in_one/YoloData/obj.data" -num_of_clusters 9 -width 416 -height 416
        //darknet.exe detector train "G:/Datasets/FOLDER_WITHOUT_CRAZY_SPACEBARS/all_dataset_in_one/YoloData/obj.data" "G:/Datasets/FOLDER_WITHOUT_CRAZY_SPACEBARS/all_dataset_in_one/YoloData/yolov4-custom.cfg" ""G:/Datasets/FOLDER_WITHOUT_CRAZY_SPACEBARS/all_dataset_in_one/YoloData/yolov4.conv.137" -map
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(trainList.begin(), trainList.end(), g);
    auto trainListFile = std::ofstream(outputFolder + "/trainList.txt");
    auto iteratorSplit = std::prev(trainList.cend(), (trainList.size() / 10));
    std::copy(trainList.cbegin(), iteratorSplit, std::ostream_iterator<std::string>(trainListFile, "\n"));
    auto validListFile = std::ofstream(outputFolder + "/validList.txt");
    std::copy(iteratorSplit, trainList.cend(), std::ostream_iterator<std::string>(validListFile, "\n"));
    auto namesFile = std::ofstream(outputFolder + "/obj.names");
    std::copy(labels.cbegin(), labels.cend(), std::ostream_iterator<std::string>(namesFile, "\n"));
    auto dataFile = std::ofstream(outputFolder + "/obj.data");
    dataFile << "classes " << labels.size() << "\n"
             << "train  = " << outputFolder + "/trainList.txt" << "\n"
             << "valid  = " << outputFolder + "/validList.txt" << "\n"
             << "names = " << outputFolder + "/obj.names" << "\n"
             << "backup = " << outputFolder + "/backup" << std::endl;
    fs::create_directories(outputFolder + "/backup");
}

void GetCroppedMaskForSelectedLabelInMultipleDatasetFolders(std::vector<std::pair<std::string, std::string>> const& datasetFolderPathes,
                                                            std::vector<std::string> const& labels,
                                                            uint32_t alignFactor = 1,
                                                            cv::Size downscale = {1, 1},
                                                            std::map<std::string, cv::Scalar> const& colorToClass = {},
                                                            uint32_t rotate = 0,
                                                            std::function<bool(cv::Mat const&)>&& skipPredicat = [](cv::Mat const&) -> bool { return false; })
{
    switch(rotate)
    {
        case 90:
            std::swap(downscale.width, downscale.height);
            break;
        case 0:
        default:
            break;
    }
    for (auto datasetFolderPath : datasetFolderPathes)
    {
        fs::create_directories(datasetFolderPath.second + "/images");
        fs::create_directories(datasetFolderPath.second + "/masks");
        for (auto file : fs::directory_iterator(datasetFolderPath.first))
        {
            if (fs::is_directory(file))
            {
                continue;
            }
            auto rects = ConvertPolygonsToRects(file.path().string(), true);
            cv::Rect cropRect;
            for(auto& label : labels)
            {
                if (rects.second[label].empty())
                {
                    continue;
                }
                cropRect |= rects.second[label].front();
            }
            cv::Mat mask = ConvertPolygonsToMask(file.path().string(), colorToClass);
            if (skipPredicat(mask))
            {
                continue;
            }
            auto filename = file.path().filename().string();
            filename = filename.substr(0, filename.size() - 4);
            cropRect.width = std::min(mask.cols - 1 - cropRect.x, cropRect.width);
            cropRect.height = std::min(mask.rows - 1 - cropRect.y, cropRect.height);
            auto widthModAlignFactor = cropRect.width % (alignFactor * downscale.width);
            auto neededWidthAddition = (alignFactor * downscale.width) - widthModAlignFactor;
            auto heightModAlignFactor = cropRect.height % (alignFactor * downscale.height);
            auto neededHeightAddition = (alignFactor * downscale.height) - heightModAlignFactor;
            if (widthModAlignFactor != 0)
            {
                if ((cropRect.width + cropRect.x + neededWidthAddition) > mask.cols)
                {
                    cropRect.width -= widthModAlignFactor;
                }
                else
                {
                    cropRect.width += neededWidthAddition;
                }
            }
            if (heightModAlignFactor != 0)
            {
                if ((cropRect.height + cropRect.y + neededHeightAddition) > mask.rows)
                {
                    cropRect.height -= heightModAlignFactor;
                }
                else
                {
                    cropRect.height += neededHeightAddition;
                }
            }
            if (cropRect.width > mask.cols) {
                std::cout << "Really?" << std::endl;
            }
            if (cropRect.height > mask.rows) {
                std::cout << "Really?" << std::endl;
            }
            //auto downscaleModX = (downscale.width > 1) ? (cropRect.width % downscale.width) : 0;
            //cropRect.width -= downscaleModX;
            //cropRect.width /= downscale.width;
            //auto downscaleModY = cropRect.y % downscale.height;
            if ((cropRect.width == 0) || (cropRect.height == 0))
            {
                std::cout << "Skipped cropping for image: " << file.path() << std::endl;
                continue;
            }
            cv::Mat maskCropped = mask(cropRect);
            cv::resize(maskCropped, maskCropped, cv::Size{maskCropped.cols / downscale.width, maskCropped.rows / downscale.height}, cv::INTER_NEAREST);
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(file.path().string(), pt);
            auto imagePath = pt.get<std::string>("imagePath");
            cv::Mat imageCropped = cv::imread(imagePath, cv::IMREAD_COLOR);
            imageCropped = imageCropped(cropRect);
            cv::resize(imageCropped, imageCropped, cv::Size{imageCropped.cols / downscale.width, imageCropped.rows / downscale.height}, cv::INTER_NEAREST);
            switch(rotate) {
                case 90:
                    cv::rotate(maskCropped, maskCropped, cv::ROTATE_90_CLOCKWISE);
                    cv::rotate(imageCropped, imageCropped, cv::ROTATE_90_CLOCKWISE);
                    break;
                case 0:
                default:
                    break;
            }
            cv::imwrite(datasetFolderPath.second + "/masks/" + filename + "png", maskCropped);
            cv::imwrite(datasetFolderPath.second + "/images/" + filename + "png", imageCropped);
        }
    }
}

auto CountingLabeledObjectsInMultipleDatasetFolders(std::vector<std::string> const& datasetFolderPathes) -> std::map<std::string, uint32_t>
{
   std::map<std::string, uint32_t> countResult;
   for (auto const& datasetFolderPath : datasetFolderPathes)
   {
      for (auto file : fs::directory_iterator(datasetFolderPath))
      {
         CountingLabeledObjects(countResult, file.path().string(), true);
      }
   }
   return countResult;
}

void ConvertPolygonsToMaskInMultipleDatasetFolders(std::vector<std::pair<std::string, std::string>> const& datasetFolderPathes,
                                                   cv::Rect const& roi = {},
                                                   std::map<std::string, cv::Scalar> const& colorToClass = {},
                                                   std::function<bool(cv::Mat const&)>&& skipPredicat = [](cv::Mat const&) -> bool { return false; })
{
   for (auto datasetFolderPath : datasetFolderPathes)
   {
      fs::create_directories(datasetFolderPath.second);
      for (auto file : fs::directory_iterator(datasetFolderPath.first))
      {
         if (fs::is_directory(file))
         {
           continue;
         }
         cv::Mat mask = ConvertPolygonsToMask(file.path().string(), colorToClass);
         if (skipPredicat(mask))
         {
            continue;
         }
         auto filename = file.path().filename().string();
         filename = filename.substr(0, filename.size() - 4);
         cv::Mat maskCropped = roi.empty() ? mask : mask(roi);
         cv::imwrite(datasetFolderPath.second + filename + "png", maskCropped);
      }
   }
}

void ConvertImagesInMultipleDatasetFolders(std::vector<std::pair<std::string, std::string>> const& datasetFolderPathes,
                                           cv::Rect const& roi = {}, bool isClahe = false, cv::Size const& newSize = {},
                                           std::function<bool(cv::Mat const&)>&& skipPredicat = [](cv::Mat const&) -> bool { return false; })
{
   for (auto datasetFolderPath : datasetFolderPathes)
   {
      fs::create_directories(datasetFolderPath.second);
      for (auto file : fs::directory_iterator(datasetFolderPath.first))
      {
         cv::Mat image = cv::imread(file.path().string());
         if (skipPredicat(image))
         {
            continue;
         }
         if (isClahe)
         {
           auto clahe = cv::createCLAHE();
           cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
           clahe->apply(image, image);
         }
         auto filename = file.path().filename().string();
         filename = filename.substr(0, filename.size() - 4);
         cv::Mat imageCropped = roi.empty() ? image : image(roi);
         if (!newSize.empty())
         {
             cv::resize(imageCropped, imageCropped, newSize, cv::INTER_NEAREST);
         }
         cv::imwrite(datasetFolderPath.second + filename + ".png", imageCropped);
      }
   }
}

void ShuffleAndSplitIntoTrainAndValid(std::string path, size_t divisor)
{
    fs::create_directories(path + "\\ImagesV");
    fs::create_directories(path + "\\MasksV");
    std::vector<fs::directory_entry> datasetList(fs::directory_iterator{path + "\\images"}, fs::directory_iterator{});
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(datasetList.begin(), datasetList.end(), g);
    datasetList.erase(datasetList.cbegin() + datasetList.size() / divisor, datasetList.cend());
    for(auto const& item : datasetList) {
        fs::rename(item.path().string(),
                   path + "\\ImagesV\\" + item.path().filename().string());
        fs::rename(path + "\\masks\\" + item.path().filename().string(),
                   path + "\\MasksV\\" + item.path().filename().string());
    }
    fs::rename(path + "\\images", path + "\\ImagesT");
    fs::rename(path + "\\masks", path + "\\MasksT");
}

void iterateOverLinesInContours(std::vector<cv::Point> contour, std::function<void(cv::Point const& p1, cv::Point const& p2)>&& callback)
{
   if (contour.size() < 2)
   {
      return;
   }
   for (size_t i = 0; i < (contour.size() - 1); ++i)
   {
      callback(contour[i], contour[i + 1]);
   }
}

void TestDatasetCreateList(std::pair<std::string, std::string> const& outputDirs,
                           cv::Size const& size,
                           std::vector<cv::Scalar> const& classColors,
                           size_t count = 100)
{
   fs::create_directories(outputDirs.first);
   fs::create_directories(outputDirs.second);
   for (auto current = 0U; current < count; ++current)
   {
      auto datasetTest = TestDatasetCreate(size, classColors);
      datasetTest.first *= 255;
      datasetTest.first.convertTo(datasetTest.first, CV_8UC3);

      cv::imwrite(outputDirs.first + std::to_string(current) + ".png", datasetTest.first);
      cv::imwrite(outputDirs.second + std::to_string(current) + ".png", datasetTest.first);
   }
}

auto throw_an_error = [](std::string const& message)
{
  char const* const help = "\nUsage:\n\ttrain_unet_darknet2d options"
                           "\n\nNote:"
                           "\n\t <path-to-checkpoints-output>            - Should have enough storage capacity."
                           "\n\t"
                           "\n\nOptions"
                           "\n\t--help                                   - Prints this information."
                           "\n\t"
                           "\n\t--test-dataset-create=<path-images>,<path-masks>,<width>,<height>,<count>,<b>,<g>,<r>,<b>,<g>,<r>..."
                           "\n\t                                         - Creates test dataset with circles, rectangles etc to specified "
                           "\n\t                                         folder pathes."
                           "\n\t"
                           "\n\t--count-labels=<path>,<path>,...         - Select directories for counting labels."
                           "\n\t"
                           "\n\t--convert-polygons-to-masks=<left>,<top>,<width>,<height>,<path-polygons>,<path-out-masks>,<path-polygons>,<path-out-masks>,..."
                           "\n\t                                         - Select directories for convertation polygons to masks."
                           "\n\t"
                           "\n\t--convert-images=<left>,<top>,<width>,<height>,<path-images>,<path-masks>..."
                           "\n\t                                         - Select directories for convertation images to needed size."
                           "\n\t"
                           "\n\t--shuffle-and-split=<path>,<divisor>     - Shuffle and split dataset intro train and validation."
                           "\n\t"
                           "\n\t--convert-labelme-xml-to-json=<path>,[<path>]"
                           "\n\t                                         - Convert labelme xml to json."
                           "\n\t"
                           "\n\t--convert-polygons-to-yolo-darknet-boundingboxes=<path>,[<path>]"
                           "\n\t                                         - Convert to yolo darknet bounding boxes format"
                           "\n\t"
                           "\n\t--crop-classes-list=<classname>,[<classname>]"
                           "\n\t                                         - class names list"
                           "\n\t"
                           "\n\t--model-darknet=<path-to-darknet-model>[,<pretrained-weights-path>]"
                           "\n\t                                         - Path to darknet unet model."
                           "\n\t                                           Optionally - path to weights learned previous."
                           "\n\t"
                           "\n\t--epochs=<count>                         - Epochs count."
                           "\n\t"
                           "\n\t--batch-count=<count>                    - Batch count."
                           "\n\t"
                           "\n\t--checkpoints-output=<path-to-checkpoints-output>"
                           "\n\t                                         - Path to output where will be created checkpoints folder."
                           "\n\t"
                           "\n\t--train-directories=<path-images>,<path-masks>,<path-images>,<path-masks>,..."
                           "\n\t                                         - List with pairs paths to images and to masks."
                           "\n\t"
                           "\n\t--valid-directories=<path-images>,<path-masks>,<path-images>,<path-masks>,..."
                           "\n\t                                         - List with pairs paths to images and to masks."
                           "\n\t"
                           "\n\t--colors-to-class-map=<classname>,<b>,<g>,<r>,..."
                           "\n\t                                         - List with tuple of 4 values class name and 3 color component "
                           "\n\t                                           coresponded to its mask color in the dataset."
                           "\n\t"
                           "\n\t--selected-classes-and-thresholds=<classname>,<threshold>,<classname>,<threshold>,..."
                           "\n\t                                         - List with pairs class name and threshold."
                           "\n\t"
                           "\n\t--size-downscaled=<width>,<height>       - Width and height."
                           "\n\t"
                           "\n\t--grayscale=yes|no                       - Turn on|off color mode."
                           "\n\t"
                           "\n\t--best_weights_only=yes|no               - Turn on|off saving best weights only."
                           "\n\t"
                           "\n\nExample:"
                           "\n\t./build_host/train_unet_darknet2d \\"
                           "\n\t --model-darknet=./model/unet3c2cl2l8f.cfg \\"
                           "\n\t --epochs=100 \\"
                           "\n\t --checkpoints-output=./checkpoints_128x128_test \\"
                           "\n\t --train-directories=./dataset/train/imgs,./dataset/train/masks \\"
                           "\n\t --valid-directories=./dataset/valid/imgs,./dataset/valid/masks \\"
                           "\n\t --colors-to-class-map=\"circle,0,0,255,rectangle,0,0,255,disk,0,0,255\" \\"
                           "\n\t --selected-classes-and-thresholds=circle,0.3,rectangle,0.3,disk,0.3 \\"
                           "\n\t --batch-count=1 \\"
                           "\n\t --size-downscaled=128,128 \\"
                           "\n\t --best-weights-only=no";
  std::cout << message + "\nSee usage:\n\n" + help << std::endl;
};

auto ParseOptions(int argc, char *argv[]) -> std::map<std::string, std::vector<std::string>>
{
   std::map<std::string, std::vector<std::string>> params;
   for (auto i = 0; i < argc; ++i)
   {
      auto currentOption = std::string(argv[i]);
      auto delimiter = currentOption.find_first_of('=');
      if (delimiter != std::string::npos)
      {
         auto input = currentOption.substr(delimiter + 1);
         boost::split(params[currentOption.substr(0, delimiter)], input, boost::is_any_of(","));
      }
   }
   return params;
}

void runOpts(std::map<std::string, std::vector<std::string>> params)
{
   if (params["--generate-custom-unet"].size() == 5)
   {
      auto isGrayscale = (std::stoi(params["--generate-custom-unet"][0]) == 1);
      auto classCount = std::stoi(params["--generate-custom-unet"][1]);
      auto levelsCount = std::stoi(params["--generate-custom-unet"][2]);
      auto featuresCount = std::stoi(params["--generate-custom-unet"][3]);
      auto modelFile = std::ofstream(params["--generate-custom-unet"][4] + "/unet_" +
                                     params["--generate-custom-unet"][0] + "c" +
                                     params["--generate-custom-unet"][1] + "cl" +
                                     params["--generate-custom-unet"][2] + "l" +
                                     params["--generate-custom-unet"][3] + "f" + ".cfg");
      modelFile << "[net]\n"
                   "# Training\n"
                   "width=256\n"
                   "height=256\n"
                   "#now supported only grayscale and 3 color\n"
                   "channels=" << (isGrayscale ? 1 : 3) << "\n"
                   "learning_rate=1e-7\n"
                   "batch=20\n"
                   "eps=1e-05\n"
                   "momentum=0.1\n"
                   "\n"
                   "decay=0.0005\n"
                   "adam=0\n"
                   "B1=0.9\n"
                   "B2=0.999\n"
                   "max_batches = 400\n";
      for (auto currentLevel = 0; currentLevel < levelsCount; ++currentLevel)
      {
         modelFile << "\n###encoder" << (currentLevel + 1) <<"\n"
                      "[convolutional]\n"
                      "batch_normalize=1\n"
                      "filters=" << featuresCount << "\n"
                      "size=3\n"
                      "stride=1\n"
                      "pad=1\n"
                      "activation=leaky\n"
                      "\n"
                      "[convolutional]\n"
                      "batch_normalize=1\n"
                      "filters=" << featuresCount << "\n"
                      "size=3\n"
                      "stride=1\n"
                      "pad=1\n"
                      "activation=leaky\n"
                      "\n"
                      "[maxpool]\n"
                      "size=2\n"
                      "stride=2\n";
         featuresCount *= 2;
      }
      modelFile << "\n###bottleneck\n"
                   "[convolutional]\n"
                   "batch_normalize=1\n"
                   "filters=" << featuresCount << "\n"
                   "size=3\n"
                   "stride=1\n"
                   "pad=1\n"
                   "activation=leaky\n"
                   "\n"
                   "[convolutional]\n"
                   "batch_normalize=1\n"
                   "filters=" << featuresCount << "\n"
                   "size=3\n"
                   "stride=1\n"
                   "pad=1\n"
                   "activation=leaky\n";
      for (auto currentLevel = 0; currentLevel < levelsCount; ++currentLevel)
      {
         featuresCount /= 2;
         modelFile << "\n###decoder" << levelsCount - currentLevel << "\n"
                      "[upsample]\n"
                      "stride=2\n"
                      "\n"
                      "[convolutional]\n"
                      "filters=" << featuresCount << "\n"
                      "size=3\n"
                      "stride=1\n"
                      "pad=1\n"
                      "activation=leaky\n"
                      "\n"
                      "[route]\n"
                      "layers = -1, -" << (2 + (4 * (currentLevel + 1)) + ((1 + 1 + 2) * currentLevel)) << "\n"
                      "\n"
                      "[convolutional]\n"
                      "batch_normalize=1\n"
                      "filters=" << featuresCount << "\n"
                      "size=3\n"
                      "stride=1\n"
                      "pad=1\n"
                      "activation=leaky\n"
                      "\n"
                      "[convolutional]\n"
                      "batch_normalize=1\n"
                      "filters=" << featuresCount << "\n"
                      "size=3\n"
                      "stride=1\n"
                      "pad=1\n"
                      "activation=leaky\n";
      }
      modelFile << "\n################################\n"
                   "[convolutional]\n"
                   "filters=" << classCount << "\n"
                   "size=1\n"
                   "stride=1\n"
                   "activation=logistic\n"
                   "################################";
      return;
   }
   if ((params["--test-dataset-create"].size() >= 8) && (((params["--test-dataset-create"].size() - 5) % 3) == 0))
   {
      std::vector<cv::Scalar> classColors;
      for (auto colorsIt = std::next(params["--test-dataset-create"].cbegin(), 5);
           colorsIt != params["--test-dataset-create"].cend();
           colorsIt += 3)
      {
         classColors.emplace_back(cv::Scalar(std::stoi(*colorsIt),
                                             std::stoi(*std::next(colorsIt, 1)),
                                             std::stoi(*std::next(colorsIt, 2))));
      }
      TestDatasetCreateList(std::make_pair(params["--test-dataset-create"][0],
                                           params["--test-dataset-create"][1]),
                            cv::Size{std::stoi(params["--test-dataset-create"][2]),
                                     std::stoi(params["--test-dataset-create"][3])},
                            classColors,
                            std::stoi(params["--test-dataset-create"][4]));
   }
   if (!params["--count-labels"].empty())
   {
      auto map = CountingLabeledObjectsInMultipleDatasetFolders(params["--count-labels"]);
      std::cout << "Count labels list: " << std::endl;
      for (auto const& item : map)
      {
         std::cout << "\t" << item.first << " : " << item.second << std::endl;
      }
      return;
   }
   if (params.count("--convert-polygons-to-yolo-darknet-boundingboxes") &&
       ((params["--convert-polygons-to-yolo-darknet-boundingboxes"].size() % 2) == 0) &&
       !params["--crop-classes-list"].empty())
   {
       auto pathesList = std::vector<std::pair<std::string, std::string>>{};
       auto& inputAndOutputPathes = params["--convert-polygons-to-yolo-darknet-boundingboxes"];
       for(auto i = 0; i < inputAndOutputPathes.size(); i += 2)
       {
           pathesList.emplace_back(std::make_pair(inputAndOutputPathes[i], inputAndOutputPathes[i+1]));
       }
       ConvertPolygonsToBoundingBoxesYoloDarknet(pathesList, params["--crop-classes-list"]);
       return;
   }
   std::map<std::string, cv::Scalar> colorsToClass;
   if (!params["--colors-to-class-map"].empty() && (params["--colors-to-class-map"].size() % 4) == 0)
   {
      for (auto colorToClassIt = params["--colors-to-class-map"].cbegin();
           colorToClassIt != params["--colors-to-class-map"].cend();
           colorToClassIt += 4)
      {
         colorsToClass[*colorToClassIt] = cv::Scalar(
            std::stoi(*std::next(colorToClassIt, 1)), std::stoi(*std::next(colorToClassIt, 2)), std::stoi(*std::next(colorToClassIt, 3)));
      }
   }
   if ((params["--convert-polygons-to-masks"].size() >= 6) && (((params["--convert-polygons-to-masks"].size() - 4) % 2) == 0))
   {
      auto roi = cv::Rect{std::stoi(params["--convert-polygons-to-masks"][0]),
                          std::stoi(params["--convert-polygons-to-masks"][1]),
                          std::stoi(params["--convert-polygons-to-masks"][2]),
                          std::stoi(params["--convert-polygons-to-masks"][3])};
      if (!colorsToClass.empty())
      {
         std::vector<std::pair<std::string, std::string>> polygonsPathsInputAndOutput;
         for (auto polygonsInputAndOutputIt = std::next(params["--convert-polygons-to-masks"].cbegin(), 4);
              polygonsInputAndOutputIt != params["--convert-polygons-to-masks"].cend();
              polygonsInputAndOutputIt += 2)
         {
            polygonsPathsInputAndOutput.emplace_back(std::make_pair(*polygonsInputAndOutputIt, *std::next(polygonsInputAndOutputIt)));
         }
         ConvertPolygonsToMaskInMultipleDatasetFolders(polygonsPathsInputAndOutput, roi, colorsToClass);
      }
      else
      {
         std::cout << "Skipped apply option --convert-polygons-to-masks since no color to masks applied!" << std::endl;
      }
   }
   else
   {
      std::cout << "Skipped apply option --convert-polygons-to-masks since values count less than 6 or values count - 4 not even!" << std::endl;
   }
   if (params.count("--convert-images"))
   {
       if ((params["--convert-images"].size() >= 6) && (((params["--convert-images"].size() - 4) % 2) == 0))
       {
           auto newSize = cv::Size{};
           if (params["--size-downscaled"].size() == 2)
           {
               newSize = cv::Size{std::stoi(params["--size-downscaled"][0]),
                                  std::stoi(params["--size-downscaled"][1])};
           }

           auto roi = cv::Rect{std::stoi(params["--convert-images"][0]),
                               std::stoi(params["--convert-images"][1]),
                               std::stoi(params["--convert-images"][2]),
                               std::stoi(params["--convert-images"][3])};
           std::vector<std::pair<std::string, std::string>> imagesPathsInputAndOutput;
           for (auto imagesInputAndOutputIt = std::next(params["--convert-images"].cbegin(), 4);
                imagesInputAndOutputIt != params["--convert-images"].cend();
                imagesInputAndOutputIt += 2)
           {
               imagesPathsInputAndOutput.emplace_back(
                       std::make_pair(*imagesInputAndOutputIt, *std::next(imagesInputAndOutputIt)));
           }
           ConvertImagesInMultipleDatasetFolders(imagesPathsInputAndOutput, roi,
                                                 (!params["--clahe"].empty() && params["--clahe"][0] == "yes"),
                                                 newSize);
       }
       else
       {
           std::cout << "Skipped apply option --convert-images since values count less than 6 or values count - 4 not even!" << std::endl;
       }
   }
   if (params.count("--shuffle-and-split"))
   {
       if (params["--shuffle-and-split"].size() == 2)
       {
          ShuffleAndSplitIntoTrainAndValid(params["--shuffle-and-split"][0], std::stoi(params["--shuffle-and-split"][1]));
       }
       else
       {
           std::cout << "Expected two parameters!" << std::endl;
       }
       return;
   }
   if (params.count("--convert-labelme-xml-to-json"))
   {
#if 0
       auto pathToDatasetList = std::vector<std::string>{
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_101_jobs_103(only109photos)\\Dataset\\default"
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_117_jobs_133\\Dataset\\default",
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_138_jobs_214\\Dataset\\default",
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_138_jobs_215\\Dataset\\default",
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_157_jobs_255\\Dataset\\default",
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_157_jobs_256\\Dataset\\default",
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_157_jobs_257\\Dataset\\default",
//        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\a_makazan\\tasks_157_jobs_258\\Dataset\\default",
        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\gashingrigorii\\tasks_75_jobs_65\\Dataset\\default",
        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\gashingrigorii\\tasks_94_jobs_82\\Dataset\\default",
        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\gashingrigorii\\tasks_102_jobs_104\\Dataset\\default",
        "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\gashingrigorii\\tasks_154_jobs_249\\Dataset\\default",
 //       "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\magic07\\tasks_140_jobs_220\\Dataset\\default",
 //       "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\natalie6562\\tasks_119_jobs_135(empty)\\Dataset\\default",
 //       "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\natalie6562\\tasks_159_jobs_265\\Dataset\\default",
 //       "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\natalie6562\\tasks_159_jobs_266\\Dataset\\default",
 //       "G:\\Datasets\\Cable (U-Net)\\Annotation_31.05.2021\\natalie6562\\tasks_159_jobs_267\\Dataset_rotate\\default"
    };
#endif
       for (auto const& item : params["--convert-labelme-xml-to-json"])
       {
           std::cout << "Start conversion dataset: " << item << std::endl;
           ConvertingXmlLabelmeToJsonLabelmeInFolder(item, "..\/Image");
       }
   }
   if (params.count("--get-cropped-mask-for-selected-labels") &&
       params.count("--downscaling") &&
       params.count("--crop-classes-list") &&
       params.count("--colors-to-class-map") &&
       params.count("--rotate"))
   {
       auto pathesList = std::vector<std::pair<std::string, std::string>>{};
       auto& inputAndOutputPathes = params["--get-cropped-mask-for-selected-labels"];
       for(auto i = 0; i < inputAndOutputPathes.size(); i += 2)
       {
           pathesList.emplace_back(std::make_pair(inputAndOutputPathes[i], inputAndOutputPathes[i+1]));
       }
        GetCroppedMaskForSelectedLabelInMultipleDatasetFolders(pathesList,
                                                               params["--crop-classes-list"],
                                                               std::atoi(params["--align-factor"][0].c_str()),
                                                               cv::Size{std::atoi(params["--downscaling"][0].c_str()),
                                                                        std::atoi(params["--downscaling"][1].c_str())},
                                                               colorsToClass,
                                                               std::atoi(params["--rotate"][0].c_str()));
       return;
   }
   if (params["--epochs"].empty() || (std::stoi(params["--epochs"][0]) == 0))
   {
      std::cout << "Skip training since epochs option absent count equal to 0!" << std::endl;
     // return;
   }
   const auto kNumberOfEpochs = std::stoi(params["--epochs"][0]);

   if (params["--checkpoints-output"].empty())
   {
      std::cout <<"Skip training since --checkpoints-output has not been set!" << std::endl;
      //return;
   }
   if (colorsToClass.empty())
   {
      throw_an_error("Skip training since --colors-to-class-map has not been set!");
      return;
   }
   std::vector<std::tuple<std::string, std::string>> trainDirectories;
   if (params["--train-directories"].size() % 2 != 0)
   {
      throw_an_error("Train directories should be even count!\n");
      return;
   }
   for (auto i = 0U; i < params["--train-directories"].size(); i += 2)
   {
      trainDirectories.push_back(std::make_tuple(params["--train-directories"][i],
                                                 params["--train-directories"][i + 1]));
   }

   std::vector<std::tuple<std::string, std::string>> validDirectories;
   if (!(params["--valid-directories"].size() % 2 == 0))
   {
      throw_an_error("--valid-directories should be even count!\n");
      return;
   }
   for (auto i = 0U; i < params["--valid-directories"].size(); i += 2)
   {
      validDirectories.push_back(std::make_tuple(params["--valid-directories"][i], params["--valid-directories"][i + 1]));
   }

   if (params["--model-darknet"].empty())
   {
      throw_an_error("Option --model-darknet does not set!\n");
      return;
   }

   if ((params["--selected-classes-and-thresholds"].size() < 2) ||
       ((params["--selected-classes-and-thresholds"].size() % 2) != 0))
   {
      // TODO: Check for each class selected exist in classesToColor
      throw_an_error("--selected-classes-and-thresholds has not been set!\n");
      return;
   }

   std::vector<cv::Scalar> classColors;
   std::vector<float> thresholds(classColors.size(), 0.3);
   for (auto selectedClassAndThresholdIt = params["--selected-classes-and-thresholds"].cbegin();
        selectedClassAndThresholdIt != params["--selected-classes-and-thresholds"].cend();
        selectedClassAndThresholdIt += 2)
   {
      classColors.push_back(colorsToClass[*selectedClassAndThresholdIt]);
      thresholds.push_back(std::stof(*std::next(selectedClassAndThresholdIt)));
   }

   int batchSize = params["--batch-count"].empty() ? 1 : std::stoi(params["--batch-count"][0]);

   if (params["--size-downscaled"].size() != 2)
   {
      throw_an_error("--size-downscaled should be set!");
      return;
   }

   bool isGrayscale = false;
   if (!params["--grayscale"].empty())
   {
      isGrayscale = (params["--grayscale"][0] == "yes");
   }

   bool isEval = false;
   std::vector<float> thresholdEvalSteps(thresholds.size());
   if (!params["--eval"].empty())
   {
       isEval = (params["--eval"][0] == "yes");
       for (auto i = 0; i < thresholds.size(); ++i)
       {
           thresholdEvalSteps[i] = (1.0f - thresholds[i]) / kNumberOfEpochs;
       }
   }

   auto size = cv::Size{std::stoi(params["--size-downscaled"][0]),
                        std::stoi(params["--size-downscaled"][1])};

   torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
   std::cout << "Using device: " << device << std::endl;

   auto train_dataset = UNetDataset(trainDirectories, classColors, size, isGrayscale);
   auto valid_dataset = UNetDataset(validDirectories, classColors, size, isGrayscale);

   auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset.map(torch::data::transforms::Stack<>())), batchSize);
   auto valid_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(valid_dataset.map(torch::data::transforms::Stack<>())), batchSize);

   Darknet model = Darknet(params["--model-darknet"][0], cv::Size{0, 0}, true);
   std::cout << model->_moduleList << std::endl;

   if (params["--model-darknet"].size() == 2)
   {
       model->load_weights(params["--model-darknet"][1]);
       model->save_weights(params["--model-darknet"][1] + "saved");
   }

   model->to(device);

   std::cout << "UNet2d: " << c10::str(model) << std::endl;

   torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.00001));
   std::map<std::string, std::vector<cv::Point>> trainingLossData;
   std::string outputDirectory = params["--checkpoints-output"][0];
   for (size_t epoch = 1U; epoch <= kNumberOfEpochs; ++epoch)
   {
      if (!isEval)
      {
          train(epoch, model, device, *train_loader, train_dataset.count(), optimizer, [](int32_t count, int32_t current) {
              if ((current % (count / 10)) == 0)
              {
                  std::cout << "\rProgress: [" << count << " / " << current << "]" << std::endl;
              }
              cv::waitKey(1);
          });
          std::cout << std::endl;
      }

      auto i = 0;
      valid(model, device, *valid_loader, valid_dataset.count(), [&](torch::Tensor& predict, torch::Tensor& targets)
      {
        auto const batches = predict.size(0);
        auto const height = predict.size(2);
        auto const width = predict.size(3);
        int sz[] = {batches, static_cast<int>(classColors.size()), width, height};
        auto predictCpu = predict.contiguous().cpu();
        cv::Mat predictMat = cv::Mat(4, sz, CV_32FC1, predictCpu.data_ptr());
        auto masks = toClassesMapsThreshold(predictMat, {}, thresholds);
        for (auto ii = 0U; ii < masks.size(); ++ii)
        {
           cv::imshow(std::string("Predict_") + std::to_string(ii % classColors.size()) + "_" + std::to_string(ii / classColors.size()), masks[ii]);
        }

        auto targetsCpu = targets.contiguous().cpu();
        cv::Mat targetMat = cv::Mat(4, sz, CV_32FC1, targetsCpu.data_ptr());
        masks = toClassesMapsThreshold(targetMat, {}, thresholds);
        for (auto ii = 0U; ii < masks.size(); ++ii)
        {
           cv::imshow(std::string("Target_") + std::to_string(ii % classColors.size()) + "_" + std::to_string(ii / classColors.size()), masks[ii]);
        }
        cv::waitKey(1);
        ++i;
      }, [&](std::map<std::string, float> metrics)
      {
        static std::map<std::string, cv::Scalar> chartColors = {std::make_pair("bce", cv::Scalar(0x00, 0x00, 0xFF)),
                                                                std::make_pair("dice", cv::Scalar(0x00, 0xFF, 0x00)),
                                                                std::make_pair("loss", cv::Scalar(0xFF, 0x00, 0x00))};
        static std::map<std::string, std::vector<cv::Point>> validationLossData;
        cv::Mat validationChart = cv::Mat(500, 500, CV_8UC3, cv::Scalar{255, 255, 255});
        for (auto const& item : metrics)
        {
           validationLossData[item.first].emplace_back(static_cast<int>((static_cast<float>(500)/kNumberOfEpochs) * epoch),
                                                       500 - (item.second * 500));
           // MSVC lead to error when in lambda below was auto const& parameters(for x86 compiler)
           iterateOverLinesInContours(validationLossData[item.first], [&](cv::Point const& p1, cv::Point const& p2){
             cv::line(validationChart, p1, p2, chartColors[item.first], 3);
           });
           cv::imshow(std::string("Validation chart"), validationChart);
           cv::imwrite(outputDirectory + "/chart.png", validationChart);
           cv::waitKey(1);
        }
        static auto minLoss = 10000.0f;
        static auto bestEpoch = 0;
        if (metrics["loss"] < minLoss)
        {
           if (!isEval)
           {
               fs::remove(outputDirectory + "/" + "best_" + std::to_string(bestEpoch) + ".weights");
               model->save_weights(outputDirectory + "/" + "best_" + std::to_string(epoch) + ".weights");
           }
           minLoss = metrics["loss"];
           bestEpoch = epoch;
           std::cout << "Best epoch: " << epoch << std::endl;
        }
      });
      if (isEval)
      {
          std::cout << "Thresholds: ";
          for (auto const& threshold : thresholds)
          {
              std::cout << threshold << ", ";
          }
          for (auto ii = 0; ii < thresholds.size(); ++ii)
          {
              thresholds[ii] += thresholdEvalSteps[ii];
          }
          std::cout << std::endl;
          continue;
      }
      fs::create_directory(outputDirectory);
      if (params["best_weights_only"].empty() || (params["best_weights_only"][0] == "no"))
      {
          model->save_weights(outputDirectory + "/" + std::to_string(epoch) + ".weights");
      }
   }
}

