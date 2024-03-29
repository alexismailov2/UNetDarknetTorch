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
void train(size_t epoch, Darknet& model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer, std::function<void(int32_t, int32_t)>&& step = [](int32_t, int32_t){})
{
   std::cout << "=======================" << std::endl;
   std::cout << "Epoch: " << epoch << std::endl;
   std::map<std::string, float> metrics;

   std::cout << "device:" << device << std::endl;
   model->train();
   auto const count = std::distance(data_loader.begin(), data_loader.end());
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
   printMetrics(metrics, std::distance(data_loader.begin(), data_loader.end()));
}

template <typename DataLoader>
void valid(Darknet& model,
           torch::Device device,
           DataLoader& data_loader,
           std::function<void(torch::Tensor&, torch::Tensor&)>&& handler = [](torch::Tensor&, torch::Tensor&){},
           std::function<void(std::map<std::string, float>)>&& metricsHandler = [](auto&&){})
{
   std::cout << "-----------------------" << std::endl;
   std::cout << "Validation: " << std::endl;
   std::map<std::string, float> metrics;

   model->eval();
   torch::NoGradGuard no_grad;
   for (const auto& batch : data_loader)
   {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);
      auto output = model->forward(data);

      calcLoss(output, targets, metrics);
      handler(output, targets);
   }
   auto datasetSize = std::distance(data_loader.begin(), data_loader.end());
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

char const* const help = "\nUsage:\n\ttrain_unet_darknet2d options"
                         "\n\nNote:"
                         "\n\t <path-to-checkpoints-output>            - Should have enough storage capacity."
                         "\n\t"
                         "\n\nOptions"
                         "\n\t--help                                   - Prints this information."
                         "\n\t"
                         "\n\t--generate-custom-unet=<input-channels-count>,<output-classes-count>,<encoder-decoder-levels-count>,<initial-feature-count>,<model path>"
                         "\n\t                                         - Creates custom unet model config file(NOTE: without weights!)"
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

auto throw_an_error = [](std::string const& message)
{
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
   if (params.count("--help"))
   {
       std::cout << help << std::endl;
       return;
   }
   if (params.count("--generate-custom-unet"))
   {
      if (params["--generate-custom-unet"].size() != 5)
      {
         throw_an_error(std::string("Expected 5 parameters but provided ") + std::to_string(params["--generate-custom-unet"].size()));
         return;
      }
      fs::create_directories(params["--generate-custom-unet"][4]);
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
      return;
   }
   if (!params["--count-labels"].empty())
   {
      auto map = CountingLabeledObjectsInMultipleDatasetFolders(params["--count-labels"]);
      std::cout << "Count labels list: " << std::endl;
      for (auto const& item : map)
      {
         std::cout << "\t" << item.first << " : " << item.second << std::endl;
      }
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
         imagesPathsInputAndOutput.emplace_back(std::make_pair(*imagesInputAndOutputIt, *std::next(imagesInputAndOutputIt)));
      }
      ConvertImagesInMultipleDatasetFolders(imagesPathsInputAndOutput, roi, (!params["--clahe"].empty() && params["--clahe"][0] == "yes"), newSize);
   }
   else
   {
      std::cout << "Skipped apply option --convert-images since values count less than 6 or values count - 4 not even!" << std::endl;
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

   auto train_dataset = UNetDataset(trainDirectories, classColors, size, isGrayscale).map(torch::data::transforms::Stack<>());
   auto valid_dataset = UNetDataset(validDirectories, classColors, size, isGrayscale).map(torch::data::transforms::Stack<>());

   auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batchSize);
   auto valid_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(valid_dataset), batchSize);

   if (!fs::exists(params["--model-darknet"][0]))
   {
       std::cout << "ERROR: " << params["--model-darknet"][0] << " does not exist!" << std::endl;
       return;
   }
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
          train(epoch, model, device, *train_loader, optimizer, [](int32_t count, int32_t current) {
              if ((current % (count / 10)) == 0)
              {
                  std::cout << "\rProgress: [" << count << " / " << current << "]" << std::endl;
              }
              cv::waitKey(1);
          });
          std::cout << std::endl;
      }

      auto i = 0;
      std::vector<double> lossesByClass(classColors.size());
      valid(model, device, *valid_loader, [&](torch::Tensor& predict, torch::Tensor& targets)
      {
        auto const batches = predict.size(0);
        auto const height = predict.size(2);
        auto const width = predict.size(3);
        int sz[] = {batches, static_cast<int>(classColors.size()), width, height};
        auto predictCpu = predict.contiguous().cpu();
        cv::Mat predictMat = cv::Mat(4, sz, CV_32FC1, predictCpu.data_ptr());
        auto masksPredict = toClassesMapsThreshold(predictMat, {}, thresholds);
        for (auto ii = 0U; ii < masksPredict.size(); ++ii)
        {
           cv::imshow(std::string("Predict_") + std::to_string(ii % classColors.size()) + "_" + std::to_string(ii / classColors.size()), masksPredict[ii]);
        }

        auto targetsCpu = targets.contiguous().cpu();
        cv::Mat targetMat = cv::Mat(4, sz, CV_32FC1, targetsCpu.data_ptr());
        auto masksTarget = toClassesMapsThreshold(targetMat, {}, thresholds);
        for (auto ii = 0U; ii < masksTarget.size(); ++ii)
        {
           cv::imshow(std::string("Target_") + std::to_string(ii % classColors.size()) + "_" + std::to_string(ii / classColors.size()), masksTarget[ii]);
        }

        for (auto ii = 0U; ii < masksPredict.size(); ++ii)
        {
            cv::Mat diffMat;
            cv::absdiff(masksTarget[ii], masksPredict[ii], diffMat);
            lossesByClass[ii % classColors.size()] += (cv::sum(diffMat/255.0) / (masksTarget[ii].rows * masksTarget[ii].cols))[0] / classColors.size();
        }
        cv::waitKey(1);
        ++i;
      }, [&](std::map<std::string, float> metrics)
      {
        for (auto ii = 0; ii < lossesByClass.size(); ++ii)
        {
            lossesByClass[ii] /= i;
            std::cout << "Diff for " << std::to_string(ii % lossesByClass.size()) + "_" + std::to_string(ii / lossesByClass.size()) << ": " << lossesByClass[ii] << std::endl;
        }
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

