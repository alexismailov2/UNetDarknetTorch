#include <UNet/UNetDataset.hpp>

#include <experimental/filesystem>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace {

void LabelsEnumerator(std::string const& annotationDir, std::function<void(std::string const&, std::set<std::string> const&)> callback)
{
   for (auto file : std::experimental::filesystem::directory_iterator(annotationDir))
   {
      boost::property_tree::ptree pt;
      boost::property_tree::read_json(file.path().string(), pt);

      std::set<std::string> classesSetForImage;
      auto shapes = pt.get_child("shapes");
      for (auto const& shape : shapes)
      {
         auto label = shape.second.get<std::string>("label");
         classesSetForImage.insert(label);
      }
      callback(file.path().filename().string(), classesSetForImage);
   }
}

void CountingLabeledObjects(std::map<std::string, uint32_t>& map, std::string const& polygonInfo, bool forImage = false)
{
   boost::property_tree::ptree pt;
   boost::property_tree::read_json(polygonInfo, pt);

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
}

auto ConvertPolygonsToMask(std::string const& polygonInfo, std::map<std::string, cv::Scalar> colorToClass) -> cv::Mat {
   boost::property_tree::ptree pt;
   boost::property_tree::read_json(polygonInfo, pt);

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

auto colorsToMasks(cv::Mat const& colorMasks, std::vector<cv::Scalar> const& classColors, cv::Size const& size) -> cv::Mat
{
   int sz[] = {static_cast<int>(classColors.size()), size.width, size.height};
   cv::Mat masks = cv::Mat::zeros(3, sz, CV_8UC1);

   auto const rows = masks.size[2];
   auto const cols = masks.size[1];
   auto const channels = masks.size[0];

   for (auto ch = 0; ch < channels; ++ch)
   {
      cv::Mat currentClassMask = cv::Mat(rows, cols, CV_8UC1, const_cast<uint8_t*>(masks.ptr<uint8_t>(ch, 0, 0)));
      cv::inRange(colorMasks, classColors[ch], classColors[ch], currentClassMask);
   }
   masks.convertTo(masks, CV_32FC1, 1.0 / 255.0);
   return masks;
}

auto toClassesMapsThreshold(cv::Mat const& score,
                            cv::Size const& inputSize,
                            std::vector<float> threshold) -> std::vector<cv::Mat>
{
   auto const rows = score.size[2];
   auto const cols = score.size[1];
   auto const channels = score.size[0];

   std::vector<cv::Mat> classesMaps(static_cast<size_t>(channels));
   for (auto ch = 0; ch < channels; ++ch)
   {
      cv::Mat channelScore = cv::Mat(rows, cols, CV_32FC1, const_cast<float *>(score.ptr<float>(ch, 0, 0)));
      cv::inRange(channelScore, threshold[ch], 1000000.0, classesMaps[ch]);
      std::cout << "dataset tresholding:" << classesMaps[ch] << std::endl;
//      if ((inputSize.width != 0) && (inputSize.height != 0))
//      {
//         cv::resize(classesMaps[ch], classesMaps[ch], inputSize);
//      }
   }
   return classesMaps;
}
} /// end namespace anonymous

auto TestDatasetCreate(cv::Size const& size, std::vector<cv::Scalar> const& classes) -> std::pair<cv::Mat, cv::Mat>
{
   cv::Mat image = cv::Mat::zeros(size.height, size.width, CV_8UC3);
   auto count = rand() % 10;
   for (auto i = 0; i < count; ++i)
   {
      auto type = rand() % classes.size();
      auto x = rand() % size.width;
      auto y = rand() % size.height;
      auto width = 30;//rand() % 5 + 5;
      auto height = width;

      switch(type)
      {
         case 0:
            cv::circle(image, {x, y}, width, classes[type], -1);
            break;
         case 1:
            cv::rectangle(image, {x, y, width, height}, classes[type], -1);
            break;
         case 2:
            cv::circle(image, {x, y}, width, classes[type], 2);
            break;
         case 3:
            cv::rectangle(image, {x, y, width, height}, classes[type], 2);
            break;
         case 4:
            cv::line(image, {x - width/2, y}, {x + width/2, y}, classes[type], 2);
            cv::line(image, {x, y - height/2}, {x, y + height/2}, classes[type], 2);
            break;
         case 5:
         default:
            cv::fillPoly(image, std::vector<std::vector<cv::Point>>{{{x, y - height/2}, {x + width/2, y + height/2}, {x - width/2, y + height/2}, {x, y - height/2}}}, classes[type], 2);
            break;
      }
   }
   auto masks = colorsToMasks(image, classes, size);

   image.convertTo(image, CV_32FC3, 1.0 / 255.0);
   return std::make_pair(image.clone(), masks);
}

UNetDataset::UNetDataset(std::vector<std::tuple<std::string, std::string>> const &datasetDirsPath,
                         std::vector<cv::Scalar> const& classColors,
                         cv::Size size,
                         bool grayscale)
  : _datasetDirsPath{datasetDirsPath}
  , _size{size}
  , _classColors{classColors}
{
   for (auto const& datasetDirPath : datasetDirsPath)
   {
      for (auto& p : std::experimental::filesystem::directory_iterator(std::get<0>(datasetDirPath)))
      {
         _imagesAndMasks.emplace_back(std::make_pair(p.path().string(), std::get<1>(datasetDirPath) + "/" + p.path().filename().string()));
         auto const& imageAndMask = _imagesAndMasks.back();
         if (imageAndMask.second.substr(imageAndMask.second.size() - 3) == "jpg")
         {
            _imagesAndMasks.back().second.replace(imageAndMask.second.size() - 3, 3, "png");
         }
         _image.emplace_back(cv::imread(imageAndMask.first, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
         cv::resize(_image.back(), _image.back(), _size, 0, 0, cv::INTER_NEAREST);

         _image.back().convertTo(_image.back(), grayscale ? CV_32FC1 : CV_32FC3, 1.0 / 255.0);

         torch::Tensor inputImage = torch::from_blob(_image.back().data, {_size.height, _size.width, grayscale ? 1 : 3}, torch::kFloat);
         inputImage = inputImage.permute({2, 0, 1});

         cv::Mat colorMasks = cv::imread(imageAndMask.second);
         cv::resize(colorMasks, colorMasks, _size, 0, 0, cv::INTER_NEAREST);

         _mask.emplace_back(colorsToMasks(colorMasks, classColors, _size));

         torch::Tensor maskImage =
            torch::from_blob(_mask.back().data, {static_cast<int>(classColors.size()), _size.height, _size.width}, torch::kFloat);

         _data.emplace_back(inputImage, maskImage);
      }
   }
}

auto UNetDataset::get(size_t index) -> torch::data::Example<>
{
   return _data[index];
}

auto UNetDataset::size() const -> c10::optional<size_t>
{
  return _data.size();
}
