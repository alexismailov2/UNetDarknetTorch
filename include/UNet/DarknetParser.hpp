#pragma once

#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>

#include <stack>
#include <string>
#include <fstream>
#include <experimental/filesystem>

struct SwishImpl : torch::nn::Module
{
    auto forward(const torch::Tensor& x) -> torch::Tensor
    {
       return x * torch::sigmoid(x);
    }
};
TORCH_MODULE_IMPL(Swish, SwishImpl);

struct HardSwishImpl : torch::nn::Module
{
    auto forward(const torch::Tensor& x) -> torch::Tensor
    {
       return x * torch::nn::functional::hardtanh(x + 3, torch::nn::HardtanhOptions{}.min_val(0.).max_val(6.).inplace(true)) / 6.;
    }
};
TORCH_MODULE_IMPL(HardSwish, HardSwishImpl);

struct MishImpl : torch::nn::Module
{
    auto forward(const torch::Tensor& x) -> torch::Tensor
    {
       return x * torch::nn::functional::softplus(x).tanh();
    }
};
TORCH_MODULE_IMPL(Mish, MishImpl);

//struct SigmoidImpl : torch::nn::Module
//{
//   auto forward(const torch::Tensor& x) -> torch::Tensor
//   {
//      return x.sigmoid();
//   }
//};
//TORCH_MODULE_IMPL(Sigmoid, SigmoidImpl);

struct ConcatImpl : torch::nn::Module
{
    ConcatImpl(uint32_t dimension = 1)
       : _dimension{dimension}
    {}

    auto forward(const torch::Tensor& x) -> torch::Tensor
    {
       return torch::cat(x, 1);
    }
    uint32_t _dimension{};
};
TORCH_MODULE_IMPL(Concat, ConcatImpl);

struct FeatureConcatImpl : torch::nn::Module
{
    FeatureConcatImpl(std::vector<int32_t> const& layers)
      : _layers{layers}
    {}

    auto forward(torch::Tensor const& x, std::vector<torch::Tensor> const& outputs) -> torch::Tensor
    {
       switch(_layers.size())
       {
          case 2:
             return torch::cat({outputs[_layers[0]], outputs[_layers[1]]}, 1);
          case 3:
             return torch::cat({outputs[_layers[0]], outputs[_layers[1]], outputs[_layers[2]]}, 1);
          case 4:
             return torch::cat({outputs[_layers[0]], outputs[_layers[1]], outputs[_layers[2]], outputs[_layers[3]]}, 1);
          case 1:
          default:
             return torch::cat(outputs[_layers[0]]);
             break;
       }
       return {};
    }
    std::vector<int32_t> _layers{};
};
TORCH_MODULE_IMPL(FeatureConcat, FeatureConcatImpl);

class DarknetParser {
public:
    DarknetParser(std::string const& path)
       : _path{path}
    {
       if (_path.substr(_path.size()-4) != ".cfg")
       {
          _path += ".cfg";
       }
       if (!std::experimental::filesystem::exists(_path) &&
           std::experimental::filesystem::exists("cfg/" + _path))
       {
          _path = "cfg/" + _path;
       }
       std::ifstream cfgFile(_path);
       std::string line;
       while (std::getline(cfgFile, line))
       {
          boost::trim(line);
          if (line.empty() || (line[0] == '#'))
          {
             continue;
          }
          if (line[0] == '[')
          {
             auto item = std::make_pair(line.substr(1, line.size() - 2), std::map<std::string, std::string>{std::make_pair("batch_normalize", "0")});
             mdefs.push_back(item);
          }
          else
          {
             std::vector<std::string> keyVal;
             boost::split(keyVal, line, boost::is_any_of("="));
             boost::trim(keyVal[0]);
             boost::trim(keyVal[1]);
             if (keyVal[0] == "anchors")
             {

             }
             else if (false)
             {

             }
             else
             {
                mdefs.back().second[keyVal[0]] = keyVal[1];
             }
          }
       }
       auto supported = std::vector<std::string>{"type", "batch_normalize", "filters", "size", "stride", "pad", "activation", "layers", "groups",
               "from", "mask", "anchors", "classes", "num", "jitter", "ignore_thresh", "truth_thresh", "random",
               "stride_x", "stride_y", "weights_type", "weights_normalization", "scale_x_y", "beta_nms", "nms_kind",
               "iou_loss", "iou_normalizer", "cls_normalizer", "iou_thresh", "probability"};
    }

    auto createModules(cv::Size const& size, std::string const& cfg) -> std::pair<torch::nn::ModuleList, std::vector<bool>>
    {
       using namespace torch::nn;

       uint32_t channels = std::atoi(mdefs[0].second["channels"].c_str());
       auto momentum = std::atof(mdefs[0].second["momentum"].c_str());
       auto eps = std::atof(mdefs[0].second["eps"].c_str());

       std::vector<int32_t> routs;
       auto output_filters = std::vector<uint32_t>{channels};
       uint32_t filters = 0;
       ModuleList moduleList;

       bool upsamplePrev = false;
       for (auto i = 1; i < mdefs.size(); ++i)
       {
          auto& mdef = mdefs[i].second;
          auto const& type = mdefs[i].first;
          Sequential modules;
          if (type == "convolutional")
          {
             auto bn = static_cast<bool>(std::atoi(mdef["batch_normalize"].c_str()));
             filters = std::atoi(mdef["filters"].c_str());
             auto k = std::atoi(mdef["size"].c_str());
             auto stride_x = mdef.count("stride") ? std::atoi(mdef["stride"].c_str()) : std::atoi(mdef["stride_x"].c_str());
             auto stride_y = mdef.count("stride") ? std::atoi(mdef["stride"].c_str()) : std::atoi(mdef["stride_y"].c_str());

             //if (isinstance(k, int))
             {  // single-size conv
                if (upsamplePrev)
                {
                   output_filters.back() /= 2;
                }
                modules->push_back("Conv2d",
                                   Conv2d(Conv2dOptions(output_filters.back(), filters, k)
                                   .stride({stride_x, stride_y})
                                   .padding(mdef.count("pad") ? (k / 2) : 0)
                                   .groups(mdef.count("groups") ? std::atoi(mdef["groups"].c_str()) : 1)
                                   .bias(!bn)));
                if (upsamplePrev)
                {
                   filters *= 2;
                   upsamplePrev = false;
                }
             }
//             else
//             { // multiple-size conv
//                modules->push_back("MixConv2d",
//                                    MixConv2d(in_ch = output_filters[-1], out_ch = filters, k = k, stride = stride, bias = not bn));
//             }

             if (bn)
             {
                modules->push_back("BatchNorm2d", BatchNorm2d(BatchNorm2dOptions{filters}.momentum(momentum).eps(eps)));
             }
             else
             {
                // ???
                //routs.append(i);
             }

             if (mdef["activation"] == "leaky")
             {
               // activation study https://github.com/ultralytics/yolov3/issues/441
                modules->push_back("activation", LeakyReLU(LeakyReLUOptions{}.negative_slope(0.1).inplace(true)));
             }
             else if (mdef["activation"] == "swish")
             {
                modules->push_back("activation", Swish());
             }
             else if (mdef["activation"] == "mish")
             {
                modules->push_back("activation", Mish());
             }
             else if (mdef["activation"] == "relu")
             {
                modules->push_back("activation", ReLU(ReLUOptions{}.inplace(true)));
             }
             else if (mdef["activation"] == "logistic")
             {
                modules->push_back("activation", Sigmoid{});
             }
          }
          else if (type == "BatchNorm2d")
          {
             filters = output_filters.back();
             modules->push_back(BatchNorm2d(BatchNorm2dOptions{filters}.momentum(0.03).eps(1e-4)));
             //moduleList->push_back(BatchNorm2d(BatchNorm2dOptions{filters}.momentum(0.03).eps(1e-4)));
             if ((i == 0) && (filters == 3))
             {
                //modules->runnimg_mean = Tensor({0.485, 0.456, 0.406});
                //modules->running_var = Tensor({0.0524, 0.0502, 0.0506});
             }
          }
          else if (type == "maxpool")
          {
             auto k = std::atoi(mdef["size"].c_str());
             auto stride = std::atoi(mdef["stride"].c_str());
             if ((k == 2) && (stride == 1)) // yolov3-tiny
             {
                modules->push_back("ZeroPad2d", ZeroPad2d(ZeroPad2dOptions{{0, 1, 0, 1}}));
                //moduleList->push_back(ZeroPad2d(ZeroPad2dOptions{{0, 1, 0, 1}}));
             }
             //moduleList->push_back(MaxPool2d(MaxPool2dOptions{k}.stride(stride).padding((k - 1) / 2)));
             modules->push_back("MaxPool2d", MaxPool2d(MaxPool2dOptions{k}.stride(stride).padding((k - 1) / 2)));
          }
          else if (type == "upsample")
          {
//             if (ONNX_EXPORT) // explicitly state size, avoid scale_factor
//             {
//                auto g = ((yolo_index + 1) * 2) / 32; // gain
//                // img_size = (320, 192)
//                modules->push_back(Upsample{UpsampleOptions{size=tuple(int(x * g) for x in img_size)}});
//             }
//             else
             {
                double scale_factor = std::atof(mdef["stride"].c_str());
                //moduleList->push_back(Upsample{UpsampleOptions{}.scale_factor({{scale_factor}})});
                modules->push_back("Upsample", Upsample{UpsampleOptions{}.scale_factor({{scale_factor, scale_factor}}).mode(torch::kBilinear).align_corners(true)});
                upsamplePrev = true;
                //modules->push_back("Conv2d", Conv2d{Conv2dOptions{output_filters.back(), filters/2, 3}.stride({1,1}).padding({1, 1})});
             }
          }
          else if (type == "convolution_transpose")
          {
             modules->push_back("ConvTranspose2d", ConvTranspose2d(ConvTranspose2dOptions{output_filters.back(), filters/2, 2}.stride(2)));
          }
          else if (type == "route") // Sequential() placeholder for 'route' layer
          {
             std::vector<std::string> layersStr;
             boost::split(layersStr, mdef["layers"], boost::is_any_of(","));
             std::vector<int32_t> layers;
             for (auto layerStr : layersStr)
             {
                layers.emplace_back(std::atoi(layerStr.c_str()));
             }
             auto sum = 0;
             for (auto l : layers)
             {
                sum += output_filters[l > 0 ? l + 1 : i + l];
                routs.push_back(l < 0 ? i + l - 1 : l - 1);
             }
             for (auto& l : layers)
             {
                l += i - 1;
             }
             moduleList->push_back(FeatureConcat(layers));
          }
          else if (type == "shortcut") // Sequential() placeholder for 'shortcut' layer
          {
             //layers = mdef['from']
             //filters = output_filters[-1]
             //routs.extend([i + l if l < 0 else l for l in layers])
             //modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
          }
          else if (type == "reorg3d") // yolov3-spp-pan-scale
          {
             //pass
          }
          else if (type == "yolo")
          {
             // TODO:
          }
          else if (type == "dropout")
          {
             auto perc = std::atof(mdef["probability"].c_str());
             modules->push_back(Dropout{DropoutOptions{perc}});
          }
          else
          {
             std::cout << "Warning: Unrecognized Layer Type: " << type << std::endl;
          }
          // Register module list and number of output filters
          if (!modules->is_empty())
          {
             moduleList->push_back(modules);
          }
          output_filters.push_back(upsamplePrev ? filters*2 : filters);
       }
       auto routs_binary = std::vector<bool>(mdefs.size() + 1, false);
       for (auto const& i : routs)
       {
          routs_binary[i] = true;
       }
       return std::make_pair(moduleList, routs_binary);
    }
private:
    std::string _path;
    std::vector<std::pair<std::string, std::map<std::string, std::string>>> mdefs;
};

struct DarknetImpl : torch::nn::Module
{
   DarknetImpl(std::string const& cfg, cv::Size size, bool verbose = false)
   {
      std::tie(_moduleList, _routs) = DarknetParser(cfg).createModules(size, cfg);
      register_module("unet", _moduleList);
   }

   auto forward(torch::Tensor x) -> torch::Tensor
   {
      using namespace torch::nn;

      auto img_size = cv::Size(x.size(2), x.size(3));
      std::vector<torch::Tensor> outs;
      for (auto i = 0; i < _moduleList->size(); ++i)
      {
         if (auto featureConcat = std::dynamic_pointer_cast<FeatureConcatImpl>(_moduleList[i]))
         {
//            auto l = featureConcat->_layers;
//            std::cout << "Concat: ";
//            for (auto const& item : l)
//            {
//               std::cout << item << ", ";
//            }
//            std::cout << std::endl;
            x = featureConcat->forward(x, outs);
         }
         else if (auto sequential = std::dynamic_pointer_cast<SequentialImpl>(_moduleList[i]))
         {
            x = sequential->forward(x);
         }
         outs.push_back(_routs[i] ? x : torch::Tensor{});
      }
      return x;
   }

   void load_weights(std::string const& darknetWeightsFile)
   {
      using namespace torch::nn;

      auto file = std::ifstream(darknetWeightsFile, std::ios::binary);
      int32_t version[3] = {};
      int64_t seen = {};
      file.read((char*)&version, sizeof(int32_t)*3);
      file.read((char*)&seen, sizeof(int64_t));
      for (int i = 0; i < _moduleList->size(); ++i)
      {
         auto submodules = _moduleList[i]->modules(false);
         for (auto j = 0; j < submodules.size(); j++)
         {
            //std::cout << submodules[j]->name() << std::endl;
            if (submodules[j]->name() == "torch::nn::Conv2dImpl")
            {
               //std::cout << "Conv2d: " << std::endl;
               bool bnFlag = false;
               if (((j + 1) < submodules.size()) &&
                   (submodules[j + 1]->name() == "torch::nn::BatchNorm2dImpl"))
               {
                  auto bn = std::dynamic_pointer_cast<BatchNorm2dImpl>(submodules[j + 1]);
                  if (bn != nullptr)
                  {
                     //std::cout << "\tLoading batch norm 2d" << std::endl;
                     auto batchNorm2dBias = bn->bias.data().cpu();
                     auto batchNorm2dWeights = bn->weight.data().cpu();
                     auto batchNorm2dRunningMean = bn->running_mean.data().cpu();
                     auto batchNorm2dRunningVar = bn->running_var.data().cpu();
                     //std::cout << "\t\tbatchNorm2dBias.nbytes(): " << batchNorm2dBias.nbytes() << std::endl;
                     //std::cout << "\t\tbatchNorm2dWeights.nbytes(): " << batchNorm2dWeights.nbytes() << std::endl;
                     //std::cout << "\t\tbatchNorm2dRunningMean.nbytes(): " << batchNorm2dRunningMean.nbytes() << std::endl;
                     //std::cout << "\t\tbatchNorm2dRunningVar.nbytes(): " << batchNorm2dRunningVar.nbytes() << std::endl;
                     file.readsome((char*)batchNorm2dBias.data_ptr(), batchNorm2dBias.nbytes());
                     file.readsome((char*)batchNorm2dWeights.data_ptr(), batchNorm2dWeights.nbytes());
                     file.readsome((char*)batchNorm2dRunningMean.data_ptr(), batchNorm2dRunningMean.nbytes());
                     file.readsome((char*)batchNorm2dRunningVar.data_ptr(), batchNorm2dRunningVar.nbytes());
                     bnFlag = true;
                  }
               }
               auto conv2d = std::dynamic_pointer_cast<Conv2dImpl>(submodules[j]);
               if (!bnFlag)
               {
                  //std::cout << "\tLoading bias: " << std::endl;
                  auto conv2dBias = conv2d->bias.data().cpu();
                  //std::cout << "\t\tconv2dBias.nbytes(): " << conv2dBias.nbytes() << std::endl;
                  file.readsome((char*)conv2dBias.data_ptr(), conv2dBias.nbytes());
               }
               //std::cout << "\tLoading weights: " << std::endl;
               auto conv2dWeights = conv2d->weight.data().cpu();
               //std::cout << "\t\tconv2dWeights.nbytes(): " << conv2dWeights.nbytes() << std::endl;
               file.readsome((char*)conv2dWeights.data_ptr(), conv2dWeights.nbytes());
            }
         }
      }
   }

   void save_weights(std::string const& darknetWeightsFile)
   {
      using namespace torch::nn;

      auto file = std::ofstream(darknetWeightsFile, std::ios::binary);
      int32_t version[3] = {0, 2, 5};
      int64_t seen = {0};
      file.write((char*)&version, sizeof(int32_t)*3);
      file.write((char*)&seen, sizeof(int64_t));
      for (int i = 0; i < _moduleList->size(); ++i)
      {
         auto submodules = _moduleList[i]->modules(false);
         for (auto j = 0; j < submodules.size(); j++)
         {
            //std::cout << submodules[j]->name() << std::endl;
            if (submodules[j]->name() == "torch::nn::Conv2dImpl")
            {
               //std::cout << "Conv2d: " << std::endl;
               bool bnFlag = false;
               if (((j + 1) < submodules.size()) &&
                   (submodules[j + 1]->name() == "torch::nn::BatchNorm2dImpl"))
               {
                  auto bn = std::dynamic_pointer_cast<BatchNorm2dImpl>(submodules[j + 1]);
                  if (bn != nullptr)
                  {
                     //std::cout << "\tSaving batch norm 2d" << std::endl;
                     auto batchNorm2dBias = bn->bias.data().cpu();
                     auto batchNorm2dWeights = bn->weight.data().cpu();
                     auto batchNorm2dRunningMean = bn->running_mean.data().cpu();
                     auto batchNorm2dRunningVar = bn->running_var.data().cpu();
                     //std::cout << "\t\tbatchNorm2dBias.nbytes(): " << batchNorm2dBias.nbytes() << std::endl;
                     //std::cout << "\t\tbatchNorm2dWeights.nbytes(): " << batchNorm2dWeights.nbytes() << std::endl;
                     //std::cout << "\t\tbatchNorm2dRunningMean.nbytes(): " << batchNorm2dRunningMean.nbytes() << std::endl;
                     //std::cout << "\t\tbatchNorm2dRunningVar.nbytes(): " << batchNorm2dRunningVar.nbytes() << std::endl;
                     file.write((char*)batchNorm2dBias.data_ptr(), batchNorm2dBias.nbytes());
                     file.write((char*)batchNorm2dWeights.data_ptr(), batchNorm2dWeights.nbytes());
                     file.write((char*)batchNorm2dRunningMean.data_ptr(), batchNorm2dRunningMean.nbytes());
                     file.write((char*)batchNorm2dRunningVar.data_ptr(), batchNorm2dRunningVar.nbytes());
                     bnFlag = true;
                  }
               }
               auto conv2d = std::dynamic_pointer_cast<Conv2dImpl>(submodules[j]);
               if (!bnFlag)
               {
                  //std::cout << "\tSaving bias: " << std::endl;
                  auto conv2dBias = conv2d->bias.data().cpu();
                  //std::cout << "\t\tconv2dBias.nbytes(): " << conv2dBias.nbytes() << std::endl;
                  file.write((char*)conv2dBias.data_ptr(), conv2dBias.nbytes());
               }
               //std::cout << "\tSaving weights: " << std::endl;
               auto conv2dWeights = conv2d->weight.data().cpu();
               //std::cout << "\t\tconv2dWeights.nbytes(): " << conv2dWeights.nbytes() << std::endl;
               file.write((char*)conv2dWeights.data_ptr(), conv2dWeights.nbytes());
            }
         }
      }
   }

   torch::nn::ModuleList _moduleList;
   std::vector<bool>     _routs;
};
TORCH_MODULE_IMPL(Darknet, DarknetImpl);