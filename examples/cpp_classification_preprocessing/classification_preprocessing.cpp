#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

#include "object_store/FullBladeObjectStore.h"
#include "utils/Stats.h"
#include "client/BladeClient.h"
#include "client/TCPClient.h"
#include "cache_manager/CacheManager.h"
#include "cache_manager/LRAddedEvictionPolicy.h"

#ifdef USE_OPENCV
using namespace caffe;
using std::string;

#define IP "10.10.49.94"
#define PORT "12345"

/** Format:
  * rows (uint64_t)
  * cols (uint64_t)
  * type (uint64_t)
  * data size (uint64_t)
  * raw data
  */
class image_serializer : public cirrus::Serializer<cv::Mat> {
    public:
        uint64_t size(const cv::Mat& img) const override {
            uint64_t data_size = img.total() * img.elemSize();
            uint64_t total_size = sizeof(uint64_t) * 4 + data_size;
            return total_size;
        }   

        void serialize(const cv::Mat& img, void* mem) const override {
            uint64_t* ptr = reinterpret_cast<uint64_t*>(mem);
            *ptr++ = img.rows;
            *ptr++ = img.cols;
            *ptr++ = img.type();

            uint64_t data_size = img.total() * img.elemSize();
            *ptr++ = data_size;

            //std::cout << "Serializing."
            //    << " rows: " << img.rows
            //    << " cols: " << img.cols
            //    << " type: " << img.type()
            //    << " data_size: " << data_size
            //    << std::endl;

            memcpy(ptr, img.data, data_size);
        }   
    private:
};

class image_deserializer {
    public:
        cv::Mat operator()(const void* data, unsigned int des_size) {
            const uint64_t* ptr = reinterpret_cast<const uint64_t*>(data);
            uint64_t rows = *ptr++;
            uint64_t cols = *ptr++;
            uint64_t type = *ptr++;
            uint64_t data_size = *ptr++;

            if (des_size != data_size + sizeof(double) * 4) {
                throw std::runtime_error("Wrong size");
            }

            void* m = new char[data_size];
            std::memcpy(m, ptr, data_size);
            
            //std::cout << "Deserializing."
            //    << " rows: " << rows
            //    << " cols: " << cols
            //    << " type: " << type
            //    << " data_size: " << data_size
            //    << std::endl;

            return cv::Mat(rows, cols, type, m);
        }

    private:
};

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

uint64_t get_time_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux
    return ts.tv_nsec + ts.tv_sec * 1000000000UL;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }
    
  image_serializer ser;
  image_deserializer deser;

  cirrus::TCPClient client;
  cirrus::ostore::FullBladeObjectStoreTempl<cv::Mat>
      image_store(IP, PORT, &client, ser, deser);

  ::google::InitGoogleLogging(argv[0]);

  uint64_t count = 0;
  while (1) {
      std::cout << "---------- Preprocessing for "
          << count++ << " ----------" << std::endl;

      auto now = get_time_ns();
      cv::Mat img = image_store.get(0);
      auto elapsed_store = get_time_ns() - now;

      CHECK(!img.empty()) << "Unable to decode image ";
      
      now = get_time_ns();
      std::vector<Prediction> predictions = classifier.Classify(img);
      auto elapsed_pred = get_time_ns() - now;
      
      std::cout << "Elapsed (ns). Store: " << elapsed_store
          << " pred: " << elapsed_pred << std::endl;
  
      /* Print the top N predictions. */
      for (size_t i = 0; i < predictions.size(); ++i) {
          Prediction p = predictions[i];
          std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
      }
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
