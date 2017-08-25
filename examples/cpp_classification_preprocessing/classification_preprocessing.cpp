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

#include "../cpp_classification/Serializers.h"

#ifdef USE_OPENCV
using namespace caffe;
using std::string;
  
cv::Size input_geometry_(227, 227);

#define IP "10.10.49.94"
#define PORT "12345"

#define RAW_SAMPLES (0)
#define PREPROCESSED_SAMPLES (1000000000UL)

#if 0
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
#endif

int num_channels = 3;
cv::Mat preprocess(const cv::Mat& img, const cv::Mat& mean) {
  cv::Mat sample;

  if (img.channels() == 3 && num_channels == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean, sample_normalized);

  return sample_normalized;
}

uint64_t get_time_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux
    return ts.tv_nsec + ts.tv_sec * 1000000000UL;
}

cv::Mat getMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  return cv::Mat(input_geometry_, mean.type(), channel_mean);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0]
              << " mean_file"
              << std::endl;
    return 1;
  }
    
  image_serializer ser;
  image_deserializer deser;

  cirrus::TCPClient client;
  cirrus::ostore::FullBladeObjectStoreTempl<cv::Mat>
      image_store(IP, PORT, &client, ser, deser);

  ::google::InitGoogleLogging(argv[0]);
  
  string mean_file = argv[1];
  cv::Mat mean = getMean(mean_file);

  uint64_t count = 0;
  while (1) {
      std::cout << "---------- Preprocessing sample #"
          << count << " ----------" << std::endl;

      auto now = get_time_ns();
      cv::Mat img = image_store.get(0);
      auto elapsed_store = get_time_ns() - now;

      CHECK(!img.empty()) << "Unable to decode image ";

      image_store.put(PREPROCESSED_SAMPLES + count, preprocess(img, mean));

      count++;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
