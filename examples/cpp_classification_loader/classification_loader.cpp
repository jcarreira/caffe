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

#include "object_store/FullBladeObjectStore.h"
#include "utils/Stats.h"
#include "client/BladeClient.h"
#include "client/TCPClient.h"
#include "cache_manager/CacheManager.h"
#include "cache_manager/LRAddedEvictionPolicy.h"

/**
 * This code loads raw image bytes to the Cirrus store
 */

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
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

            std::cout << "Serializing."
                << " rows: " << img.rows
                << " cols: " << img.cols
                << " type: " << img.type()
                << " data_size: " << data_size
                << std::endl;

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
            
            std::cout << "Deserializing."
                << " rows: " << rows
                << " cols: " << cols
                << " type: " << type
                << " data_size: " << data_size
                << std::endl;

            return cv::Mat(rows, cols, type, m);
        }

    private:
};

uint64_t image_checksum(const cv::Mat& m) {
    uint64_t res = 0;
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            res += m.at<ushort>(i, j);
        }
    }
    return res % 1337;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
            << "img.jpg" << std::endl;
        return 1;
    }

    image_serializer ser;
    image_deserializer deser;

    cirrus::TCPClient client;
    cirrus::ostore::FullBladeObjectStoreTempl<cv::Mat>
        image_store(IP, PORT, &client, ser, deser);

    ::google::InitGoogleLogging(argv[0]);

    string file = argv[1];

    std::cout << "Loading file: "
        << file << " ----------"
        << std::endl;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;

    if (!img.isContinuous()) {
        std::cout << "Error: img is not continuous" << std::endl;
    }

    std::cout << "Image element type: " << img.type() << std::endl;
    std::cout << "Loaded raw data" << std::endl;

    std::cout << "Storing image with checksum: "
        << image_checksum(img)
        << std::endl;

    for (int i = 0; i < 10000; ++i) {
        image_store.put(0, img);
    }
        
    std::cout << "Image stored. Getting it again"
        << std::endl;

    auto img2 = image_store.get(0);
    std::cout << "Received image with checksum: " << image_checksum(img2)
        << std::endl;

    return 0;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
    return 0;
}
#endif  // USE_OPENCV
