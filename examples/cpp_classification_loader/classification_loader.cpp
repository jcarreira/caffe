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
  * array of values (doubles)
  */
class image_serializer : public cirrus::Serializer<cv::Mat> {
    public:
        uint64_t size(const cv::Mat& img) const override {
            uint64_t rows = img.rows;
            uint64_t cols = img.cols;

            uint64_t size = sizeof(uint64_t) * 2 + rows * cols * sizeof(double);
            return size;
        }   

        void serialize(const cv::Mat& img, void* mem) const override {
            uint64_t* ptr = reinterpret_cast<uint64_t*>(mem);
            *ptr++ = img.rows;
            *ptr++ = img.cols;
            //XXX fix

            memcpy(ptr, img.data, img.total() * img.elemSize());
        }   
    private:
};

class image_deserializer {
    public:
        cv::Mat operator()(const void* data, unsigned int des_size) {

            //XXX fix
        }

    private:
};


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

    //int rows = img.rows;
    //int cols = img.cols;

    //uint64_t size = sizeof(uint64_t) * 2 + sizeof(double) * rows * cols;
    //std::shared_ptr<char> p(new char[size]);

    //double* d = p.get();
    //for (int r = 0; r < rows; ++r) {
    //    for (int c = 0; c < cols; ++c) {
    //        d[r * cols + c] = img.at(r, c);
    //    }
    //}

    std::cout << "Loaded raw data" << std::endl;

    //image_store.put(0, p);

    return 0;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
    return 0;
}
#endif  // USE_OPENCV
