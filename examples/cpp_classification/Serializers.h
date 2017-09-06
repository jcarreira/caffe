#ifndef _SERIALIZERS_H_
#define _SERIALIZERS_H_

#include <common/Serializer.h>

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

#endif  // _SERIALIZERS_H_
