#ifndef _STRING_SERIALIZERS_H_
#define _STRING_SERIALIZERS_H_

class string_serializer : public cirrus::Serializer<std::string> {
    public:
        uint64_t size(const std::string& img) const override {
            return img.size();
        }   

        void serialize(const std::string& img, void* mem) const override {
            std::memcpy(mem, img.data(), img.size());
        }   
    private:
};

class string_deserializer {
    public:
        std::string operator()(const void* data, unsigned int des_size) {
            return std::string((char*)data, (char*)data + des_size);
        }

    private:
};


#endif  // _SERIALIZERS_H_
