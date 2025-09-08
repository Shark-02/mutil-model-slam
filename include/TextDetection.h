#pragma once
#include "ImageDetections.h"
#include <string>
#include <vector>
#include <memory>

namespace ORB_SLAM2 {

class TextDetectionsFromFile : public ImageDetectionsManager {
public:
    TextDetectionsFromFile(const std::string &folder) : folder_(folder) {}

    std::vector<Detection::Ptr> detect(const std::string &image_filename) const override;

    std::vector<Detection::Ptr> detect(unsigned int idx) const override {
        std::cerr << "This function is not available for TextDetectionsFromFile\n";
        return {};
    }
    std::vector<Detection::Ptr> detect(cv::Mat img) const override {
        std::cerr << "This function is not available for TextDetectionsFromFile\n";
        return {};
    }

private:
    std::string folder_;
};

} // namespace ORB_SLAM2
