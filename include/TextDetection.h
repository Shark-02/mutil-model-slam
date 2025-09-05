// TextDetectionsFromFile.h
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM2 {

struct TextDetection {
    using Ptr = std::shared_ptr<TextDetection>;
    std::vector<cv::Point2f> quad;  // 四边形顶点
    std::string text;
    float score;
};

class TextDetectionsFromFile {
public:
    TextDetectionsFromFile(const std::string &folder);

    std::vector<TextDetection::Ptr> detect(const std::string &image_filename) const;

private:
    std::string folder_;
};

}
