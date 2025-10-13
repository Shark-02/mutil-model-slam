// TextMatcher.h

#ifndef TEXTMATCHER_H
#define TEXTMATCHER_H

#include "Frame.h"
#include "MapText.h"
#include <vector>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM2
{

enum class ProjectionMode {
    BOUNDING_BOX,
    HOMOGRAPHY
};

class TextMatcher
{
public:
    /**
     * @brief 构造函数
     * @param znccThreshold ZNCC外观相似度阈值，低于此值认为可能被遮挡
     * @param minIoU 几何重叠度阈值，低于此值认为位置不匹配
     * @param minTextSimilarity 文本内容相似度阈值，低于此值认为内容不匹配
     */
    TextMatcher(float znccThreshold = 0.4f, float minIoU = 0.3f, float minTextSimilarity = 0.6f);

    // 几何投影辅助函数
    bool ProjectTextToImage(MapText* pMT, const cv::Mat& Tcw, const cv::Mat& K, 
                            cv::Rect2f& out_bbox, const Frame& F);
    
    // 外观验证辅助函数
    float CalculateZNCC(const cv::Mat& img1, const cv::Rect2f& roi1, 
                        const cv::Mat& img2, const cv::Rect2f& roi2);

    /**
     * @brief 主函数：通过三重验证来跟踪地图中的文本地标
     * @param F 当前帧
     * @param vpMapTexts 视野内可见的候选3D文本地标
     * @return int 成功匹配的数量
     */
    int SearchTextByProjection(Frame &F, const std::vector<MapText*> &vpMapTexts,ProjectionMode mode=ProjectionMode::HOMOGRAPHY);

private:

    // 几何关联辅助函数
    float CalculateIoU(const cv::Rect2f& r1, const cv::Rect2f& r2);

    // 语义关联辅助函数
    int LevenshteinDistance(const std::string &s1, const std::string &s2);

private:
    float mfZNCCThreshold;
    float mfMinIoU;
    float mfMinTextSimilarity;
};

} // namespace ORB_SLAM2

#endif // TEXTMATCHER_H