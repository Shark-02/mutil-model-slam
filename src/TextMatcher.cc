// TextMatcher.cc

#include "TextMatcher.h"
#include <opencv2/imgproc/imgproc.hpp> // For cv::matchTemplate and cv::resize
#include <algorithm> // For std::min/max

namespace ORB_SLAM2
{

TextMatcher::TextMatcher(float znccThreshold, float minIoU, float minTextSimilarity)
    : mfZNCCThreshold(znccThreshold), mfMinIoU(minIoU), mfMinTextSimilarity(minTextSimilarity)
{
}

// TextMatcher.cc

int TextMatcher::SearchTextByProjection(Frame &F, const std::vector<MapText*> &vpMapTexts,ProjectionMode mode)
{
    int nMatches = 0;  
    // 使用一个布尔向量来确保每个2D检测框只被匹配一次
    std::vector<bool> vbMatchedDetection(F.vTextDetections.size(), false);
    /*std::cout << "[TextMatcher] Frame " << F.mnId
              << " #detections=" << F.vTextDetections.size()
              << " #candidates=" << vpMapTexts.size() << std::endl;*/

    // 遍历所有可见的3D文本地标
    for (MapText* pMT : vpMapTexts)
    {
        if (!pMT || pMT->isBad())
            continue;
        /*std::cout << "[TextMatcher] trying MapText " << pMT->mnId
                  << " text='" << pMT->mTextInfo.mean << "'" << std::endl;*/

        // 阶段一: 几何投影
        cv::Rect2f proj_bbox;
        bool ok = false;
        if (mode == ProjectionMode::BOUNDING_BOX)
        {
            // 用 MapText::GetWorldBoundingBox()
            ok = ProjectTextToImage(pMT, F.mTcw, F.mK, proj_bbox, F);
        }
        else if (mode == ProjectionMode::HOMOGRAPHY)
        {
            // 用单应性投影
            auto proj_corners = pMT->ProjectByHomography(F);
            if (proj_corners.size() == 4) {
                float minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
                for (auto &pt : proj_corners) {
                    minx = std::min(minx, pt.x);
                    miny = std::min(miny, pt.y);
                    maxx = std::max(maxx, pt.x);
                    maxy = std::max(maxy, pt.y);
                }
                proj_bbox = cv::Rect2f(minx, miny, maxx - minx, maxy - miny);
                ok = true;
            }
        }

        if (!ok) {
            std::cout << "  -> projection failed / not in frustum\n";
            continue;
        }

        /*std::cout << "  -> proj_bbox: [" << proj_bbox.width << " x " << proj_bbox.height
                  << " from (" << proj_bbox.x << "," << proj_bbox.y << ")]" << std::endl;*/


        // 阶段二: 外观验证 (ZNCC预筛选)
        KeyFrame* pRefKF = pMT->GetReferenceKeyFrame();
        if(!pRefKF) continue;
        size_t ref_det_idx = pMT->GetIndexInKeyFrame(pRefKF);
        if(ref_det_idx >= pRefKF->vTextDetections.size()) continue;
        const auto& ref_det_bbox_vec = pRefKF->vTextDetections[ref_det_idx]->bbox;
        cv::Rect2f ref_bbox(ref_det_bbox_vec[0], ref_det_bbox_vec[1], ref_det_bbox_vec[2] - ref_det_bbox_vec[0], ref_det_bbox_vec[3] - ref_det_bbox_vec[1]);
        //TODO 暂时注释zccc校验
        //float zncc_score = CalculateZNCC(pRefKF->mImGray, ref_bbox, F.mImGray, proj_bbox);
        //if (zncc_score < mfZNCCThreshold)
            //continue;

        // 阶段三: 数据关联 (IoU + 语义)
        float max_score = 0.0f;
        int best_match_idx = -1;

        for (size_t i = 0; i < F.vTextDetections.size(); ++i)
        {
            if (vbMatchedDetection[i]) continue;

            const auto& det = F.vTextDetections[i];
            const auto& det_bbox_vec = det->bbox;
            cv::Rect2f det_bbox(det_bbox_vec[0], det_bbox_vec[1], det_bbox_vec[2] - det_bbox_vec[0], det_bbox_vec[3] - det_bbox_vec[1]);

            float iou = CalculateIoU(proj_bbox, det_bbox);
            //std::cout << "    det " << i << " iou=" << iou << std::endl;
            //std::cout << "    det " << det->text << std::endl;
            //if (iou < mfMinIoU) continue;

            float text_sim = 1.0f - (float)LevenshteinDistance(pMT->mTextInfo.mean, det->text) / std::max(pMT->mTextInfo.mean.length(), det->text.length());
            //std::cout << " text_sim=" << text_sim << std::endl;
            if(iou < mfMinIoU && text_sim < mfMinTextSimilarity) continue;
            //if (text_sim < mfMinTextSimilarity) continue;


            float current_score = iou + text_sim;
            //std::cout << " current_score=" << current_score << std::endl;

            if (current_score > max_score)
            {
                max_score = current_score;
                best_match_idx = i;
                /*std::cout << " max_score=" << max_score << std::endl;
                std::cout << " best_match_idx=" << best_match_idx << std::endl;*/
            }
        }

        // 如果找到了有效的最佳匹配
        if (best_match_idx != -1)
        {
            F.mvpMapTexts[best_match_idx] = pMT;

            vbMatchedDetection[best_match_idx] = true; // 标记此检测框已被使用
            nMatches++;
            /*std::cout << "Frame " << F.mnId << " matched detection " << best_match_idx 
          << " with MapText " << pMT->mnId << std::endl;*/
        }
    }
    return nMatches;
}


bool TextMatcher::ProjectTextToImage(MapText* pMT, const cv::Mat& Tcw, const cv::Mat& K, 
                                     cv::Rect2f& out_bbox, const Frame& F)
{
    //优先使用 MapText 的持久化 world corners（3x1 CV_32F）
    std::vector<cv::Mat> worldCorners = pMT->GetWorldCorners();
    if (!worldCorners.empty())
    {
        float min_x = static_cast<float>(F.mnMaxX), max_x = static_cast<float>(F.mnMinX);
        float min_y = static_cast<float>(F.mnMaxY), max_y = static_cast<float>(F.mnMinY);
        bool bInFrustum = false;

        cv::Mat R = Tcw.rowRange(0,3).colRange(0,3);
        cv::Mat t = Tcw.rowRange(0,3).col(3);

        for (const auto& p3D_world_raw : worldCorners)
        {
            if (p3D_world_raw.empty()) continue;
            cv::Mat p3D_world = p3D_world_raw;
            // defensive checks
            if (p3D_world.rows < 3) continue;
            float X = p3D_world.at<float>(0,0);
            float Y = p3D_world.at<float>(1,0);
            float Z = p3D_world.at<float>(2,0);
            if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z)) continue;

            // world -> camera
            cv::Mat p3D_cam = R * p3D_world + t;
            float z = p3D_cam.at<float>(2,0);
            if (!std::isfinite(z) || z <= 0.02f) continue;

            const float inv_z = 1.0f / z;
            const float u = K.at<float>(0, 0) * p3D_cam.at<float>(0,0) * inv_z + K.at<float>(0, 2);
            const float v = K.at<float>(1, 1) * p3D_cam.at<float>(1,0) * inv_z + K.at<float>(1, 2);

            if (u >= F.mnMinX - 1 && u <= F.mnMaxX + 1 && v >= F.mnMinY - 1 && v <= F.mnMaxY + 1)
            {
                bInFrustum = true;
                min_x = std::min(min_x, u); max_x = std::max(max_x, u);
                min_y = std::min(min_y, v); max_y = std::max(max_y, v);
            }
        }
        if (!bInFrustum) return false;
        if (max_x < min_x) std::swap(max_x, min_x);
        if (max_y < min_y) std::swap(max_y, min_y);

        // clip to image bounds
        min_x = std::max(min_x, static_cast<float>(F.mnMinX));
        min_y = std::max(min_y, static_cast<float>(F.mnMinY));
        max_x = std::min(max_x, static_cast<float>(F.mnMaxX));
        max_y = std::min(max_y, static_cast<float>(F.mnMaxY));

        if ((max_x - min_x) < 1.0f || (max_y - min_y) < 1.0f) return false;

        out_bbox = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
        return true;
    }
    //若没有持久化角点，退回到基于单应或射线-平面交的逻辑
    std::vector<cv::Mat> bbox3D_world = pMT->GetWorldBoundingBox();
    if (bbox3D_world.empty()) return false;

    float min_x = F.mnMaxX, max_x = F.mnMinX;
    float min_y = F.mnMaxY, max_y = F.mnMinY;
    bool bInFrustum = false;

    for (const auto& p3D_world : bbox3D_world)
    {
        cv::Mat p3D_cam = Tcw.rowRange(0, 3).colRange(0, 3) * p3D_world + Tcw.rowRange(0, 3).col(3);
        const float z = p3D_cam.at<float>(2);

        if (z < 0.1) continue;
        
        const float inv_z = 1.0f / z;
        const float u = K.at<float>(0, 0) * p3D_cam.at<float>(0) * inv_z + K.at<float>(0, 2);
        const float v = K.at<float>(1, 1) * p3D_cam.at<float>(1) * inv_z + K.at<float>(1, 2);

        if (u > F.mnMinX && u < F.mnMaxX && v > F.mnMinY && v < F.mnMaxY)
        {
            bInFrustum = true;
            min_x = std::min(min_x, u); max_x = std::max(max_x, u);
            min_y = std::min(min_y, v); max_y = std::max(max_y, v);
        }
    }

    if (!bInFrustum) return false;

    out_bbox = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
    return true;
}

float TextMatcher::CalculateIoU(const cv::Rect2f& r1, const cv::Rect2f& r2)
{
    cv::Rect2f intersection = r1 & r2;
    float intersection_area = intersection.area();
    if (intersection_area <= 0) return 0.0f;
    float union_area = r1.area() + r2.area() - intersection_area;
    return intersection_area / union_area;
}

int TextMatcher::LevenshteinDistance(const std::string &s1, const std::string &s2) {
    const std::size_t len1 = s1.size(), len2 = s2.size();
    std::vector<unsigned int> col(len2 + 1), prevCol(len2 + 1);
    for (unsigned int i = 0; i < prevCol.size(); i++) prevCol[i] = i;
    for (unsigned int i = 0; i < len1; i++) {
        col[0] = i + 1;
        for (unsigned int j = 0; j < len2; j++)
            col[j + 1] = std::min({ prevCol[j + 1] + 1, col[j] + 1, prevCol[j] + (s1[i] == s2[j] ? 0 : 1) });
        col.swap(prevCol);
    }
    return prevCol[len2];
}

float TextMatcher::CalculateZNCC(const cv::Mat& img1, const cv::Rect2f& roi1, 
                                 const cv::Mat& img2, const cv::Rect2f& roi2)
{
    // 检查ROI是否有效
    if(roi1.width <= 0 || roi1.height <= 0 || roi2.width <= 0 || roi2.height <= 0) return 0.0f;
    if(roi1.x < 0 || roi1.y < 0 || roi1.x + roi1.width > img1.cols || roi1.y + roi1.height > img1.rows) return 0.0f;
    if(roi2.x < 0 || roi2.y < 0 || roi2.x + roi2.width > img2.cols || roi2.y + roi2.height > img2.rows) return 0.0f;

    // 提取图像块
    cv::Mat patch1 = img1(roi1);
    cv::Mat patch2 = img2(roi2);

    if (patch1.empty() || patch2.empty()) return 0.0f;

    // 将两个图像块缩放到相同尺寸以便比较
    cv::Mat patch2_resized;
    cv::resize(patch2, patch2_resized, patch1.size());

    // 使用OpenCV的matchTemplate计算ZNCC
    cv::Mat result;
    cv::matchTemplate(patch1, patch2_resized, result, CV_TM_CCOEFF_NORMED);
    
    // result是一个1x1的矩阵，其值就是ZNCC得分
    return result.at<float>(0, 0);
}


} // namespace ORB_SLAM2