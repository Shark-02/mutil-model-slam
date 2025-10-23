#ifndef MAPTEXT_H
#define MAPTEXT_H

#include <vector>
#include <map>
#include <mutex>
#include <string>

#include <Eigen/Dense>

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"


namespace ORB_SLAM2
{

// Forward declarations
class KeyFrame;
class Frame;
class Map;

// 文本状态枚举
enum TextStatus {
    TEXTGOOD,
    TEXTIMMATURE,
    TEXTBAD
};

// 存储文本的语义信息和置信度
struct TextInfo {
    std::string mean;
    double score;
    double score_semantic; // 用于视图选择的综合得分

    TextInfo() : mean(""), score(0.0), score_semantic(99999.9) {}
};


class MapText
{
public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 如果你在类中直接使用Eigen固定大小的成员，需要这个宏

    // 构造函数 (适配ORB-SLAM2/V-OOM的版本)
    MapText(KeyFrame* pRefKF, const Eigen::Vector4d& theta, 
            const TextInfo& textInfo, Map* pMap);

    // --- 核心功能 ---
    
    // 设置/获取位姿相关
    Eigen::Vector4d GetWorldPlane(); // 获取世界坐标系下的平面方程
    KeyFrame* GetReferenceKeyFrame();

    // 观测管理
    void AddObservation(KeyFrame* pKF, size_t idx);
    void EraseObservation(KeyFrame* pKF);
    std::map<KeyFrame*, size_t> GetObservations();
    int GetIndexInKeyFrame(KeyFrame* pKF);
    std::vector<cv::Mat> GetWorldBoundingBox();
    std::vector<cv::Point2f> ProjectByHomography(const Frame& F);

    int Observations();

    // 状态管理
    void SetBadFlag();
    bool isBad();

    // --- TODO: 高级功能接口 ---

    // 更新语义信息：如果新的 TextInfo 更优（score_semantic 更低），则替换
    void UpdateSemanticInfo(const TextInfo& newInfo);
    TextStatus GetState() const;
    void SetState(const TextStatus &s);

    // 地标融合：标记为被替换
    void SetReplaced(MapText* pMT);
    bool IsInKeyFrame(KeyFrame* pKF);

    // 获取文本内容（便于语义匹配）
    std::string GetText() const;
    TextInfo GetTextInfo() const;

    // 用于回环检测后的地标融合
    void Replace(MapText* pMT);
    MapText* GetReplaced();

public:
    // --- 公共成员变量 (遵循ORB-SLAM2风格) ---
    long unsigned int mnId;                     // 全局唯一ID
    static long unsigned int nNextId;

    // 状态
    TextStatus mState;                          // 地标状态 (好/不成熟/坏)
    
    // 观测信息
    int mnVisible;                              // 在跟踪中可见的次数
    int mnFound;                                // 在跟踪中匹配上的次数

    // 语义信息
    TextInfo mTextInfo;

    long long mnFirstKFid;  // 记录创建时的关键帧ID

protected:
    // --- 内部数据与锁 ---

    // 几何信息: 存储在参考关键帧坐标系下的平面方程 [nx, ny, nz, d]
    // n'x + d = 0
    Eigen::Vector4d mThetaRef;

    // 参考关键帧
    KeyFrame* mpRefKF;
    
    // 观测列表
    std::map<KeyFrame*, size_t> mObservations;

    // 状态标志
    bool mbBad;                                 // 是否已被标记为坏点

    // TODO: 用于地标融合的指针
    MapText* mpReplaced;

    // 全局地图指针
    Map* mpMap;

    // 线程锁
    mutable std::mutex mMutexFeatures;
    std::mutex mMutexPose;
};


} // namespace ORB_SLAM2

#endif // MAPTEXT_H