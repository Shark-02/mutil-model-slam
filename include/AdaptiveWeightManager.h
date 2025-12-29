/**
 * AdaptiveWeightManager.h
 * 
 * 自适应权重调度模块
 * 
 * 核心思想：基于"环境语义熵"动态调整点云、物体、文本三种模态的权重
 * 
 * 环境语义熵 H_m 综合考量：
 *   1. 数量密度 N_m：当前帧跟踪到的特征数量
 *   2. 观测置信度 C_m：检测置信度（文本OCR得分/物体检测得分）
 *   3. 空间分布广度 S_m：特征在图像平面的分布均匀度（网格化香农熵）
 * 
 * 权重映射：使用Sigmoid门控函数 w_m = w_base + (w_max - w_base) * sigmoid(k*(H_m - H_thresh))
 */

#ifndef ADAPTIVE_WEIGHT_MANAGER_H
#define ADAPTIVE_WEIGHT_MANAGER_H

#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <Eigen/Dense>

namespace ORB_SLAM2
{

// 前向声明
class Frame;
class Detection;
class MapText;
class Object;

/**
 * @brief 模态权重结构体
 */
struct ModalityWeights
{
    double pointWeight;     // 特征点权重 (ORB点)
    double objectWeight;    // 物体权重
    double textWeight;      // 文本权重
    
    // 用于优化器的信息矩阵缩放因子
    double pointInfoScale;
    double objectInfoScale;
    double textInfoScale;
    
    // 匹配阈值调节因子
    double objectMatchThreshold;
    double textMatchThreshold;
    
    ModalityWeights() 
        : pointWeight(1.0), objectWeight(1.0), textWeight(1.0),
          pointInfoScale(1.0), objectInfoScale(1.0), textInfoScale(1.0),
          objectMatchThreshold(1.0), textMatchThreshold(1.0) {}
};

/**
 * @brief 环境语义熵统计结构体
 */
struct SemanticEntropyStats
{
    // 数量密度
    int numPoints;
    int numObjects;
    int numTexts;
    
    // 平均置信度
    double avgPointQuality;      // 点特征响应强度
    double avgObjectConfidence;  // 物体检测置信度
    double avgTextConfidence;    // 文本OCR置信度
    
    // 空间分布熵
    double pointSpatialEntropy;
    double objectSpatialEntropy;
    double textSpatialEntropy;
    
    // 综合语义熵
    double pointEntropy;
    double objectEntropy;
    double textEntropy;
    
    SemanticEntropyStats()
        : numPoints(0), numObjects(0), numTexts(0),
          avgPointQuality(0), avgObjectConfidence(0), avgTextConfidence(0),
          pointSpatialEntropy(0), objectSpatialEntropy(0), textSpatialEntropy(0),
          pointEntropy(0), objectEntropy(0), textEntropy(0) {}
};

/**
 * @brief 自适应权重管理器
 */
class AdaptiveWeightManager
{
public:
    AdaptiveWeightManager();
    
    /**
     * @brief 设置图像尺寸（用于空间分布计算）
     */
    void SetImageSize(int width, int height);
    
    /**
     * @brief 根据当前帧计算环境语义熵并更新权重
     * @param pFrame 当前帧
     * @param objectDetections 物体检测列表
     * @param textDetections 文本检测列表
     * @param matchedTexts 已匹配的文本地标
     */
    void UpdateWeights(Frame* pFrame,
                       const std::vector<std::shared_ptr<Detection>>& objectDetections,
                       const std::vector<std::shared_ptr<Detection>>& textDetections,
                       const std::vector<MapText*>& matchedTexts);
    
    /**
     * @brief 获取当前模态权重
     */
    const ModalityWeights& GetCurrentWeights() const { return mCurrentWeights; }
    
    /**
     * @brief 获取当前语义熵统计
     */
    const SemanticEntropyStats& GetCurrentStats() const { return mCurrentStats; }
    
    /**
     * @brief 获取文本优化器权重（用于PoseOptimization）
     */
    double GetTextOptWeight() const { return mCurrentWeights.textInfoScale * mBaseTextWeight; }
    
    /**
     * @brief 获取物体优化器权重
     */
    double GetObjectOptWeight() const { return mCurrentWeights.objectInfoScale * mBaseObjectWeight; }
    
    /**
     * @brief 获取TextMatcher的IoU阈值调节
     */
    double GetTextIoUThreshold() const;
    
    /**
     * @brief 获取TextMatcher的ZNCC阈值调节
     */
    double GetTextZNccThreshold() const;
    
    /**
     * @brief 打印当前权重状态
     */
    void PrintStatus() const;
    
    /**
     * @brief 判断当前环境类型
     * @return 0=弱语义, 1=富文本, 2=富物体, 3=均衡
     */
    int GetEnvironmentType() const;
    
    // ============ 参数配置 ============
    
    // Sigmoid门控参数
    void SetSigmoidParams(double k, double threshold) {
        mSigmoidK = k;
        mEntropyThreshold = threshold;
    }
    
    // 基础权重
    void SetBaseWeights(double point, double object, double text) {
        mBasePointWeight = point;
        mBaseObjectWeight = object;
        mBaseTextWeight = text;
    }
    
    // 最大权重
    void SetMaxWeights(double point, double object, double text) {
        mMaxPointWeight = point;
        mMaxObjectWeight = object;
        mMaxTextWeight = text;
    }
    
    // 空间分布网格数
    void SetGridSize(int gridX, int gridY) {
        mGridX = gridX;
        mGridY = gridY;
    }

private:
    /**
     * @brief 计算空间分布熵（网格化香农熵）
     */
    double ComputeSpatialEntropy(const std::vector<Eigen::Vector2d>& positions);
    
    /**
     * @brief Sigmoid门控函数
     */
    double SigmoidGate(double entropy);
    
    /**
     * @brief 归一化熵值到[0,1]
     */
    double NormalizeEntropy(double rawEntropy, double maxPossible);
    
    /**
     * @brief 计算综合语义熵
     */
    double ComputeModalityEntropy(int count, double avgConfidence, double spatialEntropy);

private:
    // 当前权重和统计
    ModalityWeights mCurrentWeights;
    SemanticEntropyStats mCurrentStats;
    
    // 图像尺寸
    int mImageWidth;
    int mImageHeight;
    
    // 空间分布网格
    int mGridX;
    int mGridY;
    
    // Sigmoid参数
    double mSigmoidK;           // 门控陡峭度
    double mEntropyThreshold;   // 熵阈值
    
    // 基础权重（最低权重）
    double mBasePointWeight;
    double mBaseObjectWeight;
    double mBaseTextWeight;
    
    // 最大权重
    double mMaxPointWeight;
    double mMaxObjectWeight;
    double mMaxTextWeight;
    
    // 历史平滑（EMA）
    double mAlphaEMA;
    bool mFirstFrame;
    
    // 特征数量归一化参数
    double mNormPointCount;
    double mNormObjectCount;
    double mNormTextCount;
};

} // namespace ORB_SLAM2

#endif // ADAPTIVE_WEIGHT_MANAGER_H
