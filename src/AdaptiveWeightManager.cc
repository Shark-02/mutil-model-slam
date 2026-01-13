/**
 * AdaptiveWeightManager.cc
 * 
 * 自适应权重调度模块实现
 * 
 * 核心公式：
 *   综合语义熵: H_m = α * log(1 + N_m/N_norm) + β * C_m + γ * S_m
 *   权重映射:   w_m = w_base + (w_max - w_base) * σ(k * (H_m - H_thresh))
 *   其中 σ(x) = 1 / (1 + exp(-x))
 */

#include "AdaptiveWeightManager.h"
#include "Frame.h"
#include "ImageDetections.h"
#include "MapText.h"
#include "Object.h"

#include <numeric>
#include <iomanip>

namespace ORB_SLAM2
{

AdaptiveWeightManager::AdaptiveWeightManager()
    : mImageWidth(640), mImageHeight(480),
      mGridX(4), mGridY(4),
      mSigmoidK(5.0),           // 门控陡峭度
      mEntropyThreshold(0.5),   // 熵阈值（归一化后）
      mBasePointWeight(1.0),
      mBaseObjectWeight(0.01),  // 物体基础权重较低
      mBaseTextWeight(0.01),    // 文本基础权重较低
      mMaxPointWeight(1.0),
      mMaxObjectWeight(0.5),    // 高置信时可达0.5
      mMaxTextWeight(0.5),
      mAlphaEMA(0.3),           // EMA平滑系数
      mFirstFrame(true),
      mNormPointCount(200.0),   // 期望点特征数量
      mNormObjectCount(5.0),    // 期望物体数量
      mNormTextCount(3.0)       // 期望文本数量
{
}

void AdaptiveWeightManager::SetImageSize(int width, int height)
{
    mImageWidth = width;
    mImageHeight = height;
}

void AdaptiveWeightManager::UpdateWeights(
    Frame* pFrame,
    const std::vector<std::shared_ptr<Detection>>& objectDetections,
    const std::vector<std::shared_ptr<Detection>>& textDetections,
    const std::vector<MapText*>& matchedTexts)
{
    SemanticEntropyStats newStats;
    
    // =========================================================================
    // 1. 统计特征点信息
    // =========================================================================
    if (pFrame)
    {
        newStats.numPoints = 0;
        double totalResponse = 0.0;
        std::vector<Eigen::Vector2d> pointPositions;
        
        for (int i = 0; i < pFrame->N; ++i)
        {
            if (pFrame->mvpMapPoints[i] && !pFrame->mvbOutlier[i])
            {
                newStats.numPoints++;
                totalResponse += pFrame->mvKeysUn[i].response;
                pointPositions.push_back(Eigen::Vector2d(
                    pFrame->mvKeysUn[i].pt.x,
                    pFrame->mvKeysUn[i].pt.y));
            }
        }
        
        if (newStats.numPoints > 0)
        {
            newStats.avgPointQuality = totalResponse / newStats.numPoints;
            // 归一化到[0,1]，假设最大响应为100
            newStats.avgPointQuality = std::min(1.0, newStats.avgPointQuality / 100.0);
            newStats.pointSpatialEntropy = ComputeSpatialEntropy(pointPositions);
        }
    }
    
    // =========================================================================
    // 2. 统计物体检测信息
    // =========================================================================
    newStats.numObjects = static_cast<int>(objectDetections.size());
    if (newStats.numObjects > 0)
    {
        double totalConf = 0.0;
        std::vector<Eigen::Vector2d> objectCenters;
        
        for (const auto& det : objectDetections)
        {
            totalConf += det->score;
            double cx = (det->bbox[0] + det->bbox[2]) / 2.0;
            double cy = (det->bbox[1] + det->bbox[3]) / 2.0;
            objectCenters.push_back(Eigen::Vector2d(cx, cy));
        }
        
        newStats.avgObjectConfidence = totalConf / newStats.numObjects;
        newStats.objectSpatialEntropy = ComputeSpatialEntropy(objectCenters);
    }
    
    // =========================================================================
    // 3. 统计文本检测信息
    // =========================================================================
    newStats.numTexts = static_cast<int>(textDetections.size());
    if (newStats.numTexts > 0)
    {
        double totalConf = 0.0;
        std::vector<Eigen::Vector2d> textCenters;
        
        for (const auto& det : textDetections)
        {
            // 使用 text_confidence；如果为负（未赋值）才回退到检测分数
            totalConf += (det->text_confidence >= 0 ? det->text_confidence : det->score);
            double cx = (det->bbox[0] + det->bbox[2]) / 2.0;
            double cy = (det->bbox[1] + det->bbox[3]) / 2.0;
            textCenters.push_back(Eigen::Vector2d(cx, cy));
        }
        
        newStats.avgTextConfidence = totalConf / newStats.numTexts;
        newStats.textSpatialEntropy = ComputeSpatialEntropy(textCenters);
    }
    
    // 额外考虑已匹配的GOOD文本地标的质量
    int numGoodTexts = 0;
    for (auto* pMT : matchedTexts)
    {
        if (pMT && !pMT->isBad() && pMT->GetState() == TEXTGOOD)
            numGoodTexts++;
    }
    // 如果有GOOD文本，提升文本置信度
    if (numGoodTexts > 0)
    {
        newStats.avgTextConfidence = std::min(1.0, 
            newStats.avgTextConfidence + 0.1 * numGoodTexts);
    }
    
    // =========================================================================
    // 4. 计算综合语义熵
    // =========================================================================
    // 权重系数在 ComputeModalityEntropy 中使用：数量 0.3, 置信度 0.4, 空间分布 0.3
    
    newStats.pointEntropy = ComputeModalityEntropy(
        newStats.numPoints, newStats.avgPointQuality, newStats.pointSpatialEntropy);
    
    newStats.objectEntropy = ComputeModalityEntropy(
        newStats.numObjects, newStats.avgObjectConfidence, newStats.objectSpatialEntropy);
    
    newStats.textEntropy = ComputeModalityEntropy(
        newStats.numTexts, newStats.avgTextConfidence, newStats.textSpatialEntropy);
    
    // =========================================================================
    // 5. EMA平滑（避免权重剧烈波动）
    // =========================================================================
    if (mFirstFrame)
    {
        mCurrentStats = newStats;
        mFirstFrame = false;
    }
    else
    {
        // 指数移动平均
        mCurrentStats.pointEntropy = mAlphaEMA * newStats.pointEntropy + 
                                     (1 - mAlphaEMA) * mCurrentStats.pointEntropy;
        mCurrentStats.objectEntropy = mAlphaEMA * newStats.objectEntropy + 
                                      (1 - mAlphaEMA) * mCurrentStats.objectEntropy;
        mCurrentStats.textEntropy = mAlphaEMA * newStats.textEntropy + 
                                    (1 - mAlphaEMA) * mCurrentStats.textEntropy;
        
        // 更新其他统计量
        mCurrentStats.numPoints = newStats.numPoints;
        mCurrentStats.numObjects = newStats.numObjects;
        mCurrentStats.numTexts = newStats.numTexts;
        mCurrentStats.avgPointQuality = newStats.avgPointQuality;
        mCurrentStats.avgObjectConfidence = newStats.avgObjectConfidence;
        mCurrentStats.avgTextConfidence = newStats.avgTextConfidence;
        mCurrentStats.pointSpatialEntropy = newStats.pointSpatialEntropy;
        mCurrentStats.objectSpatialEntropy = newStats.objectSpatialEntropy;
        mCurrentStats.textSpatialEntropy = newStats.textSpatialEntropy;
    }
    
    // =========================================================================
    // 6. 通过Sigmoid门控计算权重
    // =========================================================================
    double pointGate = SigmoidGate(mCurrentStats.pointEntropy);
    double objectGate = SigmoidGate(mCurrentStats.objectEntropy);
    double textGate = SigmoidGate(mCurrentStats.textEntropy);
    
    // 计算信息矩阵缩放因子
    mCurrentWeights.pointInfoScale = mBasePointWeight + 
        (mMaxPointWeight - mBasePointWeight) * pointGate;
    mCurrentWeights.objectInfoScale = mBaseObjectWeight + 
        (mMaxObjectWeight - mBaseObjectWeight) * objectGate;
    mCurrentWeights.textInfoScale = mBaseTextWeight + 
        (mMaxTextWeight - mBaseTextWeight) * textGate;
    
    // 归一化使总权重和为常数（可选）
    double totalWeight = mCurrentWeights.pointInfoScale + 
                         mCurrentWeights.objectInfoScale + 
                         mCurrentWeights.textInfoScale;
    
    // 保持点特征为基准，调整语义模态的相对权重
    mCurrentWeights.pointWeight = 1.0;
    mCurrentWeights.objectWeight = mCurrentWeights.objectInfoScale / mCurrentWeights.pointInfoScale;
    mCurrentWeights.textWeight = mCurrentWeights.textInfoScale / mCurrentWeights.pointInfoScale;
    
    // =========================================================================
    // 7. 调整匹配阈值（熵高时可以放宽阈值，熵低时收紧）
    // =========================================================================
    // 物体匹配：熵高时Wasserstein阈值可以略微放宽
    mCurrentWeights.objectMatchThreshold = 1.0 + 0.2 * (objectGate - 0.5);
    
    // 文本匹配：熵高时IoU阈值可以略微放宽
    mCurrentWeights.textMatchThreshold = 1.0 + 0.2 * (textGate - 0.5);
}

double AdaptiveWeightManager::ComputeSpatialEntropy(const std::vector<Eigen::Vector2d>& positions)
{
    if (positions.empty())
        return 0.0;
    
    // 将图像划分为 mGridX x mGridY 的网格
    int totalCells = mGridX * mGridY;
    std::vector<int> cellCounts(totalCells, 0);
    
    double cellWidth = static_cast<double>(mImageWidth) / mGridX;
    double cellHeight = static_cast<double>(mImageHeight) / mGridY;
    
    for (const auto& pos : positions)
    {
        int gx = static_cast<int>(pos[0] / cellWidth);
        int gy = static_cast<int>(pos[1] / cellHeight);
        
        gx = std::max(0, std::min(gx, mGridX - 1));
        gy = std::max(0, std::min(gy, mGridY - 1));
        
        cellCounts[gy * mGridX + gx]++;
    }
    
    // 计算香农熵
    double entropy = 0.0;
    double total = static_cast<double>(positions.size());
    
    for (int c : cellCounts)
    {
        if (c > 0)
        {
            double p = c / total;
            entropy -= p * std::log2(p);
        }
    }
    
    // 归一化到[0,1]，最大熵为 log2(totalCells)
    double maxEntropy = std::log2(static_cast<double>(totalCells));
    return (maxEntropy > 0) ? (entropy / maxEntropy) : 0.0;
}

double AdaptiveWeightManager::SigmoidGate(double entropy)
{
    // σ(k * (H - H_thresh)) = 1 / (1 + exp(-k * (H - H_thresh)))
    double x = mSigmoidK * (entropy - mEntropyThreshold);
    return 1.0 / (1.0 + std::exp(-x));
}

double AdaptiveWeightManager::NormalizeEntropy(double rawEntropy, double maxPossible)
{
    if (maxPossible <= 0)
        return 0.0;
    return std::min(1.0, rawEntropy / maxPossible);
}

double AdaptiveWeightManager::ComputeModalityEntropy(int count, double avgConfidence, double spatialEntropy)
{
    // 综合语义熵公式：
    // H = α * log(1 + N/N_norm) / log(2) + β * C + γ * S
    // 其中各项都归一化到[0,1]
    
    const double alpha = 0.3;  // 数量权重
    const double beta = 0.4;   // 置信度权重
    const double gamma = 0.3;  // 空间分布权重
    
    // 数量密度（使用对数归一化）
    double normCount = mNormPointCount; // 默认使用点特征归一化
    double countTerm = std::log(1.0 + count / normCount) / std::log(2.0);
    countTerm = std::min(1.0, countTerm); // 截断到[0,1]
    
    // 置信度项（已经在[0,1]范围）
    double confTerm = std::min(1.0, std::max(0.0, avgConfidence));
    
    // 空间分布项（已经归一化）
    double spatialTerm = std::min(1.0, std::max(0.0, spatialEntropy));
    
    return alpha * countTerm + beta * confTerm + gamma * spatialTerm;
}

double AdaptiveWeightManager::GetTextIoUThreshold() const
{
    // 基础IoU阈值0.4，根据文本熵动态调整
    // 熵高（文本可靠）时可以略微降低阈值（更宽松）
    // 熵低时提高阈值（更严格）
    double baseThreshold = 0.4;
    double adjustment = 0.1 * (mCurrentWeights.textMatchThreshold - 1.0);
    return std::max(0.2, std::min(0.6, baseThreshold - adjustment));
}

double AdaptiveWeightManager::GetTextZNccThreshold() const
{
    // 基础ZNCC阈值0.5，类似调整逻辑
    double baseThreshold = 0.5;
    double adjustment = 0.1 * (mCurrentWeights.textMatchThreshold - 1.0);
    return std::max(0.3, std::min(0.7, baseThreshold - adjustment));
}

int AdaptiveWeightManager::GetEnvironmentType() const
{
    // 判断当前环境类型
    // 0=弱语义, 1=富文本, 2=富物体, 3=均衡
    
    const double threshold = 0.4; // 语义熵阈值
    
    bool richText = mCurrentStats.textEntropy > threshold;
    bool richObject = mCurrentStats.objectEntropy > threshold;
    
    if (!richText && !richObject)
        return 0; // 弱语义环境
    else if (richText && !richObject)
        return 1; // 富文本环境
    else if (!richText && richObject)
        return 2; // 富物体环境
    else
        return 3; // 均衡环境
}

void AdaptiveWeightManager::PrintStatus() const
{
    std::cout << "[AdaptiveWeight] ======================================" << std::endl;
    std::cout << "[AdaptiveWeight] Environment Type: ";
    switch(GetEnvironmentType())
    {
        case 0: std::cout << "Weak Semantic (Point-dominant)"; break;
        case 1: std::cout << "Rich Text"; break;
        case 2: std::cout << "Rich Object"; break;
        case 3: std::cout << "Balanced"; break;
    }
    std::cout << std::endl;
    
    std::cout << "[AdaptiveWeight] Feature Counts: "
              << "Points=" << mCurrentStats.numPoints
              << " Objects=" << mCurrentStats.numObjects
              << " Texts=" << mCurrentStats.numTexts << std::endl;
    
    std::cout << "[AdaptiveWeight] Confidence: "
              << "Point=" << std::fixed << std::setprecision(3) << mCurrentStats.avgPointQuality
              << " Object=" << mCurrentStats.avgObjectConfidence
              << " Text=" << mCurrentStats.avgTextConfidence << std::endl;
    
    std::cout << "[AdaptiveWeight] Spatial Entropy: "
              << "Point=" << mCurrentStats.pointSpatialEntropy
              << " Object=" << mCurrentStats.objectSpatialEntropy
              << " Text=" << mCurrentStats.textSpatialEntropy << std::endl;
    
    std::cout << "[AdaptiveWeight] Semantic Entropy: "
              << "Point=" << mCurrentStats.pointEntropy
              << " Object=" << mCurrentStats.objectEntropy
              << " Text=" << mCurrentStats.textEntropy << std::endl;
    
    std::cout << "[AdaptiveWeight] Info Scale: "
              << "Point=" << mCurrentWeights.pointInfoScale
              << " Object=" << mCurrentWeights.objectInfoScale
              << " Text=" << mCurrentWeights.textInfoScale << std::endl;
    
    std::cout << "[AdaptiveWeight] Relative Weights: "
              << "Point=" << mCurrentWeights.pointWeight
              << " Object=" << mCurrentWeights.objectWeight
              << " Text=" << mCurrentWeights.textWeight << std::endl;
    std::cout << "[AdaptiveWeight] ======================================" << std::endl;
}

} // namespace ORB_SLAM2
