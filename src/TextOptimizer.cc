/**
 * TextOptimizer.cc
 * 
 * 文本平面参数优化器实现
 * 参考 TextSLAM 的 ThetaOptimMultiFs
 * 
 * 核心改进：使用 θ = -n/d (3维) 代替 (n,d) (4维) 进行优化
 * 优势：单应性公式 H = R + t*θ^T 完全没有除法，彻底避免数值爆炸
 */

#include "TextOptimizer.h"
#include "MapText.h"
#include "KeyFrame.h"
#include "Converter.h"

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace ORB_SLAM2
{

// ============================================================================
// θ ←→ (n, d) 转换工具函数
// ============================================================================

Eigen::Vector3d TextOptimizer::PlaneToTheta(const Eigen::Vector3d& n, double d)
{
    // θ = -n/d
    // 注意：d 不能为 0（平面不能经过坐标原点）
    if (std::abs(d) < 1e-8) {
        std::cerr << "[TextOptimizer] Warning: d ≈ 0, plane passes through origin!" << std::endl;
        // 返回一个安全的默认值（假设平面在 z=1 处）
        return Eigen::Vector3d(0, 0, -1);
    }
    return -n / d;
}

void TextOptimizer::ThetaToPlane(const Eigen::Vector3d& theta, Eigen::Vector3d& n, double& d)
{
    // θ = -n/d
    // 假设 |n| = 1，则 |θ| = 1/|d|
    // 因此 d = ±1/|θ|，n = -θ*d = ∓θ/|θ|
    double theta_norm = theta.norm();
    if (theta_norm < 1e-8) {
        std::cerr << "[TextOptimizer] Warning: θ ≈ 0, cannot recover plane!" << std::endl;
        n = Eigen::Vector3d(0, 0, 1);
        d = -1.0;
        return;
    }
    
    // 先恢复标准化的法向量，使其成为 PlaneToTheta 的逆运算
    // （具体符号将在调用方根据原始法向进行对齐）
    d = -1.0 / theta_norm;
    n = theta / theta_norm;
}

// ============================================================================
// 主优化函数
// ============================================================================

bool TextOptimizer::OptimizePlaneMultiFrame(MapText* pMT, bool bFixScale)
{
    if (!pMT || pMT->isBad())
        return false;

    // 获取所有观测
    std::map<KeyFrame*, size_t> observations = pMT->GetObservations();
    if (observations.size() < 2) {
        std::cout << "[TextOptimizer] MapText " << pMT->mnId 
                  << " has only " << observations.size() << " observations, skip." << std::endl;
        return false;
    }

    KeyFrame* pRefKF = pMT->GetReferenceKeyFrame();
    if (!pRefKF || pRefKF->isBad())
        return false;

    // =========================================================================
    // 获取当前平面参数并转换为 θ（参考帧坐标系）
    // =========================================================================
    // MapText 存储的 mThetaRef 已经是参考帧坐标系下的 (n_ref, d_ref)
    Eigen::Vector4d planeRef = pMT->GetPlaneRef();
    Eigen::Vector3d n_ref = planeRef.head<3>();
    double d_ref = planeRef[3];
    
    // 归一化法向量
    double n_norm = n_ref.norm();
    if (n_norm > 1e-6) {
        n_ref /= n_norm;
        d_ref /= n_norm;
    }
    
    // 转换为 θ
    Eigen::Vector3d theta_init = PlaneToTheta(n_ref, d_ref);
    
    std::cout << "[TextOptimizer] MapText " << pMT->mnId 
              << " | Initial n_ref=(" << n_ref.transpose() << ") d_ref=" << d_ref
              << " | theta=(" << theta_init.transpose() << ")" << std::endl;

    // =========================================================================
    // 预处理：获取各种数据
    // =========================================================================
    
    // 获取参考帧的位姿
    cv::Mat Twc_ref_cv = pRefKF->GetPoseInverse();
    Eigen::Matrix4d Twc_ref;
    cv::cv2eigen(Twc_ref_cv, Twc_ref);
    Eigen::Matrix3d R_wc = Twc_ref.block<3,3>(0,0);
    Eigen::Vector3d t_wc = Twc_ref.block<3,1>(0,3);

    // 获取参考帧的 corner rays
    auto itRef = observations.find(pRefKF);
    if (itRef == observations.end()) {
        std::cout << "[TextOptimizer] Reference KeyFrame not in observations!" << std::endl;
        return false;
    }
    size_t refDetIdx = itRef->second;
    if (refDetIdx >= pRefKF->vTextDetections.size())
        return false;

    auto& refDet = pRefKF->vTextDetections[refDetIdx];
    std::vector<Eigen::Vector2d> cornerRays = GetCornerRays(pRefKF, 
        Eigen::Vector4d(refDet->bbox[0], refDet->bbox[1], refDet->bbox[2], refDet->bbox[3]));

    if (cornerRays.size() != 4) {
        std::cout << "[TextOptimizer] Failed to get 4 corner rays, got " << cornerRays.size() << std::endl;
        return false;
    }

    // 获取点云内点
    std::vector<cv::Mat> inliers_ref = pMT->GetInlierPoints();
    int nPointEdges = static_cast<int>(inliers_ref.size());
    
    // 计算点云内点到初始平面的平均距离，用于验证内点质量
    double avgDist = 0.0;
    for (const auto& pt_mat : inliers_ref) {
        Eigen::Vector3d p(pt_mat.at<float>(0), pt_mat.at<float>(1), pt_mat.at<float>(2));
        double dist = std::abs(n_ref.dot(p) + d_ref);
        avgDist += dist;
    }
    if (nPointEdges > 0) avgDist /= nPointEdges;
    std::cout << "[TextOptimizer] Inlier quality: " << nPointEdges << " points, avg dist to plane = " 
              << avgDist << " m" << std::endl;
    
    // 计算图像对角线长度，用于归一化视觉残差
    double imgDiag = std::sqrt(pRefKF->mnMaxX * pRefKF->mnMaxX + pRefKF->mnMaxY * pRefKF->mnMaxY);
    if (imgDiag < 100.0) imgDiag = 640.0;  // 默认值
    double visualScale = 1.0 / imgDiag;    // 归一化因子，使像素误差变成 ~0.01-0.1 量级

    // =========================================================================
    // Stage 1: 仅点云约束优化
    // =========================================================================
    g2o::SparseOptimizer optimizer1;
    
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolver_3_1;
    typedef g2o::LinearSolverDense<BlockSolver_3_1::PoseMatrixType> LinearSolver;
    
    {
        auto linearSolver = new LinearSolver();
        auto solver_ptr = new BlockSolver_3_1(linearSolver);
        auto solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer1.setAlgorithm(solver);
        optimizer1.setVerbose(false);
    }

    // 顶点
    VertexTheta3D* vTheta1 = new VertexTheta3D();
    vTheta1->setEstimate(theta_init);
    vTheta1->setId(0);
    optimizer1.addVertex(vTheta1);

    int edgeId = 0;
    
    // 只添加点云约束
    for (const auto& pt_mat : inliers_ref)
    {
        Eigen::Vector3d p_ref(pt_mat.at<float>(0), pt_mat.at<float>(1), pt_mat.at<float>(2));
        
        EdgePointToPlaneTheta* e = new EdgePointToPlaneTheta();
        e->setVertex(0, vTheta1);
        e->setPoint3D(p_ref);
        e->setMeasurement(0.0);
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0);  // 权重 1，因为这是唯一约束
        
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(0.1);
        e->setRobustKernel(rk);
        
        e->setId(edgeId++);
        optimizer1.addEdge(e);
    }

    if (nPointEdges < 5) {
        std::cout << "[TextOptimizer] Too few point constraints (" << nPointEdges << "), skip optimization." << std::endl;
        return false;
    }

    // Stage 1 优化
    optimizer1.initializeOptimization();
    optimizer1.computeActiveErrors();
    double chi2_stage1_before = optimizer1.activeChi2();
    
    optimizer1.optimize(10);  // 10次迭代
    
    double chi2_stage1_after = optimizer1.activeChi2();
    Eigen::Vector3d theta_stage1 = vTheta1->estimate();

    // 计算 Stage1 后的 d 值，用于验证
    Eigen::Vector3d n_stage1;
    double d_stage1;
    ThetaToPlane(theta_stage1, n_stage1, d_stage1);
    if (n_ref.dot(n_stage1) < 0) d_stage1 = -d_stage1;  // 保持符号一致
    
    std::cout << "[TextOptimizer] Stage1 (point cloud only): Chi2 " 
              << chi2_stage1_before << " -> " << chi2_stage1_after
              << " | theta=(" << theta_stage1.transpose() << ")"
              << " | d_stage1=" << d_stage1 << " (init=" << d_ref << ")" << std::endl;

    // =========================================================================
    // Stage 2: 点云 + 视觉约束（视觉权重极低且归一化）
    // =========================================================================
    g2o::SparseOptimizer optimizer2;
    
    {
        auto linearSolver = new LinearSolver();
        auto solver_ptr = new BlockSolver_3_1(linearSolver);
        auto solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer2.setAlgorithm(solver);
        optimizer2.setVerbose(false);
    }

    // 用 Stage 1 结果作为初值
    VertexTheta3D* vTheta2 = new VertexTheta3D();
    vTheta2->setEstimate(theta_stage1);
    vTheta2->setId(0);
    optimizer2.addVertex(vTheta2);

    edgeId = 0;
    int nCornerEdges = 0;

    // 点云约束（权重 1.0）
    for (const auto& pt_mat : inliers_ref)
    {
        Eigen::Vector3d p_ref(pt_mat.at<float>(0), pt_mat.at<float>(1), pt_mat.at<float>(2));
        
        EdgePointToPlaneTheta* e = new EdgePointToPlaneTheta();
        e->setVertex(0, vTheta2);
        e->setPoint3D(p_ref);
        e->setMeasurement(0.0);
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1.0);
        
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(0.1);
        e->setRobustKernel(rk);
        
        e->setId(edgeId++);
        optimizer2.addEdge(e);
    }

    // 视觉约束（归一化 + 极低权重）
    // 权重设计：点云残差~0.1，归一化后视觉残差~0.05，权重 0.01
    // 这样视觉贡献 ≈ 0.05² * 0.01 = 0.000025，远小于点云贡献 0.1² * 1 = 0.01
    double visualWeight = 0.01;

    for (auto& [pKF, detIdx] : observations)
    {
        if (!pKF || pKF->isBad())
            continue;
        
        if (detIdx >= pKF->vTextDetections.size())
            continue;

        auto& det = pKF->vTextDetections[detIdx];
        Eigen::Vector4d bbox(det->bbox[0], det->bbox[1], det->bbox[2], det->bbox[3]);

        // 获取观测帧的位姿
        cv::Mat Tcw_cv = pKF->GetPose();
        Eigen::Matrix4d Tcw;
        cv::cv2eigen(Tcw_cv, Tcw);
        Eigen::Matrix3d R_cw = Tcw.block<3,3>(0,0);
        Eigen::Vector3d t_cw = Tcw.block<3,1>(0,3);

        // 计算相对位姿
        Eigen::Matrix3d R_rel = R_cw * R_wc;
        Eigen::Vector3d t_rel = R_cw * t_wc + t_cw;

        // 获取内参（归一化后的 K，让投影误差变成归一化坐标误差）
        Eigen::Matrix3d K;
        K << pKF->fx * visualScale, 0, pKF->cx * visualScale,
             0, pKF->fy * visualScale, pKF->cy * visualScale,
             0, 0, 1;

        // 当前帧检测框的四个角点（归一化后）
        std::vector<Eigen::Vector2d> corners2D;
        if (det->quad.size() == 4) {
            for (const auto& q : det->quad)
                corners2D.push_back(q * visualScale);
        } else {
            corners2D.push_back(Eigen::Vector2d(bbox[0], bbox[1]) * visualScale);
            corners2D.push_back(Eigen::Vector2d(bbox[2], bbox[1]) * visualScale);
            corners2D.push_back(Eigen::Vector2d(bbox[2], bbox[3]) * visualScale);
            corners2D.push_back(Eigen::Vector2d(bbox[0], bbox[3]) * visualScale);
        }

        for (int k = 0; k < 4; ++k)
        {
            EdgeHomographyProjection* e = new EdgeHomographyProjection();
            e->setVertex(0, vTheta2);
            e->setParams(cornerRays[k], R_rel, t_rel, K);
            e->setMeasurement(corners2D[k]);
            e->setInformation(Eigen::Matrix2d::Identity() * visualWeight);
            
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(0.05);  // 归一化后约 0.05 对应原来的 ~30 像素
            e->setRobustKernel(rk);
            
            e->setId(edgeId++);
            optimizer2.addEdge(e);
            nCornerEdges++;
        }
    }

    std::cout << "[TextOptimizer] MapText " << pMT->mnId 
              << ": " << nPointEdges << " point edges, " 
              << nCornerEdges << " corner edges from "
              << observations.size() << " observations"
              << " (visualScale=" << visualScale << ", visualWeight=" << visualWeight << ")" << std::endl;

    // Stage 2 优化（只做少量迭代，避免视觉约束累积影响）
    optimizer2.initializeOptimization();
    
    optimizer2.computeActiveErrors();
    double chi2_before = optimizer2.activeChi2();

    optimizer2.optimize(5);  // 只做 5 次迭代

    double chi2_after = optimizer2.activeChi2();

    // =========================================================================
    // 从优化后的 θ 恢复 (n, d)
    // =========================================================================
    Eigen::Vector3d theta_opt = vTheta2->estimate();
    Eigen::Vector3d n_ref_new;
    double d_ref_new;
    ThetaToPlane(theta_opt, n_ref_new, d_ref_new);

    // 与原始法向保持一致的符号，避免出现等价平面但法向翻转的情况
    if (n_ref.norm() > 1e-6 && n_ref.dot(n_ref_new) < 0.0) {
        n_ref_new = -n_ref_new;
        d_ref_new = -d_ref_new;
    }

    // =========================================================================
    // 检查优化结果的有效性 (三层防御机制)
    // =========================================================================
    bool isPhysicallyValid = true;
    bool largeChangeButImproved = false;  // 标记：大跳变但误差显著下降
    
    // 获取优化尝试次数，用于渐进式放宽阈值
    int nOptAttempts = pMT->GetOptimizationAttempts();
    pMT->IncrementOptimizationAttempts();  // 记录本次尝试
    
    // ===== 第一层：渐进式放宽阈值 =====
    // 首次严格，之后逐步放宽，给初始化不好的地标多次机会
    double ratio_threshold_low, ratio_threshold_high;
    if (nOptAttempts == 0) {
        ratio_threshold_low = 0.3; ratio_threshold_high = 3.0;   // 首次严格
    } else if (nOptAttempts <= 2) {
        ratio_threshold_low = 0.2; ratio_threshold_high = 5.0;   // 第2-3次放宽
    } else {
        ratio_threshold_low = 0.1; ratio_threshold_high = 10.0;  // 之后更宽松
    }
    
    // 1. 平面不能离相机太近（防止穿过相机）
    if (std::abs(d_ref_new) < 0.1) {
        std::cout << "[TextOptimizer] Warning: plane too close to camera, d=" << d_ref_new << std::endl;
        isPhysicallyValid = false;
    }
    
    // 2. 平面不能太远（超过 50 米可能是错误的）
    if (std::abs(d_ref_new) > 50.0) {
        std::cout << "[TextOptimizer] Warning: plane too far, d=" << d_ref_new << std::endl;
        isPhysicallyValid = false;
    }
    
    // 3. 物理约束：平面应该在相机前方
    //    法向量 n_z 和 d 的符号关系决定平面位置
    //    如果 n_z > 0 且 d > 0，或 n_z < 0 且 d < 0，平面在相机后方
    bool planeInFront = (n_ref_new.z() * d_ref_new <= 0) || std::abs(n_ref_new.z()) < 0.1;
    if (!planeInFront) {
        std::cout << "[TextOptimizer] Warning: plane may be behind camera, n_z=" 
                  << n_ref_new.z() << " d=" << d_ref_new << std::endl;
        // 不直接拒绝，因为某些倾斜平面可能触发这个条件
    }
    
    // 4. 法向变化不能太大（防止平面翻转）
    double cos_angle = 1.0;
    if (n_ref.norm() > 1e-6) {
        cos_angle = n_ref.dot(n_ref_new);
    }
    if (cos_angle < 0.5) {  // 超过 60 度变化
        std::cout << "[TextOptimizer] Warning: normal changed too much, cos=" << cos_angle << std::endl;
        isPhysicallyValid = false;
    }
    
    // 5. 距离变化检查（使用渐进式阈值）
    double d_change_ratio = std::abs(d_ref_new / d_ref);
    bool largeChange = (d_change_ratio < ratio_threshold_low || d_change_ratio > ratio_threshold_high);
    
    if (largeChange) {
        std::cout << "[TextOptimizer] Notice: d changed significantly, ratio=" << d_change_ratio 
                  << " (d: " << d_ref << " -> " << d_ref_new << ")"
                  << " threshold=[" << ratio_threshold_low << ", " << ratio_threshold_high << "]"
                  << " (attempt #" << nOptAttempts << ")" << std::endl;
        
        // ===== 第二层：Chi2 显著下降时的特殊通道 =====
        // 关键：使用 Stage1（纯点云）的 Chi2 来判断，因为点云约束更可靠
        double chi2_drop_ratio = chi2_stage1_after / std::max(chi2_stage1_before, 1e-6);
        bool stage1_improved_significantly = (chi2_drop_ratio < 0.4);  // Chi2 降到 40% 以下
        
        if (stage1_improved_significantly) {
            std::cout << "[TextOptimizer] >>> Accepting large change: Stage1 Chi2 dropped to " 
                      << (chi2_drop_ratio * 100) << "% (" << chi2_stage1_before << " -> " 
                      << chi2_stage1_after << ")" << std::endl;
            largeChangeButImproved = true;
            // 不设置 isPhysicallyValid = false
        } else {
            std::cout << "[TextOptimizer] Warning: large change rejected, Stage1 Chi2 ratio=" 
                      << chi2_drop_ratio << " (need < 0.4)" << std::endl;
            isPhysicallyValid = false;
        }
    }

    bool accept = (chi2_after <= chi2_before) && (isPhysicallyValid || largeChangeButImproved);
    
    // 额外检查：Stage2 相对 Stage1 的 d 变化（检测视觉约束是否过度影响）
    double d_stage2_drift = std::abs(d_ref_new - d_stage1) / std::max(std::abs(d_stage1), 0.1);
    if (d_stage2_drift > 0.2) {  // Stage2 导致 d 变化超过 20%
        std::cout << "[TextOptimizer] Warning: Stage2 caused " << (d_stage2_drift*100) 
                  << "% drift in d (stage1=" << d_stage1 << " -> final=" << d_ref_new << ")" << std::endl;
    }

    std::cout << "[TextOptimizer] MapText " << pMT->mnId 
              << " | Stage2 Chi2: " << chi2_before << " -> " << chi2_after
              << " | n_new=(" << n_ref_new.transpose() << ") d_new=" << d_ref_new
              << " | Valid: " << isPhysicallyValid
              << " | LargeButImproved: " << largeChangeButImproved
              << " | Accept: " << accept << std::endl;

    // 更新连续拒绝计数器
    if (!accept) {
        pMT->IncrementRejections();
        std::cout << "[TextOptimizer] MapText " << pMT->mnId 
                  << " consecutive rejections: " << pMT->GetConsecutiveRejections() << std::endl;
    } else {
        pMT->ResetRejections();  // 接受后重置计数器
    }

    if (accept)
    {
        if (pMT->isBad()) {
            std::cout << "[TextOptimizer] MapText became bad during optimization, abort." << std::endl;
            return false;
        }
        if (pMT->GetReferenceKeyFrame() != pRefKF) {
            std::cout << "[TextOptimizer] Reference KeyFrame changed during optimization, abort." << std::endl;
            return false;
        }

        // 更新 MapText 的平面参数（参考帧坐标系）
        // 先保存旧的世界角点，以防更新失败时恢复
        std::vector<cv::Mat> oldWorldCorners = pMT->GetWorldCorners();
        Eigen::Vector4d oldPlaneRef = pMT->GetPlaneRef();
        
        pMT->SetPlaneRef(Eigen::Vector4d(n_ref_new[0], n_ref_new[1], n_ref_new[2], d_ref_new));
        
        // 强制刷新3D角点
        pMT->UpdateCornersFromPlane();
        
        // 验证更新是否成功
        std::vector<cv::Mat> newWorldCorners = pMT->GetWorldCorners();
        if (newWorldCorners.size() != 4) {
            // 更新失败，恢复旧参数
            std::cout << "[TextOptimizer] Warning: UpdateCornersFromPlane failed, reverting to old parameters." << std::endl;
            pMT->SetPlaneRef(oldPlaneRef);
            pMT->SetWorldCorners(oldWorldCorners);
            return false;
        }
        
        return true;
    }
    
    return false;
}

// ============================================================================
// 辅助函数
// ============================================================================

std::vector<Eigen::Vector2d> TextOptimizer::GetCornerRays(
    KeyFrame* pKF,
    const Eigen::Vector4d& bbox)
{
    std::vector<Eigen::Vector2d> rays;
    
    // 四个角点：TL, TR, BR, BL
    std::vector<cv::Point2f> corners;
    corners.reserve(4);
    corners.push_back(cv::Point2f(bbox[0], bbox[1]));  // TL
    corners.push_back(cv::Point2f(bbox[2], bbox[1]));  // TR
    corners.push_back(cv::Point2f(bbox[2], bbox[3]));  // BR
    corners.push_back(cv::Point2f(bbox[0], bbox[3]));  // BL

    std::vector<cv::Point2f> corners_un;

    if(!pKF->mDistCoef.empty())
    {
        cv::Mat mat(corners);
        // undistortPoints with empty P returns normalized coordinates
        cv::undistortPoints(mat, corners_un, pKF->mK, pKF->mDistCoef);
    }
    else
    {
        double fx = pKF->fx;
        double fy = pKF->fy;
        double cx = pKF->cx;
        double cy = pKF->cy;

        corners_un.reserve(4);
        for(const auto& c : corners)
        {
            corners_un.push_back(cv::Point2f((c.x - cx)/fx, (c.y - cy)/fy));
        }
    }
    
    rays.reserve(4);
    for (const auto& c : corners_un) {
        rays.push_back(Eigen::Vector2d(c.x, c.y));
    }
    
    return rays;
}

} // namespace ORB_SLAM2
