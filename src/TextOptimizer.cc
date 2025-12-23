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
    // 设置 g2o 优化器
    // =========================================================================
    g2o::SparseOptimizer optimizer;
    
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolver_3_1;
    typedef g2o::LinearSolverDense<BlockSolver_3_1::PoseMatrixType> LinearSolver;
    
    auto linearSolver = new LinearSolver();
    auto solver_ptr = new BlockSolver_3_1(linearSolver);
    auto solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // =========================================================================
    // 添加顶点：θ = -n/d (3维，参考帧坐标系)
    // =========================================================================
    VertexTheta3D* vTheta = new VertexTheta3D();
    vTheta->setEstimate(theta_init);
    vTheta->setId(0);
    optimizer.addVertex(vTheta);

    int edgeId = 0;
    int nPointEdges = 0;
    int nCornerEdges = 0;

    // 获取参考帧的位姿
    cv::Mat Twc_ref_cv = pRefKF->GetPoseInverse();
    Eigen::Matrix4d Twc_ref;
    cv::cv2eigen(Twc_ref_cv, Twc_ref);
    Eigen::Matrix3d R_wc = Twc_ref.block<3,3>(0,0);
    Eigen::Vector3d t_wc = Twc_ref.block<3,1>(0,3);

    // =========================================================================
    // 添加点云约束（RANSAC 内点，参考帧坐标系）
    // =========================================================================
    std::vector<cv::Mat> inliers_ref = pMT->GetInlierPoints();
    
    for (const auto& pt_mat : inliers_ref)
    {
        // 内点已经在参考帧坐标系下
        Eigen::Vector3d p_ref(pt_mat.at<float>(0), pt_mat.at<float>(1), pt_mat.at<float>(2));
        
        EdgePointToPlaneTheta* e = new EdgePointToPlaneTheta();
        e->setVertex(0, vTheta);
        e->setPoint3D(p_ref);
        e->setMeasurement(0.0);  // 残差 = θ^T * X - 1，目标为0
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 100.0);
        
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(0.1);  // 对应约 0.1 的残差（因为我们的残差单位是无量纲的）
        e->setRobustKernel(rk);
        
        e->setId(edgeId++);
        optimizer.addEdge(e);
        nPointEdges++;
    }

    // =========================================================================
    // 添加角点重投影约束（所有观测帧）
    // =========================================================================
    
    // [Optimization] Pre-calculate reference frame rays
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

    // 检查 cornerRays 是否有效（去畸变可能失败导致少于4个角点）
    if (cornerRays.size() != 4) {
        std::cout << "[TextOptimizer] Failed to get 4 corner rays, got " << cornerRays.size() << std::endl;
        return false;
    }

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

        // 计算相对位姿：从参考帧到观测帧
        // T_obs_ref = T_obs_world * T_world_ref
        Eigen::Matrix3d R_rel = R_cw * R_wc;           // R_{ref→obs} = R_cw * R_wc
        Eigen::Vector3d t_rel = R_cw * t_wc + t_cw;    // t_{ref→obs}

        // 获取内参
        Eigen::Matrix3d K;
        K << pKF->fx, 0, pKF->cx,
             0, pKF->fy, pKF->cy,
             0, 0, 1;

        // 当前帧检测框的四个角点（观测值）
        std::vector<Eigen::Vector2d> corners2D;
        if (det->quad.size() == 4) {
            for (const auto& q : det->quad)
                corners2D.push_back(q);
        } else {
            corners2D.push_back(Eigen::Vector2d(bbox[0], bbox[1]));  // TL
            corners2D.push_back(Eigen::Vector2d(bbox[2], bbox[1]));  // TR
            corners2D.push_back(Eigen::Vector2d(bbox[2], bbox[3]));  // BR
            corners2D.push_back(Eigen::Vector2d(bbox[0], bbox[3]));  // BL
        }

        for (int k = 0; k < 4; ++k)
        {
            EdgeHomographyProjection* e = new EdgeHomographyProjection();
            e->setVertex(0, vTheta);
            e->setParams(cornerRays[k], R_rel, t_rel, K);
            e->setMeasurement(corners2D[k]);
            e->setInformation(Eigen::Matrix2d::Identity() * 1.0);
            
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(10.0);  // 10像素阈值
            e->setRobustKernel(rk);
            
            e->setId(edgeId++);
            optimizer.addEdge(e);
            nCornerEdges++;
        }
    }

    std::cout << "[TextOptimizer] MapText " << pMT->mnId 
              << ": " << nPointEdges << " point edges, " 
              << nCornerEdges << " corner edges from "
              << observations.size() << " observations" << std::endl;

    if (nPointEdges < 5 && nCornerEdges < 4) {
        std::cout << "[TextOptimizer] Too few constraints, skip optimization." << std::endl;
        return false;
    }

    // =========================================================================
    // 执行优化
    // =========================================================================
    optimizer.initializeOptimization();
    
    optimizer.computeActiveErrors();
    double chi2_before = optimizer.activeChi2();

    optimizer.optimize(10);

    double chi2_after = optimizer.activeChi2();

    // =========================================================================
    // 从优化后的 θ 恢复 (n, d)
    // =========================================================================
    Eigen::Vector3d theta_opt = vTheta->estimate();
    Eigen::Vector3d n_ref_new;
    double d_ref_new;
    ThetaToPlane(theta_opt, n_ref_new, d_ref_new);

    // 与原始法向保持一致的符号，避免出现等价平面但法向翻转的情况
    if (n_ref.norm() > 1e-6 && n_ref.dot(n_ref_new) < 0.0) {
        n_ref_new = -n_ref_new;
        d_ref_new = -d_ref_new;
    }

    // =========================================================================
    // 检查优化结果的有效性
    // =========================================================================
    bool isPhysicallyValid = true;
    
    // 1. 平面不能离相机太近（防止穿过相机）
    if (std::abs(d_ref_new) < 0.05) {
        std::cout << "[TextOptimizer] Warning: plane too close to camera, d=" << d_ref_new << std::endl;
        isPhysicallyValid = false;
    }
    
    // 2. 平面不能太远（超过 50 米可能是错误的）
    if (std::abs(d_ref_new) > 50.0) {
        std::cout << "[TextOptimizer] Warning: plane too far, d=" << d_ref_new << std::endl;
        isPhysicallyValid = false;
    }
    
    // 3. 法向变化不能太大（防止平面翻转）
    double cos_angle = 1.0;
    if (n_ref.norm() > 1e-6) {
        cos_angle = n_ref.dot(n_ref_new);
    }
    if (cos_angle < 0.5) {  // 超过 60 度变化
        std::cout << "[TextOptimizer] Warning: normal changed too much, cos=" << cos_angle << std::endl;
        isPhysicallyValid = false;
    }

    bool accept = (chi2_after <= chi2_before) && isPhysicallyValid;

    std::cout << "[TextOptimizer] MapText " << pMT->mnId 
              << " | Chi2: " << chi2_before << " -> " << chi2_after
              << " | n_new=(" << n_ref_new.transpose() << ") d_new=" << d_ref_new
              << " | Valid: " << isPhysicallyValid
              << " | Accept: " << accept << std::endl;

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
        pMT->SetPlaneRef(Eigen::Vector4d(n_ref_new[0], n_ref_new[1], n_ref_new[2], d_ref_new));
        
        // 强制刷新3D角点
        pMT->UpdateCornersFromPlane();
        
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
