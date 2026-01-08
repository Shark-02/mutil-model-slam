/**
 * TextOptimizer.h
 * 
 * 文本平面参数优化器
 * 参考 TextSLAM 的 ThetaOptimMultiFs 实现
 * 适配 RGBD 输入，使用 g2o 进行非线性优化
 * 
 * 核心改进：使用 θ = -n/d (3维) 代替 (n,d) (4维) 进行优化
 * 优势：单应性公式 H = R + t*θ^T 完全没有除法，彻底避免数值爆炸
 */

#ifndef TEXT_OPTIMIZER_H
#define TEXT_OPTIMIZER_H

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace ORB_SLAM2
{

class MapText;
class KeyFrame;

// ============================================================================
// g2o 顶点：θ = -n/d (3维向量，参考帧坐标系下)
// 参考 TextSLAM: 用 θ 代替 (n, d) 可以避免所有除法
// ============================================================================
class VertexTheta3D : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexTheta3D() {}

    virtual void setToOriginImpl() override {
        _estimate.setZero();
    }

    virtual void oplusImpl(const double* update) override {
        Eigen::Map<const Eigen::Vector3d> v(update);
        _estimate += v;
    }

    virtual bool read(std::istream& in) override { return true; }
    virtual bool write(std::ostream& out) const override { return true; }
};

// ============================================================================
// g2o 边：3D点到平面的距离残差 (一元边)
// 使用 θ = -n/d，平面方程变为：θ^T * X = 1 (对于平面上的点)
// 残差 = θ^T * X - 1
// ============================================================================
class EdgePointToPlaneTheta : public g2o::BaseUnaryEdge<1, double, VertexTheta3D>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePointToPlaneTheta() {}

    // 设置3D点 (参考帧坐标系，与 θ 同一坐标系)
    void setPoint3D(const Eigen::Vector3d& pt) {
        point3D_ = pt;
    }

    virtual void computeError() override {
        const VertexTheta3D* v = static_cast<const VertexTheta3D*>(_vertices[0]);
        Eigen::Vector3d theta = v->estimate();
        
        // 平面方程：n^T * X + d = 0
        // 变换为：(-n/d)^T * X = 1，即 θ^T * X = 1
        // 残差 = θ^T * X - 1
        _error[0] = theta.dot(point3D_) - 1.0;
    }

    virtual bool read(std::istream& in) override { return true; }
    virtual bool write(std::ostream& out) const override { return true; }

private:
    Eigen::Vector3d point3D_;
};

// ============================================================================
// g2o 边：基于单应性的角点重投影误差 (一元边)
// 公式：H = R_rel + t_rel * θ^T  （完全没有除法！）
// ============================================================================
class EdgeHomographyProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexTheta3D>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeHomographyProjection() {}

    // 设置参数
    void setParams(const Eigen::Vector2d& ray_ref,      // 参考帧中角点的归一化射线 (u-cx)/fx, (v-cy)/fy
                   const Eigen::Matrix3d& R_rel,        // 相对旋转：从参考帧到观测帧
                   const Eigen::Vector3d& t_rel,        // 相对平移：从参考帧到观测帧
                   const Eigen::Matrix3d& K)            // 观测帧的内参
    {
        ray_ref_ = ray_ref;
        R_rel_ = R_rel;
        t_rel_ = t_rel;
        K_ = K;
    }

    virtual void computeError() override {
        const VertexTheta3D* vTheta = static_cast<const VertexTheta3D*>(_vertices[0]);
        Eigen::Vector3d theta = vTheta->estimate();  // θ = -n/d (参考帧坐标系)
        
        // ========================================================================
        // 单应性矩阵：H = R_rel + t_rel * θ^T
        // 完全没有除法！这是 TextSLAM 的核心公式
        // ========================================================================
        Eigen::Matrix3d H = R_rel_ + t_rel_ * theta.transpose();
        
        // 参考帧中的射线（归一化坐标 + 1）
        Eigen::Vector3d ray_3d(ray_ref_[0], ray_ref_[1], 1.0);
        
        // 通过单应性矩阵投影到观测帧的归一化相机坐标系
        Eigen::Vector3d p_obs = H * ray_3d;
        
        // 检查投影点是否在相机前方（Z > 0）
        if (p_obs[2] <= 0.01) {
            _error = Eigen::Vector2d(1e6, 1e6);
            return;
        }
        
        // 投影到像素坐标
        double inv_z = 1.0 / p_obs[2];
        double u = K_(0,0) * p_obs[0] * inv_z + K_(0,2);
        double v = K_(1,1) * p_obs[1] * inv_z + K_(1,2);
        
        // 计算重投影误差
        _error = Eigen::Vector2d(u, v) - _measurement;
    }

    virtual bool read(std::istream& in) override { return true; }
    virtual bool write(std::ostream& out) const override { return true; }

private:
    Eigen::Vector2d ray_ref_;
    Eigen::Matrix3d R_rel_;
    Eigen::Vector3d t_rel_;
    Eigen::Matrix3d K_;
};

// ============================================================================
// g2o 边：基于单应性的角点重投影误差 (二元边 - LBA专用)
// 连接：VertexSE3Expmap (相机位姿) 和 VertexTheta3D (文本平面)
// 公式：H = K_j * (R_ji + t_ji * θ^T) * K_i^{-1}
// 这是紧耦合优化的关键：同时优化相机位姿和文本平面参数
// ============================================================================
class EdgeHomographyProjectBinary : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, 
                                                                g2o::VertexSE3Expmap,  // 当前帧相机位姿
                                                                VertexTheta3D>          // 文本平面参数
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeHomographyProjectBinary() {}

    /**
     * @brief 设置固定参数
     * @param ray_ref       参考帧中角点的归一化射线 (u-cx)/fx, (v-cy)/fy
     * @param T_iw          参考帧的世界到相机变换 (4x4)
     * @param K_j           当前观测帧的内参矩阵
     */
    void setParams(const Eigen::Vector2d& ray_ref,
                   const Eigen::Matrix4d& T_iw,
                   const Eigen::Matrix3d& K_j)
    {
        ray_ref_ = ray_ref;
        T_iw_ = T_iw;
        K_j_ = K_j;
        
        // 预计算参考帧的旋转和平移
        R_iw_ = T_iw_.block<3,3>(0,0);
        t_iw_ = T_iw_.block<3,1>(0,3);
    }

    virtual void computeError() override {
        // 获取优化变量
        const g2o::VertexSE3Expmap* vPoseJ = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const VertexTheta3D* vTheta = static_cast<const VertexTheta3D*>(_vertices[1]);
        
        // 当前帧位姿 T_jw
        g2o::SE3Quat SE3_jw = vPoseJ->estimate();
        Eigen::Matrix3d R_jw = SE3_jw.rotation().toRotationMatrix();
        Eigen::Vector3d t_jw = SE3_jw.translation();
        
        // 文本平面参数 θ = -n/d (参考帧坐标系)
        Eigen::Vector3d theta_i = vTheta->estimate();
        
        // ========================================================================
        // 计算相对变换：从参考帧 i 到当前帧 j
        // T_ji = T_jw * T_wi = T_jw * T_iw^{-1}
        // R_ji = R_jw * R_iw^T
        // t_ji = t_jw - R_ji * t_iw = t_jw - R_jw * R_iw^T * t_iw
        // ========================================================================
        Eigen::Matrix3d R_ji = R_jw * R_iw_.transpose();
        Eigen::Vector3d t_ji = t_jw - R_ji * t_iw_;
        
        // ========================================================================
        // 单应性矩阵：H = R_ji + t_ji * θ^T (参考帧坐标系下的 θ)
        // 注意：θ 是在参考帧坐标系下定义的，所以直接用 R_ji 和 t_ji
        // ========================================================================
        Eigen::Matrix3d H = R_ji + t_ji * theta_i.transpose();
        
        // 参考帧中的射线（归一化坐标）
        Eigen::Vector3d ray_3d(ray_ref_[0], ray_ref_[1], 1.0);
        
        // 通过单应性矩阵投影到当前帧的归一化相机坐标系
        Eigen::Vector3d p_cam_j = H * ray_3d;
        
        // 检查投影点是否在相机前方
        if (p_cam_j[2] <= 0.01) {
            _error = Eigen::Vector2d(1e6, 1e6);
            return;
        }
        
        // 投影到像素坐标
        double inv_z = 1.0 / p_cam_j[2];
        double u = K_j_(0,0) * p_cam_j[0] * inv_z + K_j_(0,2);
        double v = K_j_(1,1) * p_cam_j[1] * inv_z + K_j_(1,2);
        
        // 计算重投影误差
        _error = Eigen::Vector2d(u, v) - _measurement;
    }

    virtual bool read(std::istream& in) override { return true; }
    virtual bool write(std::ostream& out) const override { return true; }

private:
    Eigen::Vector2d ray_ref_;       // 参考帧归一化射线
    Eigen::Matrix4d T_iw_;          // 参考帧位姿 (world to camera)
    Eigen::Matrix3d R_iw_;          // 预计算的旋转
    Eigen::Vector3d t_iw_;          // 预计算的平移
    Eigen::Matrix3d K_j_;           // 当前帧内参
};

// ============================================================================
// 文本平面优化器类
// ============================================================================
class TextOptimizer
{
public:
    /**
     * @brief 使用多帧观测优化文本平面参数 (θ = -n/d)
     * 
     * @param pMT           要优化的 MapText 地标
     * @param bFixScale     是否固定尺度（对于RGBD通常为true）
     * @return true         优化成功且平面稳定
     * @return false        优化失败或平面不稳定
     */
    static bool OptimizePlaneMultiFrame(MapText* pMT, bool bFixScale = true);

    /**
     * @brief 从 (n, d) 转换到 θ = -n/d
     */
    static Eigen::Vector3d PlaneToTheta(const Eigen::Vector3d& n, double d);

    /**
     * @brief 从 θ = -n/d 恢复 (n, d)
     * 注意：只能恢复到一个尺度因子，我们假设 |n| = 1
     */
    static void ThetaToPlane(const Eigen::Vector3d& theta, Eigen::Vector3d& n, double& d);

private:
    // 获取检测框的四个角点归一化射线
    static std::vector<Eigen::Vector2d> GetCornerRays(
        KeyFrame* pKF,
        const Eigen::Vector4d& bbox);
};

} // namespace ORB_SLAM2

#endif // TEXT_OPTIMIZER_H
