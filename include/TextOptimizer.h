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
