#include "MapText.h"
#include <opencv2/core/eigen.hpp>
#include <mutex>


namespace ORB_SLAM2
{

long unsigned int MapText::nNextId = 0;

MapText::MapText(KeyFrame* pRefKF, const Eigen::Vector4d& theta, 
                 const TextInfo& textInfo, Map* pMap)
    : mpRefKF(pRefKF), mThetaRef(theta), mTextInfo(textInfo), mpMap(pMap),
      mState(TEXTIMMATURE), mbBad(false), mpReplaced(static_cast<MapText*>(NULL)),
      mnVisible(1), mnFound(1),
      mnOptimizationAttempts(0), mnConsecutiveRejections(0),  // 初始化优化状态追踪
      mnBALocalForKF(0), mnBAFixedForKF(0)  // 初始化 BA 标记
{
    mnId = nNextId++;
    mnFirstKFid = (pRefKF ? (long long)pRefKF->mnId : -1);
}

Eigen::Vector4d MapText::GetWorldPlane()
{
    std::unique_lock<std::mutex> lock(mMutexPose);

    cv::Mat Twc_ref = mpRefKF->GetPoseInverse();

    Eigen::Matrix3d R_wc;
    R_wc << Twc_ref.at<float>(0,0), Twc_ref.at<float>(0,1), Twc_ref.at<float>(0,2),
            Twc_ref.at<float>(1,0), Twc_ref.at<float>(1,1), Twc_ref.at<float>(1,2),
            Twc_ref.at<float>(2,0), Twc_ref.at<float>(2,1), Twc_ref.at<float>(2,2);

    Eigen::Vector3d t_wc;
    t_wc << Twc_ref.at<float>(0,3), Twc_ref.at<float>(1,3), Twc_ref.at<float>(2,3);

    // read mThetaRef
    Eigen::Vector3d n_c = mThetaRef.head<3>();
    double d_c = mThetaRef[3];

    // defensive: ensure normal is normalized
    double n_norm = n_c.norm();
    if (n_norm < 1e-8) {
        std::cerr << "[GetWorldPlane] WARNING: zero normal for MapText " << mnId << std::endl;
        // fallback: return something harmless
        Eigen::Vector4d theta_w; theta_w << 0,0,1, 0;
        return theta_w;
    }
    if (fabs(n_norm - 1.0) > 1e-6) {
        // 如果法向量不是单位向量，需要同时缩放 n 和 d
        // 原始平面方程：n·x + d = 0
        // 归一化后：(n/||n||)·x + (d/||n||) = 0
        n_c /= n_norm;
        d_c /= n_norm;
    }

    Eigen::Vector3d n_w = R_wc * n_c;
    double d_w = d_c - n_w.dot(t_wc);

    Eigen::Vector4d theta_w;
    theta_w.head<3>() = n_w;
    theta_w[3] = d_w;
    return theta_w;
}


KeyFrame* MapText::GetReferenceKeyFrame()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapText::AddObservation(KeyFrame* pKF, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
}

void MapText::EraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            mObservations.erase(pKF);

            // 如果被删除的是参考帧，需要重新指定一个
            if(mpRefKF == pKF)
            {
                if(!mObservations.empty())
                {
                    KeyFrame* pNewRefKF = mObservations.begin()->first;

                    // 计算从旧参考帧到新参考帧的变换 T_new_old = T_new_world * T_world_old
                    cv::Mat Tcw_new = pNewRefKF->GetPose();
                    cv::Mat Twc_old = pKF->GetPoseInverse();
                    cv::Mat T_new_old = Tcw_new * Twc_old;

                    Eigen::Matrix4d T_no;
                    cv::cv2eigen(T_new_old, T_no);
                    Eigen::Matrix3d R_no = T_no.block<3,3>(0,0);
                    Eigen::Vector3d t_no = T_no.block<3,1>(0,3);

                    // 1. 更新平面参数 mThetaRef
                    // n_new = R_no * n_old
                    // d_new = d_old - n_new.dot(t_no)
                    Eigen::Vector3d n_old = mThetaRef.head<3>();
                    double d_old = mThetaRef[3];

                    Eigen::Vector3d n_new = R_no * n_old;
                    double d_new = d_old - n_new.dot(t_no);

                    mThetaRef.head<3>() = n_new;
                    mThetaRef[3] = d_new;

                    // 2. 更新内点 mInlierPoints
                    // P_new = R_no * P_old + t_no
                    std::vector<cv::Mat> newInliers;
                    newInliers.reserve(mInlierPoints.size());
                    for(const auto& pt : mInlierPoints)
                    {
                        if(pt.empty()) continue;
                        Eigen::Vector3d p_old(pt.at<float>(0), pt.at<float>(1), pt.at<float>(2));
                        Eigen::Vector3d p_new = R_no * p_old + t_no;
                        newInliers.push_back((cv::Mat_<float>(3,1) << p_new.x(), p_new.y(), p_new.z()));
                    }
                    mInlierPoints = newInliers;

                    mpRefKF = pNewRefKF;
                }
            }
            
            if(mObservations.empty())
                bBad = true;
        }
    }

    if(bBad)
        SetBadFlag();
}

std::map<KeyFrame*, size_t> MapText::GetObservations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapText::Observations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations.size();
}

void MapText::SetBadFlag()
{
    std::map<KeyFrame*, size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPose);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    // 通知所有观测到它的KF，将它移除
    for(auto& pair : obs)
    {
        pair.first->EraseMapTextMatch(pair.second);
    }

    mpMap->EraseMapText(this);
}

bool MapText::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mbBad;
}

void MapText::Replace(MapText* pMT)
{
    if (!pMT || pMT->mnId == mnId)
        return;

    int nvisible = 0, nfound = 0;
    std::map<KeyFrame*, size_t> obs;

    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(pMT->mMutexFeatures);

        // --- 防御性检查 ---
        if (mbBad) return;

        // --- 转移观测 ---
        obs = mObservations;  // 拷贝旧观测
        mObservations.clear(); // 清空自身
        mbBad = true;          // 标记为坏点
        mpReplaced = pMT;      // 指向被替代者
        nvisible = mnVisible;
        nfound = mnFound;
    }

    // --- 遍历旧观测并迁移 ---
    for (auto& [pKF, idx] : obs)
    {
        if (!pKF || pKF->isBad()) continue;

        // 如果新地标未在该关键帧中被观测，则添加
        if (!pMT->IsInKeyFrame(pKF))
        {
            pMT->AddObservation(pKF, idx);
            pKF->AddMapTextMatch(pMT, idx); // 
        }
        else
        {
            // 如果已有相同索引，移除旧观测
            pKF->EraseMapTextMatch(idx);
        }
    }

    // --- 累积观测信息 ---
    {
        std::unique_lock<std::mutex> lock(pMT->mMutexFeatures);
        pMT->mnVisible += nvisible;
        pMT->mnFound += nfound;
        pMT->mnFirstKFid = std::min(pMT->mnFirstKFid, mnFirstKFid);
    }

    // --- 从地图中清除旧地标 ---
    if (mpMap)
        mpMap->EraseMapText(this);

    std::cout << "[MapText::Replace] MapText " << mnId
              << " replaced by " << pMT->mnId
              << " ('" << pMT->GetText() << "')" << std::endl;
}


MapText* MapText::GetReplaced()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mpReplaced;
}

/**
 * @brief 计算并返回该文本地标在世界坐标系下的3D包围盒的4个【平面】角点
 * * @return std::vector<cv::Mat> 4个3x1的Mat，每个代表一个角点的世界坐标
 */
// 内部辅助函数：根据当前平面参数计算世界角点
// 注意：调用者需要持有 mMutexFeatures 锁
std::vector<cv::Mat> MapText::ComputeCornersFromPlaneInternal()
{
    std::vector<cv::Mat> result;
    
    // 检查前置条件
    if (!mpRefKF || !mObservations.count(mpRefKF)) {
        return result;
    }
    
    size_t ref_det_idx = mObservations[mpRefKF];
    if (ref_det_idx >= mpRefKF->vTextDetections.size()) {
        return result;
    }
    
    const auto& quad_pts = mpRefKF->vTextDetections[ref_det_idx]->quad;
    if (quad_pts.size() != 4) {
        return result;
    }
    
    // 获取2D四边形的四个角点 (像素坐标)
    std::vector<cv::Point2f> corners_2d;
    corners_2d.reserve(4);
    for (int i = 0; i < 4; i++) {
        corners_2d.push_back(cv::Point2f(quad_pts[i].x(), quad_pts[i].y()));
    }
    
    // 将2D角点反投影到参考帧相机坐标系的3D平面上
    std::vector<cv::Mat> corners_3d_cam;
    const Eigen::Vector3d n_c = mThetaRef.head<3>();
    const double d_c = mThetaRef[3];
    
    for (const auto& corner : corners_2d) {
        float u_norm = (corner.x - mpRefKF->cx) / mpRefKF->fx;
        float v_norm = (corner.y - mpRefKF->cy) / mpRefKF->fy;
        Eigen::Vector3d ray(u_norm, v_norm, 1.0);
        
        double n_dot_ray = n_c.dot(ray);
        if (std::fabs(n_dot_ray) < 1e-6) continue;
        
        double t = -d_c / n_dot_ray;
        if (t <= 0.02) continue;
        
        Eigen::Vector3d p3D_cam_eig = t * ray;
        corners_3d_cam.push_back((cv::Mat_<float>(3, 1) << p3D_cam_eig.x(), p3D_cam_eig.y(), p3D_cam_eig.z()));
    }
    
    if (corners_3d_cam.size() != 4) {
        return result;
    }
    
    // 转换到世界坐标系
    cv::Mat Twc_ref = mpRefKF->GetPoseInverse();
    cv::Mat Rwc_ref = Twc_ref.rowRange(0, 3).colRange(0, 3);
    cv::Mat Ow_ref = Twc_ref.rowRange(0, 3).col(3);
    
    result.reserve(4);
    for (const auto& p_cam : corners_3d_cam) {
        result.push_back(Rwc_ref * p_cam + Ow_ref);
    }
    
    return result;
}

std::vector<cv::Mat> MapText::GetWorldBoundingBox()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    
    // 优先返回持久化的角点
    if (!mWorldCorners.empty()) {
        return mWorldCorners;
    }

    // 回退逻辑 (如果 mWorldCorners 为空, e.g. 从旧地图加载)
    std::cout << "[GetWorldBoundingBox] 警告: MapText " << mnId 
              << " 缺少 mWorldCorners, 正在从RefKF回退计算..." << std::endl;
    
    std::vector<cv::Mat> computed = ComputeCornersFromPlaneInternal();
    
    if (computed.size() == 4) {
        mWorldCorners = computed;  // 缓存结果
        return mWorldCorners;
    } else {
        std::cerr << "[GetWorldBoundingBox] Fallback失败: MapText " << mnId 
                  << " 计算角点失败." << std::endl;
        return computed;  // 返回空
    }
}

std::vector<cv::Point2f> MapText::ProjectByHomography(const Frame& F)
{
    std::vector<cv::Point2f> proj_corners;

    if (!mObservations.count(mpRefKF)) return proj_corners;
    size_t ref_det_idx = mObservations[mpRefKF];
    if (ref_det_idx >= mpRefKF->vTextDetections.size()) return proj_corners;

    // 参考帧 bbox 四角点
    const auto& det_bbox = mpRefKF->vTextDetections[ref_det_idx]->bbox;
    std::vector<cv::Point2f> corners_ref = {
        {static_cast<float>(det_bbox[0]), static_cast<float>(det_bbox[1])},
        {static_cast<float>(det_bbox[2]), static_cast<float>(det_bbox[1])},
        {static_cast<float>(det_bbox[2]), static_cast<float>(det_bbox[3])},
        {static_cast<float>(det_bbox[0]), static_cast<float>(det_bbox[3])}
    };

    // 相机位姿
    cv::Mat Tcw_ref = mpRefKF->GetPose();  // world->refCam
    cv::Mat Tcw_cur = F.mTcw;              // world->curCam
    cv::Mat Twc_ref = Tcw_ref.inv();       // refCam->world
    cv::Mat Tcr = Tcw_cur * Twc_ref;       // refCam->curCam


    cv::Mat R, t;
    Tcr.rowRange(0,3).colRange(0,3).convertTo(R, CV_64F);
    Tcr.rowRange(0,3).col(3).convertTo(t, CV_64F);

    // 平面参数 (相机坐标系)
    Eigen::Vector3d n_eig = mThetaRef.head<3>();
    double d = mThetaRef[3];
    cv::Mat n = (cv::Mat_<double>(3,1) << n_eig.x(), n_eig.y(), n_eig.z());

    // 单应性
    cv::Mat Kd;
    F.mK.convertTo(Kd, CV_64F);
    cv::Mat H = Kd * (R - (t * n.t()) / d) * Kd.inv();

    // 投影
    cv::perspectiveTransform(corners_ref, proj_corners, H);
    return proj_corners;
}


bool MapText::HasWorldCorners() const {
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return !mWorldCorners.empty();
}

void MapText::SetWorldCorners(const std::vector<cv::Mat>& corners_world)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mWorldCorners.clear();
    for (const auto &c : corners_world) {
        if (c.empty()) continue;
        // accept 3x1 float or 4x1 homogeneous
        if (c.rows == 3 && c.cols == 1 && c.type() == CV_32F) {
            mWorldCorners.push_back(c.clone());
        } else if (c.rows == 4 && c.cols == 1 && c.type() == CV_32F) {
            float w = c.at<float>(3,0);
            if (fabs(w) < 1e-6f) continue;
            cv::Mat nc = (cv::Mat_<float>(3,1) << c.at<float>(0,0)/w, c.at<float>(1,0)/w, c.at<float>(2,0)/w);
            mWorldCorners.push_back(nc);
        }
    }
}

std::vector<cv::Mat> MapText::GetWorldCorners() const {
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mWorldCorners; // copy
}

void MapText::ClearWorldCorners()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mWorldCorners.clear();
}
// ========================== 平面计算方法 ==========================
// 平面方程定义： n^T * X + d = 0
// 其中 n 为单位法向量，d = -n^T * X0，X0 为平面上一点。
// 因此 d 通常为负值（相机前方平面若法向量朝向相机，d < 0）。
// 坐标变换规则：
//   相机系 -> 世界系：
//      n_w = R_wc * n_c
//      d_w = d_c - n_w^T * t_wc
//   世界系 -> 相机系：
//      n_c = R_cw * n_w
//      d_c = d_w - n_c^T * t_cw
// ===============================================================
void MapText::RefitPlaneFromWorldCorners(KeyFrame* pRefKF)
{
    std::vector<cv::Mat> corners_world;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if (mWorldCorners.empty()) return;
        corners_world = mWorldCorners; // copy
    }

    if (!pRefKF) return;

    // Get Twc_ref (world <- camera_ref)
    cv::Mat Twc_ref = pRefKF->GetPoseInverse(); // 4x4
    // Convert Twc_ref to Eigen for computations
    Eigen::Matrix4d T_world_cam;
    cv::cv2eigen(Twc_ref, T_world_cam);
    Eigen::Matrix3d R_wc = T_world_cam.block<3,3>(0,0);
    Eigen::Vector3d t_wc = T_world_cam.block<3,1>(0,3);

    // Transform world corners to reference camera coords
    std::vector<Eigen::Vector3d> pts_cam;
    for (const auto &pw_mat : corners_world) {
        if (pw_mat.empty() || pw_mat.rows < 3) continue;
        Eigen::Vector3d pw(pw_mat.at<float>(0,0), pw_mat.at<float>(1,0), pw_mat.at<float>(2,0));
        // p_cam = Rcw*(pw - Ow) ; Rcw = R_wc.transpose()
        Eigen::Vector3d p_cam = R_wc.transpose() * (pw - t_wc);
        pts_cam.push_back(p_cam);
    }
    if (pts_cam.size() < 3) return;

    // Fit plane via PCA/SVD (smallest eigenvector is normal)
    Eigen::MatrixXd A(pts_cam.size(), 3);
    for (size_t i = 0; i < pts_cam.size(); ++i) {
        A(i,0) = pts_cam[i].x();
        A(i,1) = pts_cam[i].y();
        A(i,2) = pts_cam[i].z();
    }
    Eigen::Vector3d centroid = A.colwise().mean();
    Eigen::MatrixXd centered = A.rowwise() - centroid.transpose();
    Eigen::Matrix3d cov = (centered.transpose() * centered) / double(pts_cam.size());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d normal = es.eigenvectors().col(0);
    if (normal.norm() < 1e-6) return;
    normal.normalize();
    double d = -normal.dot(centroid);

    // Ensure normal points toward camera (camera center in ref cam = 0)
    if (normal.dot(Eigen::Vector3d(0,0,0) - centroid) < 0) {
        normal = -normal;
        d = -d;
    }

    // Write back into mThetaRef (in reference camera coords) — mThetaRef is Eigen::Vector4d or similar
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        // If your mThetaRef is declared differently, adapt assignment accordingly
        mThetaRef[0] = normal.x();
        mThetaRef[1] = normal.y();
        mThetaRef[2] = normal.z();
        mThetaRef[3] = d;
    }
}




int MapText::GetIndexInKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

void MapText::UpdateSemanticInfo(const TextInfo& newInfo)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if(newInfo.score_semantic < mTextInfo.score_semantic)
        mTextInfo = newInfo;
}

TextStatus MapText::GetState() const {
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mState;
}

void MapText::SetState(const TextStatus &s) {
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mState = s;
}

void MapText::SetPlaneRef(const Eigen::Vector4d& thetaRef)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mThetaRef = thetaRef;
}

Eigen::Vector4d MapText::GetPlaneRef() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mThetaRef;
}

void MapText::UpdateCornersFromPlane()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    
    // 1. 保存旧角点（以防重新计算失败时恢复）
    std::vector<cv::Mat> oldCorners = mWorldCorners;
    
    // 2. 清空当前角点
    mWorldCorners.clear();
    
    // 3. 调用内部函数计算新角点（不打印警告）
    std::vector<cv::Mat> newCorners = ComputeCornersFromPlaneInternal();
    
    // 4. 更新或恢复角点
    if (newCorners.size() == 4) {
        mWorldCorners = newCorners;
        // 成功时不打印，减少日志噪音
    } else if (oldCorners.size() == 4) {
        mWorldCorners = oldCorners;
        std::cout << "[MapText::UpdateCornersFromPlane] 计算失败，恢复旧角点 for MapText " << mnId << std::endl;
    } else {
        std::cout << "[MapText::UpdateCornersFromPlane] 计算失败，无旧角点可恢复 for MapText " << mnId << std::endl;
    }
}

void MapText::SetInlierPoints(const std::vector<cv::Mat>& points)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mInlierPoints = points;
}

std::vector<cv::Mat> MapText::GetInlierPoints()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mInlierPoints;
}

void MapText::SetReplaced(MapText* pMT)
{
    if(pMT->mnId == mnId)
        return;
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mpReplaced = pMT;
}

bool MapText::IsInKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF) != 0);
}

std::string MapText::GetText() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mTextInfo.mean;
}

TextInfo MapText::GetTextInfo() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mTextInfo;
}

} // namespace ORB_SLAM2