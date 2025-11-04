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
      mnVisible(1), mnFound(1)
{
    mnId = nNextId++;
    mnFirstKFid = (pRefKF ? (long long)pRefKF->mnId : -1);
}

Eigen::Vector4d MapText::GetWorldPlane()
{
    std::unique_lock<std::mutex> lock(mMutexPose);

    // 获取参考关键帧的位姿 T_wc (world from camera)
    cv::Mat Twc_ref = mpRefKF->GetPoseInverse();
    
    // 提取旋转 R_wc 和平移 t_wc (即相机光心 O_w)
    cv::Mat Rwc_ref_cv = Twc_ref.rowRange(0, 3).colRange(0, 3);
    cv::Mat Ow_ref_cv = Twc_ref.rowRange(0, 3).col(3);

    Eigen::Matrix3d R_wc;
    R_wc << Twc_ref.at<float>(0,0), Twc_ref.at<float>(0,1), Twc_ref.at<float>(0,2),
            Twc_ref.at<float>(1,0), Twc_ref.at<float>(1,1), Twc_ref.at<float>(1,2),
            Twc_ref.at<float>(2,0), Twc_ref.at<float>(2,1), Twc_ref.at<float>(2,2);
    
    Eigen::Vector3d t_wc; // this is Ow
    t_wc << Ow_ref_cv.at<float>(0), Ow_ref_cv.at<float>(1), Ow_ref_cv.at<float>(2);

    // 平面方程 Pi_c = (n_c, d_c) 变换到世界坐标系 Pi_w = (n_w, d_w)
    // 公式: n_w = R_wc * n_c
    //      d_w = d_c - n_w^T * t_wc
    
    Eigen::Vector3d n_c = mThetaRef.head<3>();
    double d_c = mThetaRef[3];

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
                    mpRefKF = mObservations.begin()->first;
                    // TODO: 需要重新计算 mThetaRef
                    // 这是一个复杂操作，暂不实现。在当前框架下，参考帧不应被轻易删除。
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
 * @brief 计算并返回该文本地标在世界坐标系下的3D包围盒的8个角点
 * @return std::vector<cv::Mat> 8个3x1的Mat，每个代表一个角点的世界坐标
 */
std::vector<cv::Mat> MapText::GetWorldBoundingBox()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (!mWorldCorners.empty()) {
        return mWorldCorners;
    }

    std::vector<cv::Mat> bbox3D_world;

    // --- 1. 从参考关键帧获取原始的2D检测框 ---
    if (!mObservations.count(mpRefKF))
        return bbox3D_world; // 如果没有参考帧的观测信息，则无法计算

    size_t ref_det_idx = mObservations[mpRefKF];
    if (ref_det_idx >= mpRefKF->vTextDetections.size())
        return bbox3D_world;

    const auto& det_bbox_vec = mpRefKF->vTextDetections[ref_det_idx]->bbox;
    
    // 获取2D检测框的四个角点 (像素坐标)
    std::vector<cv::Point2f> corners_2d;
    corners_2d.push_back(cv::Point2f(det_bbox_vec[0], det_bbox_vec[1])); // top-left
    corners_2d.push_back(cv::Point2f(det_bbox_vec[2], det_bbox_vec[1])); // top-right
    corners_2d.push_back(cv::Point2f(det_bbox_vec[2], det_bbox_vec[3])); // bottom-right
    corners_2d.push_back(cv::Point2f(det_bbox_vec[0], det_bbox_vec[3])); // bottom-left

    // --- 2. 将2D角点反投影到参考帧相机坐标系的3D平面上 ---
    std::vector<cv::Mat> front_face_3d_cam;
    const Eigen::Vector3d n_c = mThetaRef.head<3>(); // 平面法向量 (相机坐标系)
    const double d_c = mThetaRef[3];               // 平面距离 (相机坐标系)

    for (const auto& corner : corners_2d)
    {
        // 将2D像素坐标转换为归一化图像坐标
        float u_norm = (corner.x - mpRefKF->cx) / mpRefKF->fx;
        float v_norm = (corner.y - mpRefKF->cy) / mpRefKF->fy;
        Eigen::Vector3d ray(u_norm, v_norm, 1.0);
        ray.normalize();

        // 射线与平面求交，得到3D点在相机坐标系下的坐标
        // 平面方程: n·p + d = 0, 射线上点 p = t*ray
        // t * (n·ray) + d = 0  =>  t = -d / (n·ray)
        double n_dot_ray = n_c.dot(ray);
        if (std::fabs(n_dot_ray) < 1e-6) continue; // 射线与平面平行，忽略

        double t = -d_c / n_dot_ray;
        Eigen::Vector3d p3D_cam_eig = t * ray;
        /*std::cout << "[GetWorldBoundingBox] refKF " << mpRefKF->mnId
              << " 2D=(" << corner.x << "," << corner.y << ")"
              << " -> 3D_cam=(" << p3D_cam_eig.transpose() << ")" 
              << " t=" << t << std::endl;*/
        
        front_face_3d_cam.push_back((cv::Mat_<float>(3, 1) << p3D_cam_eig.x(), p3D_cam_eig.y(), p3D_cam_eig.z()));
    }
    
    if (front_face_3d_cam.size() != 4) return bbox3D_world; // 必须要有4个角点

    // --- 3. 根据平面法线方向，构造一个有厚度的包围盒 ---
    const float BOX_THICKNESS = 0.02; // 假设文本有2cm的厚度
    Eigen::Vector3d n_c_norm = n_c.normalized();
    cv::Mat normal_cv = (cv::Mat_<float>(3, 1) << n_c_norm.x(), n_c_norm.y(), n_c_norm.z());
    
    std::vector<cv::Mat> back_face_3d_cam;
    for (const auto& p_front : front_face_3d_cam) {
        back_face_3d_cam.push_back(p_front - BOX_THICKNESS * normal_cv);
    }
    
    // --- 4. 将包围盒的8个3D角点从参考帧相机坐标系转换到世界坐标系 ---
    cv::Mat Twc_ref = mpRefKF->GetPoseInverse();
    cv::Mat Rwc_ref = Twc_ref.rowRange(0, 3).colRange(0, 3);
    cv::Mat Ow_ref = Twc_ref.rowRange(0, 3).col(3);

    for (size_t i = 0; i < front_face_3d_cam.size(); i++) {
        cv::Mat p_world = Rwc_ref * front_face_3d_cam[i] + Ow_ref;
        /*std::cout << "[GetWorldBoundingBox] world corner " << i
                << " = (" << p_world.at<float>(0) << ","
                << p_world.at<float>(1) << ","
                << p_world.at<float>(2) << ")" << std::endl;*/
        bbox3D_world.push_back(p_world);
    }
    for (const auto& p_cam : back_face_3d_cam) {
        bbox3D_world.push_back(Rwc_ref * p_cam + Ow_ref);
    }
    
    return bbox3D_world;
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
        {det_bbox[0], det_bbox[1]},
        {det_bbox[2], det_bbox[1]},
        {det_bbox[2], det_bbox[3]},
        {det_bbox[0], det_bbox[3]}
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