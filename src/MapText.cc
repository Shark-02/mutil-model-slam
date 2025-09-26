#include "MapText.h"

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
}

Eigen::Vector4d MapText::GetWorldPlane()
{
    std::unique_lock<std::mutex> lock(mMutexPose);

    // 获取参考关键帧的位姿 T_wc (world from camera)
    cv::Mat Twc_ref = mpRefKF->GetPose();
    
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
    // TODO: 实现地标融合逻辑 (高级功能)
    // 1. 获取 pMT 的所有观测
    // 2. 将这些观测添加到当前 MapText 对象
    // 3. 通知所有观测到 pMT 的 KeyFrame，让它们更新指针指向当前对象
    // 4. 将 pMT 设为 bad，并更新其 mpReplaced 指针
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
        std::cout << "[GetWorldBoundingBox] refKF " << mpRefKF->mnId
              << " 2D=(" << corner.x << "," << corner.y << ")"
              << " -> 3D_cam=(" << p3D_cam_eig.transpose() << ")" 
              << " t=" << t << std::endl;
        
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
        std::cout << "[GetWorldBoundingBox] world corner " << i
                << " = (" << p_world.at<float>(0) << ","
                << p_world.at<float>(1) << ","
                << p_world.at<float>(2) << ")" << std::endl;
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



int MapText::GetIndexInKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}


} // namespace ORB_SLAM2