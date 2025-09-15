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


} // namespace ORB_SLAM2