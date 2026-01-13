/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "ObjectMatcher.h"
#include "TextMatcher.h"
#include "TextOptimizer.h"

#include<mutex>
#include <unistd.h>
#include <cmath>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular, int use_objects):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    use_objects_(use_objects)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{
    std::cout << "Start Local Mapping: ";

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Check recent Objects
            ObjectCulling();

            FuseTextLandmarks();      // 执行融合

            //文本地标状态更新and剔除
            TextStatusUpdate();

            // Triangulate new MapPoints
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())
            {
                //SearchByObjects();
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                //std::cout<<"lm3"<<std::endl;
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap, mpTracker);
                //std::cout<<"lm4"<<std::endl;
                // Check redundant local Keyframes
                KeyFrameCulling();
                TextLandmarkCulling();
                //std::cout<<"lm5"<<std::endl;
            }
            
            /*BBox2 img_bbox(0, 0, mpCurrentKeyFrame->im.cols, mpCurrentKeyFrame->im.rows);
            //auto objects = mpMap->GetAllMapObjects();
            auto Rt = mpCurrentKeyFrame->GetPose();
            Matrix34d Rt_eigen = cvToEigenMatrix<double, float, 3, 4>(Rt);
            auto K_ = cvToEigenMatrix<double, float, 3, 3>(mpCurrentKeyFrame->mK);
            Matrix34d P;
            P = K_ * Rt_eigen;
            for(auto [node_id, attribute]:mpCurrentKeyFrame->graph->attributes){
                auto bb =  attribute.bbox;
                cv::rectangle(mpCurrentKeyFrame->im, cv::Point2i(bb[0], bb[1]),
                                        cv::Point2i(bb[2], bb[3]),
                                        cv::Scalar(255, 255, 255),
                                        2);
                auto obj = attribute.obj;
                if(!obj) continue;
                
                // 检查物体是否在相机前方
                Eigen::Vector3d obj_center = obj->GetEllipsoid().GetCenter();
                Eigen::Vector4d obj_center_homo;
                obj_center_homo << obj_center, 1.0;
                double z = Rt_eigen.row(2).dot(obj_center_homo);
                if(z < 0.1) continue;
                
                auto proj = attribute.obj->GetEllipsoid().project(P);
                auto c = proj.GetCenter();
                auto axes = proj.GetAxes();
                double angle = proj.GetAngle();
                // 检查椭圆参数有效性，避免 OpenCV 异常（包括 NaN）
                if(std::isnan(axes[0]) || std::isnan(axes[1]) || axes[0] <= 0.0 || axes[1] <= 0.0) continue;
                if(obj->isBad())
                    cv::ellipse(mpCurrentKeyFrame->im, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(0, 0, 255), 2);
                else
                    cv::ellipse(mpCurrentKeyFrame->im, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(0, 255, 255), 2);
            }

            const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

            for(size_t i=0; i<vpMapPointMatches.size(); i++)
            {
                MapPoint* pMP = vpMapPointMatches[i];
                if(pMP)
                {
                    if(!pMP->isBad())
                    {
                        cv::circle(mpCurrentKeyFrame->im, mpCurrentKeyFrame->mvKeysUn[i].pt,1,cv::Scalar(0,255,0),-1);
                        cv::Mat p3d = pMP->GetWorldPos();
                        cv::Mat p3Dc = Rt(cv::Rect(0, 0, 3, 3))*p3d + Rt(cv::Rect(3, 0, 1, 3));
                        // Project into Image
                        float invz = 1/p3Dc.at<float>(2);
                        float x = p3Dc.at<float>(0)*invz;;
                        float y = p3Dc.at<float>(1)*invz;;
                        float u = mpCurrentKeyFrame->fx*x+mpCurrentKeyFrame->cx;
                        float v = mpCurrentKeyFrame->fy*y+mpCurrentKeyFrame->cy;
                        cv::circle(mpCurrentKeyFrame->im, cv::Point2f(u, v),1,cv::Scalar(0,255,255),-1);
                        cv::line(mpCurrentKeyFrame->im, mpCurrentKeyFrame->mvKeysUn[i].pt, cv::Point2f(u, v), cv::Scalar(255,255,255), 1, cv::LINE_AA);
                    }
                }
            }

            std::string image_path = "/home/yutong/OA-SLAM/bin/tmp/lm/" + to_string(mpCurrentKeyFrame->mnId) + "_lm.png";
            cv::imwrite(image_path.c_str(), mpCurrentKeyFrame->im);*/

            //mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    //物体
    for(auto [node_id, attribute] : mpCurrentKeyFrame->graph->attributes){
        auto obj = attribute.obj;
        if(obj){
            if(!obj->isBad()){
                mlpRecentAddedObjects.push_back(obj);
            }
        }
    }

    //文本
    UpdateTextObservations(mpCurrentKeyFrame);

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::AddRecentAddedMapText(MapText* pMT)
{
    if(!pMT) return;
    std::unique_lock<std::mutex> lock(mMutexRecentAddedMapTexts);
    mlpRecentAddedMapTexts.push_back(pMT);
}


void LocalMapping::ObjectCulling()
{
    list<Object*>::iterator lit = mlpRecentAddedObjects.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;
    
    while(lit!=mlpRecentAddedObjects.end())
    {
        Object* obj = *lit;
        if(obj->isBad())
        {
            lit = mlpRecentAddedObjects.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)obj->mnLastKFid)>=5 && obj->GetObservationNumber()<=2){
            obj->SetBadFlag();
            lit = mlpRecentAddedObjects.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)obj->mnLastKFid)>=6)
            lit = mlpRecentAddedObjects.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::UpdateTextObservations(KeyFrame* pKF)
{
    if(!pKF) return;

    const std::vector<MapText*>& vpTexts = pKF->GetMapTextMatches();

    int nAdded = 0;
    for(size_t i = 0; i < vpTexts.size(); ++i)
    {
        MapText* pMT = vpTexts[i];
        if(!pMT || pMT->isBad()) continue;

        // 检查地标是否已经记录过该关键帧
        if(!pMT->IsInKeyFrame(pKF))
        {
            pMT->AddObservation(pKF, i);
            std::cout << " addedObs MapTextID=" << pMT->mnId 
                    << " text=\"" << pMT->GetText() << "\"" 
                    << " totalObs=" << pMT->GetObservations().size()
                    << std::endl;
            nAdded++;
        }
    }

    if(nAdded > 0)
        std::cout << "[LocalMapping] Updated " << nAdded
                  << " text observations for KF " << pKF->mnId << std::endl;
}


void LocalMapping::TextStatusUpdate()
{
    if (!mpCurrentKeyFrame) {
    // std::cout << "[LocalMapping] Skip TextStatusUpdate: no current keyframe (tracking lost)" << std::endl;
        return;
    }

    std::unique_lock<std::mutex> lock(mMutexRecentAddedMapTexts);
    if (mlpRecentAddedMapTexts.empty())
        return;

    list<MapText*>::iterator lit = mlpRecentAddedMapTexts.begin();
    long long nCurrentKFid = (long long) mpCurrentKeyFrame->mnId;
    // ---- 参数配置（可根据实验调整） ----
    const int minObsForGood = 3;        // 晋升所需最少观测次数
    const int minAgeForGood = 2;        // 至少经过2个KF考验
    const int maxAgeForCulling = 10;     // IMMATURE状态最大存活周期

    while (lit != mlpRecentAddedMapTexts.end())
    {
        MapText* pMT = *lit;

        if (!pMT || pMT->isBad())
        {
            lit = mlpRecentAddedMapTexts.erase(lit);
            continue;
        }

        std::cout << "MapText " << pMT->mnId << " obs=" << pMT->Observations()
          << " mcurrentKf" << nCurrentKFid
          << " age=" << (nCurrentKFid - pMT->mnFirstKFid)
          << " state=" << pMT->mState << std::endl;


        if (pMT->mState == TEXTIMMATURE)
        {
            // 如果该文字创建在未来关键帧，跳过
            if (pMT->mnFirstKFid > nCurrentKFid){
                ++lit;
                continue;
            }
                       
            int nObs = pMT->Observations();
            long long nAge = (pMT->mnFirstKFid >= 0) ? (nCurrentKFid - pMT->mnFirstKFid) : -1;
            if (nAge < 0) {
                // 防御性日志（便于调试）
                std::cout << "[LocalMapping] Warning: invalid mnFirstKFid for MapText " << pMT->mnId << std::endl;
                // 不做剔除判断，等下次再评估
                ++lit; continue;
            }

            // ---------- 晋升逻辑 ----------
            if (nObs >= minObsForGood && nAge >= minAgeForGood)
            {
                // 调用多帧平面参数优化器
                // 返回 true 表示优化成功且平面参数稳定
                bool optimization_successful = TextOptimizer::OptimizePlaneMultiFrame(pMT, true);

                if (optimization_successful)
                {
                    pMT->mState = TEXTGOOD;
                    std::cout << "[LocalMapping] MapText " << pMT->mnId
                              << " ('" << pMT->GetText()
                              << "') promoted to GOOD after optimization. Obs=" << nObs
                              << " Age=" << nAge << std::endl;

                    lit = mlpRecentAddedMapTexts.erase(lit);
                    continue;
                }
                else
                {
                    // 优化失败但观测数足够，可能是几何不一致
                    // 给予更多机会，如果超过最大年龄则剔除
                    std::cout << "[LocalMapping] MapText " << pMT->mnId
                              << " optimization failed, remain IMMATURE. Obs=" << nObs
                              << " Age=" << nAge << std::endl;
                }
            }

            // ---------- 剔除逻辑 ----------
            // 原有条件：超龄且观测不足
            if (nAge >= maxAgeForCulling && nObs < minObsForGood)
            {
                pMT->SetBadFlag();
                lit = mlpRecentAddedMapTexts.erase(lit);
                std::cout << "[LocalMapping] MapText " << pMT->mnId
                          << " ('" << pMT->GetText()
                          << "') culled (IMMATURE, few obs)." << std::endl;
                continue;
            }
            
            // ===== 第三层：僵尸清除机制 =====
            // 连续多次被优化器拒绝 + 存活时间较长 → 确定是死锁，强制清除
            const int maxConsecutiveRejections = 5;  // 连续 5 次被拒绝
            const int minAgeForZombieCulling = 15;   // 至少存活 15 帧
            
            int nRejections = pMT->GetConsecutiveRejections();
            if (nRejections >= maxConsecutiveRejections && nAge >= minAgeForZombieCulling)
            {
                pMT->SetBadFlag();
                lit = mlpRecentAddedMapTexts.erase(lit);
                std::cout << "[LocalMapping] MapText " << pMT->mnId
                          << " ('" << pMT->GetText()
                          << "') culled as ZOMBIE (" << nRejections 
                          << " consecutive rejections, age=" << nAge << ")" << std::endl;
                continue;
            }
        }

        // 已经变为 GOOD 或 BAD 的直接移除
        if (pMT->mState != TEXTIMMATURE)
            lit = mlpRecentAddedMapTexts.erase(lit);
        else
            ++lit;
    }
}


void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchByObjects()
{
    auto K_ = cvToEigenMatrix<double, float, 3, 3>(mpCurrentKeyFrame->mK);
    ObjectMatcher obj_matcher = ObjectMatcher(K_);
    for(auto [node_id, attribute] : mpCurrentKeyFrame->graph->attributes){
        auto bb_det =  attribute.bbox;
        auto obj = attribute.obj;
        if(!obj) continue;
        auto sMPs = obj->GetAssociatedMapPoints();
        std::vector<MapPoint*> vMPs(sMPs.begin(), sMPs.end());
        auto vIndices_in_box = mpCurrentKeyFrame->GetFeaturesInBox(bb_det[0], bb_det[2], bb_det[1], bb_det[3]);
        obj_matcher.SearchInBox(mpCurrentKeyFrame, vMPs, vIndices_in_box);
    }
    
}

// --- 辅助函数：平面间角度（°） ---
inline double AngleBetweenPlanes(const Eigen::Vector4d& plane1, const Eigen::Vector4d& plane2)
{
    double cosTheta = plane1.head<3>().normalized().dot(plane2.head<3>().normalized());
    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
    return std::acos(cosTheta) * 180.0 / M_PI;
}

// --- 辅助函数：点到平面距离 ---
inline double PointToPlaneDist(const Eigen::Vector3d& pt, const Eigen::Vector4d& plane)
{
    return std::abs(plane.head<3>().dot(pt) + plane[3]);
}

// --- 辅助函数：语义相似度 ---
inline std::string NormalizeTextForSim(const std::string &s) {
    // 简单 normalize: 去两侧空格，去控制字符，替换全角符号为半角的常见替换，去重复空格
    std::string t;
    t.reserve(s.size());
    for (char c : s) {
        // 去掉不可见字符
        if (c == '\r' || c == '\n' || c == '\t') continue;
        // 可考虑把全角->半角的处理放在这里（略）
        t.push_back(c);
    }
    // trim spaces
    size_t start = 0;
    while (start < t.size() && isspace((unsigned char)t[start])) ++start;
    size_t end = t.size();
    while (end > start && isspace((unsigned char)t[end-1])) --end;
    std::string out = t.substr(start, end-start);

    // Collapse multiple spaces
    std::string out2;
    out2.reserve(out.size());
    bool last_space = false;
    for (char c: out) {
        if (isspace((unsigned char)c)) {
            if (!last_space) { out2.push_back(' '); last_space = true; }
        } else {
            out2.push_back(c);
            last_space = false;
        }
    }
    return out2;
}

inline float Jaccard2Gram(const std::string &a, const std::string &b) {
    // character 2-gram Jaccard (works for Chinese too as characters)
    auto make_grams = [](const std::string &s) {
        std::unordered_set<std::string> grams;
        if (s.size() == 0) return grams;
        if (s.size() == 1) { grams.insert(s); return grams; }
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            grams.insert(s.substr(i, 2));
        }
        return grams;
    };
    auto A = make_grams(a);
    auto B = make_grams(b);
    if (A.empty() && B.empty()) return 1.0f;
    size_t inter = 0;
    for (auto &g : A) if (B.count(g)) ++inter;
    size_t uni = A.size() + B.size() - inter;
    if (uni == 0) return 0.0f;
    return static_cast<float>(inter) / static_cast<float>(uni);
}

inline float ComputeSemanticSim(const std::string& s1_orig, const std::string& s2_orig)
{
    std::string s1 = NormalizeTextForSim(s1_orig);
    std::string s2 = NormalizeTextForSim(s2_orig);

    // If either is empty after normalize
    if (s1.empty() && s2.empty()) return 1.0f;
    if (s1.empty() || s2.empty()) return 0.0f;

    // Short-text special handling
    if (s1.size() <= 2 || s2.size() <= 2) {
        // exact or substring match => high sim
        if (s1 == s2) return 1.0f;
        if (s1.find(s2) != std::string::npos || s2.find(s1) != std::string::npos) return 0.9f;
        // else use len-normalized edit but len small -> be permissive
        TextMatcher tm(0,0,0);
        int dist = tm.LevenshteinDistance(s1, s2);
        int maxLen = std::max((int)s1.size(), (int)s2.size());
        float levSim = 1.0f - static_cast<float>(dist) / static_cast<float>(maxLen);
        if (levSim < 0.0f) levSim = 0.0f;
        // boost a bit
        return std::max(levSim, 0.4f);
    }

    // Normal-length text: combine normalized Levenshtein and 2-gram Jaccard
    TextMatcher tm(0,0,0);
    int dist = tm.LevenshteinDistance(s1, s2);
    int maxLen = std::max((int)s1.size(), (int)s2.size());
    float levSim = 1.0f - static_cast<float>(dist) / static_cast<float>(maxLen);
    if (levSim < 0.0f) levSim = 0.0f;

    float jacc = Jaccard2Gram(s1, s2); // 0..1

    // combine: weighted average or max
    // weights tuned to favor exact/partial matches but be robust to typos
    float sim = std::max(levSim, jacc); // conservative: take whichever is larger
    // optionally blend: sim = 0.6f*levSim + 0.4f*jacc;

    return sim;
}


// --- 主函数：文本地标融合 ---
void LocalMapping::FuseTextLandmarks()
{
    KeyFrame* pCurrentKF = mpCurrentKeyFrame;
    if (!pCurrentKF || pCurrentKF->isBad()) return;

    // 收集当前关键帧及其邻居的文本地标
    std::set<MapText*> spLocalMapTexts;
    auto addTexts = [&](KeyFrame* pKF)
    {
        if (!pKF || pKF->isBad()) return;
        const auto& texts = pKF->GetMapTextMatches();
        for (MapText* pT : texts)
            if (pT && !pT->isBad()) spLocalMapTexts.insert(pT);
    };

    addTexts(pCurrentKF);
    for (KeyFrame* pKFi : pCurrentKF->GetBestCovisibilityKeyFrames(10))
        addTexts(pKFi);

    if (spLocalMapTexts.size() < 2) return;

    vector<MapText*> vTexts(spLocalMapTexts.begin(), spLocalMapTexts.end());
    vector<bool> vbFused(vTexts.size(), false);
    //std::cout << "[LocalMapping] Running FuseTextLandmarks(), candidates=" << spLocalMapTexts.size() << std::endl;

    const float minSemanticSim = 0.85f;
    const double maxAngleDiff = 10.0;   // degrees
    const double maxPlaneDist = 0.08;   // meters
    const int minCommonKFs = 2;

    int nFused = 0;

    for (size_t i = 0; i < vTexts.size(); ++i)
    {
        MapText* pMT1 = vTexts[i];
        if (!pMT1 || pMT1->isBad() || vbFused[i]) continue;

        Eigen::Vector4d plane1 = pMT1->GetWorldPlane();
        Eigen::Vector3d center1 = -plane1.head<3>().normalized() * plane1[3];
        std::string text1 = pMT1->GetText();
        auto obs1 = pMT1->GetObservations();

        for (size_t j = i + 1; j < vTexts.size(); ++j)
        {
            MapText* pMT2 = vTexts[j];
            if (!pMT2 || pMT2->isBad() || vbFused[j]) continue;

            // --- Step 1: 语义一致性 ---
            std::string text2 = pMT2->GetText();
            float semanticSim = ComputeSemanticSim(text1, text2);
            if (semanticSim < minSemanticSim) continue;

            // --- Step 2: 几何一致性 ---
            Eigen::Vector4d plane2 = pMT2->GetWorldPlane();
            double angleDiff = AngleBetweenPlanes(plane1, plane2);
            if (angleDiff > maxAngleDiff) continue;

            Eigen::Vector3d center2 = -plane2.head<3>().normalized() * plane2[3];
            double d12 = PointToPlaneDist(center1, plane2);
            double d21 = PointToPlaneDist(center2, plane1);
            if (d12 > maxPlaneDist || d21 > maxPlaneDist) continue;

            // --- Step 3: 共视一致性 ---
            auto obs2 = pMT2->GetObservations();
            int nCommon = 0;
            for (const auto& it : obs1)
                if (obs2.count(it.first)) nCommon++;
            if (nCommon < minCommonKFs) continue;

            // --- Step 4: 选择保留者 ---
            MapText* pKeep = nullptr;
            MapText* pReplace = nullptr;

            if (pMT1->Observations() > pMT2->Observations()) {
                pKeep = pMT1; pReplace = pMT2; vbFused[j] = true;
            } else if (pMT2->Observations() > pMT1->Observations()) {
                pKeep = pMT2; pReplace = pMT1; vbFused[i] = true;
            } else {
                // 状态优先（GOOD 胜出），否则 ID 小者胜
                if (pMT1->GetState() == TEXTGOOD && pMT2->GetState() != TEXTGOOD)
                    pKeep = pMT1, pReplace = pMT2, vbFused[j] = true;
                else if (pMT2->GetState() == TEXTGOOD && pMT1->GetState() != TEXTGOOD)
                    pKeep = pMT2, pReplace = pMT1, vbFused[i] = true;
                else if (pMT1->mnId < pMT2->mnId)
                    pKeep = pMT1, pReplace = pMT2, vbFused[j] = true;
                else
                    pKeep = pMT2, pReplace = pMT1, vbFused[i] = true;
            }

            if (!pKeep || !pReplace || pKeep == pReplace) continue;

            // --- Step 5: 执行融合 ---
            pReplace->Replace(pKeep);
            nFused++;

            cout << "[LocalMapping] Fused MapText " << pReplace->mnId << " ('" << pReplace->GetText()
                 << "') → " << pKeep->mnId << " ('" << pKeep->GetText() << "')"
                 << " [Sim=" << semanticSim << ", Angle=" << angleDiff
                 << ", Dist=" << (d12 + d21) / 2.0 << ", CommonKFs=" << nCommon << "]" << endl;

            if (vbFused[i]) break; // 当前已被融合，无需继续
        }
    }

    if (nFused > 0)
        cout << "[LocalMapping] ✓ Fused " << nFused << " text landmarks." << endl;
}

void LocalMapping::TextLandmarkCulling()
{
    // 获取当前活动地图中的所有文本地标
    // 注意：GetAllMapTexts() 内部应该处理好锁
    std::vector<MapText*> vpAllMapTexts = mpMap->GetAllMapTexts();

    if (vpAllMapTexts.empty())
        return;

    int culledCount = 0;
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId; // 用于可能的基于时间的剔除

    // --- 参数配置 ---
    // 对于已经是 TEXTGOOD 状态的地标，如果观测数变得太少，则剔除
    const int minObservationsForGood = 2; // GOOD 状态至少需要的观测数 (比晋升时的要求低)
    // 可选：类似于 TextSLAM 论文 [cite: 366] 的剔除条件
    // const float maxBadRatio = 0.9f; // 如果坏帧比例过高
    // const int maxBadFrames = 40;   // 如果坏帧绝对数量过多
    // 注意：实现这些需要 MapText 跟踪好帧/坏帧观测次数，这里暂时不用

    for (MapText* pMT : vpAllMapTexts)
    {
        // 基本检查
        if (!pMT || pMT->isBad()) // 跳过空指针或已标记为坏点的地标
            continue;

        TextStatus state = pMT->GetState(); // 获取当前状态

        // --- 仅检查 GOOD 状态的地标 ---
        // IMMATURE 状态的剔除主要在 TextStatusUpdate 中完成
        if (state == TEXTGOOD)
        {
            int nObs = pMT->Observations(); // 获取当前观测数

            // 条件1: 观测次数是否低于维持 GOOD 状态的最低要求？
            // 这通常发生在关联的关键帧被 KeyFrameCulling 移除后
            if (nObs < minObservationsForGood)
            {
                pMT->SetBadFlag(); // 标记为坏点
                culledCount++;
                std::cout << "[LocalMapping] Culled GOOD MapText " << pMT->mnId
                          << " ('" << pMT->GetText() << "') due to low observation count: "
                          << nObs << " (min required: " << minObservationsForGood << ")" << std::endl;
                continue; // 处理下一个
            }

            // 可选条件2: 检查观测质量 (类似TextSLAM论文 [cite: 366])
            // 需要 MapText 记录被认为是内点/外点的帧数
            /*
            int goodFrames = pMT->GetGoodObservationsCount(); // 假设有这个函数
            int badFrames = pMT->GetBadObservationsCount();   // 假设有这个函数
            if (badFrames > maxBadFrames || (goodFrames > 0 && (float)badFrames / goodFrames > maxBadRatio))
            {
                pMT->SetBadFlag();
                culledCount++;
                std::cout << "[LocalMapping] Culled GOOD MapText " << pMT->mnId
                          << " ('" << pMT->GetText() << "') due to poor observation quality (bad/good): "
                          << badFrames << "/" << goodFrames << std::endl;
                continue;
            }
            */

             // 可选条件3：如果需要，可以添加基于找到率 (Found Ratio) 的剔除
             // 需要 MapText 像 MapPoint 一样维护 mnVisible 和 mnFound
             /*
             if (pMT->GetFoundRatio() < 0.25f && pMT->Observations() > 3) // 仅在有足够观测时检查
             {
                 pMT->SetBadFlag();
                 culledCount++;
                 std::cout << "[LocalMapping] Culled GOOD MapText " << pMT->mnId
                           << " ('" << pMT->GetText() << "') due to low found ratio: "
                           << pMT->GetFoundRatio() << std::endl;
                 continue;
             }
             */
        }
        // 对于 IMMATURE 状态，主要依赖 TextStatusUpdate 进行剔除
        // 也可以在这里添加一个非常宽松的绝对年龄上限（例如，存在超过 50 个关键帧还未转正）
        /*
        else if (state == TEXTIMMATURE)
        {
            long long firstKFid = pMT->mnFirstKFid;
            long long kfSpan = (firstKFid >= 0) ? (nCurrentKFid - firstKFid) : -1;
            const int absoluteMaxKFSpan = 50; // 绝对最大存活期
            if (kfSpan > absoluteMaxKFSpan)
            {
                 pMT->SetBadFlag();
                 culledCount++;
                 std::cout << "[LocalMapping] Culled IMMATURE MapText " << pMT->mnId
                           << " ('" << pMT->GetText() << "') due to exceeding absolute max KFSpan: "
                           << kfSpan << std::endl;
                 continue;
            }
        }
        */
    }

    if (culledCount > 0)
        std::cout << "[LocalMapping] ✓ TextLandmarkCulling removed " << culledCount << " landmarks." << std::endl;
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    //const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    //////////////////////////////////
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestObjectCovisibilityKeyFrames(nn);
    
    //////////////////////////////////
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        //std::cout<<"pKFi id:"<<pKFi->mnId<<std::endl;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }
    //std::cout<<"vpTargetKFs:"<<vpTargetKFs.size()<<std::endl;

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }
    //std::cout<<"lm1"<<std::endl;
    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
    //std::cout<<"lm2"<<std::endl;
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
    //std::cout<<"lm6"<<std::endl;
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs){
            //std::cout<<"SetBadFlag"<<std::endl;
            pKF->SetBadFlag();
        }
            
    }
    //std::cout<<"lm7"<<std::endl;
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mlpRecentAddedObjects.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
