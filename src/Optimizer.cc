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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"
#include "OptimizerObject.h"
#include "MapText.h"
#include "TextOptimizer.h"
#include "AdaptiveWeightManager.h"

#include<mutex>

namespace ORB_SLAM2
{


void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }

    // --- PLAN A: Text Corner Optimization ---
    // Added by Copilot: Optimize pose using 3D Text corners projected to 2D detection corners
    struct TextEdgeInfo {
        size_t detIdx;
        vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    };
    vector<TextEdgeInfo> vTextEdges;

    // Iterate over all text detections in this frame
    for(size_t idx=0; idx<pFrame->vTextDetections.size(); idx++)
    {
        if(idx >= pFrame->mvpMapTexts.size()) continue;
        MapText* pMT = pFrame->mvpMapTexts[idx];
        if(!pMT) continue;
        if(pMT->GetState() != TEXTGOOD) continue;

        // Get 3D corners (TL, TR, BR, BL)
        std::vector<cv::Mat> worldCorners = pMT->GetWorldCorners();
        if(worldCorners.size() != 4) continue;

        Detection::Ptr det = pFrame->vTextDetections[idx];
        
        // Get 2D corners
        std::vector<Eigen::Vector2d> corners2d;
        if(det->quad.size() == 4) {
            corners2d = det->quad;
        } else {
            // Fallback to bbox (TL, TR, BR, BL)
            double x1 = det->bbox[0];
            double y1 = det->bbox[1];
            double x2 = det->bbox[2];
            double y2 = det->bbox[3];
            corners2d.push_back(Eigen::Vector2d(x1, y1)); // TL
            corners2d.push_back(Eigen::Vector2d(x2, y1)); // TR
            corners2d.push_back(Eigen::Vector2d(x2, y2)); // BR
            corners2d.push_back(Eigen::Vector2d(x1, y2)); // BL
        }

        TextEdgeInfo edgeInfo;
        edgeInfo.detIdx = idx;

        // Create 4 edges for the 4 corners
        for(int k=0; k<4; ++k)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            
            e->setMeasurement(corners2d[k]);
            
            // Low weight for text boxes (high uncertainty)
            // Standard points use invSigma2. We use a fixed low weight (e.g. sigma=10px -> info=0.01)
            // This ensures text doesn't break the optimization if the box is jittery
            e->setInformation(Eigen::Matrix2d::Identity() * 0.01); 

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono); // Use same delta as mono points

            e->fx = pFrame->fx;
            e->fy = pFrame->fy;
            e->cx = pFrame->cx;
            e->cy = pFrame->cy;

            cv::Mat Xw_cv = worldCorners[k];
            e->Xw[0] = Xw_cv.at<float>(0);
            e->Xw[1] = Xw_cv.at<float>(1);
            e->Xw[2] = Xw_cv.at<float>(2);

            optimizer.addEdge(e);
            edgeInfo.edges.push_back(e);
        }
        vTextEdges.push_back(edgeInfo);
        nInitialCorrespondences++; 
    }

    // [DEBUG] Print text optimization info
    if(!vTextEdges.empty()) {
        std::cout << "[Optimizer] Added " << vTextEdges.size() << " text constraints (4 corners each)." << std::endl;
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        // --- PLAN A: Text Outlier Rejection ---
        for(auto& info : vTextEdges)
        {
            // Check if this detection was marked as outlier
            if(pFrame->mvbOutlierText.size() > info.detIdx && pFrame->mvbOutlierText[info.detIdx])
            {
                for(auto* e : info.edges) e->computeError();
            }

            int nBadCorners = 0;
            for(auto* e : info.edges)
            {
                if(e->chi2() > chi2Mono[it]) 
                {
                    nBadCorners++;
                }
            }

            // If more than 2 corners are bad, mark detection as outlier
            bool isOutlier = (nBadCorners >= 2);
            
            if(isOutlier)
            {
                if(pFrame->mvbOutlierText.size() > info.detIdx)
                    pFrame->mvbOutlierText[info.detIdx] = true;
                
                for(auto* e : info.edges) e->setLevel(1);
                nBad++; // Count as bad observation
            }
            else
            {
                if(pFrame->mvbOutlierText.size() > info.detIdx)
                    pFrame->mvbOutlierText[info.detIdx] = false;
                
                for(auto* e : info.edges) e->setLevel(0);
            }
            
            if(it==2)
                for(auto* e : info.edges) e->setRobustKernel(0);
        }

        // [DEBUG] Print text error stats
        if(!vTextEdges.empty()) {
            float totalChi2 = 0;
            int nActive = 0;
            for(auto& info : vTextEdges) {
                if(pFrame->mvbOutlierText.size() > info.detIdx && !pFrame->mvbOutlierText[info.detIdx]) {
                    for(auto* e : info.edges) totalChi2 += e->chi2();
                    nActive++;
                }
            }
            if(nActive > 0)
                std::cout << "[Optimizer] Iter " << it << ": Active Texts=" << nActive << " AvgChi2=" << totalChi2/(nActive*4) << std::endl;
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

/**
 * @brief 带自适应权重的位姿优化
 * @param pFrame 当前帧
 * @param weights 模态权重结构
 * @return 内点数量
 */
int Optimizer::PoseOptimization(Frame *pFrame, const OptimizationWeights& weights)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                // 应用点特征权重缩放
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * weights.pointInfoScale);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                // 应用点特征权重缩放
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2 * weights.pointInfoScale;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }
    }
    }

    // --- Text Corner Optimization with Adaptive Weights ---
    struct TextEdgeInfo {
        size_t detIdx;
        vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    };
    vector<TextEdgeInfo> vTextEdges;

    for(size_t idx=0; idx<pFrame->vTextDetections.size(); idx++)
    {
        if(idx >= pFrame->mvpMapTexts.size()) continue;
        MapText* pMT = pFrame->mvpMapTexts[idx];
        if(!pMT) continue;
        if(pMT->GetState() != TEXTGOOD) continue;

        std::vector<cv::Mat> worldCorners = pMT->GetWorldCorners();
        if(worldCorners.size() != 4) continue;

        Detection::Ptr det = pFrame->vTextDetections[idx];
        
        std::vector<Eigen::Vector2d> corners2d;
        if(det->quad.size() == 4) {
            corners2d = det->quad;
        } else {
            double x1 = det->bbox[0];
            double y1 = det->bbox[1];
            double x2 = det->bbox[2];
            double y2 = det->bbox[3];
            corners2d.push_back(Eigen::Vector2d(x1, y1));
            corners2d.push_back(Eigen::Vector2d(x2, y1));
            corners2d.push_back(Eigen::Vector2d(x2, y2));
            corners2d.push_back(Eigen::Vector2d(x1, y2));
        }

        TextEdgeInfo edgeInfo;
        edgeInfo.detIdx = idx;

        for(int k=0; k<4; ++k)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            
            e->setMeasurement(corners2d[k]);
            
            // 使用自适应文本权重（基础权重 * 自适应缩放因子）
            double textWeight = weights.textInfoScale;
            e->setInformation(Eigen::Matrix2d::Identity() * textWeight);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pFrame->fx;
            e->fy = pFrame->fy;
            e->cx = pFrame->cx;
            e->cy = pFrame->cy;

            cv::Mat Xw_cv = worldCorners[k];
            e->Xw[0] = Xw_cv.at<float>(0);
            e->Xw[1] = Xw_cv.at<float>(1);
            e->Xw[2] = Xw_cv.at<float>(2);

            optimizer.addEdge(e);
            edgeInfo.edges.push_back(e);
        }
        vTextEdges.push_back(edgeInfo);
        nInitialCorrespondences++;
    }

    // Debug: Print adaptive weights info
    if(!vTextEdges.empty()) {
        std::cout << "[Optimizer] Adaptive weights - Text: " << weights.textInfoScale 
                  << " Point: " << weights.pointInfoScale 
                  << " | " << vTextEdges.size() << " text constraints" << std::endl;
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];
            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        // Text Outlier Rejection
        for(auto& info : vTextEdges)
        {
            if(pFrame->mvbOutlierText.size() > info.detIdx && pFrame->mvbOutlierText[info.detIdx])
            {
                for(auto* e : info.edges) e->computeError();
            }

            int nBadCorners = 0;
            for(auto* e : info.edges)
            {
                if(e->chi2() > chi2Mono[it]) 
                {
                    nBadCorners++;
                }
            }

            bool isOutlier = (nBadCorners >= 2);
            
            if(isOutlier)
            {
                if(pFrame->mvbOutlierText.size() > info.detIdx)
                    pFrame->mvbOutlierText[info.detIdx] = true;
                
                for(auto* e : info.edges) e->setLevel(1);
                nBad++;
            }
            else
            {
                if(pFrame->mvbOutlierText.size() > info.detIdx)
                    pFrame->mvbOutlierText[info.detIdx] = false;
                
                for(auto* e : info.edges) e->setLevel(0);
            }
            
            if(it==2)
                for(auto* e : info.edges) e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, Tracking* pTracker)
{    
    // ========== 0. 获取自适应权重 ==========
    double textInfoScale = 0.01;    // 默认值
    double objectInfoScale = 0.01;
    double pointInfoScale = 1.0;
    
    if (pTracker && pTracker->GetAdaptiveWeightManager()) {
        ModalityWeights weights = pTracker->GetAdaptiveWeightManager()->GetCurrentWeights();
        textInfoScale = weights.textInfoScale;
        objectInfoScale = weights.objectInfoScale;
        pointInfoScale = weights.pointInfoScale;
    }
    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // ========== 新增：收集 Local MapTexts (只要 TEXTGOOD 状态) ==========
    list<MapText*> lLocalMapTexts;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapText*> vpMTs = (*lit)->GetMapTextMatches();
        for(auto pMT : vpMTs)
        {
            if(pMT && !pMT->isBad() && 
               pMT->GetState() == TEXTGOOD &&  // 只取 GOOD 状态
               pMT->mnBALocalForKF != pKF->mnId)
            {
                lLocalMapTexts.push_back(pMT);
                pMT->mnBALocalForKF = pKF->mnId;
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }
    
    // 同时处理 MapTexts 的观测 KF (确保文本观测帧也被加入固定相机)
    for(auto pMT : lLocalMapTexts)
    {
        map<KeyFrame*, size_t> observations = pMT->GetObservations();
        for(auto& obs : observations)
        {
            KeyFrame* pKFi = obs.first;
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::Solver *solver_ptr = nullptr;

    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    solver_ptr = new g2o::BlockSolver_6_3(linearSolver);


    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int maxMPid = 0;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        maxMPid = std::max(maxMPid, id);
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    // ========== 新增：添加 MapText 顶点和二元边 ==========
    // 使用二元边 EdgeHomographyProjectBinary 同时优化相机位姿和文本平面参数
    
    // 安全计算 ID 偏移（避免与 MapPoint ID 冲突）
    int nextTextVertexId = maxMPid + 100;  // 留出安全距离
    
    // 文本边的 Huber 阈值（像素单位，比点特征稍宽松）
    const float thHuberText = sqrt(15.0);
    
    // 存储 MapText 与其 Vertex ID 的映射
    map<MapText*, int> mtToVertexId;
    
    // 存储文本边以便后续检查
    vector<EdgeHomographyProjectBinary*> vpEdgesText;
    vector<MapText*> vpMapTextEdge;
    
    // 记录每个 MapText 的参考帧及其优化前位姿（用于后续姿态增量校正）
    // 如果参考帧在本次 LBA 中被优化，需要用姿态增量重新表达 θ
    map<MapText*, std::pair<KeyFrame*, Eigen::Matrix4d>> mtRefKFPoseOld;
    
    for(auto pMT : lLocalMapTexts)
    {
        if(pMT->isBad()) continue;
        
        // --- 添加文本平面顶点 (VertexTheta3D) ---
        VertexTheta3D* vTheta = new VertexTheta3D();
        Eigen::Vector4d planeRef = pMT->GetPlaneRef();
        Eigen::Vector3d n_ref = planeRef.head<3>();
        double d_ref = planeRef[3];
        // 使用 PlaneToTheta 正确转换：θ = -n/d
        Eigen::Vector3d theta = TextOptimizer::PlaneToTheta(n_ref, d_ref);
        vTheta->setEstimate(theta);
        vTheta->setId(nextTextVertexId);
        vTheta->setMarginalized(true);  // 必须边缘化！BlockSolver_6_3 期望 3 维顶点是 landmark
        optimizer.addVertex(vTheta);
        
        mtToVertexId[pMT] = nextTextVertexId;
        nextTextVertexId++;
        
        // --- 准备参考帧信息 ---
        KeyFrame* pRefKF = pMT->GetReferenceKeyFrame();
        if(!pRefKF || pRefKF->isBad()) continue;
        
        // 参考帧位姿 T_iw (world to camera)
        cv::Mat Tcw_ref = pRefKF->GetPose();
        Eigen::Matrix4d T_iw = Eigen::Matrix4d::Identity();
        for(int r = 0; r < 3; r++)
            for(int c = 0; c < 4; c++)
                T_iw(r, c) = Tcw_ref.at<float>(r, c);
        
        // 记录参考帧及其优化前位姿（用于后续姿态增量校正）
        mtRefKFPoseOld[pMT] = std::make_pair(pRefKF, T_iw);
        
        // 获取参考帧上的检测索引
        int refIdx = pMT->GetIndexInKeyFrame(pRefKF);
        if(refIdx < 0 || refIdx >= (int)pRefKF->vTextDetections.size()) continue;
        
        auto refDet = pRefKF->vTextDetections[refIdx];
        
        // 计算参考帧的 4 个角点归一化射线
        std::vector<Eigen::Vector2d> refRays;
        double fx_ref = pRefKF->fx, fy_ref = pRefKF->fy;
        double cx_ref = pRefKF->cx, cy_ref = pRefKF->cy;
        
        // 从 bbox 获取 4 个角点 (TL, TR, BR, BL)
        double x1 = refDet->bbox[0], y1 = refDet->bbox[1];
        double x2 = refDet->bbox[2], y2 = refDet->bbox[3];
        refRays.push_back(Eigen::Vector2d((x1 - cx_ref) / fx_ref, (y1 - cy_ref) / fy_ref));  // TL
        refRays.push_back(Eigen::Vector2d((x2 - cx_ref) / fx_ref, (y1 - cy_ref) / fy_ref));  // TR
        refRays.push_back(Eigen::Vector2d((x2 - cx_ref) / fx_ref, (y2 - cy_ref) / fy_ref));  // BR
        refRays.push_back(Eigen::Vector2d((x1 - cx_ref) / fx_ref, (y2 - cy_ref) / fy_ref));  // BL
        
        // --- 添加二元边 (观测帧 -> 文本平面) ---
        map<KeyFrame*, size_t> observations = pMT->GetObservations();
        
        for(auto& obs : observations)
        {
            KeyFrame* pKFj = obs.first;
            size_t detIdx = obs.second;
            
            if(pKFj->isBad()) continue;
            
            // 检查这个 KF 是否在优化图中（Local 或 Fixed）
            g2o::VertexSE3Expmap* vPoseJ = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFj->mnId));
            if(!vPoseJ) continue;
            
            // 获取当前帧的观测
            if(detIdx >= pKFj->vTextDetections.size()) continue;
            auto det = pKFj->vTextDetections[detIdx];
            
            // 当前帧内参矩阵
            Eigen::Matrix3d K_j = Eigen::Matrix3d::Identity();
            K_j(0,0) = pKFj->fx; K_j(1,1) = pKFj->fy;
            K_j(0,2) = pKFj->cx; K_j(1,2) = pKFj->cy;
            
            // 获取当前帧的 4 个角点观测值（像素坐标）
            std::vector<Eigen::Vector2d> obsCorners;
            double ox1 = det->bbox[0], oy1 = det->bbox[1];
            double ox2 = det->bbox[2], oy2 = det->bbox[3];
            obsCorners.push_back(Eigen::Vector2d(ox1, oy1));  // TL
            obsCorners.push_back(Eigen::Vector2d(ox2, oy1));  // TR
            obsCorners.push_back(Eigen::Vector2d(ox2, oy2));  // BR
            obsCorners.push_back(Eigen::Vector2d(ox1, oy2));  // BL
            
            // 为 4 个角点分别添加二元边
            for(int k = 0; k < 4; ++k)
            {
                EdgeHomographyProjectBinary* e = new EdgeHomographyProjectBinary();
                
                // 连接两个顶点：相机位姿 和 文本平面
                e->setVertex(0, vPoseJ);               // 优化变量：当前相机位姿
                e->setVertex(1, vTheta);               // 优化变量：文本平面参数
                
                // 设置固定参数
                e->setParams(refRays[k], T_iw, K_j);
                
                // 设置观测值
                e->setMeasurement(obsCorners[k]);
                
                // 设置信息矩阵（自适应权重）
                e->setInformation(Eigen::Matrix2d::Identity() * textInfoScale);
                
                // 设置鲁棒核函数
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberText);
                
                optimizer.addEdge(e);
                vpEdgesText.push_back(e);
                vpMapTextEdge.push_back(pMT);
            }
        }
    }


    // ========== LBA 文本优化统计 ==========
    int nTextVertices = mtToVertexId.size();
    int nTextEdges = vpEdgesText.size();
    std::cout << "\n[LBA-Text] ========== 文本优化开始 =========="
              << "\n[LBA-Text] 参与优化的文本地标数: " << nTextVertices
              << "\n[LBA-Text] 文本边数量: " << nTextEdges << " (每个角点一条边)"
              << "\n[LBA-Text] 文本权重 textInfoScale: " << textInfoScale << std::endl;

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    
    // 记录优化前的 chi2
    optimizer.computeActiveErrors();
    double chi2_text_before = 0.0;
    for(auto* e : vpEdgesText) {
        chi2_text_before += e->chi2();
    }
    
    optimizer.optimize(5);


    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }
    
    // 检查文本边异常值（使用更宽松的阈值）
    int nTextOutliers = 0;
    double chi2_text_stage1 = 0.0;
    for(size_t i=0, iend=vpEdgesText.size(); i<iend; i++)
    {
        EdgeHomographyProjectBinary* e = vpEdgesText[i];
        MapText* pMT = vpMapTextEdge[i];
        
        chi2_text_stage1 += e->chi2();
        
        if(pMT->isBad())
            continue;
        
        // 文本检测框较大，使用更宽松的阈值 (chi2 = 15.0 对应约 3.87 像素)
        if(e->chi2() > 15.0)
        {
            e->setLevel(1);
            nTextOutliers++;
        }
        
        e->setRobustKernel(0);
    }
    
    std::cout << "[LBA-Text] Stage1 完成: chi2 " << chi2_text_before << " -> " << chi2_text_stage1
              << " | 文本异常边: " << nTextOutliers << "/" << vpEdgesText.size() << std::endl;

    // Optimize again without the outliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    
    // 记录 Stage2 后的 chi2
    double chi2_text_stage2 = 0.0;
    for(auto* e : vpEdgesText) {
        if(e->level() == 0) {
            chi2_text_stage2 += e->chi2();
        }
    }
    std::cout << "[LBA-Text] Stage2 完成: chi2 -> " << chi2_text_stage2 << std::endl;

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
    
    // ========== 新增：恢复优化后的文本平面参数 ==========
    // 使用 TextOptimizer::ThetaToPlane 从 θ 恢复 (n, d)，然后做符号校正
    // 关键修复：如果参考帧在本次 LBA 中被优化了，需要用姿态增量重新表达 θ
    int nTextUpdated = 0, nTextSkipped = 0, nPoseIncrementApplied = 0, nCornerUpdateFailed = 0;
    
    for(auto pMT : lLocalMapTexts)
    {
        if(pMT->isBad()) continue;
        
        auto it = mtToVertexId.find(pMT);
        if(it == mtToVertexId.end()) continue;
        
        VertexTheta3D* vTheta = static_cast<VertexTheta3D*>(optimizer.vertex(it->second));
        if(vTheta)
        {
            Eigen::Vector3d optimizedTheta = vTheta->estimate();
            double theta_norm = optimizedTheta.norm();
            
            // 检查 θ 的模长是否有效（太小表示平面太远，太大表示平面太近）
            if(theta_norm < 1e-6 || theta_norm > 100.0)
            {
                nTextSkipped++;
                std::cout << "[LBA-Text] 跳过 MapText " << pMT->mnId 
                          << ": theta_norm=" << theta_norm << " 异常" << std::endl;
                continue;  // 跳过异常情况，保留原始平面参数
            }
            
            // 使用 ThetaToPlane 恢复 (n, d)
            Eigen::Vector3d new_n;
            double new_d;
            TextOptimizer::ThetaToPlane(optimizedTheta, new_n, new_d);
            
            // 符号校正：与原始法向量保持一致，避免法向量翻转
            Eigen::Vector4d oldPlane = pMT->GetPlaneRef();
            Eigen::Vector3d old_n = oldPlane.head<3>();
            if(old_n.norm() > 1e-6 && old_n.dot(new_n) < 0.0)
            {
                new_n = -new_n;
                new_d = -new_d;
            }
            
            // ========== 姿态增量校正 ==========
            // 检查参考帧是否在本次 LBA 中被优化了
            // 如果是，需要用姿态增量把 θ 从"旧参考帧坐标系"变换到"新参考帧坐标系"
            auto refIt = mtRefKFPoseOld.find(pMT);
            if(refIt != mtRefKFPoseOld.end())
            {
                KeyFrame* pRefKF = refIt->second.first;
                const Eigen::Matrix4d& Tcw_old = refIt->second.second;
                
                // 检查参考帧是否在优化图中且不是固定的
                g2o::VertexSE3Expmap* vRefPose = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pRefKF->mnId));
                if(vRefPose && !vRefPose->fixed())
                {
                    // 参考帧被优化了，获取优化后的位姿
                    g2o::SE3Quat SE3_new = vRefPose->estimate();
                    Eigen::Matrix4d Tcw_new = SE3_new.to_homogeneous_matrix();
                    
                    // 计算位姿增量：T_new_old = Tcw_new * Twc_old
                    // Twc_old = Tcw_old^(-1)
                    Eigen::Matrix4d Twc_old = Tcw_old.inverse();
                    Eigen::Matrix4d T_new_old = Tcw_new * Twc_old;
                    
                    Eigen::Matrix3d R_new_old = T_new_old.block<3,3>(0,0);
                    Eigen::Vector3d t_new_old = T_new_old.block<3,1>(0,3);
                    
                    // 计算位姿增量的大小（用于日志）
                    double delta_t = t_new_old.norm();
                    Eigen::AngleAxisd aa(R_new_old);
                    double delta_r_deg = aa.angle() * 180.0 / M_PI;
                    
                    // 平面变换规则（从旧参考系到新参考系）：
                    // n_new = R_new_old * n_old
                    // d_new = d_old - n_new.dot(t_new_old)
                    Eigen::Vector3d n_before = new_n;
                    double d_before = new_d;
                    
                    Eigen::Vector3d n_transformed = R_new_old * new_n;
                    double d_transformed = new_d - n_transformed.dot(t_new_old);
                    
                    new_n = n_transformed;
                    new_d = d_transformed;
                    
                    nPoseIncrementApplied++;
                    if(delta_t > 0.001 || delta_r_deg > 0.1) {
                        std::cout << "[LBA-Text] MapText " << pMT->mnId 
                                  << " 参考帧KF" << pRefKF->mnId << " 位姿被优化"
                                  << " | delta_t=" << delta_t*1000 << "mm"
                                  << " delta_r=" << delta_r_deg << "deg"
                                  << " | d: " << d_before << "->" << new_d << std::endl;
                    }
                }
            }
            
            // 构建新的平面参数 [n_x, n_y, n_z, d]
            Eigen::Vector4d newPlaneRef;
            newPlaneRef.head<3>() = new_n;
            newPlaneRef[3] = new_d;
            
            pMT->SetPlaneRef(newPlaneRef);
            
            // 根据新的平面参数更新世界角点
            std::vector<cv::Mat> oldCorners = pMT->GetWorldCorners();
            pMT->UpdateCornersFromPlane();
            std::vector<cv::Mat> newCorners = pMT->GetWorldCorners();
            
            if(newCorners.size() != 4) {
                nCornerUpdateFailed++;
            }
            
            nTextUpdated++;
        }
    }
    
    // 输出文本优化总结
    std::cout << "[LBA-Text] ========== 文本优化完成 =========="
              << "\n[LBA-Text] 更新成功: " << nTextUpdated << " | 跳过: " << nTextSkipped
              << "\n[LBA-Text] 姿态增量校正次数: " << nPoseIncrementApplied
              << "\n[LBA-Text] 角点更新失败: " << nCornerUpdateFailed
              << "\n[LBA-Text] ==================================\n" << std::endl;

}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
