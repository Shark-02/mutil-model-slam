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

#include "Map.h"
#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
    graph_3d = new Graph();//nullptr;
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}



void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();

    for(set<Object*>::iterator sit=mspObjects.begin(), send=mspObjects.end(); sit!=send; sit++)
        delete *sit;
    mspObjects.clear();
}

void Map::AddObject(Object *obj){
    unique_lock<mutex> lock(mMutexMap);
    mspObjects.insert(obj);
}

std::vector<Object*> Map::GetAllObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return std::vector<Object*>(mspObjects.begin(),mspObjects.end());
}

void Map::AddMapText(MapText* pMT)
{
    // 加锁，确保线程安全
    std::unique_lock<std::mutex> lock(mMutexMapTexts);
    
    // 将新的MapText指针添加到容器中
    mspMapTexts.push_back(pMT);
}

void Map::EraseMapText(MapText* pMT)
{
    // 加锁，确保线程安全
    std::unique_lock<std::mutex> lock(mMutexMapTexts);

    // 在vector中查找要删除的MapText指针
    for (auto it = mspMapTexts.begin(); it != mspMapTexts.end(); ++it)
    {
        if (*it == pMT)
        {
            // 找到后，从vector中移除
            mspMapTexts.erase(it);
            break; // 找到并删除后即可退出循环
        }
    }
}

std::vector<MapText*> Map::GetAllMapTexts()
{
    // 加锁，确保线程安全
    std::unique_lock<std::mutex> lock(mMutexMapTexts);
    
    // 返回容器的副本
    return mspMapTexts;
}

/*
std::vector<MapText*> Map::GetAllMapTexts(const TextStatus &state)
{
    std::unique_lock<std::mutex> lock(mMutexMapTexts);
    std::vector<MapText*> vMapTextObjsout;
    for(MapText* pMT : mspMapTexts)
    {
        if(pMT && !pMT->isBad() && pMT->mState == state)
            vMapTextObjsout.push_back(pMT);
    }
    return vMapTextObjsout;
}*/


/**
 * @brief 根据当前普通帧，获取时间上最近的两个关键帧 (std::set)
 * @param F 当前普通帧
 * @return 包含最多两个KeyFrame指针的vector，[0]是最新的, [1]是次新的
 */
std::vector<KeyFrame*> Map::GetNeighborKF(const Frame &F)
{
    std::unique_lock<std::mutex> lock(mMutexMap); // 加锁以安全访问mspKeyFrames
    
    KeyFrame* pClosestKF = nullptr;
    KeyFrame* pSecondClosestKF = nullptr;
    int minDiff = INT_MAX;
    int secondMinDiff = INT_MAX;
    long unsigned int FId = F.mnId;

    // 遍历set中的所有关键帧
    for (KeyFrame* pKF : mspKeyFrames)
    {
        if (!pKF || pKF->isBad()) continue;

        long unsigned int KFId = pKF->mnFrameId;
        int IdDiff = FId - KFId;
        
        if (IdDiff < 0) continue; // 只考虑在当前帧之前的关键帧

        // 核心逻辑：同时更新最近和次近的关键帧
        if (IdDiff < minDiff)
        {
            // 如果找到了一个更近的，那么之前的“最近”就变成了“次近”
            secondMinDiff = minDiff;
            pSecondClosestKF = pClosestKF;
            
            // 更新“最近”
            minDiff = IdDiff;
            pClosestKF = pKF;
        }
        else if (IdDiff < secondMinDiff)
        {
            // 如果只是比“次近”的更近，那就只更新“次近”
            secondMinDiff = IdDiff;
            pSecondClosestKF = pKF;
        }
    }
    
    std::vector<KeyFrame*> vResult;
    // 将找到的结果添加到返回向量中
    if (pClosestKF)
        vResult.push_back(pClosestKF);
    if (pSecondClosestKF)
        vResult.push_back(pSecondClosestKF);
    
    return vResult;
}

} //namespace ORB_SLAM
