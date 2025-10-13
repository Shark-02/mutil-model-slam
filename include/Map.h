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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Ellipsoid.h"
#include <set>
#include <unordered_map>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class MapObject;
class Graph;
class Object;
class MapText;
class Frame;
class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<KeyFrame*> GetNeighborKF(const Frame &F);
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<Object*> GetAllObjects();
    std::vector<MapText*> GetAllMapTexts();
    //std::vector<MapText*> GetAllMapTexts(const TextStatus &state);
    std::vector<MapPoint*> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    std::vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    // const std::unordered_map<unsigned int, Eigen::Matrix<double, 3, Eigen::Dynamic>>& GetAllMapObjectsPoints() {
    //     return ellipsoids_points_;

    void AddObject(Object *obj);

    void AddMapText(MapText* pMT);
    void EraseMapText(MapText* pMT);
    


    //MapObject* GetObjWithTrId(int tr_id);

    size_t GetNumberMapObjects() const {
        return mspObjects.size();
    }
    size_t GetNumberPoints() const {
        return mspMapPoints.size();
    }
    //ADDED TOBE DELETED
    std::set<KeyFrame*> getKeyFrames() const {
        return mspKeyFrames;
    }

    Graph *graph_3d;


protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;
    std::set<Object*> mspObjects;
    std::vector<MapText*> mspMapTexts;
    std::mutex mMutexMapTexts;

    std::vector<MapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;

    // std::unordered_map<unsigned int, Eigen::Matrix<double, 3, Eigen::Dynamic>> ellipsoids_points_;
};

} //namespace ORB_SLAM

#endif // MAP_H
