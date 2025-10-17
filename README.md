# MMS-SLAM: 一个融合多模态语义的视觉SLAM系统



本项目是一个基于 ORB-SLAM2 框架的同步定位与建图（SLAM）系统。系统的核心创新在于融合了多种信息模态，融合了**特征点（Point）**、**物体（Object）** 与**文本（Text）**三种视觉要素，旨在在不同语义环境下实现更鲁棒、更可解释的定位与建图。

## 项目结构

```
mms-slam/
│
├── CMakeLists.txt              # 主 CMake 配置文件，用于构建整个项目
├── build.sh                    # 编译项目的Shell脚本
│
├── Vocabulary/
│   └── ORBvoc.txt.tar.gz         # ORB-SLAM2所需的词袋模型文件
│
├── Cameras/                      # 存放不同相机参数的配置文件
│   ├── TUM1.yaml                 # TUM 数据集相机参数
│   └── ...                       # 其他相机参数文件
│
├── Examples/
│   ├── rgbd_tum_with_ellipse.cc  # TUM RGB-D数据集的运行示例，集成了文本和物体检测
│   └── ...                       # 其他示例程序
│
├── include/                      # 头文件目录
│   ├── System.h                  # 系统主类，负责管理所有线程
│   ├── Tracking.h                # 前端跟踪线程接口，处理图像帧并估计位姿
│   ├── LocalMapping.h            # 局部建图线程接口，负责维护和优化局部地图
│   ├── LoopClosing.h             # 回环检测线程接口，负责检测和修正累积误差
│   ├── Map.h                     # 地图数据结构定义
│   ├── KeyFrame.h                # 关键帧数据结构定义
│   ├── MapPoint.h                # 地图点数据结构定义
│   ├── Object.h                  # 物体地标定义，表示为三维椭球体
│   ├── MapText.h                 # 文本地标定义，表示为三维平面
│   ├── ORBextractor.h            # ORB特征提取器
│   ├── ORBmatcher.h              # ORB特征匹配器
│   ├── ObjectMatcher.h           # 物体匹配器，用于数据关联
│   ├── TextMatcher.h             # 文本匹配器，用于文本地标的数据关联
│   ├── Viewer.h                  # 2D/3D可视化接口
│   ├── ARViewer.h                # AR视图实现
│   ├── Osmap.h                   # 地图保存与加载模块接口
│   ├── Optimizer.h               # BA优化相关接口
│   └── ...                       # 其他辅助类头文件
│
├── src/                        # 源文件目录
│   ├── System.cc                 # 系统主类实现
│   ├── Tracking.cc               # 前端跟踪主逻辑，融合了特征点、物体和文本的跟踪
│   ├── LocalMapping.cc           # 局部地图维护逻辑
│   ├── LoopClosing.cc            # 回环检测与修正逻辑
│   ├── Object.cc                 # 物体地标的实现，包括椭球体优化
│   ├── MapText.cc                # 文本地标的实现，包括平面拟合与状态管理
│   ├── ObjectMatcher.cc          # 物体匹配算法实现
│   ├── TextMatcher.cc            # 文本匹配算法实现
│   ├── MapDrawer.cc              # 地图绘制实现，支持物体和文本的可视化
│   ├── ARViewer.cc               # AR视图的渲染逻辑
│   ├── Osmap.cpp                 # 地图序列化与反序列化实现
│   ├── Optimizer.cc              # BA优化实现
│   └── rgbd_tum_with_ellipse.cc  # 示例程序的具体实现
│
├── Thirdparty/                 # 第三方库
│   ├── DBoW2/                    # 词袋模型库，用于重定位和回环检测
│   ├── g2o/                      # 图优化库，用于BA优化
│   ├── json/                     # JSON库，用于处理检测结果文件
│   └── Osmap/                    # 地图序列化库
│
└── script/                       # 存放辅助脚本（根据您的描述添加）
    ├── text_extraction.py        # 文本检测与识别脚本
    ├── build_database.py         # 构建数据集脚本
    └── prepare_dataset.py        # 预处理数据脚本
```

## 当前主要功能实现

## 已实现的功能

### 特征点（Point）

使用 ORB 特征提取与匹配；

维持传统帧间姿态估计；

可与语义特征协同优化。

### 物体（Object）

基于 YOLOv8-seg 掩码拟合椭圆；

在 Tracking中，当地图中的物体椭球体被投影到当前帧时，ObjectMatcher 会计算投影椭圆与当前帧检测椭圆之间的 Wasserstein 距离，实现稳健的数据关联

在新的关键帧中，系统会根据深度信息和多视角几何约束，自动初始化新的 Object 地标。

通过 AddDetection() 接口，一个物体地标可以累积来自不同关键帧的观测，并通过 OptimizerObject 中的图优化方法，不断优化三维椭球体的参数。

### 文本（Text）

从 OCR 模块提取文字检测框；

通过平面参数 $(n, d)$ 建立文字平面；

在 Tracking 阶段完成匹配、初始化与状态更新；

维护语义质量评分 score_semantic；

MapText 拥有 TEXTIMMATURE（不成熟）和 TEXTGOOD（成熟）状态。一个地标在被稳定观测和跟踪多次后，会由不成熟状态转为成熟状态。

可通过 GetWorldBoundingBox()投影可视化文字平面。

## 预期目标

1. **多模态统一地图表示：**
    支持点（Point）+ 物体（Object）+ 文本（Text）三类特征共存与交叉约束。

2. **语义自适应建图策略：**

   文本密集场景：以文本平面为主要约束，提高语义一致性。

   物体密集场景：以物体几何为主，实现稳健定位。

   无语义场景：自动退化为传统点特征 SLAM。

3. **长期语义维护：**
    基于语义代价的地标更新、融合与剔除机制，使地图语义结构长期一致。

4. **全局语义约束优化：**
    将文字、物体平面等语义信息统一纳入 BA 优化图，实现全局一致性。

##  关键代码模块

| 模块类别                     | 文件                                    | 功能概要                                                     |
| ---------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| **传统特征点**               | `MapPoint.h / MapPoint.cc``Tracking.cc` | 传统 ORB 特征的提取、匹配与三角化，提供基础的几何约束与位姿估计；在语义稀疏场景下作为主要定位源。 |
| **物体语义（Object）**       | `Object.h / Object.cc`                  | 物体级地标定义（椭球模型），负责物体检测结果的几何拟合、三维重投影与观测融合；支持多帧观测下的物体持续跟踪。 |
| **文本语义（Text）**         | `MapText.h / MapText.cc`                | 文本平面地标定义与管理，支持语义置信度更新、地标融合与替换；在文字密集场景中提供高层平面约束。 |
| **前端跟踪（Tracking）**     | `Tracking.cc`                           | 统一管理多模态输入（点、物体、文本），进行特征匹配、姿态估计与地标初始化；包含语义代价计算与文本状态更新逻辑。 |
| **后端建图（LocalMapping）** | `LocalMapping.cc`                       | 处理关键帧插入、局部地图优化与语义更新；未来将实现文字与物体的地标融合（Replace/Fuse）与多模态局部 BA。 |
| **全局优化（Optimizer）**    | `Optimizer.cc`                          | 负责 Bundle Adjustment 与位姿图优化；未来扩展为联合优化点、椭球和平面三种语义约束。 |
| **地图与可视化**             | `Map.cc / MapDrawer.cc`                 | 管理所有类型地标的全局集合；在可视化中以不同颜色和形状区分点特征、物体和文本的状态与类型。 |
| **回环检测**                 | `LoopClosing.cc `                       | 当前仅支持基于点特征的几何回环；计划扩展语义辅助回环检测（文字字符串相似性 + 物体标签共现）。 |

## 致谢

本项目基于以下开源成果进行改进与拓展：

**VOOM: Robust Visual Object Odometry and Mapping using Hierarchical Landmarks** 

**TextSLAM: Visual SLAM with Semantic Planar Text Features**

**ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**

------

## 作者

**Author:** *Sun Yu*
**Affiliation:** Beijing University of Posts and Telecommunications (BUPT)
**Research Focus:** Semantic SLAM, Object Mapping, Text Understanding，Multimodal fusion
**Contact:** [[shark@bupt.edu.cn](mailto:your_email@example.com)]

