# PTZ-Calib 辅助工具

## 工具列表

| 工具 | 用途 |
|------|------|
| **gcp_annotator.py** ⭐ | **地面控制点标注器**（2D图像 + 3D模型）→ 生成 `annotation.json` |
| **spatial_gcp_annotator.py** ⭐ | **空间控制点标注器**（全景图/单图 + 正射影像/DSM/3D模型/手动XYZ） |
| **build_panorama_opencv.py** ⭐ | 基于 OpenCV detailed stitching 生成全景图（相机估计 + BA + warper + seam + blender） |
| build_panorama_vismatch.py | 用 vismatch 按图像顺序生成简易全景图（快速诊断，不作为首选全景方案） |
| build_pano_homographies_vismatch.py | 用 vismatch 估计每张PTZ图像到全景图的单应矩阵 |
| project_pano_gcp.py | 将全景图控制点批量投影到PTZ图像并生成 `annotation.json` |
| control_point_picker.py | 纯3D模型点选（求解坐标系相似变换） |
| apply_transformation.py | 批量应用相似变换到点/相机 |

---

## ⭐ 全景图/单图空间控制点工作流

如果你不想给每张 PTZ 图像逐个标注控制点，可以先在全景图上标注空间控制点，再通过 vismatch 自动映射回每张原图。

完整教程见：[SPATIAL_GCP_WORKFLOW.md](SPATIAL_GCP_WORKFLOW.md)

最常用的全景路线：

```bash
# 0. 如果还没有全景图，先用 OpenCV detailed stitching 生成
python build_panorama_opencv.py \
    --images /data/ptz_images \
    --output /data/pano/panorama.jpg \
    --features sift \
    --warp spherical \
    --seam gc_color \
    --blend multiband

# 1. 在全景图上标注空间控制点
python spatial_gcp_annotator.py \
    --target /data/pano/panorama.jpg \
    --target_type panorama \
    --output /data/gcp/pano_gcp.json \
    --ortho /data/map/ortho.png \
    --ortho_extent "500000,3456000,501000,3457000" \
    --dsm /data/map/dsm.npy

# 2. 匹配每张PTZ图像到全景图
python build_pano_homographies_vismatch.py \
    --images /data/ptz_images \
    --panorama /data/pano/panorama.jpg \
    --output /data/gcp/pano_homographies.json

# 3. 批量生成PTZ-Calib annotation.json
python project_pano_gcp.py \
    --pano_gcp /data/gcp/pano_gcp.json \
    --mappings /data/gcp/pano_homographies.json \
    --images /data/ptz_images \
    --output /data/gcp/annotation.json
```

---

## ⭐ gcp_annotator.py — 地面控制点标注器（主要工具）

这是**PTZ-Calib的核心配套工具**。它为`run_ptz_ba`生成必需的`annotation.json`文件。

### 原理

PTZ-Calib需要地面控制点（GCP）作为 **2D-3D约束** 来求解相机外参：
- **2D**: 图像上的像素坐标（用户在图像上点选）
- **3D**: 对应的世界坐标（用户在倾斜摄影模型上点选）

### 安装

```bash
pip install open3d opencv-python numpy
```

### 使用方法

```bash
# 1. 准备3D模型
#    如果是OSGB，先用CloudCompare/osgconv转换：
osgconv model.osgb model.obj

# 2. 运行标注工具
python gcp_annotator.py \
    --images /path/to/ptz_images/ \
    --model /path/to/model.obj \
    --output annotation.json

# 3. 生成的annotation.json可直接用于PTZ-Calib
python ../run_ptz_ba.py --annotation annotation.json --images /path/to/ptz_images/ ...
```

### 交互流程

工具会打开两个窗口协同工作：

**图像窗口（左）**：显示PTZ相机图像
**3D模型窗口（右）**：显示倾斜摄影三维模型

标注流程：

```
┌────────────────────────────────────────────────────────────┐
│  Step 1: 在图像窗口左键点击某个特征点（如建筑角点）             │
│          → 图像上出现橙色十字标记                              │
├────────────────────────────────────────────────────────────┤
│  Step 2: 按 '3' 键打开3D模型窗口                              │
├────────────────────────────────────────────────────────────┤
│  Step 3: 在3D窗口找到同一特征点，Shift+左键点选               │
│          → 按 'Q' 关闭3D窗口                                  │
├────────────────────────────────────────────────────────────┤
│  Step 4: 2D-3D点对自动保存，图像上出现绿色圆圈标记             │
├────────────────────────────────────────────────────────────┤
│  Step 5: 继续标注下一个点（每张图建议≥4个点）                  │
├────────────────────────────────────────────────────────────┤
│  Step 6: 按 'n' 切换到下一张图像，重复Step 1-5                │
├────────────────────────────────────────────────────────────┤
│  Step 7: 全部标注完后按 's' 保存，或 'q' 保存并退出            │
└────────────────────────────────────────────────────────────┘
```

### 键盘快捷键

| 键 | 功能 |
|---|------|
| **左键单击（图像）** | 选择2D控制点 |
| **Shift+左键（3D）** | 选择3D控制点 |
| **3** | 打开3D窗口进行3D点选 |
| **n** | 下一张图像 |
| **p** | 上一张图像 |
| **u** | 撤销当前图像的最后一个点对 |
| **s** | 保存annotation.json |
| **h** | 显示帮助 |
| **q** | 保存并退出 |

### 命令行参数

```bash
python gcp_annotator.py \
    --images /path/to/images/ \      # 必需：PTZ图像目录
    --model /path/to/model.obj \     # 必需：3D模型路径（OBJ/PLY）
    --output annotation.json \       # 输出JSON路径
    --K "1500,0,960,0,1500,540,0,0,1" \  # 可选：初始K矩阵
    --load previous_annotation.json  # 可选：加载已有标注继续编辑
```

### 输出格式（PTZ-Calib标准）

生成的`annotation.json`符合C++ `ReadFromJson`函数的格式：

```json
{
  "cameras": {
    "image_001": {
      "K": [1000, 0, 960, 0, 1000, 540, 0, 0, 1],
      "R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
      "t": [0, 0, 0],
      "dist": [0, 0, 0, 0, 0],
      "res": [1920, 1080],
      "marker": {
        "pix": [
          [0.5234, 0.6789],
          [0.1234, 0.4567]
        ],
        "pos": [
          [500123.45, 3456789.12, 125.67],
          [500145.32, 3456812.45, 130.22]
        ]
      }
    },
    "image_002": { ... }
  }
}
```

字段说明：
- `K, R, t, dist`: 相机参数（初始估计值，会被BA优化）
- `res`: 图像分辨率 [width, height]
- `marker.pix`: 归一化像素坐标 `[x/width, y/height]`
- `marker.pos`: 对应的3D世界坐标 `[x, y, z]`

### 标注建议

- **每张图像至少标注4个点**（PnP最少4个点即可求解）
- **点位选择原则**：
  - 优先选择**清晰锐利**的角点（建筑楼角、地面标记、道路交叉口）
  - **分布均匀**（避免集中在同一区域）
  - **深度多样**（近点+远点组合）
- **多相机共享控制点**：不同PTZ图像最好包含**相同的3D控制点**，这样能形成更强的约束

### 常见问题

**Q1: OSGB打不开？**  
A: OSGB需要先转成OBJ或PLY。用 `osgconv input.osgb output.obj` 或 CloudCompare。

**Q2: 3D窗口无法点选？**  
A: 记得用 **Shift + 左键**（Open3D的picking模式），普通左键只旋转视角。

**Q3: 想中途保存进度？**  
A: 随时按 `s` 键，或退出前按 `q` 自动保存。用 `--load` 参数可以载入继续标注。

**Q4: 如何验证标注质量？**  
A: 用生成的annotation.json运行PTZ-Calib，看重投影误差：
```
error_2d3d < 5 pixel  → 好
error_2d3d < 15 pixel → 可用
error_2d3d > 30 pixel → 需要重新标注
```

---

## control_point_picker.py — 3D模型点选工具（辅助）

用于**仅在3D模型上**标注控制点，并求解**世界坐标系↔局部坐标系**的相似变换（7-DOF: R, t, s）。

适用场景：
- 将SfM得到的局部重建对齐到实际世界坐标系
- 多个坐标系之间的对齐

```bash
python control_point_picker.py --model model.obj
```

详见文件内注释。

---

## apply_transformation.py — 批量变换工具

将`control_point_picker.py`求解的变换应用到其他数据：

```bash
# 变换单个点
python apply_transformation.py \
    --transform transformation.json \
    --point 100.5 200.3 50.2

# 变换整个相机文件
python apply_transformation.py \
    --transform transformation.json \
    --cameras cameras_local.json \
    --output cameras_world.json
```

---

## 完整工作流示例

```bash
# 1. 准备PTZ相机图像
ls /data/ptz_cam/*.jpg
#   pan_0_tilt_0.jpg
#   pan_30_tilt_0.jpg
#   pan_60_tilt_-10.jpg
#   ...

# 2. 准备3D模型（从无人机倾斜摄影）
osgconv /data/model/Data.osgb /data/model/scene.obj

# 3. 标注地面控制点
cd python/tools
python gcp_annotator.py \
    --images /data/ptz_cam/ \
    --model /data/model/scene.obj \
    --output /data/annotation.json

# 4. 运行PTZ-Calib BA优化
cd ..
python run_ptz_ba.py \
    --images /data/ptz_cam/ \
    --annotation /data/annotation.json \
    --output /data/output/ \
    --dist

# 5. 查看结果
cat /data/output/registered_cameras.json
```
