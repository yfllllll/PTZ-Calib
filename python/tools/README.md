# Control Point Picker Tool

交互式3D模型控制点标注工具，用于求解世界坐标系与局部坐标系之间的相似变换参数。

## 功能特性

- ✅ 支持倾斜摄影模型（OSGB/OBJ/PLY格式）
- ✅ 交互式3D可视化点选
- ✅ 自动求解7参数相似变换（旋转R、平移t、缩放s）
- ✅ 残差分析和精度评估
- ✅ 导出控制点和变换参数

## 安装依赖

```bash
cd python
pip install open3d scipy numpy
```

## 使用方法

### 1. 准备3D模型

如果你的模型是OSGB格式，需要先转换成OBJ或PLY：

**方法A - CloudCompare（推荐）**:
```bash
# 1. 打开CloudCompare
# 2. File > Open > 选择你的.osgb文件
# 3. File > Save > 选择OBJ或PLY格式
```

**方法B - osgconv (OpenSceneGraph)**:
```bash
osgconv input.osgb output.obj
# 或
osgconv input.osgb output.ply
```

**方法C - FME Desktop**:
```
OSGB Reader -> OBJ/PLY Writer
```

### 2. 运行工具

```bash
cd python/tools

# 使用OBJ/PLY
python control_point_picker.py --model /path/to/model.obj

# 如果OSGB已转换，也可以直接指定（会自动查找同名OBJ/PLY）
python control_point_picker.py --model /path/to/model.osgb
```

### 3. 操作流程

1. **点选控制点**: 在3D窗口中用鼠标左键点击模型表面
2. **输入世界坐标**: 在终端输入对应的世界坐标（格式：`x y z`）
3. **保存控制点**: 按 `n` 键保存当前点并开始下一个
4. **求解变换**: 标注≥3个点后，按 `s` 键求解变换
5. **导出结果**: 按 `e` 键导出控制点和变换参数
6. **退出**: 按 `q` 键退出

### 4. 查看结果

运行后会在当前目录生成 `control_points_output/` 文件夹：

```
control_points_output/
├── control_points.json      # 控制点坐标对
└── transformation.json      # 变换参数
```

## 输出格式

### control_points.json
```json
{
  "num_points": 4,
  "points": [
    {
      "id": 1,
      "local": [100.5, 200.3, 50.2],
      "world": [500123.45, 3456789.12, 125.67]
    },
    ...
  ]
}
```

### transformation.json
```json
{
  "type": "similarity_7dof",
  "formula": "world = s * R * local + t",
  "scale": 1.0234,
  "rotation_matrix": [[...], [...], [...]],
  "rotation_euler_zyx_deg": [1.23, -0.45, 89.67],
  "rotation_quaternion_xyzw": [0.1, 0.2, 0.3, 0.9],
  "translation": [500000.0, 3456000.0, 100.0]
}
```

## 相似变换说明

工具求解的是7自由度相似变换：

```
world_coords = s * R * local_coords + t
```

其中：
- **s (scale)**: 尺度缩放因子
- **R (rotation)**: 3×3旋转矩阵
- **t (translation)**: 3×1平移向量

### 求解方法

使用**Procrustes分析** + **SVD分解**：

1. 对控制点中心化
2. 计算尺度比 s = ||world_centered|| / ||local_centered||
3. 归一化后通过SVD求解最优旋转矩阵R
4. 从中心点反推平移向量t

### 精度评估

工具会自动计算：
- 每个控制点的残差（residual）
- 均方根误差 RMSE
- 平均误差 Mean
- 最大误差 Max

## 应用场景

### 场景1: PTZ相机标定

将PTZ相机的局部坐标系对齐到倾斜摄影模型的世界坐标系：

1. 在倾斜模型上标注至少3个特征点（如建筑角点、地标）
2. 记录这些点在PTZ相机坐标系中的坐标
3. 求解变换参数
4. 用变换参数将PTZ相机位姿转到世界坐标系

### 场景2: 多源数据融合

将无人机倾斜模型（局部坐标）转换到国家坐标系（如CGCS2000）：

1. 在模型上标注RTK测量的控制点
2. 输入控制点的国家坐标
3. 求解变换参数
4. 批量转换模型中所有点的坐标

### 场景3: 跨项目坐标统一

将多个倾斜摄影项目统一到同一坐标系：

1. 在重叠区域标注公共控制点
2. 一个项目作为基准（世界坐标）
3. 其他项目作为待配准（局部坐标）
4. 求解每个项目的变换参数

## 技巧提示

1. **控制点数量**: 建议标注4-6个点，分布均匀
2. **点位选择**: 选择模型上清晰可辨的特征点（建筑角点、道路交叉口等）
3. **精度检查**: RMSE应小于模型精度（通常<0.5米）
4. **坐标一致**: 确保世界坐标和局部坐标的单位一致（都是米）

## 键盘快捷键

| 键 | 功能 |
|---|------|
| **Left Click** | 在3D模型上点选控制点 |
| **n** | 保存当前控制点，开始下一个 |
| **s** | 求解相似变换（需≥3个点） |
| **e** | 导出控制点和变换参数 |
| **q** | 退出程序 |

## 常见问题

### Q1: 打开OSGB文件报错？
A: OSGB需要先转换成OBJ或PLY。用CloudCompare或osgconv转换后再运行。

### Q2: 残差过大怎么办？
A: 检查：
- 世界坐标输入是否正确
- 控制点是否点选在正确位置
- 单位是否一致（都用米或都用度）
- 删除异常点重新标注

### Q3: 如何批量应用变换？
A: 读取transformation.json，用NumPy/SciPy应用变换：
```python
import numpy as np
import json

with open('transformation.json') as f:
    t = json.load(f)

R = np.array(t['rotation_matrix'])
trans = np.array(t['translation'])
s = t['scale']

# 应用变换
world = s * (R @ local) + trans
```

## 许可证

MIT License - 与PTZ-Calib项目保持一致
