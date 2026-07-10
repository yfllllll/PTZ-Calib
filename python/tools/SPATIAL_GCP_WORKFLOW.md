# 空间控制点自动准备工作流

这个工作流解决的是：只有一批 PTZ 图像时，怎样尽量少做重复标注，批量生成 PTZ-Calib 需要的 `annotation.json`。

核心思想是先在一个公共视野上标注空间控制点，再自动传播到每张 PTZ 图像：

```text
全景图/单张图像上的像素点
        +
本地正射影像/DSM、在线地图、倾斜三维模型或手动 XYZ
        |
        v
空间控制点项目 spatial_gcp.json
        |
        +-- 单张图像模式：直接导出 annotation.json
        |
        +-- 全景图模式：vismatch 匹配每张图像到全景图
                         image -> panorama 单应矩阵
                         全景 GCP 批量投影回每张图像
                         导出 annotation.json
```

## 1. 全景图标注路线

适合一组图像共同覆盖一个较大场景的情况。你只需要在全景图上标一批点，然后由程序自动投影到每张原始图像。

### 1.1 准备全景图

可以先用 `build_panorama_vismatch.py` 生成一个简易全景图，也可以用你的 stitching 仓库生成更完整的全景图。那个仓库里 `test.py` 的 `map_point_to_panorama` 体现了一个更严格的思路：保存 stitching 的相机参数、warper 和角点信息后，把原图像素映射到全景坐标。

当前这里先实现一个更通用的版本：不依赖 stitching 内部状态，而是用 `vismatch` 直接匹配每张 PTZ 图像和全景图，再用 RANSAC 估计 `image -> panorama` 单应矩阵。这样全景图来源更自由，后续也可以替换成 stitching 输出的精确 warper 参数。

按图像顺序生成全景图：

```bash
cd python/tools
python build_panorama_vismatch.py \
  --images /data/ptz_images \
  --output /data/pano/panorama.jpg \
  --matcher superpoint-lightglue \
  --resize 1024 \
  --min_matches 20 \
  --min_inliers 12
```

这个工具采用相邻图像链式拼接：第 `i` 张和第 `i+1` 张先匹配并估计单应矩阵，再统一变换到中心图像坐标系。它适合图像已经按 PTZ 扫描顺序排列、相邻视野有足够重叠的情况。输出旁边会生成 `panorama.report.json`，里面包含每一对相邻图像的匹配数、RANSAC 内点数和 RMSE。

### 1.2 在全景图上标注空间控制点

只手动输入 XYZ：

```bash
python spatial_gcp_annotator.py \
  --target /data/pano/panorama.jpg \
  --target_type panorama \
  --output /data/gcp/pano_gcp.json \
  --coordinate_system EPSG:32650
```

使用本地正射影像和默认高程：

```bash
python spatial_gcp_annotator.py \
  --target /data/pano/panorama.jpg \
  --target_type panorama \
  --output /data/gcp/pano_gcp.json \
  --ortho /data/map/ortho.png \
  --ortho_extent "500000,3456000,501000,3457000" \
  --z 35.0 \
  --coordinate_system EPSG:32650
```

使用正射影像 + DSM：

```bash
python spatial_gcp_annotator.py \
  --target /data/pano/panorama.jpg \
  --target_type panorama \
  --output /data/gcp/pano_gcp.json \
  --ortho /data/map/ortho.png \
  --ortho_extent "500000,3456000,501000,3457000" \
  --dsm /data/map/dsm.npy \
  --dsm_extent "500000,3456000,501000,3457000" \
  --coordinate_system EPSG:32650
```

使用倾斜三维模型点选 XYZ：

```bash
python spatial_gcp_annotator.py \
  --target /data/pano/panorama.jpg \
  --target_type panorama \
  --output /data/gcp/pano_gcp.json \
  --model /data/model/scene.obj \
  --coordinate_system local_model
```

使用在线底图：

```bash
python spatial_gcp_annotator.py \
  --target /data/pano/panorama.jpg \
  --target_type panorama \
  --output /data/gcp/pano_gcp.json \
  --map_source google_satellite \
  --map_bbox "116.3900,39.9000,116.4020,39.9100" \
  --map_zoom 18 \
  --z 35.0 \
  --coordinate_system EPSG:3857
```

也可以使用自定义 XYZ 瓦片模板：

```bash
python spatial_gcp_annotator.py \
  --target /data/pano/panorama.jpg \
  --target_type panorama \
  --output /data/gcp/pano_gcp.json \
  --map_template "https://example.com/tiles/{z}/{x}/{y}.png?key={key}" \
  --map_key YOUR_KEY \
  --map_bbox "116.3900,39.9000,116.4020,39.9100" \
  --map_zoom 18
```

内置 `--map_source` 包括：

| 来源 | 说明 |
|---|---|
| `osm` | OpenStreetMap 标准瓦片 |
| `google_satellite` | Google 卫星瓦片 |
| `google_roadmap` | Google 道路瓦片 |
| `amap_satellite` | 高德卫星瓦片 |
| `amap_roadmap` | 高德道路瓦片 |

在线地图点击得到的是 WebMercator 米制坐标。高德底图存在 GCJ-02 偏移问题，如果要和 WGS84/UTM 或本地测绘成果严格对齐，需要先做坐标转换或改用本地正射影像。

交互键位：

| 操作 | 作用 |
|---|---|
| 左键点击全景图/单图 | 选择目标像素 |
| 左键点击正射影像 | 给当前目标点赋世界坐标 |
| `m` | 手动输入 `X Y Z` |
| `3` | 打开 Open3D 模型点选 |
| `u` | 撤销最后一个点 |
| `s` | 保存项目 |
| `q` | 保存并退出 |

输出的 `pano_gcp.json` 是中间项目文件，记录的是：

```json
{
  "target_type": "panorama",
  "target_image": "/data/pano/panorama.jpg",
  "points": [
    {
      "target_pixel": [1234.0, 567.0],
      "world": [500123.45, 3456789.12, 35.0],
      "source": "ortho_dsm"
    }
  ]
}
```

### 1.3 匹配每张 PTZ 图像到全景图

```bash
python build_pano_homographies_vismatch.py \
  --images /data/ptz_images \
  --panorama /data/pano/panorama.jpg \
  --output /data/gcp/pano_homographies.json \
  --matcher superpoint-lightglue \
  --resize 1024 \
  --min_matches 30 \
  --min_inliers 12
```

输出文件会包含每张图像的：

- `H_image_to_pano`：图像像素到全景像素的 3x3 单应矩阵
- `num_matches`：vismatch 匹配点数
- `num_inliers`：RANSAC 内点数
- `rmse`：内点重投影均方根误差

如果某些图像和全景重叠太少、纹理太弱或匹配质量差，会被写进 `skipped` 字段。

### 1.4 批量生成 PTZ-Calib 的 annotation.json

```bash
python project_pano_gcp.py \
  --pano_gcp /data/gcp/pano_gcp.json \
  --mappings /data/gcp/pano_homographies.json \
  --images /data/ptz_images \
  --output /data/gcp/annotation.json \
  --min_points 4
```

生成的 `/data/gcp/annotation.json` 可以直接给 `run_ptz_ba.py` 使用：

```bash
cd ../
python run_ptz_ba.py \
  --images /data/ptz_images \
  --annotation /data/gcp/annotation.json \
  --output /data/output \
  --dist
```

旁边还会生成 `annotation.summary.json`，用于检查每张图像最终投影到了几个控制点。建议每张图像至少 4 个点，且点位尽量分散。

## 2. 单张图像标注路线

如果只有少量关键图像，或者某些图像没有被全景覆盖，可以直接在单张图像上标注。

```bash
python spatial_gcp_annotator.py \
  --target /data/ptz_images/000001.jpg \
  --target_type image \
  --output /data/gcp/000001_spatial_gcp.json \
  --annotation_output /data/gcp/annotation_single.json \
  --images_dir /data/ptz_images \
  --image_name 000001.jpg \
  --ortho /data/map/ortho.png \
  --ortho_extent "500000,3456000,501000,3457000" \
  --dsm /data/map/dsm.npy
```

在工具里按 `e` 可以直接导出 PTZ-Calib 的 `annotation.json`。

## 3. 关于在线地图、DSM 和坐标系

目前代码支持本地正射影像、`.npy` DSM，以及在线 XYZ 瓦片地图。在线地图使用时要注意三类额外问题：

- Google/OSM 常用 WGS84/WebMercator，高德使用 GCJ-02，和工程坐标系之间必须明确转换。
- 在线地图瓦片需要 API key、网络权限、缓存策略和服务条款约束。
- PTZ-Calib 的 BA 只关心一致的世界坐标，不强制必须是经纬度或 UTM；本地工程坐标、ENU 或模型坐标都可以，但同一批数据必须一致。

实际使用时推荐优先准备本地正射影像和 DSM，并在 `--coordinate_system` 里记录坐标系，例如 `EPSG:32650`、`local_enu` 或 `model_local`。

## 4. 质量检查建议

- 全景图上至少标注 8 到 12 个清晰点，覆盖画面左右、上下和不同深度。
- `pano_homographies.json` 中 `num_inliers` 太少或 `rmse` 很大的图像不要直接信任。
- `annotation.summary.json` 中少于 4 个点的图像不会导出，或应重新补点。
- 控制点的 3D 坐标来源要一致，不要混用未对齐的模型坐标和地图坐标。
- 最终仍以 `run_ptz_ba.py` 的 2D-3D 重投影误差判断标注质量。
