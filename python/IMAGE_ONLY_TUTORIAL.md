# Image-only PTZ-Calib Tutorial

这份教程说明在只有图像目录、没有预先提取好的特征/匹配文件时，如何用
`gmberton/vismatch` 自动准备 PTZ-Calib 输入，并运行 Python 版 PTZ-IBA。

## 1. 安装依赖

先安装 Python 版 PTZ-Calib 的基础依赖：

```bash
pip install -r python/requirements.txt
```

再安装图像匹配依赖：

```bash
pip install git+https://github.com/gmberton/vismatch.git
```

`vismatch` 首次运行某些 matcher 时会下载模型权重，例如 LightGlue/SuperPoint
权重，并缓存到本机的 torch cache 目录。

## 2. 从图片生成特征和匹配

假设图片放在：

```text
data/my_scene/images
```

运行：

```bash
python python/prepare_data_vismatch.py \
  --images data/my_scene/images \
  --output data/my_scene/features \
  --matcher superpoint-lightglue \
  --device auto \
  --resize 1024 \
  --pairing window \
  --window 5 \
  --min_matches 15
```

这一步会生成：

```text
data/my_scene/features/
  image_001.jpg.txt
  image_002.jpg.txt
  ...
  pairs_matches.txt
  prepare_summary.json
```

其中每个 `*.jpg.txt` 是对应图像的关键点/描述子文件，
`pairs_matches.txt` 是图像对之间的匹配关系。

## 3. 运行无 annotation 的 PTZ-IBA

如果你真的只有图像，没有 2D-3D 控制点 annotation，可以先只跑增量 PTZ-IBA：

```bash
PYTHONPATH=python python python/run_ptz_ba.py \
  --images data/my_scene/images \
  --features data/my_scene/features \
  --output outputs/my_scene \
  --max_iter 200
```

这会输出：

```text
outputs/my_scene/<scene_name>.json
```

注意：不提供 annotation 时，流程不会执行 georeferencing，输出相机是在内部重建坐标系下的结果。

## 4. 如果有 2D-3D annotation

如果你有 PTZ-Calib 格式的 annotation JSON，可以把它传给 `run_ptz_ba.py`：

```bash
PYTHONPATH=python python python/run_ptz_ba.py \
  --images data/my_scene/images \
  --features data/my_scene/features \
  --annotation data/my_scene/annotation.json \
  --output outputs/my_scene \
  --max_iter 200
```

这会在 PTZ-IBA 之后继续做 georeferencing，并输出带世界坐标对齐的相机参数。

## 5. `window` 配对是什么意思

`--pairing window --window 5` 表示每张图只和它后面最多 5 张图匹配。

例如图像顺序为：

```text
1, 2, 3, 4, 5, 6, 7, ...
```

`window=5` 会匹配：

```text
1-2, 1-3, 1-4, 1-5, 1-6
2-3, 2-4, 2-5, 2-6, 2-7
...
```

这样比全量两两匹配快很多，但可能漏掉关键匹配边。如果图像数量不大，或者默认
`window` 结果注册不完整，可以尝试全量匹配：

```bash
python python/prepare_data_vismatch.py \
  --images data/my_scene/images \
  --output data/my_scene/features_all_pairs \
  --matcher superpoint-lightglue \
  --device auto \
  --resize 1024 \
  --pairing all \
  --min_matches 15
```

全量匹配的 pair 数是 `N * (N - 1) / 2`。例如 30 张图是 435 对，会明显更慢，
但连通性通常更强。

## 6. 图像顺序

`prepare_data_vismatch.py` 会使用自然数字排序读取图像文件名，例如：

```text
scene_01-1.jpg
scene_01-7.jpg
scene_01-13.jpg
...
scene_01-103.jpg
```

这比普通字典序更适合按帧号或角度编号命名的数据。如果你的文件名没有包含拍摄顺序，
建议使用 `--pairing all`，或者自己准备 pairs 文件并使用 `--pairing file`。

## 7. 当前验证结论

在 synthetic `scene_01` 上做过 image-only 测试：

- 自动特征/匹配生成可以正常工作。
- `run_ptz_ba.py` 可以读取自动生成的数据并输出结果。
- 默认 `window=5` 只注册了部分图像，说明仅靠当前默认配对策略还不一定能全量重建。

如果你希望更稳地从只有图像跑完整流程，优先尝试：

- 增大 `--window`，例如 `--window 10`。
- 使用 `--pairing all` 做全量匹配。
- 降低或调整 `--min_matches` 后检查 `prepare_summary.json`。
- 如果有 2D-3D 控制点，提供 `--annotation` 做 georeferencing。

