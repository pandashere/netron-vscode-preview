# ViT-UNet 3200 节点布局对比报告

## 样本

- 模型文件：`/home/zhaochen/net_vsix_min_repo/testdata/generated/vit-unet-3200.onnx`
- ONNX 节点数：`3200`
- Netron Workbench 读取节点数：`3200`
- 边数：`3835`
- 结构：4 encoder stages + bottleneck + 4 decoder stages，带 U-Net skip merge，block 内部为简化 ViT attention/MLP 模式

## 布局差异

- Python 版 ranker：`longest-path`
- JS / Netron ranker：`longest-path`
- rank mismatch count：`0`
- max abs rank delta：`0`
- Python layer count：`2568`
- JS layer count：`2568`
- first mismatch indices：`[]`

## 性能

### 默认 Netron 策略（>3000 节点自动走 `longest-path`）

- Python median：`119.68 ms`
- JS median：`953.54 ms`
- JS / Python 比值：`7.97x`

### 强制 `network-simplex`

- Python median：`495.32 ms`
- JS median：`1096.74 ms`
- JS / Python 比值：`2.21x`

## 结论

- 对这个 3200 节点 ViT-UNet 样本，Python 版层号与 Netron JS 版 **完全一致**。
- 因为节点数大于 3000，Netron 默认不会用 `network-simplex`，而是切到 `longest-path`；Python 版已复现这一策略。
- Python 版只求“高低层 / rank”，而 JS 版执行的是完整 Dagre 布局（还包含横向排序与坐标分配），因此 JS 通常更慢，这个差距在默认策略下尤为明显。
- 如果你的目标只是得到 Netron 风格的节点高低关系，Python 版已经足够对齐，并且更适合离线批量处理。
