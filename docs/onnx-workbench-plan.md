# ONNX Workbench 需求摘要与实施计划

## 1. 需求摘要

本项目后续改造目标，按优先级和主题整理如下。

### 1.1 大 ONNX 模型加载

- 当前项目存在“无法加载过大的 ONNX 模型”的问题。
- 需要结合当前仓库相对原版 Netron 的改动，定位并修复根因。
- 重点怀疑点包括：
  - Node 扩展宿主与 Webview 之间的大对象传输限制
  - Webview 内存/主线程负载问题
  - 现有超时与串行布局策略

### 1.2 多窗口 / 多标签支持

- 需要支持在 VS Code 中同时打开两个 Netron 预览标签页。
- 两个标签页可分别打开不同模型。
- 两个标签页后续需要具备宿主统一编排的通信能力。
- Compare 场景中的跨窗口协作不能依赖面板点对点直连，应由扩展宿主统一管理状态。

### 1.3 ONNX Crop / 导出 / 推理

- 对 ONNX 子图裁剪（crop）后，需要支持直接导出为 ONNX。
- 需要支持对 ONNX crop 一键执行 inference。
- inference 输入支持两类来源：
  - 自动生成输入
  - 用户导入输入
- 需要特别注意用户防呆、状态一致性以及草稿/确认态区分。

### 1.4 双窗口 Compare

- 当两个 ONNX 模型分别在两个标签页中完成 crop 后，需要支持同时 inference 并对比差异。
- Compare 的核心定义已经明确：
  - 比较对象是“输入/输出契约兼容”的子图
  - 不要求两个子图内部结构同构
  - 不要求节点拓扑一致
  - 重点比较共享输入条件下的输出差异
- Compare 应围绕以下能力展开：
  - I/O 签名提取
  - 自动映射与手动确认映射
  - 共享输入规范
  - A/B 独立执行
  - 输出差异计算与展示

### 1.5 用户界面与体验

- 当前 NNJS 按钮/浮层与 Netron 原生视觉元素存在冲突，需要重新设计入口。
- 需要考虑以下 UI/UX 问题：
  - 与 Netron 左上 menu/titlebar、左下 toolbar、右侧 sidebar 的遮挡/冲突
  - 美化与视觉统一
  - 长任务等待时应展示什么内容
  - 结果如何查看
  - 用户常见误操作与预期偏差如何拦截

### 1.6 测试策略

- 大 ONNX 测试不能依赖外网慢速下载。
- 需要支持离线生成大模型测试样本。
- 可优先采用“大线性层 / MatMul + 随机初始化权重”的方式构造大 ONNX。
- 除大模型外，还需要构造小型多分支样本以验证 crop / compare 语义。

## 2. 当前结论（已冻结的产品与技术决策）

### 2.1 Compare 语义

- Compare 比较的是两个 `Confirmed Crop Artifact`。
- Compare 以 I/O 兼容为准，不要求子图同构。
- v1 不做内部节点或中间张量自动对比。
- v1 不做 dtype 自动转换，不做 broadcast 兼容。

### 2.2 总体架构

- ONNX 模型管理迁移到扩展宿主侧。
- Webview 只负责渲染、选择、状态展示与用户交互。
- 跨窗口共享状态统一由宿主侧 `Netron Compare` 底部 panel 管理。
- Compare 不允许面板点对点通信。

### 2.3 推理后端

- 推理后端固定为宿主侧 `onnxruntime-node`。

### 2.4 UI 入口

- 废弃当前左上 `NNJS` 浮层式入口。
- 新入口并入 Netron 底部 toolbar，统一命名为 `Model Tools`。
- `Model Tools` 展开后提供统一抽屉，而非额外悬浮层。

### 2.5 大模型测试

- 主样本使用离线脚本生成。
- 样本至少覆盖：
  - 大单文件 ONNX
  - 大 external-data ONNX
  - 小型多分支 crop 样本
  - 小型双 I/O compare 样本

### 2.6 平台支持

- 当前 VSIX 打包目标平台为：
  - Linux x64
  - Windows x64
- 为控制包体，暂不打包 Darwin / ARM64 原生运行时。

## 3. 实施计划

### 阶段 0：当前高优先级可用性修复

该阶段优先解决“已经影响日常使用”的问题，先恢复可操作性，再继续后续大重构。

1. Compare 面板可读性修复
   - 目标：解决当前底部 `Netron Compare` panel 在窄布局中严重溢出、表格难读、信息密度失控的问题。
   - 根因判断：
     - 当前 `extension.js` 中 compare view 曾使用普通 `<table>` 横向铺开全部字段。
     - `Input Bindings`、`Output Bindings`、`Results` 都没有响应式断点，也没有横向滚动容器。
     - 结果表把所有数值指标一次性平铺，导致在 VS Code 并排列宽下直接爆版。
   - 实施动作：
     - 将 `Input/Output Bindings` 改为“卡片行 + 下拉选择 + 状态标签”的响应式布局。
     - 将 `Results` 改为“两层展示”：
       - 第一层默认只展示 `A / B / Status / DType / Shape / Max Abs`。
       - 第二层将 `Mean Abs / RMSE / Max Rel / Cosine` 折叠到详情区或展开行。
     - 为绑定区和结果区增加独立 `overflow-x: auto` 容器，而不是让整页横向撑爆。
     - Summary、运行按钮、输入模式固定在上方，结果区单独滚动，避免滚动后丢失上下文。
     - 在窄宽度下将 A/B 槽位卡片从双列切换为单列堆叠。
   - 验收标准：
     - 在 VS Code `ViewColumn.Beside` 的窄面板宽度下，页面不出现整页内容截断。
     - 用户无需水平拖动整个 Webview，即可完成绑定与查看主要比较结论。
     - `Max diff output`、`A/B 来源`、当前运行状态始终可见。

2. `M` 按钮交互重做
   - 目标：解决当前必须“点开 M -> 点 Start/End -> Close -> 回图上点边”的割裂流程。
   - 根因判断：
     - 当前 `netron/source/workbench-ui.js` 里 `M` 入口直接打开固定抽屉。
     - `Start/End` 选择模式按钮位于抽屉内部，图上选择与抽屉关闭动作耦合。
     - 抽屉本身占据左下角图面空间，用户在选择边时容易被遮挡和打断。
   - 实施动作：
     - 将现有单一抽屉拆分为：
       - 常驻轻量 `Selection Bar`：只承载 `Start`、`End`、`Clear`、`Confirm`、当前计数与模式状态。
       - 次级 `Advanced Drawer`：承载 `Run`、`Compare`、`Activity`、导入输入、导出等低频动作。
     - `M` 按钮点击后默认显示轻量 bar，而不是大抽屉。
     - `Start` 与 `End` 变为互斥模式切换按钮，切换后无需关闭任何面板即可直接在图上点边。
     - 明确增加模式退出机制：
       - 再点一次当前按钮可退出模式。
       - `Esc` 可退出模式。
       - `Confirm Crop` 成功后自动退出选择模式。
     - 在轻量 bar 中持续显示：
       - 当前模式（Idle / Start / End）
       - 已选 `start/end` 数量
       - 当前草稿是否已脏
     - 保留高级抽屉，但 Crop 主路径必须不再依赖 Close 动作。
   - 验收标准：
     - 用户可在一个连续流程内完成“开启 Start -> 点边 -> 切 End -> 点边 -> Confirm”。
     - 整个流程中不需要先关闭任何 UI 才能继续图上操作。
     - 轻量 bar 不遮挡主要图区域，且状态反馈始终可见。

3. Weight Tensor 懒加载预览
   - 目标：恢复至少可用的权重张量查看能力，避免当前完全看不到 `weight tensor`。
   - 根因判断：
     - 当前宿主侧 `lib/onnx-workbench.js` 生成的 graph snapshot 只保留 initializer 的元信息：名称、类型、位置。
     - snapshot 没有携带 initializer 的实际值，因此 Netron sidebar 里的 `Show Tensor` 无法得到可渲染内容。
     - 若恢复“全量带值下发”，又会重新引入大模型传输和渲染压力。
   - 实施动作：
     - 保持默认 snapshot 轻量化，不在首次渲染时下发全部权重值。
     - 新增按需请求链路：
       - Webview 侧点击 `Show Tensor` 时发送 `requestTensorPreview(sessionId, tensorName)`。
       - 宿主侧按 tensor 名读取 initializer，并返回 `tensorPreviewResult`。
     - Preview 返回内容默认限制为“小而有用”的摘要，而不是完整 tensor：
       - `dtype`
       - `shape`
       - `location(inline/external)`
       - `elementCount`
       - 前 `N` 个值样本
       - 可选统计值：`min / max / mean`
     - 对超大 tensor 或 external-data tensor：
       - 只做 preview，不自动全量 materialize 到 webview。
       - 支持后续再扩展“load more”或“导出 tensor 摘要”。
     - 在宿主侧增加 session 级 preview cache，避免重复点击反复读取。
     - 在 sidebar 中增加 `loading / failed / truncated` 三种状态提示。
   - 验收标准：
     - 点击 weight 图标后，用户能在可接受时间内看到 tensor 的预览摘要。
     - 对大权重模型，预览不会导致主图重新加载或明显卡死。
     - external-data 权重至少能看到形状、dtype、位置与部分样本值。

4. 阶段 0 的实现顺序
   - 第 1 步：先修 Compare 面板布局，恢复 compare 结果可读性。
   - 第 2 步：再拆分 `M` 入口与选择态，恢复 crop 主路径流畅度。
   - 第 3 步：最后补 tensor lazy preview，恢复排障与对权重的基本可观察性。
   - 第 4 步：补最小手工 QA 清单，覆盖窄面板 compare、连续 start/end 选边、external-data 权重预览。

### 阶段 1：文档与状态重构

1. 新增正式设计文档，写清：
   - Compare 产品定义
   - I/O 契约
   - 宿主与 Webview 的职责边界
   - UI 状态机
   - 测试矩阵
2. 将当前全局单例 `panelState` 重构为支持多 panel 的 `PanelRegistry`。
3. 新增宿主侧数据中心：
   - `ModelSessionStore`
   - `CropArtifactStore`
   - `CompareCenterStore`

### 阶段 2：大 ONNX 加载链路迁移

1. 废弃当前 `base64 -> webview File -> BrowserFileContext` 的加载方式。
2. 改为宿主侧直接按 URI 读取 ONNX。
3. 宿主侧生成两类对象：
   - `HostOnnxModel`：用于 crop / export / inference
   - `RenderGraphSnapshot`：用于 webview 渲染
4. Webview 只接收轻量图快照，不再接收完整模型二进制。
5. 当前固定短超时改为：
   - 分阶段状态
   - 可取消
   - 无进展 watchdog

### 阶段 3：Crop 真源迁移

1. Webview 保留 tensor 选择与高亮。
2. 宿主接管 crop 计算与 artifact 生成。
3. `Confirm Crop` 后生成不可变 `CropArtifact`，包含：
   - `artifactId`
   - `modelSessionId`
   - `sourceGraphId`
   - `selection`
   - `selectedNodeIds`
   - `inputKeys/outputKeys`
   - `ioSignature`
   - `thumbnail`
   - `createdAt`
4. 草稿改变后，旧 artifact 标记为 `Stale`。

### 阶段 4：Crop ONNX 导出

1. 基于原始 ONNX proto 重建子图，而不是从渲染对象回写。
2. 导出时保留：
   - `ir_version`
   - `producer_name/version`
   - `opset_import`
   - graph metadata
   - 相关 `initializer`
   - 相关 `value_info`
   - `sparse_initializer`
   - 量化注释
3. external-data 模型默认导出为 `.onnx + sidecar 权重`。

### 阶段 5：单模型 Inference

1. 使用 `onnxruntime-node` 执行当前 `Confirmed Crop Artifact`。
2. 支持 `Use Full Graph` 明确切换为整图推理。
3. 输入准备分两步：
   - `Input Resolver`
   - `Input Builder`
4. 支持输入模式：
   - Auto / zeros
   - Auto / ones
   - Auto / random(seed 固定)
   - Import JSON
   - Import NPZ
5. 结果存入 `InferenceResult`。

### 阶段 6：Compare Panel

1. Compare 状态保存：
   - `slotA`
   - `slotB`
   - `inputBindings`
   - `outputBindings`
   - `compareRunStatus`
   - `compareResult`
2. Compare 兼容性只看参与 compare 的 I/O 绑定，不看内部结构。
3. 自动映射规则：
   - 先按同名 + 同 dtype + 同 rank
   - 再按唯一候选的同 dtype + 同 rank + 同 shape 推荐
   - 仍有歧义时必须人工确认
4. Compare 执行流程：
   - 校验 artifact
   - 解析/生成共享输入
   - 执行 A
   - 执行 B
   - 逐项输出差异计算
   - 生成总结与详情

### 阶段 7：UI / UX 重构

1. 将功能入口合并进 Netron 底部 toolbar。
2. 建立 `Model Tools` 抽屉，页签固定为：
   - `Crop`
   - `Run`
   - `Compare`
   - `Activity`
3. Compare 全局结果在底部 `Netron Compare` panel 中展示。
4. 右侧 sidebar 继续保留给：
   - 节点/连接属性
   - 推理结果详情
   - 输出张量 inspector

### 阶段 8：等待态与结果查看

1. 加载模型时保留 `welcome spinner`，但叠加：
   - 当前阶段
   - 文件名
   - 文件大小
   - 已耗时
   - 取消按钮
2. Crop / Export / Run / Compare 期间不使用全屏遮罩。
3. 等待态放在工具抽屉顶部状态区。
4. `Activity` 页签保存任务历史、错误原因与重试入口。
5. 单模型结果与 Compare 结果均支持：
   - 摘要查看
   - 详细查看
   - 导出
   - 过期标记

### 阶段 9：测试与样本

1. 新增离线大模型生成脚本。
2. 主样本使用大 MatMul / Gemm + 随机初始化权重。
3. 小样本覆盖多分支 crop 与双 I/O Compare。
4. 回归重点覆盖：
   - 双开不串状态
   - 草稿/确认/过期状态切换
   - Compare 基于 I/O 兼容而非同构
   - 长任务等待态与取消

## 4. 用户行为与防呆重点

### 4.1 草稿与确认态

- 当前选择必须区分：
  - `Draft`
  - `Confirmed`
  - `Stale`
- 只有 `Confirmed Crop` 才能导出、推理、发送 Compare。

### 4.2 Compare 防误配

- 发送到 Compare 时，必须显示来源摘要卡片：
  - 模型名
  - graph 名
  - 节点数
  - 输入输出摘要
  - 创建时间
  - panel 来源
- 替换槽位必须二次确认。

### 4.3 长任务防重复触发

- 长任务运行期间，同类按钮必须进入 disabled + busy 状态。
- 同一任务不能并发重复提交。

### 4.4 动态维度与输入导入

- 动态维度必须先补 concrete shape 后才能运行。
- 默认补 `1`，但必须明确提示这只是测试值。
- 输入导入需要先做预检摘要，再允许正式执行。

### 4.5 非模态选择流程

- Crop 主路径必须保持非模态：
  - 进入 `Start/End` 选择后，不应要求用户先关闭任何 UI。
  - 当前模式、计数和退出方式必须始终可见。
- 图上点击优先服务于选择操作，只有轻量 bar 自身区域才拦截点击。
- `Confirm`、`Clear`、模式切换都必须是可逆且即时反馈的。

### 4.6 Tensor 预览防失控

- Tensor 预览默认只加载摘要，不默认加载完整权重。
- 对超大 tensor 必须显式提示 `truncated` 或 `preview only`。
- 预览失败不能影响图浏览、crop、run、compare 主流程。

## 5. 结果查看原则

### 5.1 单模型推理

- 在 `Run` 页签中展示：
  - 状态
  - 耗时
  - artifact 名
  - 输入模式
  - 输出数
  - cache 命中情况

### 5.2 Compare

- 在底部 `Netron Compare` panel 中展示：
  - A/B 来源卡片
  - 总结卡
  - 输出差异表
  - 导出入口

### 5.3 Compare 可读性

- Compare 默认先展示“能帮助决策”的少量字段，而不是一次性平铺所有指标。
- 次要指标应进入折叠详情或次级视图。
- 在窄窗口下，绑定区与结果区必须各自可滚动，不能互相挤压。

### 5.4 Tensor 预览

- Tensor 预览应至少展示：
  - dtype
  - shape
  - location
  - element count
  - sample values
- 若数据被截断，必须明确告知当前仅为 preview。

### 5.5 过期状态

- 结果若与当前 artifact / 映射 / 输入参数不一致，必须标记为 `Outdated`。

## 6. 当前保存说明

- 本文档用于冻结当前已讨论完成的需求和实施方案，供后续实现与验收参考。
- 当前尚未开始继续实现下一阶段功能时，可优先以本文为准。
