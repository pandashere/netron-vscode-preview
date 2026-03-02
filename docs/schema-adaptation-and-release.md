# 新 Schema 适配与 VSIX 打包发布手册

本文档面向本项目（`netron-vscode-preview`）的维护者，目标是：

1. 把“非 Netron 官方仓库”的新 schema 接入当前扩展。
2. 完成本地验证、VSIX 打包、安装验收。
3. 产出最小可运行仓库，便于迁移到其他环境发布。

## 1. 架构与接入点

当前扩展由两部分组成：

1. `extension.js`：VS Code 扩展宿主侧，负责命令、Webview 通信、文件读写、剪贴板等。
2. `netron/source/**`：Netron 前端与模型解析器实现。

新 schema 的核心接入点在 Netron 的 `ModelFactoryService` 注册表：

1. 工厂注册位置：`netron/source/view.js` 中 `ModelFactoryService` 构造函数。
2. 解析调用链：`ModelFactoryService._openContext(context)` -> 各 schema 的 `ModelFactory.match/open`。

关键参考：

1. `netron/source/view.js`（注册与调度）
2. `netron/source/message.js`（较完整的对象映射样例）
3. `netron/source/imgdnn.js`（最小骨架样例）

## 2. 新 Schema 适配策略

### 2.1 推荐决策

1. 如果你的格式可稳定转 ONNX：优先“离线转换 -> ONNX 预览”，维护成本最低。
2. 如果必须原生支持：新增 `ModelFactory` 模块并注册到 `view.js`。

### 2.2 适配目标对象模型

`ModelFactory.open()` 最终返回的对象至少应满足：

1. `model.format`：字符串，如 `MySchema v1.0`
2. `model.modules`（或 `graphs`）：图数组
3. 每个图包含：
1. `inputs`：参数数组
2. `outputs`：参数数组
3. `nodes`：节点数组
4. 参数（Argument）包含：
1. `name`
2. `value: Value[]`
5. Value 包含：
1. `name`
2. `type`（可选）
3. `initializer`（可选，常量张量）

注意：常量权重请尽量走 `Value.initializer`，避免在可视图中被误识别为普通连接。

## 3. 代码实现步骤

### 步骤 1：新增 schema 模块

示例文件：`netron/source/myschema.js`

```js
const myschema = {};

myschema.ModelFactory = class {

    async match(context) {
        const identifier = context.identifier || '';
        const extension = identifier.lastIndexOf('.') > 0 ? identifier.split('.').pop().toLowerCase() : '';
        const stream = context.stream;
        if (!stream) {
            return null;
        }
        if (extension === 'mys') {
            return context.set('myschema');
        }
        return null;
    }

    async open(context) {
        // 读取源数据（按你的 schema 选择 json/protobuf/binary）
        const obj = await context.peek('json');
        if (!obj) {
            throw new myschema.Error('Invalid MySchema file content.');
        }
        return new myschema.Model(obj);
    }
};

myschema.Model = class {
    constructor(root) {
        this.format = root.version ? `MySchema v${root.version}` : 'MySchema';
        this.modules = [new myschema.Graph(root.graph)];
    }
};

myschema.Graph = class {
    constructor(graph) {
        this.name = graph.name || 'graph';
        const values = new Map();
        const valueOf = (name) => {
            if (!values.has(name)) {
                values.set(name, new myschema.Value({ name }));
            }
            return values.get(name);
        };

        // 常量张量（可选）
        for (const weight of graph.weights || []) {
            values.set(weight.name, new myschema.Value({
                name: weight.name,
                initializer: {
                    type: {
                        dataType: weight.dataType,
                        shape: { dimensions: weight.shape || [] }
                    },
                    // 可按需扩展 values/value
                    values: weight.values || null
                }
            }));
        }

        this.inputs = (graph.inputs || []).map((name) => new myschema.Argument({
            name,
            value: [valueOf(name)]
        }));
        this.outputs = (graph.outputs || []).map((name) => new myschema.Argument({
            name,
            value: [valueOf(name)]
        }));
        this.nodes = (graph.nodes || []).map((node) => new myschema.Node(node, valueOf));
    }
};

myschema.Node = class {
    constructor(node, valueOf) {
        this.type = { name: node.op || 'CustomOp' };
        this.name = node.name || '';
        this.inputs = (node.inputs || []).map((name) => new myschema.Argument({
            name: '',
            value: [valueOf(name)]
        }));
        this.outputs = (node.outputs || []).map((name) => new myschema.Argument({
            name: '',
            value: [valueOf(name)]
        }));
        this.attributes = Object.entries(node.attrs || {}).map(([k, v]) => new myschema.Argument({
            name: k,
            type: typeof v,
            value: v
        }));
    }
};

myschema.Argument = class {
    constructor(data) {
        this.name = data.name || '';
        this.value = data.value || [];
        this.type = data.type || null;
    }
};

myschema.Value = class {
    constructor(data) {
        this.name = data.name || '';
        this.initializer = data.initializer || null;
        this.type = this.initializer ? this.initializer.type : (data.type || null);
    }
};

myschema.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'MySchema Error';
    }
};

export const ModelFactory = myschema.ModelFactory;
```

### 步骤 2：注册新模块

编辑 `netron/source/view.js` 的 `ModelFactoryService` 构造函数，新增一行：

```js
this.register('./myschema', ['.mys', '.myschema']);
```

建议放在相近格式附近，便于维护。

### 步骤 3：可选 metadata

如果需要更丰富的算子展示，可在 `netron/source/*.json` 增加 metadata 文件，并在 parser 里通过 `context.metadata()` 读取（参考现有 parser）。

## 4. 本地验证流程

### 4.1 语法检查

```bash
node --check extension.js
node --check netron/source/vscode.js
node --check netron/source/myschema.js
```

### 4.2 打包并安装

```bash
npm run package:vsix
code --install-extension /absolute/path/to/dist/netron-vscode-preview-<version>.vsix --force
code --list-extensions --show-versions | rg local.netron-vscode-preview
```

### 4.3 运行验证

1. `Developer: Reload Window`
2. `Netron: Preview Model`
3. 选择你的 `*.mys` 样例
4. 验收项：
1. 预览可进入图视图（非 spinner）
2. 节点/边/输入输出正确
3. 常量权重不应错误显示为普通连接节点
4. NNJS/Crop 工具链基本流程可执行

## 5. 打包与发布规范

### 5.1 必要运行文件

当前 VSIX 运行所需最小集：

1. `package.json`
2. `extension.js`
3. `README.md`
4. `netron/LICENSE`
5. `netron/source/**`

### 5.2 `.vscodeignore` 建议（白名单）

```ignore
**/*
!package.json
!extension.js
!README.md
!netron/
!netron/source/**
!netron/LICENSE
```

### 5.3 版本管理建议

每次对外发包都递增 `package.json.version`（例如 `0.0.4 -> 0.0.5`），避免 VS Code 侧缓存与覆盖歧义。

## 6. 常见问题与排障

1. 安装后看不到新布局/新 parser：
1. 检查版本号是否递增。
2. 先卸载旧版本再安装新 VSIX。
3. `Developer: Reload Window`。

2. Webview ready 超时：
1. 查看 `Output -> Netron Preview` 日志。
2. 检查 `vscode.js` 是否在打包清单中。

3. 非模型文件一直转圈：
1. 应回落 welcome 并返回 `modelOpenFailed`。
2. 若未回落，优先检查 webview 消息链路是否断开。

4. Crop 后常量显示异常：
1. 确保常量在 parser 中以 `initializer` 表达。
2. 检查同名 Value 合并逻辑是否回填 initializer。

## 7. 最小发布命令清单

```bash
# 1) 语法检查
node --check extension.js
node --check netron/source/vscode.js

# 2) 打包
npm run package:vsix

# 3) 安装
code --install-extension /absolute/path/to/dist/netron-vscode-preview-<version>.vsix --force

# 4) 验证
code --list-extensions --show-versions | rg local.netron-vscode-preview
```

