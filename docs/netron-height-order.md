# Netron 节点高低关系整理

本文只整理 **Netron 默认纵向布局** 下，节点为什么会被放在某个“高度层”。

## 1. 从 graph 到 Dagre 输入

Netron 在 `netron/source/grapher.js` 里把内部 graph 转成 Dagre 的输入：

- 每个节点变成 `{ v, width, height, parent }`
- 每条边变成 `{ v, w, minlen, weight, width, height, labeloffset, labelpos }`
- 默认：
  - `nodesep = 20`
  - `ranksep = 20`
  - 对普通有边图，默认 `rankdir = 'TB'`

关键点：**节点的“高低层”不是在渲染阶段随便排的，而是先由 Dagre 算出 `rank`，后面再把 `rank` 转成 `y`。**

## 2. 真正决定“高低层”的步骤

在 `netron/source/dagre.js` 中，布局主流程依次执行：

1. `acyclic_run`
   - 如果图里有回边，先反转回边，得到 DAG。

2. `rank`
   - 小图默认走 `network-simplex`
   - 超过 3000 节点时走 `longest-path`
   - 这一步给每个节点打上整数层号 `rank`

3. `order`
   - 只决定**同一层内**左右顺序
   - **不会改变高低层**

4. `position`
   - 按层把 `rank -> y`

5. `translateGraph`
   - 只做整体平移，把最小坐标挪到正区间
   - **不会改变高低关系**

所以对“高低排序关系”来说，真正核心的是：

- `acyclic_run`
- `rank`
- `position` 里的 layer-to-y 映射

## 3. Netron 最终怎么把 rank 变成 y

`position()` 里的纵向规则非常直接：

```text
y = 0
for each layer in rank order:
    max_height = 当前层所有节点高度的最大值
    当前层每个节点的中心 y = y + max_height / 2
    y += max_height + ranksep
```

也就是说：

- 不同 `rank` 一定是不同高度层
- 同一 `rank` 的节点，中心 `y` 完全相同
- 节点自己的高度只影响：
  - 这一层整体占多高
  - 后续层的累计偏移
- 它**不影响**这个节点属于第几层

## 4. 对“只有邻接关系输入”的简化结论

如果输入只有稀疏邻接关系，而没有节点宽高、边标签、cluster 层级，那么：

- 你仍然可以**准确复现 Netron 的层号 `rank`**
- 也就能得到节点的**高低排序关系**
- 但你不能复现 Netron 的**绝对像素 y 值**
  - 除非再提供每个节点最终渲染出来的高度

## 5. Python 版对应实现

见：

- `scripts/netron_height_order.py`

这个脚本实现了：

- DFS 回边反转（`acyclic_run`）
- Netron 风格的 `network-simplex` / `longest-path` rank
- `rank -> layers`
- 可选 `rank -> y`

接口输入：

- `Sequence[np.ndarray]`
- `adjacency[u]` 存 `u -> v` 的所有目标节点

接口输出：

- `ranks: np.ndarray`
- `layers: List[np.ndarray]`
- `y: np.ndarray`

其中：

- `ranks[i]` 越小，节点越靠上
- `layers[k]` 是第 `k` 层的所有节点
- `y` 只有在你给 `node_heights` 时才接近 Netron 最终坐标；否则只是保序的层中心
