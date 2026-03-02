# Netron VSCode Preview
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A VS Code extension workspace that previews model files with Netron parsers and adds NNJS export and crop tools.

## Command

Use command palette (`Ctrl+Shift+P`) and run:

- `Netron: Preview Model`

You can also right-click a file in explorer and run the same command.

## Implemented Features

- Preview model graph in VS Code Webview using Netron parsers (`netron/source/*`).
- One-click `Convert To NNJS` button.
- NNJS structure and weights are separated:
  - default: no weight export
  - optional: include weights and save as separate `*.weights.json`
- Expandable crop panel:
  - `Select Start Tensor` / `Select End Tensor`
  - click once to select tensor edge, click again to unselect
  - multi-select supported
  - `Confirm Crop` computes subgraph by forward/backward traversal intersection
- Crop output behavior:
  - cropped graph is rendered in memory
- screenshot is cached in memory first
- then user can `Save Crop Screenshot` or `Copy Crop Screenshot`
- Save cropped NNJS (and cropped weights if enabled).

## VSIX 打包

```bash
cd /home/zhaochen/net_vsix
npx @vscode/vsce ls
mkdir -p dist
npx @vscode/vsce package -o dist/netron-vscode-preview-0.0.5.vsix
```

## VSCode 安装

```bash
cd /home/zhaochen/net_vsix
code --uninstall-extension local.netron-vscode-preview || true
code --install-extension dist/netron-vscode-preview-0.0.5.vsix --force
```

## 命令验证

```bash
code --list-extensions --show-versions | rg local.netron-vscode-preview
```

Expected output includes:

- `local.netron-vscode-preview@0.0.5`

## 功能验证（NNJS / Crop / Screenshot）

In VSCode:

- Press `Ctrl+Shift+P` and run `Netron: Preview Model`.
- Open a model and confirm `NNJS Tools` appears in the top-right area.
- Click `Convert To NNJS`, then `Save NNJS`.
- Expand `Crop`, use `Select Start Tensor` and `Select End Tensor`.
- Click the same tensor edge twice to verify select/unselect behavior.
- Multi-select multiple start/end tensors, then click `Confirm Crop`.
- Verify the graph updates and screenshot is cached in memory.
- Click `Save Crop Screenshot` and `Copy Crop Screenshot`.
