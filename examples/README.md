# Examples

This directory contains desensitized reference files for private IR integration.

- `private-ir/` contains a small internal provider template and a sample `.myir.json` model.
- `tools/exporters/` contains external text exporter examples.
- `tools/analyzers/` contains external analyzer examples.
  - `codex-one-shot-analysis` is a wrapper around `codex exec` for one-shot graph analysis. It needs a working Codex CLI login where the extension host runs.

No file in this directory contains an API key. Configure real keys through environment variables or files under `~/.netron/vscode-preview/secrets`.

Read `docs/private-ir-integration-guide.md` before copying these files into a real integration.
