# Examples

This directory contains desensitized reference files for private IR integration.

- `private-ir/` contains a small internal provider template and a sample `.myir.json` model.
- `tools/exporters/` contains external text exporter examples.
- `tools/analyzers/` contains external analyzer examples.
  - `line-count-analysis` is the smallest analyzer and is useful for checking the export -> analyze pipe.
  - `deepseek-graph-analysis` calls DeepSeek with exported graph text and no UI prompt fields.
  - `codex-one-shot-analysis` is a wrapper around `codex exec` for one-shot graph analysis. It demonstrates analyzer `description` and optional user prompt fields. It needs a working Codex CLI login where the extension host runs.
  - `deepseek-prompt-template-analysis` is a minimal DeepSeek analyzer that combines exported graph text with two user inputs and sends the final prompt to the model.

No file in this directory contains an API key. Configure real keys through environment variables or files under `~/.netron/vscode-preview/secrets`.

Read `docs/private-ir-integration-guide.md` before copying these files into a real integration.
