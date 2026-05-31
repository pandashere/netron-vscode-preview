#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

function readStdin() {
  return new Promise((resolve) => {
    let input = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      input += chunk;
    });
    process.stdin.on('end', () => resolve(input));
  });
}

function parseExtraArgs(value) {
  if (!value || !value.trim()) {
    return [];
  }
  const trimmed = value.trim();
  if (trimmed.startsWith('[')) {
    const parsed = JSON.parse(trimmed);
    if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== 'string')) {
      throw new Error('CODEX_EXTRA_ARGS JSON must be an array of strings.');
    }
    return parsed;
  }
  const result = [];
  let current = '';
  let quote = '';
  let escaped = false;
  for (const char of trimmed) {
    if (escaped) {
      current += char;
      escaped = false;
      continue;
    }
    if (char === '\\') {
      escaped = true;
      continue;
    }
    if (quote) {
      if (char === quote) {
        quote = '';
      } else {
        current += char;
      }
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }
    if (/\s/.test(char)) {
      if (current) {
        result.push(current);
        current = '';
      }
      continue;
    }
    current += char;
  }
  if (escaped) {
    current += '\\';
  }
  if (quote) {
    throw new Error('CODEX_EXTRA_ARGS contains an unterminated quote.');
  }
  if (current) {
    result.push(current);
  }
  return result;
}

function buildPrompt(graphText) {
  const prompt = process.env.CODEX_ANALYSIS_PROMPT && process.env.CODEX_ANALYSIS_PROMPT.trim()
    ? process.env.CODEX_ANALYSIS_PROMPT.trim()
    : [
        '你是神经网络图结构分析助手。',
        '输入是一段由 Netron VSCode Workbench exporter 生成的图结构文本，通常是 confirmed crop 的 edge-list 或摘要。',
        '请只基于输入内容做分析，不要修改文件，不要联网，不要运行会改变环境的命令。',
        '请用中文输出，保持结论短小、可执行，重点覆盖：',
        '1. 主干路径和关键算子。',
        '2. 分支、汇合、常量或 initializer 的作用。',
        '3. 输入输出边界是否清晰。',
        '4. 可能影响裁剪、推理或跨格式 compare 的风险。',
        '5. 下一步建议。'
      ].join('\n');
  return [
    prompt,
    '',
    '<graph-text>',
    graphText.trim(),
    '</graph-text>'
  ].join('\n');
}

function runCodex(prompt) {
  return new Promise((resolve, reject) => {
    const command = process.env.CODEX_COMMAND || 'codex';
    const workdir = process.env.CODEX_WORKDIR && process.env.CODEX_WORKDIR.trim()
      ? process.env.CODEX_WORKDIR.trim()
      : os.homedir();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-codex-analysis-'));
    const outputPath = path.join(tempDir, 'last-message.txt');
    const args = [
      '--ask-for-approval',
      process.env.CODEX_APPROVAL || 'never',
      'exec',
      '--sandbox',
      process.env.CODEX_SANDBOX || 'read-only',
      '--ephemeral',
      '--skip-git-repo-check',
      '--color',
      'never',
      '--output-last-message',
      outputPath
    ];
    if (process.env.CODEX_MODEL && process.env.CODEX_MODEL.trim()) {
      args.push('--model', process.env.CODEX_MODEL.trim());
    }
    args.push('--cd', workdir);
    args.push(...parseExtraArgs(process.env.CODEX_EXTRA_ARGS || ''));
    args.push('-');

    let stdout = '';
    let stderr = '';
    const child = spawn(command, args, {
      cwd: workdir,
      shell: false,
      env: process.env
    });

    const killChild = (signal) => {
      if (!child.killed) {
        child.kill(signal);
      }
    };
    process.once('SIGTERM', () => {
      killChild('SIGTERM');
      process.exit(143);
    });
    process.once('SIGINT', () => {
      killChild('SIGINT');
      process.exit(130);
    });

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', (error) => {
      fs.rmSync(tempDir, { recursive: true, force: true });
      error.stderr = stderr;
      reject(error);
    });
    child.on('close', (code, signal) => {
      try {
        if (code !== 0) {
          const error = new Error(`codex exec exited with code ${code}${signal ? ` (${signal})` : ''}.`);
          error.stdout = stdout;
          error.stderr = stderr;
          reject(error);
          return;
        }
        const finalText = fs.existsSync(outputPath) ? fs.readFileSync(outputPath, 'utf8') : stdout;
        if (!finalText.trim()) {
          const error = new Error('codex exec produced no final analysis.');
          error.stdout = stdout;
          error.stderr = stderr;
          reject(error);
          return;
        }
        resolve(finalText.trim());
      } finally {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
    });
    child.stdin.end(prompt);
  });
}

async function main() {
  const graphText = await readStdin();
  if (!graphText.trim()) {
    throw new Error('Analyzer input is empty.');
  }
  const result = await runCodex(buildPrompt(graphText));
  process.stdout.write(result);
}

main().catch((error) => {
  console.error(error && error.message ? error.message : String(error));
  if (error && error.stderr) {
    console.error(error.stderr.trim());
  }
  process.exit(1);
});
