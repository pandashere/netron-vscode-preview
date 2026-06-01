#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const https = require('https');

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

function parseInput(text) {
  try {
    const payload = JSON.parse(text);
    if (payload && payload.kind === 'netron-analyzer-input') {
      return {
        graphText: payload.exportedText || '',
        input1: payload.userInputs && payload.userInputs.input1 ? String(payload.userInputs.input1) : '',
        input2: payload.userInputs && payload.userInputs.input2 ? String(payload.userInputs.input2) : ''
      };
    }
  } catch {
    // Plain stdin remains useful when testing this script outside VS Code.
  }
  return { graphText: text, input1: '', input2: '' };
}

function readApiKey() {
  if (process.env.DEEPSEEK_API_KEY) {
    return process.env.DEEPSEEK_API_KEY.trim();
  }
  const keyFile = process.env.DEEPSEEK_API_KEY_FILE
    || path.join(os.homedir(), '.netron', 'vscode-preview', 'secrets', 'deepseek_api_key');
  try {
    return fs.readFileSync(keyFile, 'utf8').trim();
  } catch {
    return '';
  }
}

function buildPrompt({ graphText, input1, input2 }) {
  return [
    '你是一个严谨的神经网络图分析专家，擅长阅读模型计算图、识别关键算子链路、分析数据流依赖，并将结构信息转化为可执行的调试和优化建议。',
    '',
    '现在我们有如下图结构。它来自 Netron 当前选区的结构化文本导出，描述了节点、连接关系、部分属性以及 tensor 元信息：',
    '',
    graphText.trim(),
    '',
    '用户给出如下输入：',
    input1.trim() || '(未提供)',
    '',
    '注意事项：',
    input2.trim() || '(未提供)',
    '',
    '请结合图结构分析用户输入，并按注意事项完成图分析。请用中文输出，先给出结论，再列出关键路径、可疑点和建议动作。不要输出 Markdown 表格。'
  ].join('\n');
}

function requestJson(urlText, apiKey, payload, timeoutMs) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlText);
    const body = JSON.stringify(payload);
    const transport = url.protocol === 'http:' ? http : https;
    const request = transport.request({
      method: 'POST',
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port || undefined,
      path: url.pathname + url.search,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey,
        'Content-Length': Buffer.byteLength(body)
      },
      timeout: timeoutMs
    }, (response) => {
      let responseBody = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => {
        responseBody += chunk;
      });
      response.on('end', () => {
        let parsed = null;
        try {
          parsed = responseBody ? JSON.parse(responseBody) : null;
        } catch {
          // keep raw response for error diagnostics
        }
        if (response.statusCode < 200 || response.statusCode >= 300) {
          const error = new Error('DeepSeek request failed with HTTP ' + response.statusCode + '.');
          error.responseBody = responseBody.slice(0, 2000);
          reject(error);
          return;
        }
        resolve(parsed);
      });
    });
    request.on('timeout', () => {
      request.destroy(new Error('DeepSeek request timed out.'));
    });
    request.on('error', reject);
    request.end(body);
  });
}

async function main() {
  const input = parseInput(await readStdin());
  if (!input.graphText.trim()) {
    throw new Error('Analyzer input is empty.');
  }
  const apiKey = readApiKey();
  if (!apiKey) {
    throw new Error('DEEPSEEK_API_KEY is required, or create ~/.netron/vscode-preview/secrets/deepseek_api_key.');
  }

  const baseUrl = (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\/+$/, '');
  const model = process.env.DEEPSEEK_MODEL || 'deepseek-chat';
  const timeoutMs = Number(process.env.DEEPSEEK_TIMEOUT_MS || 120000);
  const result = await requestJson(baseUrl + '/chat/completions', apiKey, {
    model,
    messages: [
      {
        role: 'system',
        content: '你是一个图分析专家。请严格基于用户提供的图结构和要求分析，不要编造不存在的节点或边。'
      },
      {
        role: 'user',
        content: buildPrompt(input)
      }
    ],
    temperature: 0.2,
    max_tokens: 2048,
    stream: false
  }, Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 120000);

  const text = result && result.choices && result.choices[0] && result.choices[0].message
    ? result.choices[0].message.content
    : '';
  if (!text || !text.trim()) {
    throw new Error('DeepSeek returned an empty analysis result.');
  }
  process.stdout.write(text.trim());
}

main().catch((error) => {
  console.error(error && error.message ? error.message : String(error));
  if (error && error.responseBody) {
    console.error(error.responseBody);
  }
  process.exit(1);
});
