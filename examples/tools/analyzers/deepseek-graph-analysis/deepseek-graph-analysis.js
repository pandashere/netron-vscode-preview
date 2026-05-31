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
          // keep raw body in the error path
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
  const graphText = await readStdin();
  if (!graphText.trim()) {
    throw new Error('Analyzer input is empty.');
  }
  const apiKey = readApiKey();
  if (!apiKey) {
    throw new Error('DEEPSEEK_API_KEY is required, or create ~/.netron/vscode-preview/secrets/deepseek_api_key.');
  }
  const baseUrl = (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\/+$/, '');
  const model = process.env.DEEPSEEK_MODEL || 'deepseek-v4-flash';
  const timeoutMs = Number(process.env.DEEPSEEK_TIMEOUT_MS || 120000);
  const result = await requestJson(baseUrl + '/chat/completions', apiKey, {
    model,
    messages: [
      {
        role: 'system',
        content: [
          '你是神经网络图结构分析助手。',
          '输入是 Netron crop 图的 edge-list 文本。',
          '请用中文输出，重点分析主干路径、分支/汇合、输入输出、可疑断点、可并行区域、以及对后续推理/裁剪/比较的建议。',
          '不要输出 Markdown 表格；保持短小、可直接阅读。'
        ].join('\n')
      },
      { role: 'user', content: graphText }
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
