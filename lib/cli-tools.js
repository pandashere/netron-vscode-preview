const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const DEFAULT_MAX_TIMEOUT_MS = 300000;

function ensureArray(value) {
    return Array.isArray(value) ? value : [];
}

function toPublicEntry(entry) {
    return {
        key: entry.key,
        id: entry.id || null,
        label: entry.label || entry.id || entry.fileName || '(invalid)',
        status: entry.status,
        reason: entry.reason || '',
        path: entry.manifestPath || '',
        details: entry.details || []
    };
}

function normalizeTimeout(value, defaultTimeoutMs, maxTimeoutMs) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) {
        return defaultTimeoutMs;
    }
    return Math.min(Math.floor(numeric), maxTimeoutMs);
}

function validateManifest(raw, manifestPath, options = {}) {
    const defaultTimeoutMs = options.defaultTimeoutMs || 30000;
    const maxTimeoutMs = options.maxTimeoutMs || DEFAULT_MAX_TIMEOUT_MS;
    const dir = path.dirname(manifestPath);
    const fileName = path.basename(dir);
    const details = [];
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        return {
            key: `invalid:${manifestPath}`,
            status: 'error',
            fileName,
            label: `Invalid ${options.kind || 'tool'} config`,
            manifestPath,
            reason: 'Manifest must be a JSON object.',
            details: ['Manifest must be a JSON object.']
        };
    }
    const id = typeof raw.id === 'string' && raw.id.trim() ? raw.id.trim() : '';
    const label = typeof raw.label === 'string' && raw.label.trim() ? raw.label.trim() : id || fileName;
    const command = typeof raw.command === 'string' && raw.command.trim() ? raw.command.trim() : '';
    const args = raw.args === undefined ? [] : raw.args;
    const env = raw.env === undefined ? {} : raw.env;
    if (!id) {
        details.push('Missing required field: id.');
    }
    if (!command) {
        details.push('Missing required field: command.');
    }
    if (!Array.isArray(args) || args.some((item) => typeof item !== 'string')) {
        details.push('Field args must be an array of strings.');
    }
    if (!env || typeof env !== 'object' || Array.isArray(env) || Object.values(env).some((item) => typeof item !== 'string')) {
        details.push('Field env must be an object with string values.');
    }
    const base = {
        key: id ? `id:${id}` : `invalid:${manifestPath}`,
        id: id || null,
        label,
        command,
        args: Array.isArray(args) ? args.slice() : [],
        env: env && typeof env === 'object' && !Array.isArray(env) ? { ...env } : {},
        timeoutMs: normalizeTimeout(raw.timeoutMs, defaultTimeoutMs, maxTimeoutMs),
        dir,
        manifestPath,
        fileName
    };
    if (details.length > 0) {
        return {
            ...base,
            status: 'error',
            reason: details[0],
            details
        };
    }
    return {
        ...base,
        status: 'ready',
        reason: '',
        details: []
    };
}

class ToolRegistry {
    constructor(options) {
        this.kind = options.kind || 'tool';
        this.rootDir = options.rootDir;
        this.defaultTimeoutMs = options.defaultTimeoutMs || 30000;
        this.maxTimeoutMs = options.maxTimeoutMs || DEFAULT_MAX_TIMEOUT_MS;
        this.logger = typeof options.logger === 'function' ? options.logger : () => {};
        this.entries = [];
        this.watchers = [];
        this.listeners = new Set();
        this.refreshTimer = null;
    }

    onChange(listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    _emitChange() {
        const snapshot = this.getSnapshot();
        for (const listener of this.listeners) {
            try {
                listener(snapshot);
            } catch (error) {
                this.logger('warn', `${this.kind} registry listener failed`, { message: error.message });
            }
        }
    }

    getSnapshot() {
        return {
            kind: this.kind,
            rootDir: this.rootDir,
            entries: this.entries.map(toPublicEntry)
        };
    }

    getEntry(idOrKey) {
        return this.entries.find((entry) => entry.id === idOrKey || entry.key === idOrKey) || null;
    }

    getFirstReady() {
        return this.entries.find((entry) => entry.status === 'ready') || null;
    }

    refresh() {
        const entries = [];
        try {
            fs.mkdirSync(this.rootDir, { recursive: true });
            const children = fs.readdirSync(this.rootDir, { withFileTypes: true });
            for (const child of children) {
                if (!child.isDirectory()) {
                    continue;
                }
                const manifestPath = path.join(this.rootDir, child.name, `${this.kind}.json`);
                if (!fs.existsSync(manifestPath)) {
                    continue;
                }
                try {
                    const raw = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
                    entries.push(validateManifest(raw, manifestPath, this));
                } catch (error) {
                    entries.push({
                        key: `invalid:${manifestPath}`,
                        id: null,
                        label: `Invalid ${this.kind} config`,
                        status: 'error',
                        reason: `Failed to parse ${this.kind}.json.`,
                        details: [error.message],
                        manifestPath,
                        dir: path.dirname(manifestPath),
                        fileName: child.name
                    });
                }
            }
        } catch (error) {
            entries.push({
                key: `registry:${this.rootDir}`,
                id: null,
                label: `${this.kind} registry unavailable`,
                status: 'error',
                reason: error.message,
                details: [error.message],
                manifestPath: this.rootDir,
                dir: this.rootDir,
                fileName: path.basename(this.rootDir)
            });
        }

        const byId = new Map();
        for (const entry of entries) {
            if (entry.id) {
                if (!byId.has(entry.id)) {
                    byId.set(entry.id, []);
                }
                byId.get(entry.id).push(entry);
            }
        }
        for (const [id, matches] of byId.entries()) {
            if (matches.length <= 1) {
                continue;
            }
            const paths = matches.map((entry) => entry.manifestPath);
            for (const entry of matches) {
                entry.status = 'error';
                entry.reason = `${this.kind} naming/configuration conflict.`;
                entry.details = [`Duplicate ${this.kind} id: ${id}`, ...paths];
            }
        }

        this.entries = entries.sort((a, b) => String(a.label).localeCompare(String(b.label)));
        this._emitChange();
        return this.getSnapshot();
    }

    startWatching() {
        this.stopWatching();
        fs.mkdirSync(this.rootDir, { recursive: true });
        const watchDir = (dir) => {
            try {
                const watcher = fs.watch(dir, () => this.scheduleRefresh());
                this.watchers.push(watcher);
            } catch (error) {
                this.logger('warn', `${this.kind} watcher failed`, { dir, message: error.message });
            }
        };
        watchDir(this.rootDir);
        try {
            for (const child of fs.readdirSync(this.rootDir, { withFileTypes: true })) {
                if (child.isDirectory()) {
                    watchDir(path.join(this.rootDir, child.name));
                }
            }
        } catch (error) {
            this.logger('warn', `${this.kind} watcher child scan failed`, { message: error.message });
        }
    }

    scheduleRefresh() {
        if (this.refreshTimer) {
            clearTimeout(this.refreshTimer);
        }
        this.refreshTimer = setTimeout(() => {
            this.refreshTimer = null;
            this.refresh();
            this.startWatching();
        }, 250);
    }

    stopWatching() {
        if (this.refreshTimer) {
            clearTimeout(this.refreshTimer);
            this.refreshTimer = null;
        }
        while (this.watchers.length > 0) {
            const watcher = this.watchers.pop();
            try {
                watcher.close();
            } catch {
                // ignore close errors
            }
        }
    }
}

function runTool(entry, input, options = {}) {
    return new Promise((resolve, reject) => {
        if (!entry || entry.status !== 'ready') {
            reject(new Error(entry && entry.reason ? entry.reason : 'Selected tool is not ready.'));
            return;
        }
        let settled = false;
        let stdout = '';
        let stderr = '';
        const child = spawn(entry.command, ensureArray(entry.args), {
            cwd: entry.dir,
            shell: false,
            env: {
                ...process.env,
                ...entry.env,
                NETRON_TOOL_ID: entry.id || '',
                NETRON_TOOL_KIND: options.kind || ''
            }
        });
        if (typeof options.onProcess === 'function') {
            options.onProcess(child);
        }
        const timer = setTimeout(() => {
            if (settled) {
                return;
            }
            child.kill('SIGTERM');
            settled = true;
            const error = new Error(`${options.label || 'Tool'} timed out after ${entry.timeoutMs}ms.`);
            error.code = 'ETIMEDOUT';
            error.stdout = stdout;
            error.stderr = stderr;
            reject(error);
        }, entry.timeoutMs);
        child.stdout.on('data', (chunk) => {
            stdout += chunk.toString('utf8');
        });
        child.stderr.on('data', (chunk) => {
            stderr += chunk.toString('utf8');
        });
        child.stdin.on('error', (error) => {
            if (error && error.code !== 'EPIPE') {
                stderr += `${error.message}\n`;
            }
        });
        child.on('error', (error) => {
            if (settled) {
                return;
            }
            clearTimeout(timer);
            settled = true;
            error.stdout = stdout;
            error.stderr = stderr;
            reject(error);
        });
        child.on('close', (code, signal) => {
            if (settled) {
                return;
            }
            clearTimeout(timer);
            settled = true;
            if (code !== 0) {
                const error = new Error(`${options.label || 'Tool'} exited with code ${code}${signal ? ` (${signal})` : ''}.`);
                error.exitCode = code;
                error.signal = signal;
                error.stdout = stdout;
                error.stderr = stderr;
                reject(error);
                return;
            }
            if (stdout.trim().length === 0) {
                const error = new Error(`${options.label || 'Tool'} produced no output.`);
                error.stdout = stdout;
                error.stderr = stderr;
                reject(error);
                return;
            }
            resolve({ stdout, stderr, exitCode: code, signal });
        });
        try {
            child.stdin.end(input || '');
        } catch (error) {
            if (error && error.code !== 'EPIPE') {
                stderr += `${error.message}\n`;
            }
        }
    });
}

module.exports = {
    ToolRegistry,
    runTool,
    validateManifest,
    toPublicEntry
};
