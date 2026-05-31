#!/usr/bin/env node
const fs = require('fs');
const os = require('os');
const path = require('path');
const { ToolRegistry, runTool } = require('../lib/cli-tools');

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(predicate, label, timeoutMs = 3000) {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
        const value = predicate();
        if (value) {
            return value;
        }
        await sleep(25);
    }
    throw new Error(`Timed out waiting for ${label}.`);
}

function writeToolManifest(root, kind, name, manifest) {
    const dir = path.join(root, name);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${kind}.json`), JSON.stringify(manifest, null, 2));
}

function writeManifest(root, name, manifest) {
    writeToolManifest(root, 'exporter', name, manifest);
}

async function main() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-cli-tools-'));
    try {
        writeManifest(root, 'cat', {
            id: 'cat',
            label: 'Cat',
            command: '/bin/cat',
            args: []
        });
        writeManifest(root, 'empty', {
            id: 'empty',
            label: 'Empty',
            command: '/usr/bin/true',
            args: []
        });
        writeManifest(root, 'slow', {
            id: 'slow',
            label: 'Slow',
            command: '/bin/sh',
            args: ['-c', 'sleep 1; printf late'],
            timeoutMs: 50
        });
        writeManifest(root, 'snapshot', {
            id: 'snapshot',
            label: 'Snapshot',
            command: '/bin/sh',
            args: ['-c', 'sleep 0.15; printf "%s" "$NETRON_SNAPSHOT_VALUE"'],
            env: {
                NETRON_SNAPSHOT_VALUE: 'old'
            },
            timeoutMs: 1000
        });
        writeManifest(root, 'future-snapshot', {
            id: 'future-snapshot',
            label: 'Future Snapshot',
            command: '/bin/sh',
            args: ['-c', 'printf "%s" "$NETRON_SNAPSHOT_VALUE"'],
            env: {
                NETRON_SNAPSHOT_VALUE: 'before-refresh'
            },
            timeoutMs: 1000
        });
        writeManifest(root, 'over-timeout', {
            id: 'over-timeout',
            label: 'Over Timeout',
            command: '/bin/cat',
            args: [],
            timeoutMs: 999999
        });
        writeManifest(root, 'bad-env', {
            id: 'bad-env',
            label: 'Bad Env',
            command: '/bin/cat',
            args: [],
            env: {
                BAD: 1
            }
        });
        writeManifest(root, 'broken', {
            id: 'broken',
            label: 'Broken',
            args: []
        });
        writeManifest(root, 'dup-a', {
            id: 'dup',
            label: 'Duplicate A',
            command: '/bin/cat',
            args: []
        });
        writeManifest(root, 'dup-b', {
            id: 'dup',
            label: 'Duplicate B',
            command: '/bin/cat',
            args: []
        });
        const registry = new ToolRegistry({
            kind: 'exporter',
            rootDir: root,
            defaultTimeoutMs: 1000,
            maxTimeoutMs: 200
        });
        const snapshot = registry.refresh();
        assert(snapshot.entries.some((entry) => entry.id === 'cat' && entry.status === 'ready'), 'Ready exporter missing.');
        assert(snapshot.entries.some((entry) => entry.id === 'broken' && entry.status === 'error'), 'Invalid exporter missing.');
        assert(snapshot.entries.some((entry) => entry.id === 'bad-env' && entry.status === 'error'), 'Invalid env exporter missing.');
        assert(snapshot.entries.filter((entry) => entry.id === 'dup' && entry.status === 'error').length === 2, 'Duplicate exporters should both be disabled.');
        assert(registry.getEntry('over-timeout').timeoutMs === 200, 'Manifest timeout should be capped by registry max timeout.');

        const result = await runTool(registry.getEntry('cat'), 'hello', { label: 'Exporter' });
        assert(result.stdout === 'hello', `Unexpected stdout: ${JSON.stringify(result.stdout)}`);

        let emptyFailed = false;
        try {
            await runTool(registry.getEntry('empty'), 'x', { label: 'Exporter' });
        } catch (error) {
            emptyFailed = /no output/i.test(error.message);
        }
        assert(emptyFailed, 'Empty stdout should fail.');

        let timeoutFailed = false;
        try {
            await runTool(registry.getEntry('slow'), 'x', { label: 'Exporter' });
        } catch (error) {
            timeoutFailed = error.code === 'ETIMEDOUT' && /timed out/i.test(error.message);
        }
        assert(timeoutFailed, 'Timeout should fail with ETIMEDOUT.');

        const runningEntry = registry.getEntry('snapshot');
        const running = runTool(runningEntry, '', { label: 'Exporter' });
        writeManifest(root, 'snapshot', {
            id: 'snapshot',
            label: 'Snapshot Updated',
            command: '/bin/sh',
            args: ['-c', 'printf "%s" "$NETRON_SNAPSHOT_VALUE"'],
            env: {
                NETRON_SNAPSHOT_VALUE: 'new'
            },
            timeoutMs: 1000
        });
        registry.refresh();
        const runningResult = await running;
        assert(runningResult.stdout === 'old', 'Running task should use the manifest snapshot resolved at task start.');

        const futureEntry = registry.getEntry('future-snapshot');
        writeManifest(root, 'future-snapshot', {
            id: 'future-snapshot',
            label: 'Future Snapshot Updated',
            command: '/bin/sh',
            args: ['-c', 'printf "%s" "$NETRON_SNAPSHOT_VALUE"'],
            env: {
                NETRON_SNAPSHOT_VALUE: 'after-refresh'
            },
            timeoutMs: 1000
        });
        registry.refresh();
        const futureResult = await runTool(futureEntry, '', { label: 'Exporter' });
        assert(futureResult.stdout === 'before-refresh', 'Resolved entry should remain a start-time snapshot even after registry refresh.');

        const analyzerRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-analyzer-tools-'));
        try {
            writeToolManifest(analyzerRoot, 'analyzer', 'ok', {
                id: 'ok',
                label: 'OK Analyzer',
                command: '/bin/cat',
                args: []
            });
            writeToolManifest(analyzerRoot, 'analyzer', 'dup-a', {
                id: 'dup',
                label: 'Duplicate Analyzer A',
                command: '/bin/cat',
                args: []
            });
            writeToolManifest(analyzerRoot, 'analyzer', 'dup-b', {
                id: 'dup',
                label: 'Duplicate Analyzer B',
                command: '/bin/cat',
                args: []
            });
            const analyzerRegistry = new ToolRegistry({
                kind: 'analyzer',
                rootDir: analyzerRoot,
                defaultTimeoutMs: 1000
            });
            const analyzers = analyzerRegistry.refresh();
            assert(analyzers.entries.some((entry) => entry.id === 'ok' && entry.status === 'ready'), 'Ready analyzer missing.');
            assert(analyzers.entries.filter((entry) => entry.id === 'dup' && entry.status === 'error').length === 2, 'Duplicate analyzers should both be disabled.');
        } finally {
            fs.rmSync(analyzerRoot, { recursive: true, force: true });
        }

        const watchedRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'netron-watch-tools-'));
        const watchedRegistry = new ToolRegistry({
            kind: 'exporter',
            rootDir: watchedRoot,
            defaultTimeoutMs: 1000
        });
        let latestSnapshot = watchedRegistry.refresh();
        const unsubscribe = watchedRegistry.onChange((next) => {
            latestSnapshot = next;
        });
        try {
            watchedRegistry.startWatching();
            writeManifest(watchedRoot, 'hot', {
                id: 'hot',
                label: 'Hot Exporter',
                command: '/bin/cat',
                args: []
            });
            await waitFor(
                () => latestSnapshot.entries.some((entry) => entry.id === 'hot' && entry.status === 'ready'),
                'hot-plug exporter registry refresh'
            );
            writeManifest(watchedRoot, 'hot', {
                id: 'hot',
                label: 'Hot Exporter',
                args: []
            });
            await waitFor(
                () => latestSnapshot.entries.some((entry) => entry.id === 'hot' && entry.status === 'error' && /command/.test(entry.reason)),
                'hot-edit exporter invalidation'
            );
            fs.rmSync(path.join(watchedRoot, 'hot'), { recursive: true, force: true });
            await waitFor(
                () => !latestSnapshot.entries.some((entry) => entry.id === 'hot'),
                'hot-delete exporter removal'
            );
        } finally {
            unsubscribe();
            watchedRegistry.stopWatching();
            fs.rmSync(watchedRoot, { recursive: true, force: true });
        }

        console.log('cli tools ok', { entries: snapshot.entries.length });
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
});
