#!/usr/bin/env node
/**
 * Smoke test for the Tree-RAG Memory MCP Hub (`dashboard/server.js`).
 *
 * Seeds a temporary base directory from `examples/sample_base` (2 fake workspaces:
 * `sample_project` and `demo_workspace`), starts the HTTP server on an ephemeral port,
 * asserts `/api/memory/status`, `/api/memory/tree`, `/api/memory/records`,
 * `/api/memory/search`, `/api/memory/save`, `/api/memory/neighbors`, `/api/memory/update`,
 * `/api/memory/mermaid`, `/api/memory/delete`, `/api/memory/pending`, `/api/memory/merge`,
 * and `/api/memory/workspace-sync`, then cleans up and exits 0.
 */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { createHubServer } = require('../server');

function copyDirRecursive(src, dest) {
    fs.mkdirSync(dest, { recursive: true });
    for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);
        if (entry.isDirectory()) {
            copyDirRecursive(srcPath, destPath);
        } else {
            fs.copyFileSync(srcPath, destPath);
        }
    }
}

async function requestJson(baseUrl, route, options = {}) {
    const res = await fetch(`${baseUrl}${route}`, {
        method: options.method || 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...(options.headers || {}),
        },
        body: options.body ? JSON.stringify(options.body) : undefined,
    });
    const text = await res.text();
    let json;
    try {
        json = JSON.parse(text);
    } catch (err) {
        throw new Error(`Invalid JSON from ${route} (status ${res.status}): ${text.slice(0, 200)}`);
    }
    return { status: res.status, data: json };
}

async function runSmokeTest() {
    const tmpBase = fs.mkdtempSync(path.join(os.tmpdir(), 'treerag-hub-smoke-'));
    const sampleBase = path.resolve(__dirname, '..', '..', 'examples', 'sample_base');
    copyDirRecursive(sampleBase, tmpBase);

    const server = createHubServer({ baseDir: tmpBase });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const addr = server.address();
    const port = typeof addr === 'object' && addr ? addr.port : 0;
    const baseUrl = `http://127.0.0.1:${port}`;

    try {
        // 1. Static HTML & JS assets
        const htmlRes = await fetch(`${baseUrl}/`);
        assert.strictEqual(htmlRes.status, 200, 'GET / should return 200');
        const htmlText = await htmlRes.text();
        assert.ok(
            htmlText.includes('Memory MCP Hub — Hierarchical Tree RAG'),
            'index.html should contain hub header'
        );

        // 2. GET /api/memory/status & /api/memory/health
        const statusRes = await requestJson(baseUrl, '/api/memory/status');
        assert.strictEqual(statusRes.status, 200);
        assert.strictEqual(statusRes.data.ok, true);
        assert.strictEqual(statusRes.data.workspaces.length, 2, 'Expected 2 seeded workspaces');
        assert.strictEqual(statusRes.data.totalDocs, 10, 'Expected 10 seeded docs across 2 workspaces');

        const healthRes = await requestJson(baseUrl, '/api/memory/health');
        assert.strictEqual(healthRes.status, 200);
        assert.strictEqual(healthRes.data.ok, true);

        // 3. GET /api/memory/tree
        const treeRes = await requestJson(baseUrl, '/api/memory/tree');
        assert.strictEqual(treeRes.status, 200);
        assert.strictEqual(treeRes.data.ok, true);
        assert.strictEqual(treeRes.data.summary.totalWorkspaces, 2);
        assert.strictEqual(treeRes.data.summary.totalDocs, 10);
        assert.ok(treeRes.data.summary.layers.L3 >= 3, 'Should have L3 (decision/skill) records');
        assert.ok(treeRes.data.summary.layers.L2 >= 3, 'Should have L2 (code/wiki) records');
        assert.ok(treeRes.data.summary.layers.L1 >= 2, 'Should have L1 (bug/assumption) records');
        assert.ok(treeRes.data.summary.layers.L0 >= 2, 'Should have L0 (note) records');
        assert.ok(treeRes.data.summary.pillars.skill >= 3);
        assert.ok(treeRes.data.summary.pillars.code >= 4);

        // 4. GET /api/memory/records
        const recRes = await requestJson(baseUrl, '/api/memory/records?workspace=sample_project&limit=10');
        assert.strictEqual(recRes.status, 200);
        assert.strictEqual(recRes.data.ok, true);
        assert.strictEqual(recRes.data.total, 6);
        assert.strictEqual(recRes.data.records.length, 6);

        // 5. POST /api/memory/save
        const saveRes = await requestJson(baseUrl, '/api/memory/save', {
            method: 'POST',
            body: {
                workspace: 'sample_project',
                category: 'decision',
                source: 'docs/adr/003-circuit-breaker.md',
                tags: ['acme-weather-api', 'circuit-breaker'],
                text: 'acme-weather-api decision: Trip upstream radar circuit breaker after 5 consecutive 5xx responses.',
            },
        });
        assert.strictEqual(saveRes.status, 200);
        assert.strictEqual(saveRes.data.ok, true);
        assert.strictEqual(saveRes.data.layer, 'L3');
        assert.strictEqual(saveRes.data.pillar, 'skill');
        const createdId = saveRes.data.id || (saveRes.data.doc && saveRes.data.doc.id);
        assert.ok(createdId, 'Save response must return a document id');

        // 6. POST /api/memory/search
        const searchRes = await requestJson(baseUrl, '/api/memory/search', {
            method: 'POST',
            body: {
                query: 'circuit breaker radar 5xx responses',
                scope: 'workspace',
                workspace: 'sample_project',
                top_k: 5,
            },
        });
        assert.strictEqual(searchRes.status, 200);
        assert.strictEqual(searchRes.data.ok, true);
        assert.ok(searchRes.data.results.length >= 1, 'Search should return at least 1 hit');
        assert.ok(
            searchRes.data.highlights.workspaces.includes('sample_project'),
            'Search highlights should include sample_project'
        );

        // 7. POST /api/memory/neighbors
        const neighRes = await requestJson(baseUrl, '/api/memory/neighbors', {
            method: 'POST',
            body: {
                workspace: 'sample_project',
                id: createdId,
                top_k: 3,
            },
        });
        assert.strictEqual(neighRes.status, 200);
        assert.strictEqual(neighRes.data.ok, true);
        assert.strictEqual(neighRes.data.target.id, createdId);
        assert.ok(Array.isArray(neighRes.data.semantic_neighbors));

        // 8. POST /api/memory/update
        const updRes = await requestJson(baseUrl, '/api/memory/update', {
            method: 'POST',
            body: {
                workspace: 'sample_project',
                id: createdId,
                category: 'decision',
                source: 'docs/adr/003-circuit-breaker.md',
                tags: ['acme-weather-api', 'circuit-breaker', 'updated'],
                text: 'acme-weather-api decision: Trip upstream radar circuit breaker after 4 consecutive 5xx responses (30s cooldown).',
            },
        });
        assert.strictEqual(updRes.status, 200);
        assert.strictEqual(updRes.data.ok, true);

        // 9. GET /api/memory/mermaid
        const mmdGlobal = await requestJson(baseUrl, '/api/memory/mermaid?workspace=global');
        assert.strictEqual(mmdGlobal.status, 200);
        assert.strictEqual(mmdGlobal.data.ok, true);
        assert.ok(mmdGlobal.data.mermaid.startsWith('flowchart LR'));

        const mmdWs = await requestJson(baseUrl, '/api/memory/mermaid?workspace=sample_project');
        assert.strictEqual(mmdWs.status, 200);
        assert.strictEqual(mmdWs.data.ok, true);
        assert.ok(mmdWs.data.mermaid.includes('sample_project'));

        // 10. POST /api/memory/delete
        const delRes = await requestJson(baseUrl, '/api/memory/delete', {
            method: 'POST',
            body: {
                workspace: 'sample_project',
                id: createdId,
            },
        });
        assert.strictEqual(delRes.status, 200);
        assert.strictEqual(delRes.data.ok, true);

        // 11. GET /api/memory/pending, POST /api/memory/merge, POST /api/memory/workspace-sync
        const pendRes = await requestJson(baseUrl, '/api/memory/pending');
        assert.strictEqual(pendRes.status, 200);
        assert.strictEqual(pendRes.data.ok, true);

        const mergeRes = await requestJson(baseUrl, '/api/memory/merge', {
            method: 'POST',
            body: {},
        });
        assert.strictEqual(mergeRes.status, 200);
        assert.strictEqual(mergeRes.data.ok, true);

        const syncRes = await requestJson(baseUrl, '/api/memory/workspace-sync', {
            method: 'POST',
            body: {},
        });
        assert.strictEqual(syncRes.status, 200);
        assert.strictEqual(syncRes.data.ok, true);

        // 12. Verify deterministic pure-Node lexical fallback mode (forceLexical: true)
        const lexServer = createHubServer({ baseDir: tmpBase, forceLexical: true });
        await new Promise((resolve) => lexServer.listen(0, '127.0.0.1', resolve));
        const lexPort = lexServer.address().port;
        const lexUrl = `http://127.0.0.1:${lexPort}`;
        try {
            const lexSearch = await requestJson(lexUrl, '/api/memory/search', {
                method: 'POST',
                body: { query: 'token-bucket rate limiting', scope: 'global', top_k: 3 },
            });
            assert.strictEqual(lexSearch.status, 200);
            assert.strictEqual(lexSearch.data.ok, true);
            assert.strictEqual(lexSearch.data.backend, 'lexical');
            assert.ok(lexSearch.data.results.length >= 1);
        } finally {
            await new Promise((resolve) => lexServer.close(resolve));
        }

        console.log('✓ dashboard/test/smoke.js: all endpoints passed.');
    } finally {
        await new Promise((resolve) => server.close(resolve));
        fs.rmSync(tmpBase, { recursive: true, force: true });
    }
}

runSmokeTest().catch((err) => {
    console.error('✗ dashboard/test/smoke.js failed:', err);
    process.exit(1);
});
