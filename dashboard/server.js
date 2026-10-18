#!/usr/bin/env node
/**
 * Tree-RAG Memory MCP Hub — Standalone HTTP Server & API Bridge
 *
 * Zero runtime dependencies (Node.js 18+ built-ins only).
 * Reads the same on-disk layout as `ScopedMemoryStore` (`<base_dir>/<workspace>/docs.json` + `index.faiss`).
 *
 * Vector search & CRUD delegate to `dashboard/search_sidecar.py` when Python + `scoped_memory`
 * are available, and gracefully fall back to deterministic lexical scoring over `docs.json` otherwise.
 */

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const net = require('net');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

const ALLOWED_CATEGORIES = new Set([
    'code',
    'bug',
    'decision',
    'assumption',
    'note',
    'skill',
    'wiki',
]);

const CATEGORY_TO_LAYER = {
    decision: 'L3',
    skill: 'L3',
    code: 'L2',
    wiki: 'L2',
    bug: 'L1',
    assumption: 'L1',
    note: 'L0',
};

const CATEGORY_TO_PILLAR = {
    note: 'chat',
    assumption: 'chat',
    decision: 'skill',
    skill: 'skill',
    wiki: 'wiki',
    code: 'code',
    bug: 'code',
};

const LAYER_TITLES = {
    L3: 'L3 Persona / Decisions & Skills',
    L2: 'L2 Scenario / Code & Wiki',
    L1: 'L1 Atomic Facts / Bugs & Assumptions',
    L0: 'L0 Raw Context / Notes & Logs',
};

const IGNORED_DIRS = new Set(['_pending', 'backups', '_archived', '__pycache__']);

function parseCliArgs(argv) {
    const out = {
        port: Number(process.env.TREE_RAG_DASHBOARD_PORT || process.env.PORT || 8095),
        host: process.env.TREE_RAG_DASHBOARD_HOST || '127.0.0.1',
        baseDir: process.env.TREE_RAG_BASE_DIR || path.resolve(process.cwd(), '.treerag_data'),
        forceLexical: process.env.TREE_RAG_FORCE_LEXICAL === '1',
    };
    for (let i = 2; i < argv.length; i++) {
        const arg = argv[i];
        if ((arg === '--port' || arg === '-p') && argv[i + 1]) {
            out.port = Number(argv[++i]);
        } else if (arg.startsWith('--port=')) {
            out.port = Number(arg.split('=')[1]);
        } else if (arg === '--host' && argv[i + 1]) {
            out.host = argv[++i];
        } else if (arg.startsWith('--host=')) {
            out.host = arg.split('=')[1];
        } else if ((arg === '--base-dir' || arg === '-d') && argv[i + 1]) {
            out.baseDir = path.resolve(argv[++i]);
        } else if (arg.startsWith('--base-dir=')) {
            out.baseDir = path.resolve(arg.split('=')[1]);
        } else if (arg === '--lexical-only') {
            out.forceLexical = true;
        }
    }
    out.baseDir = path.resolve(out.baseDir);
    return out;
}

function sanitizeWorkspace(name) {
    const cleaned = String(name || 'default')
        .trim()
        .replace(/[^a-zA-Z0-9_-]+/g, '_')
        .replace(/^_+|_+$/g, '')
        .toLowerCase();
    return cleaned || 'default';
}

function extractTags(text, source, explicitTags) {
    const tags = [];
    const pushTag = (raw) => {
        const t = String(raw || '').trim().replace(/^#+/, '').toLowerCase();
        if (t && t.length >= 2 && !tags.includes(t)) {
            tags.push(t);
        }
    };
    if (Array.isArray(explicitTags)) {
        explicitTags.forEach(pushTag);
    } else if (typeof explicitTags === 'string' && explicitTags.trim()) {
        explicitTags.split(/[,;\s]+/).forEach(pushTag);
    }
    const hashMatches = String(text || '').match(/(?:^|\s)#([a-zA-Z0-9_-]{2,28})/g) || [];
    hashMatches.forEach((m) => pushTag(m.trim()));
    const bracketMatches = String(source || '').match(/\[([a-zA-Z0-9_-]{2,24})\]/g) || [];
    bracketMatches.forEach((m) => pushTag(m.slice(1, -1)));
    return tags.slice(0, 10);
}

function enrichDoc(doc, workspace) {
    const catRaw = String((doc && doc.category) || 'note').toLowerCase();
    const category = ALLOWED_CATEGORIES.has(catRaw) ? catRaw : 'note';
    const layer = (doc && doc.layer) || CATEGORY_TO_LAYER[category] || 'L0';
    const pillar = (doc && doc.pillar) || CATEGORY_TO_PILLAR[category] || 'chat';
    const text = String((doc && doc.text) || '');
    const source = String((doc && doc.source) || 'manual');
    const tags = Array.isArray(doc && doc.tags) && doc.tags.length
        ? doc.tags.map((t) => String(t))
        : extractTags(text, source, null);
    return {
        id: String((doc && doc.id) || ''),
        workspace,
        category,
        layer,
        pillar,
        source,
        text,
        time: String((doc && doc.time) || ''),
        updated_at: String((doc && doc.updated_at) || ''),
        tags,
    };
}

function formatNow() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function appendTelemetry(baseDir, entry) {
    try {
        fs.mkdirSync(baseDir, { recursive: true });
        const telemFile = path.join(baseDir, '_telemetry.jsonl');
        const payload = {
            ts: new Date().toISOString().slice(0, 19),
            op: entry.op || 'unknown',
            workspace: entry.workspace || 'global',
            category: entry.category || '',
            query: String(entry.query || '').slice(0, 160),
            hits: Number(entry.hits || 0),
            top_score: Number(entry.top_score || 0),
            latency_ms: Number(entry.latency_ms || 0),
            status: entry.status || 'ok',
            client: entry.client || 'hub-ui',
        };
        fs.appendFileSync(telemFile, JSON.stringify(payload) + '\n', 'utf8');
    } catch (_) {
        // Ignore telemetry write errors on read-only filesystems
    }
}

function discoverWorkspaces(baseDir) {
    const workspaces = [];
    if (!fs.existsSync(baseDir)) return workspaces;
    const seenInodes = new Set();
    const entries = fs.readdirSync(baseDir, { withFileTypes: true })
        .sort((a, b) => a.name.localeCompare(b.name));
    for (const ent of entries) {
        if (!ent.isDirectory() || ent.isSymbolicLink()) continue;
        if (IGNORED_DIRS.has(ent.name) || ent.name.startsWith('.')) continue;
        const wsDir = path.join(baseDir, ent.name);
        const docsPath = path.join(wsDir, 'docs.json');
        if (!fs.existsSync(docsPath)) continue;
        try {
            const st = fs.statSync(wsDir);
            if (st.ino && seenInodes.has(st.ino)) continue;
            if (st.ino) seenInodes.add(st.ino);
        } catch (_) {}
        workspaces.push(ent.name);
    }
    return workspaces;
}

function loadWorkspaceDocs(baseDir, wsName) {
    const docsPath = path.join(baseDir, wsName, 'docs.json');
    if (!fs.existsSync(docsPath)) return [];
    try {
        const parsed = JSON.parse(fs.readFileSync(docsPath, 'utf8'));
        return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
        return [];
    }
}

function writeWorkspaceDocsAtomic(baseDir, wsName, docs) {
    const wsDir = path.join(baseDir, wsName);
    fs.mkdirSync(wsDir, { recursive: true });
    const docsPath = path.join(wsDir, 'docs.json');
    const tmpPath = path.join(wsDir, `.docs.json.tmp.${process.pid}.${Date.now()}`);
    fs.writeFileSync(tmpPath, JSON.stringify(docs, null, 2), 'utf8');
    fs.renameSync(tmpPath, docsPath);
}

function scanPendingDir(baseDir) {
    const pendingDir = path.join(baseDir, '_pending');
    const byWs = {};
    const items = [];
    let total = 0;
    if (!fs.existsSync(pendingDir)) {
        return { total, byWorkspace: byWs, items };
    }
    try {
        const subdirs = fs.readdirSync(pendingDir, { withFileTypes: true });
        for (const d of subdirs) {
            if (!d.isDirectory() || d.isSymbolicLink()) continue;
            const wsPending = path.join(pendingDir, d.name);
            const files = fs.readdirSync(wsPending).filter((f) => f.endsWith('.md')).sort();
            if (files.length > 0) {
                byWs[d.name] = files.length;
                total += files.length;
                for (const fname of files.slice(0, 50)) {
                    const fpath = path.join(wsPending, fname);
                    let preview = '';
                    try {
                        preview = fs.readFileSync(fpath, 'utf8').slice(0, 300);
                    } catch (_) {}
                    items.push({ workspace: d.name, file: fname, path: fpath, preview });
                }
            }
        }
    } catch (_) {}
    return { total, byWorkspace: byWs, items };
}

function checkTcpPort(host, port, timeoutMs = 600) {
    return new Promise((resolve) => {
        if (!host || !port) return resolve(false);
        const sock = new net.Socket();
        let done = false;
        const finish = (ok) => {
            if (done) return;
            done = true;
            sock.destroy();
            resolve(ok);
        };
        sock.setTimeout(timeoutMs);
        sock.once('connect', () => finish(true));
        sock.once('timeout', () => finish(false));
        sock.once('error', () => finish(false));
        sock.connect(port, host);
    });
}

async function probeEndpointUrl(rawUrl) {
    if (!rawUrl || !String(rawUrl).trim()) {
        return { configured: false, reachable: false, endpoint: '' };
    }
    try {
        const u = new URL(rawUrl.includes('://') ? rawUrl : `http://${rawUrl}`);
        const host = u.hostname || '127.0.0.1';
        const port = Number(u.port || (u.protocol === 'https:' ? 443 : 80));
        const reachable = await checkTcpPort(host, port);
        return { configured: true, reachable, endpoint: rawUrl };
    } catch (_) {
        return { configured: true, reachable: false, endpoint: rawUrl };
    }
}

// --- Sidecar execution & lexical fallback ---

const SIDECAR_PATH = path.join(__dirname, 'search_sidecar.py');
let cachedPythonCmd = undefined;

function detectPythonSidecar(forceLexical) {
    if (forceLexical) return null;
    if (cachedPythonCmd !== undefined) return cachedPythonCmd;
    const candidates = [
        process.env.TREE_RAG_PYTHON,
        'python3',
        'python',
    ].filter(Boolean);
    for (const cmd of candidates) {
        try {
            const res = spawnSync(
                cmd,
                ['-c', 'import sys; sys.path.insert(0, "."); import scoped_memory; print("OK")'],
                {
                    cwd: path.resolve(__dirname, '..'),
                    encoding: 'utf8',
                    timeout: 3000,
                }
            );
            if (res.status === 0 && String(res.stdout || '').includes('OK')) {
                cachedPythonCmd = cmd;
                return cmd;
            }
        } catch (_) {}
    }
    cachedPythonCmd = null;
    return null;
}

function callPythonSidecar(payload, forceLexical = false) {
    const py = detectPythonSidecar(forceLexical);
    if (!py || !fs.existsSync(SIDECAR_PATH)) {
        return null;
    }
    try {
        const res = spawnSync(py, [SIDECAR_PATH], {
            cwd: path.resolve(__dirname, '..'),
            input: JSON.stringify(payload) + '\n',
            encoding: 'utf8',
            timeout: 15000,
            env: {
                ...process.env,
                TREE_RAG_BASE_DIR: payload.base_dir || process.env.TREE_RAG_BASE_DIR || '',
            },
        });
        if (res.status !== 0 || !res.stdout) {
            return null;
        }
        const lines = res.stdout.trim().split('\n').filter(Boolean);
        if (!lines.length) return null;
        return JSON.parse(lines[lines.length - 1]);
    } catch (_) {
        return null;
    }
}

function tokenizeFeatures(text) {
    const cleaned = String(text || '')
        .toLowerCase()
        .replace(/[^a-z0-9_]+/g, ' ')
        .trim();
    const tokens = cleaned.split(/\s+/).filter((t) => t.length >= 2);
    const weights = new Map();
    const addFeat = (f, w) => weights.set(f, (weights.get(f) || 0) + w);
    for (const tok of tokens) {
        addFeat(`w:${tok}`, 2.0);
        if (tok.length >= 4) addFeat(`p:${tok.slice(0, 4)}`, 1.0);
        for (let i = 0; i <= tok.length - 3; i++) {
            addFeat(`g:${tok.slice(i, i + 3)}`, 0.5);
        }
    }
    return weights;
}

function lexicalSimilarity(queryText, docText) {
    const qMap = tokenizeFeatures(queryText);
    const dMap = tokenizeFeatures(docText);
    if (!qMap.size || !dMap.size) return 0;
    let dot = 0;
    let qNorm = 0;
    let dNorm = 0;
    for (const [k, v] of qMap.entries()) {
        qNorm += v * v;
        if (dMap.has(k)) {
            dot += v * dMap.get(k);
        }
    }
    for (const v of dMap.values()) {
        dNorm += v * v;
    }
    const denom = Math.sqrt(qNorm) * Math.sqrt(dNorm);
    let score = denom > 0 ? dot / denom : 0;
    const qLower = String(queryText || '').toLowerCase().trim();
    const dLower = String(docText || '').toLowerCase();
    if (qLower.length >= 3 && dLower.includes(qLower)) {
        score = Math.min(0.999, score + 0.25);
    }
    return Number(score.toFixed(4));
}

function runLexicalSearch(baseDir, payload) {
    const t0 = Date.now();
    const query = String(payload.query || '').trim();
    if (!query) {
        return { ok: false, error: 'Search query cannot be empty.' };
    }
    const scope = String(payload.scope || 'global').toLowerCase() === 'global' ? 'global' : 'workspace';
    const wsFilter = scope === 'workspace' ? sanitizeWorkspace(payload.workspace || 'default') : '';
    const catFilter = String(payload.category || '').trim().toLowerCase();
    const topK = Math.max(1, Math.min(30, Number(payload.top_k || 8)));
    const mode = String(payload.mode || 'base').toLowerCase();

    const targetWorkspaces = wsFilter ? [wsFilter] : discoverWorkspaces(baseDir);
    const scored = [];
    for (const ws of targetWorkspaces) {
        const docs = loadWorkspaceDocs(baseDir, ws);
        for (const d of docs) {
            const ed = enrichDoc(d, ws);
            if (catFilter && ed.category !== catFilter) continue;
            const searchable = `${ed.text} ${ed.source} ${(ed.tags || []).join(' ')}`;
            const score = lexicalSimilarity(query, searchable);
            if (score > 0.01) {
                scored.push({ ...ed, score });
            }
        }
    }
    scored.sort((a, b) => b.score - a.score);
    const results = scored.slice(0, topK);

    const hlWorkspaces = [];
    const hlCategories = [];
    const hlSources = [];
    const hlIds = [];
    for (const r of results) {
        if (!hlWorkspaces.includes(r.workspace)) hlWorkspaces.push(r.workspace);
        const ck = `${r.workspace}::${r.category}`;
        if (!hlCategories.includes(ck)) hlCategories.push(ck);
        const sk = `${r.workspace}::${r.category}::${r.source}`;
        if (!hlSources.includes(sk)) hlSources.push(sk);
        if (r.id && !hlIds.includes(r.id)) hlIds.push(r.id);
    }

    const latencyMs = Date.now() - t0;
    appendTelemetry(baseDir, {
        op: 'memory_search',
        workspace: wsFilter || 'global',
        category: catFilter,
        query,
        hits: results.length,
        top_score: results[0] ? results[0].score : 0,
        latency_ms: latencyMs,
        status: results.length ? 'ok' : 'empty',
    });

    return {
        ok: true,
        backend: 'lexical',
        mode,
        query,
        scope,
        workspace: wsFilter || 'global',
        count: results.length,
        latency_ms: latencyMs,
        results,
        highlights: {
            workspaces: hlWorkspaces,
            categories: hlCategories,
            sources: hlSources,
            ids: hlIds,
        },
        warnings: [],
    };
}

function runLexicalNeighbors(baseDir, payload) {
    const t0 = Date.now();
    const workspace = sanitizeWorkspace(payload.workspace || '');
    const docId = String(payload.id || payload.doc_id || '').trim();
    const topK = Math.max(1, Math.min(15, Number(payload.top_k || 6)));
    if (!workspace || !docId) {
        return { ok: false, error: "Fields 'workspace' and 'id' are required." };
    }
    const docs = loadWorkspaceDocs(baseDir, workspace);
    const targetRaw = docs.find((d) => d.id === docId);
    if (!targetRaw) {
        return { ok: false, error: `Document ${docId} not found in workspace [${workspace}].` };
    }
    const target = enrichDoc(targetRaw, workspace);
    const sameSource = [];
    const candidates = [];
    for (const d of docs) {
        if (d.id === docId) continue;
        const ed = enrichDoc(d, workspace);
        if (ed.source === target.source) {
            sameSource.push(ed);
        }
        const sim = lexicalSimilarity(target.text, ed.text);
        candidates.push({ ...ed, score: sim });
    }
    candidates.sort((a, b) => b.score - a.score);
    return {
        ok: TrueBoolean(true),
        backend: 'lexical',
        target,
        same_source: sameSource.slice(0, 15),
        semantic_neighbors: candidates.slice(0, topK),
        latency_ms: Date.now() - t0,
    };
}

function TrueBoolean(v) {
    return Boolean(v);
}

function runLexicalSave(baseDir, payload) {
    const t0 = Date.now();
    const workspace = sanitizeWorkspace(payload.workspace || 'default');
    const catRaw = String(payload.category || 'note').trim().toLowerCase();
    const category = ALLOWED_CATEGORIES.has(catRaw) ? catRaw : 'note';
    const text = String(payload.text || '').trim();
    const source = String(payload.source || 'hub-ui').trim() || 'hub-ui';
    if (!text) {
        return { ok: false, error: "Field 'text' cannot be empty." };
    }
    const docs = loadWorkspaceDocs(baseDir, workspace);
    const hash = crypto.createHash('sha256').update(text).digest('hex');
    for (const existing of docs) {
        const eh = crypto.createHash('sha256').update(String(existing.text || '').trim()).digest('hex');
        if (eh === hash) {
            const ed = enrichDoc(existing, workspace);
            return {
                ok: true,
                status: 'duplicate',
                id: ed.id,
                workspace,
                category: ed.category,
                layer: ed.layer,
                pillar: ed.pillar,
                doc: ed,
                message: `Duplicate — entry already exists in [${workspace}] (id=${ed.id}).`,
            };
        }
    }
    const newDoc = {
        id: 'mem_' + crypto.randomBytes(6).toString('hex'),
        text,
        source,
        category,
        layer: CATEGORY_TO_LAYER[category] || 'L0',
        pillar: CATEGORY_TO_PILLAR[category] || 'chat',
        tags: extractTags(text, source, payload.tags),
        time: formatNow(),
    };
    docs.push(newDoc);
    writeWorkspaceDocsAtomic(baseDir, workspace, docs);
    appendTelemetry(baseDir, {
        op: 'memory_save',
        workspace,
        category,
        query: text.slice(0, 120),
        hits: 1,
        latency_ms: Date.now() - t0,
        status: 'saved',
    });
    return {
        ok: true,
        status: 'saved',
        id: newDoc.id,
        workspace,
        category: newDoc.category,
        layer: newDoc.layer,
        pillar: newDoc.pillar,
        doc: enrichDoc(newDoc, workspace),
        message: `Saved to [${workspace}] (id=${newDoc.id}, category=${newDoc.category}, layer=${newDoc.layer}).`,
    };
}

function runLexicalUpdate(baseDir, payload) {
    const workspace = sanitizeWorkspace(payload.workspace || '');
    const docId = String(payload.id || '').trim();
    if (!workspace || !docId) {
        return { ok: false, error: "Fields 'workspace' and 'id' are required." };
    }
    const docs = loadWorkspaceDocs(baseDir, workspace);
    const idx = docs.findIndex((d) => d.id === docId);
    if (idx < 0) {
        return { ok: false, error: `Document id=${docId} not found in ${workspace}` };
    }
    const cur = docs[idx];
    const newText = payload.text !== undefined ? String(payload.text).trim() : String(cur.text || '');
    if (!newText) {
        return { ok: false, error: 'Document text cannot be empty' };
    }
    const catRaw = String(payload.category || cur.category || 'note').toLowerCase();
    const newCat = ALLOWED_CATEGORIES.has(catRaw) ? catRaw : 'note';
    const newSrc = payload.source !== undefined ? String(payload.source).trim() : String(cur.source || 'manual');
    cur.text = newText;
    cur.category = newCat;
    cur.layer = CATEGORY_TO_LAYER[newCat] || 'L0';
    cur.pillar = CATEGORY_TO_PILLAR[newCat] || 'chat';
    cur.source = newSrc;
    cur.tags = extractTags(newText, newSrc, payload.tags !== undefined ? payload.tags : cur.tags);
    cur.updated_at = formatNow();
    writeWorkspaceDocsAtomic(baseDir, workspace, docs);
    return {
        ok: true,
        doc: enrichDoc(cur, workspace),
        message: `Updated record ${docId} in [${workspace}].`,
    };
}

function runLexicalDelete(baseDir, payload) {
    const workspace = sanitizeWorkspace(payload.workspace || '');
    const docId = String(payload.id || '').trim();
    if (!workspace || !docId) {
        return { ok: false, error: "Fields 'workspace' and 'id' are required." };
    }
    const docs = loadWorkspaceDocs(baseDir, workspace);
    const idx = docs.findIndex((d) => d.id === docId);
    if (idx < 0) {
        return { ok: false, error: `Document id=${docId} not found in ${workspace}` };
    }
    const [removed] = docs.splice(idx, 1);
    writeWorkspaceDocsAtomic(baseDir, workspace, docs);
    return {
        ok: true,
        deleted: removed,
        remaining: docs.length,
    };
}

function runLexicalMerge(baseDir) {
    const pendingDir = path.join(baseDir, '_pending');
    let merged = 0;
    if (fs.existsSync(pendingDir)) {
        const subdirs = fs.readdirSync(pendingDir, { withFileTypes: true });
        for (const d of subdirs) {
            if (!d.isDirectory() || d.isSymbolicLink()) continue;
            const wsName = sanitizeWorkspace(d.name);
            const wsPending = path.join(pendingDir, d.name);
            const files = fs.readdirSync(wsPending).filter((f) => f.endsWith('.md')).sort();
            for (const f of files) {
                const fpath = path.join(wsPending, f);
                try {
                    const raw = fs.readFileSync(fpath, 'utf8');
                    let category = 'note';
                    let source = `pending:${f}`;
                    let body = raw.trim();
                    const m = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
                    if (m) {
                        const front = m[1];
                        body = m[2].trim();
                        for (const line of front.split('\n')) {
                            const sep = line.indexOf(':');
                            if (sep > 0) {
                                const k = line.slice(0, sep).trim();
                                const v = line.slice(sep + 1).trim();
                                if (k === 'category') category = v;
                                if (k === 'source') source = v;
                            }
                        }
                    }
                    if (body) {
                        runLexicalSave(baseDir, { workspace: wsName, category, text: body, source });
                        merged++;
                    }
                    fs.unlinkSync(fpath);
                } catch (_) {}
            }
        }
    }
    return {
        ok: true,
        merged,
        deduped: 0,
        remaining: scanPendingDir(baseDir).total,
        output: `Merged ${merged} pending files.`,
    };
}

function cleanMermaidText(text, maxLen = 48) {
    const cleaned = String(text || '')
        .replace(/["[\]{}()<>|`#&;\\]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    if (cleaned.length > maxLen) return cleaned.slice(0, maxLen - 1) + '…';
    return cleaned || 'entry';
}

function buildMermaidFallback(baseDir, workspace) {
    const ws = String(workspace || 'global').trim();
    if (ws && ws.toLowerCase() !== 'global') {
        const docs = loadWorkspaceDocs(baseDir, ws).map((d) => enrichDoc(d, ws));
        const lines = [
            'flowchart LR',
            `    WS["Workspace: ${ws} (${docs.length} docs)"]`,
        ];
        const byLayer = { L3: [], L2: [], L1: [], L0: [] };
        for (const d of docs) {
            if (!byLayer[d.layer]) byLayer[d.layer] = [];
            byLayer[d.layer].push(d);
        }
        for (const lyr of ['L3', 'L2', 'L1', 'L0']) {
            const arr = byLayer[lyr] || [];
            if (!arr.length) continue;
            const lid = `LYR_${lyr}`;
            lines.push(`    ${lid}["${LAYER_TITLES[lyr]} (${arr.length})"]`);
            lines.push(`    WS --> ${lid}`);
            arr.slice(-5).forEach((item, idx) => {
                const nid = `N_${lyr}_${idx}`;
                const shortId = item.id ? item.id.slice(-6) : `idx${idx}`;
                lines.push(`    ${nid}["[${shortId}] ${cleanMermaidText(item.text, 44)}<br/>src: ${cleanMermaidText(item.source, 20)}"]`);
                lines.push(`    ${lid} --> ${nid}`);
            });
        }
        return { ok: true, workspace: ws, mermaid: lines.join('\n'), docs_count: docs.length };
    }

    const wsNames = discoverWorkspaces(baseDir);
    const wsStats = wsNames.map((wn) => {
        const docs = loadWorkspaceDocs(baseDir, wn);
        const cats = {};
        for (const d of docs) {
            const c = String(d.category || 'note').toLowerCase();
            cats[c] = (cats[c] || 0) + 1;
        }
        return { name: wn, count: docs.length, cats };
    }).sort((a, b) => b.count - a.count);

    const totalDocs = wsStats.reduce((acc, x) => acc + x.count, 0);
    const lines = [
        'flowchart LR',
        `    ROOT["Global Memory Hub (${wsStats.length} workspaces / ${totalDocs} docs)"]`,
    ];
    wsStats.slice(0, 12).forEach((w, idx) => {
        const wid = `WS_${idx}`;
        const catStr = Object.entries(w.cats).map(([k, v]) => `${k}:${v}`).join(', ') || 'empty';
        lines.push(`    ${wid}["${w.name} (${w.count})<br/>${catStr}"]`);
        lines.push(`    ROOT --> ${wid}`);
    });
    return { ok: true, workspace: 'global', mermaid: lines.join('\n'), docs_count: totalDocs };
}

function readJsonBody(req) {
    return new Promise((resolve, reject) => {
        let body = '';
        req.on('data', (chunk) => {
            body += chunk;
            if (body.length > 2 * 1024 * 1024) {
                reject(new Error('Payload too large'));
            }
        });
        req.on('end', () => {
            if (!body.trim()) return resolve({});
            try {
                resolve(JSON.parse(body));
            } catch (err) {
                reject(new Error('Invalid JSON payload'));
            }
        });
        req.on('error', reject);
    });
}

function sendJson(res, statusCode, payload) {
    res.writeHead(statusCode, {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store',
    });
    res.end(JSON.stringify(payload));
}

function createHubServer(options = {}) {
    const baseDir = path.resolve(
        options.baseDir || process.env.TREE_RAG_BASE_DIR || path.resolve(process.cwd(), '.treerag_data')
    );
    const forceLexical = Boolean(options.forceLexical || process.env.TREE_RAG_FORCE_LEXICAL === '1');
    fs.mkdirSync(path.join(baseDir, '_pending'), { recursive: true });

    const server = http.createServer(async (req, res) => {
        try {
            const parsedUrl = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
            const pathname = parsedUrl.pathname;

            // Static files
            if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
                const htmlPath = path.join(__dirname, 'index.html');
                res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
                res.end(fs.readFileSync(htmlPath, 'utf8'));
                return;
            }
            if (req.method === 'GET' && pathname === '/app.js') {
                const jsPath = path.join(__dirname, 'app.js');
                res.writeHead(200, { 'Content-Type': 'application/javascript; charset=utf-8' });
                res.end(fs.readFileSync(jsPath, 'utf8'));
                return;
            }
            if (req.method === 'GET' && pathname === '/style.css') {
                const cssPath = path.join(__dirname, 'style.css');
                res.writeHead(200, { 'Content-Type': 'text/css; charset=utf-8' });
                res.end(fs.readFileSync(cssPath, 'utf8'));
                return;
            }

            // GET /api/memory/status & GET /api/memory/health
            if ((pathname === '/api/memory/status' || pathname === '/api/memory/health') && req.method === 'GET') {
                const wsNames = discoverWorkspaces(baseDir);
                const workspaces = [];
                let totalDocs = 0;
                let totalIndexBytes = 0;

                for (const name of wsNames) {
                    const wsDir = path.join(baseDir, name);
                    const docsPath = path.join(wsDir, 'docs.json');
                    const faissPath = path.join(wsDir, 'index.faiss');
                    const docs = loadWorkspaceDocs(baseDir, name);
                    const indexBytes = fs.existsSync(faissPath) ? fs.statSync(faissPath).size : 0;
                    const docsBytes = fs.existsSync(docsPath) ? fs.statSync(docsPath).size : 0;
                    const mtime = fs.existsSync(docsPath) ? fs.statSync(docsPath).mtime.toISOString() : null;
                    const cats = {};
                    for (const d of docs) {
                        const c = String(d.category || 'note').toLowerCase();
                        cats[c] = (cats[c] || 0) + 1;
                    }
                    totalDocs += docs.length;
                    totalIndexBytes += indexBytes;
                    workspaces.push({
                        name,
                        docs: docs.length,
                        indexBytes,
                        docsBytes,
                        mtime,
                        categories: cats,
                    });
                }
                workspaces.sort((a, b) => b.docs - a.docs);

                const pending = scanPendingDir(baseDir);
                const embedUrl = process.env.TREE_RAG_EMBED_URL || '';
                const embedPlusUrl = process.env.TREE_RAG_EMBED_URL_PLUS || '';
                const [baseProbe, plusProbe] = await Promise.all([
                    probeEndpointUrl(embedUrl),
                    probeEndpointUrl(embedPlusUrl),
                ]);
                const sidecarCmd = detectPythonSidecar(forceLexical);
                const primaryWs = workspaces[0] || { name: 'default', docs: 0, indexBytes: 0 };

                sendJson(res, 200, {
                    ok: true,
                    basePath: baseDir,
                    backend: sidecarCmd ? 'faiss-sidecar' : 'lexical-fallback',
                    totalDocs,
                    totalIndexBytes,
                    workspaces,
                    primaryWorkspace: primaryWs,
                    pendingTotal: pending.total,
                    pendingByWorkspace: pending.byWorkspace,
                    embedUrl: embedUrl || 'MockEmbedder (local CPU)',
                    embedPlusUrl: embedPlusUrl || '',
                    embedReachable: baseProbe.configured ? baseProbe.reachable : true,
                    embedPlusReachable: plusProbe.configured ? plusProbe.reachable : false,
                    concurrentActive: false,
                    concurrencyReason: 'idle (direct FAISS writes enabled)',
                });
                return;
            }

            // GET /api/memory/pending
            if (pathname === '/api/memory/pending' && req.method === 'GET') {
                const pending = scanPendingDir(baseDir);
                sendJson(res, 200, { ok: true, count: pending.total, byWorkspace: pending.byWorkspace, items: pending.items });
                return;
            }

            // POST /api/memory/merge
            if (pathname === '/api/memory/merge' && req.method === 'POST') {
                const body = await readJsonBody(req);
                const sidecarRes = callPythonSidecar({ op: 'merge', base_dir: baseDir, ...body }, forceLexical);
                const out = sidecarRes || runLexicalMerge(baseDir);
                sendJson(res, 200, out);
                return;
            }

            // GET /api/memory/logs
            if (pathname === '/api/memory/logs' && req.method === 'GET') {
                const telemFile = path.join(baseDir, '_telemetry.jsonl');
                const entries = [];
                if (fs.existsSync(telemFile)) {
                    const lines = fs.readFileSync(telemFile, 'utf8').trim().split('\n').filter(Boolean);
                    for (const line of lines.slice(-60).reverse()) {
                        try {
                            entries.push(JSON.parse(line));
                        } catch (_) {}
                    }
                }
                sendJson(res, 200, { ok: true, entries });
                return;
            }

            // GET /api/memory/tree
            if (pathname === '/api/memory/tree' && req.method === 'GET') {
                const wsNames = discoverWorkspaces(baseDir);
                const globalCategories = {
                    decision: 0,
                    skill: 0,
                    code: 0,
                    wiki: 0,
                    bug: 0,
                    assumption: 0,
                    note: 0,
                };
                const globalLayers = { L3: 0, L2: 0, L1: 0, L0: 0 };
                const globalPillars = { chat: 0, skill: 0, wiki: 0, code: 0 };
                const globalTags = {};
                const timelineByDay = {};
                const workspaceNodes = [];
                let totalDocs = 0;
                let totalIndexBytes = 0;

                for (const name of wsNames) {
                    const wsDir = path.join(baseDir, name);
                    const docsPath = path.join(wsDir, 'docs.json');
                    const faissPath = path.join(wsDir, 'index.faiss');
                    const indexBytes = fs.existsSync(faissPath) ? fs.statSync(faissPath).size : 0;
                    const docsBytes = fs.existsSync(docsPath) ? fs.statSync(docsPath).size : 0;
                    const mtime = fs.existsSync(docsPath) ? fs.statSync(docsPath).mtime.toISOString() : null;
                    const rawDocs = loadWorkspaceDocs(baseDir, name);

                    const catMap = {};
                    const layerCounts = { L3: 0, L2: 0, L1: 0, L0: 0 };
                    const pillarCounts = { chat: 0, skill: 0, wiki: 0, code: 0 };
                    const wsTags = {};
                    let latestTime = '';

                    for (const raw of rawDocs) {
                        const ed = enrichDoc(raw, name);
                        globalCategories[ed.category] = (globalCategories[ed.category] || 0) + 1;
                        globalLayers[ed.layer] = (globalLayers[ed.layer] || 0) + 1;
                        layerCounts[ed.layer] = (layerCounts[ed.layer] || 0) + 1;
                        globalPillars[ed.pillar] = (globalPillars[ed.pillar] || 0) + 1;
                        pillarCounts[ed.pillar] = (pillarCounts[ed.pillar] || 0) + 1;

                        for (const tg of ed.tags) {
                            globalTags[tg] = (globalTags[tg] || 0) + 1;
                            wsTags[tg] = (wsTags[tg] || 0) + 1;
                        }

                        if (ed.time && ed.time > latestTime) latestTime = ed.time;
                        const day = ed.time ? ed.time.slice(0, 10) : '';
                        if (/^\d{4}-\d{2}-\d{2}$/.test(day)) {
                            timelineByDay[day] = (timelineByDay[day] || 0) + 1;
                        }

                        if (!catMap[ed.category]) {
                            catMap[ed.category] = { count: 0, layer: ed.layer, pillar: ed.pillar, sources: {} };
                        }
                        catMap[ed.category].count++;
                        catMap[ed.category].sources[ed.source] = (catMap[ed.category].sources[ed.source] || 0) + 1;
                    }

                    const categoryChildren = Object.entries(catMap)
                        .sort((a, b) => b[1].count - a[1].count)
                        .map(([catName, info]) => ({
                            name: catName,
                            type: 'category',
                            layer: info.layer,
                            pillar: info.pillar,
                            count: info.count,
                            children: Object.entries(info.sources)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 25)
                                .map(([srcName, cnt]) => ({
                                    name: srcName,
                                    type: 'source',
                                    count: cnt,
                                })),
                        }));

                    totalDocs += rawDocs.length;
                    totalIndexBytes += indexBytes;

                    workspaceNodes.push({
                        name,
                        type: 'workspace',
                        count: rawDocs.length,
                        indexBytes,
                        docsBytes,
                        mtime,
                        latestTime,
                        layers: layerCounts,
                        pillars: pillarCounts,
                        topTags: Object.entries(wsTags)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 12)
                            .map(([tag, count]) => ({ tag, count })),
                        children: categoryChildren,
                    });
                }

                workspaceNodes.sort((a, b) => b.count - a.count);
                const pending = scanPendingDir(baseDir);

                // Telemetry summary
                const telemFile = path.join(baseDir, '_telemetry.jsonl');
                const recentTelemetry = [];
                const opCounts = {};
                let totalLatencyMs = 0;
                let latencySamples = 0;
                let calls24h = 0;
                const cutoff24h = Date.now() - 24 * 3600 * 1000;

                if (fs.existsSync(telemFile)) {
                    try {
                        const lines = fs.readFileSync(telemFile, 'utf8').trim().split('\n').filter(Boolean);
                        for (const l of lines.slice(-300)) {
                            try {
                                const obj = JSON.parse(l);
                                opCounts[obj.op] = (opCounts[obj.op] || 0) + 1;
                                if (typeof obj.latency_ms === 'number') {
                                    totalLatencyMs += obj.latency_ms;
                                    latencySamples++;
                                }
                                if (obj.ts) {
                                    const tsMs = Date.parse(obj.ts);
                                    if (!Number.isNaN(tsMs) && tsMs >= cutoff24h) calls24h++;
                                }
                                recentTelemetry.push(obj);
                            } catch (_) {}
                        }
                    } catch (_) {}
                }
                recentTelemetry.reverse();

                const timeline = Object.entries(timelineByDay)
                    .sort((a, b) => a[0].localeCompare(b[0]))
                    .slice(-21)
                    .map(([date, count]) => ({ date, count }));

                const topTags = Object.entries(globalTags)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 30)
                    .map(([tag, count]) => ({ tag, count }));

                sendJson(res, 200, {
                    ok: true,
                    basePath: baseDir,
                    summary: {
                        totalDocs,
                        totalWorkspaces: workspaceNodes.length,
                        totalIndexBytes,
                        pendingTotal: pending.total,
                        categories: globalCategories,
                        layers: globalLayers,
                        pillars: globalPillars,
                        topTags,
                        timeline,
                    },
                    telemetry: {
                        totalCalls: Object.values(opCounts).reduce((a, b) => a + b, 0),
                        calls24h,
                        avgLatencyMs: latencySamples ? Number((totalLatencyMs / latencySamples).toFixed(1)) : 0,
                        opCounts,
                        recentCalls: recentTelemetry.slice(0, 35),
                    },
                    tree: {
                        name: 'global',
                        type: 'root',
                        count: totalDocs,
                        children: workspaceNodes,
                    },
                    flatWorkspaces: workspaceNodes.map((w) => ({
                        name: w.name,
                        count: w.count,
                        indexBytes: w.indexBytes,
                        layers: w.layers,
                        pillars: w.pillars,
                    })),
                });
                return;
            }

            // GET /api/memory/records
            if (pathname === '/api/memory/records' && req.method === 'GET') {
                const wsFilter = (parsedUrl.searchParams.get('workspace') || '').trim();
                const catFilter = (parsedUrl.searchParams.get('category') || '').trim().toLowerCase();
                const srcFilter = (parsedUrl.searchParams.get('source') || '').trim();
                const layerFilter = (parsedUrl.searchParams.get('layer') || '').trim().toUpperCase();
                const pillarFilter = (parsedUrl.searchParams.get('pillar') || '').trim().toLowerCase();
                const tagFilter = (parsedUrl.searchParams.get('tag') || '').trim().toLowerCase();
                const q = (parsedUrl.searchParams.get('q') || '').trim().toLowerCase();
                const limit = Math.max(1, Math.min(250, parseInt(parsedUrl.searchParams.get('limit') || '60', 10)));
                const offset = Math.max(0, parseInt(parsedUrl.searchParams.get('offset') || '0', 10));

                const targetWorkspaces = (wsFilter && wsFilter !== 'global')
                    ? [wsFilter]
                    : discoverWorkspaces(baseDir);

                const matched = [];
                for (const wsName of targetWorkspaces) {
                    const docs = loadWorkspaceDocs(baseDir, wsName);
                    for (let i = docs.length - 1; i >= 0; i--) {
                        const ed = enrichDoc(docs[i], wsName);
                        if (catFilter && ed.category !== catFilter) continue;
                        if (srcFilter && ed.source !== srcFilter) continue;
                        if (layerFilter && ed.layer !== layerFilter) continue;
                        if (pillarFilter && ed.pillar !== pillarFilter) continue;
                        if (tagFilter && !ed.tags.includes(tagFilter)) continue;
                        if (q) {
                            const hay = `${ed.text} ${ed.source} ${ed.category} ${ed.workspace} ${ed.tags.join(' ')}`.toLowerCase();
                            if (!hay.includes(q)) continue;
                        }
                        matched.push(ed);
                    }
                }

                matched.sort((a, b) => (b.time || '').localeCompare(a.time || ''));
                const total = matched.length;
                const page = matched.slice(offset, offset + limit);

                sendJson(res, 200, {
                    ok: true,
                    total,
                    offset,
                    limit,
                    records: page,
                });
                return;
            }

            // POST /api/memory/search
            if (pathname === '/api/memory/search' && req.method === 'POST') {
                const payload = await readJsonBody(req);
                const sidecarRes = callPythonSidecar({ op: 'search', base_dir: baseDir, ...payload }, forceLexical);
                const out = sidecarRes || runLexicalSearch(baseDir, payload);
                sendJson(res, 200, out);
                return;
            }

            // POST /api/memory/neighbors
            if (pathname === '/api/memory/neighbors' && req.method === 'POST') {
                const payload = await readJsonBody(req);
                const sidecarRes = callPythonSidecar({ op: 'neighbors', base_dir: baseDir, ...payload }, forceLexical);
                const out = sidecarRes || runLexicalNeighbors(baseDir, payload);
                sendJson(res, 200, out);
                return;
            }

            // POST /api/memory/save | /api/memory/update | /api/memory/delete
            if ((pathname === '/api/memory/save' || pathname === '/api/memory/update' || pathname === '/api/memory/delete') && req.method === 'POST') {
                const op = pathname.split('/').pop();
                const payload = await readJsonBody(req);
                const sidecarRes = callPythonSidecar({ op, base_dir: baseDir, ...payload }, forceLexical);
                let out = sidecarRes;
                if (!out) {
                    if (op === 'save') out = runLexicalSave(baseDir, payload);
                    else if (op === 'update') out = runLexicalUpdate(baseDir, payload);
                    else if (op === 'delete') out = runLexicalDelete(baseDir, payload);
                }
                sendJson(res, 200, out);
                return;
            }

            // GET /api/memory/mermaid
            if (pathname === '/api/memory/mermaid' && req.method === 'GET') {
                const ws = (parsedUrl.searchParams.get('workspace') || 'global').trim();
                const sidecarRes = callPythonSidecar({ op: 'mermaid', base_dir: baseDir, workspace: ws }, forceLexical);
                const out = sidecarRes || buildMermaidFallback(baseDir, ws);
                sendJson(res, 200, out);
                return;
            }

            // POST /api/memory/workspace-sync
            if (pathname === '/api/memory/workspace-sync' && req.method === 'POST') {
                const sidecarRes = callPythonSidecar({ op: 'workspace-sync', base_dir: baseDir }, forceLexical);
                if (sidecarRes) {
                    sendJson(res, 200, sidecarRes);
                } else {
                    const wsNames = discoverWorkspaces(baseDir);
                    sendJson(res, 200, {
                        ok: true,
                        workspaces_synced: wsNames.length,
                        output: `Verified ${wsNames.length} workspace branches in ${baseDir}.`,
                    });
                }
                return;
            }

            sendJson(res, 404, { ok: false, error: `Route not found: ${req.method} ${pathname}` });
        } catch (err) {
            sendJson(res, 500, { ok: false, error: err.message || String(err) });
        }
    });

    return server;
}

if (require.main === module) {
    const cfg = parseCliArgs(process.argv);
    const srv = createHubServer({ baseDir: cfg.baseDir, forceLexical: cfg.forceLexical });
    srv.listen(cfg.port, cfg.host, () => {
        const addr = srv.address();
        const actualPort = typeof addr === 'object' && addr ? addr.port : cfg.port;
        console.log(`Tree-RAG Memory MCP Hub listening on http://${cfg.host}:${actualPort}`);
        console.log(`Base directory: ${cfg.baseDir}`);
    });
}

module.exports = {
    createHubServer,
    parseCliArgs,
    runLexicalSearch,
    runLexicalNeighbors,
    runLexicalSave,
    runLexicalUpdate,
    runLexicalDelete,
    buildMermaidFallback,
};
