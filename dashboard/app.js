/**
 * Tree-RAG Memory MCP Hub — Frontend Controller (`dashboard/app.js`)
 *
 * Interactive Hierarchical Tree RAG Explorer, L0->L3 Pyramid, 4 Memory Pillars,
 * Live Semantic Search with Tree Branch Auto-Highlighter, 2D Graph Canvas,
 * FAISS Vector Neighbors Drill-Down, Mermaid Symbolic Canvas, and Telemetry.
 */

'use strict';

const memRagState = {
    data: null,
    viewMode: 'tree',        // 'tree' | 'graph'
    graphLayout: 'radial',   // 'radial' | 'horizontal'
    hideEmpty: true,
    expandedWorkspaces: new Set(['sample_project', 'demo_workspace']),
    expandedCategories: new Set(),
    selectedWorkspace: 'global',
    selectedCategory: '',
    selectedSource: '',
    selectedLayer: '',
    selectedPillar: '',
    selectedTag: '',
    highlights: {
        workspaces: new Set(),
        categories: new Set(),
        sources: new Set(),
        ids: new Set(),
    },
    recordsOffset: 0,
    recordsLimit: 40,
    currentRecords: [],
    graphTransform: { x: 0, y: 0, scale: 1 },
    graphNodes: [],
    graphEdges: [],
    graphDragging: false,
    graphDragStart: { x: 0, y: 0 },
    graphHoveredNode: null,
    canvasInitialized: false,
    lastMermaidCode: '',
    recordsDebounceTimer: null,
};

const MEMRAG_CAT_META = {
    decision:   { icon: '👑', color: '#c084fc', bg: 'rgba(168,85,247,0.16)', border: 'rgba(168,85,247,0.38)', layer: 'L3' },
    skill:      { icon: '⚡', color: '#f472b6', bg: 'rgba(236,72,153,0.16)', border: 'rgba(236,72,153,0.38)', layer: 'L3' },
    code:       { icon: '💻', color: '#60a5fa', bg: 'rgba(59,130,246,0.16)', border: 'rgba(59,130,246,0.38)', layer: 'L2' },
    wiki:       { icon: '📖', color: '#38bdf8', bg: 'rgba(14,165,233,0.16)', border: 'rgba(14,165,233,0.38)', layer: 'L2' },
    bug:        { icon: '🐛', color: '#f87171', bg: 'rgba(239,68,68,0.16)',  border: 'rgba(239,68,68,0.38)',  layer: 'L1' },
    assumption: { icon: '📐', color: '#fbbf24', bg: 'rgba(245,158,11,0.16)', border: 'rgba(245,158,11,0.38)', layer: 'L1' },
    note:       { icon: '📝', color: '#34d399', bg: 'rgba(16,185,129,0.16)', border: 'rgba(16,185,129,0.38)', layer: 'L0' },
};

function escMemHtml(str) {
    return String(str == null ? '' : str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function escJsArg(str) {
    return String(str == null ? '' : str)
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'");
}

function formatMemBytes(bytes) {
    const b = Number(bytes || 0);
    if (b >= 1024 * 1024) return (b / (1024 * 1024)).toFixed(2) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
    return b + ' B';
}

async function refreshMemoryStatus() {
    const badge = document.getElementById('status-memoryMcp');
    const primaryDocsEl = document.getElementById('memory-stat-primary-docs');
    const primaryIdxEl = document.getElementById('memory-stat-primary-index');
    const workspacesEl = document.getElementById('memory-stat-workspaces');
    const embedEl = document.getElementById('memory-stat-embed');
    const pendingEl = document.getElementById('memory-stat-pending');
    const concurrentEl = document.getElementById('memory-stat-concurrent');
    const basePathEl = document.getElementById('memory-base-path');
    const wsListEl = document.getElementById('memory-workspaces-list');

    if (!badge) return;
    badge.textContent = 'Checking...';
    badge.className = 'status-badge status-starting';

    try {
        const res = await fetch('/api/memory/status');
        const d = await res.json();
        if (!res.ok || !d.ok) throw new Error(d.error || 'Status check failed');

        if (d.concurrentActive) {
            badge.textContent = 'Read-Only Guard Active';
            badge.className = 'status-badge status-starting';
        } else if (d.pendingTotal > 0) {
            badge.textContent = `Active (${d.pendingTotal} Pending)`;
            badge.className = 'status-badge status-starting';
        } else {
            badge.textContent = `Active (${d.backend || 'FAISS'})`;
            badge.className = 'status-badge status-running';
        }

        const primary = d.primaryWorkspace || (d.workspaces && d.workspaces[0]) || { name: 'default', docs: 0, indexBytes: 0 };
        if (primaryDocsEl) primaryDocsEl.textContent = `${primary.name}: ${primary.docs} docs`;
        if (primaryIdxEl) primaryIdxEl.textContent = `index.faiss: ${formatMemBytes(primary.indexBytes)}`;
        if (workspacesEl) workspacesEl.textContent = `${(d.workspaces || []).length} branches (${d.totalDocs || 0} docs)`;
        if (embedEl) {
            const ok = d.embedReachable !== false;
            embedEl.textContent = `${ok ? '✓' : '⚠'} ${d.embedUrl || 'MockEmbedder'}`;
            embedEl.style.color = ok ? '#34d399' : '#fbbf24';
        }
        if (pendingEl) {
            pendingEl.textContent = `${d.pendingTotal || 0} files`;
            pendingEl.style.color = (d.pendingTotal || 0) > 0 ? '#fbbf24' : '#34d399';
        }
        if (concurrentEl) {
            concurrentEl.textContent = d.concurrentActive
                ? `⛔ ${d.concurrencyReason || 'blocked'}`
                : `✓ ${d.concurrencyReason || 'direct writes enabled'}`;
            concurrentEl.style.color = d.concurrentActive ? '#fbbf24' : '#34d399';
        }
        if (basePathEl) basePathEl.textContent = d.basePath || '';

        if (wsListEl) {
            const wsArr = d.workspaces || [];
            if (!wsArr.length) {
                wsListEl.innerHTML = `<span style="color:var(--text-secondary);">No initialized workspaces found in baseDir.</span>`;
            } else {
                wsListEl.innerHTML = wsArr.map((w, idx) => {
                    const isPrimary = idx === 0;
                    const safeW = escJsArg(w.name);
                    return `<div onclick="selectMemRagNode('${safeW}','','')"
                                 style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;gap:10px;background:${isPrimary ? 'rgba(168,85,247,0.12)' : 'rgba(0,0,0,0.25)'};border:1px solid ${isPrimary ? 'rgba(168,85,247,0.3)' : 'rgba(255,255,255,0.07)'};border-radius:7px;padding:6px 10px;">
                        <span style="font-weight:${isPrimary ? 700 : 500};color:${isPrimary ? '#c084fc' : 'white'};">📁 ${escMemHtml(w.name)}${isPrimary ? ' ★' : ''}</span>
                        <span style="color:var(--text-secondary);font-family:monospace;">${w.docs} docs · ${formatMemBytes(w.indexBytes)}</span>
                    </div>`;
                }).join('');
            }
        }
    } catch (err) {
        badge.textContent = 'Error: ' + err.message;
        badge.className = 'status-badge status-stopped';
    }
}

async function loadMemoryRagTree(force = false) {
    try {
        const res = await fetch('/api/memory/tree');
        const data = await res.json();
        if (!data || !data.ok) throw new Error((data && data.error) || 'Failed to load RAG tree');
        memRagState.data = data;

        const s = data.summary || {};
        const tel = data.telemetry || {};
        const layers = s.layers || {};

        const el = (id) => document.getElementById(id);
        if (el('memrag-kpi-docs')) el('memrag-kpi-docs').textContent = Number(s.totalDocs || 0).toLocaleString('en-US');
        if (el('memrag-kpi-ws')) el('memrag-kpi-ws').textContent = `in ${s.totalWorkspaces || 0} workspaces`;
        if (el('memrag-kpi-bytes')) el('memrag-kpi-bytes').textContent = formatMemBytes(s.totalIndexBytes || 0);
        if (el('memrag-kpi-l3')) el('memrag-kpi-l3').textContent = String(layers.L3 || 0);
        if (el('memrag-kpi-l2')) el('memrag-kpi-l2').textContent = String(layers.L2 || 0);
        if (el('memrag-kpi-l1')) el('memrag-kpi-l1').textContent = String(layers.L1 || 0);
        if (el('memrag-kpi-telemetry')) el('memrag-kpi-telemetry').textContent = `${tel.totalCalls || 0} calls`;
        if (el('memrag-kpi-pending')) {
            el('memrag-kpi-pending').textContent = `pending: ${s.pendingTotal || 0} · avg: ${tel.avgLatencyMs || 0} ms`;
        }

        const wsChildren = (data.tree && data.tree.children) || [];
        if (force && wsChildren.length > 0 && memRagState.expandedWorkspaces.size === 0) {
            wsChildren.slice(0, 3).forEach((w) => memRagState.expandedWorkspaces.add(w.name));
        }

        populateMemRagWorkspaceSelects(wsChildren);
        renderMemRagPyramidAndPillars(s);
        renderMemRagTelemetrySection(s, tel);
        renderMemRagTreeExplorer();
        initMemRagGraphCanvasListeners();
        renderMemRagGraphCanvas();
        await loadMemRagRecords();
    } catch (err) {
        console.error('loadMemoryRagTree error:', err);
    }
}

function populateMemRagWorkspaceSelects(workspaces) {
    const searchSel = document.getElementById('memrag-search-ws');
    const mmdSel = document.getElementById('memrag-mermaid-ws');
    const curSearch = searchSel ? searchSel.value : 'global';
    const curMmd = mmdSel ? mmdSel.value : 'global';

    const opts = [`<option value="global">🌐 Global (all workspaces)</option>`]
        .concat(workspaces.map((w) => `<option value="${escMemHtml(w.name)}">📁 ${escMemHtml(w.name)} (${w.count})</option>`))
        .join('');

    if (searchSel) {
        searchSel.innerHTML = opts;
        if ([...searchSel.options].some((o) => o.value === curSearch)) searchSel.value = curSearch;
    }
    if (mmdSel) {
        mmdSel.innerHTML = opts;
        if ([...mmdSel.options].some((o) => o.value === curMmd)) mmdSel.value = curMmd;
    }
}

function renderMemRagPyramidAndPillars(summary) {
    const pyrEl = document.getElementById('memrag-pyramid-bars');
    const pilEl = document.getElementById('memrag-pillars-grid');
    const clearLayerBtn = document.getElementById('btn-memrag-clear-layer');
    const clearPillarBtn = document.getElementById('btn-memrag-clear-pillar');

    if (clearLayerBtn) clearLayerBtn.style.display = memRagState.selectedLayer ? 'inline-block' : 'none';
    if (clearPillarBtn) clearPillarBtn.style.display = memRagState.selectedPillar ? 'inline-block' : 'none';

    const total = Math.max(1, Number(summary.totalDocs || 1));
    const layers = summary.layers || {};
    const layerDefs = [
        { id: 'L3', label: '👑 L3 Persona · Decisions & Skills', desc: 'Architectural decisions, invariants, reusable SOP patterns', count: layers.L3 || 0, color: '#a855f7', widthPct: '66%' },
        { id: 'L2', label: '🧩 L2 Scenario · Code & Wiki', desc: 'Code modules, API contracts, technical documentation', count: layers.L2 || 0, color: '#3b82f6', widthPct: '78%' },
        { id: 'L1', label: '🔬 L1 Atomic Facts · Bugs & Assumptions', desc: 'Root-cause analyses, bug fixes, verified system assumptions', count: layers.L1 || 0, color: '#f59e0b', widthPct: '90%' },
        { id: 'L0', label: '💬 L0 Raw Context · Notes & Logs', desc: 'Session observations, operational notes, scratchpad context', count: layers.L0 || 0, color: '#10b981', widthPct: '100%' },
    ];

    if (pyrEl) {
        pyrEl.innerHTML = layerDefs.map((l) => {
            const active = memRagState.selectedLayer === l.id;
            const pct = Math.round((l.count / total) * 100);
            return `<div onclick="setMemRagLayerFilter('${l.id}')"
                         style="width:${l.widthPct};margin:0 auto;cursor:pointer;padding:8px 12px;border-radius:9px;background:${active ? 'rgba(168,85,247,0.26)' : 'rgba(0,0,0,0.36)'};border:1px solid ${active ? l.color : 'rgba(255,255,255,0.09)'};transition:0.18s;display:flex;justify-content:space-between;align-items:center;gap:8px;box-shadow:${active ? '0 0 12px ' + l.color + '44' : 'none'};">
                <div style="min-width:0;">
                    <div style="font-size:0.8rem;font-weight:700;color:white;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${l.label}</div>
                    <div style="font-size:0.68rem;color:var(--text-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${l.desc}</div>
                </div>
                <div style="text-align:right;flex-shrink:0;">
                    <span style="font-size:0.85rem;font-weight:800;color:${l.color};font-family:monospace;">${l.count}</span>
                    <span style="font-size:0.68rem;color:var(--text-secondary);display:block;">${pct}%</span>
                </div>
            </div>`;
        }).join('');
    }

    const pillars = summary.pillars || {};
    const pillarDefs = [
        { id: 'chat',  icon: '💬', title: 'Chat & Context',      sub: 'note · assumption', count: pillars.chat || 0,  color: '#34d399' },
        { id: 'skill', icon: '⚡', title: 'Skills & Decisions',  sub: 'decision · skill',  count: pillars.skill || 0, color: '#c084fc' },
        { id: 'wiki',  icon: '📖', title: 'Knowledge & Wiki',    sub: 'wiki · runbooks',   count: pillars.wiki || 0,  color: '#38bdf8' },
        { id: 'code',  icon: '🕸️', title: 'Code & Bug Graph',    sub: 'code · bug fixes',  count: pillars.code || 0,  color: '#60a5fa' },
    ];

    if (pilEl) {
        pilEl.innerHTML = pillarDefs.map((p) => {
            const active = memRagState.selectedPillar === p.id;
            return `<div onclick="setMemRagPillarFilter('${p.id}')"
                         style="cursor:pointer;padding:12px 14px;border-radius:10px;background:${active ? 'rgba(59,130,246,0.22)' : 'rgba(0,0,0,0.34)'};border:1px solid ${active ? p.color : 'rgba(255,255,255,0.08)'};transition:0.18s;display:flex;flex-direction:column;justify-content:space-between;gap:6px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:1.3rem;">${p.icon}</span>
                    <span style="font-size:1.2rem;font-weight:800;color:${p.color};">${p.count}</span>
                </div>
                <div>
                    <div style="font-size:0.82rem;font-weight:700;color:white;">${p.title}</div>
                    <div style="font-size:0.7rem;color:var(--text-secondary);">${p.sub}</div>
                </div>
            </div>`;
        }).join('');
    }
}

function setMemRagLayerFilter(layer) {
    memRagState.selectedLayer = (memRagState.selectedLayer === layer) ? '' : layer;
    memRagState.recordsOffset = 0;
    if (memRagState.data) renderMemRagPyramidAndPillars(memRagState.data.summary || {});
    loadMemRagRecords();
}

function setMemRagPillarFilter(pillar) {
    memRagState.selectedPillar = (memRagState.selectedPillar === pillar) ? '' : pillar;
    memRagState.recordsOffset = 0;
    if (memRagState.data) renderMemRagPyramidAndPillars(memRagState.data.summary || {});
    loadMemRagRecords();
}

function switchMemRagExplorerMode(mode) {
    memRagState.viewMode = mode;
    const treeDiv = document.getElementById('memrag-tree-container');
    const graphDiv = document.getElementById('memrag-graph-container');
    const btnTree = document.getElementById('btn-memrag-view-tree');
    const btnGraph = document.getElementById('btn-memrag-view-graph');
    if (treeDiv) treeDiv.style.display = mode === 'tree' ? 'block' : 'none';
    if (graphDiv) graphDiv.style.display = mode === 'graph' ? 'block' : 'none';
    if (btnTree) {
        btnTree.style.background = mode === 'tree' ? 'rgba(168, 85, 247, 0.3)' : 'transparent';
        btnTree.style.borderColor = mode === 'tree' ? '#a855f7' : 'transparent';
        btnTree.style.color = mode === 'tree' ? 'white' : 'var(--text-secondary)';
    }
    if (btnGraph) {
        btnGraph.style.background = mode === 'graph' ? 'rgba(168, 85, 247, 0.3)' : 'transparent';
        btnGraph.style.borderColor = mode === 'graph' ? '#a855f7' : 'transparent';
        btnGraph.style.color = mode === 'graph' ? 'white' : 'var(--text-secondary)';
    }
    if (mode === 'graph') {
        renderMemRagGraphCanvas();
    }
}

function expandCollapseAllMemRagTree(expand) {
    if (!memRagState.data || !memRagState.data.tree) return;
    if (!expand) {
        memRagState.expandedWorkspaces.clear();
        memRagState.expandedCategories.clear();
    } else {
        for (const ws of (memRagState.data.tree.children || [])) {
            if (ws.count > 0) {
                memRagState.expandedWorkspaces.add(ws.name);
                for (const c of (ws.children || [])) {
                    memRagState.expandedCategories.add(`${ws.name}::${c.name}`);
                }
            }
        }
    }
    renderMemRagTreeExplorer();
    renderMemRagGraphCanvas();
}

function selectMemRagNode(workspace, category = '', source = '') {
    memRagState.selectedWorkspace = workspace || 'global';
    memRagState.selectedCategory = category || '';
    memRagState.selectedSource = source || '';
    memRagState.recordsOffset = 0;

    if (workspace && workspace !== 'global') {
        memRagState.expandedWorkspaces.add(workspace);
    }
    if (workspace && category) {
        memRagState.expandedCategories.add(`${workspace}::${category}`);
    }
    renderMemRagTreeExplorer();
    renderMemRagGraphCanvas();
    loadMemRagRecords();
}

function toggleMemRagWorkspaceNode(wsName, event) {
    if (event) event.stopPropagation();
    if (memRagState.expandedWorkspaces.has(wsName)) {
        memRagState.expandedWorkspaces.delete(wsName);
    } else {
        memRagState.expandedWorkspaces.add(wsName);
    }
    selectMemRagNode(wsName, '', '');
}

function toggleMemRagCategoryNode(wsName, catName, event) {
    if (event) event.stopPropagation();
    const key = `${wsName}::${catName}`;
    if (memRagState.expandedCategories.has(key)) {
        memRagState.expandedCategories.delete(key);
    } else {
        memRagState.expandedCategories.add(key);
    }
    selectMemRagNode(wsName, catName, '');
}

function renderMemRagTreeExplorer() {
    const container = document.getElementById('memrag-tree-container');
    if (!container || !memRagState.data || !memRagState.data.tree) return;

    const filterTxt = (document.getElementById('memrag-tree-filter')?.value || '').trim().toLowerCase();
    const hideEmpty = memRagState.hideEmpty;
    const root = memRagState.data.tree;
    const workspaces = (root.children || []).filter((ws) => {
        if (hideEmpty && ws.count === 0) return false;
        if (!filterTxt) return true;
        if (ws.name.toLowerCase().includes(filterTxt)) return true;
        return (ws.children || []).some((c) =>
            c.name.toLowerCase().includes(filterTxt) ||
            (c.children || []).some((s) => s.name.toLowerCase().includes(filterTxt))
        );
    });

    const isRootSelected = memRagState.selectedWorkspace === 'global' && !memRagState.selectedCategory;

    let html = `
        <div onclick="selectMemRagNode('global','','')"
             style="display:flex;justify-content:space-between;align-items:center;padding:8px 11px;border-radius:8px;cursor:pointer;margin-bottom:6px;background:${isRootSelected ? 'rgba(139,92,246,0.28)' : 'rgba(255,255,255,0.04)'};border:1px solid ${isRootSelected ? '#a855f7' : 'rgba(255,255,255,0.08)'};">
            <div style="display:flex;align-items:center;gap:8px;font-weight:700;color:white;">
                <span>🌐</span>
                <span>Global Memory Hub (All Branches)</span>
            </div>
            <span style="background:rgba(168,85,247,0.25);color:#e9d5ff;padding:2px 9px;border-radius:999px;font-size:0.75rem;font-weight:700;font-family:monospace;">${root.count} docs</span>
        </div>
    `;

    for (const ws of workspaces) {
        const isWsExp = memRagState.expandedWorkspaces.has(ws.name) || Boolean(filterTxt);
        const isWsSel = memRagState.selectedWorkspace === ws.name && !memRagState.selectedCategory;
        const isWsHit = memRagState.highlights.workspaces.has(ws.name);

        html += `
        <div style="margin-left:10px;border-left:1px dashed rgba(255,255,255,0.12);padding-left:8px;margin-bottom:4px;">
            <div onclick="toggleMemRagWorkspaceNode('${escMemHtml(ws.name)}', event)"
                 style="display:flex;justify-content:space-between;align-items:center;padding:6px 10px;border-radius:7px;cursor:pointer;background:${isWsSel ? 'rgba(59,130,246,0.26)' : (isWsHit ? 'rgba(16,185,129,0.16)' : 'rgba(0,0,0,0.28)')};border:1px solid ${isWsSel ? '#60a5fa' : (isWsHit ? '#34d399' : 'rgba(255,255,255,0.06)')};transition:0.15s;">
                <div style="display:flex;align-items:center;gap:7px;min-width:0;">
                    <span style="font-size:0.72rem;color:var(--text-secondary);width:12px;">${ws.children && ws.children.length ? (isWsExp ? '▼' : '▶') : '•'}</span>
                    <span>📁</span>
                    <span style="font-weight:600;color:${ws.count > 0 ? 'white' : 'var(--text-secondary)'};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escMemHtml(ws.name)}</span>
                    ${isWsHit ? '<span style="font-size:0.66rem;background:rgba(16,185,129,0.25);color:#6ee7b7;padding:1px 6px;border-radius:4px;">🔥 HIT</span>' : ''}
                </div>
                <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                    <span style="font-size:0.68rem;color:var(--text-secondary);font-family:monospace;">${formatMemBytes(ws.indexBytes)}</span>
                    <span style="background:${ws.count > 0 ? 'rgba(59,130,246,0.22)' : 'rgba(255,255,255,0.06)'};color:${ws.count > 0 ? '#93c5fd' : '#64748b'};padding:1px 8px;border-radius:999px;font-size:0.73rem;font-weight:700;font-family:monospace;">${ws.count}</span>
                </div>
            </div>
        `;

        if (isWsExp && ws.children && ws.children.length) {
            for (const cat of ws.children) {
                const catKey = `${ws.name}::${cat.name}`;
                const isCatExp = memRagState.expandedCategories.has(catKey) || Boolean(filterTxt);
                const isCatSel = memRagState.selectedWorkspace === ws.name && memRagState.selectedCategory === cat.name && !memRagState.selectedSource;
                const isCatHit = memRagState.highlights.categories.has(catKey);
                const meta = MEMRAG_CAT_META[cat.name] || { icon: '🏷️', color: '#cbd5e1', bg: 'rgba(255,255,255,0.08)', layer: 'L0' };

                html += `
                <div style="margin-left:14px;border-left:1px dashed rgba(255,255,255,0.1);padding-left:8px;margin-top:3px;">
                    <div onclick="toggleMemRagCategoryNode('${escMemHtml(ws.name)}','${escMemHtml(cat.name)}', event)"
                         style="display:flex;justify-content:space-between;align-items:center;padding:5px 9px;border-radius:6px;cursor:pointer;background:${isCatSel ? meta.bg : (isCatHit ? 'rgba(16,185,129,0.14)' : 'rgba(0,0,0,0.2)')};border:1px solid ${isCatSel ? meta.color : (isCatHit ? '#34d399' : 'rgba(255,255,255,0.04)')};">
                        <div style="display:flex;align-items:center;gap:6px;">
                            <span style="font-size:0.68rem;color:var(--text-secondary);width:10px;">${cat.children && cat.children.length ? (isCatExp ? '▼' : '▶') : '•'}</span>
                            <span>${meta.icon}</span>
                            <span style="font-weight:600;color:${meta.color};">${escMemHtml(cat.name)}</span>
                            <span style="font-size:0.65rem;padding:1px 5px;border-radius:4px;background:rgba(255,255,255,0.06);color:var(--text-secondary);font-family:monospace;">${cat.layer}</span>
                            ${isCatHit ? '<span style="font-size:0.64rem;color:#6ee7b7;">🔥</span>' : ''}
                        </div>
                        <span style="background:${meta.bg};color:${meta.color};padding:1px 7px;border-radius:999px;font-size:0.71rem;font-weight:700;font-family:monospace;">${cat.count}</span>
                    </div>
                `;

                if (isCatExp && cat.children && cat.children.length) {
                    const maxSources = filterTxt ? 50 : 25;
                    for (const src of cat.children.slice(0, maxSources)) {
                        if (
                            filterTxt &&
                            !src.name.toLowerCase().includes(filterTxt) &&
                            !ws.name.toLowerCase().includes(filterTxt) &&
                            !cat.name.toLowerCase().includes(filterTxt)
                        ) {
                            continue;
                        }
                        const srcKey = `${ws.name}::${cat.name}::${src.name}`;
                        const isSrcSel = memRagState.selectedWorkspace === ws.name && memRagState.selectedCategory === cat.name && memRagState.selectedSource === src.name;
                        const isSrcHit = memRagState.highlights.sources.has(srcKey);
                        const safeWs = escJsArg(ws.name);
                        const safeCat = escJsArg(cat.name);
                        const safeSrc = escJsArg(src.name);

                        html += `
                        <div onclick="selectMemRagNode('${safeWs}','${safeCat}','${safeSrc}')"
                             style="margin-left:14px;margin-top:2px;display:flex;justify-content:space-between;align-items:center;padding:4px 8px;border-radius:5px;cursor:pointer;background:${isSrcSel ? 'rgba(168,85,247,0.22)' : (isSrcHit ? 'rgba(16,185,129,0.14)' : 'transparent')};border:1px solid ${isSrcSel ? '#c084fc' : (isSrcHit ? '#34d399' : 'transparent')};">
                            <div style="display:flex;align-items:center;gap:6px;min-width:0;">
                                <span style="color:var(--text-secondary);font-size:0.72rem;">📄</span>
                                <span style="font-size:0.75rem;color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:210px;" title="${escMemHtml(src.name)}">${escMemHtml(src.name)}</span>
                            </div>
                            <span style="font-size:0.7rem;color:var(--text-secondary);font-family:monospace;flex-shrink:0;">${src.count}</span>
                        </div>`;
                    }
                }
                html += `</div>`;
            }
        }
        html += `</div>`;
    }

    container.innerHTML = html;
}

// --- 2D Interactive Graph Canvas ---

function resetMemRagGraphView() {
    memRagState.graphTransform = { x: 0, y: 0, scale: 1 };
    drawMemRagGraphFrame();
}

function zoomMemRagGraph(factor) {
    const next = Math.max(0.35, Math.min(3.0, memRagState.graphTransform.scale * factor));
    memRagState.graphTransform.scale = next;
    drawMemRagGraphFrame();
}

function toggleMemRagGraphLayout() {
    memRagState.graphLayout = memRagState.graphLayout === 'radial' ? 'horizontal' : 'radial';
    const btn = document.getElementById('btn-memrag-graph-layout');
    if (btn) {
        btn.textContent = `Layout: ${memRagState.graphLayout === 'radial' ? 'Radial' : 'Horizontal'}`;
    }
    resetMemRagGraphView();
    renderMemRagGraphCanvas();
}

function toggleMemRagHideEmpty() {
    memRagState.hideEmpty = !memRagState.hideEmpty;
    const btn = document.getElementById('btn-memrag-hide-empty');
    if (btn) {
        btn.textContent = `Hide Empty: ${memRagState.hideEmpty ? 'ON' : 'OFF'}`;
    }
    renderMemRagTreeExplorer();
    renderMemRagGraphCanvas();
}

function initMemRagGraphCanvasListeners() {
    if (memRagState.canvasInitialized) return;
    const canvas = document.getElementById('memrag-graph-canvas');
    if (!canvas) return;
    memRagState.canvasInitialized = true;

    let moved = false;
    canvas.addEventListener('mousedown', (e) => {
        memRagState.graphDragging = true;
        moved = false;
        memRagState.graphDragStart = {
            x: e.clientX - memRagState.graphTransform.x,
            y: e.clientY - memRagState.graphTransform.y,
        };
        canvas.style.cursor = 'grabbing';
    });

    window.addEventListener('mouseup', () => {
        if (memRagState.graphDragging) {
            memRagState.graphDragging = false;
            canvas.style.cursor = 'grab';
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (canvas.height / rect.height);

        if (memRagState.graphDragging) {
            moved = true;
            memRagState.graphTransform.x = e.clientX - memRagState.graphDragStart.x;
            memRagState.graphTransform.y = e.clientY - memRagState.graphDragStart.y;
            drawMemRagGraphFrame();
            return;
        }

        const gx = (mx - canvas.width / 2 - memRagState.graphTransform.x) / memRagState.graphTransform.scale;
        const gy = (my - canvas.height / 2 - memRagState.graphTransform.y) / memRagState.graphTransform.scale;
        let hovered = null;
        for (let i = memRagState.graphNodes.length - 1; i >= 0; i--) {
            const n = memRagState.graphNodes[i];
            const dx = gx - n.x;
            const dy = gy - n.y;
            if (Math.sqrt(dx * dx + dy * dy) <= n.r + 5) {
                hovered = n;
                break;
            }
        }
        if (hovered !== memRagState.graphHoveredNode) {
            memRagState.graphHoveredNode = hovered;
            canvas.style.cursor = hovered ? 'pointer' : 'grab';
            const tip = document.getElementById('memrag-graph-tooltip');
            if (tip) {
                tip.textContent = hovered
                    ? `${hovered.label} (${hovered.count} records) — click to inspect`
                    : 'Click a node to inspect records · Drag to pan · Scroll to zoom';
            }
            drawMemRagGraphFrame();
        }
    });

    canvas.addEventListener('click', () => {
        if (moved) return;
        const n = memRagState.graphHoveredNode;
        if (!n) return;
        if (n.type === 'root') {
            selectMemRagNode('global', '', '');
        } else if (n.type === 'workspace') {
            toggleMemRagWorkspaceNode(n.workspace);
        } else if (n.type === 'category') {
            toggleMemRagCategoryNode(n.workspace, n.category);
        } else if (n.type === 'source') {
            selectMemRagNode(n.workspace, n.category, n.source);
        }
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 0.89;
        zoomMemRagGraph(factor);
    }, { passive: false });
}

function renderMemRagGraphCanvas() {
    if (!memRagState.data || !memRagState.data.tree) return;
    const hideEmpty = memRagState.hideEmpty;
    const root = memRagState.data.tree;
    const workspaces = (root.children || []).filter((w) => !hideEmpty || w.count > 0);

    const nodes = [];
    const edges = [];

    const rootNode = {
        id: 'root',
        type: 'root',
        label: 'Global Hub',
        count: root.count,
        x: 0,
        y: 0,
        r: 20,
        color: '#a855f7',
    };
    nodes.push(rootNode);

    if (memRagState.graphLayout === 'radial') {
        const wsCount = Math.max(1, workspaces.length);
        const r1 = 115;
        const r2 = 185;
        const r3 = 240;

        workspaces.forEach((ws, idx) => {
            const angle = (2 * Math.PI * idx) / wsCount - Math.PI / 2;
            const wx = Math.cos(angle) * r1;
            const wy = Math.sin(angle) * r1;
            const isSel = memRagState.selectedWorkspace === ws.name;
            const isHit = memRagState.highlights.workspaces.has(ws.name);
            const wsNode = {
                id: `ws:${ws.name}`,
                type: 'workspace',
                workspace: ws.name,
                label: ws.name,
                count: ws.count,
                x: wx,
                y: wy,
                r: Math.max(10, Math.min(18, 8 + Math.log10(ws.count + 1) * 4)),
                color: isHit ? '#10b981' : (isSel ? '#60a5fa' : '#3b82f6'),
                hit: isHit,
                selected: isSel,
            };
            nodes.push(wsNode);
            edges.push({ from: rootNode, to: wsNode, hit: isHit });

            if (memRagState.expandedWorkspaces.has(ws.name) && ws.children && ws.children.length) {
                const cats = ws.children;
                const spread = Math.min(Math.PI / 2.2, Math.max(0.35, (2 * Math.PI) / Math.max(3, wsCount) * 0.85));
                cats.forEach((cat, cIdx) => {
                    const offset = cats.length === 1 ? 0 : ((cIdx / (cats.length - 1)) - 0.5) * spread;
                    const cAngle = angle + offset;
                    const cx = Math.cos(cAngle) * r2;
                    const cy = Math.sin(cAngle) * r2;
                    const catKey = `${ws.name}::${cat.name}`;
                    const meta = MEMRAG_CAT_META[cat.name] || { color: '#cbd5e1' };
                    const isCatHit = memRagState.highlights.categories.has(catKey);
                    const isCatSel = memRagState.selectedWorkspace === ws.name && memRagState.selectedCategory === cat.name;
                    const catNode = {
                        id: `cat:${catKey}`,
                        type: 'category',
                        workspace: ws.name,
                        category: cat.name,
                        label: `${cat.name} (${cat.layer})`,
                        count: cat.count,
                        x: cx,
                        y: cy,
                        r: Math.max(8, Math.min(14, 6 + Math.log10(cat.count + 1) * 3.5)),
                        color: meta.color,
                        hit: isCatHit,
                        selected: isCatSel,
                    };
                    nodes.push(catNode);
                    edges.push({ from: wsNode, to: catNode, hit: isCatHit });

                    if (memRagState.expandedCategories.has(catKey) && cat.children && cat.children.length) {
                        const topSrcs = cat.children.slice(0, 5);
                        topSrcs.forEach((src, sIdx) => {
                            const sOff = topSrcs.length === 1 ? 0 : ((sIdx / (topSrcs.length - 1)) - 0.5) * 0.28;
                            const sAngle = cAngle + sOff;
                            const sx = Math.cos(sAngle) * r3;
                            const sy = Math.sin(sAngle) * r3;
                            const srcNode = {
                                id: `src:${catKey}::${src.name}`,
                                type: 'source',
                                workspace: ws.name,
                                category: cat.name,
                                source: src.name,
                                label: src.name.slice(0, 18),
                                count: src.count,
                                x: sx,
                                y: sy,
                                r: 7,
                                color: '#94a3b8',
                                hit: memRagState.highlights.sources.has(`${catKey}::${src.name}`),
                                selected: memRagState.selectedSource === src.name,
                            };
                            nodes.push(srcNode);
                            edges.push({ from: catNode, to: srcNode, hit: srcNode.hit });
                        });
                    }
                });
            }
        });
    } else {
        rootNode.x = -150;
        rootNode.y = 0;
        const wsCount = Math.max(1, workspaces.length);
        const totalHeight = Math.max(320, wsCount * 48);
        workspaces.forEach((ws, idx) => {
            const wy = wsCount === 1 ? 0 : ((idx / (wsCount - 1)) - 0.5) * totalHeight;
            const wx = -20;
            const isSel = memRagState.selectedWorkspace === ws.name;
            const isHit = memRagState.highlights.workspaces.has(ws.name);
            const wsNode = {
                id: `ws:${ws.name}`,
                type: 'workspace',
                workspace: ws.name,
                label: ws.name,
                count: ws.count,
                x: wx,
                y: wy,
                r: 12,
                color: isHit ? '#10b981' : (isSel ? '#60a5fa' : '#3b82f6'),
                hit: isHit,
                selected: isSel,
            };
            nodes.push(wsNode);
            edges.push({ from: rootNode, to: wsNode, hit: isHit });

            if (memRagState.expandedWorkspaces.has(ws.name) && ws.children) {
                ws.children.forEach((cat, cIdx) => {
                    const cy = wy + (cIdx - (ws.children.length - 1) / 2) * 28;
                    const cx = 120;
                    const catKey = `${ws.name}::${cat.name}`;
                    const meta = MEMRAG_CAT_META[cat.name] || { color: '#cbd5e1' };
                    const catNode = {
                        id: `cat:${catKey}`,
                        type: 'category',
                        workspace: ws.name,
                        category: cat.name,
                        label: `${cat.name} (${cat.layer})`,
                        count: cat.count,
                        x: cx,
                        y: cy,
                        r: 9,
                        color: meta.color,
                        hit: memRagState.highlights.categories.has(catKey),
                        selected: memRagState.selectedWorkspace === ws.name && memRagState.selectedCategory === cat.name,
                    };
                    nodes.push(catNode);
                    edges.push({ from: wsNode, to: catNode, hit: catNode.hit });
                });
            }
        });
    }

    memRagState.graphNodes = nodes;
    memRagState.graphEdges = edges;
    drawMemRagGraphFrame();
}

function drawMemRagGraphFrame() {
    const canvas = document.getElementById('memrag-graph-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(canvas.width / 2 + memRagState.graphTransform.x, canvas.height / 2 + memRagState.graphTransform.y);
    ctx.scale(memRagState.graphTransform.scale, memRagState.graphTransform.scale);

    if (memRagState.graphLayout === 'radial') {
        [115, 185, 240].forEach((radius) => {
            ctx.beginPath();
            ctx.arc(0, 0, radius, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255,255,255,0.04)';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    for (const e of memRagState.graphEdges) {
        ctx.beginPath();
        ctx.moveTo(e.from.x, e.from.y);
        const mx = (e.from.x + e.to.x) / 2;
        const my = (e.from.y + e.to.y) / 2;
        ctx.quadraticCurveTo(mx, my, e.to.x, e.to.y);
        ctx.strokeStyle = e.hit ? 'rgba(52, 211, 153, 0.75)' : 'rgba(148, 163, 184, 0.25)';
        ctx.lineWidth = e.hit ? 2.2 : 1.1;
        ctx.stroke();
    }

    for (const n of memRagState.graphNodes) {
        const isHovered = memRagState.graphHoveredNode === n;
        if (n.hit || n.selected || isHovered) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r + 5, 0, Math.PI * 2);
            ctx.fillStyle = n.hit ? 'rgba(16, 185, 129, 0.28)' : 'rgba(168, 85, 247, 0.28)';
            ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
        ctx.lineWidth = (n.selected || isHovered) ? 2.5 : 1.2;
        ctx.strokeStyle = n.hit ? '#6ee7b7' : (n.selected ? '#ffffff' : 'rgba(255,255,255,0.35)');
        ctx.stroke();

        ctx.font = 'bold 9px monospace';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (n.r >= 10) {
            ctx.fillText(String(n.count), n.x, n.y);
        }

        ctx.font = (n.type === 'root' || n.selected) ? 'bold 10.5px sans-serif' : '10px sans-serif';
        ctx.fillStyle = n.hit ? '#6ee7b7' : '#e2e8f0';
        ctx.textAlign = 'center';
        const labelText = n.r < 10 ? `${n.label} [${n.count}]` : n.label;
        ctx.fillText(labelText, n.x, n.y + n.r + 11);
    }

    ctx.restore();
}

// --- Right Panel: Node Records Inspector & Drill-Down ---

function debounceLoadMemRagRecords() {
    clearTimeout(memRagState.recordsDebounceTimer);
    memRagState.recordsDebounceTimer = setTimeout(() => {
        memRagState.recordsOffset = 0;
        loadMemRagRecords();
    }, 220);
}

async function loadMemRagRecords() {
    const listEl = document.getElementById('memrag-records-list');
    const breadEl = document.getElementById('memrag-breadcrumb');
    const subEl = document.getElementById('memrag-inspector-subtitle');
    const pagEl = document.getElementById('memrag-records-pagination');
    const tagsBar = document.getElementById('memrag-tags-bar');
    if (!listEl) return;

    const q = (document.getElementById('memrag-records-q')?.value || '').trim();
    const params = new URLSearchParams({
        workspace: memRagState.selectedWorkspace || 'global',
        category: memRagState.selectedCategory || '',
        source: memRagState.selectedSource || '',
        layer: memRagState.selectedLayer || '',
        pillar: memRagState.selectedPillar || '',
        tag: memRagState.selectedTag || '',
        q,
        limit: String(memRagState.recordsLimit),
        offset: String(memRagState.recordsOffset),
    });

    if (breadEl) {
        const parts = [`<span onclick="selectMemRagNode('global','','')" style="cursor:pointer;color:#c084fc;">🌐 Global Hub</span>`];
        if (memRagState.selectedWorkspace && memRagState.selectedWorkspace !== 'global') {
            const safeW = memRagState.selectedWorkspace.replace(/'/g, "'");
            parts.push(`<span style="color:var(--text-secondary);">›</span> <span onclick="selectMemRagNode('${safeW}','','')" style="cursor:pointer;color:#93c5fd;">📁 ${escMemHtml(memRagState.selectedWorkspace)}</span>`);
        }
        if (memRagState.selectedCategory) {
            const safeW = memRagState.selectedWorkspace.replace(/'/g, "'");
            const safeC = memRagState.selectedCategory.replace(/'/g, "'");
            parts.push(`<span style="color:var(--text-secondary);">›</span> <span onclick="selectMemRagNode('${safeW}','${safeC}','')" style="cursor:pointer;color:#fcd34d;">🏷️ ${escMemHtml(memRagState.selectedCategory)}</span>`);
        }
        if (memRagState.selectedSource) {
            parts.push(`<span style="color:var(--text-secondary);">›</span> <span style="color:#6ee7b7;">📄 ${escMemHtml(memRagState.selectedSource)}</span>`);
        }
        if (memRagState.selectedLayer) {
            parts.push(`<span style="background:rgba(168,85,247,0.25);color:#e9d5ff;padding:1px 7px;border-radius:4px;font-size:0.72rem;">Layer: ${memRagState.selectedLayer}</span>`);
        }
        if (memRagState.selectedPillar) {
            parts.push(`<span style="background:rgba(59,130,246,0.25);color:#bfdbfe;padding:1px 7px;border-radius:4px;font-size:0.72rem;">Pillar: ${memRagState.selectedPillar}</span>`);
        }
        breadEl.innerHTML = parts.join(' ');
    }

    try {
        const res = await fetch('/api/memory/records?' + params.toString());
        const d = await res.json();
        if (!d || !d.ok) throw new Error((d && d.error) || 'Failed to load records');

        memRagState.currentRecords = d.records || [];
        if (subEl) {
            subEl.textContent = `Found ${d.total} records (showing ${d.records.length ? (d.offset + 1) + '–' + (d.offset + d.records.length) : 0})`;
        }

        if (tagsBar && memRagState.data && memRagState.data.summary && Array.isArray(memRagState.data.summary.topTags)) {
            const topTags = memRagState.data.summary.topTags.slice(0, 14);
            if (topTags.length) {
                tagsBar.innerHTML = topTags.map((t) => {
                    const active = memRagState.selectedTag === t.tag;
                    return `<button onclick="toggleMemRagTagFilter('${escMemHtml(t.tag)}')"
                                    style="background:${active ? 'rgba(168,85,247,0.3)' : 'rgba(255,255,255,0.05)'};border:1px solid ${active ? '#c084fc' : 'rgba(255,255,255,0.1)'};color:${active ? 'white' : 'var(--text-secondary)'};border-radius:999px;padding:2px 8px;font-size:0.7rem;cursor:pointer;">
                                #${escMemHtml(t.tag)} (${t.count})
                            </button>`;
                }).join('');
            } else {
                tagsBar.innerHTML = '';
            }
        }

        if (!d.records.length) {
            listEl.innerHTML = `<div style="text-align:center;padding:35px 15px;color:var(--text-secondary);font-size:0.84rem;">No records match the selected filter in this RAG branch.</div>`;
        } else {
            listEl.innerHTML = d.records.map((rec, idx) => renderMemRagRecordCard(rec, idx)).join('');
        }

        if (pagEl) {
            const hasPrev = d.offset > 0;
            const hasNext = d.offset + d.limit < d.total;
            pagEl.innerHTML = `
                <span>Page ${Math.floor(d.offset / d.limit) + 1} of ${Math.max(1, Math.ceil(d.total / d.limit))} (${d.total} records)</span>
                <div style="display:flex;gap:6px;">
                    <button ${hasPrev ? '' : 'disabled'} onclick="pageMemRagRecords(-1)" style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.14);color:${hasPrev ? 'white' : '#475569'};border-radius:6px;padding:4px 10px;font-size:0.75rem;cursor:${hasPrev ? 'pointer' : 'default'};">← Previous</button>
                    <button ${hasNext ? '' : 'disabled'} onclick="pageMemRagRecords(1)" style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.14);color:${hasNext ? 'white' : '#475569'};border-radius:6px;padding:4px 10px;font-size:0.75rem;cursor:${hasNext ? 'pointer' : 'default'};">Next →</button>
                </div>
            `;
        }
    } catch (err) {
        listEl.innerHTML = `<div style="color:#f87171;padding:12px;">Error: ${escMemHtml(err.message)}</div>`;
    }
}

function toggleMemRagTagFilter(tag) {
    memRagState.selectedTag = (memRagState.selectedTag === tag) ? '' : tag;
    memRagState.recordsOffset = 0;
    loadMemRagRecords();
}

function pageMemRagRecords(dir) {
    memRagState.recordsOffset = Math.max(0, memRagState.recordsOffset + dir * memRagState.recordsLimit);
    loadMemRagRecords();
}

function renderMemRagRecordCard(rec, idx, extraScore = null) {
    const meta = MEMRAG_CAT_META[rec.category] || { icon: '🏷️', color: '#cbd5e1', bg: 'rgba(255,255,255,0.08)', border: 'rgba(255,255,255,0.18)', layer: 'L0' };
    const isHit = memRagState.highlights.ids.has(rec.id);
    const safeWs = escJsArg(rec.workspace || '');
    const safeId = escJsArg(rec.id || '');
    const drillBoxId = `memrag-drill-${rec.workspace}-${rec.id}`.replace(/[^a-zA-Z0-9_-]/g, '_');

    return `
    <div style="background:${isHit ? 'rgba(16,185,129,0.09)' : 'rgba(0,0,0,0.34)'};border:1px solid ${isHit ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:12px 14px;transition:0.15s;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                <span style="background:${meta.bg};border:1px solid ${meta.border};color:${meta.color};padding:2px 8px;border-radius:6px;font-size:0.72rem;font-weight:700;">
                    ${meta.icon} ${escMemHtml(rec.category)} · ${escMemHtml(rec.layer || meta.layer)}
                </span>
                <span onclick="selectMemRagNode('${safeWs}','','')" style="cursor:pointer;background:rgba(59,130,246,0.14);color:#93c5fd;padding:2px 8px;border-radius:6px;font-size:0.72rem;font-family:monospace;">
                    📁 ${escMemHtml(rec.workspace)}
                </span>
                <span style="font-size:0.72rem;color:var(--text-secondary);">src: <strong style="color:#cbd5e1;">${escMemHtml(rec.source)}</strong></span>
                ${extraScore !== null ? `<span style="background:rgba(16,185,129,0.22);color:#6ee7b7;padding:2px 8px;border-radius:6px;font-size:0.72rem;font-weight:700;font-family:monospace;">score=${Number(extraScore).toFixed(3)}</span>` : ''}
            </div>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.68rem;color:var(--text-secondary);font-family:monospace;">
                <span>${escMemHtml(rec.time || '')}</span>
                <span>· ${escMemHtml(rec.id)}</span>
            </div>
        </div>

        <div style="font-size:0.82rem;line-height:1.52;color:#f1f5f9;white-space:pre-wrap;word-break:break-word;max-height:180px;overflow-y:auto;padding:6px 8px;background:rgba(0,0,0,0.25);border-radius:6px;border:1px solid rgba(255,255,255,0.04);">${escMemHtml(rec.text)}</div>

        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;flex-wrap:wrap;gap:6px;">
            <div style="display:flex;gap:4px;flex-wrap:wrap;">
                ${(rec.tags || []).map((t) => `<span style="font-size:0.66rem;background:rgba(168,85,247,0.14);color:#d8b4fe;padding:1px 6px;border-radius:4px;">#${escMemHtml(t)}</span>`).join('')}
            </div>
            <div style="display:flex;gap:6px;">
                <button onclick="toggleMemRagDrillDown('${safeWs}','${safeId}','${drillBoxId}')"
                        style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.35);color:#93c5fd;border-radius:6px;padding:4px 9px;font-size:0.72rem;font-weight:600;cursor:pointer;">
                    🔗 Drill-Down &amp; Neighbors
                </button>
                <button onclick="openMemRagEditModal('${safeWs}','${safeId}')"
                        style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.15);color:#e2e8f0;border-radius:6px;padding:4px 9px;font-size:0.72rem;cursor:pointer;">
                    ✏️ Edit
                </button>
                <button onclick="deleteMemRagRecord('${safeWs}','${safeId}')"
                        style="background:rgba(239,68,68,0.14);border:1px solid rgba(239,68,68,0.32);color:#fca5a5;border-radius:6px;padding:4px 9px;font-size:0.72rem;cursor:pointer;">
                    🗑️ Delete
                </button>
            </div>
        </div>

        <div id="${drillBoxId}" style="display:none;margin-top:10px;padding-top:10px;border-top:1px dashed rgba(255,255,255,0.14);"></div>
    </div>`;
}

// --- Drill-Down & Traceability (Same-Source Siblings + Nearest Neighbors) ---

async function toggleMemRagDrillDown(workspace, docId, containerId) {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (box.style.display === 'block') {
        box.style.display = 'none';
        return;
    }
    box.style.display = 'block';
    box.innerHTML = `<div style="font-size:0.76rem;color:#93c5fd;">⏳ Reconstructing vector from index.faiss and computing nearest neighbors...</div>`;
    try {
        const res = await fetch('/api/memory/neighbors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workspace, id: docId, top_k: 5 }),
        });
        const d = await res.json();
        if (!d || !d.ok) throw new Error((d && d.error) || 'Failed to load neighbors');

        const siblings = d.same_source || [];
        const neighbors = d.semantic_neighbors || [];

        let html = `<div style="font-size:0.74rem;color:#a78bfa;margin-bottom:6px;font-family:monospace;">✓ Traceability &amp; Vector Neighbors (${d.latency_ms} ms via ${escMemHtml(d.backend || 'faiss')}):</div>`;
        html += `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;">`;

        html += `<div style="background:rgba(0,0,0,0.3);border:1px solid rgba(59,130,246,0.25);border-radius:8px;padding:8px 10px;">
            <div style="font-size:0.75rem;font-weight:700;color:#93c5fd;margin-bottom:6px;">🧭 Top-${neighbors.length} Nearest Vector Neighbors:</div>`;
        if (!neighbors.length) {
            html += `<div style="font-size:0.72rem;color:var(--text-secondary);">No other records in this workspace index.</div>`;
        } else {
            html += neighbors.map((n) => `
                <div style="padding:5px 7px;margin-bottom:5px;background:rgba(255,255,255,0.03);border-radius:6px;border-left:2px solid #60a5fa;">
                    <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:var(--text-secondary);">
                        <span>[${escMemHtml(n.category)} · ${escMemHtml(n.layer)}] ${escMemHtml(n.id)}</span>
                        <span style="color:#6ee7b7;font-weight:700;font-family:monospace;">cos=${Number(n.score).toFixed(3)}</span>
                    </div>
                    <div style="font-size:0.73rem;color:#e2e8f0;margin-top:2px;">${escMemHtml(n.text.slice(0, 170))}${n.text.length > 170 ? '…' : ''}</div>
                </div>
            `).join('');
        }
        html += `</div>`;

        html += `<div style="background:rgba(0,0,0,0.3);border:1px solid rgba(168,85,247,0.25);border-radius:8px;padding:8px 10px;">
            <div style="font-size:0.75rem;font-weight:700;color:#d8b4fe;margin-bottom:6px;">📄 Same-Source Entries (${siblings.length}):</div>`;
        if (!siblings.length) {
            html += `<div style="font-size:0.72rem;color:var(--text-secondary);">This record is the only entry from this source.</div>`;
        } else {
            html += siblings.slice(0, 5).map((s) => `
                <div style="padding:5px 7px;margin-bottom:5px;background:rgba(255,255,255,0.03);border-radius:6px;border-left:2px solid #c084fc;">
                    <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:var(--text-secondary);">
                        <span>[${escMemHtml(s.category)} · ${escMemHtml(s.layer)}]</span>
                        <span>${escMemHtml(s.time)}</span>
                    </div>
                    <div style="font-size:0.73rem;color:#e2e8f0;margin-top:2px;">${escMemHtml(s.text.slice(0, 170))}${s.text.length > 170 ? '…' : ''}</div>
                </div>
            `).join('');
        }
        html += `</div></div>`;

        box.innerHTML = html;
    } catch (err) {
        box.innerHTML = `<div style="color:#f87171;font-size:0.75rem;">Drill-down error: ${escMemHtml(err.message)}</div>`;
    }
}

// --- Live Semantic Search & Tree Highlighter ---

async function runMemRagSemanticSearch() {
    const q = (document.getElementById('memrag-search-input')?.value || '').trim();
    if (!q) return;
    const ws = document.getElementById('memrag-search-ws')?.value || 'global';
    const cat = document.getElementById('memrag-search-cat')?.value || '';
    const mode = document.getElementById('memrag-search-mode')?.value || 'base';
    const btn = document.getElementById('btn-memrag-search');
    const clearBtn = document.getElementById('btn-memrag-search-clear');
    const metaEl = document.getElementById('memrag-search-meta');
    const resEl = document.getElementById('memrag-search-results');

    if (btn) { btn.disabled = true; btn.textContent = '⏳ Searching...'; }
    if (resEl) {
        resEl.style.display = 'block';
        resEl.innerHTML = `<div style="font-size:0.82rem;color:#93c5fd;">⏳ Querying indices (${mode === 'plus' ? 'Custom Plus 4096d' : 'e5-small 384d'})...</div>`;
    }

    try {
        const res = await fetch('/api/memory/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: q,
                scope: ws === 'global' ? 'global' : 'workspace',
                workspace: ws === 'global' ? '' : ws,
                category: cat,
                mode,
                top_k: 8,
            }),
        });
        const d = await res.json();
        if (!d || !d.ok) throw new Error((d && d.error) || 'Semantic search failed');

        const hl = d.highlights || {};
        memRagState.highlights = {
            workspaces: new Set(hl.workspaces || []),
            categories: new Set(hl.categories || []),
            sources: new Set(hl.sources || []),
            ids: new Set(hl.ids || []),
        };

        for (const w of (hl.workspaces || [])) {
            memRagState.expandedWorkspaces.add(w);
        }

        if (clearBtn) clearBtn.style.display = 'inline-block';
        if (metaEl) {
            metaEl.textContent = `${d.count} hits in ${d.latency_ms} ms (${(d.mode || 'base').toUpperCase()} · ${d.backend || 'faiss'}) · highlighted ${(hl.workspaces || []).length} branches`;
        }

        renderMemRagTreeExplorer();
        renderMemRagGraphCanvas();

        if (resEl) {
            if (!d.results || !d.results.length) {
                resEl.innerHTML = `<div style="font-size:0.82rem;color:var(--text-secondary);">No semantic hits found for this query.</div>`;
            } else {
                resEl.innerHTML = `
                    <div style="font-size:0.78rem;color:#6ee7b7;margin-bottom:8px;font-weight:600;">
                        🔥 Semantic Hits (click a branch in the tree or inspect below):
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:10px;">
                        ${d.results.map((r, i) => renderMemRagRecordCard(r, i, r.score)).join('')}
                    </div>
                `;
            }
        }
    } catch (err) {
        if (resEl) resEl.innerHTML = `<div style="color:#f87171;font-size:0.82rem;">Search error: ${escMemHtml(err.message)}</div>`;
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '🔍 Search FAISS'; }
    }
}

function clearMemRagSearchHighlights() {
    memRagState.highlights = {
        workspaces: new Set(),
        categories: new Set(),
        sources: new Set(),
        ids: new Set(),
    };
    const resEl = document.getElementById('memrag-search-results');
    const clearBtn = document.getElementById('btn-memrag-search-clear');
    const metaEl = document.getElementById('memrag-search-meta');
    if (resEl) { resEl.style.display = 'none'; resEl.innerHTML = ''; }
    if (clearBtn) clearBtn.style.display = 'none';
    if (metaEl) metaEl.textContent = '';
    renderMemRagTreeExplorer();
    renderMemRagGraphCanvas();
    loadMemRagRecords();
}

// --- CRUD Modal & Actions (Save / Edit / Delete / Merge Pending / Workspace Sync) ---

function openMemRagSaveModal() {
    const modal = document.getElementById('memrag-modal-backdrop');
    if (!modal) return;
    document.getElementById('memrag-modal-title').textContent = '➕ Add New Entry to Memory MCP';
    document.getElementById('memrag-modal-id').value = '';
    document.getElementById('memrag-modal-ws').value = (memRagState.selectedWorkspace && memRagState.selectedWorkspace !== 'global')
        ? memRagState.selectedWorkspace
        : 'sample_project';
    document.getElementById('memrag-modal-ws').disabled = false;
    document.getElementById('memrag-modal-cat').value = memRagState.selectedCategory || 'decision';
    document.getElementById('memrag-modal-source').value = memRagState.selectedSource || 'hub-ui';
    document.getElementById('memrag-modal-tags').value = '';
    document.getElementById('memrag-modal-text').value = '';
    document.getElementById('memrag-modal-status').textContent = '';
    modal.style.display = 'flex';
}

function openMemRagEditModal(workspace, docId) {
    const rec = (memRagState.currentRecords || []).find((r) => r.id === docId && r.workspace === workspace);
    const modal = document.getElementById('memrag-modal-backdrop');
    if (!modal || !rec) return;
    document.getElementById('memrag-modal-title').textContent = `✏️ Edit Entry ${docId} [${workspace}]`;
    document.getElementById('memrag-modal-id').value = docId;
    document.getElementById('memrag-modal-ws').value = workspace;
    document.getElementById('memrag-modal-ws').disabled = true;
    document.getElementById('memrag-modal-cat').value = rec.category || 'note';
    document.getElementById('memrag-modal-source').value = rec.source || '';
    document.getElementById('memrag-modal-tags').value = (rec.tags || []).join(', ');
    document.getElementById('memrag-modal-text').value = rec.text || '';
    document.getElementById('memrag-modal-status').textContent = '';
    modal.style.display = 'flex';
}

function closeMemRagModal() {
    const modal = document.getElementById('memrag-modal-backdrop');
    if (modal) modal.style.display = 'none';
}

async function submitMemRagModal() {
    const docId = document.getElementById('memrag-modal-id').value.trim();
    const workspace = document.getElementById('memrag-modal-ws').value.trim() || 'sample_project';
    const category = document.getElementById('memrag-modal-cat').value;
    const source = document.getElementById('memrag-modal-source').value.trim() || 'hub-ui';
    const tags = document.getElementById('memrag-modal-tags').value.trim();
    const text = document.getElementById('memrag-modal-text').value.trim();
    const statusEl = document.getElementById('memrag-modal-status');
    const btn = document.getElementById('btn-memrag-modal-save');

    if (!text) {
        if (statusEl) { statusEl.style.color = '#f87171'; statusEl.textContent = 'Observation text cannot be empty.'; }
        return;
    }

    if (btn) btn.disabled = true;
    if (statusEl) { statusEl.style.color = '#93c5fd'; statusEl.textContent = '⏳ Embedding and persisting to FAISS...'; }

    try {
        const endpoint = docId ? '/api/memory/update' : '/api/memory/save';
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: docId, workspace, category, source, tags, text }),
        });
        const d = await res.json();
        if (!d || !d.ok) throw new Error((d && (d.error || d.message)) || 'Save failed');
        if (statusEl) { statusEl.style.color = '#34d399'; statusEl.textContent = d.message || '✓ Saved successfully!'; }
        setTimeout(() => {
            closeMemRagModal();
            loadMemoryRagTree(true);
            refreshMemoryStatus();
        }, 400);
    } catch (err) {
        if (statusEl) { statusEl.style.color = '#f87171'; statusEl.textContent = 'Error: ' + err.message; }
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function deleteMemRagRecord(workspace, docId) {
    if (!confirm(`Delete record ${docId} from workspace [${workspace}] and rebuild its FAISS index?`)) return;
    try {
        const res = await fetch('/api/memory/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workspace, id: docId }),
        });
        const d = await res.json();
        if (!d || !d.ok) throw new Error((d && d.error) || 'Failed to delete record');
        await loadMemoryRagTree(true);
        await refreshMemoryStatus();
    } catch (err) {
        alert('Delete error: ' + err.message);
    }
}

async function triggerMemRagMergePending() {
    const btn = document.getElementById('btn-memrag-merge');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Merging...'; }
    try {
        const res = await fetch('/api/memory/merge', { method: 'POST', body: '{}' });
        const d = await res.json();
        alert((d && d.output) || (d && d.error) || 'Pending queue merge complete.');
        await loadMemoryRagTree(true);
        await refreshMemoryStatus();
    } catch (err) {
        alert('Merge error: ' + err.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '⚡ Merge Pending'; }
    }
}

async function triggerMemRagWorkspaceSync() {
    const btn = document.getElementById('btn-memrag-wssync');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Syncing...'; }
    try {
        const res = await fetch('/api/memory/workspace-sync', { method: 'POST', body: '{}' });
        const d = await res.json();
        alert((d && d.output) || (d && d.error) || 'Workspace synchronization complete.');
        await loadMemoryRagTree(true);
        await refreshMemoryStatus();
    } catch (err) {
        alert('Workspace sync error: ' + err.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '🔗 Workspace Sync'; }
    }
}

// --- Mermaid Symbolic Memory Canvas ---

function toggleMemRagMermaidPanel() {
    const p = document.getElementById('memrag-mermaid-panel');
    if (!p) return;
    if (p.style.display === 'none') {
        p.style.display = 'block';
        const sel = document.getElementById('memrag-mermaid-ws');
        const ws = (memRagState.selectedWorkspace && memRagState.selectedWorkspace !== 'global')
            ? memRagState.selectedWorkspace
            : (sel ? sel.value : 'global');
        if (sel) sel.value = ws;
        loadMemRagMermaid(ws);
    } else {
        p.style.display = 'none';
    }
}

async function loadMemRagMermaid(workspace) {
    const renderEl = document.getElementById('memrag-mermaid-render');
    const codeEl = document.getElementById('memrag-mermaid-code');
    if (!renderEl) return;
    renderEl.innerHTML = `<div style="color:#f9a8d4;font-size:0.82rem;">⏳ Generating Symbolic Memory Canvas for [${escMemHtml(workspace)}]...</div>`;
    try {
        const res = await fetch('/api/memory/mermaid?workspace=' + encodeURIComponent(workspace || 'global'));
        const d = await res.json();
        if (!d || !d.ok) throw new Error((d && d.error) || 'Failed to generate Mermaid flowchart');
        memRagState.lastMermaidCode = d.mermaid || '';
        if (codeEl) codeEl.textContent = d.mermaid || '';

        if (window.mermaid && typeof window.mermaid.render === 'function') {
            try {
                window.mermaid.initialize({ startOnLoad: false, theme: 'dark' });
                const svgId = 'mmd-svg-' + Date.now();
                const out = await window.mermaid.render(svgId, d.mermaid);
                if (out && out.svg) {
                    renderEl.innerHTML = out.svg;
                    return;
                }
            } catch (_) {
                // Fall through to structured card renderer
            }
        }

        const lines = (d.mermaid || '').split('\n').slice(1);
        const nodesMap = {};
        const edges = [];
        for (const line of lines) {
            const mNode = line.match(/^\s*([A-Za-z0-9_]+)\["(.*)"\]\s*$/);
            if (mNode) {
                nodesMap[mNode[1]] = mNode[2];
                continue;
            }
            const mEdge = line.match(/^\s*([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)\s*$/);
            if (mEdge) {
                edges.push([mEdge[1], mEdge[2]]);
            }
        }

        const childrenOf = {};
        for (const [u, v] of edges) {
            if (!childrenOf[u]) childrenOf[u] = [];
            childrenOf[u].push(v);
        }
        const rootId = Object.keys(nodesMap)[0];
        if (!rootId) {
            renderEl.innerHTML = `<pre style="color:#e2e8f0;font-size:0.78rem;">${escMemHtml(d.mermaid)}</pre>`;
            return;
        }

        const midIds = childrenOf[rootId] || [];
        renderEl.innerHTML = `
            <div style="display:flex;flex-direction:column;gap:12px;">
                <div style="align-self:flex-start;background:linear-gradient(135deg,rgba(168,85,247,0.28),rgba(59,130,246,0.25));border:1px solid #a855f7;color:white;padding:8px 16px;border-radius:10px;font-weight:700;font-size:0.86rem;">
                    ${nodesMap[rootId]}
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;">
                    ${midIds.map((mid) => {
                        const leaves = childrenOf[mid] || [];
                        return `<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);border-radius:9px;padding:10px 12px;">
                            <div style="font-size:0.8rem;font-weight:700;color:#93c5fd;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:5px;">
                                ${nodesMap[mid] || mid}
                            </div>
                            <div style="display:flex;flex-direction:column;gap:6px;">
                                ${leaves.map((lid) => `<div style="font-size:0.74rem;color:#e2e8f0;background:rgba(0,0,0,0.35);padding:6px 8px;border-radius:6px;border-left:2px solid #f472b6;">${nodesMap[lid] || lid}</div>`).join('')}
                            </div>
                        </div>`;
                    }).join('')}
                </div>
            </div>
        `;
    } catch (err) {
        renderEl.innerHTML = `<div style="color:#f87171;font-size:0.8rem;">Error: ${escMemHtml(err.message)}</div>`;
    }
}

function copyMemRagMermaidCode() {
    if (!memRagState.lastMermaidCode) return;
    navigator.clipboard.writeText(memRagState.lastMermaidCode);
    alert('Copied Mermaid Symbolic Canvas code to clipboard!');
}

// --- Bottom Section: Category Breakdown, Timeline & Telemetry Log ---

function renderMemRagTelemetrySection(summary, telemetry) {
    const catEl = document.getElementById('memrag-category-breakdown');
    const timeEl = document.getElementById('memrag-timeline-chart');
    const telEl = document.getElementById('memrag-telemetry-table');

    const total = Math.max(1, Number(summary.totalDocs || 1));
    const cats = summary.categories || {};

    if (catEl) {
        const entries = Object.entries(cats).sort((a, b) => b[1] - a[1]);
        catEl.innerHTML = `
            <div style="display:flex;height:10px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,0.06);margin-bottom:8px;">
                ${entries.map(([k, v]) => {
                    const m = MEMRAG_CAT_META[k] || { color: '#94a3b8' };
                    const w = (v / total) * 100;
                    return w > 0 ? `<div style="width:${w.toFixed(2)}%;background:${m.color};" title="${k}: ${v}"></div>` : '';
                }).join('')}
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;">
                ${entries.map(([k, v]) => {
                    const m = MEMRAG_CAT_META[k] || { icon: '•', color: '#94a3b8' };
                    return `<span style="font-size:0.73rem;color:var(--text-secondary);display:inline-flex;align-items:center;gap:4px;background:rgba(0,0,0,0.28);padding:3px 8px;border-radius:6px;border:1px solid rgba(255,255,255,0.05);">
                        <span>${m.icon}</span>
                        <strong style="color:${m.color};">${escMemHtml(k)}</strong>:
                        <span style="color:white;font-family:monospace;">${v}</span>
                    </span>`;
                }).join('')}
            </div>
        `;
    }

    if (timeEl) {
        const timeline = summary.timeline || [];
        if (!timeline.length) {
            timeEl.innerHTML = `<div style="color:var(--text-secondary);font-size:0.75rem;margin:auto;">No timeline data available</div>`;
        } else {
            const maxCnt = Math.max(1, ...timeline.map((x) => x.count));
            timeEl.innerHTML = timeline.map((item) => {
                const hPct = Math.max(8, Math.round((item.count / maxCnt) * 100));
                return `<div style="flex:1;min-width:18px;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;" title="${item.date}: ${item.count} entries">
                    <div style="font-size:0.6rem;color:#cbd5e1;margin-bottom:2px;font-family:monospace;">${item.count}</div>
                    <div style="width:100%;height:${hPct}%;background:linear-gradient(180deg,#a855f7,#3b82f6);border-radius:4px 4px 0 0;"></div>
                    <div style="font-size:0.58rem;color:var(--text-secondary);margin-top:3px;">${item.date.slice(5)}</div>
                </div>`;
            }).join('');
        }
    }

    if (telEl) {
        const recent = (telemetry && telemetry.recentCalls) || [];
        if (!recent.length) {
            telEl.innerHTML = `<div style="color:var(--text-secondary);padding:14px;text-align:center;">No calls recorded in _telemetry.jsonl yet — run a search or save an entry to view live telemetry.</div>`;
        } else {
            telEl.innerHTML = `
                <table style="width:100%;border-collapse:collapse;text-align:left;">
                    <thead>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.1);color:var(--text-secondary);font-size:0.7rem;">
                            <th style="padding:5px 6px;">Time</th>
                            <th style="padding:5px 6px;">Operation</th>
                            <th style="padding:5px 6px;">Workspace</th>
                            <th style="padding:5px 6px;">Query / Preview</th>
                            <th style="padding:5px 6px;">Hits</th>
                            <th style="padding:5px 6px;">Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${recent.map((r) => `
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                                <td style="padding:5px 6px;color:var(--text-secondary);white-space:nowrap;">${escMemHtml((r.ts || '').slice(11))}</td>
                                <td style="padding:5px 6px;color:#c084fc;font-weight:600;">${escMemHtml(r.op)}</td>
                                <td style="padding:5px 6px;color:#93c5fd;">${escMemHtml(r.workspace)}</td>
                                <td style="padding:5px 6px;color:#e2e8f0;max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escMemHtml(r.query)}">${escMemHtml(r.query || '—')}</td>
                                <td style="padding:5px 6px;color:#34d399;">${r.hits ?? 0}${r.top_score ? ` (${Number(r.top_score).toFixed(2)})` : ''}</td>
                                <td style="padding:5px 6px;color:#fbbf24;">${r.latency_ms ?? 0} ms</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    refreshMemoryStatus();
    loadMemoryRagTree(true);
});
