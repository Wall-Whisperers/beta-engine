/* Beta Engine — wall grid editor (vanilla JS) */
(() => {
  'use strict';

  const TYPE_COLORS = {
    jug:      '#22c55e',
    crimp:    '#ef4444',
    sloper:   '#f59e0b',
    pinch:    '#3b82f6',
    foothold: '#a855f7',
  };
  const HOLD_TYPES = Object.keys(TYPE_COLORS);
  const SIZES = ['small', 'medium', 'large'];

  const state = {
    cols: 10,
    rows: 14,
    wallId: 'my-wall',
    holds: new Map(), // key "x,y" -> hold
    selectedKey: null,
    nextNum: 1,
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    grid:      $('grid'),
    legend:    $('legend'),
    cellInfo:  $('cellInfo'),
    status:    $('status'),
    counts:    $('counts'),
    cols:      $('cols'),
    rows:      $('rows'),
    btnResize: $('btnResize'),
    holdType:  $('holdType'),
    size:      $('size'),
    color:     $('color'),
    orient:    $('orient'),
    orientVal: $('orientVal'),
    isStart:   $('isStart'),
    isFinish:  $('isFinish'),
    btnApply:  $('btnApply'),
    btnRemove: $('btnRemove'),
    wallId:    $('wallId'),
    wallList:  $('wallList'),
    btnLoad:   $('btnLoad'),
    btnSave:   $('btnSave'),
    btnNew:    $('btnNew'),
    btnDelete: $('btnDelete'),
    json:      $('json'),
    btnApplyJson: $('btnApplyJson'),
    btnFormat: $('btnFormat'),
    btnDownload: $('btnDownload'),
    fileUpload: $('fileUpload'),
  };

  const key = (x, y) => `${x},${y}`;
  const setStatus = (msg, kind = '') => {
    els.status.textContent = msg;
    els.status.className = 'status' + (kind ? ' ' + kind : '');
  };
  const inBounds = (x, y) => x >= 0 && y >= 0 && x < state.cols && y < state.rows;

  function nextHoldId() {
    while (true) {
      const id = `h_${String(state.nextNum).padStart(3, '0')}`;
      state.nextNum += 1;
      const taken = [...state.holds.values()].some((h) => h.hold_id === id);
      if (!taken) return id;
    }
  }

  function paletteHold() {
    return {
      hold_type: els.holdType.value,
      orientation_deg: Number(els.orient.value) % 360,
      size: els.size.value,
      color: els.color.value,
      is_start: els.isStart.checked,
      is_finish: els.isFinish.checked,
    };
  }

  function writeHoldToPalette(h) {
    els.holdType.value = h.hold_type;
    els.size.value = h.size;
    els.color.value = h.color && h.color.startsWith('#') ? h.color : (TYPE_COLORS[h.hold_type] || '#22c55e');
    els.orient.value = Math.round(h.orientation_deg);
    els.orientVal.textContent = `${Math.round(h.orientation_deg)}°`;
    els.isStart.checked = !!h.is_start;
    els.isFinish.checked = !!h.is_finish;
  }

  function render() {
    renderLegend();
    renderGrid();
    renderJson();
    renderCounts();
  }

  function renderLegend() {
    els.legend.innerHTML = '';
    HOLD_TYPES.forEach((t) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.innerHTML = `<span class="dot" style="background:${TYPE_COLORS[t]}"></span>${t}`;
      els.legend.appendChild(chip);
    });
    const start = document.createElement('span');
    start.className = 'chip';
    start.innerHTML = `<span class="dot" style="background:#0b1220;outline:2px solid var(--start)"></span>start`;
    const fin = document.createElement('span');
    fin.className = 'chip';
    fin.innerHTML = `<span class="dot" style="background:#0b1220;outline:2px solid var(--finish)"></span>finish`;
    els.legend.appendChild(start);
    els.legend.appendChild(fin);
  }

  function renderGrid() {
    const g = els.grid;
    g.style.gridTemplateColumns = `repeat(${state.cols}, 44px)`;
    g.innerHTML = '';
    // Render top row first (highest y) so origin (0,0) is bottom-left visually.
    for (let y = state.rows - 1; y >= 0; y -= 1) {
      for (let x = 0; x < state.cols; x += 1) {
        const cell = document.createElement('button');
        cell.type = 'button';
        cell.className = 'cell';
        cell.dataset.x = x;
        cell.dataset.y = y;
        cell.setAttribute('role', 'gridcell');
        cell.setAttribute('aria-label', `cell ${x},${y}`);
        const k = key(x, y);
        if (state.selectedKey === k) cell.classList.add('selected');
        const hold = state.holds.get(k);
        if (hold) cell.appendChild(holdElement(hold));
        cell.addEventListener('click', (e) => onCellClick(x, y, e));
        cell.addEventListener('contextmenu', (e) => { e.preventDefault(); onCellRightClick(x, y); });
        g.appendChild(cell);
      }
    }
    if (state.selectedKey) {
      const [sx, sy] = state.selectedKey.split(',').map(Number);
      const h = state.holds.get(state.selectedKey);
      els.cellInfo.textContent = h
        ? `selected ${h.hold_id}  @ (${sx}, ${sy})  · ${h.hold_type} ${h.size} ${Math.round(h.orientation_deg)}°`
        : `cell (${sx}, ${sy}) — empty`;
    } else {
      els.cellInfo.textContent = 'no selection';
    }
  }

  function holdElement(h) {
    const el = document.createElement('div');
    el.className = `hold size-${h.size}` +
      (h.is_start ? ' is-start' : '') + (h.is_finish ? ' is-finish' : '');
    el.style.background = h.color || TYPE_COLORS[h.hold_type] || '#888';
    el.title = `${h.hold_id} • ${h.hold_type} • ${Math.round(h.orientation_deg)}°` +
      (h.is_start ? ' • START' : '') + (h.is_finish ? ' • FINISH' : '');
    let label = h.hold_type[0].toUpperCase();
    if (h.is_start) label = 'S';
    if (h.is_finish) label = 'F';
    if (h.is_start && h.is_finish) label = 'SF';
    el.textContent = label;
    const arrow = document.createElement('span');
    arrow.className = 'arrow';
    // 0° points up; rotates clockwise. The triangle's natural point is downward,
    // so we translate up by half its size first then rotate around the hold center.
    arrow.style.transform = `rotate(${h.orientation_deg}deg) translate(-5px, -16px)`;
    el.appendChild(arrow);
    return el;
  }

  function renderCounts() {
    const counts = { jug: 0, crimp: 0, sloper: 0, pinch: 0, foothold: 0 };
    let starts = 0, finishes = 0;
    for (const h of state.holds.values()) {
      counts[h.hold_type] = (counts[h.hold_type] || 0) + 1;
      if (h.is_start) starts += 1;
      if (h.is_finish) finishes += 1;
    }
    const total = state.holds.size;
    const parts = [`${total} hold${total === 1 ? '' : 's'}`];
    if (starts) parts.push(`${starts} start`);
    if (finishes) parts.push(`${finishes} finish`);
    HOLD_TYPES.forEach((t) => { if (counts[t]) parts.push(`${counts[t]} ${t}`); });
    els.counts.textContent = parts.join(' · ');
  }

  function renderJson() {
    els.json.value = JSON.stringify(currentWall(), null, 2);
  }

  function currentWall() {
    const holds = [...state.holds.values()]
      .slice()
      .sort((a, b) => (a.grid_y - b.grid_y) || (a.grid_x - b.grid_x))
      .map((h) => ({
        hold_id: h.hold_id,
        grid_x: h.grid_x,
        grid_y: h.grid_y,
        hold_type: h.hold_type,
        orientation_deg: Number(h.orientation_deg),
        size: h.size,
        color: h.color,
        is_start: !!h.is_start,
        is_finish: !!h.is_finish,
      }));
    return {
      wall_id: state.wallId,
      grid: { cols: state.cols, rows: state.rows },
      holds,
    };
  }

  function loadWallFromObject(obj) {
    if (!obj || typeof obj !== 'object') throw new Error('not an object');
    if (typeof obj.wall_id !== 'string') throw new Error('wall_id missing');
    if (!Array.isArray(obj.holds)) throw new Error('holds must be array');

    let cols = 10, rows = 14;
    if (obj.grid && Number.isFinite(obj.grid.cols) && Number.isFinite(obj.grid.rows)) {
      cols = Math.max(1, obj.grid.cols);
      rows = Math.max(1, obj.grid.rows);
    } else {
      for (const h of obj.holds) {
        if (Number.isFinite(h.grid_x)) cols = Math.max(cols, h.grid_x + 1);
        if (Number.isFinite(h.grid_y)) rows = Math.max(rows, h.grid_y + 1);
      }
    }

    const holds = new Map();
    let maxNum = 0;
    for (const h of obj.holds) {
      if (!Number.isInteger(h.grid_x) || !Number.isInteger(h.grid_y))
        throw new Error(`hold ${h.hold_id || '?'} has non-integer grid coords`);
      const k = key(h.grid_x, h.grid_y);
      if (holds.has(k)) throw new Error(`duplicate cell ${k}`);
      holds.set(k, {
        hold_id: String(h.hold_id),
        grid_x: h.grid_x,
        grid_y: h.grid_y,
        hold_type: HOLD_TYPES.includes(h.hold_type) ? h.hold_type : 'jug',
        orientation_deg: ((Number(h.orientation_deg) || 0) % 360 + 360) % 360,
        size: SIZES.includes(h.size) ? h.size : 'medium',
        color: typeof h.color === 'string' && h.color ? h.color : (TYPE_COLORS[h.hold_type] || '#22c55e'),
        is_start: !!h.is_start,
        is_finish: !!h.is_finish,
      });
      const m = /^h_(\d+)$/.exec(String(h.hold_id));
      if (m) maxNum = Math.max(maxNum, Number(m[1]));
    }

    state.cols = cols; state.rows = rows;
    state.holds = holds;
    state.selectedKey = null;
    state.wallId = obj.wall_id;
    state.nextNum = maxNum + 1;
    els.cols.value = cols; els.rows.value = rows;
    els.wallId.value = obj.wall_id;
  }

  function onCellClick(x, y, ev) {
    const k = key(x, y);
    const existing = state.holds.get(k);

    if (ev.shiftKey && state.selectedKey && !existing) {
      const src = state.holds.get(state.selectedKey);
      if (src) {
        const clone = { ...src, hold_id: nextHoldId(), grid_x: x, grid_y: y };
        state.holds.set(k, clone);
        state.selectedKey = k;
        render();
        setStatus(`Cloned to (${x}, ${y}).`, 'ok');
        return;
      }
    }

    if (existing) {
      state.selectedKey = k;
      writeHoldToPalette(existing);
      render();
      return;
    }

    const hold = { hold_id: nextHoldId(), grid_x: x, grid_y: y, ...paletteHold() };
    state.holds.set(k, hold);
    state.selectedKey = k;
    render();
    setStatus(`Placed ${hold.hold_id} at (${x}, ${y}).`, 'ok');
  }

  function onCellRightClick(x, y) {
    const h = state.holds.get(key(x, y));
    if (!h) return;
    h.orientation_deg = (h.orientation_deg + 15) % 360;
    if (state.selectedKey === key(x, y)) writeHoldToPalette(h);
    render();
  }

  function applyPaletteToSelected() {
    if (!state.selectedKey) { setStatus('Nothing selected.', 'error'); return; }
    const h = state.holds.get(state.selectedKey);
    if (!h) return;
    Object.assign(h, paletteHold());
    render();
    setStatus(`Updated ${h.hold_id}.`, 'ok');
  }

  function removeSelected() {
    if (!state.selectedKey) return;
    const h = state.holds.get(state.selectedKey);
    state.holds.delete(state.selectedKey);
    state.selectedKey = null;
    render();
    if (h) setStatus(`Removed ${h.hold_id}.`, 'ok');
  }

  function moveSelection(dx, dy) {
    if (!state.selectedKey) return;
    const [x, y] = state.selectedKey.split(',').map(Number);
    const nx = x + dx, ny = y + dy;
    if (!inBounds(nx, ny)) return;
    state.selectedKey = key(nx, ny);
    const h = state.holds.get(state.selectedKey);
    if (h) writeHoldToPalette(h);
    render();
  }

  function resizeGrid() {
    const c = Math.max(1, Math.min(40, Number(els.cols.value) || 1));
    const r = Math.max(1, Math.min(40, Number(els.rows.value) || 1));
    let dropped = 0;
    for (const k of [...state.holds.keys()]) {
      const [x, y] = k.split(',').map(Number);
      if (x >= c || y >= r) { state.holds.delete(k); dropped += 1; }
    }
    state.cols = c; state.rows = r;
    if (state.selectedKey) {
      const [x, y] = state.selectedKey.split(',').map(Number);
      if (x >= c || y >= r) state.selectedKey = null;
    }
    render();
    setStatus(dropped ? `Resized — dropped ${dropped} out-of-bounds hold(s).` : 'Resized.', dropped ? 'error' : 'ok');
  }

  async function refreshWallList() {
    try {
      const r = await fetch('/api/walls');
      const data = await r.json();
      const cur = els.wallList.value;
      els.wallList.innerHTML = '<option value="">— saved walls —</option>' +
        (data.walls || []).map((id) => `<option value="${id}">${id}</option>`).join('');
      if (cur && (data.walls || []).includes(cur)) els.wallList.value = cur;
    } catch (e) {
      setStatus(`List failed: ${e.message}`, 'error');
    }
  }

  async function saveWall() {
    const id = els.wallId.value.trim();
    if (!/^[A-Za-z0-9_\-]{1,64}$/.test(id)) {
      setStatus('Wall ID must be 1–64 chars: letters, digits, "-" or "_".', 'error');
      return;
    }
    state.wallId = id;
    const body = currentWall();
    try {
      const r = await fetch(`/api/walls/${encodeURIComponent(id)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      setStatus(`Saved "${id}" (${body.holds.length} holds).`, 'ok');
      refreshWallList();
    } catch (e) {
      setStatus(`Save failed: ${e.message}`, 'error');
    }
  }

  async function loadWall(id) {
    if (!id) return;
    try {
      const r = await fetch(`/api/walls/${encodeURIComponent(id)}`);
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      loadWallFromObject(data);
      render();
      setStatus(`Loaded "${id}".`, 'ok');
    } catch (e) {
      setStatus(`Load failed: ${e.message}`, 'error');
    }
  }

  async function deleteWall() {
    const id = els.wallId.value.trim();
    if (!id) return;
    if (!confirm(`Delete saved wall "${id}"? This cannot be undone.`)) return;
    try {
      const r = await fetch(`/api/walls/${encodeURIComponent(id)}`, { method: 'DELETE' });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      setStatus(`Deleted "${id}".`, 'ok');
      refreshWallList();
    } catch (e) {
      setStatus(`Delete failed: ${e.message}`, 'error');
    }
  }

  function newWall() {
    state.holds.clear();
    state.selectedKey = null;
    state.nextNum = 1;
    state.wallId = els.wallId.value.trim() || 'my-wall';
    render();
    setStatus('New empty wall.', 'ok');
  }

  function applyJson() {
    try {
      const obj = JSON.parse(els.json.value);
      loadWallFromObject(obj);
      render();
      setStatus('JSON applied.', 'ok');
    } catch (e) {
      setStatus(`JSON error: ${e.message}`, 'error');
    }
  }
  function formatJson() { renderJson(); setStatus('Formatted.', 'ok'); }

  function downloadJson() {
    const id = state.wallId || 'wall';
    const blob = new Blob([JSON.stringify(currentWall(), null, 2) + '\n'], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${id}.json`; a.click();
    URL.revokeObjectURL(url);
  }

  function uploadJson(file) {
    const reader = new FileReader();
    reader.onload = () => {
      els.json.value = String(reader.result || '');
      applyJson();
    };
    reader.readAsText(file);
  }

  function bind() {
    els.btnResize.addEventListener('click', resizeGrid);
    els.btnApply.addEventListener('click', applyPaletteToSelected);
    els.btnRemove.addEventListener('click', removeSelected);
    els.orient.addEventListener('input', () => {
      els.orientVal.textContent = `${els.orient.value}°`;
    });
    els.holdType.addEventListener('change', () => {
      // If user just changed type and palette color matches a previous default, update color too.
      const def = TYPE_COLORS[els.holdType.value];
      if (Object.values(TYPE_COLORS).includes(els.color.value)) els.color.value = def;
    });

    els.btnSave.addEventListener('click', saveWall);
    els.btnLoad.addEventListener('click', () => {
      const pick = els.wallList.value || els.wallId.value.trim();
      if (pick) { els.wallId.value = pick; loadWall(pick); }
    });
    els.wallList.addEventListener('change', () => {
      if (els.wallList.value) els.wallId.value = els.wallList.value;
    });
    els.btnNew.addEventListener('click', newWall);
    els.btnDelete.addEventListener('click', deleteWall);
    els.wallId.addEventListener('change', () => { state.wallId = els.wallId.value.trim(); renderJson(); });

    els.btnApplyJson.addEventListener('click', applyJson);
    els.btnFormat.addEventListener('click', formatJson);
    els.btnDownload.addEventListener('click', downloadJson);
    els.fileUpload.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) uploadJson(f);
      e.target.value = '';
    });

    document.addEventListener('keydown', (e) => {
      const tag = document.activeElement && document.activeElement.tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
      if (e.key === 'Delete' || e.key === 'Backspace') { removeSelected(); e.preventDefault(); }
      else if (e.key === 'r' || e.key === 'R') {
        if (!state.selectedKey) return;
        const h = state.holds.get(state.selectedKey);
        if (!h) return;
        h.orientation_deg = (h.orientation_deg + 15) % 360;
        writeHoldToPalette(h);
        render();
      }
      else if (e.key === 'ArrowUp')    { moveSelection(0, 1); e.preventDefault(); }
      else if (e.key === 'ArrowDown')  { moveSelection(0, -1); e.preventDefault(); }
      else if (e.key === 'ArrowLeft')  { moveSelection(-1, 0); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { moveSelection(1, 0); e.preventDefault(); }
    });
  }

  async function boot() {
    bind();
    render();
    await refreshWallList();
    try {
      const r = await fetch('/api/walls');
      const data = await r.json();
      const ids = data.walls || [];
      if (ids.includes('example-v2-boulder')) {
        await loadWall('example-v2-boulder');
      } else if (ids.length) {
        await loadWall(ids[0]);
      }
    } catch (_e) {
      /* empty wall is fine */
    }
  }

  boot();
})();
