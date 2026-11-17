// Interactive 97-Syllable Atlas & Event Jump Inspector (`src/components/syllable-atlas-inspector.js`)
export function mountSyllableAtlasInspector(container, atlasCatalog, store, onJumpSec) {
  const sorted = [...atlasCatalog].sort((a, b) => b.count - a.count);
  let uiEvents = [];

  container.innerHTML = `
    <div class="panel-head">
      <h2 class="panel-title">V8.5 全曲去重共享音节声韵字典 (97 个普通话音节原型驱动全曲 286 次发声事件)</h2>
      <span class="panel-badge">点击任意音节高亮全曲复用位置并瞬移试听</span>
    </div>
    <div class="atlas-layout">
      <div class="syllable-grid" id="syl-grid">
        ${sorted
          .map(
            (p) => `
          <button class="syl-pill" data-sid="${p.id}">
            <span>${p.pinyin}</span>
            <span class="syl-count">×${p.count}</span>
          </button>
        `
          )
          .join('')}
      </div>
      <div class="occ-box" id="occ-box">
        <span class="metric-label">选择左侧任意音节原型（如高频复用 di ×26、da ×14），查看并点击跳转全曲每一次演唱实例：</span>
        <div class="occ-list" id="occ-list"></div>
      </div>
    </div>
  `;

  const gridEl = container.querySelector('#syl-grid');
  const occList = container.querySelector('#occ-list');

  function renderOccurrences(sid) {
    for (const pill of gridEl.querySelectorAll('.syl-pill')) {
      pill.classList.toggle('selected', Number(pill.dataset.sid) === sid);
    }
    if (sid === null || sid === undefined) {
      occList.innerHTML = '';
      return;
    }
    const matches = uiEvents.filter((ev) => ev.syllableId === sid);
    occList.innerHTML = matches
      .map(
        (ev) => `
        <button class="occ-jump-btn mono" data-sec="${ev.startSec.toFixed(3)}">
          ${ev.text} (${ev.startSec.toFixed(1)}s)
        </button>
      `
      )
      .join('');
  }

  gridEl.addEventListener('click', (e) => {
    const pill = e.target.closest('.syl-pill');
    if (!pill) return;
    const sid = Number(pill.dataset.sid);
    const nextSid = store.getState().highlightedSyllableId === sid ? null : sid;
    store.setState({ highlightedSyllableId: nextSid });
    renderOccurrences(nextSid);
  });

  occList.addEventListener('click', (e) => {
    const btn = e.target.closest('.occ-jump-btn');
    if (!btn) return;
    onJumpSec(Number(btn.dataset.sec));
  });

  return {
    setEvents(evs) {
      uiEvents = evs || [];
      if (uiEvents.length > 0 && store.getState().highlightedSyllableId === null) {
        const topSid = sorted[0]?.id ?? 0;
        store.setState({ highlightedSyllableId: topSid });
        renderOccurrences(topSid);
      } else {
        renderOccurrences(store.getState().highlightedSyllableId);
      }
    },
  };
}
