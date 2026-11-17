// V9R1 4-Tab Live Dual-Dictionary & Physics Inspector (`src/components/dictionary-inspector.js`)
import { SYLLABLE_ATLAS } from '../../models/syllable-atlas.js';
import { BAND_MOTIF_ATLAS } from '../../models/band-motif-atlas.js';
import { V9R1_TIERS } from '../../data/tiers-registry.js';
import { V9R1_PHYSICS_EQUATIONS } from '../../data/benchmarks.js';
import { studioStore } from '../state/studio-store.js';

export function mountDictionaryInspector(container) {
  const sylEntries = Object.entries(SYLLABLE_ATLAS);

  container.innerHTML = `
    <div class="section-heading">
      <div>
        <h2 class="section-title">纯代码双重字典与可执行乐谱透视镜 (Live ES Module Inspector)</h2>
        <div class="section-sub">所有参数均直接读取自内存中的纯 JS 字典与乐谱模块，无任何二进制块或隐藏采样</div>
      </div>
    </div>

    <div class="inspector-tabs" id="v9r1-insp-tabs">
      <button type="button" class="insp-tab-btn is-active" data-tab="syllables">1. 97音节声韵字典 (${sylEntries.length})</button>
      <button type="button" class="insp-tab-btn" data-tab="motifs">2. 86小节伴奏原型字典 (18+22+22)</button>
      <button type="button" class="insp-tab-btn" data-tab="score">3. 可执行乐谱与稀疏残差源码</button>
      <button type="button" class="insp-tab-btn" data-tab="equations">4. 10轨物理声学微分方程</button>
    </div>

    <div id="v9r1-insp-body"></div>
  `;

  const tabsWrap = container.querySelector('#v9r1-insp-tabs');
  const bodyEl = container.querySelector('#v9r1-insp-body');
  const tabBtns = Array.from(tabsWrap.querySelectorAll('.insp-tab-btn'));

  tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      studioStore.setState({ inspectorTab: btn.dataset.tab });
    });
  });

  let lastTab = null;
  let lastTier = null;

  function renderSyllablesTab() {
    const filterVal = studioStore.getState().syllableFilter;
    bodyEl.innerHTML = `
      <div class="insp-toolbar">
        <input type="search" class="insp-search" id="v9r1-syl-search" placeholder="筛选拼音或声母类别 (如 jian / sibilant / plosive)..." value="${filterVal}" />
        <span class="section-sub">每个音节含 声母3锚点×40阶ERB + 韵母5锚点×40阶ERB (零行无损裁剪 0.00dB)</span>
      </div>
      <div class="syl-grid" id="v9r1-syl-grid"></div>
    `;

    const searchInput = bodyEl.querySelector('#v9r1-syl-search');
    const gridEl = bodyEl.querySelector('#v9r1-syl-grid');

    const populate = (q) => {
      const kw = q.trim().toLowerCase();
      const filtered = sylEntries.filter(([py, tup]) => {
        if (!kw) return true;
        return py.includes(kw) || tup[0].includes(kw) || tup[1].includes(kw) || tup[2].includes(kw);
      });
      gridEl.innerHTML = filtered
        .map(([py, tup]) => {
          const [cls, ini, fin, cMcep, , vMcep] = tup;
          const c0 = cMcep[0] && cMcep[0][0] !== undefined ? cMcep[0][0] : 0;
          const v0 = vMcep[2] && vMcep[2][0] !== undefined ? vMcep[2][0] : 0;
          return `
            <div class="syl-card">
              <div class="syl-card-top">
                <span>${py}</span>
                <span class="syl-class-tag">${cls}</span>
              </div>
              <div class="syl-decomp">声母 <strong>${ini}</strong> + 韵母 <strong>${fin}</strong></div>
              <div class="syl-decomp">锚点 ${cMcep.length}+${vMcep.length} · C0(${c0}→${v0})</div>
            </div>
          `;
        })
        .join('');
    };

    populate(filterVal);
    searchInput.addEventListener('input', () => {
      studioStore.getState().syllableFilter = searchInput.value;
      populate(searchInput.value);
    });
  }

  function renderMotifsTab() {
    const dMotifs = BAND_MOTIF_ATLAS.drums.motifs;
    const bMotifs = BAND_MOTIF_ATLAS.bass.motifs;
    const hMotifs = BAND_MOTIF_ATLAS.harmony.motifs;

    bodyEl.innerHTML = `
      <div class="motif-atlas-cols">
        <div class="motif-col-box">
          <div class="motif-col-title">鼓组原型字典 · ${dMotifs.length} Motifs (245帧/小节)</div>
          ${dMotifs
            .map(
              (m) => `
              <div class="motif-item">
                <span><strong>D:M${m.id}</strong> (取自 Bar #${m.srcBar + 1})</span>
                <span>${m.tag} · K:${m.kick[1].length}</span>
              </div>`
            )
            .join('')}
        </div>
        <div class="motif-col-box">
          <div class="motif-col-title">贝斯原型字典 · ${bMotifs.length} Motifs (122.5帧/小节)</div>
          ${bMotifs
            .map(
              (m) => `
              <div class="motif-item">
                <span><strong>B:M${m.id}</strong> (取自 Bar #${m.srcBar + 1})</span>
                <span>${m.tag} · F0:${m.f0[1].length}</span>
              </div>`
            )
            .join('')}
        </div>
        <div class="motif-col-box">
          <div class="motif-col-title">吉他/键盘复调字典 · ${hMotifs.length} Motifs (98帧/小节)</div>
          ${hMotifs
            .map((m) => {
              const nNotes = Object.keys(m.notes || {}).length;
              return `
              <div class="motif-item">
                <span><strong>H:M${m.id}</strong> (取自 Bar #${m.srcBar + 1})</span>
                <span>${m.tag} · ${nNotes}音复调</span>
              </div>`;
            })
            .join('')}
        </div>
      </div>
    `;
  }

  function renderScoreTab(tierId) {
    const tier = V9R1_TIERS[tierId] || Object.values(V9R1_TIERS)[0];
    const sampleVocal = tier.vocalScore.slice(0, 5).map((ev) => JSON.stringify(ev)).join(',\n  ');
    const sampleBars = tier.bandArrangement.bars
      .slice(0, 12)
      .map((b, idx) => `  /* Bar #${String(idx + 1).padStart(2, '0')} */ [${b.join(', ')}]`)
      .join(',\n');
    const resKeys = Object.keys(tier.bandArrangement.residuals || {});

    bodyEl.innerHTML = `
      <pre class="code-preview-pre">// 当前档位: ${tier.vocalProfile.fullLabel}
// 1) 可执行人声连音乐谱 (${tier.vocalScore.length} 个音节事件 · ${tier.vocalProfile.pcaDims} 维李宇春 PCA 轨迹):
export const VOCAL_SCORE_${tierId.toUpperCase()}_SAMPLE = [
  ${sampleVocal}
  // ... 共 ${tier.vocalScore.length} 条纯数学事件元组
];

// 2) 86 小节伴奏原型调度表 [drumId, drumGainQ64, bassId, bassGainQ64, harmId, harmGainQ64]
//    稀疏过门/微变奏残差小节数: ${resKeys.length} / 86 (${resKeys.slice(0, 12).map((k) => 'Bar#' + (Number(k) + 1)).join(', ')}${resKeys.length > 12 ? ' ...' : ''})
export const BAND_ARRANGEMENT_${tierId.toUpperCase()}_BARS_01_12 = [
${sampleBars}
];</pre>
    `;
  }

  function renderEquationsTab() {
    bodyEl.innerHTML = `
      <div class="eq-grid">
        ${V9R1_PHYSICS_EQUATIONS.map(
          (eq) => `
          <div class="eq-card">
            <h4>${eq.title}</h4>
            <div class="eq-formula-line">${eq.formula}</div>
            <p class="eq-desc">${eq.desc}</p>
          </div>`
        ).join('')}
      </div>
    `;
  }

  const sync = (st) => {
    if (st.inspectorTab === lastTab && st.tierId === lastTier) return;
    lastTab = st.inspectorTab;
    lastTier = st.tierId;

    tabBtns.forEach((b) => {
      b.classList.toggle('is-active', b.dataset.tab === st.inspectorTab);
    });

    if (st.inspectorTab === 'syllables') renderSyllablesTab();
    else if (st.inspectorTab === 'motifs') renderMotifsTab();
    else if (st.inspectorTab === 'score') renderScoreTab(st.tierId);
    else renderEquationsTab();
  };

  sync(studioStore.getState());
  studioStore.subscribe(sync);
}
