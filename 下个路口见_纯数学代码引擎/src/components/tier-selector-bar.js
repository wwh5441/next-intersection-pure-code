// 7-Tier Rate-Distortion Live Switcher Component (for Unified Studio)

import { formatByteSize } from './header-metrics.js';

export function mountTierSelectorBar(container, store, onSelectTier) {
  if (!container) return;
  const initSt = store.getState();
  if (initSt.studioMode !== 'unified' || !initSt.tiersCatalog.length) {
    container.style.display = 'none';
    return;
  }

  container.innerHTML = `
    <div class="canvas-header">
      <span><strong>七级纯数学压缩阶梯实时切换矩阵（V9.4.1 → V9.4.7）</strong> · 播放中点击任意档位，按 15ms 相位锁定无缝接力同位置盲听对比</span>
      <span id="tier-switch-hint" class="status-hint">按需动态加载 ES 模块 · 零中断切换</span>
    </div>
    <div class="tier-cards-grid" id="tier-cards-grid"></div>
  `;

  const gridEl = container.querySelector('#tier-cards-grid');
  const hintEl = container.querySelector('#tier-switch-hint');
  const btnMap = new Map();

  for (const item of initSt.tiersCatalog) {
    const btn = document.createElement('button');
    btn.className = 'tier-card-btn';
    btn.dataset.tierId = item.tierId;
    btn.innerHTML = `
      <div class="tier-card-top">
        <span>${item.badgeIndex} ${item.versionCode}</span>
        <span>${item.passbandShort}</span>
      </div>
      <div class="tier-card-name">${item.tierName}</div>
      <div class="tier-card-size">明文 ${formatByteSize(item.totalAudioCorePlainJsBytes)}</div>
      <div class="tier-card-desc">Gzip ${formatByteSize(item.totalAudioCoreGzipBytes)} · ${item.shortTag}</div>
    `;
    btn.addEventListener('click', () => {
      if (onSelectTier) onSelectTier(item.tierId);
    });
    btnMap.set(item.tierId, btn);
    gridEl.appendChild(btn);
  }

  function syncCards(state) {
    const busy = Boolean(state.loadingTierId);
    for (const [tid, btn] of btnMap.entries()) {
      btn.classList.toggle('active', tid === state.activeTierId);
      btn.disabled = busy;
    }
    if (state.loadingTierId) {
      const target = state.tiersCatalog.find((x) => x.tierId === state.loadingTierId);
      hintEl.textContent = `正在加载 ${target ? target.versionCode + ' ' + target.tierName : state.loadingTierId}...`;
    } else {
      hintEl.textContent = `已缓存 ${state.loadedTierIds.length} / ${state.tiersCatalog.length} 档 · 点击同秒盲听对比`;
    }
  }

  syncCards(store.getState());
  store.subscribe((state, keys) => {
    if (
      keys.includes('activeTierId') ||
      keys.includes('loadingTierId') ||
      keys.includes('loadedTierIds')
    ) {
      syncCards(state);
    }
  });
}
