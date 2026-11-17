export class SyllableAtlasPanelComponent {
  constructor(rootEl, { atlasCatalog, onSelectSyllable, onJumpToEvent }) {
    this.root = rootEl;
    this.prototypes = [...(atlasCatalog.prototypes || [])].sort((a, b) => b.count - a.count);
    this.stats = atlasCatalog.stats;
    this.onSelectSyllable = onSelectSyllable;
    this.onJumpToEvent = onJumpToEvent;
    this.uiEvents = [];
    this.selectedId = null;
    this._buildDom();
  }

  _buildDom() {
    const chips = this.prototypes
      .map(
        (p) => `
      <button class="atlas-chip" data-sid="${p.id}">
        <span>${p.pinyin}</span>
        <span class="atlas-chip-cnt">×${p.count}</span>
      </button>
    `
      )
      .join('');

    this.root.innerHTML = `
      <p class="atlas-summary">
        全曲共检测 <strong>${this.stats.totalEvents}</strong> 次人声发声事件（歌词 ${this.stats.lyricEvents} 字 + 前/间/尾奏 Hook ${this.stats.hookEvents} 次），归并为 <strong>${this.stats.uniqueSyllablePrototypes}</strong> 个共享音节声韵原型。点击任意音节可查看并跳转它在全曲不同段落的每次演唱表达：
      </p>
      <div class="atlas-chip-grid">${chips}</div>
      <div id="atlas-occ-box" class="atlas-occurrences">点击上方音节胶囊（如 di、da、ji、xia、jian）定位它在全曲的历次演唱差异</div>
    `;

    this.occBox = this.root.querySelector('#atlas-occ-box');
    this.chipBtns = [...this.root.querySelectorAll('.atlas-chip')];
    for (const btn of this.chipBtns) {
      btn.addEventListener('click', () => {
        const sid = Number(btn.dataset.sid);
        this.selectedId = this.selectedId === sid ? null : sid;
        this.onSelectSyllable(this.selectedId);
        this._renderOccurrences();
      });
    }
  }

  setEvents(uiEvents) {
    this.uiEvents = uiEvents || [];
    this._renderOccurrences();
  }

  _renderOccurrences() {
    for (const btn of this.chipBtns) {
      btn.classList.toggle('selected', Number(btn.dataset.sid) === this.selectedId);
    }
    if (this.selectedId === null) {
      this.occBox.innerHTML = '点击上方音节胶囊（如 di、da、ji、xia、jian）定位它在全曲的历次演唱差异';
      return;
    }
    const matches = this.uiEvents.filter((e) => e.syllableId === this.selectedId);
    const proto = this.prototypes.find((p) => p.id === this.selectedId);
    const btns = matches
      .slice(0, 18)
      .map(
        (ev) =>
          `<button class="occ-jump-btn" data-sec="${ev.startSec}">${ev.displayText} @ ${ev.startSec.toFixed(1)}s</button>`
      )
      .join('');
    this.occBox.innerHTML = `<span>音节 <strong>[${proto?.pinyin}]</strong> 共演唱 ${matches.length} 次（点击试听不同段落发音方式）：</span> ${btns}`;
    this.occBox.querySelectorAll('.occ-jump-btn').forEach((b) => {
      b.addEventListener('click', () => this.onJumpToEvent(Number(b.dataset.sec)));
    });
  }
}
