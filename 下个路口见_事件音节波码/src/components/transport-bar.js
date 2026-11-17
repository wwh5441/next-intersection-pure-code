function fmtSec(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
}

export class TransportBarComponent {
  constructor(rootEl, { profilesMeta, sizes, lyrics, onTogglePlay, onSeek, onSelectProfile }) {
    this.root = rootEl;
    this.profilesMeta = profilesMeta;
    this.sizes = sizes;
    this.lyrics = lyrics.lines || [];
    this.onTogglePlay = onTogglePlay;
    this.onSeek = onSeek;
    this.onSelectProfile = onSelectProfile;
    this._buildDom();
  }

  _buildDom() {
    const tabsHtml = this.profilesMeta.available
      .map((pid) => {
        const kb = this.sizes.profiles[pid]?.totalKB ?? '';
        const label = pid === 'ultra' ? `极简音节档 (${kb} KB)` : `精修事件档 (${kb} KB)`;
        return `<button class="profile-tab" data-profile="${pid}">${label}</button>`;
      })
      .join('');

    this.root.innerHTML = `
      <div class="transport-top">
        <div class="transport-controls">
          <button id="master-play-btn" class="play-btn" disabled>加载中…</button>
          <span id="time-readout" class="time-readout">00:00 / 03:30</span>
        </div>
        <div class="profile-switcher" id="profile-tabs">${tabsHtml}</div>
      </div>
      <div class="scrubber-wrap">
        <input id="timeline-scrubber" class="scrubber-slider" type="range" min="0" max="210.373" step="0.02" value="0" />
      </div>
      <div class="lyric-stage">
        <div id="lyric-line-display" class="lyric-current">准备就绪 · 点击播放聆听纯事件数学衍生人声</div>
        <div id="active-event-pill" class="active-event-pill">待触发事件</div>
      </div>
    `;

    this.playBtn = this.root.querySelector('#master-play-btn');
    this.timeEl = this.root.querySelector('#time-readout');
    this.scrubber = this.root.querySelector('#timeline-scrubber');
    this.lyricEl = this.root.querySelector('#lyric-line-display');
    this.eventPill = this.root.querySelector('#active-event-pill');
    this.tabBtns = [...this.root.querySelectorAll('.profile-tab')];

    this.playBtn.addEventListener('click', () => this.onTogglePlay());
    this.scrubber.addEventListener('input', (e) => this.onSeek(Number(e.target.value)));
    for (const btn of this.tabBtns) {
      btn.addEventListener('click', () => this.onSelectProfile(btn.dataset.profile));
    }
  }

  update(state, activeEvent) {
    this.playBtn.disabled = state.loading;
    this.playBtn.textContent = state.loading
      ? '合成中…'
      : state.playing
        ? '暂停播放'
        : '播放人声';

    this.scrubber.max = String(state.duration || 210.373);
    if (document.activeElement !== this.scrubber) {
      this.scrubber.value = String(state.currentTime);
    }
    this.timeEl.textContent = `${fmtSec(state.currentTime)} / ${fmtSec(state.duration)}`;

    for (const btn of this.tabBtns) {
      btn.classList.toggle('active', btn.dataset.profile === state.profile);
    }

    // Active lyric line lookup
    const t = state.currentTime;
    let curLine = '';
    for (let i = 0; i < this.lyrics.length; i++) {
      const ln = this.lyrics[i];
      if (t >= ln.startSec && t < ln.endSec) {
        curLine = ln.text;
        break;
      }
    }
    this.lyricEl.textContent = curLine || (t < 22.8 ? '♪ 前奏律动段（稀疏 DI DI DA 事件触发，空白段 0 字节）' : '♪ 间奏 / 呼吸留白段（无发声事件，0 字节开销）');

    if (activeEvent) {
      const hz = activeEvent.pitchPreview?.length ? `${activeEvent.pitchPreview[0]} Hz` : '清辅音/气声';
      this.eventPill.textContent = `#${activeEvent.index + 1} 「${activeEvent.displayText}」 → 音节 [${activeEvent.pinyin}] · ${hz}`;
    } else {
      this.eventPill.textContent = '静默段 · 0 事件运行';
    }
  }
}
