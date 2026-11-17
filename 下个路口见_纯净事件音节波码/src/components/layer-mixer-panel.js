export class LayerMixerPanelComponent {
  constructor(rootEl, { layers, onChangeLayer }) {
    this.root = rootEl;
    this.layers = layers;
    this.onChangeLayer = onChangeLayer;
    this.meterEls = new Map();
    this._buildDom();
  }

  _buildDom() {
    this.root.innerHTML = this.layers
      .map(
        (layer) => `
      <div class="layer-row" data-layer-id="${layer.id}">
        <div>
          <div class="layer-info-title">
            <span class="layer-dot" style="background:${layer.color}"></span>
            <span>0${layer.index} · ${layer.name}</span>
          </div>
          <div class="layer-sub">${layer.subtitle}</div>
        </div>
        <div class="layer-actions">
          <button class="mini-toggle btn-solo" data-id="${layer.id}" title="独听此数学分量">S</button>
          <button class="mini-toggle btn-mute" data-id="${layer.id}" title="静音此数学分量">M</button>
          <input class="layer-gain-slider" type="range" min="0" max="1.5" step="0.02" value="${layer.defaultGain}" data-id="${layer.id}" />
        </div>
        <div class="layer-meter-track">
          <div class="layer-meter-fill" id="meter-${layer.id}" style="background:${layer.color}"></div>
        </div>
      </div>
    `
      )
      .join('');

    for (const layer of this.layers) {
      this.meterEls.set(layer.id, this.root.querySelector(`#meter-${layer.id}`));
    }

    this.root.querySelectorAll('.btn-solo').forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.dataset.id;
        const active = !btn.classList.contains('solo-on');
        this.onChangeLayer(id, { solo: active });
      });
    });

    this.root.querySelectorAll('.btn-mute').forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.dataset.id;
        const active = !btn.classList.contains('mute-on');
        this.onChangeLayer(id, { muted: active });
      });
    });

    this.root.querySelectorAll('.layer-gain-slider').forEach((sl) => {
      sl.addEventListener('input', (e) => {
        this.onChangeLayer(sl.dataset.id, { gain: Number(e.target.value) });
      });
    });
  }

  syncButtons(layerSnap) {
    for (const [id, st] of Object.entries(layerSnap)) {
      const row = this.root.querySelector(`.layer-row[data-layer-id="${id}"]`);
      if (!row) continue;
      row.querySelector('.btn-solo').classList.toggle('solo-on', Boolean(st.solo));
      row.querySelector('.btn-mute').classList.toggle('mute-on', Boolean(st.muted));
    }
  }

  updateMeters(levels) {
    for (const [id, val] of Object.entries(levels)) {
      const el = this.meterEls.get(id);
      if (el) {
        el.style.width = `${Math.round(Math.min(1, val) * 100)}%`;
      }
    }
  }
}
