// CabronWEC (CWC-5X) Mathematical Equation Inspector, Multi-Tier Compression Proof & Cue Sheet Deck
import { CWC_LYRICS } from '../../data/cwc-lyrics.js';
import { CWC_EDITIONS, CWC_WEC_PLAYER } from '../../data/cwc-models.js';
import { CWC_HOP_M, CWC_SAMPLE_RATE } from '../dsp/mdct-core.js';
import { cwcEngine } from '../modules/cwc-audio-engine.js';
import { cwcStore } from '../state/cwc-store.js';

function formatClock(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = (sec % 60).toFixed(2).padStart(5, '0');
  return `${m}:${s}`;
}

function fmtSize(bytes) {
  if (bytes >= 1000000) {
    return `${(bytes / 1000000).toFixed(2)} MB (${(bytes / 1048576).toFixed(2)} MiB)`;
  }
  return `${(bytes / 1000).toFixed(1)} KB (${(bytes / 1024).toFixed(1)} KiB)`;
}

export function mountDeckInspector(deckBodyEl) {
  let cachedMetrics = null;

  fetch('/data/cwc-metrics.json')
    .then((r) => (r.ok ? r.json() : null))
    .then((m) => {
      cachedMetrics = m;
      if (cwcStore.getState().activeDeckTab === 'metrics') render();
    })
    .catch(() => {});

  const render = () => {
    const st = cwcStore.getState();

    if (st.activeDeckTab === 'equations') {
      const fIdx = st.isPlaying ? st.currentFrame : st.inspectedFrame;
      const info = cwcEngine.inspectFrame(fIdx, st.activeTier);
      const frameSec = (fIdx * CWC_HOP_M) / CWC_SAMPLE_RATE;
      const ed = CWC_EDITIONS[st.activeTier] || CWC_EDITIONS['1m'];
      const inspectorLabel =
        st.activeTier === 'wec'
          ? `WEC (${st.loadedWecMeta?.tierLabel || '500K'})`
          : ed.shortLabel;

      deckBodyEl.innerHTML = `
        <div class="cwc-eq-grid">
          <div class="cwc-eq-block">
            <h3 class="cwc-eq-title">CWC-5X Pure-JS &amp; "CWC5" .wec Container Pipeline</h3>
            <pre class="cwc-code-pre">1. Adaptive Edler Bridge MDCT-TDAC (N_L = 4096 ↔ 8 × N_S = 512):
   X[k] = √(2/M) · Σ w_type[n] · x[n] · cos((π/M)(n + 0.5 + M/2)(k + 0.5))

2. Coherent Block-Aligned Air Extension + Least-Squares Gain + M/S Coupling:
   • &lt;500K / 319K .wec: 0–4.4 kHz Tonal + 4.4–19.6 kHz Coherent Air (Zero Static)
   • &lt;1MB  / 730K .wec: 0–6.4 kHz Tonal + 6.4–19.6 kHz Coherent Air (Zero Static)
   • Hi-Fi / 1.99M .wec: 0–12 kHz Full Tonal + Coherent Air + 4-Band Side

3. "CWC5" .wec Lossless Binary Container + wec-play.mjs (37.7 KB):
   32B Header + CRC-32 + 2D Scale Pred + 48-Ctx Inter-Frame Harmonic Coding
   (Bit-exact 0.000000 sample diff vs Pure-JS chunks, -31.3% smaller).</pre>
          </div>
          <div class="cwc-eq-block">
            <div class="cwc-stem-top">
              <h3 class="cwc-eq-title">${inspectorLabel} CSR Inspector · Frame #${fIdx} (${formatClock(frameSec)})</h3>
              <button type="button" class="cwc-preset-btn" id="btn-inspect-chorus">Jump Chorus #1250</button>
            </div>
            ${
              info
                ? `
                  <div class="cwc-eq-meta tabular">
                    Window: ${info.winName} · Active Tonal Atoms: ${info.midNzCount} Mid + ${info.sideNzCount} Side
                  </div>
                  <pre class="cwc-code-pre">${info.atomsPreview || '(Harmonic SBR / Velvet diffuse frame — 0 sparse tonal atoms)'}</pre>
                  <div class="cwc-eq-meta tabular">Bands: ${info.scalesM.slice(0, 8).join(' · ')}</div>
                  <div class="cwc-eq-meta tabular">M/S Pan &amp; Air: ${info.pans.slice(0, 4).join(' · ')} · ${info.pnsM.join(' · ')}</div>
                `
                : `<div class="cwc-eq-meta tabular">Streaming ${inspectorLabel} data for Frame #${fIdx}...</div>`
            }
          </div>
        </div>
      `;
      const jumpBtn = deckBodyEl.querySelector('#btn-inspect-chorus');
      if (jumpBtn) {
        jumpBtn.addEventListener('click', () => {
          cwcEngine.seek((1250 * CWC_HOP_M) / CWC_SAMPLE_RATE);
        });
      }
      return;
    }

    if (st.activeDeckTab === 'metrics') {
      const ed1m = cachedMetrics?.editions?.['1m'] || CWC_EDITIONS['1m'];
      const ed500k = cachedMetrics?.editions?.['500k'] || CWC_EDITIONS['500k'];
      const edHifi = cachedMetrics?.editions?.hifi || CWC_EDITIONS.hifi;
      const wecPlayer = cachedMetrics?.wecPlayer || CWC_WEC_PLAYER;

      const curBaseKey =
        st.activeTier === 'wec' ? st.loadedWecMeta?.sourceTier || '500k' : st.activeTier;
      const curEd =
        curBaseKey === '500k' ? ed500k : curBaseKey === 'hifi' ? edHifi : ed1m;

      const s500Bytes = ed500k.standalone?.rawBytes || ed500k.cwcRawBytes;
      const w500Bytes = ed500k.wec?.bytes || 328207;
      const w1mBytes = ed1m.wec?.bytes || 735120;
      const wHifiBytes = edHifi.wec?.bytes || 1978301;
      const legacyRaw = cachedMetrics?.legacyCwc4xRawBytes || 7059651;
      const legacyGz = cachedMetrics?.legacyCwc4xGzipBytes || 4382044;
      const mp3Bytes = cachedMetrics?.mp3Bytes || 9620866;
      const wavBytes = cachedMetrics?.pcmWavBytes || 37109900;
      const activeWecSource = st.activeTier === 'wec' ? st.loadedWecMeta?.sourceTier : null;

      deckBodyEl.innerHTML = `
        <div class="cwc-kpi-row">
          <div class="cwc-kpi-cell">
            <span class="cwc-kpi-label">500K Lossless .wec Container (9R2 Style)</span>
            <span class="cwc-kpi-value tabular">${(w500Bytes / 1000).toFixed(1)}<span class="cwc-kpi-unit">KB (${(w500Bytes / 1024).toFixed(1)} KiB)</span></span>
            <span class="cwc-kpi-delta tabular">-30.9% vs ${(s500Bytes / 1000).toFixed(1)} KB .js · Bit-Exact 0 Diff</span>
          </div>
          <div class="cwc-kpi-cell">
            <span class="cwc-kpi-label">1M &amp; Hi-Fi Lossless .wec Containers</span>
            <span class="cwc-kpi-value tabular">${(w1mBytes / 1000).toFixed(1)}<span class="cwc-kpi-unit">KB · ${(wHifiBytes / 1000000).toFixed(2)} MB Hi-Fi</span></span>
            <span class="cwc-kpi-delta tabular">48-Ctx Inter-Frame Coding + CRC-32 Verified</span>
          </div>
          <div class="cwc-kpi-cell">
            <span class="cwc-kpi-label">wec-play.mjs Player + Fidelity (${curEd.shortLabel})</span>
            <span class="cwc-kpi-value tabular">${(wecPlayer.rawBytes / 1000).toFixed(1)}<span class="cwc-kpi-unit">KB Raw (${(wecPlayer.gzBytes / 1000).toFixed(1)} KB Gzip)</span></span>
            <span class="cwc-kpi-delta tabular">SNR ${curEd.snrLeftDb}/${curEd.snrRightDb} dB · Bark ${curEd.avg24BarkSnrDb} dB · ${curEd.realtimeSpeedupX}× RT</span>
          </div>
          <div class="cwc-kpi-cell">
            <span class="cwc-kpi-label">Download .wec Binary &amp; CLI Executable</span>
            <div class="cwc-preset-grid cwc-dl-grid">
              <a class="cwc-preset-btn cwc-dl-link tabular" href="/bundles/next-intersection-500k.wec" download="next-intersection-500k.wec">500k.wec (${Math.round(w500Bytes / 1000)}KB)</a>
              <a class="cwc-preset-btn cwc-dl-link tabular" href="/bundles/next-intersection-1m.wec" download="next-intersection-1m.wec">1m.wec (${Math.round(w1mBytes / 1000)}KB)</a>
              <a class="cwc-preset-btn cwc-dl-link tabular" href="/bundles/next-intersection-hifi.wec" download="next-intersection-hifi.wec">hifi.wec (${(wHifiBytes / 1000000).toFixed(2)}MB)</a>
              <a class="cwc-preset-btn cwc-dl-link tabular" href="/bundles/wec-play.mjs" download="wec-play.mjs">wec-play (${Math.round(wecPlayer.rawBytes / 1000)}KB)</a>
            </div>
          </div>
        </div>
        <div class="cwc-table-wrap">
          <table class="cwc-table">
            <thead>
              <tr>
                <th>Edition / Container (Click to Audition)</th>
                <th class="num">Raw File Size</th>
                <th class="num">Gzip / CRC-32</th>
                <th class="num">Boot / Unpack</th>
                <th class="num">Stereo SNR (L / R)</th>
                <th class="num">24-Bark SNR</th>
                <th>Acoustic &amp; Entropy Architecture</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Uncompressed 16-bit PCM WAV</td>
                <td class="num">${(wavBytes / 1048576).toFixed(2)} MiB</td>
                <td class="num">~33.80 MiB</td>
                <td class="num">N/A</td>
                <td class="num">Reference</td>
                <td class="num">Reference</td>
                <td>Raw PCM Buffer (~36 MB RAM)</td>
              </tr>
              <tr data-select-edition="ref" class="cwc-row-clickable ${st.audioSource === 'ref' ? 'cwc-row-highlight' : ''}">
                <td>Original 320 kbps MP3 Master</td>
                <td class="num">${(mp3Bytes / 1048576).toFixed(2)} MiB</td>
                <td class="num">${(mp3Bytes / 1048576).toFixed(2)} MiB</td>
                <td class="num">~150 ms</td>
                <td class="num">Reference</td>
                <td class="num">Reference</td>
                <td>Browser Native Codec (No Stems)</td>
              </tr>
              <tr>
                <td>CabronWEC (CWC-4X) Legacy Baseline</td>
                <td class="num">${(legacyRaw / 1048576).toFixed(2)} MiB</td>
                <td class="num">${(legacyGz / 1048576).toFixed(2)} MiB</td>
                <td class="num">520 KiB</td>
                <td class="num">15.70 / 16.12 dB</td>
                <td class="num">32.01 dB</td>
                <td>Fixed N=1024 + Base62 RLE</td>
              </tr>
              <tr data-select-edition="hifi" class="cwc-row-clickable ${st.audioSource === 'cwc' && st.activeTier === 'hifi' ? 'cwc-row-highlight' : ''}">
                <td>CWC-HiFi Pure-JS (8 Chunks)</td>
                <td class="num">${fmtSize(edHifi.cwcRawBytes)}</td>
                <td class="num">${fmtSize(edHifi.cwcGzipBytes)}</td>
                <td class="num">${(edHifi.firstChunkGzipBytes / 1024).toFixed(0)} KiB</td>
                <td class="num">${edHifi.snrLeftDb} / ${edHifi.snrRightDb} dB</td>
                <td class="num">${edHifi.avg24BarkSnrDb} dB</td>
                <td>0–12 kHz Tonal + Base91 JS</td>
              </tr>
              <tr data-select-edition="wec-hifi" class="cwc-row-clickable ${st.audioSource === 'cwc' && activeWecSource === 'hifi' ? 'cwc-row-highlight' : ''}">
                <td><strong>next-intersection-hifi.wec (Binary)</strong></td>
                <td class="num"><strong>${fmtSize(wHifiBytes)}</strong></td>
                <td class="num">0x${edHifi.wec?.crc32Hex || 'd1405459'}</td>
                <td class="num">${edHifi.wec?.unpackMs || 198} ms</td>
                <td class="num">${edHifi.snrLeftDb} / ${edHifi.snrRightDb} dB</td>
                <td class="num">${edHifi.avg24BarkSnrDb} dB</td>
                <td><strong>CWC5 Binary + 16-Ctx 2D Run-Mag (0 Diff)</strong></td>
              </tr>
              <tr data-select-edition="1m" class="cwc-row-clickable ${st.audioSource === 'cwc' && st.activeTier === '1m' ? 'cwc-row-highlight' : ''}">
                <td>CWC-1M (&lt; 1 MB Pure-JS)</td>
                <td class="num">${fmtSize(ed1m.cwcRawBytes)}</td>
                <td class="num">${fmtSize(ed1m.cwcGzipBytes)}</td>
                <td class="num">${(ed1m.firstChunkGzipBytes / 1024).toFixed(0)} KiB</td>
                <td class="num">${ed1m.snrLeftDb} / ${ed1m.snrRightDb} dB</td>
                <td class="num">${ed1m.avg24BarkSnrDb} dB</td>
                <td>0–6.4 kHz Tonal + SBR + Base91 JS</td>
              </tr>
              <tr data-select-edition="wec-1m" class="cwc-row-clickable ${st.audioSource === 'cwc' && activeWecSource === '1m' ? 'cwc-row-highlight' : ''}">
                <td><strong>next-intersection-1m.wec (Binary)</strong></td>
                <td class="num"><strong>${fmtSize(w1mBytes)}</strong></td>
                <td class="num">0x${ed1m.wec?.crc32Hex || '2c7306dd'}</td>
                <td class="num">${ed1m.wec?.unpackMs || 180} ms</td>
                <td class="num">${ed1m.snrLeftDb} / ${ed1m.snrRightDb} dB</td>
                <td class="num">${ed1m.avg24BarkSnrDb} dB</td>
                <td><strong>CWC5 Binary + 48-Ctx Inter-Frame (0 Diff)</strong></td>
              </tr>
              <tr data-select-edition="500k" class="cwc-row-clickable ${st.audioSource === 'cwc' && st.activeTier === '500k' ? 'cwc-row-highlight' : ''}">
                <td>CWC-500K (&lt; 500 KB Pure-JS)</td>
                <td class="num">${fmtSize(ed500k.cwcRawBytes)}</td>
                <td class="num">${fmtSize(ed500k.cwcGzipBytes)}</td>
                <td class="num">${(ed500k.firstChunkGzipBytes / 1024).toFixed(0)} KiB</td>
                <td class="num">${ed500k.snrLeftDb} / ${ed500k.snrRightDb} dB</td>
                <td class="num">${ed500k.avg24BarkSnrDb} dB</td>
                <td>0–4.4 kHz Tonal + SBR + Base91 JS</td>
              </tr>
              <tr data-select-edition="wec-500k" class="cwc-row-clickable ${st.audioSource === 'cwc' && activeWecSource === '500k' ? 'cwc-row-highlight' : ''}">
                <td><strong>next-intersection-500k.wec (Binary)</strong></td>
                <td class="num"><strong>${fmtSize(w500Bytes)}</strong></td>
                <td class="num"><strong>0x${ed500k.wec?.crc32Hex || 'c82126ff'}</strong></td>
                <td class="num"><strong>${ed500k.wec?.unpackMs || 117} ms</strong></td>
                <td class="num"><strong>${ed500k.snrLeftDb} / ${ed500k.snrRightDb} dB</strong></td>
                <td class="num"><strong>${ed500k.avg24BarkSnrDb} dB</strong></td>
                <td><strong>CWC5 Binary + 48-Ctx Inter-Frame (-30.9% vs .js)</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      `;
      return;
    }

    if (st.activeDeckTab === 'lyrics') {
      let activeIdx = -1;
      for (let i = 0; i < CWC_LYRICS.length; i++) {
        if (st.currentTimeSec >= CWC_LYRICS[i].timeSec) activeIdx = i;
      }
      deckBodyEl.innerHTML = `
        <div class="cwc-cue-grid">
          ${CWC_LYRICS.map(
            (line, i) => `
              <button
                type="button"
                class="cwc-cue-item ${i === activeIdx ? 'is-active' : ''}"
                data-seek-sec="${line.timeSec}"
              >
                <span class="cwc-cue-time tabular">${formatClock(line.timeSec)}</span>
                <span>${line.text}</span>
              </button>
            `
          ).join('')}
        </div>
      `;
    }
  };

  deckBodyEl.addEventListener('click', (ev) => {
    const row = ev.target.closest('tr[data-select-edition]');
    if (row) {
      cwcEngine.selectEdition(row.dataset.selectEdition);
      return;
    }
    const cue = ev.target.closest('button[data-seek-sec]');
    if (!cue) return;
    cwcEngine.seek(Number(cue.dataset.seekSec));
  });

  render();
  return { render };
}
