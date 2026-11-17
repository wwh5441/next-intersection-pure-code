// V9R1 Dual-Layer Interactive Canvas: 285 Coarticulated Vocal Notes + 86-Bar Motif Atlas Schedule (`src/components/score-motif-canvas.js`)
import { SINGER_CORE } from '../../models/singer-core.js';
import { SONG_STRUCTURE } from '../../data/song-structure.js';
import { V9R1_TIERS } from '../../data/tiers-registry.js';
import { studioStore } from '../state/studio-store.js';
import { audioMixer } from '../modules/audio-mixer.js';

export function mountScoreMotifCanvas(container) {
  container.innerHTML = `
    <div class="section-heading">
      <h2 class="section-title">双字典时间轴总谱 · 上层 285 字连音音高曲线 + 下层 86 小节伴奏原型调度矩阵</h2>
      <span class="section-sub">点击画布任意位置可精准跳转 · 局部视窗 18 秒跟随游标</span>
    </div>
    <div class="canvas-wrap">
      <canvas class="score-motif-canvas" id="v9r1-canvas"></canvas>
    </div>
    <div class="canvas-legend">
      <div class="legend-items">
        <span><i class="legend-dot dot-vocal"></i>主唱音节与F0包络</span>
        <span><i class="legend-dot dot-coart"></i><code>--</code> 协同发音桥</span>
        <span><i class="legend-dot dot-drum"></i>鼓组原型 (M1..M18)</span>
        <span><i class="legend-dot dot-bass"></i>电贝斯原型 (M1..M22)</span>
        <span><i class="legend-dot dot-harm"></i>吉他/电钢原型 (M1..M22)</span>
        <span><i class="legend-dot dot-res"></i>稀疏加花残差 Δ</span>
      </div>
      <span id="v9r1-master-meter-txt">Master RMS: -∞ dBFS</span>
    </div>
  `;

  const canvas = container.querySelector('#v9r1-canvas');
  const meterTxt = container.querySelector('#v9r1-master-meter-txt');
  const ctx = canvas.getContext('2d');
  const frameDur = SINGER_CORE.hopSamples / SINGER_CORE.sampleRate;
  const winSpanSec = 18.0;

  function resizeCanvas() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(200, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    draw(studioStore.getState());
  }

  function getWindowBounds(curSec, totalSec) {
    let tStart = curSec - winSpanSec * 0.28;
    if (tStart < 0) tStart = 0;
    if (tStart + winSpanSec > totalSec) tStart = Math.max(0, totalSec - winSpanSec);
    return { tStart, tEnd: tStart + winSpanSec };
  }

  canvas.addEventListener('click', (ev) => {
    const st = studioStore.getState();
    const rect = canvas.getBoundingClientRect();
    const ratioX = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
    const { tStart, tEnd } = getWindowBounds(st.currentTime, st.durationSec);
    const targetSec = tStart + ratioX * (tEnd - tStart);
    audioMixer.seek(targetSec);
  });

  function draw(st) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#14171d';
    ctx.fillRect(0, 0, W, H);

    const tier = V9R1_TIERS[st.tierId] || Object.values(V9R1_TIERS)[0];
    const score = tier.vocalScore;
    const arrangement = tier.bandArrangement;
    const { tStart, tEnd } = getWindowBounds(st.currentTime, st.durationSec);
    const timeToX = (t) => ((t - tStart) / (tEnd - tStart)) * W;

    const splitY = Math.floor(H * 0.56);

    // Draw bar grid lines across both lanes
    const barDur = SONG_STRUCTURE.barDurationSec;
    const bFirst = Math.max(0, Math.floor(tStart / barDur));
    const bLast = Math.min(SONG_STRUCTURE.totalBars - 1, Math.ceil(tEnd / barDur));

    ctx.font = '10px "SF Mono", Menlo, monospace';
    for (let b = bFirst; b <= bLast; b++) {
      const bx0 = timeToX(b * barDur);
      const bx1 = timeToX((b + 1) * barDur);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx0, 0);
      ctx.lineTo(bx0, H);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.34)';
      ctx.fillText(`B${b + 1}`, bx0 + 4, 12);

      // Lower lane: 3 stacked motif tracks (Drums / Bass / Harmony)
      const tuple = arrangement.bars[b] || [0, 0, 0, 0, 0, 0];
      const laneTop = splitY + 8;
      const rowH = Math.floor((H - laneTop - 8) / 3);
      const bw = Math.max(2, bx1 - bx0 - 2);

      const tracks = [
        { mId: tuple[0], gQ: tuple[1], color: '#d69e2e', prefix: 'D' },
        { mId: tuple[2], gQ: tuple[3], color: '#38a169', prefix: 'B' },
        { mId: tuple[4], gQ: tuple[5], color: '#63b3ed', prefix: 'H' },
      ];

      tracks.forEach((tr, rIdx) => {
        const ry = laneTop + rIdx * rowH;
        if (tr.mId > 0 && tr.gQ > 0) {
          const alpha = Math.max(0.25, Math.min(0.92, (tr.gQ / 64) * 0.72));
          ctx.fillStyle = tr.color;
          ctx.globalAlpha = alpha;
          ctx.fillRect(bx0 + 1, ry + 1, bw, rowH - 3);
          ctx.globalAlpha = 1.0;
          if (bw > 28) {
            ctx.fillStyle = '#0f1116';
            ctx.font = 'bold 10px "SF Mono", Menlo, monospace';
            ctx.fillText(`${tr.prefix}${tr.mId}`, bx0 + 4, ry + rowH - 6);
          }
        } else {
          ctx.fillStyle = 'rgba(255,255,255,0.03)';
          ctx.fillRect(bx0 + 1, ry + 1, bw, rowH - 3);
        }
      });

      if (arrangement.residuals && arrangement.residuals[String(b)]) {
        ctx.fillStyle = '#f687b3';
        ctx.beginPath();
        ctx.arc(Math.min(W - 4, bx1 - 6), splitY + 13, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Divider line
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.beginPath();
    ctx.moveTo(0, splitY);
    ctx.lineTo(W, splitY);
    ctx.stroke();

    // Upper lane: Pitch reference lines (MIDI 52..74)
    const midiMin = 51;
    const midiMax = 74;
    const midiToY = (m) => {
      const norm = (m - midiMin) / (midiMax - midiMin);
      return splitY - 12 - Math.max(0, Math.min(1, norm)) * (splitY - 30);
    };

    for (let m = 54; m <= 72; m += 6) {
      const py = midiToY(m);
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(W, py);
      ctx.stroke();
    }

    // Draw vocal notes & `--` coarticulation bridges
    for (let i = 0; i < score.length; i++) {
      const ev = score[i];
      const evStart = ev[0] * frameDur;
      const evEnd = (ev[0] + ev[1]) * frameDur;
      if (evEnd < tStart - 0.5 || evStart > tEnd + 0.5) continue;

      const x0 = timeToX(evStart);
      const x1 = timeToX(evEnd);
      const baseMidi = ev[8] / 50.0;
      const y0 = midiToY(baseMidi);
      const isCur = st.currentTime >= evStart && st.currentTime < evEnd;

      ctx.fillStyle = isCur ? '#ff7b5c' : '#e85d3f';
      const barW = Math.max(4, x1 - x0 - 1);
      ctx.fillRect(x0, y0 - 6, barW, 12);

      // Draw pitch contour inside note
      const f0Steps = ev[9];
      if (f0Steps && f0Steps.length > 0) {
        ctx.strokeStyle = '#fff5eb';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        let mAcc = baseMidi;
        ctx.moveTo(x0, midiToY(mAcc));
        for (let k = 0; k < f0Steps.length; k++) {
          mAcc += f0Steps[k] / 50.0;
          const kx = x0 + ((k + 1) / f0Steps.length) * barW;
          ctx.lineTo(kx, midiToY(mAcc));
        }
        ctx.stroke();
      }

      // Coarticulation bridge `--`
      if ((ev[5] & 2) !== 0 && i + 1 < score.length) {
        const nextEv = score[i + 1];
        const nx0 = timeToX(nextEv[0] * frameDur);
        const ny0 = midiToY(nextEv[8] / 50.0);
        if (nx0 - x1 < 45) {
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(x1, y0);
          ctx.lineTo(nx0, ny0);
          ctx.stroke();
        }
      }

      if (barW > 12) {
        ctx.fillStyle = '#f8f5ef';
        ctx.font = '11px "PingFang SC", sans-serif';
        ctx.fillText(ev[3], x0 + 2, y0 - 9);
      }
    }

    // Playhead cursor
    const playX = timeToX(st.currentTime);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, H);
    ctx.stroke();

    // Master RMS readout
    const levels = audioMixer.getLiveLevels();
    const db = levels.masterRms > 1e-5 ? (20 * Math.log10(levels.masterRms)).toFixed(1) : '-∞';
    meterTxt.textContent = `Master RMS: ${db} dBFS`;

    ctx.restore();
  }

  window.addEventListener('resize', resizeCanvas);
  studioStore.subscribe(draw);
  requestAnimationFrame(resizeCanvas);
}
