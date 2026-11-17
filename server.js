import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const STATIC_ROOT = path.join(__dirname, 'cabron-wec');
const REF_MP3_PATH = path.join(__dirname, 'public/assets/aistudio/____-___________80eeb7cc.mp3');
const PORT = Number(process.env.PORT) || 3000;
const HOST = '0.0.0.0';

const app = express();

app.use((_req, res, next) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
  next();
});

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', engine: 'CabronWEC-CWC4X' });
});

// Strictly decoupled optional A/B ear-comparison reference stream (never touched by CWC Worklet)
app.get('/ref-audio/master.mp3', (_req, res) => {
  res.setHeader('Content-Type', 'audio/mpeg');
  res.sendFile(REF_MP3_PATH);
});

app.use(express.static(STATIC_ROOT, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.js') || filePath.endsWith('.mjs')) {
      res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
    } else if (filePath.endsWith('.css')) {
      res.setHeader('Content-Type', 'text/css; charset=utf-8');
    } else if (filePath.endsWith('.json')) {
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
    }
  },
}));

app.get('*', (_req, res) => {
  res.sendFile(path.join(STATIC_ROOT, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log(`Serving CabronWEC (CWC-4X) Studio on http://${HOST}:${PORT}`);
});
