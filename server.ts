import compression from 'compression';
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

app.use(compression());

app.use((_req, res, next) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
  next();
});

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', engine: 'CabronWEC-CWC5X' });
});

// Decoupled optional A/B ear-comparison reference stream (supports HTTP 206 Range seeking)
app.get('/ref-audio/master.mp3', (req, res) => {
  res.setHeader('Content-Type', 'audio/mpeg');
  res.sendFile(REF_MP3_PATH, (err) => {
    if (err && !res.headersSent && !req.aborted) {
      res.status(404).json({ error: 'Reference master MP3 not found' });
    }
  });
});

// Legacy /dist/* alias mapped safely to persistent /bundles/*
app.use(
  '/dist',
  express.static(path.join(STATIC_ROOT, 'bundles'), {
    setHeaders: (res, filePath) => {
      if (filePath.endsWith('.js') || filePath.endsWith('.mjs')) {
        res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
      } else if (filePath.endsWith('.wec')) {
        res.setHeader('Content-Type', 'application/octet-stream');
      }
    },
  })
);

app.use(
  express.static(STATIC_ROOT, {
    setHeaders: (res, filePath) => {
      if (filePath.endsWith('.js') || filePath.endsWith('.mjs')) {
        res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
      } else if (filePath.endsWith('.css')) {
        res.setHeader('Content-Type', 'text/css; charset=utf-8');
      } else if (filePath.endsWith('.json')) {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
      } else if (filePath.endsWith('.wec')) {
        res.setHeader('Content-Type', 'application/octet-stream');
      }
    },
  })
);

app.use((req, res) => {
  if (
    req.path.startsWith('/bundles/') ||
    req.path.startsWith('/dist/') ||
    req.path.startsWith('/data/') ||
    req.path.startsWith('/src/') ||
    req.path.startsWith('/styles/') ||
    req.path.startsWith('/api/') ||
    path.extname(req.path) !== ''
  ) {
    res.status(404).json({ error: `Asset not found: ${req.path}` });
    return;
  }
  res.sendFile(path.join(STATIC_ROOT, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log(`Serving CabronWEC (CWC-5X) Studio on http://${HOST}:${PORT}`);
});
