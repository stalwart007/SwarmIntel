import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = 'http://localhost:5001';

// Proxy /api requests to the Flask backend
app.use('/api', createProxyMiddleware({
  target: API_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api': '/api', // keep /api prefix
  },
}));

// Serve static files from the Vue build
app.use(express.static(path.join(__dirname, 'frontend/dist')));

// Handle SPA routing: all other requests go to index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend/dist/index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 SwarmIntel Proxy running on port ${PORT}`);
  console.log(`🔗 Proxying /api to ${API_URL}`);
});
