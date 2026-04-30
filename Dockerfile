# --- Stage 1: Build Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app
# Copy the locales folder first (needed for i18n)
COPY locales/ ./locales/
# Copy frontend and build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Final Production Image ---
FROM python:3.11-slim

# Install Node.js 20 runtime and UV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY package.json ./
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Install backend deps from lockfile (torch comes from PyPI via camel-ai / camel-oasis)
RUN npm install --omit=dev \
    && cd backend && uv sync --frozen

# Copy only the necessary source code
COPY backend/ ./backend/
COPY start-prod.sh proxy.js locales/ ./locales/ ./

# Copy the ALREADY BUILT frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

RUN chmod +x start-prod.sh

# Launch SwarmIntel
CMD ["./start-prod.sh"]