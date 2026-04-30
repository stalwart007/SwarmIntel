#!/bin/bash

# MiroFish Oracle Cloud Deployment Script
echo "🚀 Starting MiroFish Deployment Setup..."

# 1. Update and install Docker
echo "📦 Installing Docker and Docker Compose..."
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER

# 2. Open Firewall ports
echo "🔥 Opening Firewall ports (80, 443)..."
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

echo "✅ Environment setup complete!"
echo "👉 Next steps:"
echo "1. Log out and log back in (to activate docker group permissions)"
echo "2. Create your .env file: 'nano .env'"
echo "3. Start the app: 'docker compose -f docker-compose.prod.yml up -d --build'"
echo "4. Check logs: 'docker compose -f docker-compose.prod.yml logs -f'"
