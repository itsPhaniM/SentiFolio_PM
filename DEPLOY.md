# Deploying SentiFolio to the VPS

Deploys the FastAPI API + the built React dashboard behind Caddy, on the server's
IP over HTTP. (Automatic HTTPS needs a domain pointed at the IP — see the last section.)

- **Server:** `<SERVER_IP>`, SSH on a non-default port
- **Bundle:** `sentifolio-deploy.tar.gz` (built locally, ~4.6 MB)

## 1. Copy the bundle to the server (run locally, in PowerShell)

```powershell
scp -P <SSH_PORT> sentifolio-deploy.tar.gz root@<SERVER_IP>:/root/
```
(You'll be prompted for the root password — you type it, over an encrypted SSH channel.)

## 2. SSH in

```powershell
ssh -p <SSH_PORT> root@<SERVER_IP>
```

## 3. On the server — install Docker, unpack, deploy

```bash
# install Docker (auto-detects the distro)
curl -fsSL https://get.docker.com | sh

# unpack the bundle
mkdir -p /opt/sentifolio && tar xzf /root/sentifolio-deploy.tar.gz -C /opt/sentifolio
cd /opt/sentifolio

# build + start (API container + Caddy)
docker compose -f docker-compose.prod.yml up -d --build

# open the firewall for HTTP (harmless if no firewall is active)
ufw allow 80/tcp 2>/dev/null || true
ufw allow 443/tcp 2>/dev/null || true
```

## 4. Verify (on the server)

```bash
docker compose -f docker-compose.prod.yml ps          # both api + caddy "Up"
curl -s localhost/api/health                          # {"status":"ok"}
curl -s "localhost/api/portfolio?arm=price_only" | head -c 120
```

Then open **http://<SERVER_IP>** in a browser — the dashboard should load.

## Updating later

Rebuild the bundle locally, `scp` it over, then on the server:
```bash
tar xzf /root/sentifolio-deploy.tar.gz -C /opt/sentifolio
cd /opt/sentifolio && docker compose -f docker-compose.prod.yml up -d --build
```

## Enabling HTTPS (once DNS is set)

`<your-domain>` currently has **no DNS record**. To get automatic HTTPS:

1. In your domain's DNS, add an **A record** → `<SERVER_IP>` (e.g. `vps` on a domain you own).
2. Wait for it to resolve, then edit `Caddyfile`: replace the `:80 {` line with your
   domain (e.g. `<your-domain> {`) — the commented block at the bottom shows the form.
3. `docker compose -f docker-compose.prod.yml restart caddy` — Caddy fetches a
   Let's Encrypt certificate automatically.

## Security note

Deploy to a server you control and substitute your own values for `<SERVER_IP>`
and `<SSH_PORT>`. Use key-based SSH rather than a root password, and set
`PermitRootLogin prohibit-password` in `/etc/ssh/sshd_config` (then
`systemctl restart ssh`).
