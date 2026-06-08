# Droplet bootstrap — one-time setup

Run these once after provisioning a fresh control droplet. The
GitHub Actions deploy job (`.github/workflows/deploy.yml`) takes over
from there on every push to `main`.

**Droplet:** Ubuntu 24.04, `s-2vcpu-4gb`, IP `143.198.223.46`.

---

## 1. SSH in as root

From your Mac:

```bash
ssh root@143.198.223.46
```

(Your `gravity-enterprise` key is already trusted — DO injected it at
provision time.)

## 2. Install Docker + git

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker
```

Verify:

```bash
docker --version && docker compose version
```

## 3. Create persistent data dirs

```bash
mkdir -p /var/lib/dlp/{db,hf-cache,caddy-data,caddy-config}
```

These survive container replacement; the prod compose file bind-mounts
them into `service` (`/srv`), `worker` (`/hf-cache`), and `caddy`
(`/data`, `/config`).

## 4. Clone the repo into `/opt/dlp`

```bash
git clone https://github.com/payner35/trackform.git /opt/dlp
cd /opt/dlp
```

## 5. Create `.env` (NOT committed)

```bash
cp deploy/.env.example .env
# Generate a strong token:
sed -i "s|replace-with-openssl-rand-hex-32-output|$(openssl rand -hex 32)|" .env
# Paste your HF token (same one as ~/Documents/Dev/dj-loop-service/.env on your Mac)
nano .env
```

Print the generated token so you can copy it into the Player's settings:

```bash
grep DLP_API_TOKEN .env
```

## 6. Pull images + start

```bash
# Log in to GHCR with a personal access token (classic, scope: read:packages).
# Only needed if the GHCR packages are private — by default they're public
# once first pushed, in which case skip this step.
# echo $GHCR_PAT | docker login ghcr.io -u payner35 --password-stdin

docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

## 7. Add a deploy SSH key for GitHub Actions

On your Mac:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/trackform_deploy -C "github-actions@trackform" -N ""
cat ~/.ssh/trackform_deploy.pub
```

On the droplet:

```bash
echo "<paste public key>" >> /root/.ssh/authorized_keys
```

In GitHub repo Settings → Secrets and variables → Actions, create:

| Secret | Value |
|---|---|
| `DROPLET_HOST` | `143.198.223.46` |
| `DROPLET_USER` | `root` |
| `DROPLET_SSH_KEY` | Contents of `~/.ssh/trackform_deploy` (the **private** key) |

## 8. DO Cloud Firewall (recommended)

In the DO web UI: Networking → Firewalls → Create:

| Direction | Protocol | Port | Sources |
|---|---|---|---|
| Inbound | TCP | 22 | Your IP only |
| Inbound | TCP | 443 | All IPv4 + All IPv6 |
| Inbound | TCP | 80 | All IPv4 + All IPv6 |
| Outbound | All | All | All |

Apply to the `trackform-control` droplet.

## 9. Smoke test from your Mac

```bash
# Health (unauthenticated — exempt)
curl -k https://143.198.223.46/v1/health

# Authenticated endpoint
TOKEN=<paste from step 5>
curl -k -H "Authorization: Bearer $TOKEN" https://143.198.223.46/v1/tracks
```

`-k` skips cert verification (self-signed). Replace with a real domain
in Phase 4b to drop `-k` and use a Let's Encrypt cert.

---

## What happens on every `git push`

1. Actions builds `service` + `worker` images in parallel
2. Pushes to `ghcr.io/payner35/trackform-{service,worker}:latest` (+ short SHA)
3. SSHes to the droplet, `git pull`, `docker compose pull`, `docker compose up -d`
4. Restart is rolling per container (compose replaces one at a time)

Manual deploy (skip a push):

```bash
ssh root@143.198.223.46 'cd /opt/dlp && git pull && docker compose -f docker-compose.prod.yml pull && docker compose -f docker-compose.prod.yml up -d'
```
