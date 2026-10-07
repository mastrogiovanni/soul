# Kubernetes on the shared spesa cluster

Soul follows spesa's Kind + Kustomize + Contour deployment pattern. The controller container serves the admin panel, APIs, docs and node WebSockets under **`/soul`** on the existing HTTPS domain. With the current `local-cluster/config.env`, that is `https://mastrogiovanni.ddns.net/soul/`. The actual domain is read from `ingress-root`; Soul does not create a second virtual host or replace other applications' routes.

## Deploy

Prerequisites: Docker, Kind, kubectl, Make, Bash and OpenSSL; the shared `kind-k8s` cluster with namespace `k8s-local`, Contour `ingress-root`, a ready TLS certificate, and the host directory mounted at `/data-plane` inside the Kind control-plane node.

From this project:

```sh
make up
```

This validates the target cluster and HTTPS prerequisites, builds the locked controller-only image, loads it into Kind, prepares storage and a Secret, applies the manifests, registers the `/soul` include, and waits for deployment and Contour readiness. It never selects an unrelated current context or creates/recreates the shared cluster. If the cluster/certificate is absent, first run the platform's setup and `make cert` in its own checkout.

`K8S_DIR` finds `../local-cluster` first, then spesa's older `../control-plane/k8s` or `../k8s` layout. Set it explicitly when needed:

```sh
make up K8S_DIR=/path/to/local-cluster
```

Advanced overrides: `KIND_CLUSTER`, `KUBE_CONTEXT`, `KIND_NODE`, `NAMESPACE`, `SOUL_IMAGE`, `INGRESS_ROOT`, `TLS_SECRET`, and `CREDENTIALS_FILE`. The target must retain the same Contour/shared-host architecture and the control-plane `/data-plane` mount. This is a single Soul installation: PV/PVC names and `/data-plane/soul` are fixed.

```sh
make render   # inspect rendered Kubernetes objects without connecting to a cluster
make status
make logs
make redeploy # rebuild, reload and restart after source changes
make down     # remove Soul's route, Deployment and Service only
```

`make down` retains the PVC, PV, host data and credentials so `make up` can reuse them. It never deletes the shared cluster, certificate, ingress root, or other applications. Controller restarts interrupt active requests; nodes reconnect and report their cached models.

## Credentials and existing ingress authentication

On the first deployment, `make up` creates independent random API, admin and node keys, saved in **`.soul/kubernetes/credentials.env`** with mode `0600`. This file is ignored by Git and Docker. Read the `SOUL_ADMIN_KEY` entry to connect the admin panel. The values populate Secret `soul-credentials`, and subsequent deploys reuse the existing Secret without rotating credentials.

To choose initial values, export `SOUL_API_KEY`, `SOUL_ADMIN_KEY` and `SOUL_NODE_TOKEN` before the first deployment, or prepare `CREDENTIALS_FILE` in kubectl's `KEY=value` env-file format. Optional `HF_TOKEN` is included when set during creation. Editing that file does not automatically update an existing Secret; deliberate rotation can be applied with `kubectl create secret generic soul-credentials --from-env-file=... --dry-run=client -o yaml | kubectl apply -f -` using the explicit context/namespace on both commands, then `make rollout`. Update node credentials when rotating their token.

The child route inherits the shared ingress's TLS and Basic Auth if enabled. Soul does not disable that gate. In the browser, log into the site normally, then enter the Soul admin key. The panel sends `X-Soul-Admin-Key`, leaving `Authorization` available for the site's Basic Auth. API and node callers likewise support `X-Soul-API-Key` and `X-Soul-Node-Token`; ordinary Bearer credentials still work where no outer Basic Auth gate is present.

Connect each native Mac or NVIDIA node to the **prefixed** controller URL:

```sh
export SOUL_NODE_TOKEN='<node key from the credentials file>'
# Only when the shared site requires HTTP Basic Auth:
export SOUL_INGRESS_USERNAME='<site username>'
export SOUL_INGRESS_PASSWORD='<site password>'

uv run --locked --extra node soul node \
  --controller https://mastrogiovanni.ddns.net/soul --name my-mac
```

For NVIDIA, substitute the chosen `node-cu126`, `node-cu128` or `node-cu129` extra. These environment variables keep site credentials out of the URL and process arguments. Nodes validate the public TLS certificate normally. The controller image contains no PyTorch and does not run GPU nodes.

For an API call through Basic Auth:

```sh
curl --user "$SOUL_INGRESS_USERNAME:$SOUL_INGRESS_PASSWORD" \
  -H "X-Soul-API-Key: $SOUL_API_KEY" \
  https://mastrogiovanni.ddns.net/soul/v1/models
```

OpenAI SDK clients use `base_url="https://mastrogiovanni.ddns.net/soul/v1"`. When Basic Auth is enabled, set `default_headers` with `Authorization: Basic <base64(username:password)>` and `X-Soul-API-Key: <Soul inference key>`; both gates must pass. Anthropic uses the base `https://mastrogiovanni.ddns.net/soul` and can retain its `x-api-key` header alongside Basic Auth.

## Paths, storage and lifecycle

| Path | Purpose |
| --- | --- |
| `/soul` | Redirect to `/soul/`, preserving HTTPS |
| `/soul/` | Admin panel |
| `/soul/static/...` | Panel assets |
| `/soul/docs` / `/soul/openapi.json` | API documentation/schema |
| `/soul/admin/...` | Authenticated administration |
| `/soul/v1/...` / `/soul/models` | Inference APIs and executable model list |
| `/soul/internal/nodes` | Authenticated WSS connections |
| `/soul/healthz` | Application readiness |
| `/healthz` | Direct pod startup/liveness probe |

The prefix is preserved by Contour, not stripped. [WebSocket upgrades](https://projectcontour.io/docs/1.33/config/websockets/) are enabled, and the route's response/idle timeouts are disabled to allow long preloads and streamed inference; Soul retains its own request, queue, node heartbeat and RPC limits. See the [Contour timeout reference](https://projectcontour.io/docs/1.33/config/api/#projectcontour.io/v1.TimeoutPolicy).

The Deployment has one replica and `Recreate` updates because SQLite, live WebSocket connections and scheduling belong to one controller. The container runs as numeric UID/GID 1000, without privilege escalation, Linux capabilities or a service-account token, and with a read-only root filesystem. `/tmp` is ephemeral; `/data` is persistent. Requests/limits and startup/readiness/liveness probes are defined in the Deployment.

The retained PV maps `/data-plane/soul` on the Kind control-plane node to `/data` in the container. `make up` discovers the node's actual host mount, creates only the `soul` subdirectory, and sets its owner/mode for the container. It stores SQLite and Hugging Face **metadata**, while tensor weights and KV state stay on the external nodes. Back up this host directory together with the credentials. The 1 GiB hostPath capacity is a scheduling declaration, not a filesystem quota. General multi-node Kubernetes/storage deployments require replacing this Kind-specific PV and node selection.

The Docker image defaults to `CONTEXT_ROOT=/soul`. Native development still defaults to `/`; opt into the same prefix with `uv run --locked soul controller --dev --context-root /soul`, or set `CONTEXT_ROOT`. Set `CONTEXT_ROOT=` explicitly for a container that must retain the old root-level URLs.
