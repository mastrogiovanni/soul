#!/usr/bin/env bash
set -euo pipefail

SOUL_DIR=${SOUL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
K8S_DIR=${K8S_DIR:-$SOUL_DIR/../local-cluster}
KIND_CLUSTER=${KIND_CLUSTER:-k8s}
KUBE_CONTEXT=${KUBE_CONTEXT:-kind-$KIND_CLUSTER}
KIND_NODE=${KIND_NODE:-$KIND_CLUSTER-control-plane}
NAMESPACE=${NAMESPACE:-k8s-local}
SOUL_IMAGE=${SOUL_IMAGE:-soul-controller:kind}
INGRESS_ROOT=${INGRESS_ROOT:-ingress-root}
TLS_SECRET=${TLS_SECRET:-echo-app-tls}
CREDENTIALS_FILE=${CREDENTIALS_FILE:-$SOUL_DIR/.soul/kubernetes/credentials.env}
INCLUDE_SCRIPT=$K8S_DIR/scripts/httpproxy-include.sh

fail() { printf '%s\n' "$*" >&2; exit 1; }
k() { kubectl --context "$KUBE_CONTEXT" "$@"; }
root_field() { k get httpproxy "$INGRESS_ROOT" -n "$NAMESPACE" -o "jsonpath={$1}"; }
data_mount() {
    docker inspect "$KIND_NODE" --format '{{range .Mounts}}{{if eq .Destination "/data-plane"}}{{.Source}}{{end}}{{end}}'
}

render() (
    [[ "$NAMESPACE" =~ ^[a-z0-9][a-z0-9-]*$ ]] || fail "Invalid namespace"
    [[ "$SOUL_IMAGE" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/@-]*$ ]] || fail "Invalid image reference"
    local temporary
    temporary=$(mktemp -d)
    trap 'rm -rf "$temporary"' EXIT
    cp "$SOUL_DIR"/manifests/*.yaml "$temporary/"
    sed "s/^namespace: k8s-local$/namespace: $NAMESPACE/" "$SOUL_DIR/manifests/kustomization.yaml" > "$temporary/kustomization.yaml"
    cat >> "$temporary/kustomization.yaml" <<EOF
patches:
  - target:
      kind: Deployment
      name: soul-controller
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/image
        value: $SOUL_IMAGE
EOF
    kubectl kustomize "$temporary"
)

case "${1:-}" in
    check)
        for tool in docker kind kubectl openssl; do
            command -v "$tool" >/dev/null || fail "Install $tool before running make up."
        done
        kubectl config get-contexts -o name | grep -Fxq "$KUBE_CONTEXT" ||
            fail "Context $KUBE_CONTEXT is missing. Set up the shared cluster in $K8S_DIR first; no other context will be used."
        test -f "$INCLUDE_SCRIPT" || fail "Set K8S_DIR to the shared cluster checkout (missing $INCLUDE_SCRIPT)."
        k get ns "$NAMESPACE" >/dev/null
        k get httpproxy "$INGRESS_ROOT" -n "$NAMESPACE" >/dev/null
        secret=$(root_field '.spec.virtualhost.tls.secretName')
        secret=${secret:-$TLS_SECRET}
        k get secret "$secret" -n "$NAMESPACE" >/dev/null 2>&1 ||
            fail "HTTPS certificate $secret is missing. Run make cert in $K8S_DIR before make up here."
        test -n "$(data_mount)" ||
            fail "Kind node $KIND_NODE has no /data-plane host mount. Configure the shared cluster's extraMounts first."
        ;;
    render) render ;;
    data-plane)
        mount=$(data_mount)
        test -n "$mount" || fail "Missing /data-plane host mount on $KIND_NODE"
        # Prepare only Soul's directory, using the actual Kind host mount rather than a guessed path.
        docker exec "$KIND_NODE" sh -eu -c 'mkdir -p /data-plane/soul; chown 1000:1000 /data-plane/soul; chmod 0770 /data-plane/soul'
        printf 'Soul data: %s/soul\n' "$mount"
        ;;
    secrets)
        if k get secret soul-credentials -n "$NAMESPACE" >/dev/null 2>&1; then
            printf 'Reusing Secret soul-credentials (keys are unchanged).\n'
            exit 0
        fi
        if ! test -f "$CREDENTIALS_FILE"; then
            umask 077
            mkdir -p "$(dirname "$CREDENTIALS_FILE")"
            temporary=$(mktemp "$CREDENTIALS_FILE.XXXXXX")
            trap 'rm -f "$temporary"' EXIT
            for name in SOUL_API_KEY SOUL_ADMIN_KEY SOUL_NODE_TOKEN; do
                value=${!name:-}
                value=${value:-$(openssl rand -hex 32)}
                [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || fail "$name must be one line"
                printf '%s=%s\n' "$name" "$value" >> "$temporary"
            done
            if test -n "${HF_TOKEN:-}"; then
                [[ "$HF_TOKEN" != *$'\n'* && "$HF_TOKEN" != *$'\r'* ]] || fail "HF_TOKEN must be one line"
                printf 'HF_TOKEN=%s\n' "$HF_TOKEN" >> "$temporary"
            fi
            mv "$temporary" "$CREDENTIALS_FILE"
            printf 'Initial credentials saved to %s (mode 0600).\n' "$CREDENTIALS_FILE"
        fi
        chmod 0600 "$CREDENTIALS_FILE"
        k create secret generic soul-credentials -n "$NAMESPACE" --from-env-file="$CREDENTIALS_FILE" --dry-run=client -o yaml |
            k apply -f -
        ;;
    tls)
        if test -z "$(root_field '.spec.virtualhost.tls.secretName')"; then
            k patch httpproxy "$INGRESS_ROOT" -n "$NAMESPACE" --type merge \
                -p "{\"spec\":{\"virtualhost\":{\"tls\":{\"secretName\":\"$TLS_SECRET\"}}}}"
        fi
        ;;
    apply)
        render | k apply -f -
        bash "$INCLUDE_SCRIPT" add "$KUBE_CONTEXT" "$NAMESPACE" "$INGRESS_ROOT" soul "$NAMESPACE" /soul
        ;;
    down)
        if k get httpproxy "$INGRESS_ROOT" -n "$NAMESPACE" >/dev/null 2>&1; then
            bash "$INCLUDE_SCRIPT" remove "$KUBE_CONTEXT" "$NAMESPACE" "$INGRESS_ROOT" soul "$NAMESPACE" /soul
        fi
        k delete deployment/soul-controller service/soul-service httpproxy/soul -n "$NAMESPACE" --ignore-not-found
        printf 'Soul removed. PVC, PV, credentials, host data and the shared cluster are retained.\n'
        ;;
    url)
        printf '\nSoul admin: https://%s/soul/\n' "$(root_field '.spec.virtualhost.fqdn')"
        printf 'API base: /soul/v1; node controller URL: the same HTTPS /soul base.\n'
        ;;
    *) fail "Usage: $0 check|render|data-plane|secrets|tls|apply|down|url" ;;
esac
