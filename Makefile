SOUL_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# Support both the current sibling checkout and spesa's older directory layout.
K8S_DIR ?= $(or $(firstword $(wildcard $(SOUL_DIR)/../local-cluster $(SOUL_DIR)/../control-plane/k8s $(SOUL_DIR)/../k8s)),$(SOUL_DIR)/../local-cluster)
-include $(K8S_DIR)/config.env

KIND_CLUSTER ?= k8s
KUBE_CONTEXT ?= kind-$(KIND_CLUSTER)
KIND_NODE ?= $(KIND_CLUSTER)-control-plane
NAMESPACE ?= k8s-local
SOUL_IMAGE ?= soul-controller:kind
INGRESS_ROOT ?= ingress-root
TLS_SECRET ?= echo-app-tls
CREDENTIALS_FILE ?= $(SOUL_DIR)/.soul/kubernetes/credentials.env
export SOUL_DIR K8S_DIR KIND_CLUSTER KUBE_CONTEXT KIND_NODE NAMESPACE SOUL_IMAGE INGRESS_ROOT TLS_SECRET CREDENTIALS_FILE

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := up
# Image loading, credentials, storage and routing must finish before rollout, even with make -j.
.NOTPARALLEL:
.PHONY: up check build load image render data-plane secrets tls apply rollout redeploy down status logs

up: check image apply rollout
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" url

check:
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" check

build:
	docker build --target controller -t "$(SOUL_IMAGE)" "$(SOUL_DIR)"

load: build
	kind load docker-image "$(SOUL_IMAGE)" --name "$(KIND_CLUSTER)"

image: load

render:
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" render

data-plane: check
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" data-plane

secrets: check
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" secrets

tls: check
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" tls

apply: data-plane secrets tls
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" apply

rollout:
	kubectl --context "$(KUBE_CONTEXT)" -n "$(NAMESPACE)" rollout restart deployment/soul-controller
	kubectl --context "$(KUBE_CONTEXT)" -n "$(NAMESPACE)" rollout status deployment/soul-controller --timeout=180s
	kubectl --context "$(KUBE_CONTEXT)" -n "$(NAMESPACE)" wait httpproxy/soul --for=jsonpath='{.status.currentStatus}'=valid --timeout=60s

redeploy: up

down:
	@bash "$(SOUL_DIR)/scripts/kubernetes.sh" down

status:
	kubectl --context "$(KUBE_CONTEXT)" -n "$(NAMESPACE)" get deployment,pod,service,pvc,httpproxy -l app=soul-controller

logs:
	kubectl --context "$(KUBE_CONTEXT)" -n "$(NAMESPACE)" logs deployment/soul-controller --tail=100 -f
