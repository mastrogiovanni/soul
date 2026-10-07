import test from "node:test";
import assert from "node:assert/strict";
import { estimateNetwork, estimateModel } from "../../site/model-estimator.mjs";

test("pooled capacity and conservative CPU weight sizing", () => {
  const gpu = estimateNetwork(4, 8, 2);
  const cpu = estimateNetwork(4, 8, 4);
  assert.equal(gpu.capacityGiB, 32);
  // Qwen3-8B: ~15.26 GiB fp16 weights + 25% + 2 GiB, rounded up.
  assert.deepEqual(estimateModel(8190735360, gpu), {
    requiredGiB: 22, machinesNeeded: 3, additionalMachines: 0, status: "likely",
  });
  assert.deepEqual(estimateModel(8190735360, cpu), {
    requiredGiB: 41, machinesNeeded: 6, additionalMachines: 2, status: "more-memory",
  });
});

test("zero contributors cannot fit even the smallest model", () => {
  const estimate = estimateModel(494032768, estimateNetwork(0, 8, 2));
  assert.equal(estimate.status, "more-memory");
  assert.equal(estimate.additionalMachines, 1);
});

test("exact capacity is tight; 10% extra gives the likely label", () => {
  // One GiB of fp16 weights requires ceil(1 * 1.25 + 2) = 4 GiB.
  const parameters = 1024 ** 3 / 2;
  assert.equal(estimateModel(parameters, estimateNetwork(1, 3.75, 2)).status, "more-memory");
  assert.equal(estimateModel(parameters, estimateNetwork(1, 4, 2)).status, "tight");
  assert.equal(estimateModel(parameters, estimateNetwork(1, 4.25, 2)).status, "tight");
  assert.equal(estimateModel(parameters, estimateNetwork(1, 4.5, 2)).status, "likely");
});

test("invalid and out-of-range inputs never produce plausible estimates", () => {
  for (const machines of [NaN, Infinity, -1, 1.5, 1001]) {
    assert.equal(estimateNetwork(machines, 8, 2), null);
  }
  for (const memory of [NaN, Infinity, 0, -8, 1025]) {
    assert.equal(estimateNetwork(4, memory, 2), null);
  }
  assert.equal(estimateNetwork(4, 8, 1), null);
  assert.equal(estimateModel(8190735360, null), null);
  assert.equal(estimateModel(NaN, estimateNetwork(4, 8, 2)), null);
});

test("a larger network fits larger models without assuming lower latency", () => {
  const parameters = 72706203648;
  const small = estimateModel(parameters, estimateNetwork(4, 8, 4));
  const large = estimateModel(parameters, estimateNetwork(32, 16, 4));
  assert.equal(small.requiredGiB, 341);
  assert.equal(small.additionalMachines, 39);
  assert.equal(large.requiredGiB, small.requiredGiB);
  assert.equal(large.status, "likely");
});
