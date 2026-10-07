import { estimateNetwork, estimateModel } from "./model-estimator.mjs?v=20261007-models";

const form = document.querySelector("#network-estimate");
const controls = form.querySelector("fieldset");
const machines = document.querySelector("#network-machines");
const memory = document.querySelector("#network-memory");
const precision = document.querySelector("#network-precision");
const search = document.querySelector("#model-search");
const fitOnly = document.querySelector("#model-fit-only");
const rows = [...document.querySelectorAll("[data-model]")];
const format = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });
const labels = { likely: "Likely to fit", tight: "Tight fit", "more-memory": "More memory needed" };

function update() {
  const network = form.checkValidity()
    ? estimateNetwork(machines.valueAsNumber, memory.valueAsNumber, Number(precision.value))
    : null;
  const query = search.value.trim().toLowerCase();
  let likely = 0;
  let tight = 0;
  let visible = 0;

  document.querySelector("#estimate-error").hidden = Boolean(network);
  document.querySelector("#network-capacity").textContent = network ? format.format(network.capacityGiB) : "—";
  document.querySelector("#capacity-description").textContent = network
    ? `${network.machines} machines × ${format.format(network.memoryGiB)} GiB contributed each`
    : "Enter a valid network estimate to compare models.";

  rows.forEach((row) => {
    const estimate = estimateModel(Number(row.dataset.parameters), network);
    const badge = row.querySelector(".fit-status");
    const detail = row.querySelector(".fit-detail");
    row.querySelector(".model-memory").textContent = estimate ? `~${format.format(estimate.requiredGiB)} GiB` : "—";
    badge.dataset.fit = estimate?.status ?? "unknown";
    badge.textContent = estimate ? labels[estimate.status] : "Estimate unavailable";
    if (estimate) {
      if (estimate.status === "likely") likely++;
      if (estimate.status === "tight") tight++;
      detail.textContent = estimate.additionalMachines > 0
        ? `About ${estimate.additionalMachines} more ${format.format(network.memoryGiB)} GiB machine${estimate.additionalMachines === 1 ? "" : "s"} needed`
        : `Memory equivalent: ~${estimate.machinesNeeded} machine${estimate.machinesNeeded === 1 ? "" : "s"}`;
    } else {
      detail.textContent = "Check the network inputs above.";
    }
    const matches = `${row.dataset.model} ${row.dataset.architecture}`.toLowerCase().includes(query);
    row.hidden = !matches || (fitOnly.checked && (!estimate || estimate.status !== "likely"));
    if (!row.hidden) visible++;
  });

  document.querySelector("#fit-summary").textContent = network
    ? `${likely} likely to fit · ${tight} tight fit · ${rows.length - likely - tight} need more memory`
    : "Model fit estimates are unavailable until the inputs are valid.";
  document.querySelector("#model-result-count").textContent = `${visible} of ${rows.length} models shown`;
  document.querySelector("#model-empty").hidden = visible > 0;
  document.querySelector("#model-filter-reset").hidden = visible > 0;
  document.querySelectorAll("[data-network-preset]").forEach((button) => {
    button.setAttribute("aria-pressed", String(Boolean(network) &&
      machines.valueAsNumber === Number(button.dataset.machines) &&
      memory.valueAsNumber === Number(button.dataset.memory)));
  });
}

form.addEventListener("submit", (event) => event.preventDefault());
form.addEventListener("input", update);
search.addEventListener("input", update);
fitOnly.addEventListener("change", update);
document.querySelectorAll("[data-network-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    machines.value = button.dataset.machines;
    memory.value = button.dataset.memory;
    update();
  });
});
document.querySelector("#model-filter-reset").addEventListener("click", () => {
  search.value = "";
  fitOnly.checked = false;
  update();
  search.focus();
});
controls.disabled = false;
search.disabled = false;
fitOnly.disabled = false;
update();
