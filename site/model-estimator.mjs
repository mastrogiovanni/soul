// Deliberately coarse: a planning aid, not Soul's placement scheduler.
const GIB = 1024 ** 3;

export function estimateNetwork(machines, memoryGiB, bytesPerParameter) {
  if (
    !Number.isInteger(machines) || machines < 0 || machines > 1000 ||
    !Number.isFinite(memoryGiB) || memoryGiB < 1 || memoryGiB > 1024 ||
    ![2, 4].includes(bytesPerParameter)
  ) return null;

  return { machines, memoryGiB, bytesPerParameter, capacityGiB: machines * memoryGiB };
}

export function estimateModel(parameters, network) {
  if (!network || !Number.isFinite(parameters) || parameters <= 0) return null;

  const requiredGiB = Math.ceil(parameters * network.bytesPerParameter / GIB * 1.25 + 2);
  const machinesNeeded = Math.ceil(requiredGiB / network.memoryGiB);
  const status = network.capacityGiB < requiredGiB
    ? "more-memory"
    : network.capacityGiB < requiredGiB * 1.1 ? "tight" : "likely";

  return {
    requiredGiB,
    machinesNeeded,
    additionalMachines: Math.max(0, machinesNeeded - network.machines),
    status,
  };
}
