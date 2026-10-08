// Static pages provide the language; Italian is the default, independent of browser settings.
export function getMessages(language = "it") {
  const locale = language.toLowerCase().split("-")[0] === "en" ? "en" : "it";
  const format = new Intl.NumberFormat(locale, { maximumFractionDigits: 2 });
  const number = (value) => format.format(value);
  const machine = (count) => locale === "it"
    ? (count === 1 ? "macchina" : "macchine")
    : (count === 1 ? "machine" : "machines");

  return locale === "it" ? {
    locale, format,
    labels: { likely: "Memoria con margine", tight: "Memoria al limite", "more-memory": "Serve più memoria" },
    copied: "Copiato. Incollalo nel terminale quando vuoi.",
    copyFallback: "Testo selezionato. Usa il comando Copia del tuo dispositivo.",
    invalidNetwork: "Inserisci una stima valida della rete per confrontare i modelli.",
    unavailable: "Stima non disponibile",
    checkInputs: "Controlla i dati della rete qui sopra.",
    invalidSummary: "Le stime della memoria non sono disponibili finché i dati inseriti non sono validi.",
    networkDescription: ({ machines, memoryGiB }) => `${number(machines)} ${machine(machines)} × ${number(memoryGiB)} GiB offerti ciascuna`,
    moreMachines: (count, memoryGiB) => `Circa ${number(count)} ${machine(count)} in più da ${number(memoryGiB)} GiB`,
    memoryEquivalent: (count) => `Memoria equivalente: ~${number(count)} ${machine(count)}`,
    fitSummary: (likely, tight, total) => `${number(likely)} con margine · ${number(tight)} al limite · ${number(total - likely - tight)} richiedono più memoria`,
    resultCount: (visible, total) => `${number(visible)} ${visible === 1 ? "modello mostrato" : "modelli mostrati"} su ${number(total)}`,
  } : {
    locale, format,
    labels: { likely: "Likely to fit", tight: "Tight fit", "more-memory": "More memory needed" },
    copied: "Copied. Paste into your terminal when you’re ready.",
    copyFallback: "Text selected. Use your device’s Copy action to copy it.",
    invalidNetwork: "Enter a valid network estimate to compare models.",
    unavailable: "Estimate unavailable",
    checkInputs: "Check the network inputs above.",
    invalidSummary: "Model fit estimates are unavailable until the inputs are valid.",
    networkDescription: ({ machines, memoryGiB }) => `${number(machines)} ${machine(machines)} × ${number(memoryGiB)} GiB contributed each`,
    moreMachines: (count, memoryGiB) => `About ${number(count)} more ${number(memoryGiB)} GiB ${machine(count)} needed`,
    memoryEquivalent: (count) => `Memory equivalent: ~${number(count)} ${machine(count)}`,
    fitSummary: (likely, tight, total) => `${number(likely)} likely to fit · ${number(tight)} tight fit · ${number(total - likely - tight)} need more memory`,
    resultCount: (visible, total) => `${number(visible)} of ${number(total)} models shown`,
  };
}
