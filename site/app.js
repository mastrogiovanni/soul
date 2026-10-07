"use strict";

// All interactions are local: no analytics, form collection, or credentials.
const statusMessage = document.querySelector("#copy-status");
let statusTimer;

function announce(message) {
  clearTimeout(statusTimer);
  statusMessage.textContent = message;
  statusMessage.classList.add("visible");
  statusTimer = setTimeout(
    () => statusMessage.classList.remove("visible"),
    4500,
  );
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(source.textContent.trim());
      announce("Copied. Paste into your terminal when you’re ready.");
    } catch {
      // Keep the command usable when clipboard permission is unavailable.
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(source);
      selection.removeAllRanges();
      selection.addRange(range);
      announce("Command selected. Use your device’s Copy action to copy it.");
    }
  });
});

const galleryItems = [...document.querySelectorAll("[data-gallery]")];
const galleryDialog = document.querySelector("#gallery-dialog");
const galleryImage = document.querySelector("#gallery-image");
const galleryCaption = document.querySelector("#gallery-caption");
let currentImage = 0;

function showImage(index) {
  currentImage = (index + galleryItems.length) % galleryItems.length;
  const item = galleryItems[currentImage];
  const figure = item.closest("figure");
  galleryImage.src = item.href;
  galleryImage.alt = item.querySelector("img").alt;
  galleryCaption.textContent = `${currentImage + 1} / ${galleryItems.length} — ${figure.querySelector("h3").textContent} ${figure.querySelector("figcaption p").textContent}`;
}

galleryItems.forEach((item, index) => {
  item.addEventListener("click", (event) => {
    // Preserve opening the underlying image in another browser tab.
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
      return;
    if (typeof galleryDialog.showModal !== "function") return;
    event.preventDefault();
    showImage(index);
    galleryDialog.showModal();
    document.body.style.overflow = "hidden";
  });
});
document
  .querySelector("#gallery-close")
  .addEventListener("click", () => galleryDialog.close());
document
  .querySelector("#gallery-previous")
  .addEventListener("click", () => showImage(currentImage - 1));
document
  .querySelector("#gallery-next")
  .addEventListener("click", () => showImage(currentImage + 1));
galleryDialog.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    showImage(currentImage + (event.key === "ArrowRight" ? 1 : -1));
  }
});
galleryDialog.addEventListener("click", (event) => {
  const bounds = galleryDialog.getBoundingClientRect();
  if (
    event.target === galleryDialog &&
    (event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom)
  ) {
    galleryDialog.close();
  }
});
galleryDialog.addEventListener("close", () => {
  document.body.style.overflow = "";
});
