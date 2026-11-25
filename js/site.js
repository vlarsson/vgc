// Theme toggle and BibTeX copy.
(function () {
  const root = document.documentElement;
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try { localStorage.setItem("vgc-theme", next); } catch (e) {}
    window.dispatchEvent(new Event("themechange"));
  });

  const copyBtn = document.getElementById("copy-bib");
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById("bib-text").textContent);
      copyBtn.textContent = "Copied";
    } catch (e) {
      copyBtn.textContent = "Select & copy";
    }
    setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
  });
})();
