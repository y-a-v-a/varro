// Runs before first paint, so a remembered theme never flashes the other one.
(function () {
  try {
    var saved = localStorage.getItem("varro-theme");
    if (saved === "light" || saved === "dark") {
      document.documentElement.setAttribute("data-theme", saved);
    }
  } catch (e) { /* storage unavailable: follow the system */ }
})();
