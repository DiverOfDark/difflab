import { createApp } from "vue";
import App from "./App.vue";

// The web view's own menu offers Reload, which leaves this window black on WebKitGTK with nothing to
// bring it back; the panel has no use for the rest of it either.
document.addEventListener("contextmenu", (event) => event.preventDefault());
document.addEventListener("keydown", (event) => {
  if (event.key === "F5" || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "r")) event.preventDefault();
});

createApp(App).mount("#app");
