import { app } from "../../scripts/app.js";

// "What comes next" suggestions.
//
// Drag a link out of a Helm slot and release it on empty canvas. ComfyUI then shows either
//   - the node search, already filtered by the slot type (the default on new installs; every node that takes a
//     HELM is listed first, so the Helm nodes are what you see), or
//   - the link-release menu (Shift + release by default, or the plain release if you chose it in Settings,
//     LiteGraph > Link release). This file makes that menu short and specific: it lists the Helm nodes that make
//     sense after the node you dragged from, in the order of the chain, with their plain names.
//
// It works through LiteGraph's own `slot_types_default_out` / `slot_types_default_in` tables, set only for the
// moment the menu is built and only when the link comes from (or goes to) a Helm node. For every other node the
// tables are untouched, so other packs and the stock suggestions behave as before.

// Where the link comes from (an output) -> the nodes to suggest, by slot type.
const AFTER = {
  HelmAxis: { HELM: ["HelmShape", "HelmMix", "HelmApplyNode", "HelmAxis"] },
  HelmShape: { HELM: ["HelmApplyNode", "HelmMix"] },
  HelmMix: { HELM: ["HelmApplyNode"] },
  HelmApplyNode: { MODEL: ["KSampler", "BasicGuider", "CFGGuider", "KSamplerAdvanced"] },
  HelmDial: { MODEL: ["KSampler", "BasicGuider", "CFGGuider", "KSamplerAdvanced"] },
};

// Where the link goes to (an input) -> the nodes to suggest, by slot type: a HELM input wants an Axis.
const BEFORE = { HELM: ["HelmAxis"] };

function slotOf(node, slot, outputs) {
  const list = outputs ? node?.outputs : node?.inputs;
  if (slot && typeof slot === "object") return slot;
  if (typeof slot === "number") return list?.[slot];
  return list?.find((s) => s.name === slot);
}

function suggestionFor(options) {
  try {
    const registered = window.LiteGraph?.registered_node_types || {};
    const keep = (names) => names.filter((n) => registered[n]);
    if (options?.nodeFrom && options.slotFrom != null) {
      const slot = slotOf(options.nodeFrom, options.slotFrom, true);
      const names = AFTER[options.nodeFrom.type]?.[slot?.type];
      return names ? { table: "slot_types_default_out", type: slot.type, names: keep(names) } : null;
    }
    if (options?.nodeTo && options.slotTo != null) {
      const slot = slotOf(options.nodeTo, options.slotTo, false);
      const isHelm = String(options.nodeTo.type || "").startsWith("Helm");
      const names = isHelm ? BEFORE[slot?.type] : null;
      return names ? { table: "slot_types_default_in", type: slot.type, names: keep(names) } : null;
    }
  } catch (error) {
    console.warn("[helm] could not pick suggestions", error);
  }
  return null;
}

// The menu lists class names ("HelmApplyNode"); show the node's own title instead ("Helm · Apply").
function relabel(menu, names) {
  const root = menu?.root;
  if (!root?.querySelectorAll) return;
  const registered = window.LiteGraph?.registered_node_types || {};
  const display = (cls) => registered[cls]?.title || cls;
  for (const entry of root.querySelectorAll(".litemenu-entry")) {
    const value = entry.dataset?.value;
    if (names.includes(value)) entry.textContent = display(value);
  }
}

// The tables are read twice: when the menu is built and again when an entry is clicked (LiteGraph's
// createDefaultNodeForSlot checks that the chosen node is in the table). So the override lives until the menu closes.
let active = null;

function restoreTables() {
  if (!active) return;
  const { table, type, previous, observer } = active;
  observer?.disconnect();
  const tables = window.LiteGraph?.[table];
  if (tables) {
    if (previous === undefined) delete tables[type];
    else tables[type] = previous;
  }
  active = null;
}

function patchConnectionMenu() {
  const canvas = app.canvas;
  const proto = canvas && Object.getPrototypeOf(canvas);
  if (!proto || typeof proto.showConnectionMenu !== "function" || proto.showConnectionMenu.__helmFlow) return;
  const original = proto.showConnectionMenu;
  const patched = function (options) {
    restoreTables();
    const pick = suggestionFor(options);
    const LG = window.LiteGraph;
    if (!pick || !pick.names.length || !LG?.[pick.table]) return original.apply(this, arguments);
    active = { table: pick.table, type: pick.type, previous: LG[pick.table][pick.type], observer: null };
    LG[pick.table][pick.type] = pick.names.slice();
    let menu;
    try {
      menu = original.apply(this, arguments);
    } catch (error) {
      restoreTables();
      throw error;
    }
    const root = menu?.root;
    if (!root?.parentNode) {
      restoreTables();
      return menu;
    }
    relabel(menu, pick.names);
    const mine = active;
    const observer = new MutationObserver(() => {
      if (!root.isConnected && active === mine) restoreTables();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    mine.observer = observer;
    return menu;
  };
  patched.__helmFlow = true;
  proto.showConnectionMenu = patched;
}

app.registerExtension({
  name: "comfyui-helm.flow",
  async setup() {
    patchConnectionMenu();
  },
});
