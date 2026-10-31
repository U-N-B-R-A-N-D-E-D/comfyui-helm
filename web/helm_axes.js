import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CUSTOM = "(type or pick)";

async function fetchAxisNames() {
  try {
    const response = await api.fetchApi("/helm/axes");
    if (!response.ok) return [];
    const payload = await response.json();
    return (payload.axes || []).map((axis) => axis.name).filter(Boolean);
  } catch (error) {
    console.warn("[helm] could not load axes", error);
    return [];
  }
}

async function fetchDialChoices() {
  try {
    const response = await api.fetchApi("/helm/axes");
    if (!response.ok) return null;
    const payload = await response.json();
    const choices = (payload.dial || []).map((entry) => entry.choice).filter(Boolean);
    return choices.length ? choices : null;
  } catch (error) {
    console.warn("[helm] could not load the dial menu", error);
    return null;
  }
}

// Nodes with the library menu: the Dial ("axis" widget) and Axis ("idea" widget).
const dialNodes = new Set();
const MENU_WIDGET = { HelmDial: "axis", HelmAxis: "idea" };

async function refreshDialMenus() {
  const choices = await fetchDialChoices();
  if (!choices) return;
  for (const node of dialNodes) {
    const axis = node.widgets?.find((widget) => widget.name === MENU_WIDGET[node.comfyClass]);
    if (!axis) continue;
    axis.options.values = choices;
    // keep the current value even if it is not in the list (an axis made later in the same run)
    node.setDirtyCanvas?.(true, true);
  }
}

// Node colors are roles, and they match the template groups: purple = WHAT (Axis), orange = HOW MUCH (Shape) and the
// one-node Dial, blue = COMBINE (Mix), grey = WHERE (Apply), slate = Lab (advanced).
const COLORS = {
  axis: { color: "#3a2352", bgcolor: "#5b3a80", accent: "#ead8ff" },
  shape: { color: "#6b3300", bgcolor: "#a85200", accent: "#ffe2c2" },
  mix: { color: "#1c3a5e", bgcolor: "#2f5f94", accent: "#d6e8ff" },
  apply: { color: "#333", bgcolor: "#555", accent: "#e4e4e4" },
  lab: { color: "#2a363b", bgcolor: "#3f5159", accent: "#d8e3e8" },
};
const ROLE = {
  HelmAxis: "axis",
  HelmShape: "shape",
  HelmDial: "shape",
  HelmMakeAxis: "shape",
  HelmMix: "mix",
  HelmApplyNode: "apply",
};
const MIN_WIDTH = { HelmDial: 360, HelmMakeAxis: 480, HelmAxis: 340, HelmShape: 340, HelmMix: 340, HelmApplyNode: 300 };
const MIN_HEIGHT = { HelmMakeAxis: 420 };
const LAB_NODES = new Set(["HelmCollect", "HelmVector", "HelmApply", "HelmProbe", "HelmProbeReport"]);

function roleOf(name) {
  return LAB_NODES.has(name) ? "lab" : ROLE[name] || "shape";
}

function applyHelmStyle(node, name) {
  const palette = COLORS[roleOf(name)];
  node.color = palette.color;
  node.bgcolor = palette.bgcolor;
  const minWidth = MIN_WIDTH[name];
  if (minWidth) {
    const size = node.computeSize?.() || node.size;
    node.setSize?.([Math.max(node.size?.[0] || 0, minWidth), Math.max(node.size?.[1] || 0, size[1], MIN_HEIGHT[name] || 0)]);
  }
}

// "Essentials · Light angle (raking) (Z-Image)" -> "Light angle (raking)"
function plainLabel(choice) {
  const text = String(choice || "");
  const at = text.indexOf(" · ");
  if (at < 0) return text;
  const section = text.slice(0, at);
  const rest = text.slice(at + 3);
  return section === "Custom" ? rest : rest.replace(/ \([^()]*\)$/, "");
}

const MINUS = "−";

function signedPct(value) {
  const pct = Math.round(Math.min(1, Math.max(-1, Number(value) || 0)) * 100);
  if (pct === 0) return "0%";
  return (pct > 0 ? "+" : MINUS) + Math.abs(pct) + "%";
}

function widgetValue(node, name) {
  return node.widgets?.find((w) => w.name === name)?.value;
}

function isLinked(node, name) {
  return node.inputs?.find((i) => i.name === name)?.link != null;
}

// The Dial: "Light angle (raking) at 60%"
function dialCaption(node) {
  const axis = widgetValue(node, "axis");
  const amount = widgetValue(node, "amount");
  if (axis === undefined || amount === undefined) return "";
  const pct = Math.round(Math.min(1, Math.max(0, Number(amount) || 0)) * 100);
  const label = isLinked(node, "axis_from") ? "Your new axis" : plainLabel(axis);
  return `${label} at ${pct}%` + (pct === 0 ? " (Dial off)" : "");
}

// Axis: which idea, or how many prompt pairs it will be built from.
function axisCaption(node) {
  if (widgetValue(node, "source") === "prompts") {
    const pairs = String(widgetValue(node, "pairs") || "").split("\n").filter((l) => l.includes("||")).length;
    const name = String(widgetValue(node, "name") || "").trim();
    const what = pairs === 1 ? "1 prompt pair" : `${pairs} prompt pairs`;
    return `New idea from ${what}` + (name ? `, saved as ${name}` : "");
  }
  const idea = widgetValue(node, "idea");
  const chained = isLinked(node, "helm") ? " (added to the chain)" : "";
  return idea ? `${plainLabel(idea)}${chained}` : "";
}

// Shape: "−40% (less), late steps" or "+50% → 0% (ramp), all steps"
function shapeCaption(node) {
  const amount = Number(widgetValue(node, "amount")) || 0;
  const curve = String(widgetValue(node, "curve") || "constant");
  const stage = String(widgetValue(node, "stage") || "all steps").replace(/ \(.*\)$/, "");
  let how;
  if (curve === "ramp") how = `${signedPct(amount)} → ${signedPct(widgetValue(node, "end_amount"))} (ramp)`;
  else if (curve === "keyframes") how = `${signedPct(amount)} start, keyframes`;
  else if (amount === 0) how = "Off (0%)";
  else how = `${signedPct(amount)} (${amount > 0 ? "more" : "less"})`;
  return `${how}, ${stage}`;
}

// Mix: what the report said after the last run, or how many ideas are connected before the first run.
function mixCaption(node) {
  if (node._helmReport) return node._helmReport;
  const linked = ["helm_1", "helm_2", "helm_3", "helm_4"].filter((n) => isLinked(node, n)).length;
  if (!linked) return "Connect the ideas to mix";
  return `Mixing ${linked} input${linked === 1 ? "" : "s"}` + (widgetValue(node, "orthogonalize") ? ", made independent" : "");
}

const CAPTIONS = { HelmDial: dialCaption, HelmAxis: axisCaption, HelmShape: shapeCaption, HelmMix: mixCaption };

function addCaption(node, name) {
  const textOf = CAPTIONS[name];
  const accent = COLORS[roleOf(name)].accent;
  const widget = node.addCustomWidget({
    name: "helm_caption",
    type: "helm_caption",
    value: "",
    serialize: false,
    options: { serialize: false },
    computeSize: (width) => [width, 34],
    draw(ctx, n, width, y) {
      const text = textOf(n);
      const margin = 15;
      ctx.save();
      ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
      ctx.beginPath();
      ctx.roundRect(margin, y + 4, width - margin * 2, 26, 6);
      ctx.fill();
      ctx.fillStyle = accent;
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, width / 2, y + 17, width - margin * 2 - 12);
      ctx.restore();
    },
  });
  widget.serialize = false;
  widget.serializeValue = () => undefined;
  return widget;
}

// Progressive disclosure: a node shows only the widgets that matter for what it is set to do.
// Axis is the "advanced" node: library mode shows only source + idea; prompts mode swaps the idea menu for the prompt settings.
// Shape shows the split only for early/late, and the ramp or keyframe controls only for those curves.
const AXIS_PROMPT_WIDGETS = ["pairs", "name", "debias", "debias_strength", "steps", "seed", "control_after_generate", "cfg", "block"];

const VISIBILITY = {
  HelmAxis: {
    trigger: ["source"],
    visible(node) {
      const prompts = widgetValue(node, "source") === "prompts";
      return { idea: !prompts, ...Object.fromEntries(AXIS_PROMPT_WIDGETS.map((n) => [n, prompts])) };
    },
  },
  HelmShape: {
    trigger: ["stage", "curve"],
    visible(node) {
      const curve = widgetValue(node, "curve");
      return {
        split: !String(widgetValue(node, "stage") || "").startsWith("all"),
        end_amount: curve === "ramp",
        easing: curve === "ramp",
        keyframes: curve === "keyframes",
      };
    },
  },
};

function updateVisibility(node, name, resize) {
  const rule = VISIBILITY[name];
  const wanted = rule.visible(node);
  for (const w of node.widgets || []) {
    if (!(w.name in wanted)) continue;
    w.hidden = !wanted[w.name];
  }
  if (resize) {
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], MIN_WIDTH[name] || 0), size[1]]);
  }
  node.setDirtyCanvas?.(true, true);
}

function setupVisibility(node, name) {
  const rule = VISIBILITY[name];
  for (const trigger of rule.trigger) {
    const widget = node.widgets?.find((w) => w.name === trigger);
    if (!widget) continue;
    const callback = widget.callback;
    widget.callback = function () {
      const result = callback?.apply(this, arguments);
      updateVisibility(node, name, true);
      return result;
    };
  }
  updateVisibility(node, name, true);
  // A saved workflow restores its own size, so only the visibility is applied after loading.
  const onConfigure = node.onConfigure;
  node.onConfigure = function () {
    const result = onConfigure?.apply(this, arguments);
    updateVisibility(node, name, false);
    return result;
  };
}

function trackMenuNode(node) {
  dialNodes.add(node);
  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    dialNodes.delete(node);
    return onRemoved?.apply(this, arguments);
  };
}

app.registerExtension({
  name: "comfyui-helm.dial-menu",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name?.startsWith("Helm")) return;
    const name = nodeData.name;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      if (MENU_WIDGET[name]) {
        trackMenuNode(this);
        refreshDialMenus();
      }
      if (CAPTIONS[name]) addCaption(this, name);
      if (VISIBILITY[name]) setupVisibility(this, name);
      applyHelmStyle(this, name);
      return result;
    };
    if (name === "HelmMix") {
      // Mix returns its report as a small UI message too, so the node can say it in words.
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        const result = onExecuted?.apply(this, arguments);
        const text = message?.text;
        const line = Array.isArray(text) ? String(text[0] || "").split("\n")[0] : "";
        this._helmReport = line || undefined;
        this.setDirtyCanvas?.(true, true);
        return result;
      };
    }
  },
  async setup() {
    // Make Axis writes a new file during a run; offer it in the Dial menu without a page reload.
    api.addEventListener("execution_success", () => refreshDialMenus());
  },
});

// The old Lab "Apply axis" node keeps its "axis library" picker next to the free-text axis widget.
app.registerExtension({
  name: "comfyui-helm.axis-picker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "HelmApply") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const axis = this.widgets?.find((widget) => widget.name === "axis");
      if (!axis) return result;

      const picker = this.addWidget(
        "combo",
        "axis library",
        CUSTOM,
        (value) => {
          if (value && value !== CUSTOM) {
            axis.value = value;
            this.setDirtyCanvas(true, true);
          }
        },
        { values: [CUSTOM] },
      );
      picker.serialize = false;
      picker.serializeValue = () => undefined;

      fetchAxisNames().then((names) => {
        picker.options.values = [CUSTOM, ...names];
        picker.value = names.includes(axis.value) ? axis.value : CUSTOM;
      });

      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        const configured = onConfigure?.apply(this, arguments);
        if (picker.options.values.includes(axis.value)) picker.value = axis.value;
        return configured;
      };
      return result;
    };
  },
});
