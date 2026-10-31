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

const dialNodes = new Set();

async function refreshDialMenus() {
  const choices = await fetchDialChoices();
  if (!choices) return;
  for (const node of dialNodes) {
    const axis = node.widgets?.find((widget) => widget.name === "axis");
    if (!axis) continue;
    axis.options.values = choices;
    // keep the current value even if it is not in the list (an axis made later in the same run)
    node.setDirtyCanvas?.(true, true);
  }
}

// Node colors match the template groups: orange = the Helm control you move, slate = Lab (advanced).
const COLORS = {
  main: { color: "#6b3300", bgcolor: "#a85200" },
  lab: { color: "#2a363b", bgcolor: "#3f5159" },
};
const MIN_WIDTH = { HelmDial: 360, HelmMakeAxis: 480 };
const MIN_HEIGHT = { HelmMakeAxis: 420 };
const LAB_NODES = new Set(["HelmCollect", "HelmVector", "HelmApply", "HelmProbe", "HelmProbeReport"]);

function applyHelmStyle(node, name) {
  const palette = COLORS[LAB_NODES.has(name) ? "lab" : "main"];
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

function captionText(node) {
  const axis = node.widgets?.find((w) => w.name === "axis");
  const amount = node.widgets?.find((w) => w.name === "amount");
  if (!axis || !amount) return "";
  const pct = Math.round(Math.min(1, Math.max(0, Number(amount.value) || 0)) * 100);
  const linked = node.inputs?.find((i) => i.name === "axis_from")?.link != null;
  const label = linked ? "Your new axis" : plainLabel(axis.value);
  return `${label} at ${pct}%` + (pct === 0 ? " (Dial off)" : "");
}

function addCaption(node) {
  const widget = node.addCustomWidget({
    name: "helm_caption",
    type: "helm_caption",
    value: "",
    serialize: false,
    options: { serialize: false },
    computeSize: (width) => [width, 34],
    draw(ctx, n, width, y) {
      const text = captionText(n);
      const margin = 15;
      ctx.save();
      ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
      ctx.beginPath();
      ctx.roundRect(margin, y + 4, width - margin * 2, 26, 6);
      ctx.fill();
      ctx.fillStyle = "#ffe2c2";
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

app.registerExtension({
  name: "comfyui-helm.dial-menu",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name?.startsWith("Helm")) return;
    const name = nodeData.name;
    if (name !== "HelmDial") {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = created?.apply(this, arguments);
        applyHelmStyle(this, name);
        return result;
      };
      return;
    }
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      dialNodes.add(this);
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        dialNodes.delete(this);
        return onRemoved?.apply(this, arguments);
      };
      addCaption(this);
      applyHelmStyle(this, name);
      refreshDialMenus();
      return result;
    };
  },
  async setup() {
    // Make Axis writes a new file during a run; offer it in the Dial menu without a page reload.
    api.addEventListener("execution_success", () => refreshDialMenus());
  },
});

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
