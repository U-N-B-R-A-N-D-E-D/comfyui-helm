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

app.registerExtension({
  name: "comfyui-helm.dial-menu",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "HelmDial") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      dialNodes.add(this);
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        dialNodes.delete(this);
        return onRemoved?.apply(this, arguments);
      };
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
