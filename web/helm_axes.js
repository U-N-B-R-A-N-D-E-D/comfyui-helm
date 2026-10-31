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
