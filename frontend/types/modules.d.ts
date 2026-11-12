declare module "plotly.js-dist-min" {
  import * as Plotly from "plotly.js";
  export default Plotly;
}
declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  import type { PlotParams } from "react-plotly.js";
  export default function createPlotlyComponent(plotly: object): ComponentType<PlotParams>;
}
