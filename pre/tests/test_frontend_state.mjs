import fs from "node:fs";
import vm from "node:vm";

const html = fs.readFileSync("pre/static/index.html", "utf8");
const match = html.match(/<script>([\s\S]*?)<\/script>/);
if (!match) throw new Error("index.html script not found");

const script = match[1].replace(
  /loadSummary\(\);\s*loadSamples\(\);\s*checkHealth\(\);\s*setInterval\(checkHealth,30000\);/,
  "",
);
const elements = {
  baseLive: { innerHTML: "" },
  loraLive: { innerHTML: "" },
};
const context = {
  console,
  URL,
  window: {
    alert: () => {},
    confirm: () => true,
  },
  document: {
    getElementById: (id) =>
      elements[id] || {
        innerHTML: "",
        value: "",
        textContent: "",
        className: "",
      },
  },
  fetch: () => {
    throw new Error("unexpected fetch");
  },
  setInterval: () => {},
  setTimeout: () => {},
};
vm.createContext(context);

const test = `
const result = {
  success: true,
  model: "qwen35-base",
  table: [{
    "步骤层级": "执行步骤",
    "设备类型": "测试表",
    "设备指令号": "hallucinated_voltage"
  }],
  usage: {},
  parse_success: true,
  device_retrieval: {
    enabled: true,
    ready: true,
    rows: [{
      row_index: 0,
      decision: "review_unknown_identifier",
      candidates: [{
        "设备指令主键": "CMD-001",
        "设备类型": "数字万用表",
        "设备指令号": "measure_dc_voltage"
      }]
    }]
  }
};
setLiveResult("base", result);
applyRetrievalCandidate("base", 0, 0);
if (liveState.base.finalRows[0]["设备指令号"] !== "measure_dc_voltage") {
  throw new Error("candidate was not applied");
}
if (result.table[0]["设备指令号"] !== "hallucinated_voltage") {
  throw new Error("original model result was mutated");
}
if (liveModificationList("base").length !== 1) {
  throw new Error("manual modification audit was not recorded");
}
undoRetrievalChange("base", 0);
if (liveState.base.finalRows[0]["设备指令号"] !== "hallucinated_voltage") {
  throw new Error("manual modification was not undone");
}
if (liveModificationList("base").length !== 0) {
  throw new Error("audit was not cleared after undo");
}
`;

vm.runInContext(`${script}\n${test}`, context);
console.log("frontend apply/undo state test OK");
