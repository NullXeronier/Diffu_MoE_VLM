/* Chart setup for the project page (English and Korean); labels follow <html lang> */
(function () {
  "use strict";
  var D = window.SITE_DATA, C = window.SiteCharts;
  if (!D || !C) return;
  var ko = (document.documentElement.lang || "en").indexOf("ko") === 0;
  var L = ko ? {
    steps: "환경 스텝", ret: "에피소드 리턴", rate: "성공률", stepsM: "환경 스텝 (M)",
    ppoRet: "PPO 리턴", rnnRet: "PPO-RNN 리턴", policy: "정책", score: "점수 (%)", retCol: "리턴",
    task: "태스크", stepCol: "스텝", scoreUnit: "% 점수", stepUnit: "스텝",
    ach: { collect_wood: "나무 채집", place_table: "작업대 설치", make_wood_pickaxe: "나무 곡괭이", eat_cow: "소 먹기" },
    policies: { "Random": "랜덤", "PPO · CNN + MoE (100k steps)": "PPO · CNN + MoE (10만 스텝)", "Diffusion BC (from PPO demos)": "Diffusion BC (PPO demonstration 학습)" }
  } : {
    steps: "env steps", ret: "episode return", rate: "success rate", stepsM: "Env steps (M)",
    ppoRet: "PPO return", rnnRet: "PPO-RNN return", policy: "Policy", score: "Score (%)", retCol: "Return",
    task: "Task", stepCol: "Steps", scoreUnit: "% score", stepUnit: "steps",
    ach: { collect_wood: "collect wood", place_table: "place table", make_wood_pickaxe: "wood pickaxe", eat_cow: "eat cow" },
    policies: {}
  };
  var fmt1 = function (v) { return v.toFixed(1); };
  var fmtInt = function (v) { return String(Math.round(v)); };
  var fmtM = function (v) { return (Math.round(v * 10) / 10) + "M"; };
  var ppo = D.craftax_cpu.ppo, rnn = D.craftax_cpu.ppo_rnn;

  var ret = document.getElementById("chart-return");
  if (ret) {
    C.lineChart(ret, {
      ariaLabel: L.ret,
      series: [
        { name: "PPO", color: "var(--series-1)", points: ppo.map(function (r) { return { x: r.steps, y: r["return"] }; }) },
        { name: "PPO-RNN", color: "var(--series-2)", points: rnn.map(function (r) { return { x: r.steps, y: r["return"] }; }) }
      ],
      yMax: 10, xFormat: fmtM, xUnit: L.steps, yFormat: fmt1, tickFormat: fmtInt, xLabel: L.steps, yLabel: L.ret
    });
    C.tableView(ret, [{ key: "steps", label: L.stepsM, num: true }, { key: "ppo", label: L.ppoRet, num: true },
      { key: "rnn", label: L.rnnRet, num: true }], ppo.map(function (r) {
      var m = rnn.filter(function (q) { return q.steps === r.steps; })[0];
      return { steps: r.steps, ppo: r["return"], rnn: m ? m["return"] : null };
    }));
  }

  var ach = document.getElementById("chart-ach");
  if (ach) {
    var keys = ["collect_wood", "place_table", "make_wood_pickaxe", "eat_cow"];
    C.lineChart(ach, {
      ariaLabel: L.rate,
      series: keys.map(function (k, i) {
        return { name: L.ach[k], color: "var(--series-" + (i + 1) + ")", points: ppo.map(function (r) { return { x: r.steps, y: r[k] }; }) };
      }),
      yMax: 100, xFormat: fmtM, xUnit: L.steps, yFormat: function (v) { return Math.round(v) + "%"; }, xLabel: L.steps, yLabel: L.rate
    });
    C.tableView(ach, [{ key: "steps", label: L.stepsM, num: true }].concat(keys.map(function (k) {
      return { key: k, label: L.ach[k] + " (%)", num: true };
    })), ppo);
  }

  var cr = document.getElementById("chart-crafter");
  if (cr) {
    var items = D.crafter.map(function (d) { return { label: L.policies[d.policy] || d.policy, value: d.score, ret: d["return"] }; });
    C.barChart(cr, { ariaLabel: L.score, items: items, color: "var(--series-1)", valueFormat: function (v) { return v.toFixed(2); },
      unit: L.scoreUnit, max: 5 });
    C.tableView(cr, [{ key: "label", label: L.policy }, { key: "value", label: L.score, num: true }, { key: "ret", label: L.retCol, num: true }], items);
  }

  var pl = document.getElementById("chart-planner");
  if (pl) {
    C.barChart(pl, { ariaLabel: L.stepCol, items: D.planner_steps.map(function (d) { return { label: d.task, value: d.steps }; }),
      color: "var(--series-1)", unit: L.stepUnit, max: 40 });
    C.tableView(pl, [{ key: "task", label: L.task }, { key: "steps", label: L.stepCol, num: true }], D.planner_steps);
  }
})();
