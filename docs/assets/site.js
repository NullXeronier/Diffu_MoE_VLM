/* Diffu-MoE-VLM site: theme toggle, copy buttons and small SVG charts (no dependencies) */
(function () {
  "use strict";
  var SVG = "http://www.w3.org/2000/svg";

  // ---------- theme ----------
  function storedTheme() {
    try { return localStorage.getItem("theme"); } catch (e) { return null; }
  }
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(storedTheme());
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest && ev.target.closest(".theme-toggle");
    if (!btn) return;
    var current = document.documentElement.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
  });

  // ---------- copy buttons ----------
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest && ev.target.closest(".copy-btn");
    if (!btn) return;
    var target = document.getElementById(btn.getAttribute("data-target"));
    if (!target || !navigator.clipboard) return;
    navigator.clipboard.writeText(target.textContent).then(function () {
      btn.textContent = "Copied";
      setTimeout(function () { btn.textContent = "Copy"; }, 1500);
    });
  });

  // ---------- helpers ----------
  function el(name, attrs, parent) {
    var node = document.createElementNS(SVG, name);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, str, cls, anchor) {
    var t = el("text", { x: x, y: y, "class": cls, "text-anchor": anchor || "start" }, parent);
    t.textContent = str;
    return t;
  }
  function niceMax(v) {
    var exp = Math.pow(10, Math.floor(Math.log10(v)));
    var steps = [1, 2, 2.5, 5, 7.5, 10];
    for (var i = 0; i < steps.length; i++) if (steps[i] * exp >= v) return steps[i] * exp;
    return 10 * exp;
  }
  function makeTooltip(container) {
    var tip = document.createElement("div");
    tip.className = "tooltip";
    tip.setAttribute("role", "status");
    container.appendChild(tip);
    return tip;
  }
  function placeTooltip(tip, container, px, py) {
    var cw = container.clientWidth;
    var tw = tip.offsetWidth;
    var left = px + 14;
    if (left + tw > cw) left = px - tw - 14;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = Math.max(0, py - 10) + "px";
    tip.style.opacity = 1;
  }
  function legend(container, items, shape) {
    var lg = document.createElement("div");
    lg.className = "legend";
    items.forEach(function (it) {
      var key = document.createElement("span");
      key.className = "key";
      var sw = document.createElement("span");
      sw.className = shape === "rect" ? "swatch-rect" : "swatch-line";
      sw.style.background = it.color;
      var lab = document.createElement("span");
      lab.textContent = it.name;
      key.appendChild(sw);
      key.appendChild(lab);
      lg.appendChild(key);
    });
    container.parentNode.insertBefore(lg, container);
  }
  function tableView(container, columns, rows) {
    var det = document.createElement("details");
    det.className = "table-view";
    var sum = document.createElement("summary");
    sum.textContent = (document.documentElement.lang || "").indexOf("ko") === 0 ? "데이터 표 보기" : "Show data table";
    det.appendChild(sum);
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    var table = document.createElement("table");
    var thead = table.createTHead().insertRow();
    columns.forEach(function (c) {
      var th = document.createElement("th");
      th.textContent = c.label;
      if (c.num) th.className = "num";
      thead.appendChild(th);
    });
    var tbody = table.createTBody();
    rows.forEach(function (r) {
      var tr = tbody.insertRow();
      columns.forEach(function (c) {
        var td = tr.insertCell();
        var v = r[c.key];
        td.textContent = v === undefined || v === null ? "–" : (c.format ? c.format(v) : v);
        if (c.num) td.className = "num";
      });
    });
    wrap.appendChild(table);
    det.appendChild(wrap);
    container.parentNode.appendChild(det);
  }

  // ---------- line chart ----------
  // opts: { series: [{name, color, points: [{x, y}]}], xFormat, yFormat, yLabel, xLabel, yMax }
  function responsive(container, draw) {
    var lastW = 0;
    function render() {
      var w = Math.max(300, Math.round(container.clientWidth));
      if (w === lastW) return;
      lastW = w;
      while (container.firstChild) container.removeChild(container.firstChild);
      draw(w);
    }
    render();
    var timer;
    window.addEventListener("resize", function () { clearTimeout(timer); timer = setTimeout(render, 120); });
  }

  function lineChart(container, opts) {
    if (opts.series.length > 1) legend(container, opts.series.map(function (s) { return { name: s.name, color: s.color }; }), "line");
    responsive(container, function (w) { drawLine(container, opts, w); });
  }

  function drawLine(container, opts, W) {
    var narrow = W < 560;
    var H = narrow ? 250 : 310, m = { t: 28, r: narrow ? 16 : 130, b: 38, l: 44 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var xs = [], yMaxData = 0, xMax = 0;
    opts.series.forEach(function (s) {
      s.points.forEach(function (p) {
        if (xs.indexOf(p.x) < 0) xs.push(p.x);
        yMaxData = Math.max(yMaxData, p.y);
        xMax = Math.max(xMax, p.x);
      });
    });
    xs.sort(function (a, b) { return a - b; });
    var yMax = opts.yMax || niceMax(yMaxData);
    var xMaxNice = niceMax(xMax);
    function X(v) { return m.l + (v / xMaxNice) * iw; }
    function Y(v) { return m.t + ih - (v / yMax) * ih; }

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.ariaLabel || "" }, container);
    var yTicks = 5;
    for (var i = 0; i <= yTicks; i++) {
      var v = (yMax / yTicks) * i, y = Y(v);
      el("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, "class": i === 0 ? "baseline" : "gridline" }, svg);
      text(svg, m.l - 8, y + 4, opts.tickFormat ? opts.tickFormat(v) : (opts.yFormat ? opts.yFormat(v) : String(v)), "axis-text", "end");
    }
    var xTicks = narrow ? 3 : 5;
    for (var j = 0; j <= xTicks; j++) {
      var xv = (xMaxNice / xTicks) * j;
      text(svg, X(xv), m.t + ih + 18, opts.xFormat ? opts.xFormat(xv) : String(xv), "axis-text", "middle");
    }
    if (opts.xLabel) text(svg, m.l + iw, H - 2, opts.xLabel, "axis-text", "end");
    if (opts.yLabel) text(svg, 0, 12, opts.yLabel, "axis-text", "start");

    var ends = [];
    opts.series.forEach(function (s) {
      var d = s.points.map(function (p, k) { return (k ? "L" : "M") + X(p.x).toFixed(1) + " " + Y(p.y).toFixed(1); }).join(" ");
      var path = el("path", { d: d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      path.style.stroke = s.color;
      var last = s.points[s.points.length - 1];
      var dot = el("circle", { cx: X(last.x), cy: Y(last.y), r: 4, "stroke-width": 2 }, svg);
      dot.style.fill = s.color;
      dot.style.stroke = "var(--surface)";
      ends.push({ s: s, x: X(last.x), y: Y(last.y), v: last.y });
    });
    // Direct end labels only when they do not collide
    ends.sort(function (a, b) { return a.y - b.y; });
    var collide = ends.some(function (e, k) { return k && Math.abs(e.y - ends[k - 1].y) < 16 && Math.abs(e.x - ends[k - 1].x) < 90; });
    if (!collide && !narrow && ends.length <= 4) {
      ends.forEach(function (e) {
        text(svg, e.x + 9, e.y + 4, e.s.name + " " + (opts.yFormat ? opts.yFormat(e.v) : e.v), "label-text");
      });
    }

    // Crosshair + tooltip
    var tip = makeTooltip(container);
    var cross = el("line", { y1: m.t, y2: m.t + ih, "class": "crosshair", opacity: 0 }, svg);
    var hover = el("g", {}, svg);
    var hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", tabindex: 0 }, svg);
    var focusIdx = xs.length - 1;
    function show(idx) {
      var xv = xs[idx];
      var px = X(xv);
      cross.setAttribute("x1", px);
      cross.setAttribute("x2", px);
      cross.setAttribute("opacity", 1);
      while (hover.firstChild) hover.removeChild(hover.firstChild);
      tip.textContent = "";
      var head = document.createElement("div");
      head.className = "t-head";
      head.textContent = opts.xFormat ? opts.xFormat(xv) + " " + (opts.xUnit || "") : xv;
      tip.appendChild(head);
      opts.series.forEach(function (s) {
        var p = s.points.filter(function (q) { return q.x === xv; })[0];
        if (!p) return;
        var c = el("circle", { cx: px, cy: Y(p.y), r: 4, "stroke-width": 2 }, hover);
        c.style.fill = s.color;
        c.style.stroke = "var(--surface)";
        var row = document.createElement("div");
        row.className = "t-row";
        var key = document.createElement("span");
        key.className = "t-key";
        key.style.background = s.color;
        var val = document.createElement("b");
        val.textContent = opts.yFormat ? opts.yFormat(p.y) : p.y;
        var name = document.createElement("span");
        name.textContent = s.name;
        row.appendChild(key);
        row.appendChild(val);
        row.appendChild(name);
        tip.appendChild(row);
      });
      var rect = svg.getBoundingClientRect();
      var scale = rect.width / W;
      placeTooltip(tip, container, px * scale, m.t * scale + 10);
    }
    function hide() {
      cross.setAttribute("opacity", 0);
      while (hover.firstChild) hover.removeChild(hover.firstChild);
      tip.style.opacity = 0;
    }
    function nearest(clientX) {
      var rect = svg.getBoundingClientRect();
      var vx = ((clientX - rect.left) / rect.width) * W;
      var best = 0;
      xs.forEach(function (xv, k) { if (Math.abs(X(xv) - vx) < Math.abs(X(xs[best]) - vx)) best = k; });
      return best;
    }
    hit.addEventListener("pointermove", function (e) { focusIdx = nearest(e.clientX); show(focusIdx); });
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", function () { show(focusIdx); });
    hit.addEventListener("blur", hide);
    hit.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight") { focusIdx = Math.min(xs.length - 1, focusIdx + 1); show(focusIdx); e.preventDefault(); }
      if (e.key === "ArrowLeft") { focusIdx = Math.max(0, focusIdx - 1); show(focusIdx); e.preventDefault(); }
    });

  }

  // ---------- horizontal bar chart ----------
  // opts: { items: [{label, value}], color, valueFormat, max }
  function barChart(container, opts) {
    responsive(container, function (w) { drawBars(container, opts, w); });
  }

  function drawBars(container, opts, W) {
    var narrow = W < 560;
    var barH = 22, rowH = narrow ? 54 : 40;
    var labelW = narrow ? 0 : Math.min(250, Math.round(W * 0.42));
    var barTop = narrow ? 22 : (rowH - barH) / 2;
    var H = opts.items.length * rowH + 10;
    var max = opts.max || niceMax(Math.max.apply(null, opts.items.map(function (d) { return d.value; })));
    var iw = W - labelW - 60;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.ariaLabel || "" }, container);
    el("line", { x1: labelW + 0.5, x2: labelW + 0.5, y1: 0, y2: H - 6, "class": "baseline" }, svg);
    var tip = makeTooltip(container);
    opts.items.forEach(function (d, i) {
      var y = i * rowH + barTop;
      var w = Math.max(2, (d.value / max) * iw);
      var r = Math.min(4, w);
      // square at the baseline, 4px rounded data end
      var path = "M" + labelW + " " + y + " H" + (labelW + w - r) + " Q" + (labelW + w) + " " + y + " " + (labelW + w) + " " + (y + r) +
        " V" + (y + barH - r) + " Q" + (labelW + w) + " " + (y + barH) + " " + (labelW + w - r) + " " + (y + barH) + " H" + labelW + " Z";
      var bar = el("path", { d: path }, svg);
      bar.style.fill = opts.color;
      if (narrow) text(svg, 0, y - 6, d.label, "label-text", "start");
      else text(svg, labelW - 10, y + barH / 2 + 4, d.label, "label-text", "end");
      text(svg, labelW + w + 8, y + barH / 2 + 4, opts.valueFormat ? opts.valueFormat(d.value) : d.value, "value-text");
      var hit = el("rect", { x: 0, y: i * rowH, width: W, height: rowH, fill: "transparent", tabindex: 0 }, svg);
      function show() {
        bar.style.opacity = 0.8;
        tip.textContent = "";
        var head = document.createElement("div");
        head.className = "t-head";
        head.textContent = d.label;
        var row = document.createElement("div");
        row.className = "t-row";
        var val = document.createElement("b");
        val.textContent = (opts.valueFormat ? opts.valueFormat(d.value) : d.value) + (opts.unit ? " " + opts.unit : "");
        row.appendChild(val);
        tip.appendChild(head);
        tip.appendChild(row);
        var rect = svg.getBoundingClientRect();
        var scale = rect.width / W;
        placeTooltip(tip, container, (labelW + w) * scale, y * scale);
      }
      function hide() { bar.style.opacity = 1; tip.style.opacity = 0; }
      hit.addEventListener("pointermove", show);
      hit.addEventListener("pointerleave", hide);
      hit.addEventListener("focus", show);
      hit.addEventListener("blur", hide);
    });
  }

  window.SiteCharts = { lineChart: lineChart, barChart: barChart, tableView: tableView };
})();
