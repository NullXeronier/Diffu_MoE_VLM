/* Code browser for the project site: file tree + highlighted source from window.CODE_SNAPSHOT */
(function () {
  "use strict";
  var snap = window.CODE_SNAPSHOT, root = document.getElementById("code-viewer");
  if (!snap || !root) return;
  var ko = (document.documentElement.lang || "en").indexOf("ko") === 0;
  var T = ko ? { filter: "파일 이름으로 찾기", lines: "줄", copy: "복사", copied: "복사됨", files: "파일", root: "(루트)" }
             : { filter: "Filter files", lines: "lines", copy: "Copy", copied: "Copied", files: "Files", root: "(root)" };
  var byPath = {};
  snap.files.forEach(function (f) { byPath[f.path] = f; });

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /* ---------- tokenizer ---------- */
  var PY_KW = "False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield";
  var RULES = {
    py: new RegExp([
      "(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?''')",                       // 1 docstring
      "(#[^\\n]*)",                                                        // 2 comment
      "((?:\\b[rRbBfF]{1,2})?(?:\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*'))", // 3 string
      "(@[\\w.]+)",                                                        // 4 decorator
      "(\\b(?:" + PY_KW + ")\\b)",                                         // 5 keyword
      "(\\b\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b)",                  // 6 number
      "((?<=\\b(?:def|class) )\\w+)"                                       // 7 definition name
    ].join("|"), "g"),
    yaml: new RegExp([
      "(\"\"\"[\\s\\S]*?\"\"\")",                                          // 1 (unused)
      "(#[^\\n]*)",                                                        // 2 comment
      "(\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:[^'\\n])*')",                      // 3 string
      "(^\\s*\\[[^\\]\\n]+\\])",                                           // 4 TOML table
      "(\\b(?:true|false|null|True|False|None)\\b)",                       // 5 literal
      "(\\b\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b)",                  // 6 number
      "(^[ \\t]*-?[ \\t]*[\\w.\\-]+(?=[ \\t]*[:=]))"                       // 7 key
    ].join("|"), "gm")
  };
  var CLS = [null, "tk-str", "tk-com", "tk-str", "tk-dec", "tk-kw", "tk-num", "tk-def"];

  function tokens(text, lang) {
    var re = RULES[lang], out = [], last = 0, m;
    if (!re) return [[null, text]];
    re.lastIndex = 0;
    while ((m = re.exec(text))) {
      if (m[0] === "") { re.lastIndex++; continue; }
      if (m.index > last) out.push([null, text.slice(last, m.index)]);
      for (var g = 1; g < m.length; g++) if (m[g] !== undefined) { out.push([CLS[g], m[0]]); break; }
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push([null, text.slice(last)]);
    return out;
  }

  function langOf(path) { return /\.py$/.test(path) ? "py" : (/\.(ya?ml|toml)$/.test(path) ? "yaml" : null); }

  /* ---------- layout ---------- */
  var side = el("div", "cv-side"), main = el("div", "cv-main");
  var filter = el("input", "cv-filter");
  filter.type = "search"; filter.placeholder = T.filter; filter.setAttribute("aria-label", T.filter);
  var tree = el("div", "cv-tree");
  side.appendChild(filter); side.appendChild(tree);
  var head = el("div", "cv-head"), title = el("code", "cv-path"), meta = el("span", "cv-meta"), copy = el("button", "cv-copy", T.copy);
  copy.type = "button";
  head.appendChild(title); head.appendChild(meta); head.appendChild(copy);
  var body = el("div", "cv-body");
  body.setAttribute("tabindex", "0");
  main.appendChild(head); main.appendChild(body);
  root.appendChild(side); root.appendChild(main);

  var groups = {};
  snap.files.forEach(function (f) {
    var i = f.path.lastIndexOf("/"), dir = i < 0 ? "" : f.path.slice(0, i);
    (groups[dir] = groups[dir] || []).push(f.path);
  });
  var links = {};
  Object.keys(groups).sort(function (a, b) { return a === "" ? -1 : b === "" ? 1 : a < b ? -1 : 1; }).forEach(function (dir) {
    var d = el("details", "cv-dir");
    d.open = dir === "" || dir.indexOf("diffu_moe_vlm") === 0 || dir === "configs";
    d.appendChild(el("summary", null, dir === "" ? T.root : dir + "/"));
    var ul = el("ul");
    groups[dir].forEach(function (p) {
      var li = el("li"), a = el("a", null, p.slice(dir ? dir.length + 1 : 0));
      a.href = "#f=" + p;
      links[p] = a;
      li.appendChild(a); ul.appendChild(li);
    });
    d.appendChild(ul); tree.appendChild(d);
  });

  filter.addEventListener("input", function () {
    var q = filter.value.trim().toLowerCase();
    tree.querySelectorAll(".cv-dir").forEach(function (d) {
      var any = false;
      d.querySelectorAll("li").forEach(function (li) {
        var hit = !q || li.firstChild.getAttribute("href").toLowerCase().indexOf(q) >= 0;
        li.hidden = !hit; any = any || hit;
      });
      d.hidden = !any;
      if (q && any) d.open = true;
    });
  });

  var current = null;
  function show(path, line) {
    var f = byPath[path];
    if (!f) return;
    if (current !== path) {
      current = path;
      body.textContent = "";
      var lines = [[]];
      tokens(f.text.replace(/\n$/, ""), langOf(path)).forEach(function (t) {
        t[1].split("\n").forEach(function (piece, i) {
          if (i > 0) lines.push([]);
          if (piece) lines[lines.length - 1].push([t[0], piece]);
        });
      });
      var frag = document.createDocumentFragment();
      lines.forEach(function (parts, i) {
        var row = el("div", "cl"), n = el("a", "ln", String(i + 1)), c = el("span", "lc");
        row.id = "L" + (i + 1);
        n.href = "#f=" + path + "&L=" + (i + 1);
        parts.forEach(function (p) { c.appendChild(p[0] ? el("span", p[0], p[1]) : document.createTextNode(p[1])); });
        if (!parts.length) c.textContent = " ";
        row.appendChild(n); row.appendChild(c); frag.appendChild(row);
      });
      body.appendChild(frag);
      title.textContent = path;
      meta.textContent = lines.length + " " + T.lines;
      Object.keys(links).forEach(function (p) { links[p].classList.toggle("active", p === path); });
      var a = links[path];
      if (a) { a.closest("details").open = true; }
    }
    body.querySelectorAll(".cl.hl").forEach(function (r) { r.classList.remove("hl"); });
    if (line) {
      var r = document.getElementById("L" + line);
      if (r) { r.classList.add("hl"); body.scrollTop = r.offsetTop - body.clientHeight / 3; }
    } else {
      body.scrollTop = 0;
    }
  }

  copy.addEventListener("click", function () {
    var f = byPath[current];
    if (!f || !navigator.clipboard) return;
    navigator.clipboard.writeText(f.text).then(function () {
      copy.textContent = T.copied;
      setTimeout(function () { copy.textContent = T.copy; }, 1200);
    });
  });

  function route(scroll) {
    var m = /#f=([^&]+)(?:&L=(\d+))?/.exec(location.hash);
    var path = m ? decodeURIComponent(m[1]) : root.getAttribute("data-default");
    show(path, m && m[2] ? +m[2] : 0);
    if (scroll && m) root.scrollIntoView({ block: "start" });
  }
  window.addEventListener("hashchange", function () { route(true); });
  route(false);
})();
