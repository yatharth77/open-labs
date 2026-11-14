#!/usr/bin/env python3
"""
merge_reports.py - join several generated HTML reports into one tabbed page.

Usage:
    python merge_reports.py index.html open_minute_report.html -o merged.html
    python merge_reports.py a.html:"Strangle lab" b.html:"First minute" -o merged.html

Each input keeps its own charts, tables, filters and wording; only one tab is
shown at a time. Charts and tables are built the first time their tab is opened,
so a large merged page still loads quickly.

Element ids are prefixed per tab, so two reports that both contain, say,
f_NIFTY_0 no longer clash.
"""
import argparse
import json
import re
from pathlib import Path

CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--bad:#d03b3b;--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
main{max-width:1160px;margin:0 auto;padding:0 20px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:21px;margin:44px 0 12px}
h3{font-size:15px;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 20px}
.note{color:var(--muted);font-size:13px;margin:0 0 8px}
header.top{position:sticky;top:0;z-index:20;background:var(--page);
border-bottom:1px solid var(--border);padding:14px 20px 0}
header.top .in{max-width:1160px;margin:0 auto}
header.top h1{font-size:19px;margin:0 0 10px}
.tabs{display:flex;flex-wrap:wrap;gap:4px}
.tabs button{font:inherit;font-size:14px;padding:9px 16px;border:1px solid transparent;
border-bottom:none;border-radius:8px 8px 0 0;background:transparent;color:var(--ink2);cursor:pointer}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-selected="true"]{background:var(--surface);border-color:var(--border);
color:var(--ink);font-weight:600;margin-bottom:-1px}
section.tab{display:none;padding-top:26px}section.tab.on{display:block}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0}
nav a{font-size:13px;padding:5px 12px;border:1px solid var(--border);border-radius:999px;
background:var(--surface);color:var(--ink);text-decoration:none}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:24px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:18px 18px 8px;margin:16px 0}
ul.ins{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:16px 16px 16px 34px}
ul.ins li{margin:6px 0}
.card ul{margin:6px 0 12px;padding-left:20px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}
.assume{font-size:13px;color:var(--ink2)}
.scroll{overflow:auto;margin-bottom:12px}.scroll td,.scroll th{white-space:nowrap}
.scroll.tall{max-height:520px}.scroll.tall th{position:sticky;top:0;background:var(--surface)}
.xp-bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:end;margin:8px 0 10px;
font-size:13px;color:var(--ink2)}
.xp-bar label{display:flex;flex-direction:column;gap:2px}
.xp-bar select,.xp-bar input{font:inherit;padding:5px 8px;border:1px solid var(--border);
border-radius:6px;background:var(--surface);color:var(--ink)}
.xp-wrap{overflow:auto;max-height:560px;border:1px solid var(--grid);border-radius:8px}
.xp-wrap table{font-size:13px}.xp-wrap td,.xp-wrap th{white-space:nowrap;padding:6px 10px}
.xp-wrap th{position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none}
.xp-wrap td.n{text-align:right}.xp-wrap td.neg{color:var(--bad)}
.xp-foot{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;
font-size:13px;color:var(--ink2);margin:10px 0 12px}
.xp-foot button{font:inherit;padding:4px 10px;border:1px solid var(--border);border-radius:6px;
background:var(--surface);color:var(--ink);cursor:pointer}
.xp-foot button:disabled{opacity:.4;cursor:default}
"""

SCRIPT = r"""
const TABS = __TABS__;
const base = {paper_bgcolor:"#fcfcfb",plot_bgcolor:"#fcfcfb",
  font:{family:'system-ui,-apple-system,"Segoe UI",sans-serif',color:"#52514e",size:12},
  margin:{l:64,r:20,t:10,b:56},hovermode:"closest",
  legend:{orientation:"h",y:-0.22},
  xaxis:{gridcolor:"#e1e0d9",linecolor:"#c3c2b7",zeroline:false},
  yaxis:{gridcolor:"#e1e0d9",linecolor:"#c3c2b7",zeroline:false}};
const PAGE = 50;
function cell(v, kind){
  if (v === null || v === undefined || v === "") return "–";
  if (kind === "rs") return (v < 0 ? "−₹" : "₹") + Math.abs(v).toLocaleString("en-IN", {maximumFractionDigits: 0});
  if (kind === "pct") return (v > 0 ? "+" : "") + v.toLocaleString("en-IN", {maximumFractionDigits: 2}) + "%";
  if (kind === "num") return v.toLocaleString("en-IN", {maximumFractionDigits: 2});
  return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;");
}
function drawFig(f){
  const el = document.getElementById(f.id);
  if (!el) return;
  const lay = JSON.parse(JSON.stringify(base));
  for (const k in f.layout){
    const v = f.layout[k];
    lay[k] = (v && typeof v === "object" && !Array.isArray(v)) ? Object.assign({}, lay[k]||{}, v) : v;
  }
  Plotly.newPlot(f.id, f.data, lay, {displaylogo:false, responsive:true});
}
function drawTable(t){
  const root = document.getElementById(t.id);
  if (!root || root.dataset.built) return;
  root.dataset.built = "1";
  const st = {sort:null, dir:1, page:0, f:Object.assign({}, t.defaults||{}), q:""};
  const bar = document.createElement("div"); bar.className = "xp-bar";
  for (const ci of (t.filters||[])){
    const vals = [...new Set(t.rows.map(r => String(r[ci])))];
    const numeric = vals.every(v => v !== "" && !isNaN(parseFloat(v)));
    vals.sort((a,b) => numeric ? parseFloat(a)-parseFloat(b) : a.localeCompare(b));
    const lab = document.createElement("label"); lab.textContent = t.cols[ci].label;
    const sel = document.createElement("select");
    sel.innerHTML = '<option value="">All</option>' + vals.map(v => `<option>${v}</option>`).join("");
    if (st.f[ci] !== undefined && vals.includes(st.f[ci])) sel.value = st.f[ci]; else delete st.f[ci];
    sel.onchange = () => { if (sel.value) st.f[ci] = sel.value; else delete st.f[ci]; st.page = 0; draw(); };
    lab.appendChild(sel); bar.appendChild(lab);
  }
  const sl = document.createElement("label"); sl.textContent = "Search";
  const inp = document.createElement("input"); inp.type = "search"; inp.placeholder = "e.g. 2025-04";
  inp.oninput = () => { st.q = inp.value.trim().toLowerCase(); st.page = 0; draw(); };
  sl.appendChild(inp); bar.appendChild(sl);
  const wrap = document.createElement("div"); wrap.className = "xp-wrap";
  const foot = document.createElement("div"); foot.className = "xp-foot";
  root.append(bar, wrap, foot);
  function draw(){
    let rows = t.rows.filter(r => {
      for (const k in st.f) if (String(r[k]) !== st.f[k]) return false;
      return !st.q || r.some(v => v !== null && String(v).toLowerCase().includes(st.q));
    });
    if (st.sort !== null){
      const i = st.sort, isNum = t.cols[i].kind !== "text";
      rows = rows.slice().sort((a,b) => {
        const x = a[i], y = b[i];
        if (x === null || x === "") return 1; if (y === null || y === "") return -1;
        return st.dir * (isNum ? x - y : String(x).localeCompare(String(y)));
      });
    }
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    st.page = Math.min(st.page, pages - 1);
    const view = rows.slice(st.page * PAGE, (st.page + 1) * PAGE);
    const head = t.cols.map((c,i) => `<th data-i="${i}">${c.label}${st.sort === i ? (st.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
    const body = view.map(r => "<tr>" + r.map((v,i) => {
      const k = t.cols[i].kind;
      return `<td class="${k !== "text" ? "n" : ""}${(k === "rs" || k === "pct") && v < 0 ? " neg" : ""}">${cell(v,k)}</td>`;
    }).join("") + "</tr>").join("");
    wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    wrap.querySelectorAll("th").forEach(th => th.onclick = () => {
      const i = +th.dataset.i; st.dir = st.sort === i ? -st.dir : 1; st.sort = i; draw();
    });
    let sum = "";
    if (t.sum !== null && t.sum !== undefined && rows.length){
      const tot = rows.reduce((a,r) => a + (r[t.sum] || 0), 0);
      sum = ` · total ${cell(Math.round(tot), "rs")} · average ${cell(Math.round(tot/rows.length), "rs")} per row`;
    }
    foot.innerHTML = `<span>${rows.length.toLocaleString("en-IN")} rows${sum}</span><span><button class="pv">Previous</button> page ${st.page+1} of ${pages} <button class="nx">Next</button></span>`;
    const pv = foot.querySelector(".pv"), nx = foot.querySelector(".nx");
    pv.disabled = st.page === 0; nx.disabled = st.page >= pages - 1;
    pv.onclick = () => { st.page--; draw(); }; nx.onclick = () => { st.page++; draw(); };
  }
  draw();
}
function show(key){
  for (const t of TABS){
    const on = t.key === key;
    document.getElementById("sec_" + t.key).classList.toggle("on", on);
    document.getElementById("btn_" + t.key).setAttribute("aria-selected", on ? "true" : "false");
  }
  const t = TABS.find(x => x.key === key);
  if (!t.built){ t.built = true; t.figs.forEach(drawFig); t.tables.forEach(drawTable); }
  if (location.hash.slice(1) !== key) history.replaceState(null, "", "#" + key);
  window.dispatchEvent(new Event("resize"));
}
for (const t of TABS){
  document.getElementById("btn_" + t.key).onclick = () => { show(t.key); window.scrollTo(0,0); };
}
show(TABS.find(t => t.key === location.hash.slice(1)) ? location.hash.slice(1) : TABS[0].key);
"""

PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>__CSS__</style></head><body>
<header class="top"><div class="in"><h1>__TITLE__</h1>
<div class="tabs" role="tablist">__BUTTONS__</div></div></header>
<main>__SECTIONS__</main>
<script>__SCRIPT__</script></body></html>"""


def grab(s, start, end):
    i = s.find(start)
    j = s.find(end, i + len(start))
    if i < 0 or j < 0:
        raise ValueError(f"could not find {start!r} ... {end!r}")
    return s[i + len(start):j]


def json_after(s, marker):
    """The JSON array assigned after `marker`, e.g. 'const FIGS = '."""
    i = s.find(marker)
    if i < 0:
        return []
    i += len(marker)
    while s[i] in " \t\r\n":
        i += 1
    if s[i] != "[":
        return []
    depth, in_str, esc = 0, False, False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return json.loads(s[i:j + 1])
    raise ValueError(f"unterminated array after {marker!r}")


def load(path, key):
    s = Path(path).read_text(encoding="utf-8")
    body = grab(s, "<main>", "</main>")
    figs = json_after(s, "const FIGS = ")
    tables = json_after(s, "const TABLES = ")
    pre = key + "__"
    body = re.sub(r'(\sid=")([^"]+)(")', lambda m: m.group(1) + pre + m.group(2) + m.group(3), body)
    body = re.sub(r'(\shref="#)([^"]+)(")', lambda m: m.group(1) + pre + m.group(2) + m.group(3),
                  body)
    for f in figs:
        f["id"] = pre + f["id"]
    for t in tables:
        t["id"] = pre + t["id"]
    title = re.search(r"<h1>(.*?)</h1>", body)
    return {"key": key, "body": body, "figs": figs, "tables": tables,
            "title": title.group(1) if title else path}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help='HTML files, each optionally "file.html:Tab label"')
    ap.add_argument("-o", "--out", default="merged_report.html")
    ap.add_argument("--title", default="Options research")
    a = ap.parse_args()

    tabs, buttons, sections = [], [], []
    for n, item in enumerate(a.inputs, 1):
        path, _, label = item.partition(":")
        if not Path(path).exists():
            raise SystemExit(f"no such file: {path}")
        key = f"t{n}"
        r = load(path, key)
        label = label or r["title"]
        size = Path(path).stat().st_size / 1048576
        print(f"tab {n}: {label}  ({size:.1f} MB, {len(r['figs'])} charts, "
              f"{len(r['tables'])} tables)")
        buttons.append(f'<button role="tab" id="btn_{key}" aria-selected="false">{label}</button>')
        sections.append(f'<section class="tab" id="sec_{key}" role="tabpanel">{r["body"]}</section>')
        tabs.append({"key": key, "figs": r["figs"], "tables": r["tables"]})

    script = SCRIPT.replace("__TABS__", json.dumps(tabs, ensure_ascii=False,
                                                   separators=(",", ":")).replace("</", "<\\/"))
    html = (PAGE.replace("__TITLE__", a.title).replace("__CSS__", CSS)
            .replace("__BUTTONS__", "".join(buttons))
            .replace("__SECTIONS__", "\n".join(sections))
            .replace("__SCRIPT__", script))
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"\nWrote {Path(a.out).resolve()}  ({Path(a.out).stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
