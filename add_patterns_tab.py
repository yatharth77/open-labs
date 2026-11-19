"""Adds a 'Market patterns' tab to index.html that loads patterns_report.html.
Run from the open-labs folder:  python3 add_patterns_tab.py   (safe to run twice)"""
from pathlib import Path
p = Path("index.html"); s = p.read_text(encoding="utf-8")
if 'id="btn_t3"' in s:
    print("index.html already has the Market patterns tab"); raise SystemExit
btn_anchor = '>First minute at 9:15</button>'
assert btn_anchor in s, "tab bar not found"
s = s.replace(btn_anchor, btn_anchor + '<button role="tab" id="btn_t3" aria-selected="false">Market patterns 2015–26</button>', 1)
assert '</section></main>' in s, "end of sections not found"
s = s.replace('</section></main>', '</section>\n<section class="tab" id="sec_t3" role="tabpanel">'
              '<iframe id="pat_frame" title="Market patterns backtest" style="width:100%;border:0;min-height:900px;display:block"></iframe>'
              '</section></main>', 1)
tabs_end = '}]}];\nconst base'
assert tabs_end in s, "TABS block not found"
hook = '''}]}];
TABS.push({key:"t3",figs:[],tables:[]});
(function(){
  const fr = document.getElementById("pat_frame");
  function fit(){ try { fr.style.height = fr.contentDocument.documentElement.scrollHeight + "px"; } catch(e) {} }
  function load(){ if (!fr.getAttribute("src")) fr.setAttribute("src", "patterns_report.html"); else setTimeout(() => { try { fr.contentWindow.dispatchEvent(new Event("resize")); } catch(e) {} fit(); }, 50); }
  fr.addEventListener("load", () => { fit(); try { new ResizeObserver(fit).observe(fr.contentDocument.body); } catch(e) {} });
  document.getElementById("btn_t3").addEventListener("click", load);
  if (location.hash === "#t3") load();
})();
const base'''
s = s.replace(tabs_end, hook, 1)
p.write_text(s, encoding="utf-8"); print("Added the Market patterns tab to index.html")
