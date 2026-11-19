"""Add a standalone HTML report to index.html as a new tab (loaded in a frame when first opened).

Usage, from the open-labs folder:
    python3 add_report_tab.py nifty_strangle_report.html "NIFTY strangle (1430 qty)"
Running it again with the same file does nothing.
"""
import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(__doc__)
report, label = sys.argv[1], sys.argv[2]
if not Path(report).exists():
    sys.exit(f"{report} not found next to index.html")

p = Path("index.html")
s = p.read_text(encoding="utf-8")
if f'data-report="{report}"' in s or f'"{report}"' in s:
    sys.exit(f"index.html already has a tab for {report}")

keys = [int(k) for k in re.findall(r'id="btn_t(\d+)"', s)]
key = f"t{max(keys) + 1}"

# 1. tab button, after the last existing one
last_btn = list(re.finditer(r'<button role="tab" id="btn_t\d+"[^>]*>.*?</button>', s))[-1]
btn = f'<button role="tab" id="btn_{key}" aria-selected="false">{label}</button>'
s = s[:last_btn.end()] + btn + s[last_btn.end():]

# 2. panel with an empty frame, before </main>
i = s.rindex("</main>")
panel = (f'\n<section class="tab" id="sec_{key}" role="tabpanel">'
         f'<iframe id="frame_{key}" data-report="{report}" title="{label}" '
         f'style="width:100%;border:0;min-height:900px;display:block"></iframe></section>')
s = s[:i] + panel + s[i:]

# 3. register the tab and load its frame on first open (before the tab script runs show())
anchor = "\nconst base"
assert anchor in s, "tab script not found in index.html"
hook = f'''
TABS.push({{key:"{key}",figs:[],tables:[]}});
(function(){{
  const fr = document.getElementById("frame_{key}");
  function fit(){{ try {{ fr.style.height = fr.contentDocument.documentElement.scrollHeight + "px"; }} catch(e) {{}} }}
  function load(){{ if (!fr.getAttribute("src")) fr.setAttribute("src", fr.dataset.report); else setTimeout(() => {{ try {{ fr.contentWindow.dispatchEvent(new Event("resize")); }} catch(e) {{}} fit(); }}, 50); }}
  fr.addEventListener("load", () => {{ fit(); try {{ new ResizeObserver(fit).observe(fr.contentDocument.body); }} catch(e) {{}} }});
  document.getElementById("btn_{key}").addEventListener("click", load);
  if (location.hash === "#{key}") load();
}})();'''
s = s.replace(anchor, hook + anchor, 1)
p.write_text(s, encoding="utf-8")
print(f"Added tab '{label}' ({key}) loading {report}")
