# -*- coding: utf-8 -*-
"""把精选的 *_report.html 汇成 docs/ 下的 GitHub Pages 画廊(零后端、零密钥)。
用法: py -X utf8 build_gallery.py
产物: docs/index.html + docs/reports/<slug>.html
"""
import json, os, shutil, html, re

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
RDIR = os.path.join(DOCS, "reports")

# (slug, 主题, 类型标签) —— 当前只放一份精选样例
CURATED = [
    ("anti-fouling",       "Anti Fouling",         "技术综述"),
]

def teaser(topic):
    """从 <主题>_plan.json 摘一句话当卡片副文案。"""
    try:
        d = json.load(open(os.path.join(HERE, f"{topic}_plan.json"), encoding="utf-8"))
        dt = d.get("plan", {}).get("deep_thinking", {})
        t = dt.get("context_analysis") or ""
        t = re.sub(r"\s+", " ", t).strip()
        # 优先用第一步结论(更具体)
        res = d.get("results") or []
        if res and res[0].get("main_conclusion"):
            t = re.sub(r"\s+", " ", res[0]["main_conclusion"]).strip()
        return t[:96] + ("…" if len(t) > 96 else "")
    except Exception:
        return "深度研究报告"

def main():
    os.makedirs(RDIR, exist_ok=True)
    cards = []
    for slug, topic, tag in CURATED:
        src = os.path.join(HERE, f"{topic}_report.html")
        if not os.path.isfile(src):
            print(f"  [skip] 缺报告: {topic}_report.html"); continue
        shutil.copyfile(src, os.path.join(RDIR, f"{slug}.html"))
        cards.append(f'''    <a class="card" href="reports/{slug}.html">
      <span class="tag">{html.escape(tag)}</span>
      <h3>{html.escape(topic)}</h3>
      <p>{html.escape(teaser(topic))}</p>
      <span class="go">打开报告 ›</span>
    </a>''')
        print(f"  [ok] {topic} -> reports/{slug}.html")
    page = INDEX.replace("__CARDS__", "\n".join(cards)).replace("__N__", str(len(cards)))
    open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8").write(page)
    print(f"已生成 docs/index.html({len(cards)} 张)")

INDEX = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>深度行研 Agent · 报告画廊</title>
<style>
:root{--green:#6fae7e;--green-d:#4f8a60;--ink:#23282b;--mut:#7d878c;--line:#e7e9e8;--bg:#f6f7f5}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--ink);
  background:radial-gradient(1200px 600px at 50% -10%,#fbfdfc,var(--bg))}
header{max-width:1080px;margin:0 auto;padding:64px 28px 30px;text-align:center}
header h1{font-size:38px;margin:0 0 10px;letter-spacing:1px}
header .tag{color:var(--mut);font-size:13px;letter-spacing:3px;margin-bottom:18px}
header p{color:#56606a;font-size:15px;line-height:1.8;max-width:680px;margin:0 auto}
header .breathe{display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--green);
  margin-right:8px;animation:b 1.7s ease-in-out infinite;vertical-align:middle}
@keyframes b{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1.25)}}
.grid{max-width:1080px;margin:0 auto;padding:18px 24px 70px;display:grid;
  grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}
.card{display:flex;flex-direction:column;background:#fff;border:1px solid var(--line);border-radius:16px;
  padding:22px 22px 18px;text-decoration:none;color:inherit;box-shadow:0 6px 24px rgba(80,110,90,.06);
  transition:.18s}
.card:hover{transform:translateY(-3px);border-color:var(--green);box-shadow:0 12px 34px rgba(111,174,126,.16)}
.card .tag{align-self:flex-start;font-size:11.5px;background:#eef5f0;color:var(--green-d);
  padding:3px 10px;border-radius:20px;font-weight:600;letter-spacing:.5px}
.card h3{margin:13px 0 8px;font-size:19px;line-height:1.3}
.card p{margin:0 0 16px;color:var(--mut);font-size:13px;line-height:1.6;flex:1}
.card .go{color:var(--green-d);font-weight:700;font-size:13.5px}
footer{text-align:center;color:#a7aeaa;font-size:12.5px;padding:0 24px 50px;line-height:1.9}
footer a{color:var(--green-d)}
@media (max-width:640px){
  header{padding:44px 20px 18px}
  header h1{font-size:28px}
  header p{font-size:14px;line-height:1.7}
  .grid{grid-template-columns:1fr;padding:14px 16px 50px;gap:14px}
  .card{padding:20px 18px 16px}
  .card h3{font-size:18px}
}
</style></head>
<body>
<header>
  <h1><span class="breathe"></span>深度行研 Agent</h1>
  <div class="tag">Claude 设计 · DeepSeek 驱动 · 会呼吸的研究</div>
  <p>输入一个主题,它先做 4 轮迭代规划、再按依赖顺序逐步检索与撰写,像呼吸一样实时刷新。
  下面是它产出的研究报告样例 —— 点开可左右滑动浏览每一步的思考链。</p>
</header>
<div class="grid">
__CARDS__
</div>
<footer>
  这些是静态成品快照(零后端、零密钥)。完整的"会呼吸"实时界面与源码见
  <a href="https://github.com/Sugiady/breathing-research">代码仓库 README</a>。<br>
  Built with research methodology + DeepSeek · 界面与工程由 Claude 设计。
</footer>
</body></html>"""

if __name__ == "__main__":
    main()
