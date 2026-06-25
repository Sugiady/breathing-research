# -*- coding: utf-8 -*-
"""
把 research_agent.py 产出的 <主题>_plan.json 渲染成单文件、零依赖、零网络的 HTML 查看器。
左:固定大标题 + 几大要点(每步一句话结论)。右:可鼠标/触摸左右拖拽的快照窗,每步一张卡,像思考链。
用法: py -X utf8 render_html.py "AEM电解水制氢_plan.json"
产物: 同名 .html (file:// 直接双击打开,不联网,无 CORS)
"""
import json, sys, os, re, html

# LaTeX/化学式兜底:把漏网的 $...$、\command、_{}/^{} 归一成 Unicode 与 <sub>/<sup>。
_TEX = {r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
        r"\leftrightarrow": "↔", r"\uparrow": "↑", r"\downarrow": "↓",
        r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\geq": "≥", r"\leq": "≤",
        r"\approx": "≈", r"\neq": "≠", r"\Delta": "Δ", r"\alpha": "α", r"\beta": "β",
        r"\circ": "°", r"\degree": "°", r"\%": "%", r"\,": " ", r"\ ": " ",
        r"\left": "", r"\right": "", r"\text": ""}

def mathify(s):
    """s 已是 HTML 转义后的字符串。仅做保守替换,避免误伤普通下划线。"""
    # 去 $...$ / \(...\) / \[...\] 包裹,保留里面内容
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = re.sub(r"\\\((.+?)\\\)", r"\1", s)
    s = re.sub(r"\\\[(.+?)\\\]", r"\1", s)
    for k, v in _TEX.items():
        s = s.replace(k, v)
    # 上下标:花括号形式总是转;裸形式仅紧跟字母/数字/右括号时转(像公式才转)
    s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"(?<=[A-Za-z0-9\)\]])_([0-9]+)", r"<sub>\1</sub>", s)        # 裸下标只吃数字(化学式)
    s = re.sub(r"(?<=[A-Za-z0-9\)\]])\^([0-9]*[+\-]|[0-9]+)", r"<sup>\1</sup>", s)  # 裸上标:数字/电荷
    return s

def sources_html(items):
    """渲染搜索来源脚注:可点击的 [原文] 列表。"""
    if not items:
        return ""
    lis = []
    for x in items:
        url = html.escape(x.get("url", ""), quote=True)
        name = html.escape(x.get("name", "") or url)
        meta = " · ".join([p for p in (x.get("date", ""), x.get("site", "")) if p])
        meta = f'<span class="src-meta">{html.escape(meta)}</span>' if meta else ""
        lis.append(f'<li><a href="{url}" target="_blank" rel="noopener">{name}</a> '
                   f'<a class="src-link" href="{url}" target="_blank" rel="noopener">[原文]</a>{meta}</li>')
    return '<div class="sources"><h4>📎 检索来源</h4><ol>' + "".join(lis) + "</ol></div>"

def md_to_html(text):
    """够用就好的 markdown:标题/加粗/无序列表/表格/段落。内容由我们自己的 prompt 产出,范围可控。"""
    if not text:
        return ""
    lines = text.replace("\r", "").split("\n")
    out, i = [], 0
    def esc(s): return html.escape(s, quote=False)
    def inline(s):
        s = esc(s)
        s = re.sub(r"&lt;br\s*/?\s*&gt;", "<br>", s)   # 模型爱塞字面 <br>,转义后还原成真换行
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return mathify(s)
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1; continue
        # 表格
        if ln.lstrip().startswith("|") and "|" in ln[1:]:
            tbl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i].strip()); i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in tbl]
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]  # 去分隔行
            if rows:
                ncol = len(rows[0])
                def fit(r):   # 规整到表头列数:多出的并进末列(防单元格内裸|切错),不足的补空
                    if len(r) > ncol:
                        return r[:ncol - 1] + [" ".join(r[ncol - 1:])]
                    return r + [""] * (ncol - len(r))
                out.append('<div class="tbl-wrap"><table><thead><tr>'
                           + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
                for r in rows[1:]:
                    out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in fit(r)) + "</tr>")
                out.append("</tbody></table></div>")
            continue
        # 标题(封顶 h5,避免 #### 落到浏览器默认超小的 h6)
        m = re.match(r"(#{1,6})\s+(.*)", ln)
        if m:
            lvl = min(len(m.group(1)) + 2, 5)
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>"); i += 1; continue
        # 列表
        if re.match(r"\s*([-·*]|\d+[.、)])\s+", ln):
            out.append("<ul>")
            while i < len(lines) and re.match(r"\s*([-·*]|\d+[.、)])\s+", lines[i]):
                item = re.sub(r"^\s*([-·*]|\d+[.、)])\s+", "", lines[i].rstrip())
                out.append(f"<li>{inline(item)}</li>"); i += 1
            out.append("</ul>"); continue
        out.append(f"<p>{inline(ln)}</p>"); i += 1
    return "\n".join(out)

def build(plan_path):
    data = json.load(open(plan_path, encoding="utf-8"))
    topic = data["topic"]
    plan = data["plan"]
    results = {r["id"]: r for r in data["results"]}
    steps = sorted(plan["steps"], key=lambda s: s["id"])
    dt = plan["deep_thinking"]

    # 卡片0:深度思考
    cards = []
    dt_html = ("<h3>主题研判</h3>" + md_to_html(dt.get("context_analysis", "")) +
               "<h3>研究计划</h3>" + md_to_html(dt.get("research_plan", "")) +
               "<h3>范围拆解</h3>" + md_to_html(dt.get("scope_decomposition", "")))
    cards.append(("0", "深度思考", "thinking", "", "", dt_html))
    # 各步卡片
    for s in steps:
        r = results.get(s["id"], {})
        dep = ",".join(map(str, s.get("depends_on") or []))
        body = md_to_html(r.get("detailed", "")) + sources_html(r.get("sources"))
        cards.append((str(s["id"]), s["name"], s.get("type", ""), dep,
                      r.get("main_conclusion", ""), body))

    # 左侧要点
    kp = []
    for s in steps:
        r = results.get(s["id"], {})
        kp.append(f'<li data-go="{s["id"]}"><span class="kp-no">{s["id"]}</span>'
                  f'<span class="kp-name">{html.escape(s["name"])}</span>'
                  f'<span class="kp-con">{html.escape(r.get("main_conclusion",""))}</span></li>')
    keypoints = "\n".join(kp)

    # 卡片 DOM
    card_html = []
    dots = []
    for idx, (no, name, typ, dep, con, body) in enumerate(cards):
        badge = f'<span class="tag">{html.escape(typ)}</span>' if typ else ""
        depbadge = f'<span class="dep">依赖 {dep}</span>' if dep else ""
        conhtml = f'<div class="concl">{html.escape(con)}</div>' if con else ""
        head_no = "思考" if no == "0" else no
        card_html.append(
            f'<article class="card"><div class="card-head"><span class="no">{head_no}</span>'
            f'<h2>{html.escape(name)}</h2>{badge}{depbadge}</div>{conhtml}'
            f'<div class="body">{body}</div></article>')
        dots.append(f'<button class="dot" data-i="{idx}" title="{html.escape(name)}"></button>')

    tpl = HTML_TEMPLATE
    tpl = tpl.replace("__TITLE__", html.escape(topic))
    tpl = tpl.replace("__COUNT__", str(len(cards)))
    tpl = tpl.replace("__KEYPOINTS__", keypoints)
    tpl = tpl.replace("__CARDS__", "\n".join(card_html))
    tpl = tpl.replace("__DOTS__", "\n".join(dots))
    out_path = os.path.splitext(plan_path)[0].replace("_plan", "") + "_report.html"
    open(out_path, "w", encoding="utf-8").write(tpl)
    print("已生成:", out_path)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · 深度研究</title>
<style>
:root{--green:#6fae7e;--green-d:#4f8a60;--ink:#23282b;--mut:#7d878c;--line:#e7e9e8;--bg:#f6f7f5;--card:#fff}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{display:grid;grid-template-columns:minmax(300px,38%) 1fr;gap:0;height:100vh}
.left{padding:34px 30px;overflow:auto;border-right:1px solid var(--line);background:#fbfcfb}
.left h1{font-size:30px;line-height:1.25;margin:0 0 6px}
.left .sub{color:var(--mut);font-size:13px;margin-bottom:24px}
.left h4{font-size:12px;letter-spacing:2px;color:var(--green);margin:0 0 12px;font-weight:700}
ul.kp{list-style:none;margin:0;padding:0}
ul.kp li{padding:12px 12px;border-radius:10px;cursor:pointer;display:grid;grid-template-columns:24px 1fr;gap:4px 10px;transition:.15s;border:1px solid transparent}
ul.kp li:hover{background:#fff;border-color:var(--line)}
ul.kp li.active{background:#fff;border-color:var(--green);box-shadow:0 2px 10px rgba(111,174,126,.15)}
.kp-no{grid-row:1/3;width:24px;height:24px;border-radius:50%;background:var(--green);color:#fff;font-size:13px;display:flex;align-items:center;justify-content:center;font-weight:700}
.kp-name{font-weight:600;font-size:14px}
.kp-con{color:var(--mut);font-size:12.5px;line-height:1.5}
.right{position:relative;overflow:hidden;display:flex;flex-direction:column}
.stage{flex:1;overflow:hidden;position:relative}
.track{display:flex;height:100%;will-change:transform;cursor:grab}
.track.drag{cursor:grabbing;transition:none}
.track.snap{transition:transform .42s cubic-bezier(.22,.61,.36,1)}
.card{min-width:100%;height:100%;overflow:auto;padding:40px 52px 60px}
.card-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.card-head .no{width:34px;height:34px;border-radius:9px;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px}
.card-head h2{margin:0;font-size:23px}
.tag{font-size:12px;background:#eef5f0;color:var(--green);padding:3px 10px;border-radius:20px;font-weight:600}
.dep{font-size:12px;color:var(--mut);border:1px solid var(--line);padding:3px 9px;border-radius:20px}
.concl{background:linear-gradient(0deg,#fbfcfb,#fff);border-left:3px solid var(--green);padding:14px 18px;border-radius:0 10px 10px 0;font-size:15.5px;line-height:1.7;margin-bottom:22px;font-weight:500}
.body{font-size:14.5px;line-height:1.85;color:#2c3236}
.body h3,.body h4,.body h5,.body h6{margin:22px 0 8px;font-size:15.5px}
.body p{margin:9px 0}
.body ul{margin:8px 0;padding-left:22px}.body li{margin:4px 0}
.body b{color:var(--ink)}
.tbl-wrap{overflow-x:auto;margin:14px 0}
.body table{border-collapse:collapse;width:100%;font-size:13px}
.body th,.body td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top;word-break:break-word}
.body th{background:#f1f4f2;font-weight:600}
.body tr:nth-child(even) td{background:#fbfcfb}
.body sub{font-size:.72em}.body sup{font-size:.72em}
.sources{margin-top:26px;padding-top:16px;border-top:1px dashed var(--line)}
.sources h4{font-size:12px;letter-spacing:2px;color:var(--green);margin:0 0 10px;font-weight:700}
.sources ol{margin:0;padding-left:22px;font-size:13px;line-height:1.7}
.sources li{margin:5px 0}
.sources a{color:#2c3236;text-decoration:none}.sources a:hover{text-decoration:underline}
.src-link{color:var(--green-d) !important;font-weight:600;margin:0 6px}
.src-meta{color:var(--mut);font-size:12px;margin-left:4px}
.nav{display:flex;align-items:center;justify-content:center;gap:14px;padding:12px;border-top:1px solid var(--line);background:#fbfcfb}
.nav button.arrow{border:1px solid var(--line);background:#fff;width:38px;height:38px;border-radius:50%;cursor:pointer;font-size:17px;color:var(--ink)}
.nav button.arrow:hover{border-color:var(--green);color:var(--green)}
.dots{display:flex;gap:7px}
.dot{width:9px;height:9px;border-radius:50%;border:none;background:#d3d8d4;cursor:pointer;padding:0}
.dot.active{background:var(--green);transform:scale(1.3)}
.pos{font-size:12px;color:var(--mut);min-width:54px;text-align:center}
.hint{position:absolute;top:14px;right:20px;font-size:12px;color:var(--mut);background:#fff;border:1px solid var(--line);padding:4px 10px;border-radius:20px}
/* ---- 手机/平板:两栏改上下堆叠,滑动区给足高度 ---- */
@media (max-width:820px){
  .wrap{grid-template-columns:1fr;height:auto;min-height:100vh}
  .left{max-height:38vh;border-right:none;border-bottom:1px solid var(--line);padding:20px 18px}
  .left h1{font-size:22px}.left .sub{margin-bottom:16px}
  .right{height:62vh}
  .card{padding:22px 18px 44px}
  .card-head h2{font-size:19px}
  .concl{font-size:14.5px;padding:12px 14px}
  .body{font-size:14px}.body table{font-size:12px}
  .hint{display:none}
}
@media (max-width:480px){
  .left{max-height:34vh}.right{height:66vh}
  .card{padding:18px 14px 40px}
  .card-head h2{font-size:18px}
  .nav button.arrow{width:34px;height:34px;font-size:16px}
}
</style></head>
<body><div class="wrap">
<aside class="left">
  <h1>__TITLE__</h1>
  <div class="sub">深度研究 · 思考链快照</div>
  <h4>核心要点</h4>
  <ul class="kp">__KEYPOINTS__</ul>
</aside>
<section class="right">
  <div class="hint">← 拖拽 / 方向键 →</div>
  <div class="stage"><div class="track" id="track">__CARDS__</div></div>
  <div class="nav">
    <button class="arrow" id="prev">‹</button>
    <div class="dots" id="dots">__DOTS__</div>
    <span class="pos" id="pos"></span>
    <button class="arrow" id="next">›</button>
  </div>
</section>
</div>
<script>
(function(){
  var N=__COUNT__, i=0, track=document.getElementById("track");
  var startX=0, dx=0, dragging=false, w=0;
  var kps=[].slice.call(document.querySelectorAll("ul.kp li"));
  var dots=[].slice.call(document.querySelectorAll(".dot"));
  function render(anim){
    track.classList.toggle("snap",!!anim);
    track.style.transform="translateX("+(-i*w+dx)+"px)";
    document.getElementById("pos").textContent=(i+1)+" / "+N;
    dots.forEach(function(d,k){d.classList.toggle("active",k===i)});
    kps.forEach(function(li){li.classList.toggle("active",li.getAttribute("data-go")==String(i))});
  }
  function go(n){i=Math.max(0,Math.min(N-1,n));dx=0;render(true)}
  function measure(){w=track.clientWidth;render(false)}
  window.addEventListener("resize",measure);
  document.getElementById("next").onclick=function(){go(i+1)};
  document.getElementById("prev").onclick=function(){go(i-1)};
  dots.forEach(function(d){d.onclick=function(){go(+d.getAttribute("data-i"))}});
  kps.forEach(function(li){li.onclick=function(){go(+li.getAttribute("data-go"))}});
  document.addEventListener("keydown",function(e){if(e.key==="ArrowRight")go(i+1);if(e.key==="ArrowLeft")go(i-1)});
  // 拖拽(鼠标+触摸)
  function down(x){dragging=true;startX=x;dx=0;track.classList.add("drag")}
  function move(x){if(!dragging)return;dx=x-startX;render(false)}
  function up(){if(!dragging)return;dragging=false;track.classList.remove("drag");
    var th=w*0.18; if(dx<-th)go(i+1); else if(dx>th)go(i-1); else {dx=0;render(true)} }
  track.addEventListener("mousedown",function(e){e.preventDefault();down(e.clientX)});
  window.addEventListener("mousemove",function(e){move(e.clientX)});
  window.addEventListener("mouseup",up);
  track.addEventListener("touchstart",function(e){down(e.touches[0].clientX)},{passive:true});
  track.addEventListener("touchmove",function(e){move(e.touches[0].clientX)},{passive:true});
  track.addEventListener("touchend",up);
  measure();
})();
</script>
</body></html>"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法: py -X utf8 render_html.py "<主题>_plan.json"'); sys.exit(1)
    build(sys.argv[1])
