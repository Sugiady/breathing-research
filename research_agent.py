# -*- coding: utf-8 -*-
"""
深度行研 Agent —— 对目标工具 harness 的 8 分像复刻。
流程: 规划(深度思考 + 7步计划) -> 按依赖序执行每步(可选语料/搜索) -> 汇总报告。

用法:
    py -X utf8 research_agent.py "AEM电解水制氢"
    py -X utf8 research_agent.py "主题" --corpus 语料.txt      # 给步骤1-6喂参考资料
    py -X utf8 research_agent.py "主题" --steps 7              # 自定义步骤数

产物(当前目录):
    <主题>_plan.json   规划结果(深度思考 + 每步规格)
    <主题>_report.md   最终研究报告
配置: deepseek_api.json (baseUrl/apiKey/model)
搜索: 见 search_bocha(),填入博查 key 即启用;默认关闭,新闻步走模型知识兜底。
"""
import json, sys, os, time, ssl, urllib.request, urllib.parse, http.cookiejar, html as _htmlesc, re, datetime

# 内容抓取用的宽松 SSL 上下文:中国大量内容站证书有毛病(主机名不符/链不全),默认校验会直接抛。
# 抓正文不是安全敏感场景,等价 curl -k / --ssl-no-revoke,只用于 _fetch_one。
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

TODAY = datetime.date.today().isoformat()
YEAR = datetime.date.today().year

HERE = os.path.dirname(os.path.abspath(__file__))

def _load_cfg():
    """DeepSeek 配置:优先本地 deepseek_api.json(本机开发),缺字段用环境变量兜底(服务器部署)。
    apiKey 允许为空——线上可让访客自带 key(前端字段按请求覆盖),服务器不必存密钥;
    请求真正调用时若 key 仍为空,call_llm 会带出 401,由前端提示用户填 key。"""
    cfg = {}
    fp = os.path.join(HERE, "deepseek_api.json")
    if os.path.isfile(fp):
        try:
            cfg = json.load(open(fp, encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg.setdefault("baseUrl", os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    cfg.setdefault("model", os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    if not cfg.get("apiKey"):        # 文件缺失或占位符为空 -> 环境变量兜底
        cfg["apiKey"] = os.environ.get("DEEPSEEK_API_KEY", "")
    return cfg

CFG = _load_cfg()

# 最近一次运行的状态缓存(按主题),供"重试某步"用;服务器重启后可从 _plan.json 重建
_RUNS = {}

# 并发护栏:主流程(run_stream)跑动中拒绝 refine/retry/revert,防"边跑边被插改状态"的竞态。
# 用递增序号而非布尔:打断重开时新 run 占用新序号,旧 run 收尾时按序号自检、不会误清新 run。
_ACTIVE = {}   # topic -> 当前活跃 run 序号(0/缺失=空闲)
_SEQ = [0]
def is_running(topic):
    return bool(_ACTIVE.get(topic))

# ---------------- LLM 调用 ----------------
PRO_MODEL = CFG["model"]

def _fast_model():
    """fast 模型优先读 deepseek_api_flash.json 的 model(与 pro 同 baseUrl/key);
    其次读 deepseek_api.json 里的 fastModel 字段;都没有就回退 pro,不破坏现状。"""
    fp = os.path.join(HERE, "deepseek_api_flash.json")
    if os.path.isfile(fp):
        try:
            m = json.load(open(fp, encoding="utf-8")).get("model")
            if m:
                return m
        except Exception:
            pass
    return os.environ.get("DEEPSEEK_FAST_MODEL") or CFG.get("fastModel") or CFG["model"]

FAST_MODEL = _fast_model()

def call_llm(messages, max_tokens=4000, temperature=0.4, retries=5, key=None, model=None, think=True):
    body = {"model": model or CFG["model"], "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    if not think:
        body["thinking"] = {"type": "disabled"}   # 关思考:执行步/机械精化提速,深浅靠 pro/flash 选模型扛
    req = urllib.request.Request(
        CFG["baseUrl"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + (key or CFG["apiKey"]), "Content-Type": "application/json"})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:   # 关思考后基本 20-30s 出,180s 足够;连接卡死也能较快重试
                d = json.load(r)
            m = d["choices"][0]["message"]
            return m.get("content") or "", m.get("reasoning_content") or "", d.get("usage", {})
        except Exception as e:
            last = e
            if attempt < retries:
                wait = min(2 ** attempt * 2, 8)   # 指数退避封顶 8s:2,4,8,8,8
                print(f"  [warn] LLM 调用失败(第{attempt+1}/{retries+1}次): {e}; {wait}s 后重试…")
                time.sleep(wait)
    raise RuntimeError(f"LLM 连续 {retries+1} 次失败: {last}")

def extract_json(text):
    """从模型输出里抠出第一个完整 JSON 对象(容忍 ```json 包裹和前后废话)。"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("输出里找不到 JSON:\n" + text[:500])
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 不完整(可能 max_tokens 不够):\n" + text[-500:])

# ---------------- 搜索(博查占位) ----------------
def _bocha_key():
    """从 bocha_api.json 读 key(字段名容忍 apiKey/api_key/key)。文件不存在或没 key -> None。"""
    path = os.path.join(HERE, "bocha_api.json")
    if os.path.isfile(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
            k = d.get("apiKey") or d.get("api_key") or d.get("key")
            if k:
                return k
        except Exception:
            pass
    return os.environ.get("BOCHA_API_KEY") or None      # 服务器部署:key 走环境变量

def _bocha_items(query, count=8, key=None):
    """博查 Web Search,返回结构化条目列表 [{name,url,date,site,summary}]。无 key/失败 -> []。"""
    key = key or _bocha_key()
    if not key:
        return []
    body = {"query": query, "summary": True, "count": count}
    last = None
    for attempt in range(3):          # 博查也会 IncompleteRead,重试 3 次
        req = urllib.request.Request(
            "https://api.bochaai.com/v1/web-search",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            raw = (((d.get("data") or {}).get("webPages") or {}).get("value")) or []
            return [{"name": it.get("name", ""), "url": it.get("url", ""),
                     "date": (it.get("datePublished") or it.get("dateLastCrawled") or "")[:10],
                     "site": it.get("siteName") or it.get("displayUrl") or "",
                     "summary": it.get("summary") or it.get("snippet") or ""} for it in raw]
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    print(f"  [warn] 搜索失败(3次): {last}")
    return []

# ---------------- 百度补充搜索 ----------------
# 博查索引偏窄(缺东方财富/每经/雪球/资讯流),百度资讯流补这块,且吃新鲜度、给直链。
# 关键:百度的验证码是"频控+缺预热cookie"触发,不是IP封死 —— 常驻一个先访首页预热过的
# cookie jar、复用、撞验证就重新预热重试,即可稳定通过(资讯流尤其稳)。海外IP下网页流偶发
# 仍被挡,捞不到就算,博查兜底。
_BD_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_BD_H = {"User-Agent": _BD_UA,
         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
         "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8", "Accept-Encoding": "identity",
         "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "none"}
_BD_JAR = http.cookiejar.CookieJar()
_BD_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_BD_JAR))
_BD_PRIMED = [False]

def _bd_get(url, timeout=15):
    req = urllib.request.Request(url, headers=dict(_BD_H, Referer="https://www.baidu.com/"))
    with _BD_OPENER.open(req, timeout=timeout) as r:
        return r.read(700000).decode("utf-8", "ignore")

def _bd_prime():
    try:
        _bd_get("https://www.baidu.com/"); _BD_PRIMED[0] = True; time.sleep(0.6)
    except Exception:
        pass

def _bd_capt(h):
    return (not h) or len(h) < 3000 or "百度安全验证" in h or "wappass.baidu.com" in h

def _bd_strip(s):
    return re.sub(r"\s+", " ", _htmlesc.unescape(re.sub(r"(?s)<[^>]+>", "", s))).strip()

def _bd_parse(h):
    """抽 (title,url,snippet)。结果块: <h3..><a href=...>标题</a>,摘要在其后的 abstract/content。"""
    out, seen = [], set()
    for m in re.finditer(r'<h3[^>]*>\s*<a[^>]*?href="(?P<u>[^"]+)"[^>]*>(?P<t>.*?)</a>', h, re.S):
        url = _htmlesc.unescape(m.group("u")); title = _bd_strip(m.group("t"))
        if not title or url in seen:
            continue
        seen.add(url)
        tail = h[m.end():m.end() + 2600]
        sn = (re.search(r'class="[^"]*(?:content-right|c-abstract|content)[^"]*"[^>]*>(.*?)</(?:span|div|p)>', tail, re.S)
              or re.search(r'<span[^>]*class="[^"]*"[^>]*>(.{40,}?)</span>', tail, re.S))
        out.append((title, url, _bd_strip(sn.group(1))[:180] if sn else ""))
    return out

def _baidu_items(query, count=6, news=True, retries=2):
    """百度搜一条 query,返回与博查同形状的条目。news=True 走资讯流(更稳/更新/多直链)。
    任何异常都吞掉返回 [],绝不影响主流程(博查兜底)。"""
    try:
        if not _BD_PRIMED[0]:
            _bd_prime()
        wd = urllib.parse.quote(query)
        url = ("https://www.baidu.com/s?rtt=1&tn=news&word=" + wd) if news else ("https://www.baidu.com/s?wd=" + wd)
        for attempt in range(retries + 1):
            try:
                h = _bd_get(url)
            except Exception:
                time.sleep(1.2 * (attempt + 1)); continue
            if _bd_capt(h):
                _BD_PRIMED[0] = False; _bd_prime(); time.sleep(1.0); continue
            return [{"name": t, "url": u, "date": "",
                     "site": re.sub(r"^https?://(www\.)?", "", u).split("/")[0], "summary": s}
                    for t, u, s in _bd_parse(h)[:count]]
    except Exception as e:
        print(f"  [warn] 百度补充搜索异常: {type(e).__name__}")
    return []

_URL_RE = re.compile(r"https?://[^\s)\]）】,，、；;\"']+")

# Jina Reader:r.jina.ai/<URL> 服务端渲染取正文,免费(无 key 有限速),顺带解 PDF/JS 渲染页/坏证书站。
# 定位=兜底而非首发:普通页 bounded curl 更快,只有 curl 拿不到(空壳/PDF/封站)时才请 Jina 救场,兼顾速度与覆盖。
USE_JINA = True
_JINA_BASE = "https://r.jina.ai/"

def _looks_pdf(u):
    ul = (u or "").lower()
    return ul.endswith(".pdf") or ".pdf?" in ul or "filetype=pdf" in ul

def _strip_html(raw, per):
    body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    body = re.sub(r"&[a-z#0-9]+;", " ", body)
    return re.sub(r"\s+", " ", body).strip()[:per]

def _fetch_curl(u, per=2500, timeout=12):
    """原生抓取:bounded urllib + 宽松 SSL,剥成纯文本。返回 (title, body, item);失败抛异常。"""
    req = urllib.request.Request(u, headers={
        "User-Agent": _BD_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        raw = r.read(200000).decode("utf-8", "ignore")
    title = (re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I) or [None, ""])[1].strip()[:120]
    body = _strip_html(raw, per)
    item = {"name": title or u, "url": u, "date": "",
            "site": re.sub(r"^https?://(www\.)?", "", u).split("/")[0], "summary": ""}
    return title, body, item

def _fetch_via_jina(u, per=2500, timeout=18):
    """Jina Reader 取正文(服务端渲染)。成功返回 (title, body, item);失败/太短返回 None。"""
    try:
        req = urllib.request.Request(_JINA_BASE + u, headers={
            "User-Agent": _BD_UA, "Accept": "text/plain", "X-Return-Format": "text"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(500000).decode("utf-8", "ignore")
    except Exception:
        return None
    if not raw or len(raw) < 200:
        return None
    title = ""
    m = re.search(r"^Title:\s*(.+)$", raw, re.M)     # Jina 头部有 Title:/URL Source:/Markdown Content:
    if m:
        title = m.group(1).strip()[:120]
    mc = raw.find("Markdown Content:")
    body = raw[mc + len("Markdown Content:"):] if mc >= 0 else raw
    body = re.sub(r"\s+", " ", body).strip()[:per]
    if len(body) < 200:
        return None
    item = {"name": title or u, "url": u, "date": "",
            "site": re.sub(r"^https?://(www\.)?", "", u).split("/")[0], "summary": ""}
    return title, body, item

def _fetch_one(u, per=2500, timeout=12):
    """抓单页正文,返回 (title, body, item)。策略:bounded curl 先行(快,治大多数页);
    拿不到/像空壳/是 PDF -> Jina Reader 兜底(慢但能啃 PDF/JS 渲染页/封站)。都失败抛异常由调用方跳过。"""
    if USE_JINA and _looks_pdf(u):                    # PDF:curl 抓到的是压缩流乱码,直接走 Jina
        j = _fetch_via_jina(u, per, timeout + 8)
        if j:
            return j
    title, body, item = "", "", None
    try:
        title, body, item = _fetch_curl(u, per, timeout)
    except Exception:
        item = None
    if USE_JINA and len(body) < 200:                  # curl 失败或只拿到空壳(JS渲染/封站)-> Jina 救场
        j = _fetch_via_jina(u, per, timeout + 8)
        if j:
            return j
    if item is None:
        raise RuntimeError(f"fetch failed: {u}")
    return title, body, item

def _fetch_urls(text, limit=3):
    """从文本里提取 URL 并直接抓取正文(博查只能搜不能抓,用户给的链接得真去读)。
    返回 (喂模型的证据文本, 来源条目列表)。每个 URL 失败不影响其余;失败的直接跳过,不在证据里留失败note
    (免得模型把"抓取失败/网页被限制"写进成品正文)。"""
    urls, seen = [], set()
    for u in _URL_RE.findall(text or ""):
        u = u.rstrip(".,。")
        if u not in seen:
            seen.add(u); urls.append(u)
    urls = urls[:limit]
    if not urls:
        return "", []
    blocks, items = [], []
    for u in urls:
        try:
            title, body, item = _fetch_one(u, per=2500, timeout=15)
            blocks.append(f"# 资料来源 {u}\n标题:{title}\n正文摘录:{body}")
            items.append(item)
        except Exception:
            continue   # 读不到就跳过,不在证据里留失败note——避免模型把"抓取失败/网页被限制"写进正文
    return "\n\n".join(blocks), items

def _fetch_url_list(urls, limit=3, per=2600, timeout=12):
    """执行层用:抓命中页正文(产能/份额这类表格明细常只住在全文里,snippet 取不到)。
    与 _fetch_urls 的区别——这是自动增强抓取,失败/空壳(JS渲染/验证页)一律静默跳过,
    不把失败note塞进证据,免得喧宾夺主。返回喂模型的证据文本。"""
    blocks, seen = [], set()
    for u in (urls or []):
        if len(blocks) >= limit:
            break
        u = (u or "").rstrip(".,。")
        if not u or u in seen:
            continue
        seen.add(u)
        try:
            title, body, _ = _fetch_one(u, per=per, timeout=timeout)
        except Exception:
            continue
        if len(body) < 200:        # 空壳/JS渲染/验证页:无有效正文,跳过
            continue
        blocks.append(f"# 资料来源 {u}\n标题:{title}\n正文摘录:{body}")
    return "\n\n".join(blocks)

# ---------------- 深度核实:可信度分流 + 线索抽实体 + 并行免费追搜 ----------------
# 泛化自 Suj 的人类检索快照:一条命中有两种角色——
#   定位型(可信源:财经披露/公告/年报/IR/政府)→ 数直接在里面,采信;
#   线索型(次可信:媒体排名表/整理稿/文库/自媒体)→ 不信其数,只抽候选实体名,拿去逐个再查。
# 追搜走"免费"通道(百度资讯 + fetch,不花钱、不占博查预算),所以按主体逐个查、best-effort。
_ANCHOR_DOMAINS = ("sina.com.cn", "eastmoney.com", "10jqka.com.cn", "cninfo.com.cn",
                   "sse.com.cn", "szse.cn", "stcn.com", "cs.com.cn", "nbd.com.cn",
                   "yicai.com", "cnstock.com", "gov.cn")     # 财经披露/权威财媒/政府
_LEAD_DOMAINS = ("baijiahao.baidu.com", "toutiao.com", "sohu.com", "zhihu.com",
                 "163.com", "docin.com", "doc88.com", "1688.com", "hbzhan.com",
                 "lsjxww.com", "paihang", "paiming", "rank")  # 自媒体/文库/B2B/排名站

def _source_tier(url):
    """定位型 anchor(可信、采数) / 线索型 lead(次可信、只抽实体)。未知域名保守按 lead。"""
    u = (url or "").lower()
    if any(d in u for d in _ANCHOR_DOMAINS):
        return "anchor"
    return "lead"

def _bd_cookie_header():
    """预热(一次)后导出 BAIDUID 等 cookie 字符串,供并行线程共享——暖 cookie 才不吃验证码。"""
    if not _BD_PRIMED[0]:
        _bd_prime()
    return "; ".join(f"{c.name}={c.value}" for c in _BD_JAR)

def _baidu_query_isolated(query, cookie="", count=6):
    """单条百度资讯查询,线程安全:自建 jar opener。先用共享暖 cookie;撞验证就自己冷预热重试。"""
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    def g(url, ck=""):
        h = dict(_BD_H, Referer="https://www.baidu.com/")
        if ck:
            h["Cookie"] = ck
        with op.open(urllib.request.Request(url, headers=h), timeout=15) as r:
            return r.read(700000).decode("utf-8", "ignore")
    url = "https://www.baidu.com/s?rtt=1&tn=news&word=" + urllib.parse.quote(query)
    use_shared = bool(cookie)
    for attempt in range(3):
        try:
            h = g(url, cookie if use_shared else "")     # use_shared 关闭后靠 jar 自带 cookie
        except Exception:
            time.sleep(0.6); continue
        if _bd_capt(h):
            try:
                g("https://www.baidu.com/"); time.sleep(0.6)   # 自己冷预热到本线程 jar
            except Exception:
                pass
            use_shared = False
            continue
        return [{"name": t, "url": u, "date": "",
                 "site": re.sub(r"^https?://(www\.)?", "", u).split("/")[0], "summary": s}
                for t, u, s in _bd_parse(h)[:count]]
    return []

def _baidu_parallel(queries, max_workers=2):
    """并行跑多条百度查询(免费),返回 [(query, items)]。共享暖 cookie;best-effort,失败给空。"""
    from concurrent.futures import ThreadPoolExecutor
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return []
    cookie = _bd_cookie_header()          # 预热一次,所有线程共享,避免各自冷启动撞验证
    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [(q, ex.submit(_baidu_query_isolated, q, cookie)) for q in queries]
        for q, f in futs:
            try:
                out.append((q, f.result(timeout=45)))
            except Exception:
                out.append((q, []))
    return out

_ENT_KW = ("名单", "企业", "公司", "厂商", "玩家", "格局", "产能", "份额", "排名",
           "主要", "供应商", "竞争", "标的", "参与者")
def _wants_entity_verify(step):
    """该步是否是'逐个实体核实'型(名单/产能/份额/竞争格局)。只对这类才启深度核实,省时。"""
    blob = (step.get("name", "") + step.get("type", "") + step.get("instruction", ""))
    return sum(1 for k in _ENT_KW if k in blob) >= 1

def _extract_entities(corpus_text, topic, step, llm_key=None):
    """从种子命中(标题+摘要)里抽候选'企业'名 + 一个简短追查焦点词(用便宜的 flash)。
    返回 {'entities':[...], 'focus':'...'}。绝不臆造未出现的实体。"""
    sysp = ("你在给一次事实核实做准备。下面是若干检索命中的标题与摘要。"
            "任务:(1)抽出其中真正作为核实对象的**企业/公司**名(实际从事该产品生产或销售的商业主体)——"
            "只列明确出现、与主题相关的,去重,最多8个;"
            "**务必排除**:大学/科研院所/'国家科技成果网'这类机构、期刊、'河北XX厂家'这种黄页分类词、纯产品名;"
            "以及主业与本主题产品无关的公司(如饲料/养殖/综合化工企业,除非它明确就是该产品的生产商)。"
            "(2)给一个**简短**(不超过2个词)的追查焦点词,拼在公司名后能直接搜到数据的那种,如'DHA 产能'、'市占率'、'融资'。"
            "严禁臆造未出现的企业。只输出JSON:{\"entities\":[\"...\"],\"focus\":\"...\"}")
    user = f"主题:{topic}\n核实目标:{step.get('instruction','')}\n\n[检索命中]\n{corpus_text[:3500]}"
    try:
        content, _, _ = call_llm([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                 max_tokens=500, temperature=0.3, key=llm_key, model=FAST_MODEL, think=False)
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            obj = json.loads(m.group(0))
            ents = [e.strip() for e in obj.get("entities", []) if isinstance(e, str) and e.strip()]
            focus = " ".join((obj.get("focus") or "").split()[:2])   # 焦点词砍到≤2词,避免查询过长
            return {"entities": ents[:8], "focus": focus}
    except Exception:
        pass
    return {"entities": [], "focus": ""}

def deep_verify(sources, topic, step, llm_key=None, notify=None):
    """种子命中 → 分流 → 线索页抽实体 → 并行免费百度逐实体追搜 → 定位型再抓全文。
    返回 (追加证据文本, 追加sources)。全程免费(不占博查预算)。"""
    # 抽实体用全部种子命中(公司名常在 anchor 里,别挑食);信任分流只用于'信谁的数'。
    # 公司名多在标题里 -> 所有标题先行(短、都放得下)、anchor 优先,摘要垫后,避免被截断漏名。
    ordered = sorted(sources, key=lambda s: 0 if _source_tier(s["url"]) == "anchor" else 1)
    titles = "\n".join("· " + s["name"] for s in ordered)
    summ = "\n".join(f"{s['name']}:{(s.get('summary','') or '')[:120]}" for s in ordered[:10])
    corpus = "[全部命中标题]\n" + titles + "\n\n[部分摘要]\n" + summ
    if not corpus.strip():
        return "", []
    ext = _extract_entities(corpus, topic, step, llm_key)
    ents, focus = ext["entities"], ext["focus"]
    if not ents:
        return "", []
    if notify:
        notify(f"从线索页抽出 {len(ents)} 个待核实体,免费并行追查…")
    qmap = _baidu_parallel([f"{e} {focus}".strip() for e in ents])
    add_sources, seen, blocks = [], {s["url"] for s in sources}, []
    for q, items in qmap:
        for it in items:
            if it["url"] and it["url"] not in seen:
                seen.add(it["url"]); add_sources.append(it)
        if items:
            blocks.append(f"# 关于「{q}」的补充资料\n" +
                          "\n".join(f"- [{x['site']}] {x['name']}:{x['summary']}" for x in items[:4]))
    anchor_urls = [s["url"] for s in add_sources if _source_tier(s["url"]) == "anchor"]
    anchor_ev = _fetch_url_list(anchor_urls, limit=6)     # 追查命中里的定位型再抓全文(免费)
    ev = "\n\n".join(x for x in ["\n".join(blocks), anchor_ev] if x.strip())
    return ev, add_sources

def gather_search(queries, key=None, cap=3, baidu=True):
    """跑多条查询,返回 (喂给模型的文本, 去重后的来源条目列表)。cap 限制本次最多用几条查询。
    博查起手第一发;baidu=True 时再用百度资讯流补一发,合并去重(补博查缺的财经/资讯/投融资)。"""
    hits, items, seen = None, [], set()
    for q in (queries or [])[:cap]:
        its = _bocha_items(q, key=key)
        if baidu:                              # 百度资讯流补充:博查没有的直链条目并进来
            burls = {x["url"] for x in its}
            for bx in _baidu_items(q):
                if bx["url"] and bx["url"] not in burls:
                    its.append(bx); burls.add(bx["url"])
        if not its:
            continue
        block = "\n".join(f"- [{x['date']} | {x['site']}] {x['name']}\n  {x['summary']}\n  来源: {x['url']}"
                          for x in its)
        hits = (hits or "") + f"\n# 检索: {q}\n" + block
        for x in its:
            if x["url"] and x["url"] not in seen:
                seen.add(x["url"]); items.append(x)
    return hits, items

# ---------------- 预检索:规划前先摸一眼"有哪些信息" ----------------
SEARCH_BUDGET = 9        # 一次研究总检索预算(预检索 + 执行)
PRE_SEARCH_N = 2         # 预检索至少几条
FETCH_TOP_N = 3          # 每个数据步:对命中页前几条 URL 抓全文(snippet 取不到表格/明细)
DEEP_VERIFY = False      # 逐实体深度核实(种子→抽实体→并行免费追搜)。惊艳但慢(~140s/步,受API延迟支配)且不稳,默认关;想深挖时开

def _landscape_queries(topic, detail="", llm_key=None):
    """生成 2-3 条宽口径检索词,用于规划前的情报预览。失败则回退到启发式。"""
    sysp = ("你在为一次行业研究做'检索前的情报预览'。给出 2-3 条最能快速摸清该主题信息全貌的"
            "宽口径检索词(中英文均可),覆盖 现状/市场规模/主要玩家/权威数据源 等方向。只输出 JSON 字符串数组,不要解释。")
    user = f"主题:{topic}" + (f"\n补充说明:{detail}" if detail and detail.strip() else "")
    try:
        content, _, _ = call_llm([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                 max_tokens=600, temperature=0.5, key=llm_key, model=FAST_MODEL, think=False)
        m = re.search(r"\[.*\]", content, re.S)
        if m:
            qs = [q for q in json.loads(m.group(0)) if isinstance(q, str) and q.strip()]
            if qs:
                return qs[:3]
    except Exception:
        pass
    return [topic, f"{topic} 市场 现状 主要企业 数据"]

def landscape_probe(topic, detail="", search_key=None, llm_key=None, n=PRE_SEARCH_N):
    """规划前预检索:返回 (情报摘要文本, 来源条目, 用掉的检索次数)。无搜索 key 时返回空。"""
    qs = _landscape_queries(topic, detail, llm_key)
    qs = qs[:max(n, PRE_SEARCH_N)]
    hits, items = gather_search(qs, key=search_key, cap=len(qs))
    return (hits or ""), items, (len(qs) if hits else 0)

# ---------------- 阶段1: 多轮迭代规划(每轮 = 该轮prompt + 上一轮plan) ----------------
# 设计依据:7步框架不该写死——技术/市场/政策类主题需要不同的步骤组合。
# Iter1 负责"选对研究框架"(泛化),Iter2~4 层层加精度。

ITER_LABEL = {1: "框架与骨架", 2: "类型与指令", 3: "依赖关系", 4: "质量判据与检索"}

ITER_SYS = {
1: """你是行研规划器的第1轮:框架判定与骨架。
先判断该研究主题的性质属于哪一类(如 技术/产业、市场分析、政策演变、公司/标的、材料科学 等),
再据此决定最契合的研究步骤组合——步骤的数量与命名必须贴合主题性质,不要套用任何固定模板。
参考(仅示例,不要照抄):技术类≈技术总览/演进时间线/产业链/路线对比/瓶颈/场景/近期进展;
市场类≈市场定义与规模/供需结构/价格周期/竞争格局/驱动与风险/趋势研判;
政策类≈政策脉络/关键节点/利益相关方/影响传导路径/趋势研判。
拆解时心里装着五个底层视角,但它们是"看问题的角度",不是"步骤本身":①第一性原理(回到最本质的需求/驱动/机理);②价值链(上下游谁要什么、附加值在哪);③历史演化(从哪来、什么在变);④比较(与替代/同类在统一维度的相对位置);⑤风险(什么可能颠覆主流判断)。
关键:步骤骨架要顺着主题自身的自然结构走——概览/综述类主题按其内在类别拆(如手段的物理/化学/机械、产业的细分环节、品类的子类型;若用户在补充说明里点了维度,优先依其拆),把上述视角融进各步的分析里,而不要把"第一性原理""产业链""历史演化"这些视角名直接拿来当步骤标题。只有当某个视角确实是该主题的核心矛盾时,才让它单独成步。既要避免平铺罗列同质条目,也要避免为凑视角硬造抽象步骤。
步骤数通常 7-8 个;只有主题确实复杂、维度多时才超过 8,务必避免拆得过碎或步骤间重复。
同时产出深度思考三段:context_analysis(主题性质与研究级别)、research_plan(总体思路)、scope_decomposition(为何这样拆、如何不重叠)。深度思考请充分展开、把判断的理由说透,不要惜字。
只输出 JSON:{"research_class":"...","deep_thinking":{"context_analysis":"...","research_plan":"...","scope_decomposition":"..."},"steps":[{"id":1,"name":"..."}]}""",
2: """你是行研规划器的第2轮:类型与指令。在[上一轮规划]的骨架基础上,为每个步骤补充:
- type(类型标签,如 归纳/时序/产业链/对比/瓶颈分析/场景分析/新闻整理 或更贴合该主题的标签)
- instruction(该步的研究范围与口径,明确做什么、不做什么,并写清"该追问到什么深度")
写 instruction 时,从这些分析动作里按该步性质与主题类型取用、点明深度:本质/第一性(驱动是否真实)、价值链定位、历史演化、结构性多力(竞争/供给/需求/替代/新进入者/互补/政策宏观及其近期动态)、量化对账(仅在确有可靠数据基础、且量级对结论重要时,才提自上而下+自下而上对账;否则定性或量级即可,不要无据硬凑)、比较(统一维度下与替代/同类对照、指出本质差异而非表面优劣;缺数据的维度可定性或留白)、致命风险。
深度要匹配研究层级与主题宽度:概览/综述类或宽口径主题,讲清机理脉络、量级与关键判据即可,不要钻进学术级微观参数(如分子界面能垒、DLVO 合力曲线、单一材料的精确实验常数)或逼出未必公开的精确数字——代理指标、定性区间、量级范围都算合格;只有当某个精确量真正决定结论时才深挖。
对"公司/标的"类主题,这些对应尽调维度(需求真实性、解决方案、产业链站位、市场容量上下对账、竞争本质差异、商业模式、致命风险等);对"产业/政策/技术"类,等价地落到产业组织(集中度/壁垒/扩产/替代)、利益相关方得失与传导路径、技术路线的本质约束与演进上。
保持每个步骤的 id 与 name 不变,不要增删步骤。指令尽量具体、把研究意图说透,不要写空泛套话。只输出完整 JSON(含 research_class、deep_thinking、steps[id,name,type,instruction])。""",
3: """你是行研规划器的第3轮:依赖关系。此时已有较具体的步骤与指令,可结合全局把衔接关系做实。为每个步骤补充 depends_on(依赖的前序步骤 id 列表,可为空数组),
依赖只能指向 id 更小的步骤;并结合上下文把每条 instruction 改写得更具体——点名该步要对照的对象、要量化的指标、要追的本质问题,去掉空泛套话。具体化是为了去空泛,不是越细越好:宽口径/综述类主题不要强行要求学术级精度或单一精确数字,保持在能服务研究结论的颗粒度即可。保持 id/name/type 不变。
只输出完整 JSON(steps 每项含 id,name,type,instruction,depends_on)。""",
4: """你是行研规划器的第4轮:质量判据与检索准备。在[上一轮规划]基础上,为每个步骤补充:
- quality_rubric(优秀输出应做到什么、一般输出常见的不足)。务必写明:如实标注数据边界(如"公开数据有限""以最近可得年份为准")属于高质量;含糊掩盖或编造数字属于低质量;能区分事实/推断/观点、关键数字有可靠来源、判断立足所列事实,属于高质量。尤其点明:只写有据的、略过没覆盖的(不为完整搭空框架),矩阵/对比中缺数据的格子留空或定性、不强凑数字——这属于高质量(诚实);反之堆砌无来源的精确数字、或为求对称而填满杜撰的表格,属于低质量。
- need_search(布尔)。以下两类都应为 true:① 近期新闻/进展/事件类步骤;② 任何依赖具体、时效性数字的步骤(市场规模、产量、产能、销量、份额、价格、增速、企业数量等)。纯概念/原理/框架类步骤为 false。
- search_queries(若 need_search 为 true,给出中英文检索查询词数组)。检索词遵循"按问题倒推信源类型":优先指向强约束信源(监管/交易所披露、年报季报、统计公报与年鉴、行业协会、龙头公司 IR),而不是泛搜"xx market size";数据型步骤点名 具体指标+地区+最近年份(如"浙江 纺织 产业集群 产值 2024");新闻型含关键主体/产品/事件类型;
  并按"事实种类→装它的文档体裁"配后缀,直接钓一手原始件而非泛搜:要产能/在建/工艺路线→加"环评"(环评报告含产能明细);要募投/历史产能/财务→加"招股书""年报";要扩产/重大事项→加"公告";要份额/客户→加"调研纪要""投资者关系"。只在该步要的正是这种数时才加,不要给每条查询都套体裁后缀;前沿/新兴技术或竞争格局类步骤,优先检索投融资与并购新闻(融资轮次、领投方、估值、公司介绍)——这是比咨询市场报告更硬的玩家与景气信号;允许给代理指标/邻近口径的备用词,不要钻牛角尖于单一数字。否则空数组。
  检索预算有限:整个研究执行阶段总检索约 5-7 次(规划前已另做 2 次预检索)。请把检索集中到真正依赖外部数据的步骤,每个需检索步给 1-2 条最高命中率的查询;概念/原理/框架类步骤一律空数组。结合上面的[检索情报预览]判断哪些数据查得到——对预览里明显查不到的,别硬设检索。
- deep(布尔)。该步若以"原创推理/第一性推演/多因素权衡/比较研判/致命风险与前瞻综合判断"为主、需要深度思考才能写好,标 true;若主要是梳理检索结果、罗列归纳、整理时间线/名单/事件等信息整合类工作,标 false。务必克制:多数步骤应为 false,通常只有第一性/本质分析、关键技术或路线对比研判、致命风险与前瞻这类真正吃推理的步骤才标 true(一般不超过 2-3 个)。
保持前几轮已定字段不变。只输出完整 JSON(steps 每项含 id,name,type,instruction,depends_on,quality_rubric,need_search,search_queries,deep)。""",
}

def _lang_directive(lang, mode="exec"):
    """内容语言指令,压在系统提示末尾(模型更听结尾)。mode: iter=JSON值用英文 / exec=整篇英文并覆盖'结论：'格式。
    检索词与专有名词保持能命中源站的形态。lang 非 en -> 返回空(默认中文,零改动)。"""
    if not (lang or "zh").lower().startswith("en"):
        return ""
    if mode == "iter":
        return ("\n\n[OUTPUT LANGUAGE — HIGHEST PRIORITY] Write every JSON string VALUE (research_class, all deep_thinking text, "
                "and each step's name/type/instruction/quality_rubric) in fluent, professional English. Keep JSON keys unchanged. "
                "EXCEPTION: keep `search_queries` in the language that best matches the sources (Chinese for a China-focused topic), "
                "and keep company/product proper nouns in their searchable original form.")
    return ("\n\n[OUTPUT LANGUAGE — HIGHEST PRIORITY, overrides any Chinese formatting instruction above] "
            "Write the ENTIRE answer in fluent, professional English. The first line MUST start with 'Conclusion:' (NOT '结论：'), "
            "then a blank line, then the detailed body in English Markdown. Keep cited source names and URLs as they appear.")

def _iter_messages(k, topic, prev, detail="", probe="", lang="zh"):
    datehint = (f"\n\n(当前日期 {TODAY}。涉及数据、竞争格局、玩家名单等时效内容时,"
                f"检索词与判断须覆盖 {YEAR} 年及最近 1-2 年的最新进展,不要停留在更早的年份。)")
    detailhint = (f"\n\n[用户补充的研究侧重/背景](请在规划时充分尊重并据此取舍范围与深度):\n{detail.strip()}"
                  if detail and detail.strip() else "")
    # 预检索情报只喂给 Iter1(定框架)和 Iter4(设检索词):据"哪些查得到/查不到"来设步骤与维度
    probehint = (f"\n\n[检索情报预览](规划前的宽口径预搜,反映该主题大致有哪些公开信源/数据;"
                 f"据此把步骤与量化维度收敛到'答得出'的范围,对预览里明显查不到的东西别设成需要精确数字的步骤):\n"
                 + probe.strip()[:2500]) if (probe and probe.strip() and k in (1, 4)) else ""
    if k == 1:
        user = f"研究主题:{topic}" + datehint + detailhint + probehint
    else:
        user = (f"研究主题:{topic}{datehint}{detailhint}{probehint}\n\n[上一轮规划](在此基础上精化,不要推翻已定的步骤):\n"
                + json.dumps(prev, ensure_ascii=False))
    return [{"role": "system", "content": ITER_SYS[k] + _lang_directive(lang, "iter")},
            {"role": "user", "content": user}]

def _merge_steps(prev_steps, cur_steps):
    """步骤集合锁定在 Iter1:后续轮次只能给已有步骤补字段,不能增删步骤。
    以 prev 的 id/顺序为准,把 cur 同 id 的新字段覆盖上去;cur 丢了步骤也不会少。"""
    if not prev_steps:
        return cur_steps or []
    by_id = {s.get("id"): s for s in (cur_steps or []) if isinstance(s, dict)}
    merged = []
    for ps in prev_steps:
        m = dict(ps)
        cs = by_id.get(ps.get("id"))
        if cs:
            for kk, vv in cs.items():
                if vv not in (None, "", []):      # 只用非空新值覆盖,别用空值抹掉已有
                    m[kk] = vv
        merged.append(m)
    return merged

def _sanitize_deps(steps):
    """把依赖图焊成严格 DAG:每步 depends_on 只保留"存在且 id 严格更小"的项,
    剔除自依赖、前向依赖(指向更大 id)、指向不存在步骤的悬空依赖,并去重保序。
    原则"依赖只指向更小 id"本是 Iter3 的 prompt 约定;这里把它从约定升为代码保证,
    杜绝模型偶尔越界导致的隐性上下文缺口(执行仍按 id 升序,更小 id 的上游必已算完)。
    就地修改并返回 steps。"""
    ids = {s.get("id") for s in (steps or []) if isinstance(s, dict)}
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        sid, kept, seen = s.get("id"), [], set()
        for d in (s.get("depends_on") or []):
            if d in ids and isinstance(sid, int) and isinstance(d, int) and d < sid and d not in seen:
                seen.add(d); kept.append(d)
        s["depends_on"] = kept
    return steps

def plan_iters(topic, llm_key=None, stream_iter1=True, detail="", probe="", lang="zh"):
    """生成器:逐轮规划。Iter1 可流式吐 reasoning(供"呼吸"),Iter2~4 直接返回。
    yield {type:thinking_delta|plan_iter|plan_final}。
    复杂主题(步骤多/字段长)整页 JSON 可能超 max_tokens 被截断 -> 自动升档重试。"""
    prev, dt = None, {}
    for k in (1, 2, 3, 4):
        msgs = _iter_messages(k, topic, prev, detail, probe, lang)
        cur = None
        for mt in (12000, 18000):          # token 升档:截断了就给更大上限再来一次
            if k == 1 and stream_iter1 and mt == 12000:
                content = ""
                for ev in call_llm_stream(msgs, max_tokens=mt, temperature=0.45, key=llm_key):
                    if ev[0] == "delta" and ev[1] == "reasoning":
                        yield {"type": "thinking_delta", "text": ev[2]}
                    elif ev[0] == "final":
                        content = ev[1]
            else:
                # 关思考后 flash/pro 用时基本无差、价格相近 -> 统一用 pro(质量更好)。仅 Iter1 留思考(选框架);2/3/4 关。
                content, _, _ = call_llm(msgs, max_tokens=mt, temperature=0.4, key=llm_key,
                                         model=PRO_MODEL, think=(k == 1))
            try:
                cur = extract_json(content)
                if k == 1 and not (isinstance(cur.get("steps"), list) and cur["steps"]):
                    raise ValueError("Iter1 未给出 steps 列表")   # 当截断处理,升档重试
                break
            except ValueError as e:
                print(f"  [warn] Iter{k} 规划解析问题({e}),max_tokens→{mt} 重试…")
                cur = None
        if cur is None:
            if k == 1:
                raise ValueError("Iter1 规划多次未产出有效 steps——主题可能过于复杂或模型输出异常")
            cur = dict(prev)        # 后续轮失败:退回上一轮规划,不让整轮崩掉
        if k == 1:
            dt = cur.get("deep_thinking", {})
        else:
            cur["steps"] = _merge_steps(prev.get("steps", []), cur.get("steps", []))  # 锁定步骤集合
        cur.setdefault("deep_thinking", dt)            # 把 Iter1 的深度思考贯穿到底
        if not cur.get("steps"):                        # 终极兜底:任何时候都保证有步骤
            cur["steps"] = prev.get("steps", []) if prev else []
        _sanitize_deps(cur["steps"])                    # 焊成严格 DAG:剔除自/前向/悬空依赖
        prev = cur
        yield {"type": "plan_iter", "k": k, "label": ITER_LABEL[k], "plan": cur}
    yield {"type": "plan_final", "plan": prev}

def plan(topic, n_steps=7, llm_key=None):
    """离线版:跑完4轮迭代,返回 (最终plan, 迭代记录列表)(非流式)。"""
    prev, iterations = None, []
    for ev in plan_iters(topic, llm_key=llm_key, stream_iter1=False):
        if ev["type"] == "plan_iter":
            prev = ev["plan"]
            iterations.append({"k": ev["k"], "label": ev["label"], "steps": ev["plan"]["steps"]})
            print(f"  [规划 Iter{ev['k']} {ev['label']}] 步骤数={len(prev.get('steps',[]))}")
        elif ev["type"] == "plan_final":
            prev = ev["plan"]
    return prev, iterations

# ---------------- 阶段2: 单步执行(流式 / CLI / 重试共用 _exec_one) ----------------
MIN_SOURCES = 3          # 检索来源少于此 -> 触发换词补搜
GAP_MAX = 3              # 输出里"查不到"类标记多于此 -> 触发换词补搜
_GAP_PATS = ["未找到", "检索未覆盖", "公开数据有限", "无法确认", "需核实", "未能查到",
             "缺乏公开", "未披露", "暂未", "未检索到"]

def _gap_count(text):
    return sum(text.count(p) for p in _GAP_PATS)

def _cited_sources(sources, text):
    """只保留正文实际引用到的来源(站点名/标题/url 在正文里出现过)。
    全没匹配上时保守返回原列表,绝不把有用来源连带删光。"""
    if not sources or not text:
        return sources
    kept = []
    for s in sources:
        site, name, url = s.get("site", ""), s.get("name", ""), s.get("url", "")
        if (site and site in text) or (url and url in text) or (name and len(name) >= 4 and name in text):
            kept.append(s)
    return kept or sources

_DEEP_PATS = ["第一性", "本质", "致命风险", "前瞻", "研判", "瓶颈"]
def _is_deep(step):
    """该步是否吃深推 -> 用 pro+thinking;否则 flash+thinking。
    优先用规划器(Iter4)打的 deep 标;缺省时按 type/name 关键词兜底。"""
    d = step.get("deep")
    if isinstance(d, bool):
        return d
    blob = (step.get("type", "") + step.get("name", ""))
    return any(p in blob for p in _DEEP_PATS)

def _alt_queries(topic, step, prior, llm_key=None):
    """上一轮检索覆盖不足时,让模型只在'搜什么'层面提替代检索词(绝不碰结论)。返回字符串数组。"""
    sysp = ("你在为一个检索步骤补充【替代检索词】:上一轮检索覆盖不足。基于主题与该步骤,"
            "提出一组与上一轮不同的中英文检索词——可用代理指标、邻近/更新年份、上位或相邻品类、"
            "具体企业/产品/项目名、英文术语等。只输出 JSON 字符串数组,不要解释。")
    user = (f"当前日期 {TODAY}。主题:{topic}\n步骤:{step.get('name')} — {step.get('instruction','')}\n"
            f"已用过(避免重复):{prior}")
    try:
        content, _, _ = call_llm([{"role": "system", "content": sysp},
                                  {"role": "user", "content": user}],
                                 max_tokens=1200, temperature=0.5, key=llm_key)
        m = re.search(r"\[.*\]", content, re.S)
        if m:
            return [q for q in json.loads(m.group(0)) if isinstance(q, str)][:4]
    except Exception:
        pass
    return []

def _feedback_queries(topic, step, feedback, llm_key=None):
    """据用户对该步的反馈,提取最该补搜的 1-3 条检索词(围绕用户点名的公司/链接/遗漏点)。"""
    sysp = ("用户对某研究步骤的产出提了反馈,指出遗漏、或点名了该补查的公司/链接/主体。"
            "据此生成 1-3 条最针对性的中英文检索词:若反馈含公司名/产品/链接,围绕它;若指出某遗漏维度,直指该维度。"
            "只输出 JSON 字符串数组,不要解释。")
    user = (f"主题:{topic}\n步骤:{step.get('name')} — {step.get('instruction','')}\n"
            f"用户反馈:{feedback.strip()}")
    qs = []
    try:
        content, _, _ = call_llm([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                 max_tokens=500, temperature=0.4, key=llm_key, model=FAST_MODEL, think=False)
        m = re.search(r"\[.*\]", content, re.S)
        if m:
            qs = [q for q in json.loads(m.group(0)) if isinstance(q, str) and q.strip()][:3]
    except Exception:
        pass
    return qs

def _search_block(hits):
    return ("[参考资料](仅可据此作答)\n" + hits[:9000]
            + "\n\n严格要求:只整理上述资料中真实出现的事件,每条尽量标注[日期|来源];"
              "不得补充资料里没有的事件、数字或主体;无法确认的写'公开渠道未见'。"
              "注意成品口吻:不要写'本次检索''根据检索结果'这类过程词,直接陈述结论与依据。")

def _exec_one(step, done, search_key=None, llm_key=None, corpus=None, topic="", notify=None,
              detail="", budget=None, feedback="", prior="", deep=None, lang="zh"):
    """执行一个步骤,返回 (结果dict, 详细HTML)。
    安全重搜:检索覆盖不足(来源少/缺口多)时,代码侧换检索词再搜一轮、补充真实证据后重写——
    全程不奖励"减少缺口",只给更多真实证据;判定与重试都在代码侧,不问模型缺口多不多。
    budget={'left':N} 时受全局检索预算约束(预算用尽则该步走知识兜底);None=不限(单步重试/反馈返工用)。
    feedback 非空 = 带用户反馈返工:把上一版产出 prior + 用户批评喂进去,并据反馈定向补搜。"""
    import render_html
    base = [f"当前日期:{TODAY}。涉及最新进展/竞争格局/数据时,以最近可得信息为准,优先 {YEAR} 年及近一年,不要停留在更早年份。",
            f"步骤名:{step['name']}", f"类型:{step.get('type','')}",
            f"研究范围(instruction):{step.get('instruction','')}",
            f"质量标准(rubric):{step.get('quality_rubric','')}"]
    if detail and detail.strip():
        base.append("[用户补充的研究侧重/背景](据此聚焦,不要跑题):\n" + detail.strip())
    dep_txt = "\n".join(f"【步骤{i} {done[i]['name']} 结论】{done[i]['main_conclusion']}"
                        for i in (step.get("depends_on") or []) if i in done)
    if dep_txt:
        base.append("[依赖结论]\n" + dep_txt)

    fb_queries, url_sources, prior_sources = [], [], []
    if feedback and feedback.strip():
        base.append("[上一版产出 —— 仅供参考,你要交付的是它的完整替换版](下面是这一页的旧版本。"
                    "你的输出会整页替换掉它,读者只看得到你的新版本、看不到这份旧的。"
                    "因此务必把这一页写成完整、自包含的版本:旧版里对的内容要原样保留并完整重述,"
                    "针对反馈修订该改的地方;严禁以'前面已讲过/上一版提过/此处省略'为由略写任何内容——"
                    "凡省略,读者就彻底看不到了。)\n"
                    + (prior or "(无)")[:3000])
        base.append("[修订要点 —— 必须逐条处理,但产出是成品报告、不是给用户的回信](下面是需要修订/补充的点;"
                    "指到链接或公司就围绕它补检索与核实,无法证实的如实处理、绝不编造。"
                    "关键:把该补的内容自然融进报告正文,**不要出现'用户反馈''用户要求''您提到的链接'这类字样,也不要专门开一节回应这些要点或说明某链接无法核实**;"
                    "补不到的那条,就当它在公开渠道没有相应披露来中性处理或略过,不要交代它来自反馈或链接读取失败)\n"
                    + feedback.strip())
        url_ev, url_sources = _fetch_urls(feedback)      # 用户在反馈里贴的链接:直接抓正文(博查只能搜不能抓)
        if url_ev:
            base.append("[补充资料正文 —— 据此作答与核实,并登记为来源;正文里正常引用即可,不要提'用户提供的链接'之类字样]\n" + url_ev[:6000])
        prior_sources = (done.get(step["id"]) or {}).get("sources", [])
        fb_queries = _feedback_queries(topic or step["name"], step, feedback, llm_key)

    def write(extra):   # 统一 pro + 关思考(关思考后与 flash 用时无差、质量更好)
        full, _, _ = call_llm([{"role": "system", "content": EXEC_SYS_TEXT + _lang_directive(lang, "exec")},
                               {"role": "user", "content": "\n\n".join(base + extra)}],
                              max_tokens=4000, temperature=0.4, key=llm_key, model=PRO_MODEL, think=False)
        return full

    def take(n):    # 从全局预算扣 n 次检索额度,返回实际可用次数;budget=None 不限
        if budget is None:
            return n
        avail = max(0, min(n, budget.get("left", 0)))
        budget["left"] = budget.get("left", 0) - avail
        return avail

    sources = []
    if step.get("need_search") or fb_queries:
        # 反馈驱动的检索词优先;再接原步骤检索词。反馈返工(budget=None)允许多搜一点
        queries = [q for q in (list(fb_queries) +
                   (step.get("search_queries") or ([step["name"]] if step.get("need_search") else []))) if q]
        ncap = take(min(3 if fb_queries else 2, len(queries))) if queries else 0
        full_ev, did_deep = "", False
        if ncap > 0:
            hits, sources = gather_search(queries, key=search_key, cap=ncap)
            if sources:                          # 读命中页全文:表格/产能/份额明细常只住在正文里
                if notify:
                    notify("抓取命中页原文…")
                full_ev = _fetch_url_list([s["url"] for s in sources], limit=FETCH_TOP_N)
                _deep = DEEP_VERIFY if deep is None else deep
                if _deep and _wants_entity_verify(step):   # 名单/产能/份额类:逐实体免费深度核实(非阻塞,默认关)
                    try:
                        dv_ev, dv_src = deep_verify(sources, topic or step["name"], step, llm_key, notify)
                        seen = {s["url"] for s in sources}
                        for s in dv_src:
                            if s["url"] and s["url"] not in seen:
                                seen.add(s["url"]); sources.append(s)
                        if dv_ev:
                            full_ev = (full_ev + "\n\n" + dv_ev) if full_ev else dv_ev
                        did_deep = True          # 已做深度核实,后面跳过冗余的老"补搜一轮"
                    except Exception as e:
                        if notify:
                            notify(f"深度核实跳过({type(e).__name__})")
        else:
            hits = None
            if notify:
                notify("检索预算已用尽,本步基于已有知识审慎作答")
        ev = []
        if hits:
            ev.append(_search_block(hits))
        if full_ev:
            ev.append("[来源正文摘录 —— 摘要取不到的表格/明细住在这里,优先据此核实;"
                      "只采用其中真实出现的数据,无关内容忽略]\n" + full_ev[:7000])
        full = write(ev if ev else
                     ["(本步未做新检索,基于知识审慎梳理近期进展,并标注'需核实')"])
        # —— 安全重搜:仅普通执行(非反馈返工)、覆盖不足且仍有预算时补一轮;反馈返工已带定向检索,不再自动补 ——
        if not fb_queries and not did_deep and hits and (len(sources) < MIN_SOURCES or _gap_count(full) >= GAP_MAX) and take(1) > 0:
            if notify:
                notify("覆盖不足,换检索词补搜一轮…")
            alt = _alt_queries(topic or step["name"], step, queries, llm_key)
            if alt:
                hits2, src2 = gather_search(alt, key=search_key, cap=1)
                seen = {s["url"] for s in sources}
                new_urls = []
                for s in src2:
                    if s["url"] and s["url"] not in seen:
                        seen.add(s["url"]); sources.append(s); new_urls.append(s["url"])
                ev2 = _fetch_url_list(new_urls, limit=FETCH_TOP_N)   # 补搜命中页也抓全文
                merged = ((hits or "") + "\n" + (hits2 or "")).strip()
                if merged:
                    blocks = [_search_block(merged)]
                    all_ev = "\n\n".join(x for x in (full_ev, ev2) if x)
                    if all_ev:
                        blocks.append("[命中页原文抓取 —— 优先据此核实,只采用真实出现的数据]\n" + all_ev[:7000])
                    full = write(blocks)   # 用增补后的证据重写,结论由模型据实给
    elif corpus:
        full = write(["[参考资料]\n" + corpus[:8000]])
    else:
        full = write([])

    concl, detailed = parse_text_result(full)
    if feedback and feedback.strip():
        # 反馈返工:来源累加(用户给的链接 + 本轮检索 + 上一版),去重;不做"只留引用过"过滤,免得把补充来源筛掉
        merged, seen = [], set()
        for s in (url_sources + sources + prior_sources):
            u = s.get("url")
            if u and u not in seen:
                seen.add(u); merged.append(s)
        sources = merged
    else:
        sources = _cited_sources(sources, full)      # 普通执行:只留正文真正引用到的来源
    res = {"id": step["id"], "name": step["name"], "type": step.get("type", ""),
           "main_conclusion": concl, "detailed": detailed, "sources": sources}
    return res, render_html.md_to_html(detailed) + render_html.sources_html(sources)

def _persist(topic, data, iterations, done):
    """落盘 json + 渲染 html。done 是 {id:result},按 id 排序写出。每步后调用 -> 断点不丢已完成步。"""
    import render_html
    plan_path = f"{topic}_plan.json"
    json.dump({"topic": topic, "plan": data, "plan_iterations": iterations,
               "lang": (_RUNS.get(topic) or {}).get("lang", "zh"),
               "results": [done[i] for i in sorted(done)]},
              open(plan_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    try:
        render_html.build(plan_path)
    except Exception:
        pass
    return plan_path

def _downstream_ids(steps, sid):
    """返回(传递闭包地)依赖 sid 的所有下游步骤 id,升序。steps={id:step}。"""
    sid = int(sid)
    deps_of = {i: set(s.get("depends_on") or []) for i, s in steps.items()}
    out, frontier = set(), {sid}
    changed = True
    while changed:
        changed = False
        for i, ds in deps_of.items():
            if i not in out and (frontier & ds):
                out.add(i); frontier.add(i); changed = True
    return sorted(out)

def _load_state(topic):
    """取运行状态:内存缓存优先,没有就从 _plan.json 重建(服务器重启后仍可重试/返工)。"""
    st = _RUNS.get(topic)
    if not st:
        d = json.load(open(f"{topic}_plan.json", encoding="utf-8"))   # 抛错由调用方兜
        st = {"data": d["plan"], "iterations": d.get("plan_iterations", []),
              "steps": {s["id"]: s for s in d["plan"]["steps"]},
              "done": {r["id"]: r for r in d.get("results", [])},
              "search_key": None, "llm_key": None, "corpus": None, "lang": d.get("lang", "zh")}
        _RUNS[topic] = st
    return st

def import_plan(plan_obj):
    """导入用户上传的一份 plan.json(dict):校验 → 落盘为工作副本 → 载入运行状态。
    返回 topic。校验失败抛 ValueError。只载入这一份,不自动聚合历史。"""
    if not isinstance(plan_obj, dict):
        raise ValueError("不是有效的 JSON 对象")
    plan = plan_obj.get("plan") or {}
    steps = plan.get("steps")
    if not (isinstance(steps, list) and steps):
        raise ValueError("JSON 里缺少 plan.steps —— 似乎不是本工具导出的研究文件")
    _sanitize_deps(steps)      # 导入的旧文件也焊一遍 DAG,防历史/手改文件带前向或悬空依赖
    topic = re.sub(r'[\\/:*?"<>|]', "_", (plan_obj.get("topic") or "导入的研究").strip()) or "导入的研究"
    plan_obj["topic"] = topic
    json.dump(plan_obj, open(os.path.join(HERE, f"{topic}_plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)          # 落盘为工作副本,供 refine/retry/persist 复用
    _RUNS.pop(topic, None)                            # 清掉旧内存态,强制从新文件重建
    _load_state(topic)
    return topic

def _redo_step(topic, step_id, search_key, llm_key, feedback=""):
    """重跑/返工单步:feedback 非空即带用户反馈返工。返回 step_done / step_error 事件。"""
    if is_running(topic):     # 主流程跑动中不允许插入改写(防竞态 bug③);让用户等这轮完或用"打断纠正"
        return {"type": "busy", "id": step_id,
                "msg": "研究进行中,请等这一轮跑完再返工;想改框架可用页面上方的「✋ 打断纠正」。"}
    try:
        st = _load_state(topic)
    except Exception as e:
        return {"type": "step_error", "id": step_id, "msg": f"无运行缓存且读 json 失败: {e}"}
    sid = int(step_id)
    step = st["steps"].get(sid)
    if not step:
        return {"type": "step_error", "id": sid,
                "msg": "该步骤状态已丢失(多半是后端重启过)。点顶栏「↻ 重试 / 继续」从已保存进度恢复后,再返工此页。"}
    prior_res = st["done"].get(sid)
    prior = (prior_res or {}).get("detailed", "")
    try:
        if step.get("wrapup"):                           # 收口页:重新收拢(读到各步当前最新版)
            res, html = _exec_wrap(step, list(st["steps"].values()), st["done"], topic,
                                   detail=st.get("detail", ""), llm_key=llm_key or st.get("llm_key"),
                                   lang=st.get("lang", "zh"), feedback=feedback, prior=prior,
                                   research_class=(st["data"].get("research_class") or ""))
        else:
            res, html = _exec_one(step, st["done"], search_key or st.get("search_key"),
                                  llm_key or st.get("llm_key"), st.get("corpus"), topic=topic,
                                  detail=st.get("detail", ""), feedback=feedback, prior=prior,
                                  lang=st.get("lang", "zh"))
        if feedback and prior_res:                       # 返工:先把上一版存进 prev,供一键还原
            st.setdefault("prev", {})[sid] = prior_res
        st["done"][sid] = res
        _persist(topic, st["data"], st.get("iterations", []), st["done"])
        # 这一步变了,告知前端哪些下游步骤依赖它(只在已完成的下游里提示,可选连带刷新)
        down = [i for i in _downstream_ids(st["steps"], sid) if i in st["done"]]
        return {"type": "step_done", "id": sid, "conclusion": res["main_conclusion"],
                "detailed_html": html, "sources": res["sources"], "downstream": down,
                "can_revert": bool(feedback and prior_res)}
    except Exception as e:
        return {"type": "step_error", "id": sid, "msg": f"{type(e).__name__}: {e}"}

def retry_one(topic, step_id, search_key=None, llm_key=None):
    """重跑单个步骤(前端"重试此步")。"""
    return _redo_step(topic, step_id, search_key, llm_key)

def refine_one(topic, step_id, feedback, search_key=None, llm_key=None):
    """带用户反馈返工单步(前端"反馈返工")。"""
    if not (feedback or "").strip():
        return {"type": "step_error", "id": step_id, "msg": "反馈为空"}
    return _redo_step(topic, step_id, search_key, llm_key, feedback=feedback)

def revert_one(topic, step_id):
    """还原某步到返工前的上一版(前端"还原上一版")。"""
    import render_html
    if is_running(topic):
        return {"type": "busy", "id": step_id, "msg": "研究进行中,请等这一轮跑完再操作。"}
    try:
        st = _load_state(topic)
    except Exception as e:
        return {"type": "step_error", "id": step_id, "msg": f"读状态失败: {e}"}
    sid = int(step_id)
    prev = (st.get("prev") or {}).get(sid)
    if not prev:
        return {"type": "step_error", "id": sid, "msg": "没有可还原的上一版(返工前的版本未缓存,或后端重启过)"}
    st["done"][sid] = prev
    st["prev"].pop(sid, None)
    _persist(topic, st["data"], st.get("iterations", []), st["done"])
    html = render_html.md_to_html(prev.get("detailed", "")) + render_html.sources_html(prev.get("sources", []))
    return {"type": "step_done", "id": sid, "conclusion": prev.get("main_conclusion", ""),
            "detailed_html": html, "sources": prev.get("sources", [])}

# ---------------- 编排 ----------------
def run(topic, n_steps=7, corpus=None):
    p, iterations = plan(topic, n_steps)
    steps = sorted(p["steps"], key=lambda s: s["id"])
    done, budget = {}, {"left": SEARCH_BUDGET}
    for s in steps:                       # id 递增即满足依赖(依赖均指向更小 id)
        print(f"[执行] 步骤{s['id']} {s['name']} ({s.get('type','')})"
              + ("  [需搜索]" if s.get("need_search") else ""))
        res, _ = _exec_one(s, done, corpus=corpus, topic=topic, budget=budget)
        done[s["id"]] = res
        print("  完成。")

    # 收口页:未尽问题与访谈清单(与服务版同一逻辑)
    wrap = _wrap_step_obj(steps)
    p["steps"].append(wrap); steps.append(wrap)
    print(f"[收口] {wrap['name']}")
    res, _ = _exec_wrap(wrap, steps, done, topic, research_class=p.get("research_class") or "")
    done[wrap["id"]] = res

    plan_path = f"{topic}_plan.json"
    json.dump({"topic": topic, "plan": p, "plan_iterations": iterations,
               "results": [done[i] for i in sorted(done)]},
              open(plan_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    dt = p["deep_thinking"]
    md = [f"# {topic} —— 深度研究报告\n",
          "## 0. 深度思考",
          f"**主题研判**:{dt.get('context_analysis','')}\n",
          f"**研究计划**:{dt.get('research_plan','')}\n",
          f"**范围拆解**:{dt.get('scope_decomposition','')}\n"]
    for s in steps:
        r = done[s["id"]]
        dep = ("  ·  依赖: " + ",".join(map(str, s.get("depends_on") or []))) if s.get("depends_on") else ""
        md.append(f"## {s['id']}. {s['name']}  〔{s['type']}〕{dep}")
        md.append(f"> **结论**:{r['main_conclusion']}\n")
        md.append(r["detailed"] + "\n")
    open(f"{topic}_report.md", "w", encoding="utf-8").write("\n".join(md))
    print(f"\n完成 ->  {topic}_plan.json  /  {topic}_report.md")

    # 自动渲染 HTML 查看器(失败不影响已落盘的 json/md)
    try:
        import render_html
        render_html.build(plan_path)
    except Exception as e:
        print(f"[warn] HTML 渲染跳过: {e};可手动 py render_html.py \"{plan_path}\"")

# ================= 流式版(供本地服务器 server.py 调用,实现"呼吸"界面) =================

def call_llm_stream(messages, max_tokens=4000, temperature=0.4, key=None):
    """生成器:流式产出 ('delta','reasoning'|'content',文本片段),最后产出 ('final', content, reasoning)。
    流式失败则回退到非流式 call_llm,只产出一个 ('final',...)。"""
    body = {"model": CFG["model"], "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True}
    req = urllib.request.Request(
        CFG["baseUrl"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + (key or CFG["apiKey"]), "Content-Type": "application/json"})
    cparts, rparts = [], []
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"]
                except Exception:
                    continue
                rc, c = delta.get("reasoning_content"), delta.get("content")
                if rc:
                    rparts.append(rc); yield ("delta", "reasoning", rc)
                if c:
                    cparts.append(c); yield ("delta", "content", c)
        yield ("final", "".join(cparts), "".join(rparts))
    except Exception as e:
        print(f"  [warn] 流式失败,回退非流式: {e}")
        content, reasoning, _ = call_llm(messages, max_tokens, temperature)
        yield ("final", content, reasoning)

EXEC_SYS_TEXT = """你在执行一份行研计划中的某一步,产出高质量研究结论。严格遵循该步的 instruction 与 quality_rubric。
- 若提供了[参考资料]或[检索结果],优先据其作答,不要编造;未提供则基于专业知识审慎作答,对不确定处明确标注。
- [依赖结论]是前序步骤的结论,可引用衔接,但不要整段复述。
- 公式与化学式一律用纯文本 Unicode(如 H₂O、OH⁻、O₂↑、→、Δ、≥),禁止使用 LaTeX、$ 符号或 \\command。
- 数据缺失时,如实写明"公开数据中未找到/以最近可得年份为准"即可——这是高质量的体现;严禁编造数字或用模糊措辞掩盖缺口。
- 【说有的、略没有的】只写有检索证据或扎实专业共识支撑的内容。检索没覆盖、又不是该步要害的,一句带过或直接略过即可;绝不要为了"完整/对称"先搭一个小标题或表格框架、再逐格宣告"本次检索未覆盖"——空框架是噪音,不是严谨。宁可短而实,不要长而空。
- 【数字克制】数字是为支撑结论服务的,不是越多越精确越显得专业。只有当某个量真正决定判断、且有可靠来源时,才给精确值并标来源;否则用定性、量级或方向性描述,或直接写"该维度缺乏可靠公开数据"。严禁为了显得严谨而堆砌无来源的精确数字。查不到精确值时,可用代理指标/区间逼近并说明推算路径,但若连可靠的近似基础都没有,就如实说缺,不要硬造。
- 【表格/矩阵克制】对比矩阵只填有依据的格子;没有可靠数据的格子,留"—"或用定性等级(高/中/低)并注明系定性判断。绝不用看似精确实则杜撰的数字把矩阵撑满。维度也按数据可得性取舍,不必每个维度都凑齐。
- 【就地全收,不向外联想】当一条检索命中或一段证据里,除了你要找的对象之外还并列着同类的实物数据(如同一篇文章顺带列了多家公司的产能、同一张表含多行同维度数据),把这些一并收入,别只摘匹配查询的那一条——已经摆在眼前的数据不要浪费。严格边界:这只针对"已在你手上的证据里物理存在"的数据;绝不是借此发散去联想或脑补未出现的关联领域(例如由 DHA 联想去补整个鱼油/Omega-3 市场),没在证据里出现的就是没有,不收、不造。
- 【核实按材料性排序】当要逐条核实一批数字(如一张产能/份额名单)时,把核实精力优先给"错了就会推翻结论"的大额/关键条目;尾部小量级条目可只做轻校验或标"单源待核"。不要平均用力,更不要因为尾部条目好查就把检索预算耗在那里。
- 【成品口吻,不留工作痕迹】你的输出是给读者的成品报告,不是工作记录或与助手的对话。严禁出现过程性/工具性措辞:如"根据用户提供的网址/链接""本次检索""本次核实""该网页被爬虫限制/抓取失败/需登录""参考资料中未包含"等。来源正常引用即可(标题/机构/日期)。数据确有缺失时,只中性陈述数据本身的缺口(如"公开渠道未见该产能的权威披露"),不要用"本次…未能获取"这种带动作的说法,更不要描述抓取/检索的技术过程。
- 信息纪律(下列只在"已经有数据"时适用,绝不是凭空造数或强行对标的理由):① 按可信度加权——监管/审计约束(交易所披露、年报、统计公报)> 行业协会(有口径)> 龙头公司 IR > 投融资/并购新闻(真实交易事件)> 付费库 > 咨询 PR/研报"预测数" > 博客/排名站;凡"卖东西"的信源其数字打折看。前沿/新兴技术尤其要重投融资与并购新闻——"谁拿了哪轮、谁被谁收购"是比咨询机构市场规模预览版更硬的玩家与景气信号(但融资金额/估值可能有水分,只取事件本身、估值做参考)。引用券商研报的具体数字时,先追到研报小字标注的原始出处(协会/统计/公司披露),按那个原始信源定级;研报未注明出处的数与其自身"预测数",只当线索、不进结论。② 关键数字若只有单一来源,标注"(单源,待二次信源核实)";cross-check 是"当确有两条独立来源时,对上则提高置信度",不是"必须给每个数字都配两个来源",更不是"没来源也要硬凑一个对标值"。③ 时效折扣:竞争/客户数据超 5 年、技术数据超 10 年存疑;"预测数"慎用。④ 锚点+换算:权威口径(协会/统计年鉴)未更新到最近年份时,可用龙头公司被审计披露的同比增速在旧口径上推算近似值,并写明"锚点数值+推算路径";锚必须是权威口径,不得拿研报或咨询的预测数当锚。⑤ 事实与判断分离:先呈现事实,判断须立足所列事实并说清依据。
输出格式(不要 JSON,不要任何前言):第一行以"结论："开头写一句话核心结论;然后空一行,写详细展开(markdown,可分点/表格)。"""

def parse_text_result(text):
    """把'结论：…\\n\\n详细…'拆成 (conclusion, detailed)。"""
    text = re.sub(r"```(?:markdown)?", "", text).strip()
    lines = text.split("\n")
    concl, rest_start = "", 0
    for idx, ln in enumerate(lines):
        if ln.strip():
            concl = re.sub(r"^[#>\-\s*]*(?:结论|Conclusion)\s*[：:]\s*", "", ln.strip(), flags=re.I)
            rest_start = idx + 1
            break
    detailed = "\n".join(lines[rest_start:]).strip()
    return concl or text[:80], detailed or text

# ---------------- 收口页:未尽问题与访谈清单 ----------------
# 报告末尾追加一页(不占常规 7-8 步):收拢各步如实标注的信息缺口,按材料性排序;
# 并把公开检索本质答不了、必须访谈/内部数据才能验证的尽调维度落成清单
# (源自"公司调研中值得参考的问题":客户/创始人/竞对/前员工交叉印证,财务三表需内部数据)。
WRAP_NAME = {"zh": "未尽问题与访谈清单", "en": "Open Questions & Interview Plan"}
WRAP_TYPE = {"zh": "收口", "en": "Wrap-up"}

WRAP_SYS = """你在为一份行研报告写最后的收口页:「未尽问题与访谈清单」。常规研究步骤已完成,下面给你每一步的核心结论与其中如实标注的信息缺口。产出两部分:
一、还没了解到的问题——把真正影响结论的未决问题按材料性排序(错了就会推翻或明显修正结论的排前面),每条写清:问题本身;为什么重要(动摇哪一步的哪个判断);公开渠道还可尝试的补查路径(指向具体信源体裁,如 环评公示/招股书/年报/公告/协会口径/统计公报/海关数据/投融资新闻)。只列确实悬着的,不把"数据略旧"这类小瑕疵凑数;通常 3-8 条,宁缺毋滥。
二、访谈与内部数据清单——公开检索本质上答不了的部分,按访谈对象归类,每条写清:对谁、问什么、用来验证哪一步的什么判断。问题要具体到能直接开口问(如"替换现供应商时,导入验证要多久、卡在哪道工序"),不写"了解市场情况"这类空话。
  对象参考(按主题类型取用,不硬凑):公司/标的类——客户(需求真实性、购买驱动要素、转换成本、验收口径)、创始人/管理层(战略与路线取舍)、竞争对手与前员工(交叉印证)、内部数据(财务三表、运营指标、在手订单);产业/技术/政策类——等价落到 行业专家、协会、龙头企业IR、主管部门、产线实地。
克制要求:两部分都只从下面材料里真实出现的缺口与该主题公开信息的天然边界出发,不套通用清单;某部分没实质内容就写短,不为对称硬凑。
输出格式(不要 JSON,不要前言):第一行以"结论:"开头,一句话概括这项研究离"可下判断"还差哪几块;空一行后写详细展开(markdown,两部分各一节)。"""

def _wrap_step_obj(steps, lang="zh"):
    """构造收口页的合成步骤:id=最大+1,依赖全部常规步骤,不检索。wrapup 标记供重试/续跑分流。"""
    ids = [s["id"] for s in steps if not s.get("wrapup")]
    L = "en" if (lang or "zh").lower().startswith("en") else "zh"
    return {"id": max(ids) + 1, "name": WRAP_NAME[L], "type": WRAP_TYPE[L],
            "instruction": ("收拢全篇:仍悬而未决、会影响结论的问题按材料性排序;"
                            "公开检索答不了的维度落成访谈/内部数据清单(对谁、问什么、验证哪个判断)。"),
            "quality_rubric": "问题具体可执行、直指结论要害为高质量;套通用清单、为对称硬凑条目为低质量。",
            "depends_on": ids, "need_search": False, "search_queries": [], "wrapup": True}

def _gap_lines(text, cap=10):
    """从一步的 detailed 里抽出含缺口标记的句子(未找到/单源待核/检索未覆盖…),供收口页收拢。"""
    out = []
    extra = ("单源", "未见", "待核", "有限", "尚无")   # EXEC 成品口吻常写"公开渠道未见/待二次信源核实"
    for ln in re.split(r"[\n。;；]", text or ""):
        ln = ln.strip().lstrip("-·* ")
        if any(p in ln for p in _GAP_PATS) or any(p in ln for p in extra):
            if 6 <= len(ln) <= 160 and ln not in out:
                out.append(ln)
        if len(out) >= cap:
            break
    return out

def _exec_wrap(step, steps, done, topic, detail="", llm_key=None, lang="zh",
               feedback="", prior="", research_class=""):
    """生成收口页,返回 (结果dict, 详细HTML)。不检索、无来源,一次 pro 关思考调用。"""
    import render_html
    parts = [f"当前日期:{TODAY}。研究主题:{topic}" +
             (f"(研究类别:{research_class})" if research_class else "")]
    if detail and detail.strip():
        parts.append("[用户补充的研究侧重/背景]\n" + detail.strip())
    if feedback and feedback.strip():
        if prior:
            parts.append("[上一版收口页 —— 你的输出将整页替换它,旧版对的内容要完整重述]\n" + prior[:3000])
        parts.append("[修订要点 —— 逐条处理,自然融进成品,不要出现'用户反馈'等字样]\n" + feedback.strip())
    for s in sorted(steps, key=lambda x: x["id"]):
        if s.get("wrapup"):
            continue
        r = done.get(s["id"])
        if not r:
            continue
        seg = f"【步骤{s['id']} {s['name']}】结论:{r.get('main_conclusion', '')}"
        gaps = _gap_lines(r.get("detailed", ""))
        if gaps:
            seg += "\n该步标注的缺口:\n" + "\n".join("- " + g for g in gaps)
        parts.append(seg)
    full, _, _ = call_llm([{"role": "system", "content": WRAP_SYS + _lang_directive(lang, "exec")},
                           {"role": "user", "content": "\n\n".join(parts)}],
                          max_tokens=4000, temperature=0.4, key=llm_key, model=PRO_MODEL, think=False)
    concl, detailed = parse_text_result(full)
    res = {"id": step["id"], "name": step["name"], "type": step.get("type", ""),
           "main_conclusion": concl, "detailed": detailed, "sources": []}
    return res, render_html.md_to_html(detailed)

def run_stream(topic, llm_key=None, search_key=None, n_steps=7, corpus=None, resume=False, detail="", deep_verify=False, lang="zh"):
    """生成器:逐事件 yield dict,供 SSE 推给前端。
    resume=True:断线续跑——复用缓存的 plan,跳过已完成步、补发其结果,接着跑剩下的(浏览器掉线不必重头)。"""
    import render_html
    _SEQ[0] += 1; _myseq = _SEQ[0]; _ACTIVE[topic] = _myseq   # 标记本 topic 正在跑(护栏),收尾在 finally 清
    try:
        # —— 续跑:内存缓存优先;没有(如后端重启过)就从 _plan.json 重建,跳过已完成步 ——
        st = None
        if resume:
            if topic in _RUNS:
                st = _RUNS[topic]
            else:
                try:
                    st = _load_state(topic)          # 从盘上 _plan.json 把规划+已完成结果读回来
                except Exception:
                    st = None                         # 盘上也没有 -> 退回从头规划
        if st is not None:
            data, iterations = st["data"], st.get("iterations", [])
            steps = sorted(st["steps"].values(), key=lambda s: s["id"])
            done = st["done"]
            if search_key: st["search_key"] = search_key
            if llm_key: st["llm_key"] = llm_key
            detail = st.get("detail", detail)
            lang = st.get("lang", lang)
            st.setdefault("budget", {"left": SEARCH_BUDGET})   # 重建态可能没预算,给个默认别卡住补搜
            n_done = len(done)
            yield {"type": "status",
                   "msg": f"恢复上次进度…(已完成 {n_done} 步,不重跑)" if n_done else "恢复上次进度…"}
            # 先吐一份完整框架:冷页面(导入/刷新后)据此把卡片+左边目录整份建出来,再由下面的 step_done 填充
            yield {"type": "plan_iter", "k": 4, "label": "已载入", "plan": data}
        else:
            # —— 预检索:规划前先宽口径搜一眼,让模型知道"有哪些数据查得到" ——
            probe, probe_items, pre_used = "", [], 0
            yield {"type": "status", "msg": "预检索:摸一眼信息全貌…"}
            try:
                probe, probe_items, pre_used = landscape_probe(topic, detail, search_key=search_key, llm_key=llm_key)
            except Exception as e:
                print(f"  [warn] 预检索失败,跳过: {e}")
            if pre_used:
                yield {"type": "status", "msg": f"预检索完成({pre_used} 次,命中 {len(probe_items)} 条),开始规划…"}
            # —— 多轮迭代规划:每轮整页产出 plan_iter(含完整 instruction/rubric/deep_thinking) ——
            yield {"type": "status", "msg": "深度思考中…"}
            data, iterations = None, []
            for ev in plan_iters(topic, llm_key=llm_key, stream_iter1=False, detail=detail, probe=probe, lang=lang):
                if ev["type"] == "plan_iter":
                    iterations.append({"k": ev["k"], "label": ev["label"], "steps": ev["plan"]["steps"]})
                    yield {"type": "status", "msg": f"规划 Iter{ev['k']}/4 · {ev['label']}"}
                    yield ev
                elif ev["type"] == "plan_final":
                    data = ev["plan"]
            if not data or not data.get("steps"):
                yield {"type": "error", "msg": "规划未能产出有效步骤,请重试(点上方「↻ 重试」或刷新)"}
                return
            data.setdefault("deep_thinking", {})
            steps = sorted(data["steps"], key=lambda s: s["id"])
            done = {}
            _RUNS[topic] = {"data": data, "iterations": iterations,
                            "steps": {s["id"]: s for s in steps}, "done": done,
                            "search_key": search_key, "llm_key": llm_key, "corpus": corpus,
                            "detail": detail, "lang": lang, "budget": {"left": max(0, SEARCH_BUDGET - pre_used)}}

        dt = data.get("deep_thinking", {})
        yield {"type": "meta", "topic": topic,
               "thinking": {"context": dt.get("context_analysis", ""),
                            "plan": dt.get("research_plan", ""),
                            "scope": dt.get("scope_decomposition", "")},
               "steps": [{"id": s["id"], "name": s["name"], "type": s.get("type", ""),
                          "deps": s.get("depends_on") or []} for s in steps]}

        # —— 逐步执行:已完成的(续跑)直接补发;某步失败不中断;每步存盘(断点不丢) ——
        for s in steps:
            sid = s["id"]
            if s.get("wrapup"):
                continue                         # 收口页在常规步骤全部完成后单独生成(见下)
            if sid in done:                      # 续跑时已有结果,重发让前端补画
                r = done[sid]
                html = render_html.md_to_html(r["detailed"]) + render_html.sources_html(r.get("sources", []))
                yield {"type": "step_done", "id": sid, "conclusion": r["main_conclusion"],
                       "detailed_html": html, "sources": r.get("sources", [])}
                continue
            yield {"type": "step_start", "id": sid, "name": s["name"]}
            yield {"type": "step_status", "id": sid,
                   "msg": "检索中…" if s.get("need_search") else "撰写中…"}
            try:
                res, html = _exec_one(s, done, search_key or _RUNS[topic].get("search_key"),
                                      llm_key or _RUNS[topic].get("llm_key"), corpus, topic=topic,
                                      detail=detail, budget=_RUNS[topic].get("budget"), deep=deep_verify,
                                      lang=_RUNS[topic].get("lang", lang))
                done[sid] = res
                _persist(topic, data, iterations, done)
                yield {"type": "step_done", "id": sid, "conclusion": res["main_conclusion"],
                       "detailed_html": html, "sources": res["sources"]}
            except Exception as e:
                yield {"type": "step_error", "id": sid, "msg": f"{type(e).__name__}: {e}"}

        # —— 收口页:未尽问题与访谈清单(常规步骤全部完成后追加;有失败步则先不收口,等重试) ——
        wrap = next((s for s in steps if s.get("wrapup")), None)
        try:
            if wrap and wrap["id"] in done:      # 续跑/导入:已生成过,补画即可
                r = done[wrap["id"]]
                yield {"type": "step_done", "id": wrap["id"], "conclusion": r["main_conclusion"],
                       "detailed_html": render_html.md_to_html(r.get("detailed", ""))
                                        + render_html.sources_html(r.get("sources", [])),
                       "sources": r.get("sources", [])}
            else:
                normal_ids = [s["id"] for s in steps if not s.get("wrapup")]
                if done and all(i in done for i in normal_ids):
                    st = _RUNS.get(topic) or {}
                    if wrap is None:
                        wrap = _wrap_step_obj(steps, st.get("lang", lang))
                        data["steps"].append(wrap); steps.append(wrap)
                        st.setdefault("steps", {})[wrap["id"]] = wrap
                        yield {"type": "step_new",     # 让前端补建这张卡与左目录项
                               "step": {k: wrap[k] for k in
                                        ("id", "name", "type", "instruction", "depends_on", "need_search")}}
                    yield {"type": "step_start", "id": wrap["id"], "name": wrap["name"]}
                    yield {"type": "step_status", "id": wrap["id"], "msg": "收拢未尽问题与访谈清单…"}
                    res, html = _exec_wrap(wrap, steps, done, topic, detail=detail,
                                           llm_key=llm_key or st.get("llm_key"), lang=st.get("lang", lang),
                                           research_class=data.get("research_class") or "")
                    done[wrap["id"]] = res
                    _persist(topic, data, iterations, done)
                    yield {"type": "step_done", "id": wrap["id"], "conclusion": res["main_conclusion"],
                           "detailed_html": html, "sources": []}
        except Exception as e:
            yield {"type": "step_error", "id": (wrap or {}).get("id", 0), "msg": f"{type(e).__name__}: {e}"}

        yield {"type": "saved", "plan": f"{topic}_plan.json"}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "msg": f"{type(e).__name__}: {e}"}
    finally:
        if _ACTIVE.get(topic) == _myseq:      # 仅当仍是本 run 时才清(打断重开后新 run 占位,别误清)
            _ACTIVE[topic] = 0

if __name__ == "__main__":
    args = sys.argv[1:]
    topic, n_steps, corpus = None, 7, None
    i = 0
    while i < len(args):
        if args[i] == "--steps": n_steps = int(args[i + 1]); i += 2
        elif args[i] == "--corpus":
            corpus = open(args[i + 1], encoding="utf-8").read(); i += 2
        else: topic = args[i]; i += 1
    if not topic:
        print('用法: py -X utf8 research_agent.py "研究主题" [--steps 7] [--corpus 语料.txt]'); sys.exit(1)
    run(topic, n_steps, corpus)
