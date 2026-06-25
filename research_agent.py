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
import json, sys, os, time, urllib.request, re, datetime

TODAY = datetime.date.today().isoformat()
YEAR = datetime.date.today().year

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "deepseek_api.json"), encoding="utf-8"))

# 最近一次运行的状态缓存(按主题),供"重试某步"用;服务器重启后可从 _plan.json 重建
_RUNS = {}

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
    return CFG.get("fastModel") or CFG["model"]

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
            with urllib.request.urlopen(req, timeout=300) as r:
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
    if not os.path.isfile(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return d.get("apiKey") or d.get("api_key") or d.get("key")

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

def gather_search(queries, key=None, cap=3):
    """跑多条查询,返回 (喂给模型的文本, 去重后的来源条目列表)。cap 限制本次最多用几条查询。"""
    hits, items, seen = None, [], set()
    for q in (queries or [])[:cap]:
        its = _bocha_items(q, key=key)
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
- search_queries(若 need_search 为 true,给出中英文检索查询词数组)。检索词遵循"按问题倒推信源类型":优先指向强约束信源(监管/交易所披露、年报季报、统计公报与年鉴、行业协会、龙头公司 IR),而不是泛搜"xx market size";数据型步骤点名 具体指标+地区+最近年份(如"浙江 纺织 产业集群 产值 2024");新闻型含关键主体/产品/事件类型;前沿/新兴技术或竞争格局类步骤,优先检索投融资与并购新闻(融资轮次、领投方、估值、公司介绍)——这是比咨询市场报告更硬的玩家与景气信号;允许给代理指标/邻近口径的备用词,不要钻牛角尖于单一数字。否则空数组。
  检索预算有限:整个研究执行阶段总检索约 5-7 次(规划前已另做 2 次预检索)。请把检索集中到真正依赖外部数据的步骤,每个需检索步给 1-2 条最高命中率的查询;概念/原理/框架类步骤一律空数组。结合上面的[检索情报预览]判断哪些数据查得到——对预览里明显查不到的,别硬设检索。
- deep(布尔)。该步若以"原创推理/第一性推演/多因素权衡/比较研判/致命风险与前瞻综合判断"为主、需要深度思考才能写好,标 true;若主要是梳理检索结果、罗列归纳、整理时间线/名单/事件等信息整合类工作,标 false。务必克制:多数步骤应为 false,通常只有第一性/本质分析、关键技术或路线对比研判、致命风险与前瞻这类真正吃推理的步骤才标 true(一般不超过 2-3 个)。
保持前几轮已定字段不变。只输出完整 JSON(steps 每项含 id,name,type,instruction,depends_on,quality_rubric,need_search,search_queries,deep)。""",
}

def _iter_messages(k, topic, prev, detail="", probe=""):
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
    return [{"role": "system", "content": ITER_SYS[k]}, {"role": "user", "content": user}]

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

def plan_iters(topic, llm_key=None, stream_iter1=True, detail="", probe=""):
    """生成器:逐轮规划。Iter1 可流式吐 reasoning(供"呼吸"),Iter2~4 直接返回。
    yield {type:thinking_delta|plan_iter|plan_final}。
    复杂主题(步骤多/字段长)整页 JSON 可能超 max_tokens 被截断 -> 自动升档重试。"""
    prev, dt = None, {}
    for k in (1, 2, 3, 4):
        msgs = _iter_messages(k, topic, prev, detail, probe)
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

# ---------------- 阶段2: 执行单步 ----------------
EXEC_SYS = """你在执行一份行研计划中的某一步,产出高质量研究结论。
严格遵循该步的 instruction 与 quality_rubric。
- 若提供了[参考资料]或[检索结果],优先据其作答,不要编造;未提供则基于专业知识审慎作答,对不确定处明确标注。
- [依赖结论]是前序步骤的结论,可引用衔接,但不要整段复述。
- 公式与化学式一律用纯文本 Unicode(如 H₂O、OH⁻、O₂↑、→、Δ、≥),禁止使用 LaTeX、$ 符号或 \\command。
- 数据缺失时,如实写明"公开数据中未找到/以最近可得年份为准"即可——这是高质量的体现;严禁编造数字或用模糊措辞掩盖缺口。每个量化数字都应有检索来源支撑。
只输出一个 JSON: {"main_conclusion": "一句话核心结论", "detailed": "结构化详细展开(可分点)"}"""

def execute_step(step, done, corpus):
    print(f"[执行] 步骤{step['id']} {step['name']} ({step['type']})"
          + ("  [需搜索]" if step.get("need_search") else ""))
    blocks = [f"步骤名:{step['name']}", f"类型:{step.get('type','')}",
              f"研究范围(instruction):{step.get('instruction','')}",
              f"质量标准(rubric):{step.get('quality_rubric','')}"]
    deps = step.get("depends_on") or []
    if deps:
        dep_txt = "\n".join(f"【步骤{i} {done[i]['name']} 结论】{done[i]['main_conclusion']}"
                            for i in deps if i in done)
        if dep_txt:
            blocks.append("[依赖结论]\n" + dep_txt)
    sources = []
    if step.get("need_search"):
        hits, sources = gather_search(step.get("search_queries") or [step["name"]])
        if hits:
            blocks.append("[检索结果](仅可据此作答)\n" + hits[:9000]
                          + "\n\n严格要求:只整理上述检索结果中真实出现的事件,每条尽量标注[日期|来源];"
                            "不得补充检索结果里没有的事件、数字或主体;无法从结果中确认的,明确写'检索未覆盖'。")
        else:
            blocks.append("(未接入搜索,基于你的知识审慎梳理近期进展,并标注'需核实')")
    elif corpus:
        blocks.append("[参考资料]\n" + corpus[:8000])
    content, _, usage = call_llm(
        [{"role": "system", "content": EXEC_SYS},
         {"role": "user", "content": "\n\n".join(blocks)}],
        max_tokens=4000, temperature=0.4)
    res = extract_json(content)
    res["name"] = step["name"]; res["type"] = step["type"]; res["id"] = step["id"]
    res["sources"] = sources
    print(f"  完成。tokens={usage.get('total_tokens')}")
    return res

# ---------------- 单步执行(供流式 + 重试复用) ----------------
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

def _search_block(hits):
    return ("[检索结果](仅可据此作答)\n" + hits[:9000]
            + "\n\n严格要求:只整理上述检索结果中真实出现的事件,每条尽量标注[日期|来源];"
              "不得补充检索结果里没有的事件、数字或主体;无法确认的写'检索未覆盖'。")

def _exec_one(step, done, search_key=None, llm_key=None, corpus=None, topic="", notify=None, detail="", budget=None):
    """执行一个步骤,返回 (结果dict, 详细HTML)。
    安全重搜:检索覆盖不足(来源少/缺口多)时,代码侧换检索词再搜一轮、补充真实证据后重写——
    全程不奖励"减少缺口",只给更多真实证据;判定与重试都在代码侧,不问模型缺口多不多。
    budget={'left':N} 时受全局检索预算约束(预算用尽则该步走知识兜底);None=不限(单步重试用)。"""
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

    def write(extra):   # 统一 pro + 关思考(关思考后与 flash 用时无差、质量更好)
        full, _, _ = call_llm([{"role": "system", "content": EXEC_SYS_TEXT},
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
    if step.get("need_search"):
        queries = step.get("search_queries") or [step["name"]]
        ncap = take(min(2, len(queries)))          # 每步最多 2 条,且受预算约束
        if ncap > 0:
            hits, sources = gather_search(queries, key=search_key, cap=ncap)
        else:
            hits = None
            if notify:
                notify("检索预算已用尽,本步基于已有知识审慎作答")
        full = write([_search_block(hits)] if hits else
                     ["(本步未做新检索,基于知识审慎梳理近期进展,并标注'需核实')"])
        # —— 安全重搜:只在覆盖不足且仍有预算时,换检索词补一轮真实证据再重写 ——
        if hits and (len(sources) < MIN_SOURCES or _gap_count(full) >= GAP_MAX) and take(1) > 0:
            if notify:
                notify("覆盖不足,换检索词补搜一轮…")
            alt = _alt_queries(topic or step["name"], step, queries, llm_key)
            if alt:
                hits2, src2 = gather_search(alt, key=search_key, cap=1)
                seen = {s["url"] for s in sources}
                for s in src2:
                    if s["url"] and s["url"] not in seen:
                        seen.add(s["url"]); sources.append(s)
                merged = ((hits or "") + "\n" + (hits2 or "")).strip()
                if merged:
                    full = write([_search_block(merged)])   # 用增补后的证据重写,结论由模型据实给
    elif corpus:
        full = write(["[参考资料]\n" + corpus[:8000]])
    else:
        full = write([])

    concl, detailed = parse_text_result(full)
    sources = _cited_sources(sources, full)          # 只留正文真正引用到的来源
    res = {"id": step["id"], "name": step["name"], "type": step.get("type", ""),
           "main_conclusion": concl, "detailed": detailed, "sources": sources}
    return res, render_html.md_to_html(detailed) + render_html.sources_html(sources)

def _persist(topic, data, iterations, done):
    """落盘 json + 渲染 html。done 是 {id:result},按 id 排序写出。每步后调用 -> 断点不丢已完成步。"""
    import render_html
    plan_path = f"{topic}_plan.json"
    json.dump({"topic": topic, "plan": data, "plan_iterations": iterations,
               "results": [done[i] for i in sorted(done)]},
              open(plan_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    try:
        render_html.build(plan_path)
    except Exception:
        pass
    return plan_path

def retry_one(topic, step_id, search_key=None, llm_key=None):
    """重跑单个步骤(供前端"重试此步"按钮)。缓存没有就从 _plan.json 重建。返回 step_done / step_error 事件。"""
    st = _RUNS.get(topic)
    if not st:
        try:
            d = json.load(open(f"{topic}_plan.json", encoding="utf-8"))
        except Exception as e:
            return {"type": "step_error", "id": step_id, "msg": f"无运行缓存且读 json 失败: {e}"}
        st = {"data": d["plan"], "iterations": d.get("plan_iterations", []),
              "steps": {s["id"]: s for s in d["plan"]["steps"]},
              "done": {r["id"]: r for r in d.get("results", [])},
              "search_key": None, "llm_key": None, "corpus": None}
        _RUNS[topic] = st
    sid = int(step_id)
    step = st["steps"].get(sid)
    if not step:
        return {"type": "step_error", "id": sid, "msg": "找不到该步骤"}
    try:
        res, html = _exec_one(step, st["done"], search_key or st.get("search_key"),
                              llm_key or st.get("llm_key"), st.get("corpus"), topic=topic,
                              detail=st.get("detail", ""))
        st["done"][sid] = res
        _persist(topic, st["data"], st.get("iterations", []), st["done"])
        return {"type": "step_done", "id": sid, "conclusion": res["main_conclusion"],
                "detailed_html": html, "sources": res["sources"]}
    except Exception as e:
        return {"type": "step_error", "id": sid, "msg": f"{type(e).__name__}: {e}"}

# ---------------- 编排 ----------------
def run(topic, n_steps=7, corpus=None):
    p, iterations = plan(topic, n_steps)
    steps = sorted(p["steps"], key=lambda s: s["id"])
    done = {}
    for s in steps:                       # id 递增即满足依赖(依赖均指向更小 id)
        done[s["id"]] = execute_step(s, done, corpus)

    plan_path = f"{topic}_plan.json"
    json.dump({"topic": topic, "plan": p, "plan_iterations": iterations,
               "results": list(done.values())},
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
- 信息纪律(下列只在"已经有数据"时适用,绝不是凭空造数或强行对标的理由):① 按可信度加权——监管/审计约束(交易所披露、年报、统计公报)> 行业协会(有口径)> 龙头公司 IR > 投融资/并购新闻(真实交易事件)> 付费库 > 咨询 PR/研报"预测数" > 博客/排名站;凡"卖东西"的信源其数字打折看。前沿/新兴技术尤其要重投融资与并购新闻——"谁拿了哪轮、谁被谁收购"是比咨询机构市场规模预览版更硬的玩家与景气信号(但融资金额/估值可能有水分,只取事件本身、估值做参考)。② 关键数字若只有单一来源,标注"(单源,待二次信源核实)";cross-check 是"当确有两条独立来源时,对上则提高置信度",不是"必须给每个数字都配两个来源",更不是"没来源也要硬凑一个对标值"。③ 时效折扣:竞争/客户数据超 5 年、技术数据超 10 年存疑;"预测数"慎用。④ 事实与判断分离:先呈现事实,判断须立足所列事实并说清依据。
输出格式(不要 JSON,不要任何前言):第一行以"结论："开头写一句话核心结论;然后空一行,写详细展开(markdown,可分点/表格)。"""

def parse_text_result(text):
    """把'结论：…\\n\\n详细…'拆成 (conclusion, detailed)。"""
    text = re.sub(r"```(?:markdown)?", "", text).strip()
    lines = text.split("\n")
    concl, rest_start = "", 0
    for idx, ln in enumerate(lines):
        if ln.strip():
            concl = re.sub(r"^[#>\-\s]*结论[：:]\s*", "", ln.strip())
            rest_start = idx + 1
            break
    detailed = "\n".join(lines[rest_start:]).strip()
    return concl or text[:80], detailed or text

def run_stream(topic, llm_key=None, search_key=None, n_steps=7, corpus=None, resume=False, detail=""):
    """生成器:逐事件 yield dict,供 SSE 推给前端。
    resume=True:断线续跑——复用缓存的 plan,跳过已完成步、补发其结果,接着跑剩下的(浏览器掉线不必重头)。"""
    import render_html
    try:
        if resume and topic in _RUNS:
            # —— 续跑:复用上次规划与已完成结果,不重新规划 ——
            st = _RUNS[topic]
            data, iterations = st["data"], st["iterations"]
            steps = sorted(st["steps"].values(), key=lambda s: s["id"])
            done = st["done"]
            if search_key: st["search_key"] = search_key
            if llm_key: st["llm_key"] = llm_key
            detail = st.get("detail", detail)
            yield {"type": "status", "msg": "恢复上次进度…"}
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
            for ev in plan_iters(topic, llm_key=llm_key, stream_iter1=False, detail=detail, probe=probe):
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
                            "detail": detail, "budget": {"left": max(0, SEARCH_BUDGET - pre_used)}}

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
                                      detail=detail, budget=_RUNS[topic].get("budget"))
                done[sid] = res
                _persist(topic, data, iterations, done)
                yield {"type": "step_done", "id": sid, "conclusion": res["main_conclusion"],
                       "detailed_html": html, "sources": res["sources"]}
            except Exception as e:
                yield {"type": "step_error", "id": sid, "msg": f"{type(e).__name__}: {e}"}

        yield {"type": "saved", "plan": f"{topic}_plan.json"}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "msg": f"{type(e).__name__}: {e}"}

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
