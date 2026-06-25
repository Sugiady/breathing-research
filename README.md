# 深度行研 Agent · 会呼吸的研究

> 输入一个主题,它先做 **4 轮迭代规划**,再按依赖顺序**逐步检索与撰写**,像呼吸一样在网页上实时刷新每一步的思考链。
>
> **Claude 设计 · DeepSeek 驱动**

一个把"深度研究工具"的规划—执行 harness 用 DeepSeek + Web Search 复刻出来的本地行研台,带一个左右滑动、实时更新的"呼吸"界面。

## 🔭 在线 Demo

**[→ 报告画廊（GitHub Pages）](https://Sugiady.github.io/breathing-research/)**

画廊里是一份真实产出的研究报告样例(Anti Fouling 防污技术综述)。点开可左右滑动浏览每一步的思考链——这是**零后端、零密钥**的静态成品快照。

完整的实时"呼吸"界面需要在本地带 key 运行(见下)。

## ✨ 它怎么工作

```
主题  →  ① 预检索(摸一眼有哪些数据)
      →  ② 4 轮迭代规划
            Iter1 框架与骨架(按主题自身结构拆，不套模板)
            Iter2 类型与指令
            Iter3 依赖关系
            Iter4 质量判据与检索词(按问题倒推信源类型)
      →  ③ 逐步执行(按依赖序，需要时检索真实证据)
      →  ④ 落盘 + 实时推送到"呼吸"界面
```

设计要点:

- **框架不写死**——技术/产业/政策/公司/材料类主题，由 Iter1 自选最契合的步骤组合。
- **规划前预检索**——先宽口径搜一眼，让模型据"哪些数据查得到"来设步骤与维度，少编数。
- **信息纪律**——可信度阶梯加权(监管/年报/统计公报 > 行业协会 > 龙头 IR > 投融资并购 > 咨询 PR)、单源标注待核实、事实与判断分离、**说有的略没有的、不为完整搭空框架、不用杜撰数字撑满表格**。
- **搜索预算**——一次研究总检索约 5–9 次，模型自分配、代码兜底。
- **断点续跑**——每步即存盘，断线/失败可单页重试或整体续跑。

## 🚀 本地运行

需要 Python 3.10+(Windows 上用 `py`)。

1. 复制配置模板并填入你自己的 key:

   ```bash
   cp deepseek_api.example.json deepseek_api.json   # 填 DeepSeek key
   cp bocha_api.example.json    bocha_api.json       # 填博查 Web Search key(可选，不填则新闻步走模型知识)
   ```

2. 启动本地服务(key 只留在本机、服务端调 API，浏览器只连 localhost，无 CORS):

   ```bash
   py -X utf8 server.py
   ```

3. 浏览器打开 <http://localhost:8770> ，输入主题，点亮。

> 也可命令行离线跑:`py -X utf8 research_agent.py "AEM电解水制氢"`

## 📁 结构

| 文件 | 作用 |
| --- | --- |
| `research_agent.py` | 核心引擎:规划(4 轮迭代)+ 执行 + 预检索 + 搜索预算 |
| `server.py` | 本地 SSE 服务(`/run` 流式、`/retry` 单步重试、`/ping` 版本) |
| `live.html` | "呼吸"实时界面(左右滑动、整页刷新、断点续跑) |
| `render_html.py` | Markdown→HTML 渲染 + 静态报告查看器模板 |
| `build_gallery.py` | 把精选报告汇成 `docs/` 画廊(GitHub Pages) |

## 🔒 关于密钥

`deepseek_api.json` / `bocha_api.json` 等含真实 key 的文件已在 `.gitignore` 中，**不会进仓库**。仓库里只有 `*.example.json` 占位模板。

---

界面与工程由 Claude 设计;研究方法论与主题来自一线投研实践。
