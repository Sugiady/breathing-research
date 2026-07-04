# -*- coding: utf-8 -*-
"""
深度行研 Agent · 本地服务版(会呼吸的界面)。
    py -X utf8 server.py
然后浏览器开 http://localhost:8770 。

为什么要本地服务器:界面要实时调 DeepSeek/博查,若浏览器直接调 -> CORS + key 暴露。
本服务把 key 留在本地、服务端调 API、用 SSE 把每一步流式推给浏览器;浏览器只连 localhost,无 CORS。
端口 8770(避开 8000 的记忆库)。LLM/搜索 key:界面留空则回退本地 deepseek_api.json / bocha_api.json。
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import research_agent

PORT = 8770
BUILD = "2026-07-05-wrapup"   # 改代码时改这里;GET /ping 可确认连的是不是新代码

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音访问日志
        pass

    def do_GET(self):
        if self.path == "/ping":
            data = json.dumps({"ok": True, "build": BUILD}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data); return
        if self.path in ("/", "/index.html", "/live.html"):
            try:
                html = open(os.path.join(HERE, "live.html"), encoding="utf-8").read()
            except FileNotFoundError:
                self.send_error(404, "live.html 不存在"); return
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")  # 别让浏览器缓存旧 live.html
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            return {}

    def _send_json(self, ev):
        data = json.dumps(ev, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path == "/import":          # 上传一份 plan.json 导入,返回 {ok, topic}
            req = self._read_json()
            try:
                topic = research_agent.import_plan(req.get("plan"))
                self._send_json({"ok": True, "topic": topic})
            except Exception as e:
                self._send_json({"ok": False, "msg": f"{type(e).__name__}: {e}"})
            return
        if self.path in ("/retry", "/refine", "/revert"):   # 重试/反馈返工/还原上一版,返回单条 JSON(非SSE)
            req = self._read_json()
            topic = (req.get("topic") or "").strip()
            sk = (req.get("search_key") or "").strip() or None
            lk = (req.get("llm_key") or "").strip() or None
            if self.path == "/refine":
                ev = research_agent.refine_one(topic, req.get("id"),
                                               (req.get("feedback") or "").strip(),
                                               search_key=sk, llm_key=lk)
            elif self.path == "/revert":
                ev = research_agent.revert_one(topic, req.get("id"))
            else:
                ev = research_agent.retry_one(topic, req.get("id"), search_key=sk, llm_key=lk)
            data = json.dumps(ev, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data); return
        if self.path != "/run":
            self.send_error(404); return
        req = self._read_json()
        topic = (req.get("topic") or "").strip()
        detail = (req.get("detail") or "").strip()
        llm_key = (req.get("llm_key") or "").strip() or None
        search_key = (req.get("search_key") or "").strip() or None
        resume = bool(req.get("resume"))
        deep = bool(req.get("deep"))
        lang = (req.get("lang") or "zh").strip() or "zh"
        if not topic:
            self.send_error(400, "缺少 topic"); return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def push(ev):
            self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        try:
            for ev in research_agent.run_stream(topic, llm_key=llm_key, search_key=search_key, resume=resume, detail=detail, deep_verify=deep, lang=lang):
                push(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 浏览器关了页面
        except Exception as e:
            try:
                push({"type": "error", "msg": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

if __name__ == "__main__":
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        print(f"!! 端口 {PORT} 被占用(可能有上次没退干净的进程)。先释放它,或改 server.py 里的 PORT。\n   {e}")
        sys.exit(1)
    print(f"行研 Agent 本地服务启动 -> http://localhost:{PORT}  (Ctrl+C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
