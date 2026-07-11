from __future__ import annotations

import json
import locale
import os
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = "127.0.0.1"
PORT = 5500
TIMEOUT_SECONDS = 30
SYSTEM_ENCODING = locale.getpreferredencoding(False) or "utf-8"

DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent

SCRIPT_MAP = {
    "linear": "線性回歸.py",
    "poly": "多項式回歸預測煞車.py",
    "ridge": "嶺回歸Ridge Regression預測商品銷售額與廣告關係.py",
    "lasso": "套索回歸Lasso Regression預測房屋價格.py",
    "svr": "支持向量回歸SVR預測汽車的油耗2.py",
    "logistic": "邏輯回歸預測顧客是否會購買商品2.py",
}


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO_ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            self.handle_run(parsed.query)
            return
        super().do_GET()

    def handle_run(self, query: str):
        params = parse_qs(query)
        model = params.get("model", [""])[0]
        script_name = SCRIPT_MAP.get(model)

        if not script_name:
            self.send_json(
                400,
                {
                    "ok": False,
                    "error": "未知模型代碼，請使用固定清單中的 model 參數。",
                },
            )
            return

        script_path = PROJECT_ROOT / script_name
        if not script_path.exists():
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": f"找不到對應腳本: {script_name}",
                },
            )
            return

        env = os.environ.copy()
        # Use a non-GUI backend to avoid plot windows blocking the request.
        env["MPLBACKEND"] = "Agg"

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding=SYSTEM_ENCODING,
                errors="replace",
                timeout=TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired:
            self.send_json(
                408,
                {
                    "ok": False,
                    "error": f"執行逾時（>{TIMEOUT_SECONDS} 秒）。",
                    "stdout": "",
                    "stderr": "",
                },
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.send_json(
                500,
                {
                    "ok": False,
                    "error": f"執行失敗: {exc}",
                    "stdout": "",
                    "stderr": "",
                },
            )
            return

        payload = {
            "ok": result.returncode == 0,
            "returnCode": result.returncode,
            "script": script_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        self.send_json(200, payload)

    def send_json(self, status_code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"Demo server running at http://{HOST}:{PORT}")
    with ThreadingHTTPServer((HOST, PORT), DemoHandler) as httpd:
        httpd.serve_forever()
