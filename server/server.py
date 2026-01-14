"""
一个简单的 HTTP 服务，用于接收 Broker knobs 并应用到 Mosquitto。

设计目标：
  - 通过 POST /apply_knobs 接收 JSON 形式的 broker 配置
  - 内部调用 environment.knobs.apply_knobs(knobs)，让外部系统可以独立控制 broker
  - 方便你的强化学习/调参模块与实际运行的 Broker 解耦
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from environment.knobs import apply_knobs


class KnobServerHandler(BaseHTTPRequestHandler):
    server_version = "BrokerKnobServer/0.1"

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/apply_knobs":
            self._send_json(404, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            data = json.loads(raw_body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
        except Exception as exc:  # pylint: disable=broad-except
            self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
            return

        try:
            # 直接把收到的字段当作 knobs 传入
            apply_knobs(data)
        except Exception as exc:  # pylint: disable=broad-except
            self._send_json(500, {"error": "apply_failed", "detail": str(exc)})
            return

        self._send_json(200, {"status": "ok"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # 简单打印到 stdout，避免默认带客户端地址的 noisy 日志
        print(f"[KnobServer] {format % args}")


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """
    以阻塞方式启动 HTTPServer。
    """
    server_address = (host, port)
    httpd = HTTPServer(server_address, KnobServerHandler)
    print(f"Knob server listening on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()

