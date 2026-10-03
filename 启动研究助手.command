#!/bin/zsh

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
CANDIDATE_PORTS=(5050 5051 5052 5053 5054 5055 5056 5057 5058 5059 5060)

# Finder-launched shells do not always load the Node/NVM path used by Codex CLI.
CODEX_BIN="$(command -v codex 2>/dev/null || true)"
if [[ -z "$CODEX_BIN" && -d "$HOME/.nvm/versions/node" ]]; then
  CODEX_BIN="$(find "$HOME/.nvm/versions/node" -type f -path '*/bin/codex' -print -quit 2>/dev/null)"
fi
if [[ -n "$CODEX_BIN" ]]; then
  export CODEX_CLI_PATH="$CODEX_BIN"
fi

cd "$PROJECT_DIR"

for PORT in "${CANDIDATE_PORTS[@]}"; do
  WEB_URL="http://127.0.0.1:$PORT"
  if /usr/bin/curl --silent --fail --max-time 1 "$WEB_URL/api/health" >/dev/null 2>&1; then
    open "$WEB_URL"
    exit 0
  fi
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  print "没有找到已安装的运行环境。请先按 README 完成一次安装。"
  print "按任意键关闭这个窗口。"
  read -k 1
  exit 1
fi

SELECTED_PORT=""
for PORT in "${CANDIDATE_PORTS[@]}"; do
  if ! /usr/sbin/lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    SELECTED_PORT="$PORT"
    break
  fi
done

if [[ -z "$SELECTED_PORT" ]]; then
  print "常用本机端口都被占用，无法启动研究助手。"
  print "请稍后重试，或在终端中手动指定一个端口。"
  print "按任意键关闭这个窗口。"
  read -k 1
  exit 1
fi

WEB_URL="http://127.0.0.1:$SELECTED_PORT"

print "正在启动研究助手…"
"$PYTHON_BIN" "$PROJECT_DIR/web_app.py" --host 127.0.0.1 --port "$SELECTED_PORT" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

for _ in {1..50}; do
  if /usr/bin/curl --silent --fail --max-time 1 "$WEB_URL/api/health" >/dev/null 2>&1; then
    open "$WEB_URL"
    print "研究助手已在浏览器打开（端口 $SELECTED_PORT）。保持此窗口开启即可继续运行。"
    wait "$SERVER_PID"
    exit $?
  fi
  sleep 0.2
done

print "研究助手没有正常启动，请查看上方的提示。"
print "按任意键关闭这个窗口。"
read -k 1
exit 1
