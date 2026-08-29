#!/bin/bash
# -----------------------------------------------------------
# FA 어드민 & 마켓 백엔드 서버 관리 스크립트
# 사용법: ./manage_fa_admin.sh {start|stop|restart|status}
# -----------------------------------------------------------

APP_DIR="/volume3/homes/vividian/Drive/Obsidian/blog"
PYTHON_EXEC="$APP_DIR/.venv/bin/python"
SCRIPT_PATH="$APP_DIR/scripts/fa_admin_server.py"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/fa_admin.log"
PID_FILE="$LOG_DIR/fa_admin.pid"
PORT=8095

mkdir -p "$LOG_DIR"

get_pid() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    # 프로세스 목록에서 검색
    local p_pid=$(pgrep -f "$SCRIPT_PATH" | head -n 1)
    if [ -n "$p_pid" ]; then
        echo "$p_pid"
        return 0
    fi
    return 1
}

start() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo "✅ FA 백엔드 서버가 이미 실행 중입니다 (PID: $pid, Port: $PORT)."
        return 0
    fi

    echo "🚀 FA 백엔드 서버를 시작합니다 (Port: $PORT)..."
    cd "$APP_DIR" || exit 1
    nohup "$PYTHON_EXEC" "$SCRIPT_PATH" --port "$PORT" >> "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    sleep 1

    if kill -0 "$new_pid" 2>/dev/null; then
        echo "✨ 시작 완료! (PID: $new_pid, Port: $PORT)"
    else
        echo "❌ 시작 실패. 로그 파일($LOG_FILE)을 확인하세요."
    fi
}

stop() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo "🛑 FA 백엔드 서버를 중지합니다 (PID: $pid)..."
        kill "$pid" 2>/dev/null
        sleep 1
        kill -9 "$pid" 2>/dev/null
        rm -f "$PID_FILE"
        echo "정상 종료되었습니다."
    else
        echo "실행 중인 서버 프로세스가 없습니다."
        rm -f "$PID_FILE"
    fi
}

status() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo "🟢 FA 백엔드 서버 실행 중 (PID: $pid, Port: $PORT)"
    else
        echo "🔴 FA 백엔드 서버가 중지되어 있습니다."
    fi
}

case "$1" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    *)       echo "사용법: $0 {start|stop|restart|status}" ;;
esac
