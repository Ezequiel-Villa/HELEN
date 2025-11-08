from pathlib import Path
from flask import Flask, request, jsonify, Response, render_template, stream_with_context
from flask_socketio import SocketIO
import webbrowser, threading, time, json, queue

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, template_folder=str(FRONTEND_DIR), static_folder=str(FRONTEND_DIR), static_url_path='')
app.config['SECRET_KEY'] = 'mysecret'
app.config['debug'] = True

# threading evita problemas de SSE en Windows
socket = SocketIO(cors_allowed_origins="*", async_mode="threading")

class SseBroker:
    def __init__(self):
        self._clients = set()
    def subscribe(self):
        q = queue.Queue(maxsize=64)
        self._clients.add(q)
        return q
    def unsubscribe(self, q):
        self._clients.discard(q)
    def publish(self, data: dict):
        dead = []
        for q in list(self._clients):
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

broker = SseBroker()

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.get('/')
def index():
    return render_template('index.html')

@app.get('/events')
def events():
    @stream_with_context
    def stream():
        q = broker.subscribe()
        try:
            yield _sse({"type": "hello", "status": "ok"})
            while True:
                data = q.get()
                yield _sse(data)
        finally:
            broker.unsubscribe(q)
    return Response(stream(), headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
    })

@app.get('/health')
def health():
    data = {"status": "ok"}
    socket.emit('message', data)
    broker.publish(data)
    return jsonify(data), 200

@app.post('/gestures/gesture-key')
def publish_gesture_key():
    data = request.json or {}
    print("POST /gestures/gesture-key:", data)
    socket.emit('message', data)
    broker.publish(data)
    return Response(status=200)

@app.get('/mode/get')
def mode_get():
    return jsonify({"active": "windows"}), 200

@app.post('/mode/set')
def mode_set():
    body = request.json or {}
    mode = str(body.get("mode", "windows"))
    return jsonify({"mode": mode}), 200

@app.get('/net/status')
def net_status():
    return jsonify({"online": True}), 200

@socket.on('connect')
def on_connect():
    print('client connected')

@socket.on('message')
def on_message(data):
    print(f'message: {data}')

def open_browser():
    time.sleep(1)
    webbrowser.open_new("http://localhost:5000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    socket.init_app(app, cors_allowed_origins="*")
    socket.run(app, host="127.0.0.1", port=5000, debug=True, use_reloader=False)
