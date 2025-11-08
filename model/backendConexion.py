import requests

SERVER = "http://127.0.0.1:5000/gestures/gesture-key"

def post_gesturekey(payload: dict):
    try:
        r = requests.post(SERVER, json=payload, timeout=1.5)
        return r.status_code
    except Exception as e:
        print("post_gesturekey error:", e)
        return 0
