import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from backendConexion import post_gesturekey
from helpers import labels_dict as helpers_labels
import threading
import queue

class BiGRUClassifier(nn.Module):
    """Clasificador secuencial con GRU (opcionalmente bidireccional) y cabeza lineal."""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.2, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(d * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(d * hidden_dim, num_classes)
        )

    def forward(self, x):
        """Procesa la secuencia y devuelve logits usando el último estado temporal."""
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def load_checkpoint(paths):
    """Carga el modelo y metadatos desde el primer checkpoint disponible."""
    for p in paths:
        p = Path(p)
        if p.exists():
            state = torch.load(p, map_location="cpu")
            cfg, meta = state["config"], state.get("meta", {})
            model = BiGRUClassifier(
                cfg["input_dim"], cfg["hidden_dim"], cfg["num_layers"],
                cfg["num_classes"], cfg["dropout"], cfg["bidirectional"]
            )
            model.load_state_dict(state["model_state"])
            return model, meta
    raise FileNotFoundError("No checkpoint found.")


def normalize_xy(pts_xy, eps=1e-6):
    """Normaliza coordenadas XY a [0,1] por marco para ser invariante a escala/traslación."""
    mn, mx = pts_xy.min(axis=0), pts_xy.max(axis=0)
    wh = np.maximum(mx - mn, eps)
    return (pts_xy - mn) / wh


def parse_mp_results(res):
    """Convierte la salida de MediaPipe Hands en listas con etiqueta, confianza y puntos."""
    out = []
    if not res or not res.multi_hand_landmarks or not res.multi_handedness:
        return out
    for lm, cls in zip(res.multi_hand_landmarks, res.multi_handedness):
        pts = np.array([[p.x, p.y, getattr(p, 'z', 0.0)] for p in lm.landmark], dtype=np.float32)
        out.append({
            "label": cls.classification[0].label,
            "score": float(cls.classification[0].score),
            "pts": pts
        })
    return out


def draw_pts(frame, pts):
    """Dibuja los landmarks y conexiones de la mano sobre el frame de video."""
    lms = [landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        landmark_pb2.NormalizedLandmarkList(landmark=lms),
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style()
    )


class LandmarkEMA:
    """Suaviza landmarks por mano usando media exponencial móvil."""
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.buf = {"Left": None, "Right": None}

    def apply(self, sides):
        """Aplica suavizado EMA independiente para mano izquierda y derecha."""
        out = {"Left": None, "Right": None}
        for k in ("Left", "Right"):
            pts, prev = sides[k], self.buf[k]
            if pts is None:
                out[k] = None
                self.buf[k] = None
            elif prev is None:
                out[k] = pts.copy()
                self.buf[k] = pts.copy()
            else:
                sm = self.alpha * pts + (1 - self.alpha) * prev
                out[k] = sm
                self.buf[k] = sm
        return out


class GestureSender:
    """Envía gestos al backend de forma asíncrona con cola acotada y enfriamiento por etiqueta."""
    def __init__(self, maxsize=4, cooldown=1.0, verbose=True):
        self.q = queue.Queue(maxsize=maxsize)
        self.cooldown = cooldown
        self.verbose = verbose
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._last_sent = None
        self._last_sent_t = 0.0
        self._thread.start()

    def send(self, label):
        """Encola un gesto para envío sin bloquear el hilo principal."""
        try:
            self.q.put_nowait(label)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
                self.q.put_nowait(label)
            except Exception:
                pass

    def _worker(self):
        """Consume la cola y llama a post_gesturekey respetando el cooldown."""
        while self._running:
            try:
                label = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            now = time.time()
            if label == self._last_sent and (now - self._last_sent_t) < self.cooldown:
                try:
                    self.q.task_done()
                except Exception:
                    pass
                continue
            try:
                code = post_gesturekey(label)
                if self.verbose:
                    print(f"Detectado (async): {label} (status={code})")
            except Exception as e:
                if self.verbose:
                    print(f"post_gesturekey error (async): {e}")
            self._last_sent = label
            self._last_sent_t = now
            try:
                self.q.task_done()
            except Exception:
                pass

    def stop(self, wait=1.0):
        """Detiene el hilo trabajador de envíos asíncronos."""
        self._running = False
        self._thread.join(timeout=wait)


def assign_stable_sides(hands_list, prev_centers, close_thresh=0.06):
    """Asigna de forma estable cada mano a Left/Right basándose en historial y posición."""
    sides = {"Left": None, "Right": None}
    centers = {"Left": None, "Right": None}
    n = len(hands_list)
    if n == 0:
        return sides, centers
    if n == 2:
        cur = [np.array([h["pts"][:, 0].mean(), h["pts"][:, 1].mean()]) for h in hands_list]
        dist_between = np.linalg.norm(cur[0] - cur[1])
        if prev_centers.get("Left") is not None and prev_centers.get("Right") is not None and dist_between < close_thresh:
            mapped = {"Left": None, "Right": None}
            prevL = np.array(prev_centers["Left"])
            prevR = np.array(prev_centers["Right"])
            d0L = np.linalg.norm(cur[0] - prevL)
            d0R = np.linalg.norm(cur[0] - prevR)
            if d0L <= d0R:
                mapped["Left"] = hands_list[0]["pts"]
                mapped["Right"] = hands_list[1]["pts"]
            else:
                mapped["Left"] = hands_list[1]["pts"]
                mapped["Right"] = hands_list[0]["pts"]
            sides = mapped
        else:
            mapped = {"Left": None, "Right": None}
            for h in hands_list:
                if h["label"] in mapped and mapped[h["label"]] is None and h["score"] >= 0.7:
                    mapped[h["label"]] = h["pts"]
            if mapped["Left"] is None or mapped["Right"] is None:
                xs = [h["pts"][:, 0].mean() for h in hands_list]
                iL = int(np.argmin(xs))
                iR = 1 - iL
                mapped["Left"] = hands_list[iL]["pts"]
                mapped["Right"] = hands_list[iR]["pts"]
            sides = mapped
    else:
        h = hands_list[0]
        cx = h["pts"][:, 0].mean()
        if prev_centers.get("Left") is None and prev_centers.get("Right") is None:
            if h["label"] == "Left" and h["score"] >= 0.7:
                sides["Left"] = h["pts"]
            elif h["label"] == "Right" and h["score"] >= 0.7:
                sides["Right"] = h["pts"]
            else:
                sides["Left" if cx < 0.5 else "Right"] = h["pts"]
        else:
            dL = np.inf if prev_centers.get("Left") is None else abs(cx - prev_centers["Left"][0])
            dR = np.inf if prev_centers.get("Right") is None else abs(cx - prev_centers["Right"][0])
            sides["Left" if dL <= dR else "Right"] = h["pts"]
    for k, pts in sides.items():
        centers[k] = None if pts is None else (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    return sides, centers


def extract_features_from_sides(sides, use_z=False):
    """Genera el vector de características plano por mano (izquierda/derecha) y su máscara."""
    F_per_hand = 63 if use_z else 42
    left = right = np.zeros(F_per_hand, dtype=np.float32)
    mleft = mright = 0.0
    for lbl, pts in sides.items():
        if pts is None:
            continue
        nxy = normalize_xy(pts[:, :2])
        if use_z:
            z = (pts[:, 2:3] - pts[:, 2:3].mean()) / (pts[:, 2:3].std() + 1e-6)
            feat = np.concatenate([nxy, z], axis=1).reshape(-1).astype(np.float32)
        else:
            feat = nxy.reshape(-1).astype(np.float32)
        if lbl == "Left":
            left, mleft = feat, 1.0
        else:
            right, mright = feat, 1.0
    fvec = np.concatenate([left, right], axis=0)
    mask = np.array([mleft, mright], dtype=np.float32)
    return fvec, mask


def entropy(probs, eps=1e-8):
    """Calcula la entropía (incertidumbre) de una distribución de probabilidades."""
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def hand_area(pts):
    """Aproxima el área ocupada por la mano usando el rectángulo delimitador normalizado."""
    if pts is None:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float((x.max() - x.min()) * (y.max() - y.min()))


def motion_energy(prev_sides, cur_sides):
    """Estima el movimiento medio de los landmarks entre dos frames."""
    if prev_sides is None or cur_sides is None:
        return 0.0
    tot = 0.0
    cnt = 0
    for k in ("Left", "Right"):
        a, b = prev_sides.get(k), cur_sides.get(k)
        if a is None or b is None:
            continue
        d = np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean()
        tot += d
        cnt += 1
    return 0.0 if cnt == 0 else float(tot / cnt)


def main():
    """Ejecuta el pipeline completo: captura, detección, features, inferencia y envío de gestos."""
    model, meta = load_checkpoint(["model/models/sequence_bigru.pt", "models/sequence_bigru.pt", "/mnt/data/sequence_bigru.pt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    classes_numeric = meta.get("classes_numeric", [])
    idx_to_label = {i: helpers_labels.get(orig_val, str(orig_val)) for i, orig_val in enumerate(classes_numeric)}

    seq_len = int(meta.get("seq_len", 30))
    use_z = bool(meta.get("use_z", False))
    F_per_hand = 63 if use_z else 42
    F_total = 2 * F_per_hand

    # Umbrales/heurísticas (ajustables)
    presence_enter, presence_exit = 0.85, 0.70
    prob_enter, prob_exit = 0.95, 0.75
    margin_enter, margin_exit = 0.40, 0.18
    ent_enter, ent_exit = 1.70, 1.95
    area_single, area_total = 0.025, 0.040
    mot_enter, mot_exit = 0.015, 0.005
    smooth_k, min_consec = 7, 4
    send_cooldown = 1.0
    redetect_patience, force_redetect_every = 8, 120

    # Mejoras para rapidez y precisión:
    # - Allow inference on partial sequences (faster detection) but require stronger confidence/stability
    min_seq_for_run = max(6, seq_len // 2)  # run model when we have at least this many frames
    prob_buffer_size = 7  # smooth over more outputs to reduce flicker
    early_p1 = 0.995  # very high prob threshold to allow early trigger
    early_margin = 0.5
    early_min_consec = 2  # need this many recent high-confidence entries for early trigger
    min_hand_conf_to_accept = 0.7  # require this mean hand detection confidence for acceptance
    # Evitar repeticiones: cooldown por etiqueta y silenciar prints si se desea
    per_label_cooldown = 2.0  # segundos mínimo entre envíos de la misma etiqueta
    sender_verbose = False

    # Cámara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass
    if not cap.isOpened():
        return

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands_trk = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        model_complexity=1, min_detection_confidence=0.6,
        min_tracking_confidence=0.55
    )
    hands_det = mp_hands.Hands(
        static_image_mode=True, max_num_hands=2,
        model_complexity=1, min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Buffers
    window_feat = deque(maxlen=seq_len)
    window_mask = deque(maxlen=seq_len)
    smooth_preds = deque(maxlen=smooth_k)
    prob_buffer = deque(maxlen=prob_buffer_size)
    window_hand_conf = deque(maxlen=seq_len)
    motion_buffer = deque(maxlen=seq_len)
    ema = LandmarkEMA(alpha=0.35)

    prev_centers = {"Left": None, "Right": None}
    lost_counter = 0
    frame_idx = 0
    last_sent, last_sent_t = None, 0.0

    NONE_NAME = "none_names"
    active_label = NONE_NAME
    presence_ok = False
    prev_sides_draw = None
    prev_active_label = NONE_NAME

    sender = GestureSender(maxsize=4, cooldown=send_cooldown, verbose=sender_verbose)
    # track per-label last sent times to prevent rapid repeats
    label_last_sent_times = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = hands_trk.process(rgb)
        hands_list = parse_mp_results(res)
        frame_conf = float(np.mean([h["score"] for h in hands_list])) if len(hands_list) > 0 else 0.0
        window_hand_conf.append(frame_conf)

        need_redetect = (len(hands_list) < 2 and lost_counter >= redetect_patience) or (frame_idx % force_redetect_every == 0)
        if need_redetect:
            res2 = hands_det.process(rgb)
            cand = parse_mp_results(res2)
            if len(cand) >= len(hands_list):
                hands_list = cand

        sides, centers = assign_stable_sides(hands_list, prev_centers)
        prev_centers = centers
        has_any = (sides["Left"] is not None) or (sides["Right"] is not None)
        lost_counter = 0 if has_any else (lost_counter + 1)

        sides_smooth = ema.apply(sides)

        if sides_smooth["Left"] is not None:
            draw_pts(frame, sides_smooth["Left"])
        if sides_smooth["Right"] is not None:
            draw_pts(frame, sides_smooth["Right"])

        if not has_any:
            window_feat.append(np.zeros(F_total, dtype=np.float32))
            window_mask.append(np.zeros(2, dtype=np.float32))
        else:
            fvec, mvec = extract_features_from_sides(sides_smooth, use_z=use_z)
            window_feat.append(fvec.astype(np.float32))
            window_mask.append(mvec.astype(np.float32))

        aL = hand_area(sides_smooth["Left"])
        aR = hand_area(sides_smooth["Right"])
        area_ok = (aL >= area_single) or (aR >= area_single) or ((aL + aR) >= area_total)
        mot = motion_energy(prev_sides_draw, {"Left": sides_smooth["Left"], "Right": sides_smooth["Right"]})
        motion_buffer.append(mot)
        prev_sides_draw = {
            "Left": None if sides_smooth["Left"] is None else sides_smooth["Left"].copy(),
            "Right": None if sides_smooth["Right"] is None else sides_smooth["Right"].copy()
        }

        # Run inference when we have at least a small window (partial sequence allowed)
        if len(window_feat) >= min_seq_for_run:
            seq_len_used = min(len(window_feat), seq_len)
            feats = np.stack(window_feat)[-seq_len_used:]
            masks = np.stack(window_mask)[-seq_len_used:]

            # presence: fraction of frames with at least one hand
            presence = (masks.sum(axis=1) > 0).mean()
            presence_ok = (presence_ok and presence >= presence_exit) or (not presence_ok and presence >= presence_enter)

            if not presence_ok or not area_ok:
                active_label = NONE_NAME
                smooth_preds.clear()
                prob_buffer.clear()
                cv2.putText(frame, "...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
            else:
                # Build mask feature tensor for the sequence used
                M_feat_seq = np.concatenate([
                    np.repeat(masks[:, 0:1], F_per_hand, axis=1),
                    np.repeat(masks[:, 1:2], F_per_hand, axis=1)
                ], axis=1)
                X = feats[None, ...].astype(np.float32)  # (1, T, F_total)
                M_feat = M_feat_seq[None, ...].astype(np.float32)
                X = (X * M_feat)

                with torch.no_grad():
                    out = model(torch.from_numpy(X).to(device))
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]

                prob_buffer.append(probs)
                avg_probs = np.mean(np.stack(prob_buffer), axis=0)

                p1 = float(avg_probs.max())
                top1 = int(avg_probs.argmax())
                p2 = float(np.partition(avg_probs, -2)[-2]) if avg_probs.size > 1 else 0.0
                H = entropy(avg_probs)
                margin = p1 - p2

                avg_motion = float(np.mean(motion_buffer)) if len(motion_buffer) > 0 else 0.0

                feats_stack = np.stack(window_feat)
                feats_var = float(feats_stack.std(axis=0).mean())
                avg_hand_conf = float(np.mean(window_hand_conf)) if len(window_hand_conf) > 0 else 0.0

                motion_static_thresh = 0.003
                var_static_thresh = 0.0008

                is_static = (avg_motion < motion_static_thresh) and (feats_var < var_static_thresh)

                # Strong enter condition (used when enough frames available)
                enter_ok = (p1 >= prob_enter) and (margin >= margin_enter) and (H <= ent_enter) \
                           and (avg_motion >= mot_enter) and ((aL + aR) >= area_total)

                # If the sequence is static we demand very high prob/margin and good hand confidence
                if is_static:
                    if not (p1 >= 0.995 and margin >= 0.5 and avg_hand_conf >= 0.8):
                        enter_ok = False

                # Relaxed stay condition for continuing the active label
                stay_ok = (p1 >= prob_exit) and (margin >= margin_exit) and (H <= ent_exit) and (avg_motion >= mot_exit)

                # Helper: check stability in recent prob buffer
                def recent_top_consensus(pb, idx, required):
                    if len(pb) < required:
                        return False
                    last = list(pb)[-required:]
                    return all(int(p.argmax()) == idx for p in last)

                # Early trigger: allow a faster detection if very confident and stable
                early_trigger = False
                if (seq_len_used < seq_len) and (p1 >= early_p1) and (margin >= early_margin) and (avg_hand_conf >= min_hand_conf_to_accept):
                    if recent_top_consensus(prob_buffer, top1, early_min_consec):
                        early_trigger = True

                if active_label == NONE_NAME:
                    if enter_ok or early_trigger:
                        # If early_trigger is True, still require short smoothing via smooth_preds
                        smooth_preds.append(top1)
                        if len(smooth_preds) >= min_consec:
                            vote = max(set(smooth_preds), key=smooth_preds.count)
                            if list(smooth_preds)[-min_consec:].count(vote) == min_consec:
                                # Check entropy and hand confidence once more before committing
                                if (H <= ent_enter) and (avg_hand_conf >= min_hand_conf_to_accept):
                                    active_label = idx_to_label.get(vote, str(vote))
                                    smooth_preds.clear()
                                    prob_buffer.clear()
                    else:
                        smooth_preds.clear()
                else:
                    if not stay_ok:
                        active_label = NONE_NAME

                if active_label != NONE_NAME:
                    display_conf = float(p1)
                    cv2.putText(
                        frame, f"{active_label} ({display_conf:.2f})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
                    )
                    now = time.time()
                    # Enviar cuando la etiqueta se activa (transición NONE -> label).
                    # Esto permite repetir la misma seña si antes se dejó de detectar y se vuelve a hacer.
                    if prev_active_label == NONE_NAME and active_label != NONE_NAME:
                        sender.send(active_label)
                        last_sent, last_sent_t = active_label, now
                        label_last_sent_times[active_label] = now
                else:
                    cv2.putText(frame, "...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)

        # cv2.imshow("Inference", frame)
        # record previous active label for next iteration (used to detect transitions)
        # prev_active_label = active_label
        # frame_idx += 1
        # if (cv2.waitKey(1) & 0xFF) == 27:
        #     break

    cap.release()
    cv2.destroyAllWindows()
    try:
        sender.stop()
    except Exception:
        pass
    try:
        hands_trk.close()
        hands_det.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
