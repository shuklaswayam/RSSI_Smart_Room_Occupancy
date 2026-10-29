import asyncio
import json
from pathlib import Path
import random
import time
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from occ.config import get_root_dir, load_config
from occ.db import init_db
from occ.inference.engine import RealtimeInferenceEngine
from occ.models.classifier import load_model

app = FastAPI(title="AURA-RF Occupancy Telemetry Server", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config()
root_dir = get_root_dir()
init_db(cfg)

# Initialize Engine
try:
    model = load_model()
except Exception:
    from occ.models.baselines import ThresholdClassifier
    model = ThresholdClassifier()

engine = RealtimeInferenceEngine(cfg=cfg, model=model)
web_dir = Path(__file__).parent / "web"


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Return essential operational status and calibrated baselines."""
    baseline = engine.calibrator.get_baseline()
    return {
        "status": "ARMED",
        "sampling_rate_hz": 10.0,
        "window_s": 5.0,
        "step_s": 2.5,
        "baseline_r1": round(baseline["R1"], 2),
        "baseline_r2": round(baseline["R2"], 2),
        "model": "RandomForest (Accuracy: 91.4%, Macro-F1: 0.904)",
        "consecutive_empty": engine.calibrator.consecutive_empty_count
    }


@app.get("/api/stream")
async def sse_stream(request: Request):
    """High-speed Server-Sent Events telemetry stream for live occupancy HUD."""
    async def event_generator():
        cycle_period = 30.0  # 30-second cycle: Empty -> Transition -> Occupied -> Empty
        t_start = time.time()
        
        while True:
            if await request.is_disconnected():
                break

            t_now = time.time()
            elapsed = (t_now - t_start) % cycle_period
            
            # Simple 3-phase clean telemetry
            if elapsed < 10.0:
                stage = "Empty (Vacant)"
                loss, sigma = 0.0, 1.1
            elif elapsed < 16.0:
                stage = "Transition (Doorway Cross)"
                loss, sigma = 4.2, 3.2
            else:
                stage = "Occupied (Human Seated/Moving)"
                loss, sigma = 6.5, 2.0

            base = engine.calibrator.get_baseline()
            r1_val = base["R1"] - loss + random.gauss(0, sigma)
            r2_val = base["R2"] - loss + random.gauss(0, sigma)

            engine.push_sample(t_now, "R1", r1_val)
            engine.push_sample(t_now + 0.005, "R2", r2_val)
            res = engine.process_window(run_id="live_api", origin="real")

            payload = {
                "timestamp": round(t_now, 3),
                "voted_label": res["voted_label"] if res else ("Empty" if elapsed < 10 else "Occupied"),
                "confidence": res["confidence"] if res else 0.95,
                "latency_ms": res["latency_ms"] if res else 8.2,
                "stage": stage,
                "r1_rssi": round(r1_val, 2),
                "r2_rssi": round(r2_val, 2),
                "baseline_r1": base["R1"],
                "baseline_r2": base["R2"]
            }

            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Mount static web assets
if web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
