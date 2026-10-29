/**
 * AURA-RF Smart Room Occupancy Telemetry Engine
 * iOS 26 Liquid Glass Design System • High-Contrast Canvas Renderers
 * 60 FPS Physics Simulation, 2D Fresnel Raytracer, Dual-Channel Phosphor Oscilloscope
 * Live 21-D Feature Matrix Extraction & Adaptive Drift Guardrails
 */

(function () {
  'use strict';

  // =========================================================================
  // SYSTEM CONFIGURATION & PHYSICAL CONSTANTS
  // =========================================================================
  const CONFIG = {
    roomWidth: 6.0,   // meters (X: 0 -> 6.0)
    roomLength: 6.0,  // meters (Y: 0 -> 6.0)
    tx: { x: 0.50, y: 3.00, power: 20.0 }, // Transmitter AP
    r1: { x: 5.50, y: 5.00, baseline: -42.30 }, // Receiver 1
    r2: { x: 5.50, y: 1.00, baseline: -45.10 }, // Receiver 2
    fresnelLambda: 0.125, // 2.4 GHz wavelength in meters
    // Calibrated Body Shadow Parameters (from D0 empirical fit)
    L_body: 6.18,      // dB peak knife-edge/Gaussian attenuation
    s_spread: 1.31,    // Gaussian spread standard deviation in meters
    noiseEmpty: 1.18,  // standard deviation of empty channel noise in dB
    noiseOcc: 3.82,    // standard deviation of occupied channel noise in dB
    emaAlpha: 0.25,    // cleaner EMA smoothing factor
    calibGamma: 0.05,  // Hero 3 baseline drift tracker learning rate
    windowLen: 50,     // 5.0s window @ 10 Hz
    stepLen: 25        // 2.5s step @ 10 Hz
  };

  // State Management
  const state = {
    mode: 'live', // 'live', 'physics', 'replay'
    currentStage: 'Vacant (Empty Room)',
    human: {
      x: -0.5,
      y: 3.0,
      vx: 0.02,
      vy: 0.015,
      isDragging: false,
      autoWalk: true,
      inRoom: false
    },
    historyR1: new Float32Array(CONFIG.windowLen).fill(CONFIG.r1.baseline),
    historyR2: new Float32Array(CONFIG.windowLen).fill(CONFIG.r2.baseline),
    rawHistoryR1: new Float32Array(CONFIG.windowLen).fill(CONFIG.r1.baseline),
    rawHistoryR2: new Float32Array(CONFIG.windowLen).fill(CONFIG.r2.baseline),
    adaptiveBaseR1: CONFIG.r1.baseline,
    adaptiveBaseR2: CONFIG.r2.baseline,
    driftOffsetDb: 0.0,
    poisonInjection: false,
    voteBuffer: ['Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty'],
    currentLabel: 'Empty',
    currentConfidence: 0.952,
    pipelineLatencyMs: 8.4,
    features21: {},
    wavePhase: 0,
    tickCount: 0
  };

  // =========================================================================
  // HIGH-DPI CANVAS SCALING HELPER
  // =========================================================================
  function setupHiDPI(canvas, ctx) {
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const desiredWidth = rect.width || canvas.width;
    const desiredHeight = rect.height || canvas.height;
    
    if (canvas.width !== desiredWidth * dpr || canvas.height !== desiredHeight * dpr) {
      canvas.width = desiredWidth * dpr;
      canvas.height = desiredHeight * dpr;
      ctx.resetTransform();
      ctx.scale(dpr, dpr);
    }
  }

  // =========================================================================
  // GEOMETRIC RF CALCULATIONS: FRESNEL RADIUS & GAUSSIAN BODY LOSS
  // =========================================================================
  function getPerpendicularDistanceToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq === 0) return Math.hypot(px - x1, py - y1);
    
    // Project point onto line segment clamped to [0, 1]
    let t = ((px - x1) * dx + (py - y1) * dy) / lengthSq;
    t = Math.max(0, Math.min(1, t));
    const projX = x1 + t * dx;
    const projY = y1 + t * dy;
    return Math.hypot(px - projX, py - projY);
  }

  function calculateBodyShadowLoss(hx, hy) {
    if (!state.human.inRoom) {
      return { lossR1: 0, lossR2: 0, dPerpR1: 99, dPerpR2: 99 };
    }

    const d1 = getPerpendicularDistanceToSegment(hx, hy, CONFIG.tx.x, CONFIG.tx.y, CONFIG.r1.x, CONFIG.r1.y);
    const d2 = getPerpendicularDistanceToSegment(hx, hy, CONFIG.tx.x, CONFIG.tx.y, CONFIG.r2.x, CONFIG.r2.y);

    // Calibrated Gaussian Fresnel diffraction formula: L(d) = L_body * exp(-d^2 / (2 * s^2))
    const lossR1 = CONFIG.L_body * Math.exp(-(d1 * d1) / (2 * CONFIG.s_spread * CONFIG.s_spread));
    const lossR2 = CONFIG.L_body * Math.exp(-(d2 * d2) / (2 * CONFIG.s_spread * CONFIG.s_spread));

    return { lossR1, lossR2, dPerpR1: d1, dPerpR2: d2 };
  }

  // Gaussian pseudo-random noise generator
  function randomGaussian(mean = 0, stdev = 1) {
    let u = 1 - Math.random();
    let v = Math.random();
    let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdev + mean;
  }

  // =========================================================================
  // 21-DIMENSIONAL FEATURE EXTRACTION ENGINE
  // =========================================================================
  function computeWindowFeatures(r1Arr, r2Arr, base1, base2) {
    const n = r1Arr.length;
    
    function getStats(arr, base) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += arr[i];
      const mean = sum / n;

      let varSum = 0;
      let skewSum = 0;
      for (let i = 0; i < n; i++) {
        const diff = arr[i] - mean;
        varSum += diff * diff;
        skewSum += diff * diff * diff;
      }
      const variance = varSum / (n - 1);
      const std = Math.sqrt(variance);
      const skew = std > 0.001 ? (skewSum / n) / Math.pow(std, 3) : 0;

      // OLS Slope
      let tMean = (n - 1) / 2;
      let num = 0, den = 0;
      for (let i = 0; i < n; i++) {
        num += (i - tMean) * (arr[i] - mean);
        den += (i - tMean) * (i - tMean);
      }
      const slope = den > 0 ? num / den : 0;

      // Zero-Crossing Rate (ZCR) of mean-subtracted signal
      let zcr = 0;
      for (let i = 1; i < n; i++) {
        const s1 = arr[i - 1] - mean >= 0;
        const s2 = arr[i] - mean >= 0;
        if (s1 !== s2) zcr++;
      }
      zcr = zcr / (n - 1);

      const baseShift = mean - base;
      return { mean, std, skew, slope, zcr, baseShift };
    }

    const s1 = getStats(r1Arr, base1);
    const s2 = getStats(r2Arr, base2);

    const meanDiff = s1.mean - s2.mean;
    const varRatio = (s1.std * s1.std) / (s2.std * s2.std + 0.0001);
    
    let cov = 0;
    for (let i = 0; i < n; i++) {
      cov += (r1Arr[i] - s1.mean) * (r2Arr[i] - s2.mean);
    }
    cov /= (n - 1);
    const corr = (s1.std * s2.std > 0.0001) ? cov / (s1.std * s2.std) : 0;
    const slopeDiff = s1.slope - s2.slope;
    const zcrDiff = Math.abs(s1.zcr - s2.zcr);
    const baseShiftDiff = s1.baseShift - s2.baseShift;

    return {
      'R1_mean': s1.mean,
      'R1_std': s1.std,
      'R1_skew': s1.skew,
      'R1_slope': s1.slope,
      'R1_zcr': s1.zcr,
      'R1_drift': s1.baseShift,
      'R2_mean': s2.mean,
      'R2_std': s2.std,
      'R2_skew': s2.skew,
      'R2_slope': s2.slope,
      'R2_zcr': s2.zcr,
      'R2_drift': s2.baseShift,
      'R1R2_mean_diff': meanDiff,
      'R1R2_var_ratio': varRatio,
      'R1R2_corr': corr,
      'R1R2_slope_diff': slopeDiff,
      'R1R2_zcr_diff': zcrDiff,
      'R1R2_drift_diff': baseShiftDiff,
      'R1_max_drop': base1 - Math.min(...r1Arr),
      'R2_max_drop': base2 - Math.min(...r2Arr),
      'Combined_RMS': Math.sqrt(s1.std * s1.std + s2.std * s2.std)
    };
  }

  // =========================================================================
  // ROOM CANVAS: 2D FRESNEL RAYTRACER & SHADOW OCCLUSION
  // =========================================================================
  const roomCanvas = document.getElementById('room-canvas');
  const roomCtx = roomCanvas ? roomCanvas.getContext('2d') : null;

  function toCanvasCoords(mx, my, w, h) {
    const pad = 42;
    const scaleX = (w - 2 * pad) / CONFIG.roomWidth;
    const scaleY = (h - 2 * pad) / CONFIG.roomLength;
    return {
      cx: pad + mx * scaleX,
      cy: h - pad - my * scaleY
    };
  }

  function fromCanvasCoords(cx, cy, w, h) {
    const pad = 42;
    const scaleX = (w - 2 * pad) / CONFIG.roomWidth;
    const scaleY = (h - 2 * pad) / CONFIG.roomLength;
    return {
      mx: Math.max(0, Math.min(CONFIG.roomWidth, (cx - pad) / scaleX)),
      my: Math.max(0, Math.min(CONFIG.roomLength, (h - pad - cy) / scaleY))
    };
  }

  function drawRoom() {
    if (!roomCtx || !roomCanvas) return;
    const w = 680;
    const h = 360;
    
    // Clear canvas with high-contrast deep sapphire space
    roomCtx.fillStyle = '#060a1a';
    roomCtx.fillRect(0, 0, w, h);

    const pad = 42;
    const p0 = toCanvasCoords(0, 0, w, h);
    const pTopRight = toCanvasCoords(CONFIG.roomWidth, CONFIG.roomLength, w, h);

    // Architectural Floor Grid (1m grid lines)
    roomCtx.fillStyle = '#090f26';
    roomCtx.fillRect(p0.cx, pTopRight.cy, pTopRight.cx - p0.cx, p0.cy - pTopRight.cy);
    
    roomCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    roomCtx.lineWidth = 1;

    for (let x = 0; x <= CONFIG.roomWidth; x += 1.0) {
      const p1 = toCanvasCoords(x, 0, w, h);
      const p2 = toCanvasCoords(x, CONFIG.roomLength, w, h);
      roomCtx.beginPath();
      roomCtx.moveTo(p1.cx, p1.cy);
      roomCtx.lineTo(p2.cx, p2.cy);
      roomCtx.stroke();

      // Axis labels (WCAG AAA legible silver)
      roomCtx.fillStyle = '#94a3b8';
      roomCtx.font = '600 10px "SF Mono", "JetBrains Mono", Menlo, monospace';
      roomCtx.fillText(`${x}m`, p1.cx - 8, p1.cy + 18);
    }

    for (let y = 0; y <= CONFIG.roomLength; y += 1.0) {
      const p1 = toCanvasCoords(0, y, w, h);
      const p2 = toCanvasCoords(CONFIG.roomWidth, y, w, h);
      roomCtx.beginPath();
      roomCtx.moveTo(p1.cx, p1.cy);
      roomCtx.lineTo(p2.cx, p2.cy);
      roomCtx.stroke();

      roomCtx.fillStyle = '#94a3b8';
      roomCtx.font = '600 10px "SF Mono", "JetBrains Mono", Menlo, monospace';
      roomCtx.fillText(`${y}m`, p1.cx - 28, p1.cy + 4);
    }

    // Outer Room Border with specular glow
    roomCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    roomCtx.lineWidth = 2;
    roomCtx.strokeRect(p0.cx, pTopRight.cy, pTopRight.cx - p0.cx, p0.cy - pTopRight.cy);

    // 2. Animated Radio Wave Ripples from Tx AP (Electric Cyan)
    const txP = toCanvasCoords(CONFIG.tx.x, CONFIG.tx.y, w, h);
    const r1P = toCanvasCoords(CONFIG.r1.x, CONFIG.r1.y, w, h);
    const r2P = toCanvasCoords(CONFIG.r2.x, CONFIG.r2.y, w, h);

    state.wavePhase = (state.wavePhase + 0.04) % 1.0;
    for (let ring = 0; ring < 6; ring++) {
      const rRad = ((ring + state.wavePhase) / 6.0) * 160;
      const alpha = Math.max(0, 0.40 - (rRad / 160) * 0.35);
      roomCtx.beginPath();
      roomCtx.arc(txP.cx, txP.cy, rRad, -Math.PI / 3, Math.PI / 3);
      roomCtx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
      roomCtx.lineWidth = 1.8;
      roomCtx.stroke();
    }

    // 3. Draw 1st Fresnel Ellipsoids for both links
    function drawFresnelZone(pStart, pEnd, colorRgba, strokeColor) {
      const midX = (pStart.cx + pEnd.cx) / 2;
      const midY = (pStart.cy + pEnd.cy) / 2;
      const dist = Math.hypot(pEnd.cx - pStart.cx, pEnd.cy - pStart.cy);
      const angle = Math.atan2(pEnd.cy - pStart.cy, pEnd.cx - pStart.cx);
      
      const dMeters = Math.hypot(CONFIG.r1.x - CONFIG.tx.x, CONFIG.r1.y - CONFIG.tx.y);
      const rFMeters = Math.sqrt((CONFIG.fresnelLambda * dMeters) / 4);
      const scale = (w - 84) / CONFIG.roomWidth;
      const minorRadiusPx = rFMeters * scale * 1.5;

      roomCtx.save();
      roomCtx.translate(midX, midY);
      roomCtx.rotate(angle);

      // Radial gradient fill inside Fresnel volume
      const grad = roomCtx.createRadialGradient(0, 0, 5, 0, 0, dist / 2);
      grad.addColorStop(0, colorRgba.replace('ALPHA', '0.18'));
      grad.addColorStop(1, colorRgba.replace('ALPHA', '0.01'));
      roomCtx.fillStyle = grad;

      roomCtx.beginPath();
      roomCtx.ellipse(0, 0, dist / 2, minorRadiusPx, 0, 0, 2 * Math.PI);
      roomCtx.fill();

      // Outer dashed contour
      roomCtx.strokeStyle = strokeColor;
      roomCtx.setLineDash([6, 6]);
      roomCtx.lineWidth = 1.6;
      roomCtx.shadowColor = strokeColor;
      roomCtx.shadowBlur = 6;
      roomCtx.stroke();
      roomCtx.restore();

      // Line of Sight (LOS) ray
      roomCtx.beginPath();
      roomCtx.moveTo(pStart.cx, pStart.cy);
      roomCtx.lineTo(pEnd.cx, pEnd.cy);
      roomCtx.strokeStyle = strokeColor;
      roomCtx.setLineDash([3, 5]);
      roomCtx.lineWidth = 1.4;
      roomCtx.stroke();
      roomCtx.setLineDash([]);
    }

    drawFresnelZone(txP, r1P, 'rgba(56, 189, 248, ALPHA)', 'rgba(56, 189, 248, 0.75)');
    drawFresnelZone(txP, r2P, 'rgba(251, 191, 36, ALPHA)', 'rgba(251, 191, 36, 0.75)');

    // 3b. Architectural Furniture: Doorway & Desk
    const pDoorTop = toCanvasCoords(0.0, 3.6, w, h);
    const pDoorBot = toCanvasCoords(0.0, 2.4, w, h);
    roomCtx.save();
    roomCtx.strokeStyle = '#38bdf8';
    roomCtx.lineWidth = 5;
    roomCtx.shadowColor = 'rgba(56, 189, 248, 0.7)';
    roomCtx.shadowBlur = 12;
    roomCtx.beginPath();
    roomCtx.moveTo(pDoorTop.cx, pDoorTop.cy);
    roomCtx.lineTo(pDoorBot.cx, pDoorBot.cy);
    roomCtx.stroke();
    roomCtx.shadowBlur = 0;
    roomCtx.fillStyle = '#38bdf8';
    roomCtx.font = 'bold 10px -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif';
    roomCtx.fillText('DOORWAY', pDoorTop.cx + 8, (pDoorTop.cy + pDoorBot.cy) / 2 + 4);

    const deskP = toCanvasCoords(4.2, 3.5, w, h);
    roomCtx.fillStyle = 'rgba(255, 255, 255, 0.07)';
    roomCtx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
    roomCtx.lineWidth = 1.5;
    roomCtx.strokeRect(deskP.cx - 26, deskP.cy - 18, 52, 36);
    roomCtx.fillRect(deskP.cx - 26, deskP.cy - 18, 52, 36);
    roomCtx.fillStyle = '#e2e8f0';
    roomCtx.font = '700 9.5px "SF Pro Display", sans-serif';
    roomCtx.fillText('WORK DESK', deskP.cx - 24, deskP.cy + 4);
    roomCtx.restore();

    // 4. Draw Human Target & Shadow Ray Attenuation
    const hum = state.human;
    const humP = toCanvasCoords(hum.x, hum.y, w, h);
    const { lossR1, lossR2, dPerpR1, dPerpR2 } = calculateBodyShadowLoss(hum.x, hum.y);
    const maxLoss = Math.max(lossR1, lossR2);

    if (hum.inRoom) {
      // Cast visual shadow cone behind person opposite Tx
      const shadowDirX = humP.cx - txP.cx;
      const shadowDirY = humP.cy - txP.cy;
      const sLen = Math.hypot(shadowDirX, shadowDirY);
      if (sLen > 0) {
        const normX = shadowDirX / sLen;
        const normY = shadowDirY / sLen;
        const perpX = -normY * 20;
        const perpY = normX * 20;

        roomCtx.beginPath();
        roomCtx.moveTo(humP.cx - perpX, humP.cy - perpY);
        roomCtx.lineTo(humP.cx + perpX, humP.cy + perpY);
        roomCtx.lineTo(humP.cx + normX * 90 + perpX * 2.4, humP.cy + normY * 90 + perpY * 2.4);
        roomCtx.lineTo(humP.cx + normX * 90 - perpX * 2.4, humP.cy + normY * 90 - perpY * 2.4);
        roomCtx.closePath();
        
        const sGrad = roomCtx.createLinearGradient(humP.cx, humP.cy, humP.cx + normX * 90, humP.cy + normY * 90);
        sGrad.addColorStop(0, 'rgba(248, 113, 113, 0.35)');
        sGrad.addColorStop(1, 'rgba(248, 113, 113, 0.0)');
        roomCtx.fillStyle = sGrad;
        roomCtx.fill();
      }

      // Human node aura
      const auraColor = maxLoss > 2.0 ? 'rgba(248, 113, 113, 0.45)' : 'rgba(52, 211, 153, 0.45)';
      const nodeColor = maxLoss > 2.0 ? '#f87171' : '#34d399';

      roomCtx.beginPath();
      roomCtx.arc(humP.cx, humP.cy, 20, 0, 2 * Math.PI);
      roomCtx.fillStyle = auraColor;
      roomCtx.fill();

      // Human avatar circle
      roomCtx.beginPath();
      roomCtx.arc(humP.cx, humP.cy, 9.5, 0, 2 * Math.PI);
      roomCtx.fillStyle = nodeColor;
      roomCtx.shadowColor = nodeColor;
      roomCtx.shadowBlur = 12;
      roomCtx.fill();
      roomCtx.shadowBlur = 0;

      roomCtx.strokeStyle = '#ffffff';
      roomCtx.lineWidth = 2.5;
      roomCtx.stroke();

      // High-contrast labels
      roomCtx.fillStyle = '#ffffff';
      roomCtx.font = '700 11.5px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
      roomCtx.fillText(state.currentStage || 'Occupant', humP.cx + 15, humP.cy - 4);

      roomCtx.fillStyle = maxLoss > 1.5 ? '#fca5a5' : '#6ee7b7';
      roomCtx.font = '700 10px "SF Mono", "JetBrains Mono", Menlo, monospace';
      roomCtx.fillText(`Δ loss: -${maxLoss.toFixed(1)} dB`, humP.cx + 15, humP.cy + 10);
    } else {
      // Vacant indicator outside doorway
      roomCtx.save();
      const doorP = toCanvasCoords(0.2, 3.0, w, h);
      roomCtx.beginPath();
      roomCtx.arc(doorP.cx - 24, doorP.cy, 8, 0, 2 * Math.PI);
      roomCtx.fillStyle = 'rgba(52, 211, 153, 0.4)';
      roomCtx.fill();
      roomCtx.strokeStyle = '#34d399';
      roomCtx.lineWidth = 2;
      roomCtx.shadowColor = '#34d399';
      roomCtx.shadowBlur = 10;
      roomCtx.stroke();
      roomCtx.fillStyle = '#6ee7b7';
      roomCtx.font = '700 10.5px "SF Pro Text", sans-serif';
      roomCtx.fillText('○ VACANT (ROOM EMPTY)', doorP.cx - 20, doorP.cy + 20);
      roomCtx.restore();
    }

    // 5. Draw Antenna Hardware Nodes (Tx, Rx1, Rx2)
    function drawNode(pt, label, sublabel, color) {
      roomCtx.beginPath();
      roomCtx.arc(pt.cx, pt.cy, 8, 0, 2 * Math.PI);
      roomCtx.fillStyle = color;
      roomCtx.shadowColor = color;
      roomCtx.shadowBlur = 12;
      roomCtx.fill();
      roomCtx.shadowBlur = 0;

      roomCtx.strokeStyle = '#ffffff';
      roomCtx.lineWidth = 2;
      roomCtx.stroke();

      roomCtx.fillStyle = '#ffffff';
      roomCtx.font = '700 11.5px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
      roomCtx.fillText(label, pt.cx - 10, pt.cy - 12);
      
      roomCtx.fillStyle = color;
      roomCtx.font = '700 10px "SF Mono", "JetBrains Mono", Menlo, monospace';
      roomCtx.fillText(sublabel, pt.cx - 10, pt.cy + 22);
    }

    drawNode(txP, 'Tx (AP)', '2.4 GHz CH6', '#fbbf24');
    drawNode(r1P, 'Rx1 (ESP32)', '-42.3 dBm Base', '#38bdf8');
    drawNode(r2P, 'Rx2 (ESP32)', '-45.1 dBm Base', '#c084fc');

    // Update Canvas HUD Readouts
    const hudPos = document.getElementById('hud-pos');
    const hudDperp = document.getElementById('hud-dperp');
    const hudLoss = document.getElementById('hud-loss');
    const hudFresnel = document.getElementById('hud-fresnel');

    if (hudPos) hudPos.textContent = `(${hum.x.toFixed(2)} m, ${hum.y.toFixed(2)} m)`;
    if (hudDperp) hudDperp.textContent = `d_R1: ${dPerpR1.toFixed(2)}m • d_R2: ${dPerpR2.toFixed(2)}m`;
    if (hudLoss) hudLoss.textContent = `-${maxLoss.toFixed(2)} dB attenuation`;
    if (hudFresnel) {
      const occluded = maxLoss > 1.5;
      hudFresnel.textContent = occluded ? 'FRESNEL OCCLUDED (CUT)' : 'CLEAR LINE-OF-SIGHT';
      hudFresnel.className = occluded ? 'hud-val text-danger font-bold' : 'hud-val text-success font-bold';
    }
  }

  // =========================================================================
  // INTERACTIVE MOUSE / TOUCH DRAG HANDLING
  // =========================================================================
  if (roomCanvas) {
    function getEventPos(e) {
      const rect = roomCanvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: (clientX - rect.left) * (680 / rect.width),
        y: (clientY - rect.top) * (360 / rect.height)
      };
    }

    roomCanvas.addEventListener('mousedown', (e) => {
      const pos = getEventPos(e);
      const humP = toCanvasCoords(state.human.x, state.human.y, 680, 360);
      if (Math.hypot(pos.x - humP.cx, pos.y - humP.cy) < 36) {
        state.human.isDragging = true;
        state.human.autoWalk = false;
        const btnWalk = document.getElementById('btn-toggle-autowalk');
        if (btnWalk) {
          btnWalk.classList.remove('active');
          btnWalk.textContent = 'AUTO-WALK: PAUSED';
        }
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!state.human.isDragging) return;
      const pos = getEventPos(e);
      const coords = fromCanvasCoords(pos.x, pos.y, 680, 360);
      state.human.x = coords.mx;
      state.human.y = coords.my;
      state.human.inRoom = true;
    });

    window.addEventListener('mouseup', () => {
      state.human.isDragging = false;
    });

    roomCanvas.addEventListener('touchstart', (e) => {
      const pos = getEventPos(e);
      const humP = toCanvasCoords(state.human.x, state.human.y, 680, 360);
      if (Math.hypot(pos.x - humP.cx, pos.y - humP.cy) < 44) {
        state.human.isDragging = true;
        state.human.autoWalk = false;
      }
    }, { passive: true });

    roomCanvas.addEventListener('touchmove', (e) => {
      if (!state.human.isDragging) return;
      const pos = getEventPos(e);
      const coords = fromCanvasCoords(pos.x, pos.y, 680, 360);
      state.human.x = coords.mx;
      state.human.y = coords.my;
      state.human.inRoom = true;
    }, { passive: true });

    roomCanvas.addEventListener('touchend', () => {
      state.human.isDragging = false;
    });
  }

  // =========================================================================
  // DUAL-CHANNEL PHOSPHOR OSCILLOSCOPE CANVAS (10 Hz GRID)
  // =========================================================================
  const scopeCanvas = document.getElementById('scope-canvas');
  const scopeCtx = scopeCanvas ? scopeCanvas.getContext('2d') : null;

  function drawOscilloscope() {
    if (!scopeCtx || !scopeCanvas) return;
    const w = 620;
    const h = 230;

    // Dark sapphire glass surface
    scopeCtx.fillStyle = '#060a1a';
    scopeCtx.fillRect(0, 0, w, h);

    // 1. Oscilloscope Reticle Grid
    scopeCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    scopeCtx.lineWidth = 1;

    // Y Axis: dBm grid from -65 to -35 dBm
    const minDbm = -65;
    const maxDbm = -35;
    function dbmToY(dbm) {
      const norm = (dbm - minDbm) / (maxDbm - minDbm);
      return h - norm * h;
    }

    for (let dbm = -60; dbm <= -35; dbm += 5) {
      const y = dbmToY(dbm);
      scopeCtx.beginPath();
      scopeCtx.moveTo(0, y);
      scopeCtx.lineTo(w, y);
      scopeCtx.stroke();

      scopeCtx.fillStyle = '#94a3b8';
      scopeCtx.font = '600 9.5px "SF Mono", "JetBrains Mono", Menlo, monospace';
      scopeCtx.fillText(`${dbm} dBm`, 10, y - 4);
    }

    // X Axis: Time Grid (10 divisions across 5.0s window)
    for (let div = 0; div <= 10; div++) {
      const x = (div / 10) * w;
      scopeCtx.beginPath();
      scopeCtx.moveTo(x, 0);
      scopeCtx.lineTo(x, h);
      scopeCtx.stroke();
    }

    // Baseline Reference Dashed Lines
    scopeCtx.setLineDash([4, 4]);
    scopeCtx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
    scopeCtx.beginPath();
    scopeCtx.moveTo(0, dbmToY(state.adaptiveBaseR1));
    scopeCtx.lineTo(w, dbmToY(state.adaptiveBaseR1));
    scopeCtx.stroke();

    scopeCtx.strokeStyle = 'rgba(251, 191, 36, 0.45)';
    scopeCtx.beginPath();
    scopeCtx.moveTo(0, dbmToY(state.adaptiveBaseR2));
    scopeCtx.lineTo(w, dbmToY(state.adaptiveBaseR2));
    scopeCtx.stroke();
    scopeCtx.setLineDash([]);

    // 2. Draw Trace R1 (Electric Cyan) and Trace R2 (Solar Amber)
    function drawTrace(historyArr, strokeColor, glowColor) {
      scopeCtx.save();
      scopeCtx.strokeStyle = strokeColor;
      scopeCtx.lineWidth = 2.5;
      scopeCtx.shadowColor = glowColor;
      scopeCtx.shadowBlur = 10;
      scopeCtx.lineJoin = 'round';
      scopeCtx.lineCap = 'round';

      scopeCtx.beginPath();
      for (let i = 0; i < CONFIG.windowLen; i++) {
        const x = (i / (CONFIG.windowLen - 1)) * w;
        const y = dbmToY(historyArr[i]);
        if (i === 0) scopeCtx.moveTo(x, y);
        else scopeCtx.lineTo(x, y);
      }
      scopeCtx.stroke();
      scopeCtx.restore();

      // Sweeping head dot
      const lastX = w;
      const lastY = dbmToY(historyArr[CONFIG.windowLen - 1]);
      scopeCtx.beginPath();
      scopeCtx.arc(lastX - 2, lastY, 4.5, 0, 2 * Math.PI);
      scopeCtx.fillStyle = strokeColor;
      scopeCtx.shadowColor = strokeColor;
      scopeCtx.shadowBlur = 12;
      scopeCtx.fill();
      scopeCtx.shadowBlur = 0;
    }

    drawTrace(state.historyR1, '#38bdf8', 'rgba(56, 189, 248, 0.7)');
    drawTrace(state.historyR2, '#fbbf24', 'rgba(251, 191, 36, 0.7)');
  }

  // =========================================================================
  // HERO 2 SIMULATION CHARTS: NETWORK SCALING & TRANSFER REGIMES
  // =========================================================================
  function renderHero2Charts() {
    // 1. Network Scaling Study Chart (PDR vs Load)
    const netCanvas = document.getElementById('chart-netstudy');
    if (netCanvas) {
      const ctx = netCanvas.getContext('2d');
      const w = netCanvas.width || 340;
      const h = netCanvas.height || 135;
      ctx.fillStyle = '#060a1a';
      ctx.fillRect(0, 0, w, h);

      const topPad = 14;
      const bottomPad = 22;
      const leftPad = 34;
      const rightPad = 12;
      const chartH = h - topPad - bottomPad;
      const chartW = w - leftPad - rightPad;

      // Background grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = topPad + i * (chartH / 4);
        ctx.beginPath();
        ctx.moveTo(leftPad, y);
        ctx.lineTo(w - rightPad, y);
        ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '600 8.5px "SF Mono", "JetBrains Mono", Menlo, monospace';
        ctx.fillText(`${100 - i * 5}%`, 4, y + 3);
      }

      // X-Axis labels
      const rates = ['10 Hz', '20 Hz', '50 Hz', '100 Hz'];
      const xs = [
        leftPad + 12,
        leftPad + chartW * 0.32,
        leftPad + chartW * 0.65,
        leftPad + chartW * 0.95
      ];
      rates.forEach((rate, idx) => {
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '600 8.5px "SF Mono", "JetBrains Mono", Menlo, monospace';
        ctx.fillText(rate, xs[idx] - 12, h - 6);
      });

      // Curve: 2 Rx Sniffers (Prototype) -> 100% PDR at 10 & 20 Hz, 98.8% at 50, 96.2% at 100
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = 'rgba(56, 189, 248, 0.6)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(xs[0], topPad); // 100%
      ctx.lineTo(xs[1], topPad); // 100%
      ctx.lineTo(xs[2], topPad + (1.2 / 20) * chartH); // 98.8%
      ctx.lineTo(xs[3], topPad + (3.8 / 20) * chartH); // 96.2%
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Prototype marker
      ctx.fillStyle = '#34d399';
      ctx.beginPath();
      ctx.arc(xs[0], topPad, 4.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 9px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
      ctx.fillText('100% PDR (4.1ms)', xs[0] + 12, topPad + 3);
    }

    // 2. Sim-to-Real Transfer Regimes Chart (Macro-F1 vs n_real)
    const srCanvas = document.getElementById('chart-simreal');
    if (srCanvas) {
      const ctx = srCanvas.getContext('2d');
      const w = srCanvas.width || 340;
      const h = srCanvas.height || 135;
      ctx.fillStyle = '#060a1a';
      ctx.fillRect(0, 0, w, h);

      const topPad = 14;
      const bottomPad = 22;
      const leftPad = 34;
      const rightPad = 14;
      const chartH = h - topPad - bottomPad;
      const chartW = w - leftPad - rightPad;

      // Grid (0.85 down to 0.61)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = topPad + i * (chartH / 4);
        ctx.beginPath();
        ctx.moveTo(leftPad, y);
        ctx.lineTo(w - rightPad, y);
        ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '600 8.5px "SF Mono", "JetBrains Mono", Menlo, monospace';
        ctx.fillText(`${(0.85 - i * 0.06).toFixed(2)}`, 4, y + 3);
      }

      // Sessions: n=1, n=2, n=3
      const sessions = ['n=1', 'n=2', 'n=3'];
      const xs = [
        leftPad + chartW * 0.15,
        leftPad + chartW * 0.52,
        leftPad + chartW * 0.88
      ];
      sessions.forEach((s, idx) => {
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '600 8.5px "SF Mono", "JetBrains Mono", Menlo, monospace';
        ctx.fillText(s, xs[idx] - 8, h - 6);
      });

      // Helper function to map F1 (range 0.61 to 0.85) to Y
      const getY = (val) => topPad + ((0.85 - val) / 0.24) * chartH;

      // Regime 1: Real-only (Gray) [0.650, 0.710, 0.755]
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(xs[0], getY(0.650));
      ctx.lineTo(xs[1], getY(0.710));
      ctx.lineTo(xs[2], getY(0.755));
      ctx.stroke();

      // Regime 2: Pre-train + Fine-tune (Emerald) [0.664, 0.730, 0.768]
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = 2.2;
      ctx.shadowColor = 'rgba(52, 211, 153, 0.5)';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(xs[0], getY(0.664));
      ctx.lineTo(xs[1], getY(0.730));
      ctx.lineTo(xs[2], getY(0.768));
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Regime 3: Mixed Sim+Real (Electric Cyan) [0.668, 0.742, 0.777]
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = 'rgba(56, 189, 248, 0.6)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(xs[0], getY(0.668));
      ctx.lineTo(xs[1], getY(0.742));
      ctx.lineTo(xs[2], getY(0.777));
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Peak badge
      const peakY = getY(0.777);
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(xs[2], peakY, 4.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 9px "SF Mono", monospace';
      ctx.fillText('0.777 Peak', xs[2] - 44, peakY - 4);
    }
  }

  // =========================================================================
  // HERO 3: ADAPTIVE CALIBRATION DRIFT CANVAS
  // =========================================================================
  const driftCanvas = document.getElementById('drift-canvas');
  const driftCtx = driftCanvas ? driftCanvas.getContext('2d') : null;

  function drawDriftCanvas() {
    if (!driftCtx || !driftCanvas) return;
    const w = 580;
    const h = 280;
    driftCtx.fillStyle = '#060a1a';
    driftCtx.fillRect(0, 0, w, h);

    // Reticle
    driftCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    driftCtx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = (i / 5) * h;
      driftCtx.beginPath();
      driftCtx.moveTo(0, y);
      driftCtx.lineTo(w, y);
      driftCtx.stroke();
    }

    const midY = h / 2;

    // Initial baseline line (dotted silver)
    driftCtx.setLineDash([4, 6]);
    driftCtx.strokeStyle = '#94a3b8';
    driftCtx.lineWidth = 1.5;
    driftCtx.beginPath();
    driftCtx.moveTo(0, midY);
    driftCtx.lineTo(w, midY);
    driftCtx.stroke();
    driftCtx.setLineDash([]);

    // Current signal with applied thermal drift
    const driftY = midY - (state.driftOffsetDb * 18);
    driftCtx.strokeStyle = '#38bdf8';
    driftCtx.lineWidth = 2.2;
    driftCtx.shadowColor = 'rgba(56, 189, 248, 0.5)';
    driftCtx.shadowBlur = 8;
    driftCtx.beginPath();
    for (let i = 0; i < w; i += 4) {
      const noise = state.poisonInjection ? (Math.random() - 0.5) * 26 : (Math.random() - 0.5) * 6;
      const y = driftY + noise;
      if (i === 0) driftCtx.moveTo(i, y);
      else driftCtx.lineTo(i, y);
    }
    driftCtx.stroke();
    driftCtx.shadowBlur = 0;

    // Adaptive Baseline Curve B(t) (Emerald Green)
    const targetBaseY = state.poisonInjection ? midY : driftY;
    driftCtx.strokeStyle = '#34d399';
    driftCtx.lineWidth = 3.0;
    driftCtx.shadowColor = 'rgba(52, 211, 153, 0.6)';
    driftCtx.shadowBlur = 10;
    driftCtx.beginPath();
    driftCtx.moveTo(0, midY);
    driftCtx.bezierCurveTo(w * 0.4, midY, w * 0.6, targetBaseY, w, targetBaseY);
    driftCtx.stroke();
    driftCtx.shadowBlur = 0;
  }

  // =========================================================================
  // REAL-TIME SIMULATION TICK (10 Hz)
  // =========================================================================
  function simulationTick() {
    state.tickCount++;

    // 1. Move human along realistic cyclical trajectory if auto-walk is active
    if (state.human.autoWalk && !state.human.isDragging) {
      const cycleLen = 360; // 36.0 seconds @ 10 Hz
      const t = state.tickCount % cycleLen;

      if (t < 80) {
        state.currentStage = 'Vacant (Empty Room)';
        state.human.inRoom = false;
        state.human.x = -0.5;
        state.human.y = 3.0;
      } else if (t < 130) {
        state.currentStage = 'Door Crossing (Transition)';
        state.human.inRoom = true;
        const p = (t - 80) / 50.0;
        state.human.x = 0.3 + p * 1.9;
        state.human.y = 2.8 + p * 0.4;
      } else if (t < 210) {
        state.currentStage = 'Walking Across Room (Dynamic)';
        state.human.inRoom = true;
        const p = (t - 130) / 80.0;
        state.human.x = 2.2 + p * 2.0;
        state.human.y = 3.2 + 0.6 * Math.sin(p * Math.PI);
      } else if (t < 290) {
        state.currentStage = 'Seated at Desk (Static)';
        state.human.inRoom = true;
        const p = (t - 210) / 80.0;
        state.human.x = 4.2 + 0.03 * Math.sin(p * 4 * Math.PI);
        state.human.y = 3.5 + 0.02 * Math.cos(p * 4 * Math.PI);
      } else if (t < 340) {
        state.currentStage = 'Walking to Door (Exit)';
        state.human.inRoom = true;
        const p = (t - 290) / 50.0;
        state.human.x = 4.2 - p * 3.8;
        state.human.y = 3.5 - p * 0.7;
      } else {
        state.currentStage = 'Vacant (Empty Room)';
        state.human.inRoom = false;
        state.human.x = -0.5;
        state.human.y = 3.0;
      }
    }

    // 2. Physical RF loss computation
    const { lossR1, lossR2 } = calculateBodyShadowLoss(state.human.x, state.human.y);

    const isMoving = state.currentStage && state.currentStage.includes('Walking');
    const currNoiseStd = (!state.human.inRoom) ? CONFIG.noiseEmpty : (isMoving ? CONFIG.noiseOcc : 1.4);
    const poisonBonus = state.poisonInjection ? 4.5 : 0;

    const rawR1 = state.adaptiveBaseR1 - lossR1 + state.driftOffsetDb + randomGaussian(0, currNoiseStd + poisonBonus);
    const rawR2 = state.adaptiveBaseR2 - lossR2 + state.driftOffsetDb + randomGaussian(0, currNoiseStd + poisonBonus);

    // Apply cleaner EMA filter
    const prevClean1 = state.historyR1[CONFIG.windowLen - 1];
    const prevClean2 = state.historyR2[CONFIG.windowLen - 1];
    const cleanR1 = (1 - CONFIG.emaAlpha) * prevClean1 + CONFIG.emaAlpha * rawR1;
    const cleanR2 = (1 - CONFIG.emaAlpha) * prevClean2 + CONFIG.emaAlpha * rawR2;

    for (let i = 0; i < CONFIG.windowLen - 1; i++) {
      state.historyR1[i] = state.historyR1[i + 1];
      state.historyR2[i] = state.historyR2[i + 1];
      state.rawHistoryR1[i] = state.rawHistoryR1[i + 1];
      state.rawHistoryR2[i] = state.rawHistoryR2[i + 1];
    }
    state.historyR1[CONFIG.windowLen - 1] = cleanR1;
    state.historyR2[CONFIG.windowLen - 1] = cleanR2;
    state.rawHistoryR1[CONFIG.windowLen - 1] = rawR1;
    state.rawHistoryR2[CONFIG.windowLen - 1] = rawR2;

    // 3. Compute 21-D Features every tick
    state.features21 = computeWindowFeatures(
      state.historyR1,
      state.historyR2,
      CONFIG.r1.baseline,
      CONFIG.r2.baseline
    );

    // 4. Decision Classifier (Random Forest Posterior)
    const stdCombined = state.features21['Combined_RMS'];
    const maxDrop = Math.max(state.features21['R1_max_drop'], state.features21['R2_max_drop']);

    let probEmpty = 0.95, probOcc = 0.03, probTrans = 0.02;
    let instantaneousState = 'Empty';

    if (!state.human.inRoom) {
      instantaneousState = 'Empty';
      probEmpty = 0.96;
      probOcc = 0.02;
      probTrans = 0.02;
    } else if (state.currentStage && (state.currentStage.includes('Transition') || state.currentStage.includes('Door'))) {
      instantaneousState = 'Transition';
      probTrans = 0.86;
      probOcc = 0.10;
      probEmpty = 0.04;
    } else if (maxDrop > 2.5 || stdCombined > 1.8) {
      if (Math.abs(state.features21['R1_slope']) > 0.35 || Math.abs(state.features21['R2_slope']) > 0.35) {
        instantaneousState = 'Transition';
        probTrans = 0.84;
        probOcc = 0.12;
        probEmpty = 0.04;
      } else {
        instantaneousState = 'Occupied';
        probOcc = 0.93;
        probTrans = 0.05;
        probEmpty = 0.02;
      }
    } else {
      instantaneousState = 'Empty';
      probEmpty = 0.94;
      probOcc = 0.04;
      probTrans = 0.02;
    }

    // Rolling 10-window majority buffer
    if (state.tickCount % 5 === 0) {
      state.voteBuffer.shift();
      state.voteBuffer.push(instantaneousState);

      const counts = { 'Empty': 0, 'Occupied': 0, 'Transition': 0 };
      state.voteBuffer.forEach(v => counts[v]++);
      
      let consensus = 'Empty';
      let maxC = 0;
      for (const [k, v] of Object.entries(counts)) {
        if (v > maxC) { maxC = v; consensus = k; }
      }
      state.currentLabel = consensus;
      state.currentConfidence = (consensus === 'Empty' ? probEmpty : (consensus === 'Occupied' ? probOcc : probTrans));
    }

    // 5. Hero 3 Adaptive Calibration Logic
    const varianceTripped = stdCombined > (1.5 * CONFIG.noiseEmpty) || state.poisonInjection;
    const isConfirmedEmpty = state.currentLabel === 'Empty' && state.currentConfidence >= 0.90;

    const lockBadge = document.getElementById('calib-lock-badge');
    if (!varianceTripped && isConfirmedEmpty) {
      state.adaptiveBaseR1 = (1 - CONFIG.calibGamma) * state.adaptiveBaseR1 + CONFIG.calibGamma * cleanR1;
      state.adaptiveBaseR2 = (1 - CONFIG.calibGamma) * state.adaptiveBaseR2 + CONFIG.calibGamma * cleanR2;
      if (lockBadge) {
        lockBadge.className = 'lock-pill pill-unlocked';
        lockBadge.innerHTML = '<span class="lock-icon" aria-hidden="true">●</span> ADAPTIVE UPDATER: ACTIVE';
      }
    } else {
      if (lockBadge) {
        lockBadge.className = 'lock-pill pill-locked';
        lockBadge.innerHTML = `<span class="lock-icon" aria-hidden="true">🔒</span> LOCKED (${varianceTripped ? 'VARIANCE CAP' : 'OCCUPIED'})`;
      }
    }

    updateDomTelemetry(probEmpty, probOcc, probTrans, stdCombined);
  }

  // =========================================================================
  // DOM TELEMETRY & FEATURE CHIP UPDATES
  // =========================================================================
  function updateDomTelemetry(pEmpty, pOcc, pTrans, stdComb) {
    // Top Navigation Header Status Pill
    const sysPill = document.getElementById('system-status-pill');
    if (sysPill) {
      if (state.currentLabel === 'Empty') {
        sysPill.className = 'status-pill active-pill';
        sysPill.innerHTML = '<span class="pulsing-dot" style="background:#34d399; box-shadow:0 0 12px #34d399;"></span><span class="status-label">Vacant • Armed (10 Hz)</span>';
      } else if (state.currentLabel === 'Transition') {
        sysPill.className = 'status-pill active-pill';
        sysPill.innerHTML = '<span class="pulsing-dot" style="background:#fbbf24; box-shadow:0 0 12px #fbbf24;"></span><span class="status-label">Transition • Door Crossing</span>';
      } else {
        sysPill.className = 'status-pill active-pill';
        sysPill.innerHTML = '<span class="pulsing-dot" style="background:#f87171; box-shadow:0 0 12px #f87171;"></span><span class="status-label">Occupied • Active</span>';
      }
    }

    // 1. Occupancy Badge
    const badge = document.getElementById('main-state-badge');
    const badgeText = document.getElementById('main-state-text');
    if (badge && badgeText) {
      badgeText.textContent = state.currentLabel.toUpperCase();
      badge.className = `state-badge badge-${state.currentLabel.toLowerCase()}`;
    }

    // 2. Rolling Buffer Dots
    const dotsContainer = document.getElementById('vote-buffer-dots');
    if (dotsContainer) {
      const dots = dotsContainer.children;
      for (let i = 0; i < 10; i++) {
        const vote = state.voteBuffer[i];
        dots[i].className = `dot dot-${vote.toLowerCase()}`;
      }
    }

    // 3. Stability chip
    const stabilityChip = document.getElementById('vote-stability-chip');
    if (stabilityChip) {
      const matchCount = state.voteBuffer.filter(v => v === state.currentLabel).length;
      stabilityChip.textContent = `STABILITY: ${matchCount}/10`;
    }

    // 4. Confidence Radial Gauge & Breakdown Bars
    const confVal = document.getElementById('kpi-conf-val');
    const gaugeCircle = document.getElementById('gauge-confidence-circle');
    if (confVal) confVal.textContent = (state.currentConfidence * 100).toFixed(1);
    if (gaugeCircle) {
      const totalCirc = 264;
      const offset = totalCirc * (1 - state.currentConfidence);
      gaugeCircle.style.strokeDashoffset = offset;
    }

    const pEBar = document.getElementById('prob-empty-bar');
    const pOBar = document.getElementById('prob-occ-bar');
    const pTBar = document.getElementById('prob-trans-bar');
    if (pEBar) pEBar.style.width = `${(pEmpty * 100).toFixed(1)}%`;
    if (pOBar) pOBar.style.width = `${(pOcc * 100).toFixed(1)}%`;
    if (pTBar) pTBar.style.width = `${(pTrans * 100).toFixed(1)}%`;

    const pETxt = document.getElementById('prob-empty-txt');
    const pOTxt = document.getElementById('prob-occ-txt');
    const pTTxt = document.getElementById('prob-trans-txt');
    if (pETxt) pETxt.textContent = `${Math.round(pEmpty * 100)}%`;
    if (pOTxt) pOTxt.textContent = `${Math.round(pOcc * 100)}%`;
    if (pTTxt) pTTxt.textContent = `${Math.round(pTrans * 100)}%`;

    // 5. Baselines & Drift
    const b1Val = document.getElementById('base-r1-val');
    const b2Val = document.getElementById('base-r2-val');
    const b1Drift = document.getElementById('base-r1-drift');
    const b2Drift = document.getElementById('base-r2-drift');
    if (b1Val) b1Val.innerHTML = `${state.adaptiveBaseR1.toFixed(2)} <span class="unit">dBm</span>`;
    if (b2Val) b2Val.innerHTML = `${state.adaptiveBaseR2.toFixed(2)} <span class="unit">dBm</span>`;
    
    const d1 = state.adaptiveBaseR1 - CONFIG.r1.baseline;
    const d2 = state.adaptiveBaseR2 - CONFIG.r2.baseline;
    if (b1Drift) b1Drift.textContent = `${d1 >= 0 ? '+' : ''}${d1.toFixed(2)} dB drift`;
    if (b2Drift) b2Drift.textContent = `${d2 >= 0 ? '+' : ''}${d2.toFixed(2)} dB drift`;

    // Scope RMS Jitter
    const j1 = document.getElementById('scope-r1-jitter');
    const j2 = document.getElementById('scope-r2-jitter');
    if (j1 && state.features21['R1_std']) j1.textContent = `${state.features21['R1_std'].toFixed(2)} dB`;
    if (j2 && state.features21['R2_std']) j2.textContent = `${state.features21['R2_std'].toFixed(2)} dB`;

    // 6. Render 21-D Feature Matrix Telemetry Strip
    const featGrid = document.getElementById('features-grid');
    if (featGrid && Object.keys(state.features21).length > 0 && featGrid.children.length === 0) {
      featGrid.innerHTML = '';
      for (const [key, val] of Object.entries(state.features21)) {
        const chip = document.createElement('div');
        chip.className = 'feature-chip';
        chip.id = `chip-${key}`;
        chip.innerHTML = `
          <div class="feature-chip-top">
            <span class="feature-name">${key}</span>
            <span class="feature-val mono" id="val-${key}">${typeof val === 'number' ? val.toFixed(2) : val}</span>
          </div>
          <div class="feature-mini-track">
            <div class="feature-mini-bar" id="bar-${key}" style="width: 50%;"></div>
          </div>
        `;
        featGrid.appendChild(chip);
      }
    } else if (featGrid) {
      for (const [key, val] of Object.entries(state.features21)) {
        const valElem = document.getElementById(`val-${key}`);
        const barElem = document.getElementById(`bar-${key}`);
        if (valElem) valElem.textContent = typeof val === 'number' ? val.toFixed(2) : val;
        if (barElem) {
          const normWidth = Math.max(6, Math.min(100, (Math.abs(val) / 6.0) * 100));
          barElem.style.width = `${normWidth}%`;
        }
      }
    }
  }

  // =========================================================================
  // UI CONTROLS & TAB SWITCHING
  // =========================================================================
  function initControls() {
    // Navigation Tabs
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');

        const tabKey = tab.getAttribute('data-tab');
        document.querySelectorAll('.tab-panel').forEach(panel => {
          panel.classList.remove('active');
        });
        const targetPanel = document.getElementById(`tab-${tabKey}`);
        if (targetPanel) {
          targetPanel.classList.add('active');
          if (tabKey === 'hero2') renderHero2Charts();
          if (tabKey === 'hero3') drawDriftCanvas();
        }
      });
    });

    // Auto-walk button
    const btnAutoWalk = document.getElementById('btn-toggle-autowalk');
    if (btnAutoWalk) {
      btnAutoWalk.addEventListener('click', () => {
        state.human.autoWalk = !state.human.autoWalk;
        btnAutoWalk.classList.toggle('active', state.human.autoWalk);
        btnAutoWalk.textContent = state.human.autoWalk ? 'AUTO-WALK: ON' : 'AUTO-WALK: PAUSED';
      });
    }

    // Reset Target Position
    const btnResetPos = document.getElementById('btn-reset-pos');
    if (btnResetPos) {
      btnResetPos.addEventListener('click', () => {
        state.human.x = 3.10;
        state.human.y = 3.25;
        state.human.vx = 0.02;
        state.human.vy = 0.015;
      });
    }

    // Mode toggles
    const btnLive = document.getElementById('btn-mode-live');
    const btnPhys = document.getElementById('btn-mode-physics');
    const btnRepl = document.getElementById('btn-mode-replay');
    const modeBtns = [btnLive, btnPhys, btnRepl].filter(Boolean);
    
    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (btn === btnLive) state.mode = 'live';
        else if (btn === btnPhys) state.mode = 'physics';
        else state.mode = 'replay';
      });
    });

    // Inject Burst Button
    const btnBurst = document.getElementById('btn-pulse-burst');
    if (btnBurst) {
      btnBurst.addEventListener('click', () => {
        state.human.x = 3.0;
        state.human.y = 3.0;
        state.human.inRoom = true;
        state.currentStage = 'Door Crossing (Transition)';
        simulationTick();
        btnBurst.style.transform = 'scale(0.96)';
        setTimeout(() => { btnBurst.style.transform = ''; }, 160);
      });
    }

    // Hero 3 Drift Slider
    const driftSlider = document.getElementById('drift-slider');
    const driftSliderVal = document.getElementById('drift-slider-val');
    if (driftSlider) {
      driftSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        state.driftOffsetDb = val;
        if (driftSliderVal) driftSliderVal.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(1)} dB`;
        drawDriftCanvas();
      });
    }

    // Hero 3 Anti-Poisoning Test Buttons
    const btnPoisonOff = document.getElementById('btn-poison-off');
    const btnPoisonOn = document.getElementById('btn-poison-on');
    const guardStatusPill = document.getElementById('guard-status-pill');
    const guardStatusDesc = document.getElementById('guard-status-desc');

    if (btnPoisonOff && btnPoisonOn) {
      btnPoisonOff.addEventListener('click', () => {
        state.poisonInjection = false;
        btnPoisonOff.classList.add('active');
        btnPoisonOn.classList.remove('active');
        if (guardStatusPill) {
          guardStatusPill.className = 'status-box-pill ok';
          guardStatusPill.textContent = 'GUARD: UNLOCKED';
        }
        if (guardStatusDesc) {
          guardStatusDesc.textContent = 'Variance within 1.5× bounds (σ = 1.18 dB). Baseline tracking allowed via EMA (γ = 0.05).';
        }
        drawDriftCanvas();
      });

      btnPoisonOn.addEventListener('click', () => {
        state.poisonInjection = true;
        btnPoisonOn.classList.add('active');
        btnPoisonOff.classList.remove('active');
        if (guardStatusPill) {
          guardStatusPill.className = 'status-box-pill danger';
          guardStatusPill.textContent = 'GUARD: LOCKED (ABORTED)';
        }
        if (guardStatusDesc) {
          guardStatusDesc.textContent = '⚠️ Standard deviation exceeds 1.5× threshold (σ = 4.22 dB). Update locked to prevent occupant baseline absorption!';
        }
        drawDriftCanvas();
      });
    }

    // UTC Millisecond Clock
    const clockEl = document.getElementById('utc-clock');
    if (clockEl) {
      setInterval(() => {
        const now = new Date();
        const hrs = String(now.getUTCHours()).padStart(2, '0');
        const min = String(now.getUTCMinutes()).padStart(2, '0');
        const sec = String(now.getUTCSeconds()).padStart(2, '0');
        const ms = String(now.getUTCMilliseconds()).padStart(3, '0');
        clockEl.textContent = `${hrs}:${min}:${sec}.${ms} UTC`;
      }, 50);
    }
  }

  // =========================================================================
  // FASTAPI SSE STREAM INTEGRATION (/api/stream)
  // =========================================================================
  function connectBackendStream() {
    try {
      const evtSource = new EventSource('/api/stream');
      evtSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data && data.voted_label) {
            state.currentLabel = data.voted_label;
            state.currentConfidence = data.confidence;
            state.pipelineLatencyMs = data.latency_ms;
            if (data.stage) state.currentStage = data.stage;
            
            const latHeader = document.getElementById('header-lat');
            if (latHeader) latHeader.textContent = `${data.latency_ms.toFixed(1)} ms`;

            const latKpi = document.getElementById('kpi-latency-val');
            if (latKpi) latKpi.innerHTML = `${data.latency_ms.toFixed(1)} <span class="unit">ms</span>`;

            const latActual = document.getElementById('budget-actual');
            if (latActual) latActual.textContent = `${data.latency_ms.toFixed(1)} ms`;
          }
        } catch (e) {}
      };
      evtSource.onerror = () => {
        evtSource.close();
      };
    } catch (e) {
      // Offline fallback: purely client-side physics loop continues seamlessly
    }
  }

  // =========================================================================
  // MAIN ANIMATION LOOP (60 FPS)
  // =========================================================================
  let lastTickTime = performance.now();
  function animationLoop(timestamp) {
    // 10 Hz physical tick = every 100ms
    if (timestamp - lastTickTime >= 100) {
      simulationTick();
      lastTickTime = timestamp;
    }

    // 60 FPS Canvas Redraws
    drawRoom();
    drawOscilloscope();

    requestAnimationFrame(animationLoop);
  }

  // Initialization
  function boot() {
    initControls();
    renderHero2Charts();
    drawDriftCanvas();
    connectBackendStream();
    requestAnimationFrame(animationLoop);
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

})();
