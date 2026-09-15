/**
 * web/js/canvas_renderer.js
 * High-Performance HTML5 Canvas Renderer for SUMO Roundabout Simulation
 * Coordinate Space: SUMO world coordinates (0-400, 0-400), center (200, 200)
 */

class RoundaboutCanvasRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    
    // Viewport transforms (world to canvas)
    this.worldSize = 400.0;
    this.worldCenter = { x: 200.0, y: 200.0 };
    this.scale = 1.8; // px per meter
    this.panOffset = { x: 0, y: 0 };
    this.followEgo = false;
    
    // Trajectory history for ego
    this.egoTrail = [];
    this.maxTrailLength = 45;
    
    // Mouse drag state
    this.isDragging = false;
    this.lastMousePos = { x: 0, y: 0 };
    
    this.vehicles = [];
    this.egoVehicle = null;
    this.telemetry = null;

    this.initCanvasSize();
    this.setupEventListeners();
    this.resetView();
  }

  initCanvasSize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.displaySize = { width: rect.width, height: rect.height };
  }

  setupEventListeners() {
    window.addEventListener('resize', () => this.initCanvasSize());

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastMousePos.x;
      const dy = e.clientY - this.lastMousePos.y;
      this.panOffset.x += dx;
      this.panOffset.y += dy;
      this.lastMousePos = { x: e.clientX, y: e.clientY };
      this.followEgo = false;
      this.render();
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      this.scale = Math.max(0.6, Math.min(5.0, this.scale * zoomFactor));
      this.render();
    }, { passive: false });
  }

  resetView() {
    this.scale = (this.displaySize.height / 300.0);
    this.panOffset = {
      x: this.displaySize.width / 2,
      y: this.displaySize.height / 2
    };
    this.followEgo = false;
    this.render();
  }

  centerOnEgo() {
    this.followEgo = true;
    if (this.egoVehicle) {
      this.panOffset.x = this.displaySize.width / 2 - (this.egoVehicle.x - 200) * this.scale;
      this.panOffset.y = this.displaySize.height / 2 - (200 - this.egoVehicle.y) * this.scale;
    }
    this.render();
  }

  worldToCanvas(wx, wy) {
    // SUMO: (0,0) bottom-left, (400,400) top-right. Center at (200,200).
    const cx = this.panOffset.x + (wx - 200.0) * this.scale;
    const cy = this.panOffset.y - (wy - 200.0) * this.scale;
    return { x: cx, y: cy };
  }

  updateData(vehicles, telemetry) {
    this.vehicles = vehicles || [];
    this.telemetry = telemetry || null;
    this.egoVehicle = this.vehicles.find(v => v.is_ego) || null;

    if (this.egoVehicle) {
      this.egoTrail.push({ x: this.egoVehicle.x, y: this.egoVehicle.y });
      if (this.egoTrail.length > this.maxTrailLength) {
        this.egoTrail.shift();
      }

      if (this.followEgo) {
        const targetX = this.displaySize.width / 2 - (this.egoVehicle.x - 200) * this.scale;
        const targetY = this.displaySize.height / 2 - (200 - this.egoVehicle.y) * this.scale;
        // Smooth interpolation
        this.panOffset.x += (targetX - this.panOffset.x) * 0.15;
        this.panOffset.y += (targetY - this.panOffset.y) * 0.15;
      }
    } else {
      this.egoTrail = [];
    }

    this.render();
  }

  render() {
    const ctx = this.ctx;
    const w = this.displaySize.width;
    const h = this.displaySize.height;

    // Clear canvas
    ctx.fillStyle = '#060913';
    ctx.fillRect(0, 0, w, h);

    // Draw grid background
    this.drawGrid();

    // Draw roundabout road geometry
    this.drawRoadNetwork();

    // Draw sensor rays and ego trajectory trail
    this.drawPerceptionAndTrail();

    // Draw all vehicles
    this.drawVehicles();
  }

  drawGrid() {
    const ctx = this.ctx;
    const gridSize = 20 * this.scale;
    if (gridSize < 10) return;

    ctx.save();
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.35)';
    ctx.lineWidth = 0.5;

    const startX = this.panOffset.x % gridSize;
    const startY = this.panOffset.y % gridSize;

    ctx.beginPath();
    for (let x = startX; x < this.displaySize.width; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.displaySize.height);
    }
    for (let y = startY; y < this.displaySize.height; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(this.displaySize.width, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  drawRoadNetwork() {
    const ctx = this.ctx;
    const center = this.worldToCanvas(200, 200);
    const ringRadius = 30 * this.scale;
    const laneWidth = 7.5 * this.scale;
    const armLength = 170 * this.scale;

    ctx.save();

    // 1. Draw 4 Arm Asphalt Corridors (North, South, East, West)
    ctx.fillStyle = '#141a29';
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1.5;

    // North Arm (x=200, y=200 to 400)
    const nTop = this.worldToCanvas(200, 400);
    ctx.fillRect(center.x - laneWidth, nTop.y, laneWidth * 2, (center.y - nTop.y));
    // South Arm (x=200, y=200 to 0)
    const sBot = this.worldToCanvas(200, 0);
    ctx.fillRect(center.x - laneWidth, center.y, laneWidth * 2, (sBot.y - center.y));
    // West Arm (x=0 to 200, y=200)
    const wLeft = this.worldToCanvas(0, 200);
    ctx.fillRect(wLeft.x, center.y - laneWidth, (center.x - wLeft.x), laneWidth * 2);
    // East Arm (x=200 to 400, y=200)
    const eRight = this.worldToCanvas(400, 200);
    ctx.fillRect(center.x, center.y - laneWidth, (eRight.x - center.x), laneWidth * 2);

    // 2. Draw Circulating Ring Asphalt
    ctx.beginPath();
    ctx.arc(center.x, center.y, ringRadius + laneWidth, 0, Math.PI * 2);
    ctx.fillStyle = '#141a29';
    ctx.fill();

    // 3. Central Island (Green Grass & Concrete Curb)
    ctx.beginPath();
    ctx.arc(center.x, center.y, ringRadius - laneWidth * 0.7, 0, Math.PI * 2);
    ctx.fillStyle = '#0a2318';
    ctx.fill();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Island inner ring
    ctx.beginPath();
    ctx.arc(center.x, center.y, ringRadius * 0.55, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 4. Circulating Lane Center Dashed Line
    ctx.beginPath();
    ctx.arc(center.x, center.y, ringRadius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([8, 8]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 5. Arm Divider Markings (Yellow double lines / white dashes)
    ctx.strokeStyle = '#eab308';
    ctx.lineWidth = 1.5;
    // North Arm center divider
    ctx.beginPath();
    ctx.moveTo(center.x, nTop.y);
    ctx.lineTo(center.x, center.y - ringRadius - laneWidth);
    ctx.stroke();
    // South Arm center divider
    ctx.beginPath();
    ctx.moveTo(center.x, center.y + ringRadius + laneWidth);
    ctx.lineTo(center.x, sBot.y);
    ctx.stroke();
    // West Arm center divider
    ctx.beginPath();
    ctx.moveTo(wLeft.x, center.y);
    ctx.lineTo(center.x - ringRadius - laneWidth, center.y);
    ctx.stroke();
    // East Arm center divider
    ctx.beginPath();
    ctx.moveTo(center.x + ringRadius + laneWidth, center.y);
    ctx.lineTo(eRight.x, center.y);
    ctx.stroke();

    // 6. Yield Line Markings at North Entry (Merge Point)
    const nYield = this.worldToCanvas(198, 235.34);
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.8)';
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(nYield.x - laneWidth * 0.9, nYield.y);
    ctx.lineTo(nYield.x, nYield.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Merge Zone Highlight Indicator Box
    ctx.fillStyle = 'rgba(245, 158, 11, 0.08)';
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    const mzTop = this.worldToCanvas(198, 260);
    const mzBot = this.worldToCanvas(198, 230);
    ctx.strokeRect(center.x - laneWidth, mzTop.y, laneWidth, (mzBot.y - mzTop.y));
    ctx.fillRect(center.x - laneWidth, mzTop.y, laneWidth, (mzBot.y - mzTop.y));
    ctx.setLineDash([]);

    // Merge Zone Text
    ctx.fillStyle = 'rgba(245, 158, 11, 0.7)';
    ctx.font = `600 ${Math.max(9, 10 * this.scale * 0.5)}px Inter, sans-serif`;
    ctx.fillText('MERGE ZONE (30m)', center.x - laneWidth * 2.6, (mzTop.y + mzBot.y) / 2);

    ctx.restore();
  }

  drawPerceptionAndTrail() {
    const ctx = this.ctx;

    // Draw Ego Trajectory Trail
    if (this.egoTrail.length > 1) {
      ctx.save();
      for (let i = 0; i < this.egoTrail.length - 1; i++) {
        const p1 = this.worldToCanvas(this.egoTrail[i].x, this.egoTrail[i].y);
        const p2 = this.worldToCanvas(this.egoTrail[i + 1].x, this.egoTrail[i + 1].y);
        const alpha = (i / this.egoTrail.length) * 0.7;

        ctx.strokeStyle = `rgba(244, 63, 94, ${alpha})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Draw Sensor Ray from Ego to Nearest Circulating Vehicle
    if (this.egoVehicle && this.telemetry && this.telemetry.nearest_circ_dist < 45.0) {
      // Find nearest circulating vehicle in vehicles list
      const circVeh = this.vehicles.find(v => !v.is_ego && v.lane && v.lane.includes('circ'));
      if (circVeh) {
        ctx.save();
        const egoPos = this.worldToCanvas(this.egoVehicle.x, this.egoVehicle.y);
        const targetPos = this.worldToCanvas(circVeh.x, circVeh.y);

        ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(egoPos.x, egoPos.y);
        ctx.lineTo(targetPos.x, targetPos.y);
        ctx.stroke();

        // Distance Tag
        const midX = (egoPos.x + targetPos.x) / 2;
        const midY = (egoPos.y + targetPos.y) / 2;
        ctx.fillStyle = '#38bdf8';
        ctx.font = '600 10px JetBrains Mono';
        ctx.fillText(`Gap: ${this.telemetry.gap_size.toFixed(1)}m | TTC: ${this.telemetry.ttc.toFixed(1)}s`, midX + 5, midY - 5);
        ctx.restore();
      }
    }
  }

  drawVehicles() {
    const ctx = this.ctx;
    const carLength = 4.5 * this.scale;
    const carWidth = 2.0 * this.scale;

    for (const v of this.vehicles) {
      const pos = this.worldToCanvas(v.x, v.y);
      // SUMO angle: 0 is North (up), 90 is East (right), clockwise.
      // Canvas rotation: 0 is East (right), clockwise.
      // Canvas rad = (angle - 90) * (PI / 180).
      const rad = (v.angle - 90) * (Math.PI / 180);

      ctx.save();
      ctx.translate(pos.x, pos.y);
      ctx.rotate(rad);

      // Color scheme based on vehicle type
      let bodyColor = '#38bdf8'; // HDV
      let glowColor = 'rgba(56, 189, 248, 0.4)';
      let label = 'HDV';

      if (v.is_ego) {
        bodyColor = '#f43f5e';
        glowColor = 'rgba(244, 63, 94, 0.7)';
        label = 'EGO (PPO)';
      } else if (v.type === 'av') {
        bodyColor = '#10b981';
        glowColor = 'rgba(16, 185, 129, 0.5)';
        label = 'AV';
      }

      // Draw Vehicle Glow (for ego)
      if (v.is_ego) {
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 14;
      }

      // Car Body
      ctx.fillStyle = bodyColor;
      ctx.beginPath();
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(-carLength / 2, -carWidth / 2, carLength, carWidth, 2);
      } else {
        ctx.rect(-carLength / 2, -carWidth / 2, carLength, carWidth);
      }
      ctx.fill();

      // Front Headlights
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(carLength / 2 - 1, -carWidth / 2 + 0.5, 1.5, 1.5);
      ctx.fillRect(carLength / 2 - 1, carWidth / 2 - 2, 1.5, 1.5);

      // Windshield
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.fillRect(-carLength * 0.1, -carWidth * 0.4, carLength * 0.4, carWidth * 0.8);

      ctx.restore();

      // Vehicle Tag
      const speedStr = (typeof v.speed === 'number') ? v.speed.toFixed(1) : '0.0';
      if (v.is_ego || this.scale > 1.4) {
        ctx.save();
        ctx.font = '700 9px Inter, sans-serif';
        ctx.fillStyle = bodyColor;
        ctx.textAlign = 'center';
        ctx.fillText(`${label} ${speedStr}m/s`, pos.x, pos.y - carWidth - 6);
        ctx.restore();
      }
    }
  }
}

window.RoundaboutCanvasRenderer = RoundaboutCanvasRenderer;
