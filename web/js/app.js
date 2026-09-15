/**
 * web/js/app.js
 * Main Application Controller for Roundabout RL Web Dashboard
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize subsystems
  const renderer = new RoundaboutCanvasRenderer('simCanvas');
  const charts = new TelemetryChartsManager();

  // State
  let ws = null;
  let isSimRunning = false;
  let isSimPaused = false;
  let activeModel = 'final_best_agent.zip';
  let activePreset = 'MEDIUM';
  let lastFrameTime = performance.now();
  let fps = 20.0;

  // DOM Elements
  const headerActiveModel = document.getElementById('header-active-model');
  const headerGpuStatus = document.getElementById('header-gpu-status');
  const statusSimMode = document.getElementById('status-sim-mode');
  
  const btnSimPlay = document.getElementById('btn-sim-play');
  const btnSimPlayText = document.getElementById('btn-sim-play-text');
  const btnSimPause = document.getElementById('btn-sim-pause');
  const btnSimRestart = document.getElementById('btn-sim-restart');
  const btnCenterEgo = document.getElementById('btn-center-ego');
  const btnZoomIn = document.getElementById('btn-zoom-in');
  const btnZoomOut = document.getElementById('btn-zoom-out');
  const btnZoomReset = document.getElementById('btn-zoom-reset');

  const sliderSpeedMult = document.getElementById('slider-speed-mult');
  const valSpeedMult = document.getElementById('val-speed-mult');
  const sliderHdvRatio = document.getElementById('slider-hdv-ratio');
  const valHdvRatio = document.getElementById('val-hdv-ratio');
  const selectPreset = document.getElementById('select-preset');

  // Cockpit Gauges
  const teleSpeed = document.getElementById('tele-speed');
  const teleSpeedKmh = document.getElementById('tele-speed-kmh');
  const speedArcPath = document.getElementById('speed-arc-path');
  const teleTtc = document.getElementById('tele-ttc');
  const teleTtcStatus = document.getElementById('tele-ttc-status');
  const ttcArcPath = document.getElementById('ttc-arc-path');

  const teleDistEntry = document.getElementById('tele-dist-entry');
  const teleGap = document.getElementById('tele-gap');
  const teleAccel = document.getElementById('tele-accel');
  const teleStepReward = document.getElementById('tele-step-reward');
  const teleTotalReward = document.getElementById('tele-total-reward');
  const teleMinTtc = document.getElementById('tele-min-ttc');

  // HUD & Banners
  const hudMergeState = document.getElementById('hud-merge-state');
  const hudFpsCounter = document.getElementById('hud-fps-counter');
  const eventBanner = document.getElementById('event-banner');
  const eventBannerText = document.getElementById('event-banner-text');

  // Timeline steps
  const stepApproach = document.getElementById('step-approach');
  const stepMerge = document.getElementById('step-merge');
  const stepCirc = document.getElementById('step-circ');
  const stepExit = document.getElementById('step-exit');
  const line1 = document.getElementById('line-1');
  const line2 = document.getElementById('line-2');
  const line3 = document.getElementById('line-3');

  // History table
  const epHistoryBody = document.getElementById('ep-history-body');
  const epCountBadge = document.getElementById('ep-count-badge');

  // --------------------------------------------------------------------
  // WEBSOCKET CONNECTION
  // --------------------------------------------------------------------
  function connectWebSocket() {
    const loc = window.location;
    const wsProto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${loc.host}/ws/telemetry`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WS] Connected to simulation stream.');
      statusSimMode.textContent = 'ONLINE';
      statusSimMode.style.color = '#10b981';
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected. Retrying in 2s...');
      statusSimMode.textContent = 'RECONNECTING';
      statusSimMode.style.color = '#f59e0b';
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error('[WS Error]', err);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (e) {
        console.error('Error parsing WS message:', e);
      }
    };
  }

  function handleServerMessage(msg) {
    if (msg.type === 'connected') {
      activeModel = msg.active_model;
      headerActiveModel.textContent = activeModel;
      isSimRunning = msg.running;
      isSimPaused = msg.paused;
      updateControlButtons();
      if (msg.models) renderModelCards(msg.models);
      if (msg.history) renderHistoryTable(msg.history);
    } else if (msg.type === 'telemetry') {
      handleTelemetryFrame(msg);
    } else if (msg.type === 'episode_end') {
      handleEpisodeEnd(msg.summary);
    } else if (msg.type === 'model_changed') {
      activeModel = msg.active_model;
      headerActiveModel.textContent = activeModel;
      fetchModelsList();
    }
  }

  function handleTelemetryFrame(frame) {
    // FPS calculation
    const now = performance.now();
    const delta = (now - lastFrameTime) / 1000;
    lastFrameTime = now;
    if (delta > 0) {
      fps = 0.9 * fps + 0.1 * (1 / delta);
      if (hudFpsCounter) hudFpsCounter.querySelector('span').textContent = `${fps.toFixed(1)} FPS`;
    }

    // Update 2D Canvas
    renderer.updateData(frame.vehicles, frame.telemetry);

    // Update Charts
    charts.updateTelemetry(frame.telemetry, frame.sim_time);

    // Update Cockpit Gauges
    const t = frame.telemetry;
    if (t) {
      // 1. Speed Gauge
      teleSpeed.textContent = t.speed.toFixed(1);
      teleSpeedKmh.textContent = `${t.speed_kmh.toFixed(1)} km/h`;
      const speedPct = Math.min(1.0, t.speed / 14.0);
      const maxArcOffset = 110;
      speedArcPath.style.strokeDashoffset = (maxArcOffset * (1 - speedPct)).toString();

      // 2. TTC Gauge
      if (t.ttc > 50) {
        teleTtc.textContent = 'inf';
        teleTtc.className = 'gauge-number safe';
        teleTtcStatus.textContent = 'SAFE GAP';
        ttcArcPath.style.strokeDashoffset = '0';
        ttcArcPath.style.stroke = '#10b981';
      } else {
        teleTtc.textContent = t.ttc.toFixed(1);
        const ttcPct = Math.min(1.0, t.ttc / 5.0);
        ttcArcPath.style.strokeDashoffset = (maxArcOffset * (1 - ttcPct)).toString();

        if (t.ttc < 1.5) {
          teleTtc.className = 'gauge-number danger';
          teleTtcStatus.textContent = 'CRITICAL TTC!';
          ttcArcPath.style.stroke = '#f43f5e';
        } else if (t.ttc < 3.0) {
          teleTtc.className = 'gauge-number warning';
          teleTtcStatus.textContent = 'CAUTION';
          ttcArcPath.style.stroke = '#f59e0b';
        } else {
          teleTtc.className = 'gauge-number safe';
          teleTtcStatus.textContent = 'SAFE GAP';
          ttcArcPath.style.stroke = '#10b981';
        }
      }

      // 3. Stats Grid
      teleDistEntry.textContent = `${t.dist_to_entry.toFixed(1)} m`;
      teleGap.textContent = `${t.gap_size.toFixed(1)} m`;
      teleAccel.textContent = `${t.accel >= 0 ? '+' : ''}${t.accel.toFixed(2)} m/s²`;
      teleAccel.style.color = t.accel >= 0 ? '#10b981' : '#f43f5e';
      teleStepReward.textContent = `${t.step_reward >= 0 ? '+' : ''}${t.step_reward.toFixed(2)}`;
      teleTotalReward.textContent = `${t.total_reward >= 0 ? '+' : ''}${t.total_reward.toFixed(1)}`;
      teleMinTtc.textContent = t.min_ttc < 900 ? `${t.min_ttc.toFixed(2)} s` : 'N/A';

      // 4. Merge State HUD & Timeline
      hudMergeState.className = `hud-badge hud-state ${t.merge_state}`;
      hudMergeState.querySelector('.hud-state-text').textContent = t.merge_state;
      updateTimeline(t.merge_state);

      // 5. Event Banner Toast
      if (t.event) {
        showEventBanner(t.event.name, t.event.message);
      }
    }

    if (epCountBadge) {
      epCountBadge.textContent = `Ep #${frame.episode}`;
    }
  }

  function updateTimeline(state) {
    stepApproach.className = 't-step active';
    stepMerge.className = 't-step';
    stepCirc.className = 't-step';
    stepExit.className = 't-step';
    line1.className = 't-line';
    line2.className = 't-line';
    line3.className = 't-line';

    if (state === 'MERGE_ZONE') {
      stepApproach.className = 't-step completed';
      line1.className = 't-line completed';
      stepMerge.className = 't-step active';
    } else if (state === 'CIRCULATING') {
      stepApproach.className = 't-step completed';
      line1.className = 't-line completed';
      stepMerge.className = 't-step completed';
      line2.className = 't-line completed';
      stepCirc.className = 't-step active';
    } else if (state === 'EXITED') {
      stepApproach.className = 't-step completed';
      line1.className = 't-line completed';
      stepMerge.className = 't-step completed';
      line2.className = 't-line completed';
      stepCirc.className = 't-step completed';
      line3.className = 't-line completed';
      stepExit.className = 't-step completed active';
    }
  }

  function showEventBanner(type, message) {
    eventBannerText.textContent = message;
    eventBanner.className = 'event-banner-overlay show';
    if (type === 'MERGE_SUCCESS') eventBanner.classList.add('success');
    if (type === 'COLLISION') eventBanner.classList.add('collision');

    setTimeout(() => {
      eventBanner.className = 'event-banner-overlay';
    }, 2800);
  }

  function handleEpisodeEnd(summary) {
    // Add row to table
    const row = document.createElement('tr');
    let outBadge = `<span class="badge badge-success">SUCCESS</span>`;
    if (summary.outcome === 'COLLISION') outBadge = `<span class="badge badge-danger">COLLISION</span>`;
    if (summary.outcome === 'TIMEOUT') outBadge = `<span class="badge badge-warning">TIMEOUT</span>`;

    row.innerHTML = `
      <td>#${summary.episode}</td>
      <td>${outBadge}</td>
      <td>${summary.steps}</td>
      <td>${summary.time_to_merge !== 'N/A' ? summary.time_to_merge + 's' : 'N/A'}</td>
      <td>${summary.min_ttc !== 'N/A' ? summary.min_ttc + 's' : 'N/A'}</td>
      <td class="${summary.total_reward >= 0 ? 'text-success' : 'text-danger'}">${summary.total_reward > 0 ? '+' : ''}${summary.total_reward}</td>
    `;
    if (epHistoryBody.children.length === 1 && epHistoryBody.children[0].innerText.includes('Awaiting')) {
      epHistoryBody.innerHTML = '';
    }
    epHistoryBody.insertBefore(row, epHistoryBody.firstChild);

    charts.resetTelemetryData();
  }

  function renderHistoryTable(history) {
    if (!history || history.length === 0) return;
    epHistoryBody.innerHTML = '';
    history.forEach(summary => {
      let outBadge = `<span class="badge badge-success">SUCCESS</span>`;
      if (summary.outcome === 'COLLISION') outBadge = `<span class="badge badge-danger">COLLISION</span>`;
      if (summary.outcome === 'TIMEOUT') outBadge = `<span class="badge badge-warning">TIMEOUT</span>`;

      const row = document.createElement('tr');
      row.innerHTML = `
        <td>#${summary.episode}</td>
        <td>${outBadge}</td>
        <td>${summary.steps}</td>
        <td>${summary.time_to_merge !== 'N/A' ? summary.time_to_merge + 's' : 'N/A'}</td>
        <td>${summary.min_ttc !== 'N/A' ? summary.min_ttc + 's' : 'N/A'}</td>
        <td class="${summary.total_reward >= 0 ? 'text-success' : 'text-danger'}">${summary.total_reward > 0 ? '+' : ''}${summary.total_reward}</td>
      `;
      epHistoryBody.appendChild(row);
    });
  }

  // --------------------------------------------------------------------
  // CONTROLS & EVENT HANDLERS
  // --------------------------------------------------------------------
  function updateControlButtons() {
    if (isSimRunning) {
      btnSimPlayText.textContent = 'Running';
      btnSimPlay.classList.add('btn-primary');
      btnSimPause.disabled = false;
      btnSimPause.querySelector('span').textContent = isSimPaused ? 'Resume' : 'Pause';
    } else {
      btnSimPlayText.textContent = 'Start';
      btnSimPause.disabled = true;
    }
  }

  btnSimPlay.addEventListener('click', () => {
    const config = {
      model: activeModel,
      preset: selectPreset.value,
      hdv_ratio: parseFloat(sliderHdvRatio.value) / 100.0,
      speed: parseFloat(sliderSpeedMult.value)
    };
    sendWsMessage({ action: 'start', config });
    isSimRunning = true;
    isSimPaused = false;
    updateControlButtons();
  });

  btnSimPause.addEventListener('click', () => {
    sendWsMessage({ action: 'pause' });
    isSimPaused = !isSimPaused;
    updateControlButtons();
  });

  btnSimRestart.addEventListener('click', () => {
    const config = {
      model: activeModel,
      preset: selectPreset.value,
      hdv_ratio: parseFloat(sliderHdvRatio.value) / 100.0,
      speed: parseFloat(sliderSpeedMult.value)
    };
    sendWsMessage({ action: 'start', config });
  });

  sliderSpeedMult.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    valSpeedMult.textContent = `${val.toFixed(1)}x`;
    sendWsMessage({ action: 'set_speed', speed: val });
  });

  sliderHdvRatio.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    valHdvRatio.textContent = `${val}%`;
  });

  selectPreset.addEventListener('change', (e) => {
    activePreset = e.target.value;
  });

  btnCenterEgo.addEventListener('click', () => renderer.centerOnEgo());
  btnZoomIn.addEventListener('click', () => {
    renderer.scale = Math.min(5.0, renderer.scale * 1.2);
    renderer.render();
  });
  btnZoomOut.addEventListener('click', () => {
    renderer.scale = Math.max(0.6, renderer.scale * 0.8);
    renderer.render();
  });
  btnZoomReset.addEventListener('click', () => renderer.resetView());

  function sendWsMessage(msgObj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msgObj));
    }
  }

  // --------------------------------------------------------------------
  // NAVIGATION TABS
  // --------------------------------------------------------------------
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      navTabs.forEach(t => t.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');

      if (targetId === 'tab-simulator') {
        setTimeout(() => renderer.initCanvasSize(), 50);
      }
    });
  });

  // --------------------------------------------------------------------
  // MODEL ARENA
  // --------------------------------------------------------------------
  function fetchModelsList() {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        if (data.models) renderModelCards(data.models);
      })
      .catch(err => console.error('Error loading models:', err));
  }

  function renderModelCards(models) {
    const grid = document.getElementById('models-cards-grid');
    if (!grid) return;

    grid.innerHTML = '';
    models.forEach(m => {
      const isCurActive = (m.filename === activeModel);
      const card = document.createElement('div');
      card.className = `model-card ${isCurActive ? 'active-model' : ''}`;

      let friendlyTitle = m.filename.replace('.zip', '').replace(/_/g, ' ').toUpperCase();
      let badgeHtml = isCurActive
        ? `<span class="badge badge-primary">ACTIVE POLICY</span>`
        : `<span class="badge badge-muted">CHECKPOINT</span>`;

      card.innerHTML = `
        <div class="model-card-header">
          <div>
            <h4 class="model-card-title">${friendlyTitle}</h4>
            <div style="font-size: 0.72rem; color: var(--text-dim);">${m.filename}</div>
          </div>
          ${badgeHtml}
        </div>
        <div class="model-card-meta">
          <div><span class="meta-label">Size:</span> <span class="meta-val">${(m.size_bytes / 1024).toFixed(0)} KB</span></div>
          <div><span class="meta-label">Modified:</span> <span class="meta-val">${m.modified.split(' ')[0]}</span></div>
        </div>
        <div style="margin-top: auto; display: flex; justify-content: flex-end;">
          <button class="btn btn-sm ${isCurActive ? 'btn-primary' : 'btn-secondary'}" onclick="window.selectModel('${m.filename}')">
            ${isCurActive ? '✓ Currently Loaded' : 'Load Policy'}
          </button>
        </div>
      `;
      grid.appendChild(card);
    });
  }

  window.selectModel = function(modelFilename) {
    fetch('/api/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelFilename })
    })
    .then(res => res.json())
    .then(data => {
      if (data.active_model) {
        activeModel = data.active_model;
        headerActiveModel.textContent = activeModel;
        fetchModelsList();
      }
    });
  };

  // --------------------------------------------------------------------
  // LAUNCH NATIVE SUMO-GUI CONTROLLER
  // --------------------------------------------------------------------
  const btnLaunchSumoGui = document.getElementById('btn-launch-sumo-gui');
  const btnModalLaunchGui = document.getElementById('btn-modal-launch-gui');
  const guiLaunchAlert = document.getElementById('gui-launch-alert');

  function triggerSumoGuiLaunch(model, preset, episodes, speed) {
    if (guiLaunchAlert) {
      guiLaunchAlert.className = 'launch-alert';
      guiLaunchAlert.textContent = '🚀 Launching SUMO-GUI window on desktop...';
    }

    fetch('/api/launch-sumo-gui', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model || activeModel,
        preset: preset || 'MEDIUM',
        episodes: parseInt(episodes) || 3,
        speed: parseFloat(speed) || 0.25
      })
    })
    .then(res => res.json())
    .then(data => {
      if (data.success && guiLaunchAlert) {
        guiLaunchAlert.textContent = `✓ ${data.message}`;
      } else if (guiLaunchAlert) {
        guiLaunchAlert.textContent = `Error launching SUMO-GUI: ${data.error}`;
      }
    })
    .catch(err => {
      if (guiLaunchAlert) guiLaunchAlert.textContent = `Error: ${err}`;
    });
  }

  if (btnLaunchSumoGui) {
    btnLaunchSumoGui.addEventListener('click', () => {
      const preset = selectPreset ? selectPreset.value : 'MEDIUM';
      triggerSumoGuiLaunch(activeModel, preset, 3, 0.25);
    });
  }

  if (btnModalLaunchGui) {
    btnModalLaunchGui.addEventListener('click', () => {
      const model = document.getElementById('gui-modal-model').value;
      const preset = document.getElementById('gui-modal-preset').value;
      const episodes = document.getElementById('gui-modal-episodes').value;
      const speed = document.getElementById('gui-modal-speed').value;
      triggerSumoGuiLaunch(model, preset, episodes, speed);
    });
  }

  // --------------------------------------------------------------------
  // SYSTEM STATUS & BENCHMARKS INITIALIZATION
  // --------------------------------------------------------------------
  function fetchSystemStatus() {
    fetch('/api/status')
      .then(res => res.json())
      .then(data => {
        if (data.gpu_name) {
          headerGpuStatus.textContent = data.cuda_available ? data.gpu_name.split(' ')[0] + ' CUDA' : 'CPU Mode';
          const diagGpu = document.getElementById('diag-gpu-name');
          if (diagGpu) diagGpu.textContent = data.gpu_name;
        }
        if (data.torch_version) {
          const diagTorch = document.getElementById('diag-torch-status');
          if (diagTorch) diagTorch.textContent = `PyTorch ${data.torch_version} (${data.cuda_available ? 'CUDA 12.1' : 'CPU'})`;
        }
        if (data.sumo_binary) {
          const diagSumo = document.getElementById('diag-sumo-path');
          if (diagSumo) diagSumo.textContent = data.sumo_binary;
        }
      });
  }

  function fetchResearchData() {
    fetch('/api/results-data')
      .then(res => res.json())
      .then(data => {
        charts.initResearchCharts(data);
      });
  }

  // Kickoff
  connectWebSocket();
  fetchSystemStatus();
  fetchModelsList();
  fetchResearchData();
});
