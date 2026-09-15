/**
 * web/js/telemetry_charts.js
 * Real-time Chart.js telemetry graphs & Research Study Visualizations
 */

class TelemetryChartsManager {
  constructor() {
    this.speedChart = null;
    this.safetyChart = null;
    this.hdvChart = null;
    this.ablationChart = null;

    this.maxPoints = 40;
    this.timeLabels = [];
    this.speedData = [];
    this.accelData = [];
    this.ttcData = [];
    this.gapData = [];

    this.initRealtimeCharts();
  }

  initRealtimeCharts() {
    // Chart.js dark theme defaults
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#1e293b';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // 1. Speed & Acceleration Chart
    const speedCtx = document.getElementById('speedChart')?.getContext('2d');
    if (speedCtx) {
      this.speedChart = new Chart(speedCtx, {
        type: 'line',
        data: {
          labels: this.timeLabels,
          datasets: [
            {
              label: 'Speed (m/s)',
              data: this.speedData,
              borderColor: '#38bdf8',
              backgroundColor: 'rgba(56, 189, 248, 0.1)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.25,
              yAxisID: 'y'
            },
            {
              label: 'Accel (m/s²)',
              data: this.accelData,
              borderColor: '#10b981',
              borderWidth: 1.5,
              borderDash: [4, 4],
              pointRadius: 0,
              tension: 0.2,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: { display: false },
            y: {
              type: 'linear',
              position: 'left',
              min: 0,
              max: 15,
              title: { display: true, text: 'Speed (m/s)', color: '#38bdf8', font: { size: 10 } }
            },
            y1: {
              type: 'linear',
              position: 'right',
              min: -4.5,
              max: 3.0,
              grid: { drawOnChartArea: false },
              title: { display: true, text: 'Accel (m/s²)', color: '#10b981', font: { size: 10 } }
            }
          },
          plugins: {
            legend: { labels: { boxWidth: 10, font: { size: 11 } } }
          }
        }
      });
    }

    // 2. Safety Metric (TTC & Gap) Chart
    const safetyCtx = document.getElementById('safetyChart')?.getContext('2d');
    if (safetyCtx) {
      this.safetyChart = new Chart(safetyCtx, {
        type: 'line',
        data: {
          labels: this.timeLabels,
          datasets: [
            {
              label: 'TTC (s)',
              data: this.ttcData,
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245, 158, 11, 0.1)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.25,
              yAxisID: 'y'
            },
            {
              label: 'Gap Size (m)',
              data: this.gapData,
              borderColor: '#818cf8',
              borderWidth: 1.5,
              pointRadius: 0,
              tension: 0.2,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: { display: false },
            y: {
              type: 'linear',
              position: 'left',
              min: 0,
              max: 10,
              title: { display: true, text: 'TTC (sec)', color: '#f59e0b', font: { size: 10 } }
            },
            y1: {
              type: 'linear',
              position: 'right',
              min: 0,
              max: 60,
              grid: { drawOnChartArea: false },
              title: { display: true, text: 'Gap (m)', color: '#818cf8', font: { size: 10 } }
            }
          },
          plugins: {
            legend: { labels: { boxWidth: 10, font: { size: 11 } } }
          }
        }
      });
    }
  }

  updateTelemetry(telemetry, simTime) {
    if (!telemetry) return;

    const timeLabel = `${simTime.toFixed(1)}s`;
    const speed = telemetry.speed || 0;
    const accel = telemetry.accel || 0;
    const ttc = telemetry.ttc > 10 ? 10 : telemetry.ttc;
    const gap = telemetry.gap_size || 50;

    this.timeLabels.push(timeLabel);
    this.speedData.push(speed);
    this.accelData.push(accel);
    this.ttcData.push(ttc);
    this.gapData.push(gap);

    if (this.timeLabels.length > this.maxPoints) {
      this.timeLabels.shift();
      this.speedData.shift();
      this.accelData.shift();
      this.ttcData.shift();
      this.gapData.shift();
    }

    if (this.speedChart) this.speedChart.update();
    if (this.safetyChart) this.safetyChart.update();
  }

  resetTelemetryData() {
    this.timeLabels.length = 0;
    this.speedData.length = 0;
    this.accelData.length = 0;
    this.ttcData.length = 0;
    this.gapData.length = 0;
    if (this.speedChart) this.speedChart.update();
    if (this.safetyChart) this.safetyChart.update();
  }

  initResearchCharts(data) {
    // 1. HDV Penetration Chart
    const hdvCtx = document.getElementById('hdvPenetrationChart')?.getContext('2d');
    if (hdvCtx && data.hdv_penetration) {
      const labels = data.hdv_penetration.map(d => `${(parseFloat(d.hdv_ratio) * 100).toFixed(0)}% HDV`);
      const mergeTimes = data.hdv_penetration.map(d => parseFloat(d.avg_merge_time));
      const ttcs = data.hdv_penetration.map(d => parseFloat(d.avg_ttc));

      this.hdvChart = new Chart(hdvCtx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Mean Merge Time (s)',
              data: mergeTimes,
              backgroundColor: 'rgba(56, 189, 248, 0.7)',
              borderColor: '#38bdf8',
              borderWidth: 1,
              borderRadius: 6
            },
            {
              label: 'Mean Time-To-Collision (s)',
              data: ttcs,
              backgroundColor: 'rgba(16, 185, 129, 0.7)',
              borderColor: '#10b981',
              borderWidth: 1,
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 18,
              title: { display: true, text: 'Seconds' }
            }
          }
        }
      });
    }

    // 2. Ablation Study Chart
    const ablationCtx = document.getElementById('ablationChart')?.getContext('2d');
    if (ablationCtx && data.ablation) {
      const labels = data.ablation.map(d => d.Variant.replace(/^\d+\.\s*/, ''));
      const successRates = data.ablation.map(d => parseFloat(d.success_rate));
      const collisionRates = data.ablation.map(d => parseFloat(d.collision_rate));

      this.ablationChart = new Chart(ablationCtx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Success Rate (%)',
              data: successRates,
              backgroundColor: 'rgba(16, 185, 129, 0.8)',
              borderRadius: 6
            },
            {
              label: 'Collision Rate (%)',
              data: collisionRates,
              backgroundColor: 'rgba(244, 63, 94, 0.8)',
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: { display: true, text: 'Percentage (%)' }
            }
          }
        }
      });
    }
  }
}

window.TelemetryChartsManager = TelemetryChartsManager;
