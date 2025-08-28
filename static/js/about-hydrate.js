// Hydrate About page charts on Turbo navigation
(function(){
  function on(el, ev, fn){ document.addEventListener(ev, fn, false); }
  function qs(id){ return document.getElementById(id); }

  function ensureChartJS(){
    return new Promise((resolve, reject)=>{
      if (window.Chart) return resolve();
      const src = 'https://cdn.jsdelivr.net/npm/chart.js';
      // 이미 스크립트 태그가 있다면, 로드 완료를 기다림
      let existing = document.querySelector('script[data-about-chartjs]');
      if (existing){
        if (window.Chart) return resolve();
        existing.addEventListener('load', ()=> resolve());
        existing.addEventListener('error', (e)=> reject(e));
        // 안전 타임아웃
        setTimeout(()=>{ if (window.Chart) resolve(); }, 1500);
        return;
      }
      const s = document.createElement('script'); s.src = src; s.async = true; s.dataset.aboutChartjs = '1';
      s.onload = ()=> resolve(); s.onerror = (e)=> reject(e);
      document.head.appendChild(s);
    });
  }

  function makeLineChart(ctx, hexColor){
    function hexToRgba(hex, a=1){
      const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      if(!m) return `rgba(37,99,235,${a})`;
      const r = parseInt(m[1],16), g = parseInt(m[2],16), b = parseInt(m[3],16);
      return `rgba(${r},${g},${b},${a})`;
    }

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      animations: { colors: false, x: false, y: { duration: 200 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15,23,42,0.95)',
          borderColor: 'rgba(148,163,184,0.2)',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#cbd5e1',
          displayColors: false,
          callbacks: {
            label: (ctx)=>{
              const v = Math.round(ctx.parsed.y ?? 0);
              const lbl = ctx.dataset.label || '';
              return lbl ? `${lbl}: ${v}%` : `${v}%`;
            }
          }
        }
      },
      scales: {
        x: { display: false },
        y: {
          beginAtZero: true, max: 100,
          grid: { color: 'rgba(255,255,255,0.06)', drawBorder: false, borderDash: [4,4] },
          ticks: { color: '#9fb1c8', callback: v=> v + '%' }
        }
      },
      elements: {
        point: {
          radius: (ctx)=> ctx?.dataIndex === (ctx?.dataset?.data?.length||1)-1 ? 2.5 : 0,
          hoverRadius: 3.5,
          hitRadius: 8
        },
        line: { borderWidth: 2, tension: 0.35, cubicInterpolationMode: 'monotone' }
      }
    };

    // Scriptable gradient fill respecting chart area on resize
    const dataset = {
      label: '',
      data: Array(60).fill(0),
      borderColor: hexToRgba(hexColor, 1),
      backgroundColor: (context)=>{
        const chart = context.chart; const {ctx, chartArea} = chart;
        if (!chartArea) return hexToRgba(hexColor, 0.15);
        const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
        g.addColorStop(0, hexToRgba(hexColor, 0.22));
        g.addColorStop(1, hexToRgba(hexColor, 0.00));
        return g;
      },
      fill: true,
    };

    // Optional subtle glow on the line
    const glow = {
      id: 'lineGlow',
      beforeDatasetsDraw(chart, args, pluginOptions){
        const {ctx} = chart;
        chart.data.datasets.forEach((ds, i)=>{
          const meta = chart.getDatasetMeta(i);
          if (!meta?.dataset) return;
          ctx.save();
          ctx.shadowColor = hexToRgba(hexColor, 0.6);
          ctx.shadowBlur = 8;
          ctx.lineJoin = 'round';
          ctx.lineCap = 'round';
          ctx.strokeStyle = hexToRgba(hexColor, 0.6);
          ctx.lineWidth = 0; // only glow underneath actual stroke
          // draw path with shadow by stroking the element path
          meta.dataset.draw(ctx, chart.chartArea);
          ctx.restore();
        });
      }
    };

    // Cleanup any existing chart on this canvas
    try{ const existing = Chart.getChart(ctx.canvas); if (existing) { try{ existing.destroy(); }catch{} } }catch{}
    return new Chart(ctx, { type: 'line', data: { labels: Array(60).fill(''), datasets: [dataset] }, options, plugins: [glow] });
  }

  // 안정 업데이트: 보간 제거, EMA로 완화. 1업데이트=1포인트 유지
  function updateChartStable(chart, value){
    const data = chart.data.datasets[0].data;
    const last = data.length ? data[data.length-1] : value;
    // EMA: 새값 30%, 이전 70%
    const alpha = 0.3;
    const blended = (1 - alpha) * last + alpha * (isFinite(value)? value : last);
    data.push(blended);
    while(data.length>60) data.shift();
    chart.update('none');
  }

  // 세션 저장/복원으로 탭 내 일관성 유지
  function saveChartState(){
    try{
      if (!window.__aboutCharts) return;
      const C = window.__aboutCharts;
      const payload = {
        cpu: C.cpu?.data?.datasets?.[0]?.data || [],
        mem: C.mem?.data?.datasets?.[0]?.data || [],
        gpu: C.gpu?.data?.datasets?.[0]?.data || [],
        vram: C.vram?.data?.datasets?.[0]?.data || [],
        ts: Date.now(),
      };
      sessionStorage.setItem('aboutChart:v1', JSON.stringify(payload));
    }catch{}
  }

  function loadChartState(){
    try{
      const raw = sessionStorage.getItem('aboutChart:v1');
      if (!raw) return;
      const data = JSON.parse(raw);
      const maxAge = 5*60*1000; // 5분 이내만 복원
      if (!data || !data.ts || Date.now()-data.ts > maxAge) return;
      const C = window.__aboutCharts; if (!C) return;
      const keys = ['cpu','mem','gpu','vram'];
      keys.forEach(k=>{
        const chart = C[k];
        const arr = Array.isArray(data[k]) ? data[k] : [];
        if (chart && arr.length){
          chart.data.datasets[0].data = arr.slice(-60);
          chart.update('none');
        }
      });
    }catch{}
  }

  async function updateSystemMetrics(){
    try{
      const res = await fetch('/api/system_metrics', { cache:'no-store' });
      const d = await res.json();
      const cpuP = Math.round(d.cpu_percent||0);
      const memP = Math.round(d.mem_percent||0);
      let gpuP=0, vramP=0, vramUsed=0, vramTotal=0; if (d.gpus && d.gpus.length>0){
        gpuP = Math.round(d.gpus[0].sm_util||0);
        vramP = Math.round(d.gpus[0].vram_percent||0);
        vramUsed = ((d.gpus[0].vram_used||0)/1024/1024/1024).toFixed(1);
        vramTotal = ((d.gpus[0].vram_total||0)/1024/1024/1024).toFixed(1);
      }
      const C = window.__aboutCharts; if (!C) return;
  updateChartStable(C.cpu, cpuP);
  updateChartStable(C.mem, memP);
  updateChartStable(C.gpu, gpuP);
  updateChartStable(C.vram, vramP);
      // Update small UI bars if present
      const cpuProg = qs('cpu-progress'); if(cpuProg) cpuProg.style.width = cpuP+'%';
      const cpuLbl = qs('cpu-percent'); if(cpuLbl) cpuLbl.textContent = cpuP+'%';
      const memProg = qs('mem-progress'); if(memProg) memProg.style.width = memP+'%';
      const memLbl = qs('mem-percent');
      if (memLbl && d.mem_used!=null && d.mem_total!=null){
        const memUsed = ((d.mem_used||0)/1024/1024/1024).toFixed(1);
        const memTot = ((d.mem_total||0)/1024/1024/1024).toFixed(1);
        memLbl.textContent = `${memUsed}GB / ${memTot}GB`;
      }
      const gpuProg = qs('gpu-progress'); if(gpuProg) gpuProg.style.width = gpuP+'%';
      const gpuLbl = qs('gpu-percent'); if(gpuLbl) gpuLbl.textContent = gpuP+'%';
      const vramProg = qs('vram-progress'); if(vramProg) vramProg.style.width = vramP+'%';
      const vramLbl = qs('vram-percent'); if(vramLbl && (vramUsed||vramTotal)) vramLbl.textContent = `${vramUsed}GB / ${vramTotal}GB`;
    }catch(e){ /* silent */ }
  }

  function needsInit(){
    const ids = ['cpu-chart','mem-chart','gpu-chart','vram-chart'];
    const canvases = ids.map(id=> qs(id)).filter(Boolean);
    if (canvases.length !== ids.length) return false; // DOM이 아직 준비되지 않음
    if (!window.Chart) return true;
    // 하나라도 Chart 인스턴스가 없으면 재초기화 필요
    return canvases.some(c => !(Chart.getChart && Chart.getChart(c)));
  }

  async function setup(){
    if (!(qs('cpu-chart') && qs('mem-chart') && qs('gpu-chart') && qs('vram-chart'))) return;
    await ensureChartJS();
    try{
      const cpuCtx = qs('cpu-chart').getContext('2d');
      const memCtx = qs('mem-chart').getContext('2d');
      const gpuCtx = qs('gpu-chart').getContext('2d');
      const vramCtx = qs('vram-chart').getContext('2d');
      // 캔버스별 현 상태 검사
  let cpu = (Chart.getChart && Chart.getChart(cpuCtx.canvas)); if (!cpu) cpu = makeLineChart(cpuCtx, '#60a5fa'); // blue-300
  let mem = (Chart.getChart && Chart.getChart(memCtx.canvas)); if (!mem) mem = makeLineChart(memCtx, '#22c55e'); // green-500
  let gpu = (Chart.getChart && Chart.getChart(gpuCtx.canvas)); if (!gpu) gpu = makeLineChart(gpuCtx, '#f59e0b'); // amber-500
  let vram = (Chart.getChart && Chart.getChart(vramCtx.canvas)); if (!vram) vram = makeLineChart(vramCtx, '#06b6d4'); // cyan-500
      window.__aboutCharts = { cpu, mem, gpu, vram };
  if (!window.__aboutTimer){ window.__aboutTimer = setInterval(updateSystemMetrics, 2000); }
      updateSystemMetrics();
  // 세션 복원 시도 (초기 생성 직후)
  loadChartState();
      // 크기/표시 재확인
      setTimeout(()=>{ try{ cpu.resize(); mem.resize(); gpu.resize(); vram.resize(); }catch{} }, 50);
    }catch(e){ console.error('About charts init failed:', e); }
  }

  function teardown(){
    if (window.__aboutTimer){ clearInterval(window.__aboutTimer); window.__aboutTimer = null; }
    const C = window.__aboutCharts; if (!C) return;
    try{ C.cpu && C.cpu.destroy(); }catch{}
    try{ C.mem && C.mem.destroy(); }catch{}
    try{ C.gpu && C.gpu.destroy(); }catch{}
    try{ C.vram && C.vram.destroy(); }catch{}
    window.__aboutCharts = null;
  }

  on(document, 'turbo:render', ()=>{ if (needsInit()) setup(); else { updateSystemMetrics(); } });
  on(document, 'turbo:before-cache', teardown);
  on(document, 'turbo:before-render', ()=>{ /* 페이지 교체 직전 안전 해제 */ teardown(); });
  window.addEventListener('pageshow', ()=>{ if (needsInit()) setup(); });
  // 초기 로드 또는 Turbo 미사용 시에도 동작하도록 즉시 시도
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setup();
  } else {
    document.addEventListener('DOMContentLoaded', setup, { once: true });
  }
  // 백업 재시도: 렌더 후 400ms 내 미표시 시 재시도
  setTimeout(()=>{ if (needsInit()) setup(); }, 400);

  // 저장 타이머(약 10초마다 저장)
  setInterval(saveChartState, 10000);
  window.addEventListener('beforeunload', saveChartState);
  document.addEventListener('visibilitychange', ()=>{ if (document.hidden) saveChartState(); });
})();
