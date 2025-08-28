// Hydrate About page charts on Turbo navigation
(function(){
  function on(el, ev, fn){ document.addEventListener(ev, fn, false); }
  function qs(id){ return document.getElementById(id); }

  function ensureChartJS(){
    return new Promise((resolve, reject)=>{
      if (window.Chart) return resolve();
      const src = 'https://cdn.jsdelivr.net/npm/chart.js';
      // prevent duplicate loads
      if (document.querySelector('script[data-about-chartjs]')) return resolve();
      const s = document.createElement('script'); s.src = src; s.defer = true; s.dataset.aboutChartjs = '1';
      s.onload = ()=> resolve(); s.onerror = (e)=> reject(e);
      document.head.appendChild(s);
    });
  }

  function makeLineChart(ctx, color){
    const opt = {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { display:false }, y: { beginAtZero:true, max:100, grid:{ color:'rgba(255,255,255,0.1)' }, ticks:{ color:'#cbd5e1' } } },
      elements: { point:{ radius:0 }, line:{ borderWidth:2 } }
    };
    // 이미 연결된 차트가 있으면 파괴하고 재생성하거나 재사용
    try{
      const existing = Chart.getChart(ctx.canvas);
      if (existing) {
        try { existing.destroy(); } catch{}
      }
    }catch{}
    return new Chart(ctx, { type:'line', data:{ labels:Array(60).fill(''), datasets:[{ data:Array(60).fill(0), borderColor: color, backgroundColor: color.replace('1)', '0.1)'), fill:true, tension:0.4 }] }, options: opt });
  }

  function updateChartSmooth(chart, value){
    const data = chart.data.datasets[0].data; const last = data[data.length-1]||0;
    if (Math.abs(value-last)>5){ for(let i=1;i<=3;i++){ data.push(last+(value-last)*(i/3)); } }
    else { data.push(value); }
    while(data.length>60) data.shift();
    chart.update('none');
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
      updateChartSmooth(C.cpu, cpuP);
      updateChartSmooth(C.mem, memP);
      updateChartSmooth(C.gpu, gpuP);
      updateChartSmooth(C.vram, vramP);
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

  async function setup(){
    if (!(qs('cpu-chart') && qs('mem-chart') && qs('gpu-chart') && qs('vram-chart'))) return;
    if (window.__aboutCharts) return; // already
    await ensureChartJS();
    try{
      const cpuCtx = qs('cpu-chart').getContext('2d');
      const memCtx = qs('mem-chart').getContext('2d');
      const gpuCtx = qs('gpu-chart').getContext('2d');
      const vramCtx = qs('vram-chart').getContext('2d');
      // 기존 차트 재사용/정리 후 생성
      const cpu = (Chart.getChart && Chart.getChart(cpuCtx.canvas)) || makeLineChart(cpuCtx, '#2563eb');
      const mem = (Chart.getChart && Chart.getChart(memCtx.canvas)) || makeLineChart(memCtx, '#16a34a');
      const gpu = (Chart.getChart && Chart.getChart(gpuCtx.canvas)) || makeLineChart(gpuCtx, '#f59e0b');
      const vram = (Chart.getChart && Chart.getChart(vramCtx.canvas)) || makeLineChart(vramCtx, '#06b6d4');
      window.__aboutCharts = { cpu, mem, gpu, vram };
      window.__aboutTimer = setInterval(updateSystemMetrics, 2000);
      updateSystemMetrics();
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

  on(document, 'turbo:render', setup);
  on(document, 'turbo:before-cache', teardown);
  // 초기 로드 또는 Turbo 미사용 시에도 동작하도록 즉시 시도
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setup();
  } else {
    document.addEventListener('DOMContentLoaded', setup, { once: true });
  }
})();
