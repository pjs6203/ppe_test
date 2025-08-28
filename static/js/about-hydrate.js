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
      let gpuP=0, vramP=0; if (d.gpus && d.gpus.length>0){ gpuP = Math.round(d.gpus[0].sm_util||0); vramP = Math.round(d.gpus[0].vram_percent||0); }
      const C = window.__aboutCharts; if (!C) return;
      updateChartSmooth(C.cpu, cpuP);
      updateChartSmooth(C.mem, memP);
      updateChartSmooth(C.gpu, gpuP);
      updateChartSmooth(C.vram, vramP);
      // Update small UI bars if present
      const cpuProg = qs('cpu-progress'); if(cpuProg) cpuProg.style.width = cpuP+'%';
      const cpuLbl = qs('cpu-percent'); if(cpuLbl) cpuLbl.textContent = cpuP+'%';
      const memProg = qs('mem-progress'); if(memProg) memProg.style.width = memP+'%';
    }catch(e){ /* silent */ }
  }

  async function setup(){
    if (!(qs('cpu-chart') && qs('mem-chart') && qs('gpu-chart') && qs('vram-chart'))) return;
    if (window.__aboutCharts) return; // already
    await ensureChartJS();
    try{
      const cpu = makeLineChart(qs('cpu-chart').getContext('2d'), '#2563eb');
      const mem = makeLineChart(qs('mem-chart').getContext('2d'), '#16a34a');
      const gpu = makeLineChart(qs('gpu-chart').getContext('2d'), '#f59e0b');
      const vram = makeLineChart(qs('vram-chart').getContext('2d'), '#06b6d4');
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
})();
