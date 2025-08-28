// Reusable mesh background. Call initMesh(canvasId, opts)
function initMesh(canvasId = 'mesh-canvas', opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  let width = 0, height = 0, dpr = Math.max(1, window.devicePixelRatio || 1);

  // Increase point density by ~20% (reduce cell size by sqrt(1/1.2))
  const __densityBoost = 1.3; // 20% more points
  const __scale = 1/Math.sqrt(__densityBoost); // ~0.913
  const cellW = (typeof opts.cellW === 'number') ? opts.cellW : Math.round(140 * __scale);
  const cellH = (typeof opts.cellH === 'number') ? opts.cellH : Math.round(90 * __scale);
  const jitter = (typeof opts.jitter === 'number') ? opts.jitter : 27; // 약 10% 감소로 과한 겹침 완화
  const maxConn = (typeof opts.maxConn === 'number') ? opts.maxConn : 7; // 약간 더 연결해 밀도에 균형
  const lineColorBase = opts.lineColorBase || 'rgba(88,192,255,';
  const nodeColorBase = opts.nodeColorBase || 'rgba(180,230,255,';
  const mouseStrength = (typeof opts.mouseStrength === 'number') ? opts.mouseStrength : 1.5; // 1.0 기본, 1.2 권장
  const idleStrength = (typeof opts.idleStrength === 'number') ? opts.idleStrength : 6; // 픽셀 진폭
  const idleSpeed = (typeof opts.idleSpeed === 'number') ? opts.idleSpeed : 0.01; // 느린 드리프트 속도 0.006

  let points = [];
  let mouse = {x: -9999, y: -9999};

  function resize(){
    // set canvas to viewport size only (avoid oversizing that causes scroll)
    const vw = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
    const vh = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
    width = canvas.clientWidth = Math.floor(vw);
    height = canvas.clientHeight = Math.floor(vh);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr,0,0,dpr,0,0);
    buildPoints();
  }

  function buildPoints(){
    points = [];
    const cols = Math.ceil(width / cellW) + 1;
    const rows = Math.ceil(height / cellH) + 1;
  for (let r=0;r<rows;r++){
      for (let c=0;c<cols;c++){
        const ox = c*cellW + ((r%2)?cellW/2:0);
        const oy = r*cellH;
        const px = ox + (Math.random()*2-1)*jitter;
        const py = oy + (Math.random()*2-1)*jitter;
    points.push({x:px,y:py,ox:px,oy:py,vx:0,vy:0,neighbors:[], phx: Math.random()*Math.PI*2, phy: Math.random()*Math.PI*2, amp: idleStrength*(0.6+0.8*Math.random())});
      }
    }
    for (let i=0;i<points.length;i++){
      const p = points[i];
      const dists = [];
      for (let j=0;j<points.length;j++){
        if (i===j) continue;
        const q=points[j];
        const dx=p.x-q.x, dy=p.y-q.y;
        dists.push({idx:j,dist:dx*dx+dy*dy});
      }
      dists.sort((a,b)=>a.dist-b.dist);
      p.neighbors = dists.slice(0,maxConn).map(d=>d.idx);
    }
  }

  let t = 0;
  function update(dt){
    t += dt;
    for (let i=0;i<points.length;i++){
      const p = points[i];
      // idle drifting target around origin
      const tx = p.ox + Math.sin(t*idleSpeed + p.phx) * p.amp;
      const ty = p.oy + Math.cos(t*idleSpeed + p.phy) * p.amp;
      const dx = (mouse.x - p.x);
      const dy = (mouse.y - p.y);
      const dist2 = dx*dx+dy*dy + 0.001;
      const influence = Math.max(0, 1 - Math.sqrt(dist2)/400);
      // spring to drifting target + mouse influence
      p.vx += (tx - p.x)*0.02 + (dx/dist2)*20*influence*mouseStrength;
      p.vy += (ty - p.y)*0.02 + (dy/dist2)*20*influence*mouseStrength;
      p.vx *= 0.85; p.vy *= 0.85;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
    }
  }

  function draw(){
    ctx.clearRect(0,0,width,height);
    const g = ctx.createRadialGradient(width*0.2,height*0.2,50,width*0.6,height*0.6,Math.max(width,height));
    g.addColorStop(0,'#00121b');
    g.addColorStop(1,'#000814');
    ctx.fillStyle = g;
    ctx.fillRect(0,0,width,height);

    for (let i=0;i<points.length;i++){
      const p = points[i];
      const px = p.x, py = p.y;
      for (let k=0;k<p.neighbors.length;k++){
        const q = points[p.neighbors[k]];
        const dx = px - q.x, dy = py - q.y;
        const dist = Math.sqrt(dx*dx+dy*dy);
        const alpha = Math.max(0, 0.12 - (dist/1000));
        if (alpha<=0) continue;
        ctx.beginPath();
        ctx.moveTo(px,py);
        ctx.lineTo(q.x,q.y);
        ctx.strokeStyle = lineColorBase + (alpha*1.2) + ')';
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    for (let i=0;i<points.length;i++){
      const p = points[i];
      const dx = mouse.x - p.x, dy = mouse.y - p.y;
      const dist = Math.sqrt(dx*dx+dy*dy);
      const alpha = Math.max(0, 0.25 - dist/1200);
      if (alpha>0.02){
        ctx.beginPath();
        ctx.arc(p.x,p.y,2.2,0,Math.PI*2);
        ctx.fillStyle = nodeColorBase + (Math.min(alpha,0.9)) + ')';
        ctx.fill();
      }
    }
  }

  let last = performance.now();
  function loop(now){
    const dt = Math.min(1, (now-last)/16);
    last = now;
    update(dt);
    draw();
    requestAnimationFrame(loop);
  }

  function onMove(e){
    const rect = canvas.getBoundingClientRect();
    mouse.x = (e.clientX - rect.left) * (canvas.width/canvas.clientWidth) / dpr;
    mouse.y = (e.clientY - rect.top) * (canvas.height/canvas.clientHeight) / dpr;
  }
  function onLeave(){ mouse.x = -9999; mouse.y = -9999; }

  window.addEventListener('resize', resize);
  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('mouseleave', onLeave);
  document.addEventListener('mousemove', onMove);

  resize();
  requestAnimationFrame(loop);

  return {
    destroy(){
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('mousemove', onMove);
      canvas.removeEventListener('mouseleave', onLeave);
      document.removeEventListener('mousemove', onMove);
      // no further cleanup of RAF (could be added)
    }
  };
}
