// PageFX: lightweight full-screen noise wipe transition for multi-page apps
// Usage:
//   const fx = initPageFX();
//   fx.bindAnchors();         // intercept in-site links and play outro before nav
//   fx.playIntro();           // call on page load to reveal content with animation
// Options: data attributes on #page-transition: data-duration (ms), data-color

(function(){
  function create(elTag, props){ const el = document.createElement(elTag); if(props){ Object.assign(el, props); } return el; }

  function sameOrigin(href){
    try{ const u = new URL(href, window.location.href); return u.origin === window.location.origin; }catch{ return false; }
  }

  function makeNoiseCanvas(w, h){
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(w, h);
    const data = img.data;
    for(let i=0, len=w*h; i<len; i++){
      const v = Math.floor(Math.random()*256);
      const o = i*4; data[o]=v; data[o+1]=v; data[o+2]=v; data[o+3]=255;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  function lerp(a,b,t){ return a + (b-a)*t; }

  function initPageFX(){
    // Respect user preference for reduced motion
    const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion){
      return {
        bindAnchors: function(){},
        playIntro: function(){},
        cover: function(){},
        reveal: function(){}
      };
    }
    // Deduplicate overlays; keep the first one
    let overlay = document.getElementById('page-transition');
    if(!overlay){
      overlay = create('div');
      overlay.id = 'page-transition';
      overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.35);backdrop-filter:blur(6px);z-index:9999;opacity:0;pointer-events:none;transition:opacity .25s ease;';
      document.body.appendChild(overlay);
    }
    const overlays = document.querySelectorAll('#page-transition');
    if (overlays.length > 1){
      for (let i=1;i<overlays.length;i++){ overlays[i].remove(); }
    }
    overlay.classList.remove('show');

  const color = overlay.getAttribute('data-color') || '#000714';
  const duration = parseInt(overlay.getAttribute('data-duration')||'520',10);
  const wantGL = (overlay.getAttribute('data-webgl') || '1') !== '0';
  // 강제로 블러 전환(Simple)만 사용
  const simple = true;

    // Canvas setup (low-res for perf, scaled up for stylized look)
    // Simple CSS mode: no canvas, just overlay fade
    if (simple){
      function showOverlay(show){
        overlay.classList.toggle('show', !!show);
        overlay.style.pointerEvents = show ? 'auto' : 'none';
      }
      function animateCover(toCovered){
        return new Promise(resolve=>{
          document.body.classList.add('pagefxing');
          showOverlay(true);
          // use CSS transition timing (~duration)
          setTimeout(()=>{
            resolve();
          }, duration);
        });
      }
      function playIntro(){
        // Reveal on first load
        showOverlay(true);
        return animateCover(false).then(()=>{
          showOverlay(false);
          document.body.classList.remove('pagefxing');
        });
      }
  // Safety: auto-hide overlay after a grace period
  setTimeout(()=>{ overlay.classList.remove('show'); overlay.style.pointerEvents='none'; document.body.classList.remove('pagefxing'); }, Math.max(800, duration+300));
  const api = {
        bindAnchors: function(){ if (window.Turbo) return; /* no-op when Turbo */ },
        playIntro,
        cover: ()=>animateCover(true).then(()=>{
          // keep overlay shown during cover; click should be blocked only during cover
        }),
        reveal: ()=>{
          return new Promise(resolve=>{
            overlay.classList.remove('show');
            overlay.style.pointerEvents = 'none';
            document.body.classList.remove('pagefxing');
            resolve();
          });
        }
  };
  // Turbo 환경에서는 partial-transitions가 reveal을 호출하므로 별도 바인딩 생략
  return api;
    }

    const cvs = create('canvas');
    overlay.appendChild(cvs);
    // Try WebGL first if requested
    let gl = null, useGL = false;
    if (wantGL) {
      try { gl = cvs.getContext('webgl2', {alpha:true, premultipliedAlpha:true}); } catch {}
      if (!gl) { try { gl = cvs.getContext('webgl', {alpha:true, premultipliedAlpha:true}); } catch {}
      }
      useGL = !!gl;
    }
    const ctx = useGL ? null : cvs.getContext('2d');
    const baseW = 480, baseH = 270; // ~16:9 low-res mask
    cvs.width = baseW; cvs.height = baseH;
  cvs.style.cssText = 'position:absolute;inset:0;margin:auto;width:100%;height:100%;z-index:0;pointer-events:none;';
    const noise = makeNoiseCanvas(baseW, baseH);
  // Offscreen mask canvas for soft edges
    const maskCvs = useGL ? null : create('canvas');
    const maskCtx = useGL ? null : (function(){ if(!maskCvs) return null; maskCvs.width = baseW; maskCvs.height = baseH; return maskCvs.getContext('2d'); })();

  // No light streak; keep visuals minimal (blur + spinner + soft dissolve)

    // Helpers
    function hexToRgb(hex){
      const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex||'#000000');
      return m ? [parseInt(m[1],16)/255, parseInt(m[2],16)/255, parseInt(m[3],16)/255] : [0,0,0];
    }

    // WebGL pipeline
  let prog = null, aPosLoc = -1, uTLoc = null, uTimeLoc = null, uColorLoc = null, uNoiseLoc = null, uNoiseScaleLoc = null;
    let vb = null, noiseTex = null;
    let t0 = performance.now();
    if (useGL) {
      try {
        const vsSrc = `
          attribute vec2 aPos; varying vec2 vUV;
          void main(){ vUV = (aPos + 1.0)*0.5; gl_Position = vec4(aPos,0.0,1.0); }
        `;
        const fsSrc = `
          precision mediump float; varying vec2 vUV; uniform float uT; uniform float uTime; uniform vec3 uColor; uniform sampler2D uNoise; uniform float uNoiseScale;
          void main(){
            vec2 uv = vUV;
            float n = texture2D(uNoise, uv * uNoiseScale).r; // 0..1
            float softness = 0.08;
            float cover = 1.0 - smoothstep(uT - softness, uT + softness, n);
            gl_FragColor = vec4(uColor, cover);
          }
        `;
        function compile(type, src){ const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh); if(!gl.getShaderParameter(sh, gl.COMPILE_STATUS)){ throw new Error(gl.getShaderInfoLog(sh)||'shader compile failed'); } return sh; }
        const vs = compile(gl.VERTEX_SHADER, vsSrc);
        const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
        prog = gl.createProgram(); gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
        if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog)||'program link failed');
        aPosLoc = gl.getAttribLocation(prog, 'aPos');
        uTLoc = gl.getUniformLocation(prog, 'uT');
        uTimeLoc = gl.getUniformLocation(prog, 'uTime');
        uColorLoc = gl.getUniformLocation(prog, 'uColor');
        uNoiseLoc = gl.getUniformLocation(prog, 'uNoise');
        uNoiseScaleLoc = gl.getUniformLocation(prog, 'uNoiseScale');

        // Fullscreen quad (triangle strip)
        vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
        const quad = new Float32Array([
          -1,-1, 1,-1, -1,1, 1,1
        ]);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

        // Noise texture
        const nctx = noise.getContext('2d');
        const imgData = nctx.getImageData(0,0,noise.width, noise.height);
        noiseTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, noiseTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, noise.width, noise.height, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, (function(){
          // extract single channel
          const src = imgData.data; const len = noise.width*noise.height; const buf = new Uint8Array(len);
          for (let i=0;i<len;i++){ buf[i] = src[i*4]; }
          return buf;
        })());
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      } catch (e) {
        console.warn('WebGL transition disabled:', e);
        useGL = false; gl = null;
      }
    }

    function drawMask(t){
      if (useGL && gl && prog) {
        gl.viewport(0,0,baseW,baseH);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(prog);
        gl.bindBuffer(gl.ARRAY_BUFFER, vb);
        gl.enableVertexAttribArray(aPosLoc);
        gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, 0, 0);
        const rgb = hexToRgb(color);
        gl.uniform1f(uTLoc, t);
        gl.uniform1f(uTimeLoc, (performance.now()-t0)/1000.0);
        gl.uniform3f(uColorLoc, rgb[0], rgb[1], rgb[2]);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, noiseTex);
        gl.uniform1i(uNoiseLoc, 0);
        gl.uniform1f(uNoiseScaleLoc, 1.6);
  // no streak uniform
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        return;
      }
      // t: 0 -> reveal, 1 -> fully covered
      // We map noise threshold so that pixels with noise < thresh are covered
      const nctx = noise.getContext('2d');
      const nData = nctx.getImageData(0,0,baseW,baseH).data;
      const img = maskCtx.createImageData(baseW, baseH);
      const out = img.data;
      const thresh = lerp(0, 255, t);
      for(let i=0;i<baseW*baseH;i++){
        const v = nData[i*4];
        const a = v < thresh ? 255 : Math.floor(lerp(0,160,(thresh/255))); // softer tail
        const o = i*4;
        out[o] = 0; out[o+1]=0; out[o+2]=0; out[o+3]=a;
      }
      maskCtx.putImageData(img,0,0);
      // Clear main canvas then draw blurred mask for smoother edges
      ctx.clearRect(0,0,baseW,baseH);
      ctx.save();
      ctx.filter = 'blur(0.8px)';
      ctx.drawImage(maskCvs,0,0);
      ctx.restore();

  // No light streak overlay (keep visuals minimal)

      // Tint to desired color
  ctx.save();
      ctx.globalCompositeOperation = 'destination-over';
      ctx.fillStyle = color;
      ctx.fillRect(0,0,baseW,baseH);
      ctx.restore();
    }

    function showOverlay(show){
      overlay.style.opacity = show ? '1' : '0';
      overlay.style.pointerEvents = show ? 'auto' : 'none';
    }

    function animateCover(toCovered){
      return new Promise(resolve=>{
        const start = performance.now();
        const from = toCovered ? 0 : 1;
        const to = toCovered ? 1 : 0;
        document.body.classList.add('pagefxing');
        function frame(now){
          const e = Math.min(1, (now - start) / duration);
          // easeInOutCubic for smoother accel/decel
          const eased = e < 0.5 ? 4*e*e*e : 1 - Math.pow(-2*e + 2, 3)/2;
          const t = lerp(from, to, eased);
          drawMask(t);
          showOverlay(true);
          if(e < 1) requestAnimationFrame(frame); else resolve();
        }
        requestAnimationFrame(frame);
      });
    }

    function bindAnchors(){
      // If Turbo is present, let Turbo handle navigation
      if (window.Turbo) return;
      document.addEventListener('click', (ev)=>{
        const a = ev.target.closest && ev.target.closest('a');
        if(!a) return;
        // Respect modifier/middle clicks
        if (ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
        const href = a.getAttribute('href') || '';
        const target = a.getAttribute('target') || '';
        const download = a.hasAttribute('download');
        if (ev.defaultPrevented) return;
        if (!href || href.startsWith('#') || download || target==='_blank') return;
        if (!sameOrigin(href)) return;
        const current = window.location.pathname + window.location.search + window.location.hash;
        if (href === current) return; // no-op for same URL
        ev.preventDefault();
        let navigated = false;
        const go = ()=>{ if(!navigated){ navigated = true; window.location.href = href; } };
        // Fallback in case animation errors or stalls
        setTimeout(go, (duration||420) + 300);
        Promise.resolve()
          .then(()=>animateCover(true))
          .then(go)
          .catch(go);
      }, true);
    }

    async function playIntro(){
      // Start fully covered then reveal
      drawMask(1);
      showOverlay(true);
      await animateCover(false);
      showOverlay(false);
      // Stagger removal so blur fades out after overlay starts fading
      requestAnimationFrame(()=>{
        setTimeout(()=>{
          document.body.classList.remove('pagefxing');
        }, 120);
      });
    }

  // Safety: auto-hide overlay after a grace period
  setTimeout(()=>{ showOverlay(false); document.body.classList.remove('pagefxing'); }, Math.max(800, duration+300));
  const api = {
      bindAnchors,
      playIntro,
      cover: ()=>animateCover(true),
      reveal: ()=>{
        return animateCover(false).then(()=>{
          showOverlay(false);
          document.body.classList.remove('pagefxing');
        }).catch(()=>{
          // Safety: ensure overlay is hidden even on error
          showOverlay(false);
          document.body.classList.remove('pagefxing');
        });
      }
  };
  // Turbo 환경에서는 partial-transitions가 reveal을 호출하므로 별도 바인딩 생략
  return api;
  }

  // expose
  window.initPageFX = initPageFX;
})();
