// Card-only partial transitions with Turbo Drive
// Keeps header/nav steady and animates only .page-cards region between pages.
(function(){
  if (!window.Turbo) return; // only when Turbo is present
  if (window.__partialTransitionsBound) return; // avoid duplicate bindings across Turbo navigations
  window.__partialTransitionsBound = true;

  function ensureOverlaySafety(){
    const overlays = Array.from(document.querySelectorAll('#page-transition'));
    if (overlays.length === 0) return;
    // Keep the first, hide all, remove extras
    overlays.forEach((ov, idx)=>{
      ov.classList.remove('show');
      ov.style.pointerEvents = 'none';
      if (idx > 0) {
        try { ov.remove(); } catch {}
      }
    });
    document.body.classList.remove('pagefxing');
  }

  function prepEnter(newBody){
    // Avoid pre-hiding new content to prevent blank screens if enter animation fails.
    // Overlay cover will hide the swap; we'll animate in after render.
  }

  function animateExit(oldRoot){
    return new Promise((resolve)=>{
      const oc = oldRoot.querySelector('.page-cards');
      if (!oc) return resolve();
      const anim = oc.animate([
        { opacity: 1, transform: 'none' },
        { opacity: 0, transform: 'translateY(20px) scale(0.994)' }
      ], { duration: 220, easing: 'ease-out' });
      anim.addEventListener('finish', resolve);
      anim.addEventListener('cancel', resolve);
    });
  }

  function animateEnter(){
    return new Promise((resolve)=>{
      const nc = document.querySelector('.page-cards');
      if (!nc) return resolve();
      const anim = nc.animate([
        { opacity: 0, transform: 'translateY(14px) scale(0.996)' },
        { opacity: 1, transform: 'none' }
      ], { duration: 260, easing: 'ease-out' });
      const cleanup = ()=>{ nc.style.opacity = ''; nc.style.transform = ''; resolve(); };
      anim.addEventListener('finish', cleanup);
      anim.addEventListener('cancel', cleanup);
    });
  }

  let watchdog = null;
  document.addEventListener('turbo:before-render', (ev)=>{
    // Defer render until exit animation + overlay cover completes
    ev.preventDefault();
    prepEnter(ev.detail.newBody);
    const cover = (window.__pagefx && __pagefx.cover) ? __pagefx.cover().catch(()=>{}) : Promise.resolve();
    Promise.all([ animateExit(document), cover ]).then(()=>{
      ev.detail.resume();
      // Safety watchdog: ensure we always reveal within a reasonable time
      clearTimeout(watchdog);
      watchdog = setTimeout(()=>{
        if (window.__pagefx && __pagefx.reveal) {
          __pagefx.reveal();
        } else {
          // 강제 폴백: 오버레이와 상태 초기화
          ensureOverlaySafety();
        }
        const nc = document.querySelector('.page-cards');
        if (nc){ nc.style.opacity=''; nc.style.transform=''; }
      }, 2000);
    });
  });

  document.addEventListener('turbo:render', ()=>{
    // Run enter once new content is in place, then reveal overlay
    requestAnimationFrame(()=>{
      animateEnter().then(()=>{
        if (window.__pagefx && __pagefx.reveal) {
          __pagefx.reveal();
        } else {
          // 강제 폴백: 오버레이와 상태 초기화
          ensureOverlaySafety();
        }
        clearTimeout(watchdog);
        // 한 번 더 안전 해제 (프레임 지연으로 잔류하는 경우)
        setTimeout(ensureOverlaySafety, 300);
      });
    });
  });

  // 초기 로드나 캐시 복원 시에도 안전 처리
  document.addEventListener('turbo:load', ensureOverlaySafety);
})();
