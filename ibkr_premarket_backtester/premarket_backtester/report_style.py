"""CSS + JS for the interactive report, kept out of the builder so the markup
code stays readable.

Dark mode is SELECTED, not an auto-flip: the dark values are the palette's own
dark steps (validated against the dark surface #1a1a19), declared under both the
prefers-color-scheme media query (OS setting) and a :root[data-theme] scope (the
in-page toggle), with the toggle winning both ways.
"""

CSS = """
:root{
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7;
  --dip:#2a78d6; --wedge:#eb6834; --combined:#1baf7a;
  --good:#0ca30c; --bad:#d03b3b; --warn:#fab219;
  --card:#ffffff; --line:#e6e5df;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835;
  --dip:#3987e5; --wedge:#d95926; --combined:#199e70;
  --card:#1f1f1e; --line:#2c2c2a;
}}
:root[data-theme="dark"]{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835;
  --dip:#3987e5; --wedge:#d95926; --combined:#199e70;
  --card:#1f1f1e; --line:#2c2c2a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;margin:34px 0 12px;letter-spacing:.02em;text-transform:uppercase;color:var(--ink2);
  font-weight:600;border-bottom:1px solid var(--line);padding-bottom:7px}
h3{font-size:13px;margin:0 0 10px;color:var(--ink2);font-weight:600}
.muted{color:var(--muted)}
.sm{font-size:12px}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chip{font-size:11px;padding:3px 9px;border-radius:99px;background:var(--card);
  border:1px solid var(--line);color:var(--ink2);white-space:nowrap}
.chip .sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(152px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.tile .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .val{font-size:20px;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.tile .sub{font-size:11px;color:var(--muted);margin-top:2px}
.tile.good .val{color:var(--good)} .tile.bad .val{color:var(--bad)} .tile.warn .val{color:var(--warn)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:12px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:860px){.row{grid-template-columns:1fr}}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}
.base{stroke:var(--axis);stroke-width:1;stroke-dasharray:3 3}
.zero{stroke:var(--axis);stroke-width:1}
.axl{fill:var(--muted);font-size:10px;font-family:inherit}
.eqline{fill:none;stroke:var(--combined);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ddfill{fill:var(--bad);opacity:.13}
.bar-good{fill:var(--good)} .bar-bad{fill:var(--bad)}
.lev-dip{fill:var(--dip);stroke:var(--card);stroke-width:2}
.lev-wedge{fill:var(--wedge);stroke:var(--card);stroke-width:2}
.cap{stroke:var(--bad);stroke-width:1.5;stroke-dasharray:5 3}
.cap-lab{fill:var(--bad)}
.onex{stroke:var(--axis);stroke-width:1;stroke-dasharray:2 3}
.xhair{stroke:var(--axis);stroke-width:1}
.dot{fill:var(--combined);stroke:var(--card);stroke-width:2}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--plane);
  font-size:11px;padding:5px 8px;border-radius:6px;white-space:nowrap;z-index:20;
  font-variant-numeric:tabular-nums}
.chartbox{position:relative}
.hbars{display:flex;flex-direction:column;gap:7px}
.hb{display:grid;grid-template-columns:150px 1fr 84px;align-items:center;gap:9px;font-size:12px}
.hb-l{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hb-t{background:var(--grid);height:12px;border-radius:3px;overflow:hidden}
.hb-f{height:100%;border-radius:3px}
.hb-dip{background:var(--dip)} .hb-wedge{background:var(--wedge)}
.hb-good{background:var(--good)} .hb-bad{background:var(--bad)}
.hb-neutral{background:var(--combined)}
.hb-r{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.03em;position:sticky;top:0;background:var(--card);z-index:2}
th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:color-mix(in srgb,var(--grid) 45%,transparent)}
.pos{color:var(--good)} .neg{color:var(--bad)}
.scroll{overflow-x:auto;max-height:660px;overflow-y:auto;border:1px solid var(--line);border-radius:10px}
.tagd{color:var(--dip);font-weight:600} .tagw{color:var(--wedge);font-weight:600}
.ctrls{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}
button.f{font:inherit;font-size:12px;padding:5px 12px;border-radius:99px;cursor:pointer;
  border:1px solid var(--line);background:var(--card);color:var(--ink2)}
button.f[aria-pressed="true"]{background:var(--ink);color:var(--plane);border-color:var(--ink)}
.note{font-size:12px;color:var(--muted);margin:6px 0 0}
.callout{border-left:3px solid var(--warn);padding:9px 12px;background:var(--card);
  border-radius:0 8px 8px 0;font-size:12.5px;margin-top:10px;color:var(--ink2)}
.callout b{color:var(--ink)}
.callout.crit{border-left-color:var(--bad)}
details{margin-top:10px}summary{cursor:pointer;font-size:12px;color:var(--muted);padding:4px 0}
.theme{font:inherit;font-size:12px;padding:5px 11px;border-radius:8px;border:1px solid var(--line);
  background:var(--card);color:var(--ink2);cursor:pointer}
"""

JS = """
(function(){
  var b=document.getElementById('themeBtn');
  if(b) b.onclick=function(){
    var d=document.documentElement, cur=d.getAttribute('data-theme');
    var next = cur==='dark' ? 'light' : (cur==='light' ? 'dark' :
      (matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark'));
    d.setAttribute('data-theme',next);
    b.textContent = next==='dark' ? 'Light mode' : 'Dark mode';
  };
  document.querySelectorAll('svg.equity').forEach(function(svg){
    var layer=svg.querySelector('.hoverlayer'); if(!layer) return;
    var pts=JSON.parse(layer.dataset.pts||'[]');
    var xh=layer.querySelector('.xhair'), dot=layer.querySelector('.dot');
    var box=svg.closest('.chartbox'), tip=box&&box.querySelector('.tip');
    if(!pts.length||!tip) return;
    svg.addEventListener('mousemove',function(e){
      var r=svg.getBoundingClientRect(), vb=svg.viewBox.baseVal;
      var x=(e.clientX-r.left)/r.width*vb.width;
      var best=pts[0],bd=1e9;
      for(var i=0;i<pts.length;i++){var d=Math.abs(pts[i][0]-x); if(d<bd){bd=d;best=pts[i];}}
      xh.setAttribute('x1',best[0]);xh.setAttribute('x2',best[0]);
      xh.setAttribute('y1',0);xh.setAttribute('y2',vb.height);xh.style.display='';
      dot.setAttribute('cx',best[0]);dot.setAttribute('cy',best[1]);dot.style.display='';
      tip.hidden=false;
      tip.textContent='trade '+best[3]+'  \\u00b7  $'+best[2].toLocaleString();
      var px=best[0]/vb.width*r.width, py=best[1]/vb.height*r.height;
      tip.style.left=Math.min(Math.max(px+10,4),r.width-tip.offsetWidth-4)+'px';
      tip.style.top=Math.max(py-30,2)+'px';
    });
    svg.addEventListener('mouseleave',function(){
      xh.style.display='none';dot.style.display='none';tip.hidden=true;});
  });
  var btns=document.querySelectorAll('button.f[data-filter]');
  btns.forEach(function(btn){ btn.onclick=function(){
    var f=btn.dataset.filter;
    btns.forEach(function(o){o.setAttribute('aria-pressed', o===btn ? 'true':'false');});
    document.querySelectorAll('#tradeTable tbody tr').forEach(function(tr){
      tr.style.display = (f==='all'||tr.dataset.mode===f) ? '' : 'none';
    });
  };});
})();
"""
