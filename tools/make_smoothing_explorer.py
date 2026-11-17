"""Write the interactive smoothing-window explorer, notebooks/smoothing_explorer.html.

Run from the repository folder, with the mcdb5910 environment active:
    python tools/make_smoothing_explorer.py

The explorer goes with NB1, part A (Try it 2). It is a self-contained web page (no Python, no internet needed) in
which a slider changes the Savitzky-Golay window applied to a noisy track of a synthetic run-and-tumble swimmer,
and three charts show the estimated speed against the true speed, the velocity error against the window length,
and the smoothed path around one tumble. The output is deterministic: running the script again gives the same file.

How it is computed:
 1. A synthetic swimmer runs at 25 um/s with a slow wobble of its direction; six tumbles of 0.1-0.2 s slow it to
    5 um/s and turn it by 55-110 degrees. The path is integrated on a fine grid, then sampled at the frame rate,
    so position and velocity agree exactly. Independent Gaussian noise (NOISE_SD_UM) is added to x and y.
 2. For every window and polynomial order, the velocity error splits into a bias (the filter applied to the
    noise-free path) and a noise term (sigma * the norm of the derivative coefficients). Errors are RMS values per
    velocity component, the convention of NB1, and the "raw difference" is the difference of consecutive frames,
    as in NB1 (compared with the true velocity midway between the two frames). The total error is the quadrature
    sum of bias and noise; the best window is where it is smallest. The page re-does this in the browser for the
    noise level chosen with the slider.

To apply the method to your own recordings: measure the localization noise from cells that do not move, decide
the shortest event the analysis must keep (a tumble, about 0.15 s), set FRAME_RATE_HZ and NOISE_SD_UM below, and
choose the window as a duration in seconds before converting it to an odd number of frames at your frame rate.
On real data, check the residual (raw minus smoothed position): if tumbles show in it, the window is too long.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_coeffs, savgol_filter

ROOT = Path(__file__).resolve().parent.parent

FRAME_RATE_HZ = 30.0  # Camera rate of the recordings.
NOISE_SD_UM = 0.2  # Standard deviation of the localization noise on each axis (µm).
EXAMPLE_WINDOW = 7  # The window used in NB1, in frames.
ORDER = 2  # Polynomial order of the filter (NB1 uses 2).
TOO_SMOOTH_WINDOW = 31  # A window that is too long, for comparison.
PATH_ZOOM_S = (4.6, 5.9)  # Time range of the path panel. It contains one tumble.
DURATION_S = 12.0
RANDOM_SEED = 11

dt = 1.0 / FRAME_RATE_HZ

def logistic(t, width_s):
    return 1.0 / (1.0 + np.exp(-t / width_s))


RUN_SPEED = 25.0  # µm/s
TUMBLE_SPEED = 5.0  # µm/s
TUMBLES = [(1.6, 0.15, 75.0), (3.3, 0.12, -60.0), (5.1, 0.20, 110.0), (7.4, 0.15, -85.0), (9.1, 0.10, 55.0), (10.6, 0.18, -100.0)]
# Each tumble: start (s), duration (s), turn (degrees).


def simulate_swimmer(frame_rate_hz, duration_s, oversample=20):
    # Return a smooth run-and-tumble path, sampled at the frame rate, with its exact velocity.
    # The path is integrated on a grid 20 times finer than the frames, so that position and velocity agree.
    t = np.arange(round(duration_s * frame_rate_hz * oversample)) / (frame_rate_hz * oversample)
    speed = np.full_like(t, RUN_SPEED)
    heading_deg = 20.0 * np.sin(2 * np.pi * t / 5.0)  # slow wobble of the direction during runs
    for start_s, length_s, turn_deg in TUMBLES:
        speed -= (RUN_SPEED - TUMBLE_SPEED) * (logistic(t - start_s, 0.02) - logistic(t - start_s - length_s, 0.02))
        heading_deg += turn_deg * logistic(t - start_s - length_s / 2, length_s / 6)

    heading = np.deg2rad(heading_deg)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = cumulative_trapezoid(vx, t, initial=0.0)
    y = cumulative_trapezoid(vy, t, initial=0.0)
    frames = slice(None, None, oversample)
    return {"t": t[frames], "x": x[frames], "y": y[frames], "vx": vx[frames], "vy": vy[frames], "speed": speed[frames]}


cell = simulate_swimmer(FRAME_RATE_HZ, DURATION_S)
unit_noise = np.random.default_rng(RANDOM_SEED).standard_normal((2, cell["t"].size))
x_noisy = cell["x"] + NOISE_SD_UM * unit_noise[0]
y_noisy = cell["y"] + NOISE_SD_UM * unit_noise[1]


WINDOWS = np.arange(5, 42, 2)
ORDERS = (2, 4, 6)
EDGE_FRAMES = int(WINDOWS.max()) // 2
interior = slice(EDGE_FRAMES, -EDGE_FRAMES)


def sg_velocity(x, y, window, order):
    return (
        savgol_filter(x, window, order, deriv=1, delta=dt),
        savgol_filter(y, window, order, deriv=1, delta=dt),
    )


def velocity_rms_error(vx_est, vy_est):
    # RMS error per velocity component (the two components are averaged).
    squared = ((vx_est - cell["vx"]) ** 2 + (vy_est - cell["vy"]) ** 2) / 2
    return float(np.sqrt(np.mean(squared[interior])))


def error_curve(order, noise_sd_um):
    # Rows: window, bias, noise error, total error (µm/s, per velocity component).
    rows = []
    for window in WINDOWS[WINDOWS > order]:
        bias = velocity_rms_error(*sg_velocity(cell["x"], cell["y"], window, order))
        coeffs = savgol_coeffs(window, order, deriv=1, delta=dt, use="dot")
        noise = float(noise_sd_um * np.linalg.norm(coeffs))
        rows.append((window, bias, noise, np.hypot(bias, noise)))
    return np.array(rows)


curve = error_curve(ORDER, NOISE_SD_UM)
best_window = int(curve[np.argmin(curve[:, 3]), 0])

EXPLORER_TEMPLATE = r'''<div id="__UID__">
<style>
#__UID__{background:#ffffff;color:#0b0b0b;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;padding:16px;border:1px solid #e1e0d9;border-radius:12px;max-width:820px;box-sizing:border-box}
#__UID__ .row{display:flex;align-items:center;gap:12px;margin:0 0 10px;flex-wrap:wrap}
#__UID__ .lab{font-size:13px;color:#52514e;min-width:110px}
#__UID__ .val{font-size:13px;font-weight:500;min-width:130px;font-variant-numeric:tabular-nums}
#__UID__ button,#__UID__ select{font:inherit;font-size:13px;padding:5px 12px;border:1px solid #c3c2b7;border-radius:8px;background:#ffffff;color:#0b0b0b;cursor:pointer}
#__UID__ button:hover{background:#f5f4f0}
#__UID__ .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0 6px}
#__UID__ .card{background:#f5f4f0;border-radius:8px;padding:10px 14px}
#__UID__ .card p{margin:0}
#__UID__ .cl{font-size:12px;color:#52514e}
#__UID__ .cv{font-size:20px;font-weight:500;font-variant-numeric:tabular-nums}
#__UID__ .leg{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:#52514e;margin:14px 0 4px}
#__UID__ .leg span{display:flex;align-items:center;gap:5px}
#__UID__ .sw{display:inline-block;width:16px;height:0;border-top-width:2px;border-top-style:solid}
#__UID__ .cap{font-size:13px;color:#52514e;margin:16px 0 2px;font-weight:500}
#__UID__ svg{display:block;width:100%;height:auto}
#__UID__ svg text{font-family:inherit}
#__UID__ .rd{font-size:12px;color:#52514e;min-height:16px;margin:2px 0 0;font-variant-numeric:tabular-nums}
#__UID__ .def{font-size:13px;line-height:1.5;background:#f5f4f0;border-radius:8px;padding:10px 14px;margin:10px 0 4px}
#__UID__ .def p{margin:0 0 6px}
#__UID__ .def ul{margin:0;padding-left:18px}
#__UID__ .def li{margin:0 0 5px}
</style>
<p data-out="msg" style="font-size:13px;color:#a32d2d;margin:0 0 10px">The explorer is not running. In JupyterLab, click <b>Trust HTML</b> in the toolbar above to let the page run its script. Or open smoothing_explorer.html in a web browser, which runs it right away.</p>
<div class="row">
<button data-act="raw">Raw difference</button><button data-act="best">Best window</button><button data-act="smooth">Too smooth</button>
</div>
<div class="row"><span class="lab">Window (frames)</span><input type="range" data-in="w" min="5" max="81" step="2" style="flex:1;min-width:160px"><span class="val" data-out="w"></span></div>
<div class="row"><span class="lab">Polynomial order</span><select data-in="p"><option value="2">2</option><option value="4">4</option><option value="6">6</option></select>
<span class="lab" style="margin-left:12px;min-width:0">Noise &sigma; (&micro;m)</span><input type="range" data-in="s" min="0.05" max="0.5" step="0.05" style="flex:1;min-width:120px"><span class="val" data-out="s" style="min-width:40px"></span></div>
<div class="cards">
<div class="card"><p class="cl">Raw velocity error</p><p class="cv" data-out="raw">&ndash;</p></div>
<div class="card"><p class="cl">Current velocity error</p><p class="cv" data-out="cur">&ndash;</p></div>
<div class="card"><p class="cl">Best window for these settings</p><p class="cv" data-out="best">&ndash;</p></div>
</div>
<p class="cap">Speed over time</p>
<div class="leg"><span><i class="sw" style="border-color:#0b0b0b"></i>True speed</span><span><i class="sw" style="border-color:#eb6834;border-top-style:dotted"></i>Raw difference</span><span><i class="sw" style="border-color:#2a78d6"></i>Current estimate</span></div>
<svg data-chart="spd" viewBox="0 0 780 230" role="img" aria-label="True, raw and filtered speed over time"></svg>
<p class="rd" data-out="rd-spd"></p>
<div class="def">
<p><b>Mean squared error of each velocity component = bias&sup2; + noise&sup2;</b> (the chart shows its square root, the RMS error, and the best window is its minimum). The raw difference is (<b>r</b><sub>n+1</sub> &minus; <b>r</b><sub>n</sub>) / &Delta;t, as in NB1.</p>
<p><b>The fit:</b> around frame n, the w positions in the window are fitted by least squares with a polynomial of order p in the time s = t &minus; t<sub>n</sub>,<br>
<b>r</b>(s) &asymp; <b>a</b><sub>0</sub> + <b>a</b><sub>1</sub> s + <b>a</b><sub>2</sub> s&sup2; + &hellip; + <b>a</b><sub>p</sub> s<sup>p</sup>.<br>
The smoothed position is <b>a</b><sub>0</sub> and the velocity is the slope at the center, <b>v</b><sub>n</sub> = <b>a</b><sub>1</sub>. Because the fit is linear in the positions, this slope is a weighted sum <b>v</b><sub>n</sub> = &Sigma;<sub>i</sub> c<sub>i</sub> <b>r</b><sub>n+i</sub> over i = &minus;(w&minus;1)/2 &hellip; (w&minus;1)/2, with weights c<sub>i</sub> that depend only on w, p and &Delta;t.</p>
<ul>
<li><b>Bias:</b> the error of the fit on the noise-free path. The polynomial blurs tumbles, so the bias grows with the window.</li>
<li><b>Noise:</b> the localization noise passed through the fit, &sigma; &radic;&Sigma;<sub>i</sub> c<sub>i</sub>&sup2; per component (for the raw difference, &radic;2 &sigma;/&Delta;t). Averaging more frames shrinks it.</li>
</ul>
</div>
<p class="cap">Velocity error against window length</p>
<div class="leg"><span><i class="sw" style="border-color:#eb6834;border-top-style:dashed"></i>Bias</span><span><i class="sw" style="border-color:#1baf7a;border-top-style:dotted"></i>Noise</span><span><i class="sw" style="border-color:#2a78d6"></i>Total RMS error</span><span><i class="sw" style="border-color:#898781;border-top-style:dashed"></i>Raw difference</span></div>
<svg data-chart="err" viewBox="0 0 780 230" role="img" aria-label="Bias, noise and total velocity error against window length"></svg>
<p class="rd" data-out="rd-err"></p>
<p class="cap">Path around one quick turn</p>
<div class="leg"><span><i class="sw" style="border-color:#0b0b0b"></i>True path</span><span><i class="sw" style="border-color:#898781;border-top-style:dotted"></i>Noisy samples</span><span><i class="sw" style="border-color:#2a78d6"></i>Smoothed path</span></div>
<svg data-chart="path" viewBox="0 0 780 300" role="img" aria-label="True path, noisy samples and smoothed path"></svg>
</div>
<script>
(function(){
const D=__DATA__;
const root=document.getElementById("__UID__");
const q=s=>root.querySelector(s);
q('[data-out="msg"]').style.display="none";
const C={ink:"#0b0b0b",blue:"#2a78d6",org:"#eb6834",aq:"#1baf7a",mut:"#898781",grid:"#e1e0d9"};
const NS="http://www.w3.org/2000/svg";
function el(tag,a,p){const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function ticks(lo,hi,n){const raw=(hi-lo)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),f=raw/mag,st=(f<1.5?1:f<3?2:f<7?5:10)*mag,out=[];
 for(let k=Math.ceil(lo/st-1e-9);k*st<=hi+1e-9*st;k++)out.push(k*st);return out;}
const fmt=v=>String(+v.toPrecision(6));
// Draw one chart into an SVG element. The SVG has no external dependency.
function chart(svg,o){
 const vb=svg.viewBox.baseVal,w=vb.width,h=vb.height,m={l:54,r:14,t:10,b:38},pw=w-m.l-m.r,ph=h-m.t-m.b;
 let [x0,x1]=o.x,[y0,y1]=o.y;
 if(o.equal){const s=Math.max((x1-x0)/pw,(y1-y0)/ph),cx=(x0+x1)/2,cy=(y0+y1)/2;x0=cx-s*pw/2;x1=cx+s*pw/2;y0=cy-s*ph/2;y1=cy+s*ph/2;}
 const X=v=>m.l+(v-x0)/(x1-x0)*pw,Y=v=>m.t+(1-(v-y0)/(y1-y0))*ph;
 svg.textContent="";
 const id=root.id+"-clip-"+svg.dataset.chart;
 el("rect",{x:m.l,y:m.t,width:pw,height:ph},el("clipPath",{id},el("defs",{},svg)));
 const g=el("g",{},svg);
 for(const v of ticks(y0,y1,5)){el("line",{x1:m.l,x2:m.l+pw,y1:Y(v),y2:Y(v),stroke:C.grid},g);el("text",{x:m.l-6,y:Y(v)+4,"text-anchor":"end","font-size":11,fill:C.mut},g).textContent=fmt(v);}
 for(const v of ticks(x0,x1,o.nx||8)){el("line",{x1:X(v),x2:X(v),y1:m.t,y2:m.t+ph,stroke:C.grid},g);el("text",{x:X(v),y:m.t+ph+15,"text-anchor":"middle","font-size":11,fill:C.mut},g).textContent=fmt(v);}
 el("text",{x:m.l+pw/2,y:h-4,"text-anchor":"middle","font-size":11,fill:C.mut},g).textContent=o.xl;
 el("text",{x:12,y:m.t+ph/2,"text-anchor":"middle","font-size":11,fill:C.mut,transform:"rotate(-90 12 "+(m.t+ph/2)+")"},g).textContent=o.yl;
 const pg=el("g",{"clip-path":"url(#"+id+")"},svg);
 for(const s of o.series){
  if(s.dot){for(const p of s.pts)if(p)el("circle",{cx:X(p[0]),cy:Y(p[1]),r:s.r||1.6,fill:s.color,stroke:s.ring||"none","stroke-width":s.ring?2:0},pg);continue;}
  let d="",pen=false;
  for(const p of s.pts){if(!p){pen=false;continue;}d+=(pen?"L":"M")+X(p[0]).toFixed(1)+" "+Y(p[1]).toFixed(1);pen=true;}
  el("path",{d,fill:"none",stroke:s.color,"stroke-width":s.width||2,"stroke-dasharray":s.dash||"none","stroke-linejoin":"round","stroke-linecap":"round"},pg);
 }
 if(!o.hover)return;
 const out=o.hover.out,cross=el("line",{y1:m.t,y2:m.t+ph,stroke:C.mut,visibility:"hidden"},svg);
 const hit=el("rect",{x:m.l,y:m.t,width:pw,height:ph,fill:"transparent"},svg);
 out.textContent=o.hover.idle;
 hit.addEventListener("mousemove",ev=>{const r=svg.getBoundingClientRect(),xs=(ev.clientX-r.left)*w/r.width,res=o.hover.fn(x0+(xs-m.l)/pw*(x1-x0));
  cross.setAttribute("x1",X(res.x));cross.setAttribute("x2",X(res.x));cross.setAttribute("visibility","visible");out.textContent=res.text;});
 hit.addEventListener("mouseleave",()=>{cross.setAttribute("visibility","hidden");out.textContent=o.hover.idle;});
}
const n=D.t.length,H=D.edge,fps=D.frameRate,dt=1/fps;
const filt=(a,c)=>{const h=(c.length-1)/2,o=new Array(n).fill(null);for(let t=H;t<n-H;t++){let s=0;for(let i=0;i<c.length;i++)s+=c[i]*a[t-h+i];o[t]=s;}return o;};
const rms=a=>{let s=0,k=0;for(const v of a){if(v!==null){s+=v;k++;}}return Math.sqrt(s/k);};
// RMS error per velocity component. mid: the estimate sits midway between frames i and i+1 (raw difference).
const verr=(fx,fy,mid)=>rms(fx.map((v,i)=>{if(v===null)return null;const tx=mid?(D.vx[i]+D.vx[i+1])/2:D.vx[i],ty=mid?(D.vy[i]+D.vy[i+1])/2:D.vy[i];return ((v-tx)**2+(fy[i]-ty)**2)/2;}));
const fwd=a=>{const o=new Array(n).fill(null);for(let t=H;t<n-H;t++)o[t]=(a[t+1]-a[t])/dt;return o;};
const st={w:D.defaults.window,p:D.defaults.order,s:D.defaults.noise,raw:false,best:D.defaults.window};
const W=q('[data-in="w"]'),P=q('[data-in="p"]'),S=q('[data-in="s"]');
W.min=D.wmin;W.max=D.wmax;
W.value=st.w;P.value=String(st.p);S.value=st.s;
const [P0,P1]=D.pathFrames;
const IDLE="Move the pointer over the chart to read the values.";
function curve(){return D.curves[st.p].map(([w,b,cn])=>{const nz=st.s*cn;return{w,b,nz,tot:Math.hypot(b,nz)};});}
function render(){
 const nx=D.x.map((v,i)=>v+st.s*D.ex[i]),ny=D.y.map((v,i)=>v+st.s*D.ey[i]);
 const cv=curve(),best=cv.reduce((a,b)=>b.tot<a.tot?b:a);st.best=best.w;
 const rx=fwd(nx),ry=fwd(ny),rawE=verr(rx,ry,true);
 let fx=rx,fy=ry,px=null,py=null;
 if(!st.raw){const c=D.coeffs[st.p][st.w];fx=filt(nx,c[1]);fy=filt(ny,c[1]);px=filt(nx,c[0]);py=filt(ny,c[0]);}
 const curE=verr(fx,fy,st.raw);
 q('[data-out="w"]').textContent=st.raw?"raw difference":st.w+" ("+(st.w/fps).toFixed(2)+" s)";
 q('[data-out="s"]').textContent=st.s.toFixed(2);
 q('[data-out="raw"]').textContent=rawE.toFixed(2)+" µm/s";
 q('[data-out="cur"]').textContent=curE.toFixed(2)+" µm/s";
 q('[data-out="best"]').textContent=best.w+" frames · "+(best.w/fps).toFixed(2)+" s";
 const rs=rx.map((v,i)=>v===null?null:Math.hypot(v,ry[i])),es=fx.map((v,i)=>v===null?null:Math.hypot(v,fy[i]));
 const pts=a=>a.map((v,i)=>v===null?null:[D.t[i],v]);
 const spdSeries=[{pts:rs.map((v,i)=>v===null?null:[D.t[i]+dt/2,v]),color:C.org,width:1,dash:"2 2"},{pts:pts(D.speed),color:C.ink,width:2}];
 if(!st.raw)spdSeries.push({pts:pts(es),color:C.blue,width:2});
 const top=Math.ceil(Math.max(30,...rs.filter(v=>v!==null))/5)*5;
 chart(q('[data-chart="spd"]'),{x:[D.t[0],D.t[n-1]],y:[0,top],nx:10,xl:"time (s)",yl:"speed (µm/s)",series:spdSeries,
  hover:{out:q('[data-out="rd-spd"]'),idle:IDLE,fn:xv=>{const i=Math.min(n-1,Math.max(0,Math.round(xv*fps)));
   const f=v=>v===null?"-":v.toFixed(1);
   return{x:D.t[i],text:"t = "+D.t[i].toFixed(2)+" s · true "+f(D.speed[i])+" · raw "+f(rs[i])+(st.raw?"":" · current "+f(es[i]))+" µm/s"};}}});
 const cur=cv.find(o=>o.w===st.w);
 const errSeries=[
  {pts:[[D.wmin,rawE],[D.wmax,rawE]],color:C.mut,width:1,dash:"6 4"},
  {pts:cv.map(o=>[o.w,o.b]),color:C.org,dash:"6 4"},
  {pts:cv.map(o=>[o.w,o.nz]),color:C.aq,dash:"2 3"},
  {pts:cv.map(o=>[o.w,o.tot]),color:C.blue}];
 if(!st.raw&&cur)errSeries.push({dot:true,pts:[[st.w,cur.tot]],color:C.blue,r:6,ring:"#ffffff"});
 chart(q('[data-chart="err"]'),{x:[D.wmin,D.wmax],y:[0,Math.ceil(Math.max(rawE,cv[cv.length-1].tot)*1.1)],xl:"window (frames)",yl:"RMS error per component (µm/s)",series:errSeries,
  hover:{out:q('[data-out="rd-err"]'),idle:IDLE,fn:xv=>{const o=cv.reduce((a,b)=>Math.abs(b.w-xv)<Math.abs(a.w-xv)?b:a);
   return{x:o.w,text:"window "+o.w+" frames ("+(o.w/fps).toFixed(2)+" s) · bias "+o.b.toFixed(2)+" · noise "+o.nz.toFixed(2)+" · total "+o.tot.toFixed(2)+" µm/s"};}}});
 const seg=(a,b)=>a.slice(P0,P1).map((v,i)=>[v,b[i+P0]]);
 const truePath=seg(D.x,D.y),noisyPath=seg(nx,ny),all=truePath.concat(noisyPath);
 const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]),pad=0.3;
 const pathSeries=[{dot:true,pts:noisyPath,color:C.mut,r:1.8},{pts:truePath,color:C.ink,width:2}];
 if(!st.raw)pathSeries.push({pts:seg(px,py),color:C.blue,width:2});
 chart(q('[data-chart="path"]'),{x:[Math.min(...xs)-pad,Math.max(...xs)+pad],y:[Math.min(...ys)-pad,Math.max(...ys)+pad],equal:true,xl:"x (µm)",yl:"y (µm)",series:pathSeries});
}
function setW(w){st.w=Math.min(D.wmax,Math.max(w,D.wmin,st.p+1));W.value=st.w;}
W.oninput=()=>{st.raw=false;setW(+W.value);render();};
P.onchange=()=>{st.p=+P.value;st.raw=false;setW(st.w);render();};
S.oninput=()=>{st.s=+S.value;render();};
root.querySelectorAll("button").forEach(b=>b.onclick=()=>{const a=b.dataset.act;
 if(a==="raw"){st.raw=true;}
 else if(a==="best"){st.raw=false;render();setW(st.best);}
 else{st.raw=false;setW(D.defaults.tooSmooth);}
 render();});
render();
})();
</script>'''

def rounded(values, digits=6):
    return [round(float(v), digits) for v in values]


def coefficient_list(window, order, deriv):
    coeffs = savgol_coeffs(window, order, deriv=deriv, delta=dt, use="dot")
    return [float(f"{c:.8g}") for c in coeffs]


explorer_data = {
    "frameRate": FRAME_RATE_HZ,
    "wmin": int(WINDOWS.min()),
    "wmax": int(WINDOWS.max()),
    "edge": EDGE_FRAMES,
    "pathFrames": [int(PATH_ZOOM_S[0] * FRAME_RATE_HZ), int(PATH_ZOOM_S[1] * FRAME_RATE_HZ)],
    "defaults": {
        "window": EXAMPLE_WINDOW,
        "order": ORDER,
        "noise": NOISE_SD_UM,
        "tooSmooth": TOO_SMOOTH_WINDOW,
    },
    "t": rounded(cell["t"], 4),
    "x": rounded(cell["x"]),
    "y": rounded(cell["y"]),
    "vx": rounded(cell["vx"]),
    "vy": rounded(cell["vy"]),
    "speed": rounded(cell["speed"]),
    "ex": rounded(unit_noise[0]),
    "ey": rounded(unit_noise[1]),
    "coeffs": {},
    "curves": {},
}
for order in ORDERS:
    order_curve = error_curve(order, 1.0)
    # With a noise level of 1 µm, column 2 is the norm of the derivative coefficients.
    explorer_data["curves"][order] = [[int(w), round(float(b), 6), round(float(nz), 8)] for w, b, nz, _ in order_curve]
    explorer_data["coeffs"][order] = {
        int(w): [coefficient_list(int(w), order, 0), coefficient_list(int(w), order, 1)]
        for w in WINDOWS[WINDOWS > order]
    }

explorer_html = (
    EXPLORER_TEMPLATE.replace("__UID__", "sg-explorer")   # fixed id: the file is identical after every run
    .replace("__DATA__", json.dumps(explorer_data, separators=(",", ":")))
)
EXPLORER_FILE = ROOT / "notebooks" / "smoothing_explorer.html"
with open(EXPLORER_FILE, "w", encoding="utf-8") as f:
    f.write(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Smoothing window explorer</title></head>\n"
        '<body style="margin:16px;background:#ffffff">\n' + explorer_html + "\n</body></html>\n"
    )
print(f"wrote {EXPLORER_FILE.relative_to(ROOT)} ({EXPLORER_FILE.stat().st_size // 1000} kB); open it in a web browser")
