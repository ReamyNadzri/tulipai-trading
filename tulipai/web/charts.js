/* TulipAI charts: tiny dependency-free SVG line chart + histogram.
   Colors come from CSS custom properties so light/dark themes swap in one place. */
(function (global) {
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmt = (v, d = 2) => Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const niceTicks = (lo, hi, count = 5) => {
    if (!isFinite(lo) || !isFinite(hi)) return [0];
    if (lo === hi) { lo -= 1; hi += 1; }
    const span = hi - lo, step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  const dateLabel = (ms, spanDays) => {
    const d = new Date(ms);
    return spanDays > 2
      ? d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
      : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  };

  function tooltip(host) {
    let t = host.querySelector(".viz-tip");
    if (!t) {
      t = document.createElement("div");
      t.className = "viz-tip";
      host.appendChild(t);
    }
    return t;
  }

  function legend(host, series) {
    if (series.length < 2) return;
    const lg = document.createElement("div");
    lg.className = "viz-legend";
    series.forEach((s) => {
      const item = document.createElement("span");
      const key = document.createElement("i");
      key.style.background = css(s.color);
      item.appendChild(key);
      item.appendChild(document.createTextNode(s.name));
      lg.appendChild(item);
    });
    host.appendChild(lg);
  }

  /* series: [{name, color: "--series-1", points: [[ms, value], ...]}] */
  function lineChart(host, series, opts = {}) {
    host.innerHTML = "";
    host.classList.add("viz-host");
    series = series.filter((s) => s.points && s.points.length);
    if (!series.length) {
      const p = document.createElement("p");
      p.className = "viz-empty";
      p.textContent = opts.empty || "No data yet.";
      host.appendChild(p);
      return;
    }
    legend(host, series);
    const W = Math.max(host.clientWidth, 280), H = opts.height || 240;
    const m = { l: 64, r: 16, t: 10, b: 26 };
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "line chart" }, host);
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = series.flatMap((s) => s.points.map((p) => p[1]));
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const pad = (Math.max(...ys) - Math.min(...ys)) * 0.06 || 1;
    const ticks = niceTicks(Math.min(...ys) - pad, Math.max(...ys) + pad);
    const y0 = Math.min(ticks[0], Math.min(...ys)), y1 = Math.max(ticks[ticks.length - 1], Math.max(...ys));
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);

    ticks.forEach((t) => {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "viz-grid" }, svg);
      const lab = el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "viz-axis" }, svg);
      lab.textContent = fmt(t, Math.abs(t) >= 100 ? 0 : 2);
    });
    if (opts.baseline !== undefined && opts.baseline >= y0 && opts.baseline <= y1) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(opts.baseline), y2: Y(opts.baseline), class: "viz-base" }, svg);
    }
    const spanDays = (x1 - x0) / 864e5;
    const nx = W < 520 ? 2 : 4;
    for (let i = 0; i <= nx; i++) {
      const v = x0 + ((x1 - x0) * i) / nx;
      const lab = el("text", { x: X(v), y: H - 6, "text-anchor": i === 0 ? "start" : i === nx ? "end" : "middle", class: "viz-axis" }, svg);
      lab.textContent = dateLabel(v, spanDays);
    }
    series.forEach((s) => {
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      const last = s.points[s.points.length - 1];
      el("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, fill: css(s.color), stroke: css("--surface-1"), "stroke-width": 2 }, svg);
    });

    const cross = el("line", { y1: m.t, y2: H - m.b, class: "viz-cross", visibility: "hidden" }, svg);
    const dots = series.map((s) => el("circle", { r: 4, fill: css(s.color), stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg));
    const tip = tooltip(host);
    const nearest = (pts, x) => {
      let lo = 0, hi = pts.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid][0] < x) lo = mid; else hi = mid; }
      return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? pts[lo] : pts[hi];
    };
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const xv = x0 + ((ev.clientX - r.left - m.l) / (W - m.l - m.r)) * (x1 - x0);
      const ref = nearest(series[0].points, xv);
      cross.setAttribute("x1", X(ref[0])); cross.setAttribute("x2", X(ref[0])); cross.setAttribute("visibility", "visible");
      tip.replaceChildren();
      const head = document.createElement("div");
      head.className = "viz-tip-head";
      head.textContent = new Date(ref[0]).toLocaleString();
      tip.appendChild(head);
      series.forEach((s, i) => {
        const p = nearest(s.points, ref[0]);
        dots[i].setAttribute("cx", X(p[0])); dots[i].setAttribute("cy", Y(p[1])); dots[i].setAttribute("visibility", "visible");
        const row = document.createElement("div");
        row.className = "viz-tip-row";
        const key = document.createElement("i"); key.style.background = css(s.color);
        const val = document.createElement("strong"); val.textContent = fmt(p[1]);
        const nm = document.createElement("span"); nm.textContent = s.name;
        row.append(key, val, nm);
        tip.appendChild(row);
      });
      tip.style.display = "block";
      const left = Math.min(X(ref[0]) + 12, W - tip.offsetWidth - 4);
      tip.style.left = `${Math.max(4, left)}px`;
      tip.style.top = `${m.t + 4 + (series.length < 2 ? 0 : 24)}px`;
    });
    hit.addEventListener("pointerleave", () => {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      tip.style.display = "none";
    });
  }

  /* values: numbers; marker: {value, label} drawn as a vertical rule */
  function histogram(host, values, marker, opts = {}) {
    host.innerHTML = "";
    host.classList.add("viz-host");
    if (!values || !values.length) return;
    const W = Math.max(host.clientWidth, 280), H = opts.height || 200;
    const m = { l: 40, r: 16, t: 18, b: 26 };
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "histogram" }, host);
    const all = marker ? values.concat([marker.value]) : values;
    const lo = Math.min(...all), hi = Math.max(...all);
    const nb = Math.min(24, Math.max(8, Math.round(Math.sqrt(values.length))));
    const w = (hi - lo) / nb || 1;
    const bins = new Array(nb).fill(0);
    values.forEach((v) => bins[Math.min(nb - 1, Math.floor((v - lo) / w))]++);
    const maxC = Math.max(...bins);
    const X = (v) => m.l + ((v - lo) / (hi - lo || 1)) * (W - m.l - m.r);
    const Y = (c) => H - m.b - (c / maxC) * (H - m.t - m.b);
    niceTicks(0, maxC, 3).forEach((t) => {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "viz-grid" }, svg);
      const lab = el("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end", class: "viz-axis" }, svg);
      lab.textContent = t;
    });
    const tip = tooltip(host);
    const slot = (W - m.l - m.r) / nb;
    const bw = Math.min(24, slot - 2);
    bins.forEach((c, i) => {
      if (!c) return;
      const x = m.l + i * slot + (slot - bw) / 2, y = Y(c), h = H - m.b - y, r = Math.min(4, h);
      const d = `M${x},${H - m.b}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${H - m.b}Z`;
      const bar = el("path", { d, fill: css(opts.color || "--series-2"), class: "viz-bar", tabindex: 0 }, svg);
      const show = () => {
        tip.replaceChildren();
        const v = document.createElement("strong"); v.textContent = `${c} runs`;
        const s = document.createElement("span"); s.textContent = ` ${fmt(lo + i * w, 0)} to ${fmt(lo + (i + 1) * w, 0)}`;
        tip.append(v, s);
        tip.style.display = "block";
        tip.style.left = `${Math.min(x, W - 180)}px`;
        tip.style.top = `${Math.max(0, y - 34)}px`;
      };
      bar.addEventListener("pointermove", show);
      bar.addEventListener("focus", show);
      bar.addEventListener("pointerleave", () => (tip.style.display = "none"));
      bar.addEventListener("blur", () => (tip.style.display = "none"));
    });
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      const lab = el("text", { x: X(v), y: H - 6, "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle", class: "viz-axis" }, svg);
      lab.textContent = fmt(v, 0);
    }
    if (marker) {
      el("line", { x1: X(marker.value), x2: X(marker.value), y1: m.t - 4, y2: H - m.b, stroke: css("--series-1"), "stroke-width": 2 }, svg);
      const lab = el("text", { x: X(marker.value), y: m.t - 6, "text-anchor": X(marker.value) > W - 90 ? "end" : "middle", class: "viz-mark-label" }, svg);
      lab.textContent = marker.label;
    }
  }

  global.TulipCharts = { lineChart, histogram, fmt };
})(window);
