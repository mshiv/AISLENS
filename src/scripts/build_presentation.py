#!/usr/bin/env python3
"""Generate the AISLENS IceT presentation as a self-contained HTML file."""

import base64, os

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "..",
    "reports", "figures", "presentations", "20260722-IceT")
OUT = os.path.join(os.path.dirname(__file__), "..", "..",
    "reports", "figures", "presentations", "20260722-IceT", "presentation.html")

def b64(filename):
    path = os.path.join(FIGDIR, filename)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Pre-load all images
imgs = {}
for name in ["fig_lowpass_filter.png", "fig_spread_amplification.png",
             "cross_scenario_ranking.png", "skew_kurt_relative.png",
             "relative_uncertainty_ratio.png", "misi_diagnostics.png",
             "uncertainty_partition.png", "transfer_function.png",
             "grounded_floating_partition.png", "fig_mass_budget_spread.png"]:
    try:
        imgs[name] = b64(name)
    except FileNotFoundError:
        pass

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Ice Sheet Hasn't Noticed Yet — AISLENS Ensemble Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0b1120;
  --bg-card: #131b2e;
  --bg-card2: #1a2540;
  --text: #e8ecf4;
  --text-dim: #7b8ba8;
  --accent: #2A9D8F;
  --accent2: #E9C46A;
  --accent3: #E76F51;
  --red: #D55E00;
  --blue: #0072B2;
  --purple: #CC79A7;
  --font: 'DM Sans', system-ui, sans-serif;
  --mono: 'JetBrains Mono', monospace;
  --title: clamp(1.8rem, 4.5vw, 3.2rem);
  --h2: clamp(1.2rem, 3vw, 2rem);
  --h3: clamp(0.95rem, 2vw, 1.35rem);
  --body: clamp(0.75rem, 1.4vw, 1.05rem);
  --small: clamp(0.65rem, 1vw, 0.85rem);
  --pad: clamp(1.5rem, 4vw, 4rem);
  --gap: clamp(0.6rem, 1.5vw, 1.5rem);
}
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
html { scroll-snap-type: y mandatory; scroll-behavior: smooth; height: 100%; }
body { font-family: var(--font); background: var(--bg); color: var(--text);
       overflow-x: hidden; height: 100%; line-height: 1.5; }

.slide {
  width: 100vw; height: 100vh; height: 100dvh;
  overflow: hidden; scroll-snap-align: start;
  display: flex; flex-direction: column; justify-content: center;
  position: relative; padding: var(--pad);
}
.slide-content {
  flex: 1; display: flex; flex-direction: column; justify-content: center;
  max-height: 100%; overflow: hidden;
}

/* Title slide */
.slide.title-slide {
  background: linear-gradient(135deg, #0b1120 0%, #132238 50%, #0b1120 100%);
  text-align: center; align-items: center;
}
.slide.title-slide h1 { font-size: var(--title); font-weight: 700; margin-bottom: 0.5em; }
.slide.title-slide .subtitle { font-size: var(--h3); color: var(--text-dim); margin-bottom: 0.3em; }
.slide.title-slide .meta { font-size: var(--small); color: var(--text-dim); margin-top: 1em; }

/* Section headers */
.slide h2 { font-size: var(--h2); font-weight: 700; margin-bottom: var(--gap);
             color: var(--accent); }
.slide h3 { font-size: var(--h3); font-weight: 500; margin-bottom: 0.5em; color: var(--accent2); }

/* Key takeaway banner */
.takeaway {
  background: var(--bg-card2); border-left: 4px solid var(--accent);
  padding: clamp(0.5rem, 1.2vw, 1rem) clamp(0.8rem, 2vw, 1.5rem);
  border-radius: 0 8px 8px 0; margin-top: var(--gap);
  font-size: var(--body); font-weight: 500;
}
.takeaway .label { color: var(--accent); font-size: var(--small); text-transform: uppercase;
                   letter-spacing: 0.1em; margin-bottom: 0.3em; }

/* Figure containers */
.fig-row { display: flex; gap: var(--gap); align-items: center; justify-content: center;
           flex: 1; min-height: 0; }
.fig-row.two { flex-wrap: wrap; }
.fig-row img { max-height: min(55vh, 450px); max-width: 48%; object-fit: contain; border-radius: 6px; }
.fig-row.single img { max-width: 85%; max-height: min(60vh, 500px); }
.fig-row.wide img { max-width: 92%; max-height: min(52vh, 420px); }

/* Bullet lists */
.bullets { list-style: none; padding: 0; }
.bullets li { padding: 0.35em 0; padding-left: 1.2em; position: relative;
              font-size: var(--body); }
.bullets li::before { content: '▸'; position: absolute; left: 0; color: var(--accent); }

/* Tables */
.data-table { border-collapse: collapse; font-size: var(--small); margin: var(--gap) 0; }
.data-table th { text-align: left; padding: 0.4em 0.8em; border-bottom: 2px solid var(--accent);
                 color: var(--accent); font-weight: 500; }
.data-table td { padding: 0.35em 0.8em; border-bottom: 1px solid #1e2d4a; }
.data-table tr:last-child td { border-bottom: none; }

/* Emphasis */
em { color: var(--accent2); font-style: normal; font-weight: 500; }
strong { color: var(--accent); }

/* Two-column layout */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap); align-items: start; }
@media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }

/* Scale diagram */
.scale-bar { background: var(--bg-card2); border-radius: 8px; padding: var(--gap);
             margin: var(--gap) 0; }
.scale-track { height: 6px; background: #1e2d4a; border-radius: 3px; position: relative;
               margin: 1em 0; }
.scale-marker { position: absolute; top: -8px; width: 3px; height: 22px;
                border-radius: 2px; transform: translateX(-50%); }
.scale-label { position: absolute; top: 20px; font-size: var(--small); color: var(--text-dim);
               transform: translateX(-50%); white-space: nowrap; }

/* Progress bar */
.progress { position: fixed; top: 0; left: 0; height: 3px; background: var(--accent);
            z-index: 100; transition: width 0.3s ease; }

/* Nav dots */
.nav { position: fixed; right: 1.2rem; top: 50%; transform: translateY(-50%);
       display: flex; flex-direction: column; gap: 6px; z-index: 100; }
.nav-dot { width: 8px; height: 8px; border-radius: 50%; background: #2a3a5c;
           cursor: pointer; transition: all 0.3s ease; border: none; }
.nav-dot.active { background: var(--accent); transform: scale(1.4); }

/* Animations */
.reveal { opacity: 0; transform: translateY(20px);
          transition: opacity 0.5s cubic-bezier(0.16,1,0.3,1),
                      transform 0.5s cubic-bezier(0.16,1,0.3,1); }
.slide.visible .reveal { opacity: 1; transform: translateY(0); }
.slide.visible .reveal:nth-child(1) { transition-delay: 0.05s; }
.slide.visible .reveal:nth-child(2) { transition-delay: 0.12s; }
.slide.visible .reveal:nth-child(3) { transition-delay: 0.2s; }
.slide.visible .reveal:nth-child(4) { transition-delay: 0.28s; }
.slide.visible .reveal:nth-child(5) { transition-delay: 0.36s; }

@media (max-height: 600px) {
  :root { --pad: clamp(0.8rem, 2.5vw, 1.5rem); --gap: clamp(0.3rem, 1vw, 0.6rem); }
  .fig-row img { max-height: 40vh; }
  .nav { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .reveal { transition: opacity 0.2s ease; transform: none; }
  html { scroll-behavior: auto; }
}
</style>
</head>
<body>

<div class="progress" id="progress"></div>
<nav class="nav" id="nav"></nav>

<!-- ============ SLIDE 1: TITLE ============ -->
<section class="slide title-slide" data-name="title">
  <h1 class="reveal">The Ice Sheet Hasn't Noticed Yet</h1>
  <p class="subtitle reveal">Ocean variability doesn't amplify Antarctic sea level rise — within 300 years</p>
  <p class="meta reveal">AISLENS Ensemble Analysis &middot; IceT Group Meeting &middot; July 2026</p>
  <p class="meta reveal" style="font-size:clamp(0.6rem,0.9vw,0.75rem); color:#4a5a7a; margin-top:0.5em;">Shivaprakash Muruganandham</p>
</section>

<!-- ============ SLIDE 2: MOTIVATION ============ -->
<section class="slide" data-name="motivation">
  <div class="slide-content">
    <h2 class="reveal">Why Does Ocean Variability Matter?</h2>
    <div class="two-col reveal">
      <div>
        <h3>The Concern</h3>
        <ul class="bullets">
          <li>CMIP6 forcing includes <em>interannual ocean variability</em></li>
          <li>Robel et al. (2019): variability can <em>amplify</em> retreat via MISI feedback</li>
          <li>If true, constant-forcing projections <em>underestimate</em> sea level rise</li>
        </ul>
      </div>
      <div>
        <h3>What We Tested</h3>
        <ul class="bullets">
          <li>4 ensembles: CTRL, SSP585, SSP126, SSP585 × 10 variability</li>
          <li>10–15 members each, 300+ years</li>
          <li>Does the ice sheet actually respond to variability?</li>
        </ul>
      </div>
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      Within the AISLENS timeframe (~300 yr), ocean variability does NOT significantly amplify Antarctic sea level rise. Initial conditions dominate.
    </div>
  </div>
</section>

<!-- ============ SLIDE 3: ENSEMBLE DESIGN ============ -->
<section class="slide" data-name="ensemble">
  <div class="slide-content">
    <h2 class="reveal">AISLENS Ensemble Design</h2>
    <table class="data-table reveal">
      <tr><th>Ensemble</th><th>Members</th><th>Years</th><th>Forcing</th></tr>
      <tr><td><strong>CTRL</strong></td><td>10</td><td>0 – 404</td><td>Constant present-day</td></tr>
      <tr><td><strong>SSP585</strong></td><td>10</td><td>0 – 300</td><td>CMIP6 SSP5-8.5</td></tr>
      <tr><td><strong>SSP126</strong></td><td>10</td><td>0 – 300</td><td>CMIP6 SSP1-2.6</td></tr>
      <tr><td><strong>SSP585 × 10</strong></td><td>15</td><td>0 – 300</td><td>SSP585 with 10× interannual variability</td></tr>
    </table>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      The 10× ensemble isolates the effect of variability amplitude while holding the mean forcing identical to SSP585.
    </div>
  </div>
</section>

<!-- ============ SLIDE 4: LOW-PASS FILTER ============ -->
<section class="slide" data-name="lowpass">
  <div class="slide-content">
    <h2 class="reveal">The Ice Sheet Is a Low-Pass Filter</h2>
    <div class="fig-row two reveal">
      <img src="data:image/png;base64,IMGS[fig_lowpass_filter.png]" alt="VAF Power Spectral Density">
      <img src="data:image/png;base64,IMGS[transfer_function.png]" alt="Autocorrelation Memory">
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      Interannual variability is damped. Only decadal+ signals propagate through to volume above flotation. CTRL memory timescale: 22 ± 6 years.
    </div>
  </div>
</section>

<!-- ============ SLIDE 5: SPREAD DOESN'T AMPLIFY ============ -->
<section class="slide" data-name="spread">
  <div class="slide-content">
    <h2 class="reveal">Spread Doesn't Amplify</h2>
    <div class="fig-row two reveal">
      <img src="data:image/png;base64,IMGS[fig_spread_amplification.png]" alt="Spread amplification ratio">
      <img src="data:image/png;base64,IMGS[relative_uncertainty_ratio.png]" alt="Relative uncertainty ratio">
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      σ<sub>SSP585</sub> / σ<sub>CTRL</sub> stays near 1. Forcing does not amplify ensemble spread within 300 years.
    </div>
  </div>
</section>

<!-- ============ SLIDE 6: INITIAL CONDITIONS DOMINATE ============ -->
<section class="slide" data-name="ranking">
  <div class="slide-content">
    <h2 class="reveal">Initial Conditions Dominate</h2>
    <div class="fig-row single reveal">
      <img src="data:image/png;base64,IMGS[cross_scenario_ranking.png]" alt="Cross-scenario member ranking">
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      Member ranking is preserved across scenarios: SSP585 vs SSP126 correlation <em>r = +0.745</em>. The ice sheet's initial state determines its future more than the forcing scenario.
    </div>
  </div>
</section>

<!-- ============ SLIDE 7: DISTRIBUTION STAYS GAUSSIAN ============ -->
<section class="slide" data-name="skewkurt">
  <div class="slide-content">
    <h2 class="reveal">Distribution Stays Gaussian</h2>
    <div class="fig-row single reveal">
      <img src="data:image/png;base64,IMGS[skew_kurt_relative.png]" alt="Skewness, kurtosis, relative uncertainty">
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      Skewness ≈ 0 throughout. No non-Gaussian signatures — the ice sheet is in the <em>linear regime</em>. Robel's amplification mechanism has not activated.
    </div>
  </div>
</section>

<!-- ============ SLIDE 8: NO MISI YET ============ -->
<section class="slide" data-name="misi">
  <div class="slide-content">
    <h2 class="reveal">No MISI Yet</h2>
    <div class="fig-row single reveal">
      <img src="data:image/png;base64,IMGS[misi_diagnostics.png]" alt="MISI diagnostics: grounded area and GL flux">
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      Grounded area and grounding-line flux show no accelerating retreat. MISI onset requires ~400–500 years (Robel et al. 2019). We are not there yet.
    </div>
  </div>
</section>

<!-- ============ SLIDE 9: UNCERTAINTY DECOMPOSITION ============ -->
<section class="slide" data-name="uncertainty">
  <div class="slide-content">
    <h2 class="reveal">Where Does the Uncertainty Come From?</h2>
    <div class="fig-row single reveal">
      <img src="data:image/png;base64,IMGS[uncertainty_partition.png]" alt="Hawkins-Sutton uncertainty partition">
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      Model uncertainty dominates over internal variability. Scenario uncertainty has not yet emerged — the forced signal hasn't separated from the noise.
    </div>
  </div>
</section>

<!-- ============ SLIDE 10: WHEN WILL IT MATTER? ============ -->
<section class="slide" data-name="horizon">
  <div class="slide-content">
    <h2 class="reveal">When <em>Will</em> Variability Matter?</h2>
    <div class="scale-bar reveal">
      <div style="display:flex;justify-content:space-between;font-size:var(--small);color:var(--text-dim);margin-bottom:0.5em;">
        <span>0 yr</span><span>200 yr</span><span>400 yr</span><span>600 yr</span>
      </div>
      <div class="scale-track">
        <div class="scale-marker" style="left:50%;background:var(--accent);"></div>
        <div class="scale-label" style="left:50%;">AISLENS covers<br><strong style="color:var(--accent);">here</strong></div>
        <div class="scale-marker" style="left:67%;background:var(--red);"></div>
        <div class="scale-label" style="left:67%;">MISI onset<br><span style="color:var(--red);">~400 yr</span></div>
        <div class="scale-marker" style="left:83%;background:var(--accent3);"></div>
        <div class="scale-label" style="left:83%;">Robel dominance<br><span style="color:var(--accent3);">~500 yr</span></div>
      </div>
    </div>
    <ul class="bullets reveal" style="margin-top:var(--gap);">
      <li><strong>Below ~400 yr:</strong> Ice sheet response is linear. Variability is filtered out.</li>
      <li><strong>Above ~500 yr:</strong> MISI feedback activates. Variability amplifies retreat.</li>
      <li><strong>AISLENS sits in the "not yet" window.</strong> The ice sheet hasn't noticed the variability.</li>
    </ul>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      Variability matters on centennial-to-millennial timescales. For policy-relevant projections (~100 yr), constant-forcing runs are sufficient.
    </div>
  </div>
</section>

<!-- ============ SLIDE 11: IMPLICATIONS ============ -->
<section class="slide" data-name="implications">
  <div class="slide-content">
    <h2 class="reveal">Implications & Open Questions</h2>
    <div class="two-col reveal">
      <div>
        <h3>What This Means</h3>
        <ul class="bullets">
          <li>Constant-forcing ensembles are <em>not biased low</em> for 100-yr projections</li>
          <li>Initial condition spread is the dominant uncertainty source</li>
          <li>Member ranking is predictable — first-year state determines the future</li>
        </ul>
      </div>
      <div>
        <h3>Open Questions</h3>
        <ul class="bullets">
          <li>What happens at 500+ yr when MISI activates?</li>
          <li>Does variability matter more in specific basins (Amundsen, FRIS)?</li>
          <li>How does the transfer function change under stronger forcing?</li>
        </ul>
      </div>
    </div>
    <div class="takeaway reveal">
      <div class="label">Key takeaway</div>
      For the next century, focus on constraining initial conditions, not forcing variability.
    </div>
  </div>
</section>

<!-- ============ SLIDE 12: THANK YOU ============ -->
<section class="slide title-slide" data-name="end">
  <h1 class="reveal">Thank You</h1>
  <p class="subtitle reveal">Questions?</p>
  <p class="meta reveal">AISLENS Project &middot; Ice Sheet Model Intercomparison</p>
  <p class="meta reveal" style="font-size:clamp(0.55rem,0.85vw,0.7rem); color:#3a4a6a; margin-top:2em;">
    Figures and analysis: github.com/.../aislens
  </p>
</section>

<script>
/* ===========================================
   SLIDE CONTROLLER
   Keyboard nav, scroll-snap, progress, dots
   =========================================== */
class Presentation {
  constructor() {
    this.slides = document.querySelectorAll('.slide');
    this.progress = document.getElementById('progress');
    this.nav = document.getElementById('nav');
    this.current = 0;
    this.buildNav();
    this.observe();
    this.bindKeys();
    this.updateProgress();
  }

  buildNav() {
    this.slides.forEach((s, i) => {
      const dot = document.createElement('button');
      dot.className = 'nav-dot' + (i === 0 ? ' active' : '');
      dot.setAttribute('aria-label', `Slide ${i+1}`);
      dot.addEventListener('click', () => this.goTo(i));
      this.nav.appendChild(dot);
    });
  }

  observe() {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          const idx = [...this.slides].indexOf(e.target);
          if (idx >= 0) { this.current = idx; this.updateProgress(); this.updateNav(); }
        }
      });
    }, { threshold: 0.5 });
    this.slides.forEach(s => obs.observe(s));
  }

  bindKeys() {
    document.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault(); this.goTo(Math.min(this.current + 1, this.slides.length - 1));
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault(); this.goTo(Math.max(this.current - 1, 0));
      } else if (e.key === 'Home') {
        e.preventDefault(); this.goTo(0);
      } else if (e.key === 'End') {
        e.preventDefault(); this.goTo(this.slides.length - 1);
      }
    });
  }

  goTo(i) {
    this.slides[i].scrollIntoView({ behavior: 'smooth' });
    this.current = i;
    this.updateProgress();
    this.updateNav();
  }

  updateProgress() {
    const pct = ((this.current + 1) / this.slides.length) * 100;
    this.progress.style.width = pct + '%';
  }

  updateNav() {
    this.nav.querySelectorAll('.nav-dot').forEach((d, i) => {
      d.classList.toggle('active', i === this.current);
    });
  }
}

document.addEventListener('DOMContentLoaded', () => new Presentation());
</script>
</body>
</html>"""

# Fix image src attributes — replace placeholder with actual base64 data
for name, b64data in imgs.items():
    placeholder = f'IMGS[{name}]'
    HTML = HTML.replace(placeholder, b64data)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(HTML)

print(f"Written: {OUT}")
print(f"Size: {os.path.getsize(OUT) / 1024:.0f} KB")
