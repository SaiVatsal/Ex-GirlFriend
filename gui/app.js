/**
 * PARHI — SYNAPTIC NEURAL INTERFACE
 * 8K Biological Brainbow Connectome Simulation & Interactive Neural Core
 * 
 * Features:
 * - Anatomically authentic Pyramidal Neuron arbors with apical tufts, basal dendrites & micro-spines
 * - Multi-wavelength Brainbow fluorescence (Magenta, Cyan, Amber, Emerald, Violet, Silver-Blue)
 * - Real-time biological action potential electrical spike propagation
 * - Neuroplastic Learning Growth animation (sprouting dendritic bridges on knowledge acquisition)
 * - Procedural Web Audio synthesizer (zero external sound files required)
 * - WebSocket telemetry & streaming conversation stream
 */

// ============================================================================
// 1. PROCEDURAL WEB AUDIO SYNTHESIZER (ZERO-DEPENDENCY SOUND FX)
// ============================================================================

class SoundFX {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.masterGain = null;
    this.initAudioContext();
  }

  initAudioContext() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.setValueAtTime(0.35, this.ctx.currentTime);
        this.masterGain.connect(this.ctx.destination);
      }
    } catch (e) {
      console.warn("Web Audio API not supported or blocked:", e);
    }
  }

  ensureReady() {
    if (!this.ctx) this.initAudioContext();
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
  }

  toggleSound() {
    this.enabled = !this.enabled;
    return this.enabled;
  }

  // Ethereal wake chord: C5 + E5 + G5 harmonic shimmer
  playWakeChime() {
    if (!this.enabled) return;
    this.ensureReady();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
    
    notes.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, now + idx * 0.04);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(0.18 / (idx + 1), now + idx * 0.04 + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 1.2 + idx * 0.1);

      osc.connect(gain);
      gain.connect(this.masterGain);

      osc.start(now + idx * 0.04);
      osc.stop(now + 1.4);
    });
  }

  // Whoosh / synaptic impulse for sending message
  playSendWhoosh() {
    if (!this.enabled) return;
    this.ensureReady();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    
    // Sub-pulse oscillator
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(320, now);
    osc.frequency.exponentialRampToValueAtTime(80, now + 0.28);

    gain.gain.setValueAtTime(0.22, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.3);

    osc.connect(gain);
    gain.connect(this.masterGain);
    osc.start(now);
    osc.stop(now + 0.32);

    // High shimmer noise sweep
    const bufferSize = this.ctx.sampleRate * 0.2;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    const noise = this.ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.setValueAtTime(1400, now);
    filter.frequency.exponentialRampToValueAtTime(4200, now + 0.2);
    filter.Q.setValueAtTime(4.0, now);

    const noiseGain = this.ctx.createGain();
    noiseGain.gain.setValueAtTime(0.12, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.22);

    noise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(this.masterGain);

    noise.start(now);
    noise.stop(now + 0.25);
  }

  // Response shimmer chime: Ascending synaptic cascade
  playResponseShimmer() {
    if (!this.enabled) return;
    this.ensureReady();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const freqs = [659.25, 783.99, 987.77, 1318.51]; // E5, G5, B5, E6
    
    freqs.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, now + idx * 0.06);

      gain.gain.setValueAtTime(0.001, now + idx * 0.06);
      gain.gain.linearRampToValueAtTime(0.16, now + idx * 0.06 + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.06 + 0.85);

      osc.connect(gain);
      gain.connect(this.masterGain);

      osc.start(now + idx * 0.06);
      osc.stop(now + idx * 0.06 + 0.9);
    });
  }

  // Confirmation beep for hardware / system actions
  playActionBeep() {
    if (!this.enabled) return;
    this.ensureReady();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc1.type = "sine";
    osc2.type = "triangle";

    osc1.frequency.setValueAtTime(880, now); // A5
    osc1.frequency.setValueAtTime(1760, now + 0.07); // A6

    osc2.frequency.setValueAtTime(1320, now);
    osc2.frequency.setValueAtTime(2640, now + 0.07);

    gain.gain.setValueAtTime(0.18, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(this.masterGain);

    osc1.start(now);
    osc2.start(now);
    osc1.stop(now + 0.2);
    osc2.stop(now + 0.2);
  }

  // Tactile micro click for chip taps & buttons
  playClick() {
    if (!this.enabled) return;
    this.ensureReady();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(1800, now);
    osc.frequency.exponentialRampToValueAtTime(400, now + 0.03);

    gain.gain.setValueAtTime(0.14, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.035);

    osc.connect(gain);
    gain.connect(this.masterGain);

    osc.start(now);
    osc.stop(now + 0.04);
  }

  // Neuroplastic burst sound when learning a new concept
  playLearningBurst() {
    if (!this.enabled) return;
    this.ensureReady();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    // Harmonic bell with frequency modulation
    const carrier = this.ctx.createOscillator();
    const modulator = this.ctx.createOscillator();
    const modGain = this.ctx.createGain();
    const mainGain = this.ctx.createGain();

    carrier.type = "sine";
    carrier.frequency.setValueAtTime(440, now);
    carrier.frequency.exponentialRampToValueAtTime(880, now + 0.5);

    modulator.type = "sine";
    modulator.frequency.setValueAtTime(12, now);
    modulator.frequency.exponentialRampToValueAtTime(4, now + 0.8);

    modGain.gain.setValueAtTime(60, now);
    modGain.gain.exponentialRampToValueAtTime(1, now + 0.8);

    modulator.connect(carrier.frequency);
    carrier.connect(mainGain);
    mainGain.connect(this.masterGain);

    mainGain.gain.setValueAtTime(0.01, now);
    mainGain.gain.linearRampToValueAtTime(0.25, now + 0.08);
    mainGain.gain.exponentialRampToValueAtTime(0.0001, now + 1.6);

    modulator.start(now);
    carrier.start(now);
    modulator.stop(now + 1.8);
    carrier.stop(now + 1.8);
  }
}

// ============================================================================
// 2. 8K BIOLOGICAL BRAINBOW NEURAL CONNECTOME SIMULATOR
// ============================================================================

/**
 * Anatomical Pyramidal Neuron with:
 * - Triangular/tear-drop soma & glowing nucleus
 * - Apical dendrite shaft & tufts
 * - Basal dendritic tree with micro-spines (postsynaptic density)
 * - Axon collaterals & synaptic boutons
 */
class PyramidalNeuron {
  constructor(id, x, y, colorData, scale = 1.0) {
    this.id = id;
    this.x = x;
    this.y = y;
    this.baseX = x;
    this.baseY = y;
    this.color = colorData.primary;
    this.secondaryColor = colorData.secondary;
    this.glowColor = colorData.glow;
    this.rgb = colorData.rgb;
    this.scale = scale;

    // Biological properties
    this.somaRadius = 14 * scale;
    this.membranePotential = -70; // Resting mV
    this.activation = 0.15;
    this.firing = false;
    this.pulsePhase = Math.random() * Math.PI * 2;

    // Branches & Spine structures
    this.apicalBranches = [];
    this.basalBranches = [];
    this.axons = [];
    this.spines = []; // Microscopic dendritic spines
    this.connectedNeurons = [];

    this.generateArborization();
  }

  generateArborization() {
    // 1. Apical Dendrite: main vertical trunk ascending upwards
    const apicalTrunkLength = 110 * this.scale;
    const trunkAngle = -Math.PI / 2 + (Math.random() - 0.5) * 0.25; // upwards with slight tilt
    
    const trunkEnd = {
      x: this.x + Math.cos(trunkAngle) * apicalTrunkLength,
      y: this.y + Math.sin(trunkAngle) * apicalTrunkLength
    };

    this.apicalBranches.push({
      p1: { x: this.x, y: this.y - this.somaRadius * 0.7 },
      cp1: { x: this.x + (Math.random() - 0.5) * 20, y: this.y - apicalTrunkLength * 0.45 },
      p2: trunkEnd,
      width: 4.5 * this.scale,
      level: 0
    });

    // Sub-branches of apical tuft (level 1 & 2)
    this.branchOut(trunkEnd, trunkAngle - 0.4, 65 * this.scale, 2.5 * this.scale, 1, this.apicalBranches);
    this.branchOut(trunkEnd, trunkAngle + 0.4, 65 * this.scale, 2.5 * this.scale, 1, this.apicalBranches);
    this.branchOut(trunkEnd, trunkAngle, 75 * this.scale, 2.8 * this.scale, 1, this.apicalBranches);

    // 2. Basal Dendrites: radiating lateral/downwards from base
    const numBasal = 5;
    for (let i = 0; i < numBasal; i++) {
      const angle = (Math.PI * 0.2) + (i / (numBasal - 1)) * (Math.PI * 0.6); // spreading downwards
      const length = (60 + Math.random() * 45) * this.scale;
      const startX = this.x + Math.cos(angle) * (this.somaRadius * 0.85);
      const startY = this.y + Math.sin(angle) * (this.somaRadius * 0.85);
      const endPt = {
        x: startX + Math.cos(angle) * length,
        y: startY + Math.sin(angle) * length
      };

      this.basalBranches.push({
        p1: { x: startX, y: startY },
        cp1: { x: (startX + endPt.x) / 2 + (Math.random() - 0.5) * 25, y: (startY + endPt.y) / 2 + (Math.random() - 0.5) * 25 },
        p2: endPt,
        width: 3.2 * this.scale,
        level: 0
      });

      // Branch basal dendrite further
      this.branchOut(endPt, angle - 0.35, 42 * this.scale, 1.8 * this.scale, 1, this.basalBranches);
      this.branchOut(endPt, angle + 0.35, 42 * this.scale, 1.8 * this.scale, 1, this.basalBranches);
    }

    // 3. Generate micro-spines (postsynaptic density nubs along dendrites, as in reference image 1)
    const allDendrites = [...this.apicalBranches, ...this.basalBranches];
    allDendrites.forEach(b => {
      const numSpines = Math.floor(b.width * 7);
      for (let s = 0; s < numSpines; s++) {
        const t = Math.random();
        // Point along bezier curve
        const bx = (1 - t) * (1 - t) * b.p1.x + 2 * (1 - t) * t * b.cp1.x + t * t * b.p2.x;
        const by = (1 - t) * (1 - t) * b.p1.y + 2 * (1 - t) * t * b.cp1.y + t * t * b.p2.y;
        
        // Perpendicular offset for spine stalk & head
        const spineAngle = Math.random() * Math.PI * 2;
        const spineLen = 3 + Math.random() * 6 * this.scale;
        this.spines.push({
          x: bx + Math.cos(spineAngle) * spineLen,
          y: by + Math.sin(spineAngle) * spineLen,
          baseX: bx,
          baseY: by,
          size: 1.0 + Math.random() * 1.8 * this.scale,
          alpha: 0.35 + Math.random() * 0.45
        });
      }
    });
  }

  branchOut(startPt, baseAngle, length, width, level, targetArr) {
    if (level > 2 || length < 18) return;
    const wiggle = (Math.random() - 0.5) * 0.3;
    const angle = baseAngle + wiggle;
    const endPt = {
      x: startPt.x + Math.cos(angle) * length,
      y: startPt.y + Math.sin(angle) * length
    };
    const cp = {
      x: (startPt.x + endPt.x) / 2 + (Math.random() - 0.5) * 20,
      y: (startPt.y + endPt.y) / 2 + (Math.random() - 0.5) * 20
    };

    targetArr.push({
      p1: startPt,
      cp1: cp,
      p2: endPt,
      width: width,
      level: level
    });

    // Further bifurcation
    this.branchOut(endPt, angle - 0.38, length * 0.72, width * 0.65, level + 1, targetArr);
    this.branchOut(endPt, angle + 0.38, length * 0.72, width * 0.65, level + 1, targetArr);
  }

  update(time, sensoryInput) {
    this.pulsePhase += 0.035;
    
    // Membrane potential breathing
    const restingOscillation = Math.sin(this.pulsePhase) * 0.08;
    this.activation = Math.max(0.12, Math.min(1.0, this.activation * 0.965 + restingOscillation + sensoryInput * 0.3));

    // Slight organic soma floating motion
    this.x = this.baseX + Math.sin(time * 0.8 + this.id) * 3.5;
    this.y = this.baseY + Math.cos(time * 0.9 + this.id * 2) * 3.5;
  }

  fireActionPotential() {
    this.activation = 1.0;
    this.firing = true;
  }
}

/**
 * Action Potential Spike: Moving electrical charge packet traversing an axon/dendrite
 */
class ActionPotential {
  constructor(path, color, speed = 0.015, parentNeuron = null) {
    this.path = path; // Array of points [{x, y}] or Bezier segment {p1, cp1, p2}
    this.color = color;
    this.t = 0;
    this.speed = speed * (0.8 + Math.random() * 0.4);
    this.alive = true;
    this.trail = [];
    this.parentNeuron = parentNeuron;
    this.intensity = 1.0;
  }

  update() {
    this.t += this.speed;
    if (this.t >= 1.0) {
      this.alive = false;
      return;
    }

    // Calculate current position along curve
    let curX, curY;
    if (this.path.cp1) {
      const t = this.t;
      curX = (1 - t) * (1 - t) * this.path.p1.x + 2 * (1 - t) * t * this.path.cp1.x + t * t * this.path.p2.x;
      curY = (1 - t) * (1 - t) * this.path.p1.y + 2 * (1 - t) * t * this.path.cp1.y + t * t * this.path.p2.y;
    } else if (Array.isArray(this.path)) {
      const idx = Math.floor(this.t * (this.path.length - 1));
      const nextIdx = Math.min(idx + 1, this.path.length - 1);
      const subT = (this.t * (this.path.length - 1)) - idx;
      curX = this.path[idx].x + (this.path[nextIdx].x - this.path[idx].x) * subT;
      curY = this.path[idx].y + (this.path[nextIdx].y - this.path[idx].y) * subT;
    }

    this.trail.unshift({ x: curX, y: curY, alpha: 1.0 });
    if (this.trail.length > 10) this.trail.pop();
  }

  draw(ctx) {
    if (!this.trail.length) return;

    ctx.save();
    ctx.globalCompositeOperation = "lighter";

    // Draw spark trail
    for (let i = 0; i < this.trail.length; i++) {
      const p = this.trail[i];
      const alpha = (1 - i / this.trail.length) * this.intensity;
      const size = Math.max(1, (6 - i * 0.5));

      ctx.fillStyle = `rgba(${this.color.rgb}, ${alpha * 0.9})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw intense spark head
    const head = this.trail[0];
    const grad = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, 14);
    grad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
    grad.addColorStop(0.3, `rgba(${this.color.rgb}, 0.9)`);
    grad.addColorStop(1, "rgba(0, 0, 0, 0)");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(head.x, head.y, 14, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }
}

/**
 * Neuroplastic Learning Connection (sprouts when new knowledge/thinking happens)
 */
class NeuroplasticBridge {
  constructor(n1, n2, color) {
    this.n1 = n1;
    this.n2 = n2;
    this.color = color;
    this.progress = 0; // 0 to 1 growth
    this.lifetime = 1.0;
    this.alive = true;
    this.synapticDensity = 8;
    this.seedWiggle = Math.random() * 60 - 30;
  }

  update() {
    if (this.progress < 1.0) {
      this.progress += 0.035; // fast sprout growth
    } else {
      this.lifetime -= 0.006;
      if (this.lifetime <= 0) this.alive = false;
    }
  }

  draw(ctx) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";

    const midX = (this.n1.x + this.n2.x) / 2 + this.seedWiggle;
    const midY = (this.n1.y + this.n2.y) / 2 - 40;

    // Current tip location based on growth progress
    const t = Math.min(1.0, this.progress);
    const tipX = (1 - t) * (1 - t) * this.n1.x + 2 * (1 - t) * t * midX + t * t * this.n2.x;
    const tipY = (1 - t) * (1 - t) * this.n1.y + 2 * (1 - t) * t * midY + t * t * this.n2.y;

    // Glowing new branch
    ctx.strokeStyle = `rgba(251, 191, 36, ${this.lifetime * 0.85})`; // Gold learning beam
    ctx.lineWidth = 2.5 * this.progress;
    ctx.shadowColor = "#fbbf24";
    ctx.shadowBlur = 15;

    ctx.beginPath();
    ctx.moveTo(this.n1.x, this.n1.y);
    ctx.quadraticCurveTo(midX, midY, tipX, tipY);
    ctx.stroke();

    // Secondary violet synaptogenesis halo
    ctx.strokeStyle = `rgba(168, 85, 247, ${this.lifetime * 0.5})`;
    ctx.lineWidth = 5 * this.progress;
    ctx.beginPath();
    ctx.moveTo(this.n1.x, this.n1.y);
    ctx.quadraticCurveTo(midX, midY, tipX, tipY);
    ctx.stroke();

    // Flashing synaptic junction buttons at contact points
    if (this.progress >= 0.95) {
      const pulse = (Math.sin(Date.now() * 0.01) + 1) * 0.5;
      ctx.fillStyle = `rgba(255, 255, 255, ${this.lifetime})`;
      ctx.beginPath();
      ctx.arc(this.n2.x, this.n2.y, 4 + pulse * 4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }
}

/**
 * Main 8K Brainbow Connectome Canvas Controller
 */
class BrainbowCanvas {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d");
    this.neurons = [];
    this.actionPotentials = [];
    this.neuroplasticBridges = [];
    this.interConnectAxons = [];
    this.extracellularVesicles = [];
    this.dpr = window.devicePixelRatio || 2;
    this.state = "idle"; // "idle" | "listening" | "thinking" | "learning"
    this.sensoryVolume = 0; // modulated by mic audio
    this.animationFrameId = null;

    // Biological Brainbow Color Spectrum Palette (matches user's reference images)
    this.brainbowPalette = [
      { primary: "#ec4899", secondary: "#be185d", glow: "rgba(236, 72, 153, 0.4)", rgb: "236, 72, 153" }, // Fluorescent Magenta
      { primary: "#00f0ff", secondary: "#0891b2", glow: "rgba(0, 240, 255, 0.4)", rgb: "0, 240, 255" },     // Electric Cyan
      { primary: "#fbbf24", secondary: "#d97706", glow: "rgba(251, 191, 36, 0.4)", rgb: "251, 191, 36" },   // Vivid Amber Gold
      { primary: "#22c55e", secondary: "#15803d", glow: "rgba(34, 197, 94, 0.4)", rgb: "34, 197, 94" },     // Fluorescent Lime
      { primary: "#a855f7", secondary: "#7e22ce", glow: "rgba(168, 85, 247, 0.4)", rgb: "168, 85, 247" },   // Bright Violet
      { primary: "#93c5fd", secondary: "#3b82f6", glow: "rgba(147, 197, 253, 0.4)", rgb: "147, 197, 253" }, // Confocal Silver-Blue
      { primary: "#fb923c", secondary: "#ea580c", glow: "rgba(251, 146, 60, 0.4)", rgb: "251, 146, 60" }    // Synaptic Coral
    ];

    this.initCanvasSize();
    this.initNeurons();
    this.initExtracellularVesicles();
    this.bindEvents();
    this.startRenderLoop();
  }

  initCanvasSize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width || 800;
    this.height = rect.height || 600;

    // High-DPI 8K Subpixel Buffer Scaling
    this.canvas.width = Math.floor(this.width * this.dpr);
    this.canvas.height = Math.floor(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.scale(this.dpr, this.dpr);
  }

  initNeurons() {
    this.neurons = [];
    this.interConnectAxons = [];

    // Form an anatomically distributed cortical network with 7 primary pyramidal neurons
    const count = 7;
    const w = this.width;
    const h = this.height;

    // Specific coordinate anchors to keep visual balance around central mic
    const positions = [
      { x: w * 0.22, y: h * 0.38, scale: 1.05 }, // Left upper pyramidal
      { x: w * 0.78, y: h * 0.36, scale: 1.02 }, // Right upper pyramidal
      { x: w * 0.50, y: h * 0.22, scale: 1.15 }, // Central apical queen neuron
      { x: w * 0.16, y: h * 0.72, scale: 0.90 }, // Left basal pyramidal
      { x: w * 0.84, y: h * 0.70, scale: 0.92 }, // Right basal pyramidal
      { x: w * 0.36, y: h * 0.78, scale: 0.95 }, // Mid-left lower neuron
      { x: w * 0.64, y: h * 0.80, scale: 0.96 }  // Mid-right lower neuron
    ];

    for (let i = 0; i < count; i++) {
      const pos = positions[i];
      const color = this.brainbowPalette[i % this.brainbowPalette.length];
      const neuron = new PyramidalNeuron(i, pos.x, pos.y, color, pos.scale);
      this.neurons.push(neuron);
    }

    // Generate inter-neuron axon collaterals (cross-cellular mesh)
    for (let i = 0; i < this.neurons.length; i++) {
      for (let j = i + 1; j < this.neurons.length; j++) {
        const n1 = this.neurons[i];
        const n2 = this.neurons[j];
        const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);

        if (dist < w * 0.65) {
          const cpX = (n1.x + n2.x) / 2 + (Math.random() - 0.5) * 80;
          const cpY = (n1.y + n2.y) / 2 + (Math.random() - 0.5) * 80;
          this.interConnectAxons.push({
            p1: { x: n1.x, y: n1.y },
            cp1: { x: cpX, y: cpY },
            p2: { x: n2.x, y: n2.y },
            n1: n1,
            n2: n2,
            length: dist,
            color: n1.color,
            rgb: n1.rgb
          });
        }
      }
    }
  }

  initExtracellularVesicles() {
    this.extracellularVesicles = [];
    const count = 45;
    for (let i = 0; i < count; i++) {
      this.extracellularVesicles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        radius: 0.8 + Math.random() * 2.2,
        speedX: (Math.random() - 0.5) * 0.25,
        speedY: (Math.random() - 0.5) * 0.25,
        color: this.brainbowPalette[i % this.brainbowPalette.length].rgb,
        alpha: 0.15 + Math.random() * 0.35
      });
    }
  }

  bindEvents() {
    window.addEventListener("resize", () => {
      this.initCanvasSize();
      this.initNeurons();
      this.initExtracellularVesicles();
    });
  }

  setState(newState) {
    this.state = newState;
    const badge = document.getElementById("cortex-state-text");
    if (badge) {
      if (newState === "listening") badge.innerText = "Acoustic Synaptic Transduction";
      else if (newState === "thinking") badge.innerText = "Cortical Gamma Oscillation";
      else if (newState === "learning") badge.innerText = "Long-Term Potentiation (Learning)";
      else badge.innerText = "Resting Membrane Potential";
    }
  }

  triggerSpontaneousFiring() {
    // Spontaneous action potential along random axonal or dendritic route
    if (!this.interConnectAxons.length) return;
    const axon = this.interConnectAxons[Math.floor(Math.random() * this.interConnectAxons.length)];
    const ap = new ActionPotential(axon, axon.n1, 0.016, axon.n1);
    this.actionPotentials.push(ap);
    axon.n1.fireActionPotential();
  }

  triggerLearningEvent(concept = "New Concept") {
    this.setState("learning");

    // All neurons fire in synchronized burst
    this.neurons.forEach(n => n.fireActionPotential());

    // Sprout new neuroplastic bridges across multiple pairs
    for (let k = 0; k < 4; k++) {
      const idx1 = Math.floor(Math.random() * this.neurons.length);
      let idx2 = (idx1 + 1 + Math.floor(Math.random() * (this.neurons.length - 1))) % this.neurons.length;
      const n1 = this.neurons[idx1];
      const n2 = this.neurons[idx2];
      this.neuroplasticBridges.push(new NeuroplasticBridge(n1, n2, n1.color));
    }

    // Launch flurry of golden/magenta action potentials
    for (let m = 0; m < 8; m++) {
      const axon = this.interConnectAxons[Math.floor(Math.random() * this.interConnectAxons.length)];
      if (axon) {
        this.actionPotentials.push(new ActionPotential(axon, { rgb: "251, 191, 36" }, 0.024));
      }
    }

    setTimeout(() => {
      if (this.state === "learning") this.setState("idle");
    }, 2800);
  }

  triggerThinkingCascade() {
    this.setState("thinking");
    // High-frequency oscillating action potentials
    for (let i = 0; i < 3; i++) {
      this.triggerSpontaneousFiring();
    }
  }

  startRenderLoop() {
    let lastTime = 0;

    const render = (time) => {
      const sec = time * 0.001;
      
      // Clean frame with slight persistence for motion phosphor trails
      this.ctx.fillStyle = "rgba(6, 8, 16, 0.32)";
      this.ctx.fillRect(0, 0, this.width, this.height);

      // 1. Draw Extracellular Matrix & Floating Vesicles
      this.drawVesicles();

      // 2. Update and Draw Neurons (Soma, Apical & Basal Arbors, Micro-Spines)
      this.neurons.forEach(neuron => {
        neuron.update(sec, this.sensoryVolume);
        this.drawNeuronArbor(neuron);
      });

      // 3. Draw Interconnect Axon Collateral Web
      this.drawAxonMesh();

      // 4. Update & Draw Action Potential Sparks
      this.updateAndDrawActionPotentials();

      // 5. Update & Draw Neuroplastic Growth Bridges (Learning)
      this.updateAndDrawBridges();

      // 6. Draw Pyramidal Somas & Glowing Nuclei (Top Layer for 3D depth)
      this.neurons.forEach(neuron => {
        this.drawPyramidalSoma(neuron);
      });

      // Periodic spontaneous biological activity
      if (Math.random() < 0.04 && this.state === "idle") {
        this.triggerSpontaneousFiring();
      } else if (this.state === "thinking" && Math.random() < 0.22) {
        this.triggerSpontaneousFiring();
      } else if (this.state === "listening" && Math.random() < 0.16) {
        this.triggerSpontaneousFiring();
      }

      this.animationFrameId = requestAnimationFrame(render);
    };

    this.animationFrameId = requestAnimationFrame(render);
  }

  drawVesicles() {
    this.ctx.save();
    this.extracellularVesicles.forEach(v => {
      v.x += v.speedX;
      v.y += v.speedY;

      if (v.x < 0) v.x = this.width;
      if (v.x > this.width) v.x = 0;
      if (v.y < 0) v.y = this.height;
      if (v.y > this.height) v.y = 0;

      this.ctx.fillStyle = `rgba(${v.color}, ${v.alpha})`;
      this.ctx.beginPath();
      this.ctx.arc(v.x, v.y, v.radius, 0, Math.PI * 2);
      this.ctx.fill();
    });
    this.ctx.restore();
  }

  drawNeuronArbor(neuron) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";

    // A. Draw Microscopic Dendritic Spines (Electron microscopy boutons)
    ctx.fillStyle = `rgba(${neuron.rgb}, ${0.35 + neuron.activation * 0.4})`;
    neuron.spines.forEach(spine => {
      ctx.beginPath();
      ctx.arc(spine.x, spine.y, spine.size * (0.8 + neuron.activation * 0.4), 0, Math.PI * 2);
      ctx.fill();
    });

    // B. Draw Apical Dendrite & Branches with multi-pass glow
    const drawBranches = (branchList) => {
      branchList.forEach(b => {
        // Outer fluorescent aura
        ctx.strokeStyle = `rgba(${neuron.rgb}, ${0.18 + neuron.activation * 0.35})`;
        ctx.lineWidth = b.width * 2.2;
        ctx.beginPath();
        ctx.moveTo(b.p1.x, b.p1.y);
        ctx.quadraticCurveTo(b.cp1.x, b.cp1.y, b.p2.x, b.p2.y);
        ctx.stroke();

        // Inner crisp dendritic core
        ctx.strokeStyle = neuron.color;
        ctx.lineWidth = Math.max(1.0, b.width * (0.9 + neuron.activation * 0.2));
        ctx.beginPath();
        ctx.moveTo(b.p1.x, b.p1.y);
        ctx.quadraticCurveTo(b.cp1.x, b.cp1.y, b.p2.x, b.p2.y);
        ctx.stroke();
      });
    };

    drawBranches(neuron.apicalBranches);
    drawBranches(neuron.basalBranches);

    ctx.restore();
  }

  drawAxonMesh() {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";

    this.interConnectAxons.forEach(axon => {
      const act = (axon.n1.activation + axon.n2.activation) * 0.5;
      ctx.strokeStyle = `rgba(${axon.rgb}, ${0.08 + act * 0.25})`;
      ctx.lineWidth = 1.1;

      ctx.beginPath();
      ctx.moveTo(axon.p1.x, axon.p1.y);
      ctx.quadraticCurveTo(axon.cp1.x, axon.cp1.y, axon.p2.x, axon.p2.y);
      ctx.stroke();
    });

    ctx.restore();
  }

  updateAndDrawActionPotentials() {
    for (let i = this.actionPotentials.length - 1; i >= 0; i--) {
      const ap = this.actionPotentials[i];
      ap.update();
      if (!ap.alive) {
        this.actionPotentials.splice(i, 1);
      } else {
        ap.draw(this.ctx);
      }
    }
  }

  updateAndDrawBridges() {
    for (let i = this.neuroplasticBridges.length - 1; i >= 0; i--) {
      const bridge = this.neuroplasticBridges[i];
      bridge.update();
      if (!bridge.alive) {
        this.neuroplasticBridges.splice(i, 1);
      } else {
        bridge.draw(this.ctx);
      }
    }
  }

  drawPyramidalSoma(neuron) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";

    const r = neuron.somaRadius * (1.0 + neuron.activation * 0.22);

    // 1. Broad outer diffuse luminescence
    const diffuseGrad = ctx.createRadialGradient(neuron.x, neuron.y, r * 0.2, neuron.x, neuron.y, r * 3.5);
    diffuseGrad.addColorStop(0, `rgba(${neuron.rgb}, ${0.45 + neuron.activation * 0.45})`);
    diffuseGrad.addColorStop(0.5, `rgba(${neuron.rgb}, 0.15)`);
    diffuseGrad.addColorStop(1, "rgba(0, 0, 0, 0)");

    ctx.fillStyle = diffuseGrad;
    ctx.beginPath();
    ctx.arc(neuron.x, neuron.y, r * 3.5, 0, Math.PI * 2);
    ctx.fill();

    // 2. Anatomical Pyramidal Cell Body (Teardrop shape pointing towards apical trunk)
    ctx.fillStyle = neuron.color;
    ctx.beginPath();
    ctx.moveTo(neuron.x, neuron.y - r * 1.5); // Apical pole
    ctx.bezierCurveTo(neuron.x + r * 1.2, neuron.y - r * 0.4, neuron.x + r * 1.1, neuron.y + r * 1.2, neuron.x, neuron.y + r * 1.1); // Right flank
    ctx.bezierCurveTo(neuron.x - r * 1.1, neuron.y + r * 1.2, neuron.x - r * 1.2, neuron.y - r * 0.4, neuron.x, neuron.y - r * 1.5); // Left flank
    ctx.closePath();
    ctx.fill();

    // 3. Glowing Cellular Nucleus (Organelle core)
    const nucleusGrad = ctx.createRadialGradient(neuron.x, neuron.y, 0, neuron.x, neuron.y, r * 0.65);
    nucleusGrad.addColorStop(0, "#ffffff");
    nucleusGrad.addColorStop(0.5, `rgba(${neuron.rgb}, 0.95)`);
    nucleusGrad.addColorStop(1, `rgba(${neuron.rgb}, 0.0)`);

    ctx.fillStyle = nucleusGrad;
    ctx.beginPath();
    ctx.arc(neuron.x, neuron.y, r * 0.65, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }
}

// ============================================================================
// 3. SPEECH RECOGNITION & ACOUSTIC MIC CONTROLLER
// ============================================================================

class SpeechManager {
  constructor(soundFX, onTranscript, onFinalResult) {
    this.soundFX = soundFX;
    this.onTranscript = onTranscript;
    this.onFinalResult = onFinalResult;
    this.recognition = null;
    this.isListening = false;
    this.initRecognition();
  }

  initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = "en-US";

      this.recognition.onstart = () => {
        this.isListening = true;
        this.soundFX.playWakeChime();
        this.updateMicUI(true);
      };

      this.recognition.onresult = (event) => {
        let interim = "";
        let final = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            final += event.results[i][0].transcript;
          } else {
            interim += event.results[i][0].transcript;
          }
        }
        if (this.onTranscript) this.onTranscript(interim || final);
        if (final && this.onFinalResult) {
          this.onFinalResult(final.trim());
        }
      };

      this.recognition.onerror = (event) => {
        console.warn("Speech recognition error:", event.error);
        this.isListening = false;
        this.updateMicUI(false);
      };

      this.recognition.onend = () => {
        this.isListening = false;
        this.updateMicUI(false);
      };
    } else {
      console.warn("Web Speech API not supported in this browser environment.");
    }
  }

  toggle() {
    if (!this.recognition) {
      alert("Microphone recognition requires Chrome or Edge browser support.");
      return;
    }
    if (this.isListening) {
      this.recognition.stop();
      this.isListening = false;
      this.updateMicUI(false);
    } else {
      try {
        this.recognition.start();
      } catch (e) {
        console.warn("Recognition start error:", e);
      }
    }
  }

  updateMicUI(active) {
    const btn = document.getElementById("mic-button");
    const rings = document.querySelectorAll(".synaptic-wave-ring");
    const liveBox = document.getElementById("live-transcript");
    const caption = document.getElementById("mic-caption");

    if (active) {
      btn?.classList.add("active");
      rings.forEach(r => r.classList.add("active-wave"));
      if (liveBox) liveBox.style.display = "block";
      if (caption) caption.innerHTML = "Listening to your voice...";
    } else {
      btn?.classList.remove("active");
      rings.forEach(r => r.classList.remove("active-wave"));
      if (caption) caption.innerHTML = 'Click to Speak or say <span class="highlight-word">"Parhi"</span>';
    }
  }
}

// ============================================================================
// 4. WEBSOCKET CLIENT & APP CONTROLLER
// ============================================================================

class ParhiApp {
  constructor() {
    this.soundFX = new SoundFX();
    this.canvas = null;
    this.speech = null;
    this.ws = null;
    this.messageCount = 0;
    this.currentStreamingBubble = null;

    this.initDOM();
    this.initCanvas();
    this.initSpeech();
    this.initWebSocket();
    this.initQuickChips();
    this.initHardwareActions();
  }

  initDOM() {
    this.chatHistory = document.getElementById("chat-history");
    this.userInput = document.getElementById("user-input");
    this.sendBtn = document.getElementById("send-btn");
    this.typingIndicator = document.getElementById("typing-indicator");
    this.messageCounter = document.getElementById("message-counter");
    this.clearChatBtn = document.getElementById("clear-chat-btn");
    this.soundToggleBtn = document.getElementById("sound-toggle");
    this.soundIcon = document.getElementById("sound-icon");
    this.soundLabel = document.getElementById("sound-label");

    // Input handlers
    this.sendBtn?.addEventListener("click", () => this.handleSendMessage());
    this.userInput?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.handleSendMessage();
      }
    });

    // Clear chat
    this.clearChatBtn?.addEventListener("click", () => {
      this.soundFX.playClick();
      if (this.chatHistory) {
        this.chatHistory.innerHTML = `
          <div class="message-bubble parhi-bubble welcome-bubble">
            <div class="bubble-avatar">P</div>
            <div class="bubble-content">
              <div class="bubble-author">
                <span class="author-name">Parhi</span>
                <span class="author-tag">Neural Companion</span>
              </div>
              <p class="bubble-text">Chat cleared. Synaptic memory ready for new thoughts!</p>
            </div>
          </div>
        `;
        this.messageCount = 0;
        this.updateCounter();
      }
    });

    // Sound toggle
    this.soundToggleBtn?.addEventListener("click", () => {
      const isEnabled = this.soundFX.toggleSound();
      if (this.soundIcon) this.soundIcon.innerText = isEnabled ? "🔊" : "🔇";
      if (this.soundLabel) this.soundLabel.innerText = isEnabled ? "Audio: ON" : "Audio: MUTE";
      this.soundFX.playClick();
    });
  }

  initCanvas() {
    const canvasEl = document.getElementById("neural-canvas");
    if (canvasEl) {
      this.canvas = new BrainbowCanvas(canvasEl);
    }
  }

  initSpeech() {
    this.speech = new SpeechManager(
      this.soundFX,
      (transcript) => {
        const transContent = document.getElementById("transcript-content");
        if (transContent) transContent.innerText = transcript;
        if (this.canvas) this.canvas.sensoryVolume = 0.8;
      },
      (finalText) => {
        if (this.userInput) this.userInput.value = finalText;
        this.handleSendMessage();
      }
    );

    const micBtn = document.getElementById("mic-button");
    micBtn?.addEventListener("click", () => {
      this.speech.toggle();
    });
  }

  initWebSocket() {
    const host = window.location.host || "127.0.0.1:8000";
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${host}/ws`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("[✓] Connected to Parhi Neural Core Gateway");
      const statusPill = document.getElementById("system-status-pill");
      const statusLabel = document.getElementById("status-label");
      if (statusLabel) statusLabel.innerText = "Cortex Active";
      if (statusPill) statusPill.classList.remove("offline");
      this.ws.send(JSON.stringify({ type: "get_telemetry" }));
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleServerMessage(data);
      } catch (err) {
        console.error("WS Parse Error:", err, event.data);
      }
    };

    this.ws.onerror = (err) => {
      console.warn("WebSocket status:", err);
    };

    this.ws.onclose = () => {
      const statusLabel = document.getElementById("status-label");
      if (statusLabel) statusLabel.innerText = "Reconnecting...";
      setTimeout(() => this.initWebSocket(), 3000);
    };
  }

  initQuickChips() {
    const chips = document.querySelectorAll(".chip");
    chips.forEach(chip => {
      chip.addEventListener("click", () => {
        this.soundFX.playClick();
        const text = chip.getAttribute("data-text");
        if (text && this.userInput) {
          this.userInput.value = text;
          this.handleSendMessage();
        }
      });
    });
  }

  initHardwareActions() {
    const actions = [
      { id: "btn-camera", cmd: "camera", target: "camera" },
      { id: "btn-screen", cmd: "screen", target: "screen" },
      { id: "btn-vol-up", cmd: "volume_up", target: "up" },
      { id: "btn-vol-down", cmd: "volume_down", target: "down" },
      { id: "btn-music", cmd: "music", target: "spotify" },
      { id: "btn-lock", cmd: "lock", target: "workstation" }
    ];

    actions.forEach(a => {
      const btn = document.getElementById(a.id);
      btn?.addEventListener("click", () => {
        this.soundFX.playActionBeep();
        if (this.canvas) this.canvas.triggerLearningEvent();
        this.sendWebSocketMessage({
          type: "system_command",
          command: a.cmd,
          target: a.target
        });
      });
    });
  }

  handleSendMessage() {
    if (!this.userInput) return;
    const text = this.userInput.value.trim();
    if (!text) return;

    this.soundFX.playSendWhoosh();
    this.appendMessage("user", text);
    this.userInput.value = "";

    // Trigger biological canvas reaction
    if (this.canvas) {
      this.canvas.triggerThinkingCascade();
    }

    // Show typing state
    if (this.typingIndicator) this.typingIndicator.style.display = "flex";

    // Transmit to Python backend
    this.sendWebSocketMessage({
      type: "chat_message",
      text: text
    });
  }

  sendWebSocketMessage(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  handleServerMessage(msg) {
    switch (msg.type) {
      case "thinking_start":
        if (this.canvas) this.canvas.triggerThinkingCascade();
        if (this.typingIndicator) this.typingIndicator.style.display = "flex";
        break;

      case "thinking_step":
        // Chain of thought chunk
        break;

      case "chat_response":
        if (this.typingIndicator) this.typingIndicator.style.display = "none";
        this.soundFX.playResponseShimmer();
        this.appendMessage("parhi", msg.text, msg.thinking);

        if (this.canvas) {
          // If response contained learning or tool use, trigger neuroplastic burst
          if (msg.learned || (msg.thinking && msg.thinking.length > 50)) {
            this.canvas.triggerLearningEvent();
            this.soundFX.playLearningBurst();
          } else {
            this.canvas.setState("idle");
          }
        }
        break;

      case "telemetry":
        this.updateTelemetry(msg);
        break;

      case "command_feedback":
        this.appendMessage("parhi", `⚡ ${msg.message}`);
        this.soundFX.playActionBeep();
        break;
    }
  }

  appendMessage(author, text, thinking = "") {
    if (!this.chatHistory) return;

    this.messageCount++;
    this.updateCounter();

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${author === "user" ? "user-bubble" : "parhi-bubble"}`;

    const avatarLetter = author === "user" ? "U" : "P";
    const authorName = author === "user" ? "You" : "Parhi";
    const authorTag = author === "user" ? "Human Partner" : "Neural Companion";

    let thinkingHtml = "";
    if (thinking && thinking.trim()) {
      thinkingHtml = `
        <details class="thinking-accordion">
          <summary>🧠 Synaptic Chain of Thought</summary>
          <pre class="thinking-content">${this.escapeHTML(thinking.trim())}</pre>
        </details>
      `;
    }

    bubble.innerHTML = `
      <div class="bubble-avatar">${avatarLetter}</div>
      <div class="bubble-content">
        <div class="bubble-author">
          <span class="author-name">${authorName}</span>
          <span class="author-tag">${authorTag}</span>
        </div>
        ${thinkingHtml}
        <p class="bubble-text">${this.formatLinksAndMarkdown(text)}</p>
      </div>
    `;

    this.chatHistory.appendChild(bubble);
    this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
  }

  updateCounter() {
    if (this.messageCounter) {
      this.messageCounter.innerText = `${this.messageCount} messages`;
    }
  }

  updateTelemetry(data) {
    // Battery
    if (data.battery !== undefined) {
      const bVal = document.getElementById("battery-val");
      const bBar = document.getElementById("battery-bar");
      const bSub = document.getElementById("battery-status-sub");
      if (bVal) bVal.innerText = `${data.battery}%`;
      if (bBar) bBar.style.width = `${data.battery}%`;
      if (bSub) bSub.innerText = data.battery_charging ? "⚡ AC Power Connected" : "Battery Discharging";
    }

    // WiFi
    if (data.wifi_ssid !== undefined) {
      const ssid = document.getElementById("wifi-ssid");
      const wVal = document.getElementById("wifi-val");
      const ipVal = document.getElementById("ip-val");
      if (ssid) ssid.innerText = data.wifi_ssid || "Ethernet/Connected";
      if (wVal) wVal.innerText = `${data.wifi_strength || 100}%`;
      if (ipVal) ipVal.innerText = data.local_ip || "127.0.0.1";
    }

    // Bond & Mood
    if (data.bond !== undefined) {
      const bVal = document.getElementById("bond-val");
      const bBar = document.getElementById("bond-bar");
      const bSub = document.getElementById("bond-sub");
      if (bVal) bVal.innerText = `${data.bond}/100`;
      if (bBar) bBar.style.width = `${data.bond}%`;
      if (bSub) bSub.innerText = data.bond_label || "Warm & Attentive";
    }

    if (data.mood) {
      const mText = document.getElementById("mood-text");
      const mIcon = document.getElementById("mood-icon");
      if (mText) mText.innerText = data.mood;
      if (mIcon) {
        const moodMap = { "happy": "✨", "loving": "💖", "caring": "🌸", "thoughtful": "💫", "excited": "⚡", "protective": "🛡️" };
        mIcon.innerText = moodMap[data.mood.toLowerCase()] || "💫";
      }
    }
  }

  escapeHTML(str) {
    return str.replace(/[&<>'"]/g, tag => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
    }[tag] || tag));
  }

  formatLinksAndMarkdown(text) {
    let esc = this.escapeHTML(text);
    // Simple code block format
    esc = esc.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    // Line breaks
    esc = esc.replace(/\n/g, '<br>');
    return esc;
  }
}

// Instantiate application when DOM is ready
window.addEventListener("DOMContentLoaded", () => {
  window.parhiApp = new ParhiApp();
});
