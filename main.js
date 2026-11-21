import * as THREE from 'three';
import {
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter
} from '@mediapipe/tasks-vision';

const video = document.getElementById('webcam');
const bgShaderContainer = document.getElementById('bg-shader-container');
const threeContainer = document.getElementById('three-container');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');

const statusBadge = document.getElementById('status-badge');
const combineBanner = document.getElementById('combine-banner');
const hudMenuEl = document.getElementById('hud-menu');
const shapesToggleBtn = document.getElementById('shapes-toggle-btn');
const sunToggleBtn = document.getElementById('sun-toggle-btn');
const shapesDrawerEl = document.getElementById('shapes-drawer');
const hudButtons = Array.from(document.querySelectorAll('.hud-btn'));

const handsCountEl = document.getElementById('hands-count');
const handsDotEl = document.getElementById('hands-dot');
const shapesCountEl = document.getElementById('shapes-count');
const sunDotEl = document.getElementById('sun-dot');
const sunStatusTextEl = document.getElementById('sun-status-text');
const clearAllBtn = document.getElementById('clear-all-btn');
const trashZoneEl = document.getElementById('trash-zone');
const trashRingEl = trashZoneEl.querySelector('.trash-ring');
const trashLabelEl = document.getElementById('trash-label');

// Standard 21-landmark MediaPipe hand skeleton connections
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],           // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],           // Index finger
  [5, 9], [9, 10], [10, 11], [11, 12],      // Middle finger
  [9, 13], [13, 14], [14, 15], [15, 16],    // Ring finger
  [13, 17], [17, 18], [18, 19], [19, 20],   // Pinky
  [0, 17]                                   // Palm base to pinky base
];

const SPECIAL_LANDMARKS = {
  4: { color: '#ff00ff', glow: '#ff00ff', radius: 8, label: 'Thumb Tip (4)' },
  8: { color: '#ffff00', glow: '#ffff00', radius: 9, label: 'Index Tip (8)' },
  12: { color: '#ff00ff', glow: '#ff00ff', radius: 8, label: 'Middle Tip (12)' }
};

// Colors for interactive 3D hologram shapes (pre-allocated once)
const COLOR_CYAN = new THREE.Color(0x00f3ff);
const COLOR_GOLD = new THREE.Color(0xffd700);
const COLOR_MAGENTA = new THREE.Color(0xff00ff);
const COLOR_HELD_GREEN = new THREE.Color(0x00ff66);
const COLOR_DANGER_RED = new THREE.Color(0xff264e);
const COLOR_WHITE = new THREE.Color(0xffffff);

// Pre-allocated scratch vectors to eliminate per-frame garbage collection
const _scratchVecA = new THREE.Vector3();
const _scratchVecB = new THREE.Vector3();
const _scratchProjected = new THREE.Vector3();
const _scratchMidpoint = new THREE.Vector3();

// Pre-allocated screen landmark pools for up to 2 hands (21 landmarks each)
const _screenPointsPool = [
  Array.from({ length: 21 }, () => ({ x: 0, y: 0 })),
  Array.from({ length: 21 }, () => ({ x: 0, y: 0 }))
];
const _rawHandsPool = [
  { wrist: null, indexTip: null, thumbTip: null, middleTip: null, pinchDistPx: 0 },
  { wrist: null, indexTip: null, thumbTip: null, middleTip: null, pinchDistPx: 0 }
];
const _trackedHandsBuffer = [];

// Reusable sets for per-frame interaction tracking (cleared each frame instead of new Set())
const _hoveredButtonsSet = new Set();
const _activeButtonsSet = new Set();
const _hoveredShapesSet = new Set();
const _dangerShapesSet = new Set();
const _heldShapesSet = new Set();

// Pinch, hover, cooldowns, and two-hand collision threshold
const PINCH_START_PX = 55;
const PINCH_RELEASE_PX = 75;
const HOVER_RADIUS_PX = 105;
const SUN_HOVER_RADIUS_PX = 115;
const TRASH_DROP_RADIUS_PX = 78;
const SPAWN_COOLDOWN_MS = 500;
const TOGGLE_COOLDOWN_MS = 500;
const MERGE_DISTANCE_WORLD = 1.45;
const MAX_PIXEL_RATIO = 1.25;

let handLandmarker = null;
let imageSegmenter = null;
let lastVideoTime = -1;
let lastMediaPipeTimestampMs = -1;
let segmentFrameCounter = 0;
let latestHandResults = null;
let combineBannerTimeout = null;
let lastClearTime = 0;
let lastMenuToggleTime = 0;
let lastSunToggleTime = 0;
let isShapesMenuOpen = false;
let isContextLost = false;
let animFrameId = null;

// ============================================================================
// Optional Spawnable Sun State (Starts OFF by default)
// ============================================================================
let isSunActive = false;
let sunTransition = 0.0;
let sunScale = 1.0;
const sunTargetUv = new THREE.Vector2(0.5, 0.5);
const sunCurrentUv = new THREE.Vector2(0.5, 0.5);
let sunGrabbedBySlotIndex = null;
let isMouseDraggingSun = false;

function setShapesMenuOpen(openState) {
  isShapesMenuOpen = openState;
  hudMenuEl.classList.toggle('is-open', isShapesMenuOpen);
  shapesToggleBtn.setAttribute('aria-expanded', String(isShapesMenuOpen));
  shapesDrawerEl.setAttribute('aria-hidden', String(!isShapesMenuOpen));
  if (!isShapesMenuOpen) {
    for (const btn of hudButtons) {
      btn.classList.remove('is-hovered');
    }
  }
}

function setSunActive(activeState) {
  isSunActive = activeState;
  if (isSunActive) {
    sunTargetUv.set(0.5, 0.5);
    sunCurrentUv.set(0.5, 0.5);
    sunScale = 1.0;
    bgUniforms.uSunScale.value = 1.0;
  } else {
    sunGrabbedBySlotIndex = null;
    isMouseDraggingSun = false;
    if (resizeState.isSun) {
      resizeState.active = false;
      resizeState.isSun = false;
    }
  }

  sunToggleBtn.classList.toggle('is-sun-active', isSunActive);
  sunToggleBtn.setAttribute('aria-pressed', String(isSunActive));
  sunDotEl.classList.toggle('is-sun-on', isSunActive);
  sunStatusTextEl.textContent = isSunActive ? 'Sun: ACTIVE' : 'Sun: OFF';
}

// ============================================================================
// Synthesized Web Audio API Sound Engine (Pre-allocated noise buffer)
// ============================================================================
let audioCtx = null;
let cachedNoiseBuffer = null;

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

window.addEventListener('pointerdown', () => getAudioContext(), { passive: true });
window.addEventListener('keydown', () => getAudioContext(), { passive: true });

function playPinchBlip(pitchMultiplier = 1.0) {
  const ctxAudio = getAudioContext();
  if (!ctxAudio) return;

  const now = ctxAudio.currentTime;
  const osc = ctxAudio.createOscillator();
  const filter = ctxAudio.createBiquadFilter();
  const gain = ctxAudio.createGain();

  osc.type = 'sine';
  osc.frequency.setValueAtTime(720 * pitchMultiplier, now);
  osc.frequency.exponentialRampToValueAtTime(1280 * pitchMultiplier, now + 0.075);

  filter.type = 'bandpass';
  filter.frequency.setValueAtTime(1100 * pitchMultiplier, now);
  filter.Q.setValueAtTime(2.5, now);

  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.11, now + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.095);

  osc.connect(filter);
  filter.connect(gain);
  gain.connect(ctxAudio.destination);

  osc.start(now);
  osc.stop(now + 0.1);
}

function playCombineChime() {
  const ctxAudio = getAudioContext();
  if (!ctxAudio) return;

  const now = ctxAudio.currentTime;
  const notes = [523.25, 659.25, 783.99, 987.77, 1174.66, 1567.98];

  notes.forEach((freq, idx) => {
    const startTime = now + idx * 0.045;
    const duration = 0.55;

    const osc = ctxAudio.createOscillator();
    const subOsc = ctxAudio.createOscillator();
    const gain = ctxAudio.createGain();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, startTime);
    osc.frequency.exponentialRampToValueAtTime(freq * 1.015, startTime + duration);

    subOsc.type = 'sine';
    subOsc.frequency.setValueAtTime(freq * 0.5, startTime);

    gain.gain.setValueAtTime(0.0001, startTime);
    gain.gain.exponentialRampToValueAtTime(0.085, startTime + 0.025);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    osc.connect(gain);
    subOsc.connect(gain);
    gain.connect(ctxAudio.destination);

    osc.start(startTime);
    subOsc.start(startTime);
    osc.stop(startTime + duration + 0.02);
    subOsc.stop(startTime + duration + 0.02);
  });
}

function playClearSweep() {
  const ctxAudio = getAudioContext();
  if (!ctxAudio) return;

  const now = ctxAudio.currentTime;
  const osc = ctxAudio.createOscillator();
  const gain = ctxAudio.createGain();

  osc.type = 'sawtooth';
  osc.frequency.setValueAtTime(640, now);
  osc.frequency.exponentialRampToValueAtTime(140, now + 0.24);

  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.075, now + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.25);

  osc.connect(gain);
  gain.connect(ctxAudio.destination);

  osc.start(now);
  osc.stop(now + 0.26);
}

function playTrashDeleteWhoosh() {
  const ctxAudio = getAudioContext();
  if (!ctxAudio) return;

  const now = ctxAudio.currentTime;
  const duration = 0.24;

  if (!cachedNoiseBuffer) {
    const bufferSize = Math.floor(ctxAudio.sampleRate * duration);
    cachedNoiseBuffer = ctxAudio.createBuffer(1, bufferSize, ctxAudio.sampleRate);
    const output = cachedNoiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }
  }

  const whiteNoise = ctxAudio.createBufferSource();
  whiteNoise.buffer = cachedNoiseBuffer;

  const filter = ctxAudio.createBiquadFilter();
  filter.type = 'bandpass';
  filter.frequency.setValueAtTime(1400, now);
  filter.frequency.exponentialRampToValueAtTime(180, now + duration);
  filter.Q.setValueAtTime(1.8, now);

  const noiseGain = ctxAudio.createGain();
  noiseGain.gain.setValueAtTime(0.001, now);
  noiseGain.gain.exponentialRampToValueAtTime(0.16, now + 0.03);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  whiteNoise.connect(filter);
  filter.connect(noiseGain);
  noiseGain.connect(ctxAudio.destination);

  const osc = ctxAudio.createOscillator();
  const oscGain = ctxAudio.createGain();
  osc.type = 'triangle';
  osc.frequency.setValueAtTime(480, now);
  osc.frequency.exponentialRampToValueAtTime(95, now + duration);

  oscGain.gain.setValueAtTime(0.001, now);
  oscGain.gain.exponentialRampToValueAtTime(0.11, now + 0.02);
  oscGain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  osc.connect(oscGain);
  oscGain.connect(ctxAudio.destination);

  whiteNoise.start(now);
  whiteNoise.stop(now + duration);
  osc.start(now);
  osc.stop(now + duration);
}

// ============================================================================
// LAYER 1 (Background): Pre-Allocated Float32Array & Single DataTexture
// ============================================================================
const MASK_WIDTH = 256;
const MASK_HEIGHT = 256;
const MASK_LENGTH = MASK_WIDTH * MASK_HEIGHT;

// Pre-allocate ONE reusable Float32Array and ONE reusable THREE.DataTexture at startup
const smoothedMaskBuffer = new Float32Array(MASK_LENGTH);

const maskDataTexture = new THREE.DataTexture(
  smoothedMaskBuffer,
  MASK_WIDTH,
  MASK_HEIGHT,
  THREE.RedFormat,
  THREE.FloatType
);
maskDataTexture.minFilter = THREE.LinearFilter;
maskDataTexture.magFilter = THREE.LinearFilter;
maskDataTexture.wrapS = THREE.ClampToEdgeWrapping;
maskDataTexture.wrapT = THREE.ClampToEdgeWrapping;
maskDataTexture.flipY = true;
maskDataTexture.needsUpdate = true;

const bgScene = new THREE.Scene();
const bgCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const bgRenderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: 'high-performance'
});
bgRenderer.setSize(window.innerWidth, window.innerHeight);
bgRenderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
bgShaderContainer.appendChild(bgRenderer.domElement);

const webcamTexture = new THREE.VideoTexture(video);
webcamTexture.minFilter = THREE.LinearFilter;
webcamTexture.magFilter = THREE.LinearFilter;
webcamTexture.format = THREE.RGBAFormat;

const bgVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

// Fragment shader optimized with a 24-step dithered 2D raymarch
const bgFragmentShader = /* glsl */ `
  precision highp float;

  uniform sampler2D uWebcamTex;
  uniform sampler2D uMaskTex;
  uniform vec2 uLightPos;
  uniform vec2 uResolution;
  uniform vec2 uVideoResolution;
  uniform vec2 uMaskResolution;
  uniform float uTime;
  uniform float uSunActive;
  uniform float uSunHeld;
  uniform float uSunDanger;
  uniform float uSunScale;

  varying vec2 vUv;

  float interleavedGradientNoise(vec2 pixelCoord) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pixelCoord, magic.xy)));
  }

  float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);

    float a = hash21(i + vec2(0.0, 0.0));
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }

  float solarFbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 3; i++) {
      v += amp * valueNoise(p);
      p = rot * p * 2.07;
      amp *= 0.5;
    }
    return v;
  }

  vec2 getMirroredCoverUv(vec2 screenUv) {
    float screenAspect = uResolution.x / max(uResolution.y, 1.0);
    float videoAspect = uVideoResolution.x / max(uVideoResolution.y, 1.0);

    vec2 scale = vec2(1.0);
    if (screenAspect > videoAspect) {
      scale.y = videoAspect / screenAspect;
    } else {
      scale.x = screenAspect / videoAspect;
    }

    vec2 coveredUv = (screenUv - 0.5) * scale + 0.5;
    coveredUv.x = 1.0 - coveredUv.x;
    return clamp(coveredUv, 0.0, 1.0);
  }

  // 9-tap Gaussian blur + crisp smoothstep(0.35, 0.65, blurred)
  float sampleGaussianBlurredMask(vec2 screenUv) {
    vec2 uv = getMirroredCoverUv(screenUv);
    vec2 texel = 1.35 / max(uMaskResolution, vec2(256.0));

    float blurred =
      texture2D(uMaskTex, uv + vec2(-texel.x, -texel.y)).r * 0.0625 +
      texture2D(uMaskTex, uv + vec2(     0.0, -texel.y)).r * 0.1250 +
      texture2D(uMaskTex, uv + vec2( texel.x, -texel.y)).r * 0.0625 +
      texture2D(uMaskTex, uv + vec2(-texel.x,      0.0)).r * 0.1250 +
      texture2D(uMaskTex, uv                           ).r * 0.2500 +
      texture2D(uMaskTex, uv + vec2( texel.x,      0.0)).r * 0.1250 +
      texture2D(uMaskTex, uv + vec2(-texel.x,  texel.y)).r * 0.0625 +
      texture2D(uMaskTex, uv + vec2(     0.0,  texel.y)).r * 0.1250 +
      texture2D(uMaskTex, uv + vec2( texel.x,  texel.y)).r * 0.0625;

    return smoothstep(0.35, 0.65, blurred);
  }

  // Fast bilinear sample used inside the 24-step raymarch loop
  float sampleRayMask(vec2 screenUv) {
    vec2 uv = getMirroredCoverUv(screenUv);
    return smoothstep(0.35, 0.65, texture2D(uMaskTex, uv).r);
  }

  float computeSunOcclusionFactor(vec2 aspectVec) {
    if (uSunHeld > 0.5) {
      return 0.0;
    }

    float occ = sampleGaussianBlurredMask(uLightPos) * 2.0;
    float totalWeight = 2.0;

    const float R1 = 0.024;
    const float R2 = 0.052;

    for (int i = 0; i < 6; i++) {
      float angle = float(i) * 1.04719755;
      vec2 dir = vec2(cos(angle), sin(angle)) / aspectVec;

      occ += sampleRayMask(uLightPos + dir * R1) * 1.2;
      occ += sampleRayMask(uLightPos + dir * R2) * 0.8;
      totalWeight += 2.0;
    }

    return clamp(occ / totalWeight, 0.0, 1.0);
  }

  void main() {
    vec2 videoUv = getMirroredCoverUv(vUv);
    vec3 webcamColor = texture2D(uWebcamTex, videoUv).rgb;

    // Default Holo-Lab dimmed webcam background when Sun is OFF
    vec3 defaultHoloBg = mix(webcamColor, vec3(0.015, 0.035, 0.07), 0.62);

    if (uSunActive <= 0.001) {
      gl_FragColor = vec4(defaultHoloBg, 1.0);
      return;
    }

    float aspect = uResolution.x / max(uResolution.y, 1.0);
    vec2 aspectVec = vec2(aspect, 1.0);

    float mask = sampleGaussianBlurredMask(vUv);
    float bgVisibility = mix(1.0 - mask, 1.0, uSunHeld * 0.85);

    float occlusionFactor = computeSunOcclusionFactor(aspectVec);
    float sunlightTransmission = 1.0 - smoothstep(0.05, 0.92, occlusionFactor);

    vec2 toSun = uLightPos - vUv;
    vec2 toSunAspect = toSun * aspectVec;
    float distToSun = length(toSunAspect);
    vec2 sunDir = normalize(toSunAspect + vec2(1e-5));

    // 24-Step Dithered 2D Raymarch for Volumetric God Rays & Shadows
    const int NUM_STEPS = 24;
    float dither = interleavedGradientNoise(gl_FragCoord.xy);

    float godRayAccum = 0.0;
    float shadowTransmittance = 1.0;
    float stepWeightSum = 0.0;

    for (int i = 0; i < NUM_STEPS; i++) {
      float t = (float(i) + dither) / float(NUM_STEPS);
      vec2 samplePos = mix(vUv, uLightPos, t);

      float stepMask = sampleRayMask(samplePos);
      float stepOpen = 1.0 - stepMask;

      float rayDecay = mix(0.6, 1.5, t);
      godRayAccum += stepOpen * rayDecay;
      stepWeightSum += rayDecay;

      shadowTransmittance *= (1.0 - stepMask * 0.16);
    }

    float volumetricRay = clamp(godRayAccum / max(stepWeightSum, 1.0), 0.0, 1.0);
    float crepuscularShafts = mix(volumetricRay, pow(volumetricRay, 1.65), 0.65);
    float surfaceShadow = clamp(mix(shadowTransmittance, volumetricRay, 0.45), 0.0, 1.0);

    // Realistic Animated Sun with Solar Turbulence
    float angle = atan(toSunAspect.y, toSunAspect.x);
    vec2 noiseCoord = toSunAspect * 14.0 + vec2(uTime * 0.35, -uTime * 0.25);
    float surfaceTurbulence = solarFbm(noiseCoord);

    vec2 coronaNoiseCoord = vec2(angle * 3.5, distToSun * 10.0 - uTime * 0.85);
    float coronaFlames = solarFbm(coronaNoiseCoord);

    float sunCoreRadius = 0.038 * uSunScale * mix(1.0, 0.65, uSunDanger);
    float distortedDist = distToSun - (surfaceTurbulence - 0.5) * 0.008 - (coronaFlames - 0.5) * 0.012;

    float whiteHotCore = 1.0 - smoothstep(sunCoreRadius * 0.35, sunCoreRadius * 0.95, distortedDist);
    float goldenLimb = 1.0 - smoothstep(sunCoreRadius * 0.7, sunCoreRadius * 1.45, distortedDist);
    float innerCorona = exp(-max(distortedDist - sunCoreRadius * 0.5, 0.0) * (18.0 / max(uSunScale, 0.4)));
    float outerCorona = exp(-distToSun * (4.2 / max(uSunScale, 0.5))) * (0.75 + 0.45 * coronaFlames);

    vec3 colorWhiteHot   = mix(vec3(1.00, 0.99, 0.94), vec3(1.00, 0.55, 0.65), uSunDanger);
    vec3 colorSolarGold  = mix(vec3(1.00, 0.78, 0.28), vec3(1.00, 0.15, 0.30), uSunDanger);
    vec3 colorFieryAmber = mix(vec3(0.98, 0.42, 0.08), vec3(0.85, 0.05, 0.18), uSunDanger);

    vec3 sunEmission =
      colorWhiteHot   * (whiteHotCore * 2.4) +
      colorSolarGold  * (goldenLimb * 1.35 + innerCorona * 1.15) +
      colorFieryAmber * (outerCorona * 0.95);

    vec3 occludedSunDisc = sunEmission * bgVisibility * (0.18 + 0.82 * sunlightTransmission);

    float radialAtten = 1.0 / (1.0 + distToSun * 1.4 + distToSun * distToSun * 2.0);
    float roomAmbient = mix(0.045, 0.26, sunlightTransmission);
    float directSunlight = radialAtten * surfaceShadow * sunlightTransmission * 1.35;

    vec3 bgLit = webcamColor * (roomAmbient + directSunlight * colorSolarGold * 1.25);
    bgLit += occludedSunDisc;

    float godRayFalloff = exp(-distToSun * 1.55) * (0.45 + 0.55 * radialAtten);
    float eclipseRayBoost = mix(0.22, 1.25, pow(sunlightTransmission, 0.55));
    vec3 volumetricGodRays =
      mix(colorFieryAmber, colorSolarGold, crepuscularShafts) *
      crepuscularShafts *
      godRayFalloff *
      eclipseRayBoost *
      1.15;

    // Foreground Silhouette Shading + Warm Golden Rim Lighting
    vec2 gradEps = vec2(2.75 / max(uResolution.x, 1.0), 2.75 / max(uResolution.y, 1.0));
    float maskR = sampleGaussianBlurredMask(vUv + vec2(gradEps.x, 0.0));
    float maskL = sampleGaussianBlurredMask(vUv - vec2(gradEps.x, 0.0));
    float maskU = sampleGaussianBlurredMask(vUv + vec2(0.0, gradEps.y));
    float maskD = sampleGaussianBlurredMask(vUv - vec2(0.0, gradEps.y));

    vec2 maskGrad = vec2(maskR - maskL, maskU - maskD);
    vec2 outwardNormal = normalize(-maskGrad * aspectVec + vec2(1e-5));
    float edgeMagnitude = clamp(length(maskGrad) * 2.6, 0.0, 1.0);

    float sunAlignment = max(0.0, dot(outwardNormal, sunDir));
    float eclipseBacklightWrap = exp(-distToSun * 6.5) * edgeMagnitude * 0.85;

    float edgeContour = smoothstep(0.05, 0.5, mask) * (1.0 - smoothstep(0.55, 0.98, mask));
    float rimFactor = (edgeMagnitude * 0.7 + edgeContour * 0.9) * pow(sunAlignment, 1.15) + eclipseBacklightWrap;
    rimFactor *= (0.25 + 1.45 * radialAtten) * mix(0.45, 1.15, sunlightTransmission);

    float fgInteriorLight = mix(0.10, 0.56, sunlightTransmission) +
                            radialAtten * surfaceShadow * sunlightTransmission * 0.35;

    vec3 fgLit = webcamColor * fgInteriorLight;
    fgLit += colorSolarGold * rimFactor * 1.45 + colorWhiteHot * pow(max(rimFactor, 0.0), 2.2) * 0.65;

    vec3 solarSceneColor = mix(bgLit, fgLit, mask);
    solarSceneColor += volumetricGodRays * mix(1.0, 0.38, mask);

    solarSceneColor = (solarSceneColor * (2.51 * solarSceneColor + 0.03)) /
                      (solarSceneColor * (2.43 * solarSceneColor + 0.59) + 0.14);
    solarSceneColor = pow(clamp(solarSceneColor, 0.0, 1.0), vec3(0.94));

    vec3 finalOut = mix(defaultHoloBg, solarSceneColor, uSunActive);
    gl_FragColor = vec4(finalOut, 1.0);
  }
`;

const bgUniforms = {
  uWebcamTex: { value: webcamTexture },
  uMaskTex: { value: maskDataTexture },
  uLightPos: { value: sunCurrentUv },
  uResolution: {
    value: new THREE.Vector2(window.innerWidth, window.innerHeight)
  },
  uVideoResolution: { value: new THREE.Vector2(640, 480) },
  uMaskResolution: { value: new THREE.Vector2(MASK_WIDTH, MASK_HEIGHT) },
  uTime: { value: 0.0 },
  uSunActive: { value: 0.0 },
  uSunHeld: { value: 0.0 },
  uSunDanger: { value: 0.0 },
  uSunScale: { value: 1.0 }
};

const bgShaderMaterial = new THREE.ShaderMaterial({
  vertexShader: bgVertexShader,
  fragmentShader: bgFragmentShader,
  uniforms: bgUniforms,
  depthWrite: false,
  depthTest: false
});

const bgQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgShaderMaterial);
bgScene.add(bgQuad);

/**
 * Copies the MediaPipe segmentation confidence mask into the single pre-allocated
 * `smoothedMaskBuffer` (zero allocations) and ALWAYS closes every mask & result object
 * immediately to prevent WASM/GPU memory leaks.
 */
function processSegmentationResult(result) {
  if (!result) return;

  try {
    const masks = result.confidenceMasks;
    if (masks && masks.length > 0) {
      const firstMask = masks[0];
      const w = firstMask.width;
      const h = firstMask.height;
      const srcFloats = firstMask.getAsFloat32Array();

      const ALPHA_CURRENT = 0.75;
      const ALPHA_PREV = 0.25;
      const isSingleMask = masks.length === 1;

      if (w === MASK_WIDTH && h === MASK_HEIGHT) {
        // Fast path: 1:1 overwrite into pre-allocated 256x256 buffer
        if (isSingleMask) {
          for (let i = 0; i < MASK_LENGTH; i++) {
            smoothedMaskBuffer[i] =
              ALPHA_CURRENT * srcFloats[i] + ALPHA_PREV * smoothedMaskBuffer[i];
          }
        } else {
          for (let i = 0; i < MASK_LENGTH; i++) {
            const fg = 1.0 - srcFloats[i];
            const clampedFg = fg < 0.0 ? 0.0 : fg > 1.0 ? 1.0 : fg;
            smoothedMaskBuffer[i] =
              ALPHA_CURRENT * clampedFg + ALPHA_PREV * smoothedMaskBuffer[i];
          }
        }
      } else {
        // Nearest-neighbor resample into the fixed 256x256 buffer without reallocating
        for (let y = 0; y < MASK_HEIGHT; y++) {
          const srcY = ((y * h) / MASK_HEIGHT) | 0;
          const dstRow = y * MASK_WIDTH;
          const srcRow = srcY * w;
          for (let x = 0; x < MASK_WIDTH; x++) {
            const srcX = ((x * w) / MASK_WIDTH) | 0;
            const rawVal = srcFloats[srcRow + srcX];
            const fg = isSingleMask
              ? rawVal
              : Math.max(0.0, Math.min(1.0, 1.0 - rawVal));
            const dstIdx = dstRow + x;
            smoothedMaskBuffer[dstIdx] =
              ALPHA_CURRENT * fg + ALPHA_PREV * smoothedMaskBuffer[dstIdx];
          }
        }
      }

      maskDataTexture.needsUpdate = true;
    }
  } finally {
    // Strictly close all MPMask instances and ImageSegmenterResult so WASM/WebGL memory never leaks
    if (result.confidenceMasks) {
      for (let m = 0; m < result.confidenceMasks.length; m++) {
        if (
          result.confidenceMasks[m] &&
          typeof result.confidenceMasks[m].close === 'function'
        ) {
          result.confidenceMasks[m].close();
        }
      }
    }
    if (
      result.categoryMask &&
      typeof result.categoryMask.close === 'function'
    ) {
      result.categoryMask.close();
    }
    if (typeof result.close === 'function') {
      result.close();
    }
  }
}

// ============================================================================
// LAYER 2 (Foreground 3D): Three.js Holographic Shapes Scene
// ============================================================================
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);
camera.position.set(0, 0, 6);

const ambientLight = new THREE.AmbientLight(0x80f8ff, 1.15);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(3, 5, 6);
scene.add(keyLight);

const rimLight = new THREE.PointLight(0x00f3ff, 3.5, 18);
rimLight.position.set(-4, -3, 3);
scene.add(rimLight);

const renderer = new THREE.WebGLRenderer({
  alpha: true,
  antialias: true,
  powerPreference: 'high-performance'
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
threeContainer.appendChild(renderer.domElement);

// ============================================================================
// WebGL Context Loss & Recovery Handlers (Prevents browser tab freeze/crash)
// ============================================================================
function attachContextLossHandlers(webglCanvas, label) {
  webglCanvas.addEventListener(
    'webglcontextlost',
    (event) => {
      event.preventDefault();
      isContextLost = true;
      if (animFrameId !== null) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }
      console.warn(`[WebGL] Context lost on ${label}. Render loop paused cleanly.`);
      statusBadge.textContent = 'WebGL Context Lost — Recovering GPU resources...';
    },
    false
  );

  webglCanvas.addEventListener(
    'webglcontextrestored',
    () => {
      console.info(`[WebGL] Context restored on ${label}. Resuming loop.`);
      isContextLost = false;
      maskDataTexture.needsUpdate = true;
      webcamTexture.needsUpdate = true;
      statusBadge.textContent = 'WebGL Context Restored — Holo-Lab Active';
      if (animFrameId === null) {
        lastFrameTimeMs = performance.now();
        animFrameId = requestAnimationFrame(drawScene);
      }
    },
    false
  );
}

attachContextLossHandlers(bgRenderer.domElement, 'Background Shader Canvas');
attachContextLossHandlers(renderer.domElement, '3D Shapes Canvas');

const shapes = [];
const particleBursts = [];

// Persistent per-hand tracking state keyed by physical hand slot (0 and 1)
const handSlots = [
  {
    active: false,
    lastPos: { x: 0, y: 0 },
    hasLastPos: false,
    isPinching: false,
    justPinched: false,
    grabbedShape: null,
    grabOffsetWorld: new THREE.Vector3(),
    lastSpawnTime: 0
  },
  {
    active: false,
    lastPos: { x: 0, y: 0 },
    hasLastPos: false,
    isPinching: false,
    justPinched: false,
    grabbedShape: null,
    grabOffsetWorld: new THREE.Vector3(),
    lastSpawnTime: 0
  }
];

// Two-Hand Stretch / Resize Mode State (zero-allocation reusable struct)
const resizeState = {
  active: false,
  isSun: false,
  shape: null,
  startDist: 1,
  startScale: 1
};

function buildHologramPart(geometry, baseColor, materialBucket) {
  const partGroup = new THREE.Group();
  const edgesGeo = new THREE.EdgesGeometry(geometry, 1);

  const edgeMat = new THREE.LineBasicMaterial({
    color: baseColor,
    transparent: true,
    opacity: 1.0,
    blending: THREE.AdditiveBlending
  });
  const wireframeLines = new THREE.LineSegments(edgesGeo, edgeMat);
  partGroup.add(wireframeLines);

  const glowMat = new THREE.MeshBasicMaterial({
    color: baseColor,
    wireframe: true,
    transparent: true,
    opacity: 0.38,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  const glowMesh = new THREE.Mesh(geometry, glowMat);
  glowMesh.scale.setScalar(1.05);
  partGroup.add(glowMesh);

  const coreMat = new THREE.MeshPhongMaterial({
    color: baseColor,
    emissive: baseColor,
    emissiveIntensity: 0.55,
    specular: COLOR_WHITE,
    shininess: 90,
    transparent: true,
    opacity: 0.24,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  const coreMesh = new THREE.Mesh(geometry, coreMat);
  partGroup.add(coreMesh);

  materialBucket.edgeMats.push(edgeMat);
  materialBucket.glowMats.push(glowMat);
  materialBucket.coreMats.push(coreMat);

  return partGroup;
}

function spawnShape(type, worldPosition = _scratchVecA.set(0, 0, 0)) {
  const group = new THREE.Group();
  group.position.copy(worldPosition);

  const materialBucket = {
    edgeMats: [],
    glowMats: [],
    coreMats: []
  };

  let baseColor = COLOR_CYAN;
  let isPulsingMagenta = false;
  let hoverRadiusPx = HOVER_RADIUS_PX;

  switch (type) {
    case 'sphere': {
      const geo = new THREE.SphereGeometry(0.75, 16, 12);
      group.add(buildHologramPart(geo, baseColor, materialBucket));
      break;
    }

    case 'pyramid': {
      const geo = new THREE.ConeGeometry(0.85, 1.35, 4);
      group.add(buildHologramPart(geo, baseColor, materialBucket));
      break;
    }

    case 'ring': {
      const geo = new THREE.TorusGeometry(0.62, 0.22, 12, 28);
      group.add(buildHologramPart(geo, baseColor, materialBucket));
      break;
    }

    case 'house': {
      const baseCubeGeo = new THREE.BoxGeometry(1.1, 0.95, 1.1);
      const basePart = buildHologramPart(baseCubeGeo, baseColor, materialBucket);
      basePart.position.y = -0.35;
      group.add(basePart);

      const roofConeGeo = new THREE.ConeGeometry(0.95, 0.95, 4);
      const roofPart = buildHologramPart(roofConeGeo, baseColor, materialBucket);
      roofPart.rotation.y = Math.PI / 4;
      roofPart.position.y = 0.6;
      group.add(roofPart);
      hoverRadiusPx = 125;
      break;
    }

    case 'planet': {
      const sphereGeo = new THREE.SphereGeometry(0.68, 16, 12);
      const spherePart = buildHologramPart(sphereGeo, baseColor, materialBucket);
      group.add(spherePart);

      const ringGeo = new THREE.TorusGeometry(1.15, 0.11, 10, 32);
      const ringPart = buildHologramPart(ringGeo, baseColor, materialBucket);
      ringPart.rotation.x = Math.PI * 0.36;
      ringPart.rotation.y = Math.PI * 0.12;
      group.add(ringPart);
      hoverRadiusPx = 130;
      break;
    }

    case 'super-cube': {
      baseColor = COLOR_GOLD;
      const superBoxGeo = new THREE.BoxGeometry(1.15, 1.15, 1.15);
      const part = buildHologramPart(superBoxGeo, baseColor, materialBucket);
      group.add(part);
      group.scale.setScalar(2.0);
      hoverRadiusPx = 165;
      break;
    }

    case 'mystery-crystal': {
      baseColor = COLOR_MAGENTA;
      isPulsingMagenta = true;
      const crystalGeo = new THREE.IcosahedronGeometry(0.92, 0);
      const part = buildHologramPart(crystalGeo, baseColor, materialBucket);
      group.add(part);
      hoverRadiusPx = 120;
      break;
    }

    case 'cube':
    default: {
      const geo = new THREE.BoxGeometry(1.15, 1.15, 1.15);
      group.add(buildHologramPart(geo, baseColor, materialBucket));
      break;
    }
  }

  scene.add(group);

  const shapeObj = {
    id: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    type,
    group,
    baseScale: type === 'super-cube' ? 2.0 : 1.0,
    baseColor: baseColor.clone(),
    hoverColor: baseColor.clone().lerp(COLOR_WHITE, 0.42),
    isPulsingMagenta,
    hoverRadiusPx,
    visualState: 'idle',
    ...materialBucket,
    rotSpeedX: 0.0045 + Math.random() * 0.0035,
    rotSpeedY: 0.0075 + Math.random() * 0.0045,
    floatPhase: Math.random() * Math.PI * 2
  };

  shapes.push(shapeObj);
  updateTopStatusBar(0);
  return shapeObj;
}

/**
 * Properly disposes all geometries and materials on removed meshes so GPU memory stays flat.
 */
function removeShape(targetShape) {
  if (resizeState.shape === targetShape) {
    resizeState.active = false;
    resizeState.shape = null;
  }
  const idx = shapes.indexOf(targetShape);
  if (idx !== -1) {
    shapes.splice(idx, 1);
  }

  scene.remove(targetShape.group);
  const disposedGeometries = new Set();
  const disposedMaterials = new Set();

  targetShape.group.traverse((child) => {
    if (child.geometry && !disposedGeometries.has(child.geometry)) {
      disposedGeometries.add(child.geometry);
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        for (const m of child.material) {
          if (m && !disposedMaterials.has(m)) {
            disposedMaterials.add(m);
            m.dispose();
          }
        }
      } else if (!disposedMaterials.has(child.material)) {
        disposedMaterials.add(child.material);
        child.material.dispose();
      }
    }
  });

  targetShape.edgeMats.length = 0;
  targetShape.glowMats.length = 0;
  targetShape.coreMats.length = 0;

  let activeCount = 0;
  for (let i = 0; i < handSlots.length; i++) {
    if (handSlots[i].active) activeCount++;
  }
  updateTopStatusBar(activeCount);
}

function clearAllShapes() {
  resizeState.active = false;
  resizeState.shape = null;
  for (const slot of handSlots) {
    slot.grabbedShape = null;
  }

  for (let i = shapes.length - 1; i >= 0; i--) {
    const s = shapes[i];
    triggerParticleBurst(s.group.position, s.baseColor);
    removeShape(s);
  }

  playClearSweep();
  clearAllBtn.classList.add('is-active');
  setTimeout(() => clearAllBtn.classList.remove('is-active'), 350);
  statusBadge.textContent = 'Workspace Cleared — Pinch + SHAPES or ☀ SUN to spawn';
}

// Mouse / Touch convenience fallbacks for buttons & dragging the Sun
clearAllBtn.addEventListener('click', () => clearAllShapes());

shapesToggleBtn.addEventListener('click', () => {
  const now = performance.now();
  if (now - lastMenuToggleTime >= TOGGLE_COOLDOWN_MS) {
    lastMenuToggleTime = now;
    setShapesMenuOpen(!isShapesMenuOpen);
    playPinchBlip(isShapesMenuOpen ? 1.18 : 0.88);
  }
});

sunToggleBtn.addEventListener('click', () => {
  const now = performance.now();
  if (now - lastSunToggleTime >= TOGGLE_COOLDOWN_MS) {
    lastSunToggleTime = now;
    setSunActive(!isSunActive);
    playPinchBlip(isSunActive ? 1.35 : 0.78);
    statusBadge.textContent = isSunActive
      ? '☀ SUN Spawned — Pinch & drag Sun with Index Tip (8), or eclipse it with your open hand!'
      : '☀ SUN Turned Off';
  }
});

for (const btn of hudButtons) {
  btn.addEventListener('click', () => {
    const shapeType = btn.dataset.shape;
    _scratchVecA.set((Math.random() - 0.5) * 1.5, (Math.random() - 0.5) * 1.0, 0);
    spawnShape(shapeType, _scratchVecA);
    playPinchBlip(1.1);
    setShapesMenuOpen(false);
  });
}

bgShaderContainer.addEventListener('pointerdown', (e) => {
  if (!isSunActive) return;
  isMouseDraggingSun = true;
  bgShaderContainer.setPointerCapture(e.pointerId);
  const uvX = THREE.MathUtils.clamp(e.clientX / window.innerWidth, 0.02, 0.98);
  const uvY = THREE.MathUtils.clamp(1.0 - e.clientY / window.innerHeight, 0.02, 0.98);
  sunTargetUv.set(uvX, uvY);
});

bgShaderContainer.addEventListener('pointermove', (e) => {
  if (!isMouseDraggingSun || !isSunActive) return;
  const uvX = THREE.MathUtils.clamp(e.clientX / window.innerWidth, 0.02, 0.98);
  const uvY = THREE.MathUtils.clamp(1.0 - e.clientY / window.innerHeight, 0.02, 0.98);
  sunTargetUv.set(uvX, uvY);
});

bgShaderContainer.addEventListener('pointerup', () => {
  if (isMouseDraggingSun && isSunActive) {
    const sunScreenX = sunCurrentUv.x * canvas.width;
    const sunScreenY = (1.0 - sunCurrentUv.y) * canvas.height;
    if (isPointOverTrashZone(sunScreenX, sunScreenY)) {
      const trashCenter = getTrashZoneCenterScreen();
      const trashWorld = screenToWorldOnPlane(trashCenter.x, trashCenter.y, 0, _scratchVecA);
      triggerParticleBurst(trashWorld, COLOR_DANGER_RED);
      playTrashDeleteWhoosh();
      setSunActive(false);
      setTrashZoneDangerState(false);
      statusBadge.textContent = '☀ SUN Deleted in Trash — Pinch ☀ SUN anytime to spawn it again';
    }
  }
  isMouseDraggingSun = false;
});

let _lastDisplayedHands = -1;
let _lastDisplayedShapes = -1;

function updateTopStatusBar(handsCount) {
  if (handsCount !== _lastDisplayedHands) {
    _lastDisplayedHands = handsCount;
    handsCountEl.textContent = `Hands Detected: ${handsCount}/2`;
    handsDotEl.classList.toggle('is-online', handsCount > 0);
  }
  if (shapes.length !== _lastDisplayedShapes) {
    _lastDisplayedShapes = shapes.length;
    shapesCountEl.textContent = `Active Shapes: ${shapes.length}`;
  }
}

function resolveCombinationRecipe(typeA, typeB) {
  const pair = typeA < typeB ? `${typeA}+${typeB}` : `${typeB}+${typeA}`;

  if (pair === 'cube+pyramid') {
    return {
      type: 'house',
      label: 'COMBINED: HOUSE!',
      colorHex: '#00f3ff',
      threeColor: COLOR_CYAN
    };
  }
  if (pair === 'ring+sphere') {
    return {
      type: 'planet',
      label: 'COMBINED: PLANET!',
      colorHex: '#00f3ff',
      threeColor: COLOR_CYAN
    };
  }
  if (pair === 'cube+cube') {
    return {
      type: 'super-cube',
      label: 'COMBINED: LARGE SUPER-CUBE!',
      colorHex: '#ffd700',
      threeColor: COLOR_GOLD
    };
  }
  return {
    type: 'mystery-crystal',
    label: 'COMBINED: MYSTERY CRYSTAL!',
    colorHex: '#ff00ff',
    threeColor: COLOR_MAGENTA
  };
}

function triggerParticleBurst(worldPosition, burstColor = COLOR_CYAN) {
  const count = 48;
  const positions = new Float32Array(count * 3);
  const velocities = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const idx = i * 3;
    positions[idx] = worldPosition.x;
    positions[idx + 1] = worldPosition.y;
    positions[idx + 2] = worldPosition.z;

    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const speed = 1.8 + Math.random() * 3.8;

    velocities[idx] = Math.sin(phi) * Math.cos(theta) * speed;
    velocities[idx + 1] = Math.sin(phi) * Math.sin(theta) * speed;
    velocities[idx + 2] = Math.cos(phi) * speed;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: burstColor,
    size: 0.14,
    transparent: true,
    opacity: 1.0,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  const ringGeo = new THREE.RingGeometry(0.15, 0.24, 24);
  const ringMat = new THREE.MeshBasicMaterial({
    color: burstColor,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  const shockRing = new THREE.Mesh(ringGeo, ringMat);
  shockRing.position.copy(worldPosition);
  scene.add(shockRing);

  particleBursts.push({
    points,
    geometry,
    material,
    velocities,
    count,
    shockRing,
    ringGeo,
    ringMat,
    createdAt: performance.now(),
    durationMs: 700
  });
}

function updateParticleBursts(nowInMs, dtSeconds) {
  for (let i = particleBursts.length - 1; i >= 0; i--) {
    const burst = particleBursts[i];
    const elapsed = nowInMs - burst.createdAt;
    const progress = elapsed / burst.durationMs;

    if (progress >= 1) {
      scene.remove(burst.points);
      scene.remove(burst.shockRing);
      burst.geometry.dispose();
      burst.material.dispose();
      burst.ringGeo.dispose();
      burst.ringMat.dispose();
      particleBursts.splice(i, 1);
      continue;
    }

    const posAttr = burst.geometry.getAttribute('position');
    const arr = posAttr.array;
    const vels = burst.velocities;
    const len = burst.count * 3;

    for (let k = 0; k < len; k++) {
      arr[k] += vels[k] * dtSeconds;
      vels[k] *= 0.94;
    }
    posAttr.needsUpdate = true;

    burst.material.opacity = 1 - progress;
    burst.shockRing.scale.setScalar(1 + progress * 6.5);
    burst.ringMat.opacity = (1 - progress) * 0.85;
  }
}

function showCombineAnnouncement(text, colorHex) {
  if (combineBannerTimeout) {
    clearTimeout(combineBannerTimeout);
  }

  combineBanner.textContent = text;
  combineBanner.style.color = colorHex;
  combineBanner.style.borderColor = colorHex;
  combineBanner.style.boxShadow = `0 0 28px ${colorHex}80, inset 0 0 16px ${colorHex}40`;
  combineBanner.classList.add('is-visible');

  combineBannerTimeout = setTimeout(() => {
    combineBanner.classList.remove('is-visible');
  }, 2000);
}

function setShapeVisualState(shape, state, nowInMs = 0) {
  shape.visualState = state;

  let targetColor = shape.baseColor;
  let coreOpacity = 0.24;
  let glowOpacity = 0.38;
  let emissiveIntensity = 0.55;
  const currentBaseScale = shape.baseScale || 1.0;
  let targetScale = currentBaseScale;

  if (state === 'danger') {
    targetColor = COLOR_DANGER_RED;
    coreOpacity = 0.42;
    glowOpacity = 0.68;
    emissiveIntensity = 0.95;
    targetScale = currentBaseScale * 0.68;
  } else if (state === 'resizing') {
    targetColor = COLOR_GOLD;
    coreOpacity = 0.4;
    glowOpacity = 0.72;
    emissiveIntensity = 0.92;
  } else if (state === 'held') {
    targetColor = COLOR_HELD_GREEN;
    coreOpacity = 0.36;
    glowOpacity = 0.62;
    emissiveIntensity = 0.82;
  } else if (state === 'hover') {
    targetColor = shape.hoverColor;
    coreOpacity = 0.3;
    glowOpacity = 0.48;
    emissiveIntensity = 0.68;
  } else if (shape.isPulsingMagenta) {
    const pulse = 0.5 + 0.5 * Math.sin(nowInMs * 0.006);
    coreOpacity = 0.18 + pulse * 0.24;
    glowOpacity = 0.3 + pulse * 0.45;
    emissiveIntensity = 0.4 + pulse * 0.55;
    targetScale = currentBaseScale * (1 + 0.06 * Math.sin(nowInMs * 0.006));
  }

  // When actively in two-hand resize mode, the stretch handler directly lerps group.scale
  if (state !== 'resizing') {
    const currentScale = shape.group.scale.x;
    const nextScale = THREE.MathUtils.lerp(currentScale, targetScale, 0.22);
    shape.group.scale.setScalar(nextScale);
  }

  for (let i = 0; i < shape.edgeMats.length; i++) {
    shape.edgeMats[i].color.copy(targetColor);
  }
  for (let i = 0; i < shape.glowMats.length; i++) {
    shape.glowMats[i].color.copy(targetColor);
    shape.glowMats[i].opacity = glowOpacity;
  }
  for (let i = 0; i < shape.coreMats.length; i++) {
    const m = shape.coreMats[i];
    m.color.copy(targetColor);
    if (m.emissive) {
      m.emissive.copy(targetColor);
      m.emissiveIntensity = emissiveIntensity;
    }
    m.opacity = coreOpacity;
  }
}

// Spawn the initial glowing cyan wireframe 3D Cube in the center of the screen
spawnShape('cube', _scratchVecA.set(0, 0, 0));

/**
 * Unprojects screen pixel coordinates onto z = targetZ without allocating new Vector3 objects.
 */
function screenToWorldOnPlane(screenX, screenY, targetZ = 0, outVec = _scratchVecA) {
  const ndcX = (screenX / canvas.width) * 2 - 1;
  const ndcY = -(screenY / canvas.height) * 2 + 1;

  _scratchVecB.set(ndcX, ndcY, 0.5).unproject(camera);
  _scratchVecB.sub(camera.position).normalize();

  const distance = (targetZ - camera.position.z) / _scratchVecB.z;
  return outVec.copy(camera.position).add(_scratchVecB.multiplyScalar(distance));
}

const _screenOut = { x: 0, y: 0 };
function worldToScreen(worldPos, out = _screenOut) {
  _scratchProjected.copy(worldPos).project(camera);
  out.x = (_scratchProjected.x * 0.5 + 0.5) * canvas.width;
  out.y = (-_scratchProjected.y * 0.5 + 0.5) * canvas.height;
  return out;
}

function resizeAll() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const pixelRatio = Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO);

  canvas.width = width;
  canvas.height = height;

  bgRenderer.setPixelRatio(pixelRatio);
  bgRenderer.setSize(width, height);
  bgUniforms.uResolution.value.set(width, height);

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(width, height);
}

window.addEventListener('resize', resizeAll);
resizeAll();

const _coverLayout = { drawWidth: 0, drawHeight: 0, offsetX: 0, offsetY: 0 };
function getVideoCoverLayout(videoWidth, videoHeight, canvasWidth, canvasHeight) {
  const videoAspect = videoWidth / videoHeight;
  const canvasAspect = canvasWidth / canvasHeight;

  if (canvasAspect > videoAspect) {
    _coverLayout.drawWidth = canvasWidth;
    _coverLayout.drawHeight = canvasWidth / videoAspect;
    _coverLayout.offsetX = 0;
    _coverLayout.offsetY = (canvasHeight - _coverLayout.drawHeight) / 2;
  } else {
    _coverLayout.drawHeight = canvasHeight;
    _coverLayout.drawWidth = canvasHeight * videoAspect;
    _coverLayout.offsetX = (canvasWidth - _coverLayout.drawWidth) / 2;
    _coverLayout.offsetY = 0;
  }

  return _coverLayout;
}

function isPointInsideElement(screenX, screenY, elem) {
  const rect = elem.getBoundingClientRect();
  return (
    screenX >= rect.left &&
    screenX <= rect.right &&
    screenY >= rect.top &&
    screenY <= rect.bottom
  );
}

const _trashCenter = { x: 0, y: 0 };
function getTrashZoneCenterScreen() {
  const rect = trashRingEl.getBoundingClientRect();
  _trashCenter.x = rect.left + rect.width * 0.5;
  _trashCenter.y = rect.top + rect.height * 0.5;
  return _trashCenter;
}

function isPointOverTrashZone(screenX, screenY) {
  const center = getTrashZoneCenterScreen();
  return Math.hypot(screenX - center.x, screenY - center.y) <= TRASH_DROP_RADIUS_PX;
}

let _lastTrashDangerState = false;
function setTrashZoneDangerState(isDanger) {
  if (isDanger === _lastTrashDangerState) return;
  _lastTrashDangerState = isDanger;
  trashZoneEl.classList.toggle('is-danger', isDanger);
  trashLabelEl.textContent = isDanger ? 'RELEASE TO DELETE' : 'TRASH';
}

function getHoveredHudButton(screenX, screenY) {
  if (!isShapesMenuOpen) return null;
  for (let i = 0; i < hudButtons.length; i++) {
    const btn = hudButtons[i];
    if (isPointInsideElement(screenX, screenY, btn)) {
      return btn;
    }
  }
  return null;
}

function assignHandsToPersistentSlots(rawHandsCount) {
  _trackedHandsBuffer.length = 0;

  if (rawHandsCount === 0) {
    for (let i = 0; i < handSlots.length; i++) {
      const slot = handSlots[i];
      slot.active = false;
      slot.isPinching = false;
      slot.grabbedShape = null;
      slot.hasLastPos = false;
    }
    sunGrabbedBySlotIndex = null;
    return _trackedHandsBuffer;
  }

  if (rawHandsCount === 1) {
    const hand = _rawHandsPool[0];
    let chosenSlotIdx = 0;
    if (
      handSlots[1].active &&
      handSlots[1].hasLastPos &&
      (!handSlots[0].active || !handSlots[0].hasLastPos)
    ) {
      chosenSlotIdx = 1;
    } else if (handSlots[0].hasLastPos && handSlots[1].hasLastPos) {
      const d0 = Math.hypot(
        hand.wrist.x - handSlots[0].lastPos.x,
        hand.wrist.y - handSlots[0].lastPos.y
      );
      const d1 = Math.hypot(
        hand.wrist.x - handSlots[1].lastPos.x,
        hand.wrist.y - handSlots[1].lastPos.y
      );
      chosenSlotIdx = d1 < d0 ? 1 : 0;
    }

    const otherSlotIdx = 1 - chosenSlotIdx;
    handSlots[otherSlotIdx].active = false;
    handSlots[otherSlotIdx].isPinching = false;
    handSlots[otherSlotIdx].grabbedShape = null;
    handSlots[otherSlotIdx].hasLastPos = false;
    if (sunGrabbedBySlotIndex === otherSlotIdx) {
      sunGrabbedBySlotIndex = null;
    }

    _trackedHandsBuffer.push(bindHandToSlot(hand, chosenSlotIdx));
    return _trackedHandsBuffer;
  }

  const h0 = _rawHandsPool[0];
  const h1 = _rawHandsPool[1];
  let swap = false;

  if (handSlots[0].hasLastPos && handSlots[1].hasLastPos) {
    const costDirect =
      Math.hypot(h0.wrist.x - handSlots[0].lastPos.x, h0.wrist.y - handSlots[0].lastPos.y) +
      Math.hypot(h1.wrist.x - handSlots[1].lastPos.x, h1.wrist.y - handSlots[1].lastPos.y);
    const costSwapped =
      Math.hypot(h0.wrist.x - handSlots[1].lastPos.x, h0.wrist.y - handSlots[1].lastPos.y) +
      Math.hypot(h1.wrist.x - handSlots[0].lastPos.x, h1.wrist.y - handSlots[0].lastPos.y);
    swap = costSwapped < costDirect;
  } else if (handSlots[1].active && handSlots[1].hasLastPos && !handSlots[0].active) {
    const d0 = Math.hypot(h0.wrist.x - handSlots[1].lastPos.x, h0.wrist.y - handSlots[1].lastPos.y);
    const d1 = Math.hypot(h1.wrist.x - handSlots[1].lastPos.x, h1.wrist.y - handSlots[1].lastPos.y);
    swap = d0 < d1;
  } else {
    swap = h0.wrist.x > h1.wrist.x;
  }

  if (swap) {
    _trackedHandsBuffer.push(bindHandToSlot(h1, 0), bindHandToSlot(h0, 1));
  } else {
    _trackedHandsBuffer.push(bindHandToSlot(h0, 0), bindHandToSlot(h1, 1));
  }
  return _trackedHandsBuffer;
}

function bindHandToSlot(rawHand, slotIndex) {
  const slot = handSlots[slotIndex];
  slot.active = true;
  slot.lastPos.x = rawHand.wrist.x;
  slot.lastPos.y = rawHand.wrist.y;
  slot.hasLastPos = true;

  const wasPinching = slot.isPinching;
  const isPinching = wasPinching
    ? rawHand.pinchDistPx < PINCH_RELEASE_PX
    : rawHand.pinchDistPx < PINCH_START_PX;
  const justPinched = !wasPinching && isPinching;

  if (justPinched) {
    playPinchBlip(slotIndex === 0 ? 1.0 : 1.12);
  }

  slot.isPinching = isPinching;
  slot.justPinched = justPinched;
  rawHand.slotIndex = slotIndex;
  rawHand.slot = slot;
  rawHand.isPinching = isPinching;
  rawHand.justPinched = justPinched;
  return rawHand;
}

let lastFrameTimeMs = performance.now();

function drawScene(nowInMs) {
  if (isContextLost) return;

  const dtSeconds = Math.min((nowInMs - lastFrameTimeMs) / 1000, 0.1);
  lastFrameTimeMs = nowInMs;

  const width = canvas.width;
  const height = canvas.height;

  // Clear transparent 2D overlay canvas (Layer 3)
  ctx.clearRect(0, 0, width, height);

  // Smoothly update background shader uniforms
  bgUniforms.uTime.value = nowInMs * 0.001;
  sunTransition = THREE.MathUtils.lerp(sunTransition, isSunActive ? 1.0 : 0.0, 0.14);
  bgUniforms.uSunActive.value = sunTransition;
  sunCurrentUv.lerp(sunTargetUv, 0.22);

  const isSunCurrentlyHeld =
    sunGrabbedBySlotIndex !== null || isMouseDraggingSun;
  bgUniforms.uSunHeld.value = THREE.MathUtils.lerp(
    bgUniforms.uSunHeld.value,
    isSunCurrentlyHeld ? 1.0 : 0.0,
    0.25
  );

  // Guard MediaPipe calls: only run when video.readyState >= 2 and video.currentTime has advanced
  if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
    bgUniforms.uVideoResolution.value.set(video.videoWidth, video.videoHeight);

    const layout = getVideoCoverLayout(
      video.videoWidth,
      video.videoHeight,
      width,
      height
    );

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      // Guarantee strictly monotonically increasing integer timestamps so MediaPipe never crashes
      let mpTimestamp = Math.floor(nowInMs);
      if (mpTimestamp <= lastMediaPipeTimestampMs) {
        mpTimestamp = lastMediaPipeTimestampMs + 1;
      }
      lastMediaPipeTimestampMs = mpTimestamp;

      try {
        if (handLandmarker) {
          latestHandResults = handLandmarker.detectForVideo(video, mpTimestamp);
        }

        // Only run ImageSegmenter when the Sun is actually toggled ON, and throttle to every 3rd video frame (~10-15 FPS)
        if (isSunActive && imageSegmenter) {
          segmentFrameCounter = (segmentFrameCounter + 1) % 3;
          if (segmentFrameCounter === 0) {
            imageSegmenter.segmentForVideo(
              video,
              mpTimestamp,
              processSegmentationResult
            );
          }
        }
      } catch (mpErr) {
        console.warn('MediaPipe frame warning:', mpErr);
      }
    }

    let rawHandsCount = 0;

    if (latestHandResults && latestHandResults.landmarks) {
      const numDetected = Math.min(latestHandResults.landmarks.length, 2);
      for (let h = 0; h < numDetected; h++) {
        const landmarks = latestHandResults.landmarks[h];
        const points = mapLandmarksToScreen(
          landmarks,
          layout,
          width,
          _screenPointsPool[h]
        );
        drawHand(points);

        const wrist = points[0];
        const thumbTip = points[4];
        const indexTip = points[8];
        const middleTip = points[12];
        const pinchDistPx = Math.hypot(
          thumbTip.x - middleTip.x,
          thumbTip.y - middleTip.y
        );

        const rawSlot = _rawHandsPool[h];
        rawSlot.wrist = wrist;
        rawSlot.indexTip = indexTip;
        rawSlot.thumbTip = thumbTip;
        rawSlot.middleTip = middleTip;
        rawSlot.pinchDistPx = pinchDistPx;
        rawHandsCount++;
      }
    }

    const trackedHands = assignHandsToPersistentSlots(rawHandsCount);
    updateHudShapesAndSunInteraction(trackedHands, nowInMs);
    updateTopStatusBar(trackedHands.length);
  }

  // Render Layer 1: Background Webcam + Solar Eclipse / God-Ray Shader
  bgRenderer.render(bgScene, bgCamera);

  // Subtle slow idle rotation when 3D shapes are floating untouched
  for (let i = 0; i < shapes.length; i++) {
    const shape = shapes[i];
    const isHeld =
      handSlots[0].grabbedShape === shape ||
      handSlots[1].grabbedShape === shape;
    const speedFactor = isHeld ? 0.35 : 1.0;

    shape.group.rotation.x += shape.rotSpeedX * speedFactor;
    shape.group.rotation.y += shape.rotSpeedY * speedFactor;

    if (!isHeld) {
      shape.group.position.y +=
        Math.sin(nowInMs * 0.0022 + shape.floatPhase) * 0.0009;
    }
  }

  // Update active collision particle bursts
  updateParticleBursts(nowInMs, dtSeconds);

  // Render Layer 2: Foreground 3D Holographic Shapes
  renderer.render(scene, camera);

  animFrameId = requestAnimationFrame(drawScene);
}

function updateHudShapesAndSunInteraction(trackedHands, nowInMs) {
  _hoveredButtonsSet.clear();
  _activeButtonsSet.clear();
  _hoveredShapesSet.clear();
  _dangerShapesSet.clear();
  _heldShapesSet.clear();

  let clearBtnHovered = false;
  let toggleBtnHovered = false;
  let sunBtnHovered = false;
  let anyObjectOverTrash = false;
  let isSunOverTrash = false;

  const sunScreenX = sunCurrentUv.x * canvas.width;
  const sunScreenY = (1.0 - sunCurrentUv.y) * canvas.height;

  if (isMouseDraggingSun && isSunActive) {
    if (isPointOverTrashZone(sunScreenX, sunScreenY)) {
      anyObjectOverTrash = true;
      isSunOverTrash = true;
    }
  }

  // ==========================================================================
  // 0. Check & Manage Active Two-Hand Resize Mode (3D Shapes or Sun Light Source)
  // ==========================================================================
  if (resizeState.active) {
    const h0 = trackedHands.length > 0 ? trackedHands[0] : null;
    const h1 = trackedHands.length > 1 ? trackedHands[1] : null;

    if (resizeState.isSun) {
      const bothHandsStillPinchingSun =
        isSunActive &&
        trackedHands.length === 2 &&
        h0 &&
        h1 &&
        h0.isPinching &&
        h1.isPinching;

      if (bothHandsStillPinchingSun) {
        const currentDist = Math.max(
          12,
          Math.hypot(
            h0.indexTip.x - h1.indexTip.x,
            h0.indexTip.y - h1.indexTip.y
          )
        );
        const targetScale = THREE.MathUtils.clamp(
          resizeState.startScale * (currentDist / resizeState.startDist),
          0.25,
          4.0
        );
        sunScale = THREE.MathUtils.lerp(sunScale, targetScale, 0.25);
        bgUniforms.uSunScale.value = sunScale;

        const midScreenX = (h0.indexTip.x + h1.indexTip.x) * 0.5;
        const midScreenY = (h0.indexTip.y + h1.indexTip.y) * 0.5;
        const uvX = THREE.MathUtils.clamp(midScreenX / canvas.width, 0.03, 0.97);
        const uvY = THREE.MathUtils.clamp(1.0 - midScreenY / canvas.height, 0.03, 0.97);
        sunTargetUv.set(uvX, uvY);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(h0.indexTip.x, h0.indexTip.y);
        ctx.lineTo(h1.indexTip.x, h1.indexTip.y);
        ctx.strokeStyle = 'rgba(255, 215, 0, 0.35)';
        ctx.lineWidth = 6;
        ctx.stroke();

        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();

        drawSunGrabIndicator(midScreenX, midScreenY, true, false);
        drawPointerAndPinchFeedback(h0, true, true);
        drawPointerAndPinchFeedback(h1, true, true);

        statusBadge.textContent = `RESIZING ☀ SUN: ${sunScale.toFixed(2)}x — Pull apart or push together`;
      } else {
        resizeState.active = false;
        resizeState.isSun = false;
        // Let whichever hand is still pinching keep dragging the Sun
        sunGrabbedBySlotIndex = null;
        for (let k = 0; k < trackedHands.length; k++) {
          if (trackedHands[k].isPinching) {
            sunGrabbedBySlotIndex = trackedHands[k].slotIndex;
            break;
          }
        }
        statusBadge.textContent = 'Sun Resize Complete — Continue dragging or release pinch';
      }
    } else {
      const sharedShape = resizeState.shape;
      const bothHandsStillPinchingSameShape =
        sharedShape &&
        shapes.includes(sharedShape) &&
        trackedHands.length === 2 &&
        h0 &&
        h1 &&
        h0.isPinching &&
        h1.isPinching &&
        h0.slot.grabbedShape === sharedShape &&
        h1.slot.grabbedShape === sharedShape;

      if (bothHandsStillPinchingSameShape) {
        const currentDist = Math.max(
          12,
          Math.hypot(
            h0.indexTip.x - h1.indexTip.x,
            h0.indexTip.y - h1.indexTip.y
          )
        );
        const targetScale = THREE.MathUtils.clamp(
          resizeState.startScale * (currentDist / resizeState.startDist),
          0.25,
          4.0
        );
        const nextScale = THREE.MathUtils.lerp(
          sharedShape.group.scale.x,
          targetScale,
          0.25
        );
        sharedShape.group.scale.setScalar(nextScale);
        sharedShape.baseScale = nextScale;

        const midScreenX = (h0.indexTip.x + h1.indexTip.x) * 0.5;
        const midScreenY = (h0.indexTip.y + h1.indexTip.y) * 0.5;
        const midWorld = screenToWorldOnPlane(midScreenX, midScreenY, 0, _scratchVecA);
        sharedShape.group.position.copy(midWorld);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(h0.indexTip.x, h0.indexTip.y);
        ctx.lineTo(h1.indexTip.x, h1.indexTip.y);
        ctx.strokeStyle = 'rgba(255, 215, 0, 0.35)';
        ctx.lineWidth = 6;
        ctx.stroke();

        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();

        drawPointerAndPinchFeedback(h0, true, true);
        drawPointerAndPinchFeedback(h1, true, true);

        statusBadge.textContent = `RESIZING ${sharedShape.type.toUpperCase()}: ${nextScale.toFixed(2)}x — Pull apart or push together`;
      } else {
        if (sharedShape && shapes.includes(sharedShape)) {
          sharedShape.baseScale = sharedShape.group.scale.x;

          for (let k = 0; k < trackedHands.length; k++) {
            const hk = trackedHands[k];
            if (hk.slot.grabbedShape === sharedShape) {
              if (hk.isPinching) {
                const pointerWorld = screenToWorldOnPlane(
                  hk.indexTip.x,
                  hk.indexTip.y,
                  0,
                  _scratchVecA
                );
                hk.slot.grabOffsetWorld
                  .copy(sharedShape.group.position)
                  .sub(pointerWorld);
              } else {
                hk.slot.grabbedShape = null;
              }
            }
          }
        } else {
          handSlots[0].grabbedShape = null;
          handSlots[1].grabbedShape = null;
        }

        resizeState.active = false;
        resizeState.shape = null;
        statusBadge.textContent = 'Resize Complete — Continue dragging or release pinch to place';
      }
    }
  }

  for (let i = 0; i < trackedHands.length; i++) {
    const hand = trackedHands[i];
    const slot = hand.slot;

    // Skip per-hand dragging if this hand is currently participating in two-hand resize mode
    if (
      resizeState.active &&
      (resizeState.isSun || slot.grabbedShape === resizeState.shape)
    ) {
      continue;
    }

    // 1a. If this hand is currently dragging the Sun light source
    if (sunGrabbedBySlotIndex === hand.slotIndex) {
      const overTrash =
        isPointOverTrashZone(hand.indexTip.x, hand.indexTip.y) ||
        isPointOverTrashZone(sunScreenX, sunScreenY);

      if (isSunActive && hand.isPinching) {
        const uvX = THREE.MathUtils.clamp(hand.indexTip.x / canvas.width, 0.03, 0.97);
        const uvY = THREE.MathUtils.clamp(1.0 - hand.indexTip.y / canvas.height, 0.03, 0.97);
        sunTargetUv.set(uvX, uvY);

        if (overTrash) {
          anyObjectOverTrash = true;
          isSunOverTrash = true;
          statusBadge.textContent = 'RELEASE PINCH TO DELETE ☀ SUN';
        }
        drawSunGrabIndicator(sunScreenX, sunScreenY, true, overTrash);
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      } else {
        sunGrabbedBySlotIndex = null;
        if (isSunActive && overTrash) {
          const trashCenter = getTrashZoneCenterScreen();
          const trashWorld = screenToWorldOnPlane(trashCenter.x, trashCenter.y, 0, _scratchVecA);
          triggerParticleBurst(trashWorld, COLOR_DANGER_RED);
          playTrashDeleteWhoosh();
          setSunActive(false);
          statusBadge.textContent =
            '☀ SUN Deleted in Trash — Pinch ☀ SUN anytime to spawn it again';
          continue;
        }
      }
    }

    // 1b. If this hand is currently holding a 3D shape
    if (slot.grabbedShape) {
      if (!shapes.includes(slot.grabbedShape)) {
        slot.grabbedShape = null;
      } else {
        const heldShape = slot.grabbedShape;
        const shapeScreen = worldToScreen(heldShape.group.position);
        const overTrash =
          isPointOverTrashZone(hand.indexTip.x, hand.indexTip.y) ||
          isPointOverTrashZone(shapeScreen.x, shapeScreen.y);

        if (hand.isPinching) {
          const pointerWorld = screenToWorldOnPlane(
            hand.indexTip.x,
            hand.indexTip.y,
            0,
            _scratchVecA
          );
          heldShape.group.position
            .copy(pointerWorld)
            .add(slot.grabOffsetWorld);

          if (overTrash) {
            anyObjectOverTrash = true;
            _dangerShapesSet.add(heldShape);
            statusBadge.textContent = `RELEASE PINCH TO DELETE ${heldShape.type.toUpperCase()}`;
          }
          drawPointerAndPinchFeedback(hand, true, true);
          continue;
        } else {
          slot.grabbedShape = null;
          if (overTrash) {
            const trashCenter = getTrashZoneCenterScreen();
            const trashWorld = screenToWorldOnPlane(trashCenter.x, trashCenter.y, 0, _scratchVecA);
            triggerParticleBurst(trashWorld, COLOR_DANGER_RED);
            playTrashDeleteWhoosh();
            removeShape(heldShape);
            statusBadge.textContent = `Deleted ${heldShape.type.toUpperCase()} in Trash`;
            continue;
          }
        }
      }
    }

    // 1c. Generous Second-Hand Grab for Two-Hand Stretching (3D Shapes & Sun):
    // If the OTHER hand is already holding a shape (or the Sun), check if THIS hand is within
    // a generous radius (1.8x current bounding radius) and pinches to enter resize mode!
    const otherSlotIdx = 1 - hand.slotIndex;
    const otherSlot = handSlots[otherSlotIdx];
    const otherHeldShape = otherSlot.active && otherSlot.isPinching ? otherSlot.grabbedShape : null;

    if (
      isSunActive &&
      sunGrabbedBySlotIndex === otherSlotIdx &&
      !slot.grabbedShape
    ) {
      const distToSunPx = Math.hypot(
        hand.indexTip.x - sunScreenX,
        hand.indexTip.y - sunScreenY
      );
      const generousSunGrabRadius = Math.max(
        150,
        SUN_HOVER_RADIUS_PX * Math.max(0.75, sunScale) * 1.8
      );
      if (distToSunPx <= generousSunGrabRadius) {
        if (hand.isPinching) {
          const otherHand = trackedHands.find((h) => h.slotIndex === otherSlotIdx);
          const initialDist = otherHand
            ? Math.max(
                18,
                Math.hypot(
                  hand.indexTip.x - otherHand.indexTip.x,
                  hand.indexTip.y - otherHand.indexTip.y
                )
              )
            : 60;
          resizeState.active = true;
          resizeState.isSun = true;
          resizeState.shape = null;
          resizeState.startDist = initialDist;
          resizeState.startScale = sunScale;
          drawPointerAndPinchFeedback(hand, true, true);
          statusBadge.textContent =
            'SUN RESIZE MODE LOCKED — Pull hands apart or push together to scale ☀ SUN';
          continue;
        } else {
          drawPointerAndPinchFeedback(hand, true, false);
          continue;
        }
      }
    }

    if (otherHeldShape && shapes.includes(otherHeldShape) && !slot.grabbedShape) {
      const otherShapeScreen = worldToScreen(otherHeldShape.group.position);
      const distToHeldShapePx = Math.hypot(
        hand.indexTip.x - otherShapeScreen.x,
        hand.indexTip.y - otherShapeScreen.y
      );
      const currentScaleRatio =
        otherHeldShape.group.scale.x / (otherHeldShape.type === 'super-cube' ? 2.0 : 1.0);
      const baseBoundingRadius = otherHeldShape.hoverRadiusPx || HOVER_RADIUS_PX;
      const generousSecondGrabRadius = Math.max(
        140,
        baseBoundingRadius * Math.max(0.75, currentScaleRatio) * 1.8
      );

      if (distToHeldShapePx <= generousSecondGrabRadius) {
        if (hand.isPinching) {
          // Lock BOTH hands onto that shape and enter resize mode
          slot.grabbedShape = otherHeldShape;
          const otherHand = trackedHands.find((h) => h.slotIndex === otherSlotIdx);
          const initialDist = otherHand
            ? Math.max(
                18,
                Math.hypot(
                  hand.indexTip.x - otherHand.indexTip.x,
                  hand.indexTip.y - otherHand.indexTip.y
                )
              )
            : 60;

          resizeState.active = true;
          resizeState.shape = otherHeldShape;
          resizeState.startDist = initialDist;
          resizeState.startScale = otherHeldShape.group.scale.x;

          drawPointerAndPinchFeedback(hand, true, true);
          statusBadge.textContent = `RESIZE MODE LOCKED — Pull hands apart or push together to scale ${otherHeldShape.type.toUpperCase()}`;
          continue;
        } else {
          // Hovering second hand near the already-held shape shows cyan target reticle readiness
          drawPointerAndPinchFeedback(hand, true, false);
          continue;
        }
      }
    }

    // 2. Check if Index Tip (8) is hovering over "+ SHAPES" toggle button
    if (isPointInsideElement(hand.indexTip.x, hand.indexTip.y, shapesToggleBtn)) {
      toggleBtnHovered = true;
      if (hand.isPinching && nowInMs - lastMenuToggleTime >= TOGGLE_COOLDOWN_MS) {
        lastMenuToggleTime = nowInMs;
        setShapesMenuOpen(!isShapesMenuOpen);
        shapesToggleBtn.classList.add('is-active');
        setTimeout(() => shapesToggleBtn.classList.remove('is-active'), 300);
        statusBadge.textContent = isShapesMenuOpen
          ? 'Shapes Menu Open — Hover a shape & pinch Thumb (4) + Middle (12) to spawn'
          : 'Shapes Menu Closed';
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      }
      drawPointerAndPinchFeedback(hand, true, false);
      continue;
    }

    // 3. Check if Index Tip (8) is hovering over "☀ SUN" toggle button
    if (isPointInsideElement(hand.indexTip.x, hand.indexTip.y, sunToggleBtn)) {
      sunBtnHovered = true;
      if (hand.isPinching && nowInMs - lastSunToggleTime >= TOGGLE_COOLDOWN_MS) {
        lastSunToggleTime = nowInMs;
        setSunActive(!isSunActive);
        sunToggleBtn.classList.add('is-active');
        setTimeout(() => sunToggleBtn.classList.remove('is-active'), 300);
        statusBadge.textContent = isSunActive
          ? '☀ SUN Spawned — Pinch Sun with Index Tip (8) to drag, or cover with open hand for Eclipse!'
          : '☀ SUN Turned Off';
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      }
      drawPointerAndPinchFeedback(hand, true, false);
      continue;
    }

    // 4. Check if Index Tip (8) is hovering over top bar "Clear All" button
    if (isPointInsideElement(hand.indexTip.x, hand.indexTip.y, clearAllBtn)) {
      clearBtnHovered = true;
      if (hand.isPinching && nowInMs - lastClearTime >= SPAWN_COOLDOWN_MS) {
        lastClearTime = nowInMs;
        clearAllShapes();
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      }
      drawPointerAndPinchFeedback(hand, true, false);
      continue;
    }

    // 5. Check if Index Tip (8) is hovering over an expanded HUD spawner button
    const hoveredBtn = getHoveredHudButton(hand.indexTip.x, hand.indexTip.y);
    if (hoveredBtn) {
      _hoveredButtonsSet.add(hoveredBtn);

      const canSpawn = nowInMs - slot.lastSpawnTime >= SPAWN_COOLDOWN_MS;
      if (hand.isPinching && canSpawn) {
        const shapeType = hoveredBtn.dataset.shape;
        const spawnWorldPos = screenToWorldOnPlane(
          hand.indexTip.x,
          hand.indexTip.y,
          0,
          _scratchVecA
        );
        const newShape = spawnShape(shapeType, spawnWorldPos);

        slot.grabbedShape = newShape;
        slot.grabOffsetWorld.set(0, 0, 0);
        slot.lastSpawnTime = nowInMs;
        lastMenuToggleTime = nowInMs;
        _activeButtonsSet.add(hoveredBtn);

        setShapesMenuOpen(false);

        statusBadge.textContent = `Spawned ${shapeType.toUpperCase()} — Menu auto-closed for clear workspace`;
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      }

      drawPointerAndPinchFeedback(hand, true, false);
      continue;
    }

    // 6. Check if Index Tip (8) is hovering near any unheld 3D shape OR the active Sun
    let closestShape = null;
    let closestDist = Infinity;

    for (let s = 0; s < shapes.length; s++) {
      const shape = shapes[s];
      if (
        handSlots[0].grabbedShape === shape ||
        handSlots[1].grabbedShape === shape
      ) {
        continue;
      }
      const screenPos = worldToScreen(shape.group.position);
      const dist = Math.hypot(
        hand.indexTip.x - screenPos.x,
        hand.indexTip.y - screenPos.y
      );
      const scaleMult = Math.max(
        0.7,
        shape.group.scale.x / (shape.type === 'super-cube' ? 2.0 : 1.0)
      );
      const radius = (shape.hoverRadiusPx || HOVER_RADIUS_PX) * scaleMult;
      if (dist <= radius && dist < closestDist) {
        closestDist = dist;
        closestShape = shape;
      }
    }

    let distToSunPx = Infinity;
    if (isSunActive && sunGrabbedBySlotIndex === null) {
      distToSunPx = Math.hypot(
        hand.indexTip.x - sunScreenX,
        hand.indexTip.y - sunScreenY
      );
    }

    const canGrabSun =
      distToSunPx <= SUN_HOVER_RADIUS_PX && distToSunPx < closestDist;

    if (canGrabSun) {
      if (hand.isPinching) {
        sunGrabbedBySlotIndex = hand.slotIndex;
        const uvX = THREE.MathUtils.clamp(hand.indexTip.x / canvas.width, 0.03, 0.97);
        const uvY = THREE.MathUtils.clamp(1.0 - hand.indexTip.y / canvas.height, 0.03, 0.97);
        sunTargetUv.set(uvX, uvY);
        statusBadge.textContent = 'Dragging ☀ SUN — Release pinch to drop Sun in place';
        drawSunGrabIndicator(sunScreenX, sunScreenY, true, false);
        drawPointerAndPinchFeedback(hand, true, true);
      } else {
        drawSunGrabIndicator(sunScreenX, sunScreenY, false, false);
        drawPointerAndPinchFeedback(hand, true, false);
      }
    } else if (closestShape) {
      _hoveredShapesSet.add(closestShape);

      if (hand.isPinching) {
        slot.grabbedShape = closestShape;
        const pointerWorld = screenToWorldOnPlane(
          hand.indexTip.x,
          hand.indexTip.y,
          0,
          _scratchVecA
        );
        slot.grabOffsetWorld
          .copy(closestShape.group.position)
          .sub(pointerWorld);
        drawPointerAndPinchFeedback(hand, true, true);
      } else {
        drawPointerAndPinchFeedback(hand, true, false);
      }
    } else {
      drawPointerAndPinchFeedback(hand, false, false);
    }
  }

  shapesToggleBtn.classList.toggle('is-hovered', toggleBtnHovered);
  sunToggleBtn.classList.toggle('is-hovered', sunBtnHovered);
  clearAllBtn.classList.toggle('is-hovered', clearBtnHovered);
  setTrashZoneDangerState(anyObjectOverTrash);

  bgUniforms.uSunDanger.value = THREE.MathUtils.lerp(
    bgUniforms.uSunDanger.value,
    isSunOverTrash ? 1.0 : 0.0,
    0.25
  );

  // 7. Two-Hand Shape Combining Check:
  const shapeA = handSlots[0].grabbedShape;
  const shapeB = handSlots[1].grabbedShape;

  if (shapeA && shapeB && shapeA !== shapeB) {
    const posA = shapeA.group.position;
    const posB = shapeB.group.position;
    const worldDist = posA.distanceTo(posB);

    if (worldDist < MERGE_DISTANCE_WORLD * 1.9) {
      worldToScreen(posA, _screenOut);
      const ax = _screenOut.x;
      const ay = _screenOut.y;
      worldToScreen(posB, _screenOut);
      const bx = _screenOut.x;
      const by = _screenOut.y;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.strokeStyle = 'rgba(0, 243, 255, 0.85)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 6]);
      ctx.stroke();
      ctx.restore();
    }

    if (worldDist <= MERGE_DISTANCE_WORLD) {
      _scratchMidpoint.addVectors(posA, posB).multiplyScalar(0.5);
      const recipe = resolveCombinationRecipe(shapeA.type, shapeB.type);

      removeShape(shapeA);
      removeShape(shapeB);
      handSlots[0].grabbedShape = null;
      handSlots[1].grabbedShape = null;

      triggerParticleBurst(_scratchMidpoint, recipe.threeColor);
      spawnShape(recipe.type, _scratchMidpoint);

      playCombineChime();

      showCombineAnnouncement(recipe.label, recipe.colorHex);
      statusBadge.textContent = `${recipe.label} Grab it with Index (8) + Middle-Pinch (4 & 12)`;
    }
  }

  for (let i = 0; i < hudButtons.length; i++) {
    const btn = hudButtons[i];
    btn.classList.toggle('is-hovered', _hoveredButtonsSet.has(btn));
    if (_activeButtonsSet.has(btn)) {
      btn.classList.add('is-active');
      setTimeout(() => btn.classList.remove('is-active'), 350);
    }
  }

  if (handSlots[0].grabbedShape) _heldShapesSet.add(handSlots[0].grabbedShape);
  if (handSlots[1].grabbedShape) _heldShapesSet.add(handSlots[1].grabbedShape);

  for (let i = 0; i < shapes.length; i++) {
    const shape = shapes[i];
    if (resizeState.active && resizeState.shape === shape) {
      setShapeVisualState(shape, 'resizing', nowInMs);
    } else if (_dangerShapesSet.has(shape)) {
      setShapeVisualState(shape, 'danger', nowInMs);
    } else if (_heldShapesSet.has(shape)) {
      setShapeVisualState(shape, 'held', nowInMs);
    } else if (_hoveredShapesSet.has(shape)) {
      setShapeVisualState(shape, 'hover', nowInMs);
    } else {
      setShapeVisualState(shape, 'idle', nowInMs);
    }
  }
}

function drawSunGrabIndicator(sunX, sunY, isHolding, isDanger = false) {
  ctx.save();
  const ringColor = isDanger ? '#ff264e' : isHolding ? '#00ff66' : '#ffd700';
  ctx.strokeStyle = ringColor;
  ctx.lineWidth = isHolding ? 2.5 : 1.5;
  ctx.setLineDash([8, 6]);

  ctx.beginPath();
  ctx.arc(sunX, sunY, isDanger ? 34 : isHolding ? 46 : 54, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawPointerAndPinchFeedback(hand, isHoveringTarget, isHoldingObject) {
  const { indexTip, thumbTip, middleTip, isPinching } = hand;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(thumbTip.x, thumbTip.y);
  ctx.lineTo(middleTip.x, middleTip.y);
  ctx.strokeStyle = isPinching ? '#00ff66' : 'rgba(255, 0, 255, 0.45)';
  ctx.lineWidth = isPinching ? 3 : 1.5;
  if (!isPinching) {
    ctx.setLineDash([4, 4]);
  }
  ctx.stroke();
  ctx.restore();

  ctx.save();
  const ringColor = isHoldingObject
    ? '#00ff66'
    : isHoveringTarget
      ? '#00f3ff'
      : '#ffff00';

  ctx.strokeStyle = ringColor;
  ctx.lineWidth = isHoldingObject ? 2.5 : 1.8;

  ctx.beginPath();
  ctx.arc(indexTip.x, indexTip.y, isHoldingObject ? 20 : 16, 0, Math.PI * 2);
  ctx.stroke();

  const tickInner = isHoldingObject ? 23 : 19;
  const tickOuter = isHoldingObject ? 29 : 24;
  ctx.beginPath();
  ctx.moveTo(indexTip.x - tickOuter, indexTip.y);
  ctx.lineTo(indexTip.x - tickInner, indexTip.y);
  ctx.moveTo(indexTip.x + tickInner, indexTip.y);
  ctx.lineTo(indexTip.x + tickOuter, indexTip.y);
  ctx.moveTo(indexTip.x, indexTip.y - tickOuter);
  ctx.lineTo(indexTip.x, indexTip.y - tickInner);
  ctx.moveTo(indexTip.x, indexTip.y + tickInner);
  ctx.lineTo(indexTip.x, indexTip.y + tickOuter);
  ctx.stroke();
  ctx.restore();
}

/**
 * Maps normalized MediaPipe landmarks into a pre-allocated 21-point array (zero allocations).
 */
function mapLandmarksToScreen(landmarks, layout, canvasWidth, outPoints) {
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const rawX = layout.offsetX + lm.x * layout.drawWidth;
    outPoints[i].x = canvasWidth - rawX;
    outPoints[i].y = layout.offsetY + lm.y * layout.drawHeight;
  }
  return outPoints;
}

function drawHand(points) {
  // Outer glow pass + inner crisp skeleton line pass (avoids expensive per-frame CSS drop-shadow filters)
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  ctx.beginPath();
  for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
    const conn = HAND_CONNECTIONS[i];
    const p1 = points[conn[0]];
    const p2 = points[conn[1]];
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
  }
  ctx.strokeStyle = 'rgba(0, 243, 255, 0.28)';
  ctx.lineWidth = 8;
  ctx.stroke();

  ctx.strokeStyle = '#00f3ff';
  ctx.lineWidth = 2.75;
  ctx.stroke();
  ctx.restore();

  // Draw regular landmarks (all except 4, 8, 12) in cyan
  ctx.save();
  ctx.fillStyle = '#00f3ff';
  ctx.strokeStyle = '#e0ffff';
  ctx.lineWidth = 1.5;

  for (let i = 0; i < points.length; i++) {
    if (SPECIAL_LANDMARKS[i]) continue;
    const pt = points[i];
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();

  // Highlight Landmark 8 (Yellow) and Landmarks 4 & 12 (Magenta) on top
  const specialIndices = [4, 8, 12];
  for (let k = 0; k < specialIndices.length; k++) {
    const idx = specialIndices[k];
    const spec = SPECIAL_LANDMARKS[idx];
    const pt = points[idx];
    if (!pt) continue;

    ctx.save();
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, spec.radius + 4, 0, Math.PI * 2);
    ctx.fillStyle = idx === 8 ? 'rgba(255, 255, 0, 0.28)' : 'rgba(255, 0, 255, 0.28)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(pt.x, pt.y, spec.radius, 0, Math.PI * 2);
    ctx.fillStyle = spec.color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

async function init() {
  try {
    statusBadge.textContent = 'Initializing Shared MediaPipe Vision Resolver...';

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
    );

    statusBadge.textContent = 'Loading HandLandmarker & Selfie Segmenter...';

    const handLandmarkerPromise = HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    // Use lightweight float16 256x256 selfie_segmenter for lower GPU/VRAM footprint
    const imageSegmenterPromise = ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true
    });

    [handLandmarker, imageSegmenter] = await Promise.all([
      handLandmarkerPromise,
      imageSegmenterPromise
    ]);

    statusBadge.textContent = 'Requesting Webcam Access (640x480 @ 30fps)...';

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
        facingMode: 'user'
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    statusBadge.textContent =
      'HOLO-LAB READY — Pinch + SHAPES to spawn 3D shapes, or pinch ☀ SUN for Solar Eclipse & God Rays!';
    lastFrameTimeMs = performance.now();
    animFrameId = requestAnimationFrame(drawScene);
  } catch (err) {
    console.error('Initialization error:', err);
    statusBadge.textContent = `Error: ${err.message || 'Check camera permissions'}`;
    statusBadge.style.borderColor = 'rgba(255, 70, 70, 0.7)';
    statusBadge.style.color = '#ff6b6b';
  }
}

init();
