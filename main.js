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

// Colors for interactive 3D hologram shapes
const COLOR_CYAN = new THREE.Color(0x00f3ff);
const COLOR_GOLD = new THREE.Color(0xffd700);
const COLOR_MAGENTA = new THREE.Color(0xff00ff);
const COLOR_HELD_GREEN = new THREE.Color(0x00ff66);

// Pinch, hover, cooldowns, and two-hand collision threshold
const PINCH_START_PX = 55;
const PINCH_RELEASE_PX = 75;
const HOVER_RADIUS_PX = 105;
const SUN_HOVER_RADIUS_PX = 115;
const SPAWN_COOLDOWN_MS = 500;
const TOGGLE_COOLDOWN_MS = 500;
const MERGE_DISTANCE_WORLD = 1.45;

let handLandmarker = null;
let imageSegmenter = null;
let lastVideoTime = -1;
let latestHandResults = null;
let combineBannerTimeout = null;
let lastClearTime = 0;
let lastMenuToggleTime = 0;
let lastSunToggleTime = 0;
let isShapesMenuOpen = false;

// ============================================================================
// Optional Spawnable Sun State (Starts OFF by default)
// ============================================================================
let isSunActive = false;
let sunTransition = 0.0; // Smoothly lerps 0 -> 1 when Sun is spawned
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
    // Spawn Sun in the center of the screen
    sunTargetUv.set(0.5, 0.5);
    sunCurrentUv.set(0.5, 0.5);
  } else {
    sunGrabbedBySlotIndex = null;
    isMouseDraggingSun = false;
  }

  sunToggleBtn.classList.toggle('is-sun-active', isSunActive);
  sunToggleBtn.setAttribute('aria-pressed', String(isSunActive));
  sunDotEl.classList.toggle('is-sun-on', isSunActive);
  sunStatusTextEl.textContent = isSunActive ? 'Sun: ACTIVE' : 'Sun: OFF';
}

// ============================================================================
// Synthesized Web Audio API Sound Engine (No external MP3 files required)
// ============================================================================
let audioCtx = null;

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

// ============================================================================
// LAYER 1 (Background): Webcam + Optional Sun / Solar Eclipse / God-Ray Shader
// ============================================================================
let maskWidth = 256;
let maskHeight = 256;
let smoothedMaskBuffer = new Float32Array(maskWidth * maskHeight);

function createMaskDataTexture(width, height, dataArray) {
  const tex = new THREE.DataTexture(
    dataArray,
    width,
    height,
    THREE.RedFormat,
    THREE.FloatType
  );
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.flipY = true;
  tex.needsUpdate = true;
  return tex;
}

let maskDataTexture = createMaskDataTexture(
  maskWidth,
  maskHeight,
  smoothedMaskBuffer
);

const bgScene = new THREE.Scene();
const bgCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const bgRenderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: 'high-performance'
});
bgRenderer.setSize(window.innerWidth, window.innerHeight);
bgRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
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

  float sampleRayMask(vec2 screenUv) {
    vec2 uv = getMirroredCoverUv(screenUv);
    vec2 texel = 1.2 / max(uMaskResolution, vec2(256.0));
    float blurred =
      texture2D(uMaskTex, uv).r * 0.4 +
      (texture2D(uMaskTex, uv + vec2(texel.x, 0.0)).r +
       texture2D(uMaskTex, uv - vec2(texel.x, 0.0)).r +
       texture2D(uMaskTex, uv + vec2(0.0, texel.y)).r +
       texture2D(uMaskTex, uv - vec2(0.0, texel.y)).r) * 0.15;
    return smoothstep(0.35, 0.65, blurred);
  }

  float computeSunOcclusionFactor(vec2 aspectVec) {
    // When actively grabbing/dragging the Sun with a pinched finger, keep the Sun bright
    // so the user can clearly see where they are dragging it; when released, full eclipse applies!
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
    // When the user is NOT pinching/dragging the sun, the foreground mask completely hides the sun disc (1.0 - mask)
    float bgVisibility = mix(1.0 - mask, 1.0, uSunHeld * 0.85);

    float occlusionFactor = computeSunOcclusionFactor(aspectVec);
    float sunlightTransmission = 1.0 - smoothstep(0.05, 0.92, occlusionFactor);

    vec2 toSun = uLightPos - vUv;
    vec2 toSunAspect = toSun * aspectVec;
    float distToSun = length(toSunAspect);
    vec2 sunDir = normalize(toSunAspect + vec2(1e-5));

    // 64-Step Dithered 2D Raymarch for Volumetric God Rays & Shadows
    const int NUM_STEPS = 64;
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

      shadowTransmittance *= (1.0 - stepMask * 0.068);
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

    float sunCoreRadius = 0.038;
    float distortedDist = distToSun - (surfaceTurbulence - 0.5) * 0.008 - (coronaFlames - 0.5) * 0.012;

    float whiteHotCore = 1.0 - smoothstep(sunCoreRadius * 0.35, sunCoreRadius * 0.95, distortedDist);
    float goldenLimb = 1.0 - smoothstep(sunCoreRadius * 0.7, sunCoreRadius * 1.45, distortedDist);
    float innerCorona = exp(-max(distortedDist - sunCoreRadius * 0.5, 0.0) * 18.0);
    float outerCorona = exp(-distToSun * 4.2) * (0.75 + 0.45 * coronaFlames);

    vec3 colorWhiteHot   = vec3(1.00, 0.99, 0.94);
    vec3 colorSolarGold  = vec3(1.00, 0.78, 0.28);
    vec3 colorFieryAmber = vec3(0.98, 0.42, 0.08);

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
  uVideoResolution: { value: new THREE.Vector2(1280, 720) },
  uMaskResolution: { value: new THREE.Vector2(maskWidth, maskHeight) },
  uTime: { value: 0.0 },
  uSunActive: { value: 0.0 },
  uSunHeld: { value: 0.0 }
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

function ensureMaskTextureSize(w, h) {
  if (w === maskWidth && h === maskHeight) return;
  maskWidth = w;
  maskHeight = h;
  smoothedMaskBuffer = new Float32Array(w * h);

  maskDataTexture.dispose();
  maskDataTexture = createMaskDataTexture(w, h, smoothedMaskBuffer);
  bgUniforms.uMaskTex.value = maskDataTexture;
  bgUniforms.uMaskResolution.value.set(w, h);
}

function processSegmentationResult(result) {
  if (!result || !result.confidenceMasks || result.confidenceMasks.length === 0) {
    return;
  }

  const masks = result.confidenceMasks;
  const w = masks[0].width;
  const h = masks[0].height;
  const len = w * h;

  ensureMaskTextureSize(w, h);

  // Temporal smoothing: 75% current + 25% previous frame
  const ALPHA_CURRENT = 0.75;
  const ALPHA_PREV = 0.25;

  if (masks.length === 1) {
    const currentFloats = masks[0].getAsFloat32Array();
    for (let i = 0; i < len; i++) {
      smoothedMaskBuffer[i] =
        ALPHA_CURRENT * currentFloats[i] + ALPHA_PREV * smoothedMaskBuffer[i];
    }
    masks[0].close();
  } else {
    const bgFloats = masks[0].getAsFloat32Array();
    for (let i = 0; i < len; i++) {
      const fgConfidence = Math.max(0.0, Math.min(1.0, 1.0 - bgFloats[i]));
      smoothedMaskBuffer[i] =
        ALPHA_CURRENT * fgConfidence + ALPHA_PREV * smoothedMaskBuffer[i];
    }
    for (let m = 0; m < masks.length; m++) {
      masks[m].close();
    }
  }

  maskDataTexture.needsUpdate = true;
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
  antialias: true
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
threeContainer.appendChild(renderer.domElement);

const shapes = [];
const particleBursts = [];

// Persistent per-hand tracking state keyed by physical hand slot (0 and 1)
const handSlots = [
  {
    active: false,
    lastPos: null,
    isPinching: false,
    grabbedShape: null,
    grabOffsetWorld: new THREE.Vector3(),
    lastSpawnTime: 0
  },
  {
    active: false,
    lastPos: null,
    isPinching: false,
    grabbedShape: null,
    grabOffsetWorld: new THREE.Vector3(),
    lastSpawnTime: 0
  }
];

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
    opacity: 0.34,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  const glowMesh = new THREE.Mesh(geometry, glowMat);
  glowMesh.scale.setScalar(1.045);
  partGroup.add(glowMesh);

  const coreMat = new THREE.MeshPhongMaterial({
    color: baseColor,
    emissive: baseColor,
    emissiveIntensity: 0.45,
    specular: new THREE.Color(0xffffff),
    shininess: 90,
    transparent: true,
    opacity: 0.22,
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

function spawnShape(type, worldPosition = new THREE.Vector3(0, 0, 0)) {
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
      const geo = new THREE.SphereGeometry(0.75, 18, 14);
      group.add(buildHologramPart(geo, baseColor, materialBucket));
      break;
    }

    case 'pyramid': {
      const geo = new THREE.ConeGeometry(0.85, 1.35, 4);
      group.add(buildHologramPart(geo, baseColor, materialBucket));
      break;
    }

    case 'ring': {
      const geo = new THREE.TorusGeometry(0.62, 0.22, 14, 32);
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
      const sphereGeo = new THREE.SphereGeometry(0.68, 18, 14);
      const spherePart = buildHologramPart(sphereGeo, baseColor, materialBucket);
      group.add(spherePart);

      const ringGeo = new THREE.TorusGeometry(1.15, 0.11, 12, 40);
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
    baseColor: baseColor.clone(),
    hoverColor: baseColor.clone().lerp(new THREE.Color(0xffffff), 0.42),
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

function removeShape(targetShape) {
  const idx = shapes.indexOf(targetShape);
  if (idx !== -1) {
    shapes.splice(idx, 1);
  }

  scene.remove(targetShape.group);
  targetShape.group.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}

function clearAllShapes() {
  for (const slot of handSlots) {
    slot.grabbedShape = null;
  }

  for (let i = shapes.length - 1; i >= 0; i--) {
    const s = shapes[i];
    triggerParticleBurst(s.group.position.clone(), s.baseColor);
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
    spawnShape(shapeType, new THREE.Vector3((Math.random() - 0.5) * 1.5, (Math.random() - 0.5) * 1.0, 0));
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
  isMouseDraggingSun = false;
});

function updateTopStatusBar(handsCount) {
  handsCountEl.textContent = `Hands Detected: ${handsCount}/2`;
  handsDotEl.classList.toggle('is-online', handsCount > 0);
  shapesCountEl.textContent = `Active Shapes: ${shapes.length}`;
}

function resolveCombinationRecipe(typeA, typeB) {
  const pair = [typeA, typeB].sort().join('+');

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
  const count = 76;
  const positions = new Float32Array(count * 3);
  const velocities = [];

  for (let i = 0; i < count; i++) {
    positions[i * 3] = worldPosition.x;
    positions[i * 3 + 1] = worldPosition.y;
    positions[i * 3 + 2] = worldPosition.z;

    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const speed = 1.8 + Math.random() * 3.8;

    velocities.push(
      new THREE.Vector3(
        Math.sin(phi) * Math.cos(theta) * speed,
        Math.sin(phi) * Math.sin(theta) * speed,
        Math.cos(phi) * speed
      )
    );
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

  const ringGeo = new THREE.RingGeometry(0.15, 0.24, 32);
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
    shockRing,
    ringGeo,
    ringMat,
    createdAt: performance.now(),
    durationMs: 750
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

    for (let p = 0; p < burst.velocities.length; p++) {
      const vel = burst.velocities[p];
      arr[p * 3] += vel.x * dtSeconds;
      arr[p * 3 + 1] += vel.y * dtSeconds;
      arr[p * 3 + 2] += vel.z * dtSeconds;
      vel.multiplyScalar(0.94);
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
  let coreOpacity = 0.22;
  let glowOpacity = 0.34;
  let emissiveIntensity = 0.45;

  if (state === 'held') {
    targetColor = COLOR_HELD_GREEN;
    coreOpacity = 0.34;
    glowOpacity = 0.58;
    emissiveIntensity = 0.75;
  } else if (state === 'hover') {
    targetColor = shape.hoverColor;
    coreOpacity = 0.28;
    glowOpacity = 0.46;
    emissiveIntensity = 0.6;
  } else if (shape.isPulsingMagenta) {
    const pulse = 0.5 + 0.5 * Math.sin(nowInMs * 0.006);
    coreOpacity = 0.16 + pulse * 0.24;
    glowOpacity = 0.28 + pulse * 0.45;
    emissiveIntensity = 0.35 + pulse * 0.5;
  }

  for (const m of shape.edgeMats) {
    m.color.copy(targetColor);
  }
  for (const m of shape.glowMats) {
    m.color.copy(targetColor);
    m.opacity = glowOpacity;
  }
  for (const m of shape.coreMats) {
    m.color.copy(targetColor);
    if (m.emissive) {
      m.emissive.copy(targetColor);
      m.emissiveIntensity = emissiveIntensity;
    }
    m.opacity = coreOpacity;
  }
}

// Spawn the initial glowing cyan wireframe 3D Cube in the center of the screen
spawnShape('cube', new THREE.Vector3(0, 0, 0));

function screenToWorldOnPlane(screenX, screenY, targetZ = 0) {
  const ndcX = (screenX / canvas.width) * 2 - 1;
  const ndcY = -(screenY / canvas.height) * 2 + 1;

  const vec = new THREE.Vector3(ndcX, ndcY, 0.5);
  vec.unproject(camera);

  const dir = vec.sub(camera.position).normalize();
  const distance = (targetZ - camera.position.z) / dir.z;
  return camera.position.clone().add(dir.multiplyScalar(distance));
}

function worldToScreen(worldPos) {
  const projected = worldPos.clone().project(camera);
  return {
    x: (projected.x * 0.5 + 0.5) * canvas.width,
    y: (-projected.y * 0.5 + 0.5) * canvas.height
  };
}

function sunUvToScreen(uv) {
  return {
    x: uv.x * canvas.width,
    y: (1.0 - uv.y) * canvas.height
  };
}

function screenToSunUv(screenX, screenY) {
  return {
    x: THREE.MathUtils.clamp(screenX / canvas.width, 0.03, 0.97),
    y: THREE.MathUtils.clamp(1.0 - screenY / canvas.height, 0.03, 0.97)
  };
}

function resizeAll() {
  const width = window.innerWidth;
  const height = window.innerHeight;

  canvas.width = width;
  canvas.height = height;

  bgRenderer.setSize(width, height);
  bgUniforms.uResolution.value.set(width, height);

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

window.addEventListener('resize', resizeAll);
resizeAll();

function getVideoCoverLayout(videoWidth, videoHeight, canvasWidth, canvasHeight) {
  const videoAspect = videoWidth / videoHeight;
  const canvasAspect = canvasWidth / canvasHeight;

  let drawWidth, drawHeight, offsetX, offsetY;

  if (canvasAspect > videoAspect) {
    drawWidth = canvasWidth;
    drawHeight = canvasWidth / videoAspect;
    offsetX = 0;
    offsetY = (canvasHeight - drawHeight) / 2;
  } else {
    drawHeight = canvasHeight;
    drawWidth = canvasHeight * videoAspect;
    offsetX = (canvasWidth - drawWidth) / 2;
    offsetY = 0;
  }

  return { drawWidth, drawHeight, offsetX, offsetY };
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

function getHoveredHudButton(screenX, screenY) {
  if (!isShapesMenuOpen) return null;
  for (const btn of hudButtons) {
    if (isPointInsideElement(screenX, screenY, btn)) {
      return btn;
    }
  }
  return null;
}

function assignHandsToPersistentSlots(rawHands) {
  if (rawHands.length === 0) {
    for (const slot of handSlots) {
      slot.active = false;
      slot.isPinching = false;
      slot.grabbedShape = null;
      slot.lastPos = null;
    }
    sunGrabbedBySlotIndex = null;
    return [];
  }

  if (rawHands.length === 1) {
    const hand = rawHands[0];
    let chosenSlotIdx = 0;
    if (
      handSlots[1].active &&
      handSlots[1].lastPos &&
      (!handSlots[0].active || !handSlots[0].lastPos)
    ) {
      chosenSlotIdx = 1;
    } else if (handSlots[0].lastPos && handSlots[1].lastPos) {
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
    handSlots[otherSlotIdx].lastPos = null;
    if (sunGrabbedBySlotIndex === otherSlotIdx) {
      sunGrabbedBySlotIndex = null;
    }

    return [bindHandToSlot(hand, chosenSlotIdx)];
  }

  const h0 = rawHands[0];
  const h1 = rawHands[1];
  let swap = false;

  if (handSlots[0].lastPos && handSlots[1].lastPos) {
    const costDirect =
      Math.hypot(h0.wrist.x - handSlots[0].lastPos.x, h0.wrist.y - handSlots[0].lastPos.y) +
      Math.hypot(h1.wrist.x - handSlots[1].lastPos.x, h1.wrist.y - handSlots[1].lastPos.y);
    const costSwapped =
      Math.hypot(h0.wrist.x - handSlots[1].lastPos.x, h0.wrist.y - handSlots[1].lastPos.y) +
      Math.hypot(h1.wrist.x - handSlots[0].lastPos.x, h1.wrist.y - handSlots[0].lastPos.y);
    swap = costSwapped < costDirect;
  } else if (handSlots[1].active && handSlots[1].lastPos && !handSlots[0].active) {
    const d0 = Math.hypot(h0.wrist.x - handSlots[1].lastPos.x, h0.wrist.y - handSlots[1].lastPos.y);
    const d1 = Math.hypot(h1.wrist.x - handSlots[1].lastPos.x, h1.wrist.y - handSlots[1].lastPos.y);
    swap = d0 < d1;
  } else {
    swap = h0.wrist.x > h1.wrist.x;
  }

  return swap
    ? [bindHandToSlot(h1, 0), bindHandToSlot(h0, 1)]
    : [bindHandToSlot(h0, 0), bindHandToSlot(h1, 1)];
}

function bindHandToSlot(rawHand, slotIndex) {
  const slot = handSlots[slotIndex];
  slot.active = true;
  slot.lastPos = { x: rawHand.wrist.x, y: rawHand.wrist.y };

  const wasPinching = slot.isPinching;
  const isPinching = wasPinching
    ? rawHand.pinchDistPx < PINCH_RELEASE_PX
    : rawHand.pinchDistPx < PINCH_START_PX;

  if (!wasPinching && isPinching) {
    playPinchBlip(slotIndex === 0 ? 1.0 : 1.12);
  }

  slot.isPinching = isPinching;

  return {
    ...rawHand,
    slotIndex,
    slot,
    isPinching
  };
}

let lastFrameTimeMs = performance.now();

function drawScene() {
  const nowInMs = performance.now();
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

  if (video.readyState >= 2 && video.videoWidth > 0) {
    bgUniforms.uVideoResolution.value.set(video.videoWidth, video.videoHeight);

    const layout = getVideoCoverLayout(
      video.videoWidth,
      video.videoHeight,
      width,
      height
    );

    // Run BOTH MediaPipe tasks (HandLandmarker + ImageSegmenter) when a new video frame arrives
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      if (handLandmarker) {
        latestHandResults = handLandmarker.detectForVideo(video, nowInMs);
      }

      // Only run ImageSegmenter when Sun is active or transitioning to keep max FPS
      if (imageSegmenter && (isSunActive || sunTransition > 0.01)) {
        imageSegmenter.segmentForVideo(
          video,
          nowInMs,
          processSegmentationResult
        );
      }
    }

    const rawHands = [];

    if (latestHandResults && latestHandResults.landmarks) {
      for (let h = 0; h < latestHandResults.landmarks.length; h++) {
        const landmarks = latestHandResults.landmarks[h];
        const points = mapLandmarksToScreen(landmarks, layout, width);
        drawHand(points);

        const wrist = points[0];
        const thumbTip = points[4];
        const indexTip = points[8];
        const middleTip = points[12];
        const pinchDistPx = Math.hypot(
          thumbTip.x - middleTip.x,
          thumbTip.y - middleTip.y
        );

        rawHands.push({
          wrist,
          indexTip,
          thumbTip,
          middleTip,
          pinchDistPx
        });
      }
    }

    const trackedHands = assignHandsToPersistentSlots(rawHands);
    updateHudShapesAndSunInteraction(trackedHands, nowInMs);
    updateTopStatusBar(trackedHands.length);
  }

  // Render Layer 1: Background Webcam + Solar Eclipse / God-Ray Shader
  bgRenderer.render(bgScene, bgCamera);

  // Subtle slow idle rotation when 3D shapes are floating untouched
  const heldShapes = new Set(
    handSlots.map((s) => s.grabbedShape).filter(Boolean)
  );

  for (const shape of shapes) {
    const isHeld = heldShapes.has(shape);
    const speedFactor = isHeld ? 0.35 : 1.0;

    shape.group.rotation.x += shape.rotSpeedX * speedFactor;
    shape.group.rotation.y += shape.rotSpeedY * speedFactor;

    if (!isHeld) {
      shape.group.position.y +=
        Math.sin(nowInMs * 0.0022 + shape.floatPhase) * 0.0009;
    }

    if (shape.isPulsingMagenta) {
      const pulseScale = 1 + 0.06 * Math.sin(nowInMs * 0.006);
      shape.group.scale.setScalar(pulseScale);
    }
  }

  // Update active collision particle bursts
  updateParticleBursts(nowInMs, dtSeconds);

  // Render Layer 2: Foreground 3D Holographic Shapes
  renderer.render(scene, camera);

  requestAnimationFrame(drawScene);
}

function updateHudShapesAndSunInteraction(trackedHands, nowInMs) {
  const hoveredButtons = new Set();
  const activeButtons = new Set();
  const hoveredShapes = new Set();
  let clearBtnHovered = false;
  let toggleBtnHovered = false;
  let sunBtnHovered = false;
  let sunOrbHovered = false;

  const sunScreenPos = sunUvToScreen(sunCurrentUv);

  for (const hand of trackedHands) {
    const slot = hand.slot;

    // 1a. If this hand is currently dragging the Sun light source, update uLightPos or drop on unpinch
    if (sunGrabbedBySlotIndex === hand.slotIndex) {
      if (isSunActive && hand.isPinching) {
        const newUv = screenToSunUv(hand.indexTip.x, hand.indexTip.y);
        sunTargetUv.set(newUv.x, newUv.y);
        drawSunGrabIndicator(sunScreenPos, true);
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      } else {
        sunGrabbedBySlotIndex = null;
      }
    }

    // 1b. If this hand is currently holding a 3D shape, either keep dragging it or release on unpinch
    if (slot.grabbedShape) {
      if (!shapes.includes(slot.grabbedShape)) {
        slot.grabbedShape = null;
      } else if (hand.isPinching) {
        const pointerWorld = screenToWorldOnPlane(
          hand.indexTip.x,
          hand.indexTip.y,
          0
        );
        slot.grabbedShape.group.position
          .copy(pointerWorld)
          .add(slot.grabOffsetWorld);
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      } else {
        slot.grabbedShape = null;
      }
    }

    // 2. Check if Index Tip (8) is hovering over the top-left "+ SHAPES" toggle button
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

    // 3. Check if Index Tip (8) is hovering over the "☀ SUN" toggle button
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

    // 4. Check if Index Tip (8) is hovering over the top bar "Clear All" button
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
      hoveredButtons.add(hoveredBtn);

      const canSpawn = nowInMs - slot.lastSpawnTime >= SPAWN_COOLDOWN_MS;
      if (hand.isPinching && canSpawn) {
        const shapeType = hoveredBtn.dataset.shape;
        const spawnWorldPos = screenToWorldOnPlane(
          hand.indexTip.x,
          hand.indexTip.y,
          0
        );
        const newShape = spawnShape(shapeType, spawnWorldPos);

        slot.grabbedShape = newShape;
        slot.grabOffsetWorld.set(0, 0, 0);
        slot.lastSpawnTime = nowInMs;
        lastMenuToggleTime = nowInMs;
        activeButtons.add(hoveredBtn);

        // Automatically close the menu right after spawning so the workspace stays clear
        setShapesMenuOpen(false);

        statusBadge.textContent = `Spawned ${shapeType.toUpperCase()} — Menu auto-closed for clear workspace`;
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      }

      drawPointerAndPinchFeedback(hand, true, false);
      continue;
    }

    // 6. Check if Index Tip (8) is hovering near any unheld 3D shape OR the active Sun
    const heldShapes = new Set(
      handSlots.map((s) => s.grabbedShape).filter(Boolean)
    );

    let closestShape = null;
    let closestDist = Infinity;

    for (const shape of shapes) {
      if (heldShapes.has(shape)) continue;
      const screenPos = worldToScreen(shape.group.position);
      const dist = Math.hypot(
        hand.indexTip.x - screenPos.x,
        hand.indexTip.y - screenPos.y
      );
      const radius = shape.hoverRadiusPx || HOVER_RADIUS_PX;
      if (dist <= radius && dist < closestDist) {
        closestDist = dist;
        closestShape = shape;
      }
    }

    // Distance to the Sun (if active and not already grabbed by the other hand)
    let distToSunPx = Infinity;
    if (isSunActive && sunGrabbedBySlotIndex === null) {
      distToSunPx = Math.hypot(
        hand.indexTip.x - sunScreenPos.x,
        hand.indexTip.y - sunScreenPos.y
      );
    }

    const canGrabSun =
      distToSunPx <= SUN_HOVER_RADIUS_PX && distToSunPx < closestDist;

    if (canGrabSun) {
      sunOrbHovered = true;
      if (hand.isPinching) {
        sunGrabbedBySlotIndex = hand.slotIndex;
        const newUv = screenToSunUv(hand.indexTip.x, hand.indexTip.y);
        sunTargetUv.set(newUv.x, newUv.y);
        statusBadge.textContent = 'Dragging ☀ SUN — Release pinch to drop Sun in place';
        drawSunGrabIndicator(sunScreenPos, true);
        drawPointerAndPinchFeedback(hand, true, true);
      } else {
        drawSunGrabIndicator(sunScreenPos, false);
        drawPointerAndPinchFeedback(hand, true, false);
      }
    } else if (closestShape) {
      hoveredShapes.add(closestShape);

      if (hand.isPinching) {
        slot.grabbedShape = closestShape;
        const pointerWorld = screenToWorldOnPlane(
          hand.indexTip.x,
          hand.indexTip.y,
          0
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

  // Update button hover highlights
  shapesToggleBtn.classList.toggle('is-hovered', toggleBtnHovered);
  sunToggleBtn.classList.toggle('is-hovered', sunBtnHovered);
  clearAllBtn.classList.toggle('is-hovered', clearBtnHovered);

  // 7. Two-Hand Shape Combining Check:
  const shapeA = handSlots[0].grabbedShape;
  const shapeB = handSlots[1].grabbedShape;

  if (shapeA && shapeB && shapeA !== shapeB) {
    const posA = shapeA.group.position;
    const posB = shapeB.group.position;
    const worldDist = posA.distanceTo(posB);

    const screenA = worldToScreen(posA);
    const screenB = worldToScreen(posB);
    if (worldDist < MERGE_DISTANCE_WORLD * 1.9) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(screenA.x, screenA.y);
      ctx.lineTo(screenB.x, screenB.y);
      ctx.strokeStyle = 'rgba(0, 243, 255, 0.85)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 6]);
      ctx.shadowColor = '#00f3ff';
      ctx.shadowBlur = 18;
      ctx.stroke();
      ctx.restore();
    }

    if (worldDist <= MERGE_DISTANCE_WORLD) {
      const collisionPoint = new THREE.Vector3()
        .addVectors(posA, posB)
        .multiplyScalar(0.5);

      const recipe = resolveCombinationRecipe(shapeA.type, shapeB.type);

      removeShape(shapeA);
      removeShape(shapeB);
      handSlots[0].grabbedShape = null;
      handSlots[1].grabbedShape = null;

      triggerParticleBurst(collisionPoint, recipe.threeColor);
      spawnShape(recipe.type, collisionPoint);

      playCombineChime();

      showCombineAnnouncement(recipe.label, recipe.colorHex);
      statusBadge.textContent = `${recipe.label} Grab it with Index (8) + Middle-Pinch (4 & 12)`;
    }
  }

  // Update HUD drawer button highlight classes
  for (const btn of hudButtons) {
    btn.classList.toggle('is-hovered', hoveredButtons.has(btn));
    if (activeButtons.has(btn)) {
      btn.classList.add('is-active');
      setTimeout(() => btn.classList.remove('is-active'), 350);
    }
  }

  // Update visual colors for all 3D shapes
  const currentlyHeldShapes = new Set(
    handSlots.map((s) => s.grabbedShape).filter(Boolean)
  );

  for (const shape of shapes) {
    if (currentlyHeldShapes.has(shape)) {
      setShapeVisualState(shape, 'held', nowInMs);
    } else if (hoveredShapes.has(shape)) {
      setShapeVisualState(shape, 'hover', nowInMs);
    } else {
      setShapeVisualState(shape, 'idle', nowInMs);
    }
  }
}

/**
 * Draws a subtle holographic solar target ring when hovering or dragging the Sun with Index Tip (8).
 */
function drawSunGrabIndicator(sunScreen, isHolding) {
  ctx.save();
  const ringColor = isHolding ? '#00ff66' : '#ffd700';
  ctx.strokeStyle = ringColor;
  ctx.lineWidth = isHolding ? 2.5 : 1.5;
  ctx.setLineDash([8, 6]);
  ctx.shadowColor = ringColor;
  ctx.shadowBlur = 16;

  ctx.beginPath();
  ctx.arc(sunScreen.x, sunScreen.y, isHolding ? 46 : 54, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawPointerAndPinchFeedback(hand, isHoveringTarget, isHoldingObject) {
  const { indexTip, thumbTip, middleTip, isPinching } = hand;

  // Draw middle-pinch line between Landmark 4 and Landmark 12
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(thumbTip.x, thumbTip.y);
  ctx.lineTo(middleTip.x, middleTip.y);
  ctx.strokeStyle = isPinching ? '#00ff66' : 'rgba(255, 0, 255, 0.45)';
  ctx.lineWidth = isPinching ? 3 : 1.5;
  if (!isPinching) {
    ctx.setLineDash([4, 4]);
  }
  ctx.shadowColor = isPinching ? '#00ff66' : '#ff00ff';
  ctx.shadowBlur = isPinching ? 16 : 6;
  ctx.stroke();
  ctx.restore();

  // Draw Iron Man style target reticle around Landmark 8 (Index Tip)
  ctx.save();
  const ringColor = isHoldingObject
    ? '#00ff66'
    : isHoveringTarget
      ? '#00f3ff'
      : '#ffff00';

  ctx.strokeStyle = ringColor;
  ctx.lineWidth = isHoldingObject ? 2.5 : 1.8;
  ctx.shadowColor = ringColor;
  ctx.shadowBlur = 18;

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

function mapLandmarksToScreen(landmarks, layout, canvasWidth) {
  return landmarks.map((lm) => {
    const rawX = layout.offsetX + lm.x * layout.drawWidth;
    const mirroredX = canvasWidth - rawX;
    const y = layout.offsetY + lm.y * layout.drawHeight;
    return { x: mirroredX, y };
  });
}

function drawHand(points) {
  // Draw connecting skeleton lines in glowing cyan with bloom halo
  ctx.save();
  ctx.strokeStyle = '#00f3ff';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = '#00f3ff';
  ctx.shadowBlur = 20;

  ctx.beginPath();
  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = points[startIdx];
    const p2 = points[endIdx];
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
  }
  ctx.stroke();
  ctx.restore();

  // Draw regular landmarks (all except 4, 8, 12) in glowing cyan
  ctx.save();
  ctx.fillStyle = '#00f3ff';
  ctx.strokeStyle = '#e0ffff';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = '#00f3ff';
  ctx.shadowBlur = 16;

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
  for (const [idxStr, spec] of Object.entries(SPECIAL_LANDMARKS)) {
    const idx = Number(idxStr);
    const pt = points[idx];
    if (!pt) continue;

    ctx.save();
    ctx.fillStyle = spec.color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.shadowColor = spec.glow;
    ctx.shadowBlur = 24;

    ctx.beginPath();
    ctx.arc(pt.x, pt.y, spec.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

async function init() {
  try {
    statusBadge.textContent = 'Initializing Shared MediaPipe Vision Resolver...';

    // Single shared FilesetResolver for both HandLandmarker and ImageSegmenter
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

    const imageSegmenterPromise = ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true
    }).catch(() =>
      ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true
      })
    );

    [handLandmarker, imageSegmenter] = await Promise.all([
      handLandmarkerPromise,
      imageSegmenterPromise
    ]);

    statusBadge.textContent = 'Requesting Webcam Access...';

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    statusBadge.textContent =
      'HOLO-LAB READY — Pinch + SHAPES to spawn 3D shapes, or pinch ☀ SUN for Solar Eclipse & God Rays!';
    requestAnimationFrame(drawScene);
  } catch (err) {
    console.error('Initialization error:', err);
    statusBadge.textContent = `Error: ${err.message || 'Check camera permissions'}`;
    statusBadge.style.borderColor = 'rgba(255, 70, 70, 0.7)';
    statusBadge.style.color = '#ff6b6b';
  }
}

init();
