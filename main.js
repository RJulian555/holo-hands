import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const threeContainer = document.getElementById('three-container');
const statusBadge = document.getElementById('status-badge');
const combineBanner = document.getElementById('combine-banner');
const hudButtons = Array.from(document.querySelectorAll('.hud-btn'));
const handsCountEl = document.getElementById('hands-count');
const handsDotEl = document.getElementById('hands-dot');
const shapesCountEl = document.getElementById('shapes-count');
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

// Pinch, hover, spawn cooldown, and two-hand collision threshold
const PINCH_START_PX = 55;
const PINCH_RELEASE_PX = 75;
const HOVER_RADIUS_PX = 105;
const SPAWN_COOLDOWN_MS = 500;
const MERGE_DISTANCE_WORLD = 1.45;

let handLandmarker = null;
let lastVideoTime = -1;
let latestResults = null;
let combineBannerTimeout = null;
let lastClearTime = 0;

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

// Unlock Web Audio on any pointer/key interaction as well as automatic trigger
window.addEventListener('pointerdown', () => getAudioContext(), { passive: true });
window.addEventListener('keydown', () => getAudioContext(), { passive: true });

/**
 * Soft sci-fi holographic blip when a hand engages a pinch or grabs/spawns a shape.
 */
function playPinchBlip(pitchMultiplier = 1.0) {
  const ctxAudio = getAudioContext();
  if (!ctxAudio) return;

  const now = ctxAudio.currentTime;
  const osc = ctxAudio.createOscillator();
  const filter = ctxAudio.createBiquadFilter();
  const gain = ctxAudio.createGain();

  osc.type = 'sine';
  const startFreq = 720 * pitchMultiplier;
  const endFreq = 1280 * pitchMultiplier;
  osc.frequency.setValueAtTime(startFreq, now);
  osc.frequency.exponentialRampToValueAtTime(endFreq, now + 0.075);

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

/**
 * Futuristic harmonic chord chime when two shapes combine into a new creation.
 */
function playCombineChime() {
  const ctxAudio = getAudioContext();
  if (!ctxAudio) return;

  const now = ctxAudio.currentTime;
  // Sci-fi Lydian arpeggiated shimmer notes (Hz)
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

/**
 * Subtle downward holographic sweep when "Clear All" is triggered.
 */
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
// Three.js Scene Setup (Layered transparently over the 2D canvas)
// ============================================================================
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);
camera.position.set(0, 0, 6);

// Holographic studio lighting so translucent physical/phong surfaces shimmer like an Iron Man lab
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

// Collection of all active 3D shapes and active particle bursts
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

/**
 * Builds an Iron Man style holographic mesh component:
 * 1. Bright glowing outer wireframe edges (LineSegments)
 * 2. Additive outer bloom wireframe halo
 * 3. Translucent holographic inner volume with specular highlight + internal grid wireframe
 */
function buildHologramPart(geometry, baseColor, materialBucket) {
  const partGroup = new THREE.Group();
  const edgesGeo = new THREE.EdgesGeometry(geometry, 1);

  // 1. Bright glowing crisp wireframe edges
  const edgeMat = new THREE.LineBasicMaterial({
    color: baseColor,
    transparent: true,
    opacity: 1.0,
    blending: THREE.AdditiveBlending
  });
  const wireframeLines = new THREE.LineSegments(edgesGeo, edgeMat);
  partGroup.add(wireframeLines);

  // 2. Outer bloom halo wireframe
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

  // 3. Translucent holographic shell with fresnel-like specular sheen
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

/**
 * Spawns a base or combined 3D holographic shape at the specified world position.
 */
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
    // Subtle slow idle rotation speeds when floating untouched
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

/**
 * Clears all active shapes from the workspace with particle bursts and resets state.
 */
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
  statusBadge.textContent = 'Workspace Cleared — Pinch any left HUD button to spawn shapes';
}

// Also allow mouse/touch click on "Clear All" or HUD buttons as a convenience fallback
clearAllBtn.addEventListener('click', () => {
  clearAllShapes();
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

function resizeAll() {
  const width = window.innerWidth;
  const height = window.innerHeight;

  canvas.width = width;
  canvas.height = height;

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

  // Trigger soft sci-fi blip on fresh pinch transition
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

  // 1. Fill base dark canvas
  ctx.fillStyle = '#04070d';
  ctx.fillRect(0, 0, width, height);

  if (video.readyState >= 2 && video.videoWidth > 0) {
    const layout = getVideoCoverLayout(
      video.videoWidth,
      video.videoHeight,
      width,
      height
    );

    // Draw mirrored webcam feed in the background
    ctx.save();
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(
      video,
      layout.offsetX,
      layout.offsetY,
      layout.drawWidth,
      layout.drawHeight
    );
    ctx.restore();

    // Dim the webcam feed with a deep holographic blue-black tint
    ctx.fillStyle = 'rgba(4, 9, 18, 0.62)';
    ctx.fillRect(0, 0, width, height);

    // Run hand detection when a new video frame is available
    if (handLandmarker && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      latestResults = handLandmarker.detectForVideo(video, nowInMs);
    }

    const rawHands = [];

    if (latestResults && latestResults.landmarks) {
      for (let h = 0; h < latestResults.landmarks.length; h++) {
        const landmarks = latestResults.landmarks[h];
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
    updateHudAndShapesInteraction(trackedHands, nowInMs);
    updateTopStatusBar(trackedHands.length);
  }

  // Subtle slow idle rotation when shapes are floating untouched
  const heldShapes = new Set(
    handSlots.map((s) => s.grabbedShape).filter(Boolean)
  );

  for (const shape of shapes) {
    const isHeld = heldShapes.has(shape);
    const speedFactor = isHeld ? 0.35 : 1.0;

    shape.group.rotation.x += shape.rotSpeedX * speedFactor;
    shape.group.rotation.y += shape.rotSpeedY * speedFactor;

    if (!isHeld) {
      // Gentle zero-gravity bobbing while floating untouched
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

  renderer.render(scene, camera);
  requestAnimationFrame(drawScene);
}

function updateHudAndShapesInteraction(trackedHands, nowInMs) {
  const hoveredButtons = new Set();
  const activeButtons = new Set();
  const hoveredShapes = new Set();
  let clearBtnHovered = false;

  for (const hand of trackedHands) {
    const slot = hand.slot;

    // 1. If this hand is currently holding a shape, either keep dragging it or release on unpinch
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

    // 2. Check if Index Tip (8) is hovering over the top bar "Clear All" button
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

    // 3. Check if Index Tip (8) is hovering over a left HUD spawner button
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
        activeButtons.add(hoveredBtn);

        statusBadge.textContent = `Spawned ${shapeType.toUpperCase()} — Hold 2 shapes & bring together to combine!`;
        drawPointerAndPinchFeedback(hand, true, true);
        continue;
      }

      drawPointerAndPinchFeedback(hand, true, false);
      continue;
    }

    // 4. Otherwise, check if Index Tip (8) is hovering near any unheld 3D shape
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

    if (closestShape) {
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

  // Update "Clear All" button hover highlight
  clearAllBtn.classList.toggle('is-hovered', clearBtnHovered);

  // 5. Two-Hand Shape Combining Check:
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

      // Synthesized futuristic chime on combine
      playCombineChime();

      showCombineAnnouncement(recipe.label, recipe.colorHex);
      statusBadge.textContent = `${recipe.label} Grab it with Index (8) + Middle-Pinch (4 & 12)`;
    }
  }

  // Update HUD button highlight classes
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

function drawPointerAndPinchFeedback(hand, isHoveringTarget, isHoldingShape) {
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
  const ringColor = isHoldingShape
    ? '#00ff66'
    : isHoveringTarget
      ? '#00f3ff'
      : '#ffff00';

  ctx.strokeStyle = ringColor;
  ctx.lineWidth = isHoldingShape ? 2.5 : 1.8;
  ctx.shadowColor = ringColor;
  ctx.shadowBlur = 18;

  ctx.beginPath();
  ctx.arc(indexTip.x, indexTip.y, isHoldingShape ? 20 : 16, 0, Math.PI * 2);
  ctx.stroke();

  const tickInner = isHoldingShape ? 23 : 19;
  const tickOuter = isHoldingShape ? 29 : 24;
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
    statusBadge.textContent = 'Loading MediaPipe HandLandmarker...';

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
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

    statusBadge.textContent = 'Requesting webcam access...';

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
      'STARK HOLO-LAB READY — Point with Index (8) & Pinch Thumb (4) + Middle (12)';
    requestAnimationFrame(drawScene);
  } catch (err) {
    console.error('Initialization error:', err);
    statusBadge.textContent = `Error: ${err.message || 'Check camera permissions'}`;
    statusBadge.style.borderColor = 'rgba(255, 70, 70, 0.7)';
    statusBadge.style.color = '#ff6b6b';
  }
}

init();
