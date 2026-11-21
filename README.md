# STARK // HOLO-LAB MK.V — 3D Holographic Hand-Tracking Workspace & Solar Eclipse Shader

An Iron Man-inspired, webcam-powered 3D holographic workspace and real-time volumetric lighting lab built with **Three.js**, **MediaPipe Tasks Vision**, custom **GLSL raymarching shaders**, and the **Web Audio API**. Spawn, grab, stretch, and combine translucent 3D wireframes with bare-hand gestures in mid-air, or summon a turbulent solar light source that reacts to your body with real-time solar eclipses and crepuscular god rays.

---

## Features & Functions

- **Real-Time Two-Hand Skeleton Tracking (21 Landmarks per Hand):**
  - Powered by MediaPipe `HandLandmarker` (`numHands: 2`, `VIDEO` mode, GPU delegate) with persistent left/right hand slot matching so hands never swap identity mid-drag.
  - Renders a glowing cyan skeleton overlay with specialized fingertip highlights: **Landmark 8 (Index Tip)** in bright yellow (`#ffff00`) with a dynamic target reticle, and **Landmarks 4 (Thumb Tip)** & **12 (Middle Tip)** in glowing magenta (`#ff00ff`) with a live pinch-tether line.
- **Collapsible `+ SHAPES` Holographic Menu & `☀ SUN` Toggle:**
  - Compact top-left **`+ SHAPES`** toggle button that expands/collapses a slide-out holographic drawer containing **Cube**, **Sphere**, **Pyramid**, and **Ring** spawners (with a 500ms anti-flicker cooldown and automatic drawer close on spawn).
  - Dedicated **`☀ SUN`** toggle button next to `+ SHAPES` to spawn or dismiss the interactive solar light source, plus a top telemetry status bar (`Hands Detected: 0/2`, `Active Shapes: X`, `Sun: OFF/ACTIVE`, and a pinch-activated **`Clear All`** button).
- **Custom Gesture Controls:**
  - **Index Finger Tip (Landmark 8)** acts as a 2D/3D spatial cursor for hovering HUD buttons, 3D shapes, and the Sun.
  - **Thumb Tip (Landmark 4) + Middle Finger Tip (Landmark 12) Pinch** (with 55px engage / 75px release hysteresis) selects HUD buttons, spawns objects directly into your hand, and locks 3D shapes or the Sun to your finger for smooth dragging.
- **Two-Hand Shape Combining:**
  - Each hand can independently grab and hold its own 3D shape simultaneously.
  - Bringing two held shapes together within the collision threshold (`1.45` world units) displays a dashed cyan proximity energy beam, triggers a 3D additive particle burst + expanding shockwave ring, plays a 6-note Lydian harmonic chime, displays a 2-second holographic merge banner, and fuses the two shapes according to the recipe table below:

| Shape A | Shape B | Combined Result | Visual Description |
| :--- | :--- | :--- | :--- |
| **Cube** | **Pyramid** | **House** (`COMBINED: HOUSE!`) | Grouped 3D hologram with a 4-sided pyramid roof aligned atop a cube base in glowing cyan |
| **Sphere** | **Ring** | **Planet** (`COMBINED: PLANET!`) | Grouped 3D hologram featuring a sphere core surrounded by a tilted outer torus ring |
| **Cube** | **Cube** | **Large Super-Cube** (`COMBINED: LARGE SUPER-CUBE!`) | 2× scale holographic cube styled with a bright gold (`#ffd700`) wireframe and emissive core |
| *Any Other Shape* | *Any Other Shape* | **Mystery Crystal** (`COMBINED: MYSTERY CRYSTAL!`) | Faceted 3D `IcosahedronGeometry` with a pulsing magenta (`#ff00ff`) glow and breathing animation |

- **Two-Hand Stretching & Resizing (3D Shapes & Sun):**
  - While Hand A is pinching and holding a 3D shape (or the Sun), pinching with Hand B inside a generous `1.8×` bounding radius around that same object locks both hands onto it and enters **Resize Mode**.
  - Pulling your hands apart or pushing them together smoothly scales the object (`0.25×` to `4.0×`) via zero-jitter linear interpolation (`lerp`), centers it at the midpoint between both Index Tips, tints the wireframe gold (`#ffd700`), and draws a golden energy tether between your fingers. Releasing either pinch cleanly exits resize mode while preserving the new scale.
- **Interactive Sun Light Source, Solar Eclipse & Volumetric God Rays:**
  - Runs MediaPipe `ImageSegmenter` (`selfie_segmenter` float16 model) alongside `HandLandmarker` on a shared `FilesetResolver` to generate a real-time foreground confidence mask.
  - Uploads the mask to a `THREE.DataTexture` (`THREE.RedFormat`, `THREE.FloatType`) with **75% current / 25% previous** temporal smoothing and a **9-tap 3×3 Gaussian blur** + `smoothstep(0.35, 0.65)` in GLSL for crisp, anti-aliased silhouette edges.
  - Renders an animated turbulent Sun (3-octave 2D FBM photosphere + fiery corona) strictly **behind** the foreground mask (`1.0 - mask`) and performs a **24-step screen-space dithered 2D raymarch** with concentric ring occlusion sampling—casting golden crepuscular god-ray shadows, warm golden silhouette rim lighting, and plunging the room into darkness when your open hand eclipses the Sun.
- **Bottom-Left Holographic Trash Can Drop Zone:**
  - Circular 100px × 100px holographic drop zone in the bottom-left corner.
  - Dragging any held 3D shape or the Sun over the trash zone turns the ring crimson red (`#ff264e`), pulses it with a **`RELEASE TO DELETE`** warning, and shrinks/tints the held object red.
  - Releasing your pinch over the trash zone vaporizes 3D shapes with a red particle burst and synthesized Web Audio whoosh sound (or toggles the Sun cleanly back to `OFF` so it can be respawned anytime).
- **Performance & Memory Optimizations:**
  - **Webcam Capping:** Requests `640×480 @ 30 FPS` video constraints and caps WebGL renderer pixel ratios at `1.25` for high frame rates.
  - **Zero-Allocation Render Loop:** Pre-allocates a single reusable `Float32Array(256 * 256)` and `THREE.DataTexture` at startup, along with pooled scratch `THREE.Vector3` instances, landmark arrays, and reusable `Set`s.
  - **Strict WASM & GPU Memory Cleanup:** Immediately calls `.close()` on all MediaPipe `MPMask` and `ImageSegmenterResult` instances inside a `try / finally` block, and disposes Three.js geometries/materials whenever shapes or particle bursts are removed.
  - **Throttled Segmentation & Context Recovery:** Runs `ImageSegmenter` only while the Sun is toggled ON (throttled to every 3rd video frame), guards MediaPipe calls against duplicate timestamps, and handles `webglcontextlost` / `webglcontextrestored` events without freezing the browser tab.

---

## Controls & Gestures Quick Reference

| Gesture / Action | How to Perform | What It Does on Screen |
| :--- | :--- | :--- |
| **Point / Aim** | Move **Index Finger Tip (Landmark 8)** | Moves the on-screen target reticle and highlights hovered HUD buttons (`+ SHAPES`, `☀ SUN`, `Clear All`, shape drawers) or 3D objects |
| **Single-Hand Pinch (UI / Spawn)** | Pinch **Thumb Tip (4) + Middle Finger Tip (12)** while hovering a HUD button | Toggles the `+ SHAPES` drawer, spawns a 3D shape directly into your hand (auto-closing the menu), toggles the `☀ SUN`, or triggers `Clear All` |
| **Single-Hand Pinch & Drag** | Pinch **Thumb Tip (4) + Middle Finger Tip (12)** while hovering a 3D shape or the Sun | Locks the 3D shape (tinted bright green) or the Sun to your Index Tip so you can drag it across the screen; release pinch to drop |
| **Two-Hand Pinch (Same Object)** | Hold a 3D shape (or Sun) with Hand A, then pinch near that **same** object with Hand B | Enters **Resize Mode** (gold wireframe + golden tether line); pull hands apart or push together to stretch/scale between `0.25×` and `4.0×` |
| **Two-Hand Pinch (Different Objects)** | Hold Shape A with Left Hand and Shape B with Right Hand, then bring them together | Fuses the two shapes on collision with a particle shockwave and chime into a **House**, **Planet**, **Large Super-Cube**, or **Mystery Crystal** |
| **Drop in Trash** | Drag any held 3D shape or the Sun into the bottom-left **TRASH** circle and release pinch | Permanently deletes the 3D shape (or toggles the Sun `OFF`) with a crimson particle burst and synthesized delete whoosh |
| **Cover Sun with Open Palm** | Move your open hand or body in front of the active Sun (without pinching it) | Eclipses the solar disc behind your silhouette, dims ambient room lighting, and projects volumetric golden god rays and rim lighting around your fingers |

---

## Tech Stack

- **Build Tool & Dev Server:** [Vite](https://vitejs.dev/) (`^5.4.11`) — Vanilla ES Modules (`type: "module"`), HTML5, CSS3
- **3D Engine & WebGL Rendering:** [Three.js](https://threejs.org/) (`^0.170.0`) — Dual WebGL layers (orthographic full-screen background shader quad + perspective 3D holographic scene with additive wireframe/Phong materials and 3D particle bursts)
- **Computer Vision & ML:** [MediaPipe Tasks Vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) (`@mediapipe/tasks-vision ^0.10.18`) — GPU-accelerated `HandLandmarker` (2-hand 21-point tracking) and `ImageSegmenter` (`selfie_segmenter` confidence mask pipeline)
- **Custom GLSL Shaders:** 9-tap Gaussian mask smoothing, 3-octave 2D FBM solar turbulence, concentric ring eclipse occlusion, silhouette gradient rim lighting, and 24-step Interleaved Gradient Noise dithered 2D volumetric raymarching
- **Audio Synthesis:** Native browser **Web Audio API** — procedural sci-fi pinch blips, 6-note Lydian merge chimes, workspace clear sweeps, and filtered white-noise trash delete whooshes (zero external audio files required)

---

## How to Run Locally

### Prerequisites
- **Node.js** (v18+ recommended) and **npm** installed on your machine
- A working **webcam**
- A modern desktop browser with WebGL 2 and WebAssembly support (**Google Chrome** or **Microsoft Edge** recommended)

### Installation & Startup

1. **Clone the repository and navigate into the project folder:**
   ```bash
   git clone https://github.com/RJulian555/holo-hands.git
   cd holo-hands
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the Vite development server:**
   ```bash
   npm run dev
   ```

4. **Open in your browser:**
   - Visit the local URL printed in the terminal (typically `http://localhost:5173`).
   - **Important:** Click **"Allow"** when your browser prompts for **Camera / Webcam permissions** so MediaPipe can process the live video stream.
