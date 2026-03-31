# Topology3D Implementation Guide

## Overview

The **Topology3D visualization system** renders five evolving 3D fields that capture different aspects of the ta-LBFGS optimizer's internal state. Each field is an interactive Three.js scene that updates in real-time from either SSE streams (live mode) or synthetic data generators (standalone simulation).

---

## System Architecture

### Core Components

#### 1. **topology3d.js** (ES Module)

Main visualization engine with Three.js integration.

```
├─ Initialization (init)
│  ├─ WebGL Renderer setup
│  ├─ Perspective Camera (50°, near=0.1, far=200)
│  ├─ OrbitControls (damping=0.08, autoRotateSpeed=0.8)
│  └─ 5 Scene objects (one per field)
│
├─ Field Builders (5 renderers)
│  ├─ buildLandscapeField() → Surface + Wireframe + Particles
│  ├─ buildAttentionField() → Attention mask planes
│  ├─ buildExpertField() → Expert nodes + co-activation edges
│  ├─ buildResidualField() → Jacobian coupling surface
│  └─ buildChainField() → Tube geometry for gradient trajectory
│
├─ Data Updaters (5 update functions)
│  ├─ updateLandscapeData(data)
│  ├─ updateAttentionData(data)
│  ├─ updateExpertData(data)
│  ├─ updateResidualData(data)
│  └─ updateChainData(data)
│
├─ Synthetic Generators (Standalone mode)
│  ├─ generateSyntheticLandscape(step)
│  ├─ generateSyntheticAttention(step)
│  ├─ generateSyntheticExpert(step)
│  ├─ generateSyntheticResidual(step)
│  └─ generateSyntheticChain(step)
│
├─ UI Controls
│  ├─ switchTopoField(field) → Tab switching + camera presets
│  ├─ toggleAutoRotate() → Auto-rotation toggle
│  ├─ resetTopoCamera() → Reset to field preset
│  ├─ updateLegend(field) → Field-specific legend
│  └─ updateInfoOverlay(field) → Field description
│
└─ Public API (window.Topo3D)
   ├─ update(data) → Real-time data from server
   └─ simulateStep(step) → Synthetic data for standalone mode
```

#### 2. **app.js** Integration Points

- `simulateStep()`: Calls `Topo3D.simulateStep(SIM.step)` after dashboard update
- `trySSEConnection()`: Hooks `data.topology_3d` → `Topo3D.update()`
- `resetSimulation()`: Calls `Topo3D.simulateStep(0)` to reset

#### 3. **index.html** Structure

```html
<!-- Script loading with Three.js importmap -->
<script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.168.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.168.0/examples/jsm/"
    }
  }
</script>
<script src="app.js"></script>
<script type="module" src="topology3d.js"></script>

<!-- 3D Canvas Container (section #topology-3d) -->
<div class="topo3d-container" id="topo3dContainer">
  <canvas id="topo3dCanvas"></canvas>
  <div class="topo3d-overlay">
    <div class="topo3d-info" id="topo3dInfo"></div>
  </div>
  <div class="topo3d-controls">
    <button onclick="toggleAutoRotate()">Rotate</button>
    <button onclick="resetTopoCamera()">Reset</button>
  </div>
</div>

<!-- Tab Navigation -->
<div class="topo3d-tabs">
  <button data-field="landscape" onclick="switchTopoField('landscape')">
    Landscape
  </button>
  <button data-field="attention" onclick="switchTopoField('attention')">
    Attention
  </button>
  <button data-field="expert" onclick="switchTopoField('expert')">
    Expert
  </button>
  <button data-field="residual" onclick="switchTopoField('residual')">
    Residual
  </button>
  <button data-field="chain" onclick="switchTopoField('chain')">Chain</button>
</div>
```

---

## Field Renderers (5 Types)

### Field 1: Landscape Curvature Surface

**Purpose**: Visualize condition number landscape and saddle detection across layers.

```
Geometry: PlaneGeometry (8×6, layers×window subdivisions)
Height: log₂(kappa) scaled by 0.3
Colors: Green (κ<10) → Amber (10<κ<25) → Red (κ>25)
Events: Red particles at saddle detection points
Legend: Convex Bowl | Narrow Ravine | Saddle Point | Wireframe
```

**Data Schema** from `data.landscape_field`:

```javascript
{
  kappa_grid: [[κ_00, κ_01, ...], [κ_10, ...], ...],     // μ×w grid
  status_grid: [[0, 1, 2, ...], ...],                      // 0=convex, 1=ravine, 2=saddle
  evasion_events: [{layer: 2, step: 15}, ...],             // Saddle evasion history
  memory_sizes: [5, 8, 12, 6],                             // Per-layer memory allocation
}
```

---

### Field 2: Attention Topology Volume

**Purpose**: Display attention head specialization by projecting Q/K/V masks into 3D space.

```
Geometry: PlaneGeometry per head (1.2×1.2, size×size subdivisions)
Height: mask_value * 0.5
Colors: Blue (local) | Purple (global) | Amber (causal) | Pink (sink)
Layout: Grid arrangement by layer×head, projections as Y offset
Legend: Local Head | Global Head | Causal Head | Sink Head
```

**Data Schema** from `data.attention_field`:

```javascript
{
  masks_summary: {
    "0:0:q": [[0.8, 0.2, ...], ...],      // Layer:Head:Projection → mask matrix
    "0:0:k": [[...], ...],
    "0:0:v": [[...], ...],
    ...
  },
  head_types: {
    "0:0": "local",                        // Head specialization
    "0:1": "global",
    ...
  },
  active_rederive: false,                  // Whether rederiving masks this step
}
```

---

### Field 3: Expert Routing Network

**Purpose**: Visualize mixture-of-experts load distribution and co-activation patterns.

```
Geometry: Spheres (radius ∝ load_freq) arranged in circle + line edges
Position: Circular layout (radius=3, angle = i/n * 2π)
Node Color: Indigo + emissive glow if active, gray if stale
Node Scale: 0.2 + 0.4*load_freq (pulsing animation if active)
Edges: Purple lines with opacity ∝ co-activation weight
Legend: Active Expert | Stale Expert | Co-activation Edge
```

**Data Schema** from `data.expert_field`:

```javascript
{
  nodes: [
    {id: 0, load_freq: 0.65, ttl: 2, active: true},      // Expert node state
    {id: 1, load_freq: 0.20, ttl: 4, active: false},
    ...
  ],
  edges: [
    {src: 0, dst: 2, weight: 0.45},                       // Co-activation edge
    ...
  ],
  buffer_sizes: [8, 2, 5, ...],                           // Per-expert buffer allocation
}
```

---

### Field 4: Residual Coupling Landscape

**Purpose**: Show Jacobian matrix coupling strength between layers and highlight strongly coupled zones.

```
Geometry: PlaneGeometry (6×6, layers×layers subdivisions)
Height: jacobian_coupling_norm * 3
Colors: Blue (decoupled) → gradient → Red (strongly coupled)
Markers: Pink spheres at (layer_i, layer_j) pairs with strong coupling
Wireframe: Teal overlay for structure
Legend: Decoupled | Strongly Coupled | Coupled Zone Marker
```

**Data Schema** from `data.residual_field`:

```javascript
{
  jacobian_matrix: [
    [1.0, 0.3, 0.1, 0.0],                 // nLayers × nLayers matrix
    [0.3, 1.0, 0.4, 0.1],
    [0.1, 0.4, 1.0, 0.2],
    [0.0, 0.1, 0.2, 1.0],
  ],
  coupled_zones: [[0,1], [1,2], [2,3]],   // Pairs with coupling > threshold
  hessian_strategies: ["coupled", "block_diag", "full", "low_rank"],  // Per-layer strategy
}
```

---

### Field 5: Reasoning Chain Timeline

**Purpose**: Visualize gradient trajectory through reasoning phases with pivot markers for important decision points.

```
Geometry: TubeGeometry along CatmullRomCurve3 (points from gradient norms)
X: Time index (0 → 8, normalized by trajectory length)
Y: Gradient norm * 4 (vertical height)
Z: Window scale * 3 - 1.5 (depth variation)
Colors: Blue (reasoning) | Green (answer) | Amber (verify) by segment
Markers: Icosahedron at pivot indices (gradient norm direction changes)
Legend: Reasoning | Answer | Verify | Pivot Event
```

**Data Schema** from `data.chain_field`:

```javascript
{
  grad_norms: [0.5, 0.48, 0.52, ..., 0.02],              // Gradient norm trajectory
  segments: ["reasoning", "reasoning", ..., "answer", ..., "verify", ...],  // Phase labels
  window_scales: [0.5, 0.5, ..., 1.0, ..., 0.7, ...],    // Context window size
  pivot_indices: [5, 12, 18],                            // Significant direction changes
  topology_valid: true,                                   // Validity flag
}
```

---

## Data Flow: Two Modes

### Mode 1: Standalone Simulation (Default)

```
│ Browser Loads
├─ index.html
│  ├─ Three.js importmap ready
│  ├─ app.js loads → DOMContentLoaded fired
│  └─ topology3d.js module loads → init() auto-runs
│
├─ User clicks "Start Simulation"
│  └─ startSimulation() interval fires 300ms
│     └─ simulateStep() executes (no server needed)
│        ├─ Generate loss, kappa, events (synthetic)
│        ├─ updateDashboardUI() updates charts
│        └─ Topo3D.simulateStep(SIM.step)
│           ├─ generateSyntheticLandscape(step)
│           ├─ generateSyntheticAttention(step)
│           ├─ generateSyntheticExpert(step)
│           ├─ generateSyntheticResidual(step)
│           ├─ generateSyntheticChain(step)
│           └─ updateLandscapeData() → Scene updates
│
└─ 3D visualization animates every frame in animate()
   └─ renderer.render(scenes[activeField], camera)
```

### Mode 2: Live SSE Connection (Connected to Server)

```
│ Python Server Running (port 7860)
│  └─ da_lbfgs.dashboard.server.DashboardServer
│     ├─ Serves /ui.html
│     ├─ Serves /events (SSE stream)
│     └─ Publishes state with topology_3d object
│
├─ Browser trySSEConnection()
│  └─ new EventSource('http://localhost:7860/events')
│     └─ OnMessage handler receives state
│        ├─ Parse JSON data
│        ├─ Update dashboard metrics (data.run)
│        └─ Topo3D.update(data.topology_3d)  ← Live field data
│           ├─ updateLandscapeData(data.landscape_field)
│           ├─ updateAttentionData(data.attention_field)
│           ├─ updateExpertData(data.expert_field)
│           ├─ updateResidualData(data.residual_field)
│           └─ updateChainData(data.chain_field)
│
└─ 3D visualization syncs with optimizer state
   └─ Landscapes, curvature, routing, residuals, chains update
```

---

## Camera Presets & Field-Specific Layouts

| Field     | Camera Pos | Target    | Description                                  |
| --------- | ---------- | --------- | -------------------------------------------- |
| landscape | [6, 5, 8]  | [0, 1, 0] | Elevated isometric view of curvature surface |
| attention | [5, 6, 5]  | [0, 2, 0] | Overhead view of head plane grid             |
| expert    | [7, 4, 7]  | [0, 2, 0] | Elevated circular expert layout              |
| residual  | [5, 5, 7]  | [0, 1, 0] | Isometric coupling surface                   |
| chain     | [8, 3, 4]  | [0, 2, 0] | Side-front view of gradient trajectory tube  |

---

## Color Scheme

```javascript
const C = {
  bg: 0x09090b, // Dark background
  grid: 0x222228, // Grid lines
  accent1: 0x6366f1, // Indigo (primary)
  accent2: 0xa855f7, // Purple (secondary)
  accent3: 0xec4899, // Pink (tertiary)
  green: 0x22c55e, // Green (positive)
  amber: 0xf59e0b, // Amber (warning)
  red: 0xef4444, // Red (alert)
  teal: 0x14b8a6, // Teal (info)
  blue: 0x3b82f6, // Blue (secondary)
  white: 0xfafafa, // Text
};
```

---

## Animation System

### Per-Frame Animations

All field objects animate in `animate()` loop:

```javascript
animateFieldObjects(activeField, t) {
  // t = performance.now() * 0.001 (seconds)

  if (field === 'landscape') {
    // Evasion particles: rise, shrink, fade
    evasionParticles.forEach(p => {
      p.material.opacity -= 0.005;
      p.position.y += 0.01;
      p.scale.multiplyScalar(0.995);
    });
  }

  if (field === 'expert') {
    // Active expert nodes: pulsing scale
    nodes.forEach((node, i) => {
      if (node.userData.active) {
        const pulse = 1.0 + 0.1 * Math.sin(t * 3 + i);
        node.scale.setScalar(node.userData.baseScale * pulse);
      }
    });
  }

  if (field === 'chain') {
    // Flow animation via shader uniform
    ribbon.material.uniforms.time.value = t;
  }
}
```

### Control Flow

```
requestAnimationFrame(animate)
├─ controls.update() (OrbitControls)
├─ animateFieldObjects(activeField, t)
├─ renderer.render(scenes[activeField], camera)
└─ Loop continues
```

---

## Usage Examples

### Starting Standalone Simulation

```html
<!-- Button in dashboard section -->
<button onclick="startSimulation()">Start Simulation</button>
```

```javascript
// Simulation loop (app.js)
function startSimulation() {
  SIM.running = true;
  SIM.interval = setInterval(() => {
    simulateStep(); // → Topo3D.simulateStep(SIM.step)
  }, 300);
}
```

### Switching Between Fields

```html
<!-- Tabs in topology-3d section -->
<button onclick="switchTopoField('landscape')">Landscape</button>
<button onclick="switchTopoField('attention')">Attention</button>
<!-- etc -->
```

```javascript
// Tab control (topology3d.js)
window.switchTopoField = function (field) {
  activeField = field;

  // Update UI tabs
  document.querySelectorAll(".topo3d-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.field === field);
  });

  // Reset camera to preset
  const preset = cameraPresets[field];
  camera.position.set(...preset.pos);
  controls.target.set(...preset.target);

  // Update legend & overlay
  updateLegend(field);
  updateInfoOverlay(field);
};
```

### Connecting to Live Server (Manual Testing)

```python
# Terminal 1: Start server
python -c "
from ta_lbfgs.dashboard.server import DashboardServer
server = DashboardServer()
server.start()
"

# Terminal 2: Run optimizer (publish data)
python optimize_model.py --dashboard
```

```javascript
// Browser automatically connects via trySSEConnection()
// Check browser console for connection status:
// "Connected" or "Standalone" indicator
```

---

## Debugging & Troubleshooting

### Issue: White/blank canvas

- **Check**: `topo3dContainer` has CSS width/height
- **Fix**: Ensure `resize()` function runs after init()

### Issue: Fields not responding to "Start Simulation"

- **Check**: `window.Topo3D` is defined (console: `typeof window.Topo3D`)
- **Fix**: Ensure `topology3d.js` loads without errors (dev console → Sources)

### Issue: 3D visualization jumpy/laggy

- **Reason**: Too many particles or overdraw
- **Fix**: Reduce evasion particle pool size or use object pooling

### Issue: SSE not connecting in live mode

- **Check**: Server running on `http://127.0.0.1:7860`
- **Fix**: Browser console should show "Connected" not "Standalone"
- **Alternative**: Works fine in standalone mode without server

---

## Performance Characteristics

| Field     | Polygons       | Vertices      | Textures | Overhead                |
| --------- | -------------- | ------------- | -------- | ----------------------- |
| Landscape | ~1k            | ~1k           | 0        | Low (vertex colors)     |
| Attention | ~100/head × 16 | ~64/head × 16 | 0        | Med (many planes)       |
| Expert    | 8 spheres      | 128 each      | 0        | Low (simple spheres)    |
| Residual  | ~256           | ~256          | 0        | Low (vertex colors)     |
| Chain     | ~400 points    | ~2k           | 0        | Med (tube subdivisions) |

**Total**: ~10-15k triangles per frame, 60 FPS on modern hardware.

---

## File References

- **Main Logic**: `/web-ui/topology3d.js`
- **Integration**: `/web-ui/app.js` (lines 369-376, 515-528, 498-500)
- **HTML**: `/web-ui/index.html` (lines 296-356)
- **Styles**: `/web-ui/styles.css` (`.topo3d-*` classes)
- **Server Data Source**: `/ta_lbfgs/dashboard/server.py` (lines 121-161)

---

## Future Extensions

1. **VR/AR Mode**: Export to VR headset via WebXR API
2. **Multi-dimensional fields**: Hyperparameter trajectory in ND space
3. **Neural Network Graph**: Layer connectivity visualization
4. **Performance Metrics**: Real-time FPS, memory, latency overlays
5. **Recording/Export**: Capture 3D trajectory as MP4 or WebGL buffer
