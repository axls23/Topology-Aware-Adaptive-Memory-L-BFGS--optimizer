/* ===================================================================
   ta-LBFGS — 3D Topology Field Visualizations
   Five evolving 3D fields rendered with Three.js
   =================================================================== */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── State ─────────────────────────────────────────────────────────
let renderer, camera, controls, animId;
const scenes = {};
let activeField = 'landscape';
let autoRotate = true;
const fieldObjects = {};
const FIELDS = ['landscape', 'attention', 'expert', 'residual', 'chain'];

// Color palette matching the UI theme
const C = {
  bg:       0x09090b,
  grid:     0x222228,
  accent1:  0x6366f1,  // indigo
  accent2:  0xa855f7,  // purple
  accent3:  0xec4899,  // pink
  green:    0x22c55e,
  amber:    0xf59e0b,
  red:      0xef4444,
  teal:     0x14b8a6,
  blue:     0x3b82f6,
  white:    0xfafafa,
};

// ── Initialization ────────────────────────────────────────────────
function init() {
  const canvas = document.getElementById('topo3dCanvas');
  if (!canvas) return;

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(C.bg, 1);

  camera = new THREE.PerspectiveCamera(50, 2, 0.1, 200);
  camera.position.set(6, 5, 8);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = autoRotate;
  controls.autoRotateSpeed = 0.8;
  controls.minDistance = 2;
  controls.maxDistance = 40;

  // Create all 5 scenes
  FIELDS.forEach(f => {
    scenes[f] = new THREE.Scene();
    fieldObjects[f] = {};
    addLights(scenes[f]);
    addGridHelper(scenes[f]);
  });

  // Build initial geometry for each field
  buildLandscapeField();
  buildAttentionField();
  buildExpertField();
  buildResidualField();
  buildChainField();

  resize();
  window.addEventListener('resize', resize);
  updateLegend(activeField);
  updateInfoOverlay(activeField);
  animate();
}

function addLights(scene) {
  const amb = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(amb);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 8, 5);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(C.accent2, 0.2);
  dir2.position.set(-5, 3, -5);
  scene.add(dir2);
}

function addGridHelper(scene) {
  const grid = new THREE.GridHelper(12, 24, C.grid, C.grid);
  grid.material.opacity = 0.3;
  grid.material.transparent = true;
  scene.add(grid);
}

function resize() {
  const container = document.getElementById('topo3dContainer');
  if (!container || !renderer) return;
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function animate() {
  animId = requestAnimationFrame(animate);
  controls.update();

  // Animate field-specific objects
  const t = performance.now() * 0.001;
  animateFieldObjects(activeField, t);

  renderer.render(scenes[activeField], camera);
}

function animateFieldObjects(field, t) {
  const objs = fieldObjects[field];
  if (!objs) return;

  if (field === 'landscape' && objs.evasionParticles) {
    objs.evasionParticles.forEach(p => {
      p.material.opacity = Math.max(0, p.material.opacity - 0.005);
      p.position.y += 0.01;
      p.scale.multiplyScalar(0.995);
    });
  }

  if (field === 'expert' && objs.nodes) {
    objs.nodes.forEach((node, i) => {
      if (node.userData.active) {
        const pulse = 1.0 + 0.1 * Math.sin(t * 3 + i);
        node.scale.setScalar(node.userData.baseScale * pulse);
      }
    });
  }

  if (field === 'chain' && objs.ribbon) {
    // Subtle flow animation
    const mat = objs.ribbon.material;
    if (mat.uniforms && mat.uniforms.time) {
      mat.uniforms.time.value = t;
    }
  }
}


// ═══════════════════════════════════════════════════════════════════
// FIELD 1: Landscape Curvature Surface
// ═══════════════════════════════════════════════════════════════════

function buildLandscapeField() {
  const scene = scenes.landscape;
  const nLayers = 4;
  const window_ = 30;

  // Surface geometry
  const geo = new THREE.PlaneGeometry(8, 6, nLayers - 1, window_ - 1);
  geo.rotateX(-Math.PI / 2);

  // Custom shader material for status-based coloring
  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    shininess: 40,
    transparent: true,
    opacity: 0.9,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0, 1, 0);
  scene.add(mesh);

  // Wireframe overlay
  const wireMat = new THREE.MeshBasicMaterial({
    color: C.accent1, wireframe: true, transparent: true, opacity: 0.15,
  });
  const wire = new THREE.Mesh(geo.clone(), wireMat);
  wire.position.copy(mesh.position);
  scene.add(wire);

  fieldObjects.landscape.surface = mesh;
  fieldObjects.landscape.wireframe = wire;
  fieldObjects.landscape.evasionParticles = [];
  fieldObjects.landscape.dims = { nLayers, window: window_ };

  // Initial synthetic data
  updateLandscapeData(generateSyntheticLandscape(0));
}

function updateLandscapeData(data) {
  const objs = fieldObjects.landscape;
  if (!objs.surface) return;

  const mesh = objs.surface;
  const geo = mesh.geometry;
  const pos = geo.attributes.position;
  const { nLayers, window: win } = objs.dims;

  const kappaGrid = data.kappa_grid || [];
  const statusGrid = data.status_grid || [];

  // Create color attribute if needed
  let colors = geo.attributes.color;
  if (!colors) {
    colors = new THREE.Float32BufferAttribute(new Float32Array(pos.count * 3), 3);
    geo.setAttribute('color', colors);
  }

  const statusColors = [
    new THREE.Color(C.green),   // 0 = Convex Bowl
    new THREE.Color(C.amber),   // 1 = Narrow Ravine
    new THREE.Color(C.red),     // 2 = Saddle Point
  ];

  for (let i = 0; i < pos.count; i++) {
    const xi = Math.floor(i / win) % nLayers;
    const yi = i % win;

    const kRow = kappaGrid[xi] || [];
    const sRow = statusGrid[xi] || [];
    const kappa = kRow[yi] !== undefined ? kRow[yi] : (1 + Math.random() * 10);
    const status = sRow[yi] !== undefined ? sRow[yi] : 0;

    // Displace Y by log(kappa)
    const height = Math.log2(Math.max(1, kappa)) * 0.3;
    pos.setY(i, height);

    // Color by status
    const col = statusColors[status] || statusColors[0];
    colors.setXYZ(i, col.r, col.g, col.b);
  }

  pos.needsUpdate = true;
  colors.needsUpdate = true;
  geo.computeVertexNormals();

  // Update wireframe
  if (objs.wireframe) {
    const wPos = objs.wireframe.geometry.attributes.position;
    for (let i = 0; i < pos.count && i < wPos.count; i++) {
      wPos.setY(i, pos.getY(i));
    }
    wPos.needsUpdate = true;
  }

  // Evasion event particles
  const scene = scenes.landscape;
  const events = data.evasion_events || [];
  events.forEach(evt => {
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.15, 8, 8),
      new THREE.MeshBasicMaterial({ color: C.red, transparent: true, opacity: 1.0 })
    );
    const x = -4 + (evt.layer / 3) * 8;
    sphere.position.set(x, 3, 0);
    scene.add(sphere);
    objs.evasionParticles.push(sphere);
    // Clean old particles
    if (objs.evasionParticles.length > 20) {
      const old = objs.evasionParticles.shift();
      scene.remove(old);
      old.geometry.dispose();
      old.material.dispose();
    }
  });
}


// ═══════════════════════════════════════════════════════════════════
// FIELD 2: Attention Topology Volume
// ═══════════════════════════════════════════════════════════════════

function buildAttentionField() {
  const scene = scenes.attention;
  const objs = fieldObjects.attention;
  objs.planes = [];
  objs.headLabels = [];

  updateAttentionData(generateSyntheticAttention(0));
}

function updateAttentionData(data) {
  const scene = scenes.attention;
  const objs = fieldObjects.attention;

  // Clear old planes
  objs.planes.forEach(p => { scene.remove(p); p.geometry.dispose(); p.material.dispose(); });
  objs.planes = [];

  const masks = data.masks_summary || {};
  const headTypes = data.head_types || {};

  const typeColors = {
    local:  new THREE.Color(C.blue),
    global: new THREE.Color(C.accent2),
    causal: new THREE.Color(C.amber),
    sink:   new THREE.Color(C.accent3),
  };

  const keys = Object.keys(masks).sort();
  const perHead = {};

  // Group by layer:head
  keys.forEach(k => {
    const [l, h, proj] = k.split(':');
    const key = `${l}:${h}`;
    if (!perHead[key]) perHead[key] = { l: +l, h: +h, projs: {} };
    perHead[key].projs[proj] = masks[k];
  });

  const headKeys = Object.keys(perHead).sort();
  const totalHeads = headKeys.length;

  headKeys.forEach((hk, idx) => {
    const { l, h, projs } = perHead[hk];
    const ht = headTypes[hk] || 'global';
    const color = typeColors[ht] || typeColors.global;

    // Pick 'q' projection mask as primary display
    const mask = projs.q || projs.k || projs.v || [];
    const size = mask.length || 8;

    const geo = new THREE.PlaneGeometry(1.2, 1.2, size - 1, size - 1);
    const pos = geo.attributes.position;

    // Create vertex colors
    const colors = new THREE.Float32BufferAttribute(new Float32Array(pos.count * 3), 3);
    geo.setAttribute('color', colors);

    for (let vi = 0; vi < pos.count; vi++) {
      const r = Math.floor(vi / size);
      const c = vi % size;
      const val = mask[r] ? (mask[r][c] || 0) : 0;

      // Displace Z by mask value
      pos.setZ(vi, val * 0.5);

      // Blend between dark and head-type color based on mask value
      const col = color.clone().multiplyScalar(0.3 + val * 0.7);
      colors.setXYZ(vi, col.r, col.g, col.b);
    }
    pos.needsUpdate = true;
    colors.needsUpdate = true;
    geo.computeVertexNormals();

    const mat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
      shininess: 20,
    });

    const plane = new THREE.Mesh(geo, mat);

    // Position: spread by layer (X) and head (Z), stack projections (Y)
    const cols = Math.ceil(Math.sqrt(totalHeads));
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    plane.position.set(
      (col - cols / 2) * 1.8,
      l * 0.4 + 1,
      (row - Math.ceil(totalHeads / cols) / 2) * 1.8
    );
    plane.rotation.x = -Math.PI / 4;

    scene.add(plane);
    objs.planes.push(plane);
  });
}


// ═══════════════════════════════════════════════════════════════════
// FIELD 3: Expert Routing Network
// ═══════════════════════════════════════════════════════════════════

function buildExpertField() {
  const scene = scenes.expert;
  const objs = fieldObjects.expert;
  objs.nodes = [];
  objs.edges = [];
  objs.edgeLines = null;

  updateExpertData(generateSyntheticExpert(0));
}

function updateExpertData(data) {
  const scene = scenes.expert;
  const objs = fieldObjects.expert;

  // Clear old
  objs.nodes.forEach(n => { scene.remove(n); n.geometry.dispose(); n.material.dispose(); });
  objs.nodes = [];
  if (objs.edgeLines) {
    scene.remove(objs.edgeLines);
    objs.edgeLines.geometry.dispose();
    objs.edgeLines.material.dispose();
    objs.edgeLines = null;
  }

  const nodes = data.nodes || [];
  const edges = data.edges || [];
  const nExperts = nodes.length || 8;

  // Layout: circular arrangement
  const positions = [];
  nodes.forEach((nd, i) => {
    const angle = (i / nExperts) * Math.PI * 2;
    const radius = 3;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = 1 + nd.load_freq * 2;

    const baseScale = 0.2 + nd.load_freq * 0.4;
    const geo = new THREE.SphereGeometry(baseScale, 16, 16);

    const hue = nd.active ? C.accent1 : 0x444444;
    const opacity = nd.active ? 0.9 : 0.3;
    const mat = new THREE.MeshPhongMaterial({
      color: hue,
      transparent: true,
      opacity,
      emissive: nd.active ? C.accent2 : 0x000000,
      emissiveIntensity: nd.active ? 0.3 : 0,
    });

    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(x, y, z);
    sphere.userData = { active: nd.active, baseScale: 1 };
    scene.add(sphere);
    objs.nodes.push(sphere);
    positions.push(new THREE.Vector3(x, y, z));
  });

  // Edges
  if (edges.length > 0) {
    const linePositions = [];
    const lineColors = [];
    const edgeColor = new THREE.Color(C.accent2);

    edges.forEach(e => {
      const src = positions[e.src];
      const dst = positions[e.dst];
      if (src && dst) {
        linePositions.push(src.x, src.y, src.z);
        linePositions.push(dst.x, dst.y, dst.z);
        const alpha = Math.min(1, e.weight * 2);
        lineColors.push(edgeColor.r * alpha, edgeColor.g * alpha, edgeColor.b * alpha);
        lineColors.push(edgeColor.r * alpha, edgeColor.g * alpha, edgeColor.b * alpha);
      }
    });

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    lineGeo.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
    const lineMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
    });
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lines);
    objs.edgeLines = lines;
  }
}


// ═══════════════════════════════════════════════════════════════════
// FIELD 4: Residual Coupling Landscape
// ═══════════════════════════════════════════════════════════════════

function buildResidualField() {
  const scene = scenes.residual;
  const objs = fieldObjects.residual;

  const nLayers = 4;
  const geo = new THREE.PlaneGeometry(6, 6, nLayers - 1, nLayers - 1);
  geo.rotateX(-Math.PI / 2);

  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    shininess: 60,
    transparent: true,
    opacity: 0.9,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0, 0.5, 0);
  scene.add(mesh);

  // Wireframe
  const wireMat = new THREE.MeshBasicMaterial({
    color: C.teal, wireframe: true, transparent: true, opacity: 0.2,
  });
  const wire = new THREE.Mesh(geo.clone(), wireMat);
  wire.position.copy(mesh.position);
  scene.add(wire);

  // Coupled zone highlight meshes
  objs.surface = mesh;
  objs.wireframe = wire;
  objs.coupledMarkers = [];
  objs.nLayers = nLayers;

  updateResidualData(generateSyntheticResidual(0));
}

function updateResidualData(data) {
  const objs = fieldObjects.residual;
  if (!objs.surface) return;

  const scene = scenes.residual;
  const mesh = objs.surface;
  const geo = mesh.geometry;
  const pos = geo.attributes.position;
  const nl = objs.nLayers;

  const jacMatrix = data.jacobian_matrix || [];
  const coupledZones = data.coupled_zones || [];

  let colors = geo.attributes.color;
  if (!colors) {
    colors = new THREE.Float32BufferAttribute(new Float32Array(pos.count * 3), 3);
    geo.setAttribute('color', colors);
  }

  const coldColor = new THREE.Color(C.blue);
  const hotColor = new THREE.Color(C.red);

  for (let i = 0; i < pos.count; i++) {
    const li = Math.floor(i / nl);
    const lj = i % nl;

    const jVal = (jacMatrix[li] && jacMatrix[li][lj] !== undefined)
      ? jacMatrix[li][lj] : 0.5;

    pos.setY(i, jVal * 3);

    const col = coldColor.clone().lerp(hotColor, jVal);
    colors.setXYZ(i, col.r, col.g, col.b);
  }

  pos.needsUpdate = true;
  colors.needsUpdate = true;
  geo.computeVertexNormals();

  // Update wireframe
  if (objs.wireframe) {
    const wPos = objs.wireframe.geometry.attributes.position;
    for (let i = 0; i < pos.count && i < wPos.count; i++) {
      wPos.setY(i, pos.getY(i));
    }
    wPos.needsUpdate = true;
  }

  // Coupled zone markers
  objs.coupledMarkers.forEach(m => { scene.remove(m); m.geometry.dispose(); m.material.dispose(); });
  objs.coupledMarkers = [];

  coupledZones.forEach(([li, lj]) => {
    const xStep = 6 / (nl - 1);
    const x = -3 + li * xStep;
    const z = -3 + lj * xStep;
    const h = (jacMatrix[li] && jacMatrix[li][lj]) ? jacMatrix[li][lj] * 3 : 1;

    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 8, 8),
      new THREE.MeshBasicMaterial({
        color: C.accent3,
        transparent: true,
        opacity: 0.8,
      })
    );
    marker.position.set(x, h + 0.5 + 0.2, z);
    scene.add(marker);
    objs.coupledMarkers.push(marker);
  });
}


// ═══════════════════════════════════════════════════════════════════
// FIELD 5: Reasoning Chain Timeline
// ═══════════════════════════════════════════════════════════════════

function buildChainField() {
  const scene = scenes.chain;
  const objs = fieldObjects.chain;
  objs.ribbon = null;
  objs.pivotMarkers = [];

  updateChainData(generateSyntheticChain(0));
}

function updateChainData(data) {
  const scene = scenes.chain;
  const objs = fieldObjects.chain;

  // Clean old
  if (objs.ribbon) {
    scene.remove(objs.ribbon);
    objs.ribbon.geometry.dispose();
    objs.ribbon.material.dispose();
  }
  objs.pivotMarkers.forEach(m => { scene.remove(m); m.geometry.dispose(); m.material.dispose(); });
  objs.pivotMarkers = [];

  const norms = data.grad_norms || [];
  const segments = data.segments || [];
  const scales = data.window_scales || [];
  const pivots = data.pivot_indices || [];

  if (norms.length < 2) return;

  // Build curve from (time, grad_norm, window_scale)
  const points = norms.map((gn, i) => {
    const x = (i / Math.max(norms.length - 1, 1)) * 8 - 4;
    const y = gn * 4;
    const z = (scales[i] || 0.5) * 3 - 1.5;
    return new THREE.Vector3(x, y, z);
  });

  const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
  const tubeGeo = new THREE.TubeGeometry(curve, points.length * 4, 0.08, 8, false);

  // Vertex colors based on segment
  const segColors = {
    reasoning: new THREE.Color(C.blue),
    answer:    new THREE.Color(C.green),
    verify:    new THREE.Color(C.amber),
  };

  const posAttr = tubeGeo.attributes.position;
  const colorAttr = new THREE.Float32BufferAttribute(new Float32Array(posAttr.count * 3), 3);

  for (let i = 0; i < posAttr.count; i++) {
    const t = i / posAttr.count;
    const segIdx = Math.min(Math.floor(t * norms.length), norms.length - 1);
    const seg = segments[segIdx] || 'reasoning';
    const col = segColors[seg] || segColors.reasoning;
    colorAttr.setXYZ(i, col.r, col.g, col.b);
  }
  tubeGeo.setAttribute('color', colorAttr);

  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true,
    shininess: 40,
    transparent: true,
    opacity: 0.9,
  });

  const ribbon = new THREE.Mesh(tubeGeo, mat);
  ribbon.position.set(0, 1, 0);
  scene.add(ribbon);
  objs.ribbon = ribbon;

  // Pivot markers
  pivots.forEach(pi => {
    if (pi >= 0 && pi < points.length) {
      const pt = points[pi];
      const marker = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.18, 1),
        new THREE.MeshBasicMaterial({
          color: C.accent3,
          transparent: true,
          opacity: 0.9,
        })
      );
      marker.position.set(pt.x, pt.y + 1, pt.z);
      scene.add(marker);
      objs.pivotMarkers.push(marker);
    }
  });
}


// ═══════════════════════════════════════════════════════════════════
// Synthetic Data Generators (Standalone Mode)
// ═══════════════════════════════════════════════════════════════════

function generateSyntheticLandscape(step) {
  const nLayers = 4, win = 30;
  const kappaGrid = [], statusGrid = [];
  const evasionEvents = [];

  for (let l = 0; l < nLayers; l++) {
    const kRow = [], sRow = [];
    for (let t = 0; t < win; t++) {
      const k = 2 + (l + 1) * 5 * (1 + 0.5 * Math.sin(step * 0.1 + t * 0.2 + l));
      kRow.push(k);
      if (k > 25) sRow.push(2);
      else if (k > 12) sRow.push(1);
      else sRow.push(0);
    }
    kappaGrid.push(kRow);
    statusGrid.push(sRow);
    if (Math.random() < 0.05) evasionEvents.push({ layer: l, step });
  }

  return {
    kappa_grid: kappaGrid,
    status_grid: statusGrid,
    evasion_events: evasionEvents,
    memory_sizes: [5, 8, 12, 6],
  };
}

function generateSyntheticAttention(step) {
  const masks = {}, types = {};
  const nLayers = 4, nHeads = 4;
  const headTypeNames = ['local', 'global', 'causal', 'sink'];

  for (let l = 0; l < nLayers; l++) {
    for (let h = 0; h < nHeads; h++) {
      for (const proj of ['q', 'k', 'v']) {
        const mask = [];
        for (let r = 0; r < 8; r++) {
          const row = [];
          for (let c = 0; c < 8; c++) {
            let val = 0.5 + 0.4 * Math.sin(r * 0.8 + c * 0.6 + l + h + step * 0.05);
            if (h === 0) val *= Math.max(0, 1 - Math.abs(r - c) * 0.3);
            row.push(Math.round(val * 1000) / 1000);
          }
          mask.push(row);
        }
        masks[`${l}:${h}:${proj}`] = mask;
      }
      types[`${l}:${h}`] = headTypeNames[h % 4];
    }
  }
  return { masks_summary: masks, head_types: types, active_rederive: step % 20 === 0 };
}

function generateSyntheticExpert(step) {
  const nodes = [], edges = [];
  const n = 8;
  for (let i = 0; i < n; i++) {
    const load = 0.3 + 0.5 * Math.abs(Math.sin(step * 0.1 + i * 0.8));
    nodes.push({ id: i, load_freq: load, ttl: Math.max(0, 5 - Math.floor(load * 8)), active: load > 0.4 });
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = Math.max(0, 0.3 * Math.cos(i * 0.7 + j * 0.5 + step * 0.08));
      if (w > 0.05) edges.push({ src: i, dst: j, weight: Math.round(w * 1000) / 1000 });
    }
  }
  return { nodes, edges, buffer_sizes: nodes.map(n => Math.round(n.load_freq * 12)) };
}

function generateSyntheticResidual(step) {
  const nl = 4;
  const jac = [];
  const coupled = [];
  const strategies = [];
  for (let i = 0; i < nl; i++) {
    const row = [];
    for (let j = 0; j < nl; j++) {
      if (i === j) row.push(1);
      else {
        const c = Math.max(0, 0.8 - Math.abs(i - j) * 0.25 + 0.1 * Math.sin(step * 0.1 + i + j));
        row.push(Math.round(c * 1000) / 1000);
        if (c > 0.5 && i < j) coupled.push([i, j]);
      }
    }
    jac.push(row);
    strategies.push(i < nl / 3 ? 'coupled' : 'block_diag');
  }
  return { jacobian_matrix: jac, coupled_zones: coupled, hessian_strategies: strategies };
}

function generateSyntheticChain(step) {
  const n = 20;
  const norms = [], segs = [], scales = [], pivots = [];
  for (let t = 0; t < n; t++) {
    const norm = 0.5 + 0.3 * Math.sin(t * 0.4 + step * 0.05) + 0.1 * Math.cos(t * 0.7);
    norms.push(Math.max(0.01, norm));
    if (t / n < 0.4) { segs.push('reasoning'); scales.push(0.5); }
    else if (t / n < 0.85) { segs.push('answer'); scales.push(1.0); }
    else { segs.push('verify'); scales.push(0.7); }
    if (t > 0 && Math.abs(norms[t] - norms[t - 1]) > 0.2) pivots.push(t);
  }
  return { grad_norms: norms, segments: segs, window_scales: scales, pivot_indices: pivots, topology_valid: step > 5 };
}


// ═══════════════════════════════════════════════════════════════════
// Tab Switching & Controls
// ═══════════════════════════════════════════════════════════════════

const cameraPresets = {
  landscape: { pos: [6, 5, 8], target: [0, 1, 0] },
  attention: { pos: [5, 6, 5], target: [0, 2, 0] },
  expert:    { pos: [7, 4, 7], target: [0, 2, 0] },
  residual:  { pos: [5, 5, 7], target: [0, 1, 0] },
  chain:     { pos: [8, 3, 4], target: [0, 2, 0] },
};

window.switchTopoField = function(field) {
  if (!FIELDS.includes(field)) return;
  activeField = field;

  // Update tab UI
  document.querySelectorAll('.topo3d-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.field === field);
  });

  // Reset camera to preset
  const preset = cameraPresets[field];
  if (preset) {
    camera.position.set(...preset.pos);
    controls.target.set(...preset.target);
    controls.update();
  }

  updateLegend(field);
  updateInfoOverlay(field);
};

window.toggleAutoRotate = function() {
  autoRotate = !autoRotate;
  controls.autoRotate = autoRotate;
  document.getElementById('btnAutoRotate')?.classList.toggle('active', autoRotate);
};

window.resetTopoCamera = function() {
  const preset = cameraPresets[activeField];
  if (preset) {
    camera.position.set(...preset.pos);
    controls.target.set(...preset.target);
    controls.update();
  }
};

function updateLegend(field) {
  const el = document.getElementById('topo3dLegend');
  if (!el) return;

  const legends = {
    landscape: [
      { color: '#22c55e', label: 'Convex Bowl' },
      { color: '#f59e0b', label: 'Narrow Ravine' },
      { color: '#ef4444', label: 'Saddle Point' },
      { color: '#6366f1', label: 'Wireframe' },
    ],
    attention: [
      { color: '#3b82f6', label: 'Local Head' },
      { color: '#a855f7', label: 'Global Head' },
      { color: '#f59e0b', label: 'Causal Head' },
      { color: '#ec4899', label: 'Sink Head' },
    ],
    expert: [
      { color: '#6366f1', label: 'Active Expert' },
      { color: '#444444', label: 'Stale Expert' },
      { color: '#a855f7', label: 'Co-activation Edge' },
    ],
    residual: [
      { color: '#3b82f6', label: 'Decoupled' },
      { color: '#ef4444', label: 'Strongly Coupled' },
      { color: '#ec4899', label: 'Coupled Zone Marker' },
    ],
    chain: [
      { color: '#3b82f6', label: 'Reasoning' },
      { color: '#22c55e', label: 'Answer' },
      { color: '#f59e0b', label: 'Verify' },
      { color: '#ec4899', label: 'Pivot Event' },
    ],
  };

  const items = legends[field] || [];
  el.innerHTML = items.map(item =>
    `<div class="topo3d-legend-item">
       <span class="topo3d-legend-dot" style="background:${item.color}"></span>
       ${item.label}
     </div>`
  ).join('');
}

const fieldDescriptions = {
  landscape: 'X=Layer, Y=Iteration, Z=log(kappa) | Color=landscape status',
  attention: 'Head masks displaced by intensity | Color=head type (local/global/causal/sink)',
  expert:    'Node size=load freq, position=circular layout | Edges=co-activation',
  residual:  'Height=Jacobian coupling norm | Color: blue(decoupled) to red(coupled)',
  chain:     'X=Time, Y=grad norm, Z=window scale | Color=segment type',
};

function updateInfoOverlay(field) {
  const el = document.getElementById('topo3dInfo');
  if (el) el.textContent = fieldDescriptions[field] || '';
}


// ═══════════════════════════════════════════════════════════════════
// Public API: Called from app.js when SSE data arrives
// ═══════════════════════════════════════════════════════════════════

window.Topo3D = {
  update(data) {
    if (!data) return;
    if (data.landscape_field) updateLandscapeData(data.landscape_field);
    if (data.attention_field) updateAttentionData(data.attention_field);
    if (data.expert_field) updateExpertData(data.expert_field);
    if (data.residual_field) updateResidualData(data.residual_field);
    if (data.chain_field) updateChainData(data.chain_field);
  },

  simulateStep(step) {
    updateLandscapeData(generateSyntheticLandscape(step));
    updateAttentionData(generateSyntheticAttention(step));
    updateExpertData(generateSyntheticExpert(step));
    updateResidualData(generateSyntheticResidual(step));
    updateChainData(generateSyntheticChain(step));
  },
};


// ── Bootstrap ─────────────────────────────────────────────────────
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
