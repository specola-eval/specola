/* three.js pose panel for the specola dashboard.
 *
 * The scene arrives in two pieces. A "geometry" event carries the static
 * description of every body and geom once per connection; every step bundle
 * after it carries the current body poses as a packed float32 record. This
 * module owns the three.js side of that and nothing else: dashboard.js owns
 * the DOM and the socket and reaches this code through the
 * window.SpecolaScene3D facade registered at the bottom of the file.
 *
 * Frames are drawn on the rAF dirty-flag pattern dashboard.js uses for its
 * canvases: one render per pose update or camera move, never a free-running
 * loop.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

// MuJoCo scenes are z-up, three.js defaults to y-up. Setting the default
// before anything is constructed makes OrbitControls orbit the right axis.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// A MuJoCo plane with a zero half-size is infinite in that direction.
// Nothing renders an infinite plane, so it becomes a large finite one.
const INFINITE_PLANE_EXTENT = 20;
// Default-visible groups come from the geometry export: the simulator's
// own cameras decide (robosuite hides group 0, its collision hulls; the
// plain MuJoCo viewer shows 0-2, the fallback here). dashboard.js checks
// its group boxes against the same list.
const FALLBACK_VISIBLE_GROUPS = [0, 1, 2];

const panel = {
  container: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  bodies: [],           // THREE.Group per body, indexed by body id
  byGroup: new Map(),   // geom group -> [THREE.Mesh]
  disposable: [],       // geometries and materials owned by the built scene
  nbody: 0,
  framed: false,        // the camera has been fitted to the scene once
  renderPending: false,
};

/* ---- render scheduling -------------------------------------------------- */

function requestRender() {
  if (panel.renderPending || !panel.renderer) return;
  // Collapsing the pane hides it with display:none, which zeroes the
  // client size. Poses keep arriving; drawing them would be pure waste,
  // and the ResizeObserver re-renders as soon as the pane comes back.
  if (!panel.container.clientWidth) return;
  panel.renderPending = true;
  requestAnimationFrame(() => {
    panel.renderPending = false;
    panel.renderer.render(panel.scene, panel.camera);
  });
}

function resize() {
  const width = panel.container.clientWidth;
  const height = panel.container.clientHeight;
  if (!width || !height) return;            // pane collapsed or not laid out
  panel.renderer.setSize(width, height);
  panel.camera.aspect = width / height;
  panel.camera.updateProjectionMatrix();
  requestRender();
}

/* ---- scene construction ------------------------------------------------- */

function init(container) {
  if (panel.renderer) return;               // one WebGL context per page
  panel.container = container;
  panel.scene = new THREE.Scene();
  panel.scene.background = new THREE.Color(0x0a0e14);        // --bg
  panel.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 500);
  panel.camera.position.set(2.5, -2.5, 1.8);
  panel.renderer = new THREE.WebGLRenderer({ antialias: true });
  panel.renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(panel.renderer.domElement);

  const sky = new THREE.HemisphereLight(0xdfe7f5, 0x20242c, 2.2);
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(3, -4, 6);
  panel.scene.add(sky, sun);

  panel.controls = new OrbitControls(panel.camera, panel.renderer.domElement);
  panel.controls.target.set(0, 0, 0.6);
  panel.controls.addEventListener("change", requestRender);
  panel.controls.update();

  new ResizeObserver(resize).observe(container);
  resize();
}

// socket.io hands binary attachments to a browser client as ArrayBuffer; a
// typed-array view is accepted too so a polyfilled client still works.
function bufferOf(value) {
  if (value instanceof ArrayBuffer) return { buffer: value, offset: 0 };
  if (ArrayBuffer.isView(value)) {
    return { buffer: value.buffer, offset: value.byteOffset };
  }
  return null;
}

function meshGeometries(meta, source) {
  // Vertices are float32 triples from byte 0 of the buffer; faces are
  // uint32 index triples from meta.faces_start. Both offsets count
  // elements within their own region and face indices are local to their
  // mesh, so each region slices straight into a BufferGeometry.
  return (meta.meshes || []).map((entry) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(
      new Float32Array(source.buffer, source.offset + entry.vert_offset * 12,
                       entry.vert_count * 3), 3));
    if (entry.face_count) {
      geometry.setIndex(new THREE.BufferAttribute(
        new Uint32Array(source.buffer,
                        source.offset + meta.faces_start + entry.face_offset * 12,
                        entry.face_count * 3), 1));
    }
    geometry.computeVertexNormals();
    return geometry;
  });
}

function primitiveGeometry(geom) {
  // MuJoCo sizes are half-extents and radii; the three.js primitives here
  // all take full extents. Capsules and cylinders extend along local z in
  // MuJoCo and along y in three.js, so the quarter turn is baked into the
  // geometry rather than applied to every mesh that uses it.
  const size = geom.size;
  switch (geom.type) {
    case "plane": {
      const x = size[0] > 0 ? size[0] * 2 : INFINITE_PLANE_EXTENT;
      const y = size[1] > 0 ? size[1] * 2 : INFINITE_PLANE_EXTENT;
      return new THREE.PlaneGeometry(x, y);   // faces +z, as MuJoCo planes do
    }
    case "sphere":
      return new THREE.SphereGeometry(size[0], 24, 16);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 24, 16)
        .scale(size[0], size[1], size[2]);
    case "capsule":
      // CapsuleGeometry's height is the middle section only, which is
      // exactly MuJoCo's half-length doubled.
      return new THREE.CapsuleGeometry(size[0], size[1] * 2, 6, 20)
        .rotateX(Math.PI / 2);
    case "cylinder":
      return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 24)
        .rotateX(Math.PI / 2);
    case "box":
      return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    default:
      return null;                          // exporter emitted a type we skip
  }
}

function clearScene() {
  for (const body of panel.bodies) panel.scene.remove(body);
  for (const item of panel.disposable) item.dispose();
  panel.bodies = [];
  panel.byGroup = new Map();
  panel.disposable = [];
  panel.nbody = 0;
  panel.framed = false;
}

function setGeometry(meta, meshes) {
  if (!panel.renderer || !meta) return;
  const source = bufferOf(meshes);
  clearScene();
  panel.nbody = meta.nbody || 0;
  const visibleGroups =
    new Set(meta.visible_groups ?? FALLBACK_VISIBLE_GROUPS);
  // xpos/xquat are world poses, so body groups are siblings under the
  // scene: the MuJoCo body tree is already baked into them.
  for (let i = 0; i < panel.nbody; i += 1) {
    const body = new THREE.Group();
    panel.bodies.push(body);
    panel.scene.add(body);
  }
  const meshShapes = source ? meshGeometries(meta, source) : [];
  panel.disposable.push(...meshShapes);
  for (const geom of meta.geoms || []) {
    const body = panel.bodies[geom.body];
    const shape = geom.type === "mesh"
      ? meshShapes[geom.mesh]
      : primitiveGeometry(geom);
    // Primitive geometries are created per geom and must be disposed on
    // the next setGeometry (reconnects re-send the scene); mesh shapes
    // are already in the disposable list.
    if (shape && geom.type !== "mesh") panel.disposable.push(shape);
    if (!body || !shape) continue;
    const alpha = geom.rgba[3];
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]),
      roughness: 0.75,
      metalness: 0.05,
      transparent: alpha < 1,
      opacity: alpha,
      // A plane seen from below is a plane, not a hole.
      side: geom.type === "plane" ? THREE.DoubleSide : THREE.FrontSide,
    });
    panel.disposable.push(material);
    const mesh = new THREE.Mesh(shape, material);
    // The geom's pose in its body frame is constant for a rigid geom, so
    // it is set once here and only the body group moves per step.
    mesh.position.fromArray(geom.pos);
    mesh.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
    mesh.visible = visibleGroups.has(geom.group);
    body.add(mesh);
    const siblings = panel.byGroup.get(geom.group);
    if (siblings) siblings.push(mesh);
    else panel.byGroup.set(geom.group, [mesh]);
  }
  requestRender();
}

/* ---- poses -------------------------------------------------------------- */

function frameCamera() {
  // Every body sits at the origin until the first pose record lands, so
  // the scene has a meaningful extent only from here on. Fit once.
  panel.framed = true;
  // Fit the moving bodies, not the world body: MuJoCo's body 0 holds the
  // floor plane and room walls (LIBERO's are 3 m by 3 m), and a box
  // around them put the robot a few pixels wide in the first frame.
  // The world body is the fallback when nothing else carries a geom.
  let box = new THREE.Box3();
  for (let i = 1; i < panel.bodies.length; i += 1) {
    box.expandByObject(panel.bodies[i]);
  }
  if (box.isEmpty()) box = new THREE.Box3().setFromObject(panel.scene);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(
    0.5, box.getSize(new THREE.Vector3()).length() / 2);
  panel.controls.target.copy(center);
  // 1.4 radii back along the diagonal: at one radius the topmost link of
  // an upright arm left the frame.
  panel.camera.position.set(center.x + radius * 1.4, center.y - radius * 1.4,
                            center.z + radius);
  panel.camera.near = radius / 500;
  panel.camera.far = radius * 20;
  panel.camera.updateProjectionMatrix();
  panel.controls.update();
}

function setPoses(record) {
  const source = bufferOf(record);
  const nbody = panel.nbody;
  if (!source || !nbody) return;
  // nbody*3 float32 positions, then nbody*4 float32 quaternions.
  if (source.buffer.byteLength - source.offset < nbody * 28) return;
  const xpos = new Float32Array(source.buffer, source.offset, nbody * 3);
  const xquat = new Float32Array(
    source.buffer, source.offset + nbody * 12, nbody * 4);
  for (let i = 0; i < nbody; i += 1) {
    const body = panel.bodies[i];
    if (!body) continue;
    body.position.set(xpos[i * 3], xpos[i * 3 + 1], xpos[i * 3 + 2]);
    // MuJoCo stores quaternions (w, x, y, z); three.js takes (x, y, z, w).
    const q = i * 4;
    body.quaternion.set(xquat[q + 1], xquat[q + 2], xquat[q + 3], xquat[q]);
  }
  if (!panel.framed) frameCamera();
  requestRender();
}

function setGroupVisible(group, visible) {
  for (const mesh of panel.byGroup.get(group) || []) mesh.visible = visible;
  requestRender();
}

/* ---- facade -------------------------------------------------------------- */

window.SpecolaScene3D = { init, setGeometry, setPoses, setGroupVisible };
// dashboard.js is a classic script and therefore runs before this module:
// it queues its calls and flushes them on this event.
window.dispatchEvent(new Event("specola-scene3d-ready"));
