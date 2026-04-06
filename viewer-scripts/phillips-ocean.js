import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const GRID_RESOLUTION = 96;
const APP = document.querySelector("#app");

if (!APP) {
  throw new Error("Missing #app mount element.");
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x06111d);
scene.fog = new THREE.Fog(0x06111d, 14, 32);

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.1,
  100,
);
camera.position.set(6, 6, 9);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
APP.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);
controls.minDistance = 4;
controls.maxDistance = 20;
controls.maxPolarAngle = Math.PI * 0.48;

const ambientLight = new THREE.AmbientLight(0x8db8ff, 0.85);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(7, 12, 5);
scene.add(keyLight);

const planeGeometry = new THREE.PlaneGeometry(
  10,
  10,
  GRID_RESOLUTION - 1,
  GRID_RESOLUTION - 1,
);
planeGeometry.rotateX(-Math.PI / 2);

const positionAttribute = planeGeometry.getAttribute("position");
positionAttribute.setUsage(THREE.DynamicDrawUsage);

const positions = positionAttribute.array;
const targetHeights = new Float32Array(positionAttribute.count);

const planeMaterial = new THREE.MeshStandardMaterial({
  color: 0x2ec4ff,
  metalness: 0.05,
  roughness: 0.2,
  side: THREE.DoubleSide,
});

const water = new THREE.Mesh(planeGeometry, planeMaterial);
scene.add(water);

const wireframe = new THREE.Mesh(
  planeGeometry,
  new THREE.MeshBasicMaterial({
    color: 0xb7f0ff,
    wireframe: true,
    transparent: true,
    opacity: 0.22,
  }),
);
scene.add(wireframe);

/*
Copy incoming height data into target buffer
*/
function applyHeights(heights) {
  const usableCount = Math.min(targetHeights.length, heights.length);

  for (let i = 0; i < usableCount; i++) {
    targetHeights[i] = heights[i];
  }
}

const socket = new WebSocket("ws://localhost:8080/phillips-ocean");
socket.binaryType = "arraybuffer";

socket.addEventListener("message", (event) => {
  if (!(event.data instanceof ArrayBuffer)) {
    throw new Error("Wave socket payload must be an ArrayBuffer.");
  }

  if (event.data.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Wave socket payload has an invalid byte length.");
  }

  const heights = new Float32Array(event.data);
  applyHeights(heights);
});

socket.addEventListener("error", () => {
  console.error("Wave socket connection failed.");
});

function onResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

window.addEventListener("resize", onResize);

function animate() {
  const time = performance.now() / 1000;

  // Optimized loop: avoid i*3 multiplication
  for (let i = 0, j = 1; i < targetHeights.length; i++, j += 3) {
    positions[j] += (targetHeights[i] - positions[j]) * 0.16;
  }

  positionAttribute.needsUpdate = true;

  // Improve lighting correctness ( adds some CPU cost )
  planeGeometry.computeVertexNormals();

  water.rotation.y = time * 0.08;
  wireframe.rotation.y = water.rotation.y;

  controls.update();
  renderer.render(scene, camera);

  requestAnimationFrame(animate);
}

animate();