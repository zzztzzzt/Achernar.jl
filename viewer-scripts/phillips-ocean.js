import * as THREE from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { MeshBasicNodeMaterial, MeshStandardNodeMaterial } from "three/webgpu";
import {
  cross,
  normalize,
  positionLocal,
  texture,
  uniform,
  uv,
  vec2,
  vec3,
  float,
  mix,
} from "three/tsl";

const GRID_RESOLUTION = 512;
const PLANE_SIZE = 20;
const HEIGHT_SCALE = 1.35;
const APP = document.querySelector("#app");

const ENVELOPE_HEADER_LEN = 17;
const ENVELOPE_VERSION_V1 = 1;
const CONTENT_TYPE_FLOAT32_TENSOR = 1;

if (!APP) throw new Error("Missing #app mount element.");
if (!navigator.gpu) throw new Error("WebGPU is required for the Phillips ocean viewer.");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x06111d);
scene.fog = new THREE.Fog(0x06111d, 24, 52);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(9, 8, 13);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setAnimationLoop(animate);
APP.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);
controls.minDistance = 4;
controls.maxDistance = 30;
controls.maxPolarAngle = Math.PI * 0.48;

scene.add(new THREE.AmbientLight(0x8db8ff, 0.85));

const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(7, 12, 5);
scene.add(keyLight);

const backLight = new THREE.DirectionalLight(0x7eb8ff, 0.9);
backLight.position.set(-10, 4, -8);
scene.add(backLight);

const planeGeometry = new THREE.PlaneGeometry(
  PLANE_SIZE,
  PLANE_SIZE,
  GRID_RESOLUTION - 1,
  GRID_RESOLUTION - 1
);
planeGeometry.rotateX(-Math.PI / 2);

const heightData = new Float32Array(GRID_RESOLUTION * GRID_RESOLUTION);
const heightTexture = new THREE.DataTexture(
  heightData,
  GRID_RESOLUTION,
  GRID_RESOLUTION,
  THREE.RedFormat,
  THREE.FloatType
);
heightTexture.wrapS = THREE.ClampToEdgeWrapping;
heightTexture.wrapT = THREE.ClampToEdgeWrapping;
heightTexture.magFilter = THREE.LinearFilter;
heightTexture.minFilter = THREE.LinearFilter;
heightTexture.needsUpdate = true;

const heightStep = PLANE_SIZE / (GRID_RESOLUTION - 1);
const texelStep = new THREE.Vector2(1 / GRID_RESOLUTION, 1 / GRID_RESOLUTION);

const heightScaleNode = uniform(HEIGHT_SCALE);
const texelStepNode = uniform(texelStep);
const heightTextureNode = texture(heightTexture);

function buildHeightNode(uvNode) {
  return heightTextureNode.sample(uvNode).r.mul(heightScaleNode);
}

function buildPositionNode() {
  const surfaceUv = uv();
  const height = buildHeightNode(surfaceUv);
  return positionLocal.add(vec3(0, height, 0));
}

function buildNormalNode() {
  const surfaceUv = uv();
  const stepX = vec2(texelStepNode.x, 0);
  const stepY = vec2(0, texelStepNode.y);

  const left = buildHeightNode(surfaceUv.sub(stepX));
  const right = buildHeightNode(surfaceUv.add(stepX));
  const down = buildHeightNode(surfaceUv.sub(stepY));
  const up = buildHeightNode(surfaceUv.add(stepY));

  const tangentX = vec3(float(heightStep * 2), right.sub(left), 0);
  const tangentZ = vec3(0, up.sub(down), float(heightStep * 2));

  return normalize(cross(tangentZ, tangentX));
}

const waterMaterial = new MeshStandardNodeMaterial({
  color: 0x2ec4ff,
  metalness: 0.08,
  roughness: 0.18,
  side: THREE.DoubleSide,
});
waterMaterial.positionNode = buildPositionNode();
waterMaterial.normalNode = buildNormalNode();
waterMaterial.colorNode = mix(
  vec3(0.039, 0.165, 0.263),
  vec3(0.49, 0.89, 1.0),
  heightTextureNode.sample(uv()).r.mul(0.5).add(0.5)
);

const water = new THREE.Mesh(planeGeometry, waterMaterial);
scene.add(water);

const wireframeMaterial = new MeshBasicNodeMaterial({
  color: 0xb7f0ff,
  wireframe: true,
  transparent: true,
  opacity: 0.18,
});
wireframeMaterial.positionNode = buildPositionNode();

const wireframe = new THREE.Mesh(planeGeometry, wireframeMaterial);
scene.add(wireframe);

function applyHeights(heights) {
  const usableCount = Math.min(heightData.length, heights.length);
  heightData.fill(0);
  heightData.set(heights.subarray(0, usableCount), 0);
  heightTexture.needsUpdate = true;
}

function parseEnvelopeV1(arrayBuffer) {
  if (arrayBuffer.byteLength < ENVELOPE_HEADER_LEN) {
    throw new Error("Frame too small for envelope v1.");
  }

  const view = new DataView(arrayBuffer);
  const version = view.getUint8(0);
  const contentType = view.getUint16(1, true);
  const flags = view.getUint16(3, true);
  const timestampNs = view.getBigUint64(5, true);
  const payloadLen = view.getUint32(13, true);

  if (version !== ENVELOPE_VERSION_V1) {
    throw new Error(`Unsupported envelope version: ${version}`);
  }

  const payloadOffset = ENVELOPE_HEADER_LEN;
  const payloadEnd = payloadOffset + payloadLen;

  if (arrayBuffer.byteLength !== payloadEnd) {
    throw new Error("Envelope payload length mismatch.");
  }

  // IMPORTANT : header is 17 bytes ( not 4-byte aligned ), so we copy payload
  const payloadBuffer = arrayBuffer.slice(payloadOffset, payloadEnd);

  return { contentType, flags, timestampNs, payloadBuffer };
}

// WIP : ws needs to be replaaced by wss in the future
const socket = new WebSocket("ws://localhost:8080/phillips-ocean");
socket.binaryType = "arraybuffer";

socket.addEventListener("open", () => {
  console.log("Wave socket connected.");
});

socket.addEventListener("message", (event) => {
  if (!(event.data instanceof ArrayBuffer)) {
    throw new Error("Wave socket payload must be an ArrayBuffer.");
  }

  const frame = parseEnvelopeV1(event.data);

  if (frame.contentType !== CONTENT_TYPE_FLOAT32_TENSOR) {
    // Ignore unsupported payload type for this viewer
    return;
  }

  if (frame.payloadBuffer.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Float32 payload has an invalid byte length.");
  }

  const heights = new Float32Array(frame.payloadBuffer);
  applyHeights(heights);
});

socket.addEventListener("error", () => {
  console.error("Wave socket connection failed.");
});

socket.addEventListener("close", () => {
  console.warn("Wave socket closed.");
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
  water.rotation.y = time * 0.08;
  wireframe.rotation.y = water.rotation.y;
  controls.update();
  renderer.render(scene, camera);
}