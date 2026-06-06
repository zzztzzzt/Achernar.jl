import * as THREE from "three";
import WebGPURenderer from "three/src/renderers/webgpu/WebGPURenderer.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Timer } from 'three/src/core/Timer.js';

import DirectionalLightNode from "three/src/nodes/lighting/DirectionalLightNode.js";

const APP = document.querySelector("#app");

if (!APP) {
  throw new Error("Missing #app mount element.");
}

const timer = new Timer();
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x161a20);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 28);

const renderer = new WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

renderer.library.addLight( DirectionalLightNode, THREE.DirectionalLight );

APP.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const light = new THREE.DirectionalLight(0xffffff, 2.4);
light.position.set(-8, -10, -12);
scene.add(light);

const material = new THREE.MeshPhongMaterial({
  color: 0xd9ffd9,
  specular: 0xffffff,
  shininess: 320,
  emissive: 0x0a0d12,
  flatShading: false,
  side: THREE.DoubleSide,
});

const blobGeometry = new THREE.BufferGeometry();
const blobMesh = new THREE.Mesh(blobGeometry, material);
blobMesh.scale.set(14, 14, 14);
scene.add(blobMesh);

let pendingBuffer = null;

const ENVELOPE_HEADER_LEN = 17;
const ENVELOPE_VERSION_V1 = 1;
const CONTENT_TYPE_FLOAT32_TENSOR = 1;

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

  // Copy payload to new buffer for 4-byte alignment
  const payloadBuffer = arrayBuffer.slice(payloadOffset, payloadEnd);

  return { contentType, flags, timestampNs, payloadBuffer };
}

function updateGeometry(buffer) {
  const frame = parseEnvelopeV1(buffer);
  
  if (frame.contentType !== CONTENT_TYPE_FLOAT32_TENSOR) {
    return;
  }

  const data = new Float32Array(frame.payloadBuffer);
  if (data.length === 0) return;

  const vertexFloatCount = Math.round(data[0]);
  const vertices = data.subarray(1, 1 + vertexFloatCount);
  const normals = data.subarray(1 + vertexFloatCount);

  blobGeometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  blobGeometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));

  blobGeometry.attributes.position.needsUpdate = true;
  blobGeometry.attributes.normal.needsUpdate = true;

  blobGeometry.computeBoundingSphere();
}

const socket = new WebSocket("ws://localhost:8080/metaballs");
socket.binaryType = "arraybuffer";
socket.addEventListener("message", (event) => {
  pendingBuffer = event.data;
});

function animate() {
  timer.update();
  
  if (pendingBuffer) {
    updateGeometry(pendingBuffer);
    pendingBuffer = null;
  }

  controls.update();
  renderer.render(scene, camera);
}

renderer.setAnimationLoop(animate);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});