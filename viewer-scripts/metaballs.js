import * as THREE from "three";
import WebGPURenderer from "three/src/renderers/webgpu/WebGPURenderer.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Timer } from 'three/src/core/Timer.js';

import DirectionalLightNode from "three/src/nodes/lighting/DirectionalLightNode.js";
import {
  CONTENT_TYPE_FLOAT32_TENSOR,
  normalizeSocketPayload,
  parseEnvelopeOrLegacyFloat32,
} from "./frame-parser.js";

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

function updateGeometry(buffer) {
  const frame = parseEnvelopeOrLegacyFloat32(buffer, (data) => {
    if (data.length === 0) {
      throw new Error("Legacy metaballs payload is empty.");
    }

    const vertexFloatCount = Math.round(data[0]);
    if (!Number.isFinite(vertexFloatCount) || vertexFloatCount < 0) {
      throw new Error("Legacy metaballs payload has an invalid vertex count.");
    }

    const expectedLength = 1 + vertexFloatCount * 2;
    if (data.length !== expectedLength) {
      throw new Error("Legacy metaballs payload length mismatch.");
    }
  }, "Legacy metaballs");
  
  if (frame.contentType !== CONTENT_TYPE_FLOAT32_TENSOR) {
    return;
  }

  if (frame.payloadBuffer.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Float32 payload has an invalid byte length.");
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
  pendingBuffer = normalizeSocketPayload(event.data, "Metaballs socket");
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
