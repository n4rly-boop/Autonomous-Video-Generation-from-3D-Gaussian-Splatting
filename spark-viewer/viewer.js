import * as THREE from 'three';
// Explicitly import SparkRenderer as well
import { FpsMovement, PointerControls, SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

console.log(`Three.js Version: ${THREE.REVISION}`);

// 1. Initialize Three.js Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Setup Camera
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(0, 0, 3);
camera.lookAt(0, 0, 0);

// Setup Renderer
// Force WebGL 2 context for 3D texture support
const canvas = document.createElement('canvas');
const context = canvas.getContext('webgl2', { antialias: false });
const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Controls
const fpsMovement = new FpsMovement({
    moveSpeed: 2.0,
    shiftMultiplier: 3.0,
    ctrlMultiplier: 0.4
});

const pointerControls = new PointerControls({
    canvas: renderer.domElement,
    rotateSpeed: 0.0025,
    slideSpeed: 0,
    scrollSpeed: 0
});

// 2. Initialize Spark Renderer
// This is critical for correct splat rendering and sorting
const spark = new SparkRenderer({ renderer });
scene.add(spark);

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// 3. Load the Gaussian Splat Scene
const splatMesh = new SplatMesh({
    url: './scene.ply',
    logLevel: 'info'
});
// Rotate splats into the viewer's Y-up basis (COLMAP scenes are Y-down, Z-forward).
splatMesh.rotation.x = Math.PI;

splatMesh.initialized.then(() => {
    console.log("SplatMesh initialized successfully");
}).catch(e => {
    console.error("SplatMesh failed to initialize:", e);
});

spark.add(splatMesh);

// 4. Load Camera Path and Animate
let pathData = null;
const params = new URLSearchParams(window.location.search);
const initialMode = params.get('mode') === 'interactive' ? 'interactive' : 'path';
let isPlaying = initialMode === 'path';
console.log(initialMode === 'path'
    ? 'Starting in path playback mode. Append ?mode=interactive to the URL for manual mode.'
    : 'Starting in interactive FPS mode. Append ?mode=path (default) to auto-follow the camera path.');
const movementTriggerCodes = new Set([
    'KeyW', 'KeyA', 'KeyS', 'KeyD',
    'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    'ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight',
    'KeyQ', 'KeyE', 'KeyR', 'KeyF',
    'PageUp', 'PageDown'
]);

const pauseCameraPath = () => {
    if (isPlaying) {
        isPlaying = false;
        console.log('Camera path paused; manual FPS controls active. Press P to resume path playback.');
    }
};

renderer.domElement.addEventListener('pointerdown', pauseCameraPath);
document.addEventListener('keydown', (event) => {
    if (event.code === 'KeyP') {
        if (pathData) {
            isPlaying = !isPlaying;
            console.log(isPlaying ? 'Camera path playback resumed.' : 'Camera path paused; manual FPS controls active.');
        }
        return;
    }
    if (movementTriggerCodes.has(event.code)) {
        pauseCameraPath();
    }
});

renderer.domElement.addEventListener('wheel', (event) => {
    const factor = event.deltaY < 0 ? 1.1 : 0.9;
    const nextSpeed = Math.min(Math.max(fpsMovement.moveSpeed * factor, 0.5), 20);
    fpsMovement.moveSpeed = nextSpeed;
    event.preventDefault();
}, { passive: false });

fetch('./path.json')
    .then(res => res.json())
    .then(data => {
        pathData = data;
        isPlaying = initialMode === 'path';
        console.log(`Path loaded: ${pathData.frames.length} frames, Duration: ${pathData.duration_sec}s (${isPlaying ? 'auto playback' : 'manual FPS mode'})`);
    })
    .catch(err => {
        console.error("Error loading path.json:", err);
    });

// 5. Animation Loop
let lastTime = performance.now();

renderer.setAnimationLoop((timeMs) => {
    const deltaSeconds = (timeMs - lastTime) / 1000;
    lastTime = timeMs;

    if (isPlaying && pathData && pathData.frames.length > 0) {
        // Calculate elapsed time
        const totalDuration = pathData.duration_sec;
        const elapsedSec = (timeMs / 1000) % totalDuration; 
        
        const u = elapsedSec / totalDuration;
        
        // Interpolate
        const floatIndex = u * (pathData.frames.length - 1);
        const idx0 = Math.floor(floatIndex);
        const idx1 = Math.min(idx0 + 1, pathData.frames.length - 1);
        const t = floatIndex - idx0;
        
        const f0 = pathData.frames[idx0];
        const f1 = pathData.frames[idx1];
        
        const applyFlip = (vec) => {
            vec.y *= -1;
            vec.z *= -1;
            return vec;
        };

        const p0 = applyFlip(new THREE.Vector3(...f0.position));
        const p1 = applyFlip(new THREE.Vector3(...f1.position));
        const pos = new THREE.Vector3().lerpVectors(p0, p1, t);
        
        const l0 = applyFlip(new THREE.Vector3(...f0.look_at));
        const l1 = applyFlip(new THREE.Vector3(...f1.look_at));
        const look = new THREE.Vector3().lerpVectors(l0, l1, t);
        
        camera.position.copy(pos);
        camera.lookAt(look);
        
        // Debug log every ~1 second (assuming ~60fps, every 60th frame)
        if (Math.floor(timeMs / 1000) > Math.floor((timeMs - 16) / 1000)) {
           // console.log(`Camera Pos: ${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}`);
        }
    } else {
        fpsMovement.update(deltaSeconds, camera);
        pointerControls.update(deltaSeconds, camera);
    }

    renderer.render(scene, camera);
});
