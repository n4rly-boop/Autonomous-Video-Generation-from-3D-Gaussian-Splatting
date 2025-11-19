import * as THREE from 'three';
// Explicitly import SparkRenderer as well
import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

console.log(`Three.js Version: ${THREE.REVISION}`);

// 1. Initialize Three.js Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Setup Camera
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(0, 10, 10);

// Setup Renderer
// Force WebGL 2 context for 3D texture support
const canvas = document.createElement('canvas');
const context = canvas.getContext('webgl2', { antialias: false });
const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

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

splatMesh.initialized.then(() => {
    console.log("SplatMesh initialized successfully");
}).catch(e => {
    console.error("SplatMesh failed to initialize:", e);
});

scene.add(splatMesh);

// 4. Load Camera Path and Animate
let pathData = null;
let isPlaying = false;

fetch('./path.json')
    .then(res => res.json())
    .then(data => {
        pathData = data;
        isPlaying = true;
        console.log(`Path loaded: ${pathData.frames.length} frames, Duration: ${pathData.duration_sec}s`);
    })
    .catch(err => {
        console.error("Error loading path.json:", err);
    });

// 5. Animation Loop
renderer.setAnimationLoop((timeMs) => {
    
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
        
        const p0 = new THREE.Vector3(...f0.position);
        const p1 = new THREE.Vector3(...f1.position);
        const pos = new THREE.Vector3().lerpVectors(p0, p1, t);
        
        const l0 = new THREE.Vector3(...f0.look_at);
        const l1 = new THREE.Vector3(...f1.look_at);
        const look = new THREE.Vector3().lerpVectors(l0, l1, t);
        
        camera.position.copy(pos);
        camera.lookAt(look);
        
        // Debug log every ~1 second (assuming ~60fps, every 60th frame)
        if (Math.floor(timeMs / 1000) > Math.floor((timeMs - 16) / 1000)) {
           // console.log(`Camera Pos: ${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}`);
        }
    }
    
    renderer.render(scene, camera);
});
