// Import necessary modules
import * as handpose from '@tensorflow-models/handpose';
import * as tfjs from '@tensorflow/tfjs-backend-webgl';

// Set up video parameters and camera matrix
const height = 480;
const width = 640;
const focalLength = width * 0.6;
const center = [width / 2, height / 2];
const cameraMatrix = [
    focalLength, 0, center[0],
    0, focalLength, center[1],
    0, 0, 1
];
const distortion = [0, 0, 0, 0];


let videoFileInput;
let model;
let video;

// Load the handpose model and set up the video element
(async function () {
    model = await handpose.load();

    video = document.getElementById("webcam");
    video.playbackRate = 1;

    video.addEventListener("loadeddata", () => {
        video.requestVideoFrameCallback(processVideoFrame);
    });
})();

// Set up canvas and video file input elements
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

videoFileInput = document.getElementById("videoFileInput");

videoFileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    const url = URL.createObjectURL(file);
    video.src = url;
    video.playbackRate = 0.1;
});

// Set up Three.js
const threeJsContainer = document.getElementById('threejs-container');
const scene = new THREE.Scene();
const aspect = width / height;
const frustumSize = 1;
const camera = new THREE.OrthographicCamera(
    frustumSize * aspect / -2,
    frustumSize * aspect / 2,
    frustumSize / 2,
    frustumSize / -2,
    0.1,
    1000
);
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(width, height); // Match video dimensions
threeJsContainer.appendChild(renderer.domElement);

// Create pyramids for the scene
const geometry = new THREE.CylinderGeometry(0, 0.05, 0.15, 4);
const material = new THREE.MeshBasicMaterial({ color: 0x0000FF });
const pyramid = new THREE.Mesh(geometry, material);

const edges = new THREE.EdgesGeometry(geometry);
const edgesMaterial = new THREE.LineBasicMaterial({ color: 0xFFFFFF });
const edgesMesh = new THREE.LineSegments(edges, edgesMaterial);

pyramid.add(edgesMesh);
scene.add(pyramid);



camera.position.z = 1;

// Function to process each video frame
async function processVideoFrame() {
    if (!model) {
        console.log("no model");
        video.requestVideoFrameCallback(processVideoFrame);
        return;
    }

    if (!video.paused && !video.ended) {
        video.requestVideoFrameCallback(processVideoFrame);
    }

    canvasElement.style.width = width + "px";
    canvasElement.style.height = height + "px";
    canvasElement.width = width;
    canvasElement.height = height;

    const predictions = await model.estimateHands(video);

    if (predictions.length > 0) {
        for (let i = 0; i < predictions.length; i++) {
            const keypoints = predictions[i].landmarks;

            for (let j = 0; j < keypoints.length; j++) {
                const [x, y, z] = keypoints[j];
                console.log(`Keypoint ${j}: [${x}, ${y}, ${z}]`);
            }
            updatepyramidPosition(keypoints)
        }
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.restore();

    renderer.render(scene, camera);
}

// Function to update the pyramid's position based on landmarks
function updatepyramidPosition(keypoints) {
    console.log('updating')
    pyramid.position.set(0, 0, 0);
    let quaternion = calculateFingerRotationpyramid(keypoints, 'INDEX');
    quaternion.x *= -1;
    quaternion.w *= -1;
    pyramid.setRotationFromQuaternion(quaternion);


    const rotation_90_z = new THREE.Quaternion();
    rotation_90_z.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
    pyramid.applyQuaternion(rotation_90_z);
}


// Function to calculate the rotation of the pyramid based on finger landmarks
function calculateFingerRotationpyramid(points, finger_name = 'INDEX') {
    let idx_mcp, idx_pip, idx_mcp1, idx_mcp2;

    switch (finger_name) {
        case 'INDEX':
            idx_mcp = 5;
            idx_pip = 6;
            idx_mcp1 = 5;
            idx_mcp2 = 13;
            break;
        case 'MIDDLE':
            idx_mcp = 9;
            idx_pip = 10;
            idx_mcp1 = 5;
            idx_mcp2 = 13;
            break;
        case 'RING':
            idx_mcp = 13;
            idx_pip = 14;
            idx_mcp1 = 5;
            idx_mcp2 = 13;
            break;
        case 'PINKY':
            idx_mcp = 17;
            idx_pip = 18;
            idx_mcp1 = 5;
            idx_mcp2 = 13;
            break;
        default:
            idx_mcp = 2;
            idx_pip = 3;
            idx_mcp1 = 5;
            idx_mcp2 = 9;
            break;
    }

    let point_mcp1 = new THREE.Vector3(points[idx_mcp1][0], points[idx_mcp1][1], points[idx_mcp1][2]);
    let point_mcp2 = new THREE.Vector3(points[idx_mcp2][0], points[idx_mcp2][1], points[idx_mcp2][2]);
    let point_pip = new THREE.Vector3(points[idx_pip][0], points[idx_pip][1], points[idx_pip][2]);
    let point_mcp = new THREE.Vector3(points[idx_mcp][0], points[idx_mcp][1], points[idx_mcp][2]);

    const base_vector = point_mcp2.clone().sub(point_mcp1).normalize();
    const finger_vector = point_pip.clone().sub(point_mcp).normalize();

    const zAxis = finger_vector.clone();

    let xAxis = new THREE.Vector3().crossVectors(finger_vector, base_vector).normalize();

    const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();

    const rotation_matrix = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);

    const quaternion = new THREE.Quaternion().setFromRotationMatrix(rotation_matrix);

    return quaternion;
}
