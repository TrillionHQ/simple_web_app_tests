// Импортируем необходимые модули
import * as mpHands from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

// Устанавливаем параметры видео и камеры
const height = 480;
const width = 640;

let videoFileInput;
let hands;
let video;

// Загрузка модели Mediapipe Hands и настройка видео элемента
document.addEventListener('DOMContentLoaded', async function () {
    // Настройка модели Mediapipe Hands
    hands = new mpHands.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    });

    hands.onResults(onResults);

    const video = document.getElementById("webcam");
    if (!video) {
        console.error("Video element with id 'webcam' not found");
        return;
    }

    const camera = new Camera(video, {
        onFrame: async () => {
            await hands.send({ image: video });
        },
        width: width,
        height: height
    });
    camera.start();

    video.addEventListener("loadeddata", () => {
        console.log("Video data loaded, starting frame processing");
        video.requestVideoFrameCallback(processVideoFrame);
    });

    // Настройка элементов canvas и ввода видеофайла
    const canvasElement = document.getElementById("output_canvas");
    const canvasCtx = canvasElement.getContext("2d");

    const videoFileInput = document.getElementById("videoFileInput");
    if (!videoFileInput) {
        console.error("Video input element with id 'videoFileInput' not found");
        return;
    }

    videoFileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) {
            console.error("No file selected");
            return;
        }
        const url = URL.createObjectURL(file);
        video.src = url;
        video.playbackRate = 0.1;

        video.load();
        video.play().then(() => {
            console.log("Video started playing");
        }).catch((error) => {
            console.error("Error occurred during video play:", error);
        });
    });

    // Настройка Three.js
    const threeJsContainer = document.getElementById('threejs-container');
    const scene = new THREE.Scene();
    const aspect = width / height;
    const frustumSize = 1;
    const cameraThree = new THREE.OrthographicCamera(
        frustumSize * aspect / -2,
        frustumSize * aspect / 2,
        frustumSize / 2,
        frustumSize / -2,
        0.1,
        1000
    );
    const renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(width, height);
    threeJsContainer.appendChild(renderer.domElement);

    // Создание пирамид для сцены
    const geometry = new THREE.CylinderGeometry(0, 0.05, 0.15, 4);
    const material = new THREE.MeshBasicMaterial({ color: 0x0000FF });
    const pyramid = new THREE.Mesh(geometry, material);

    const edges = new THREE.EdgesGeometry(geometry);
    const edgesMaterial = new THREE.LineBasicMaterial({ color: 0xFFFFFF });
    const edgesMesh = new THREE.LineSegments(edges, edgesMaterial);

    pyramid.add(edgesMesh);
    scene.add(pyramid);

    const material_tvec = new THREE.MeshBasicMaterial({ color: 0xf00000 });

    const edges_tvec = new THREE.EdgesGeometry(geometry);
    const edgesMesh_tvec = new THREE.LineSegments(edges_tvec, edgesMaterial);



    cameraThree.position.z = 1;

    // Функция для обработки каждого кадра видео
    async function processVideoFrame() {
        if (!video.paused && !video.ended) {
            video.requestVideoFrameCallback(processVideoFrame);
        }

        canvasElement.style.width = width + "px";
        canvasElement.style.height = height + "px";
        canvasElement.width = width;
        canvasElement.height = height;

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.restore();

        renderer.render(scene, cameraThree);
    }

    // Функция обработки результатов Mediapipe Hands
    function onResults(results) {
        if (results.multiHandWorldLandmarks && results.multiHandWorldLandmarks.length > 0) {
            const landmarks = results.multiHandWorldLandmarks[0];
            updatepyramidPosition(landmarks);
        }
    }

    // Функция для обновления позиции пирамиды на основе ориентиров
    function updatepyramidPosition(keypoints3D) {
        pyramid.position.set(0, 0, 0);
        let quaternion = calculateFingerRotationpyramid(keypoints3D, 'INDEX');

        quaternion.x *= -1;
        quaternion.w *= -1;
        pyramid.setRotationFromQuaternion(quaternion);

        const rotation_90_z = new THREE.Quaternion();
        rotation_90_z.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
        pyramid.applyQuaternion(rotation_90_z);
    }

    // Функция для расчета вращения пирамиды на основе ориентиров пальцев
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

        let point_mcp1 = new THREE.Vector3(points[idx_mcp1].x, points[idx_mcp1].y, points[idx_mcp1].z);
        let point_mcp2 = new THREE.Vector3(points[idx_mcp2].x, points[idx_mcp2].y, points[idx_mcp2].z);
        let point_pip = new THREE.Vector3(points[idx_pip].x, points[idx_pip].y, points[idx_pip].z);
        let point_mcp = new THREE.Vector3(points[idx_mcp].x, points[idx_mcp].y, points[idx_mcp].z);

        const base_vector = point_mcp2.clone().sub(point_mcp1).normalize();
        const finger_vector = point_pip.clone().sub(point_mcp).normalize();

        const zAxis = finger_vector.clone();

        let xAxis = new THREE.Vector3().crossVectors(finger_vector, base_vector).normalize();

        const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();

        const rotation_matrix = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);

        const quaternion = new THREE.Quaternion().setFromRotationMatrix(rotation_matrix);

        return quaternion;
    }
});
