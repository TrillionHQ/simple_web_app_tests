import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

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

const demosSection = document.getElementById("liveView");

let handLandmarker = undefined;
let runningMode = "VIDEO";
let videoFileInput;
let webcamRunning = false;

const createHandLandmarker = async () => {
    try {
        const vision = await FilesetResolver.forVisionTasks("/wasm");

        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `./models/hand_landmarker.task`,
                delegate: "GPU",
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.7,
                model_complexity: 0,
            },
            runningMode: runningMode,
            numHands: 2
        });
        if (!handLandmarker) {
            throw new Error("HandLandmarker initialization failed, no instance created.");
        }
        demosSection.classList.remove("invisible");
    } catch (error) {
        console.error("Failed to initialize HandLandmarker:", error);
    }
};
createHandLandmarker().then(() =>
{
    video.addEventListener("loadeddata", () => {
        video.requestVideoFrameCallback(processVideoFrame);
    });
});



const video = document.getElementById("webcam");
video.playbackRate = .1
video.addEventListener("seeked", function ()
{
    video.requestVideoFrameCallback(processVideoFrame);
})

const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

videoFileInput = document.getElementById("videoFileInput");

videoFileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    const url = URL.createObjectURL(file);
    video.src = url;
    video.playbackRate = .1
    // video.play();
});

let lastVideoTime = -1;
let results = undefined;

// Three.js setup
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

const geometry = new THREE.CylinderGeometry(0, 0.05, 0.15, 4);
const material = new THREE.MeshBasicMaterial({ color: 0x0000FF });
const pyramid = new THREE.Mesh(geometry, material);

const edges = new THREE.EdgesGeometry(geometry);
const edgesMaterial = new THREE.LineBasicMaterial({ color: 0xFFFFFFFF });
const edgesMesh = new THREE.LineSegments(edges, edgesMaterial);

pyramid.add(edgesMesh);
scene.add(pyramid);

const material_tvec = new THREE.MeshBasicMaterial({ color: 0xf00000 });
const pyramid_tvec = new THREE.Mesh(geometry, material_tvec);

const edges_tvec = new THREE.EdgesGeometry(geometry);
const edgesMesh_tvec = new THREE.LineSegments(edges_tvec, edgesMaterial);

pyramid_tvec.add(edgesMesh_tvec);

scene.add(pyramid_tvec);

camera.position.z = 1;

// function animate() {
//     video.requestVideoFrameCallback(animate);
//     renderer.render(scene, camera);
// }
// animate();

function processVideoFrame() {
    if (!handLandmarker) {
        // Если handLandmarker еще не инициализирован, просто пропускаем кадр
        console.log("no landmarks")
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

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        handLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }
    canvasCtx.save();

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.landmarks) {
        for (let i = 0; i < results.landmarks.length; i++) {
            const landmarks = results.landmarks[i];
            const worldLandmarks = results.worldLandmarks ? results.worldLandmarks[i] : null;

            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });

            if (worldLandmarks) {
                updatepyramidPosition(landmarks, worldLandmarks);
            }
        }
    }

    canvasCtx.restore();

    renderer.render(scene, camera);
}

function updatepyramidPosition(landmarks, worldlandmarks) {
    pyramid.position.set(0, 0, 0);
    pyramid_tvec.position.set(0.2, 0, 0);

    let quaternion = calculateFingerRotationpyramid(worldlandmarks, 'INDEX');
    console.log(worldlandmarks, quaternion)
    quaternion.x *= -1;
    quaternion.w *= -1;
    pyramid.setRotationFromQuaternion(quaternion);

    const imagePoints = landmarks.map(
        (l) => new THREE.Vector3(l.x * width, l.y * height, 0)
    );

    let quaternion_tvec = calculateFingerRotationpyramid(transformWorldPointsUsingPnP(cameraMatrix, distortion, worldlandmarks, imagePoints));
    quaternion_tvec.x *= -1;
    quaternion_tvec.w *= -1;
    pyramid_tvec.setRotationFromQuaternion(quaternion_tvec);

    const rotation_90_z = new THREE.Quaternion();
    rotation_90_z.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
    pyramid.applyQuaternion(rotation_90_z);
    pyramid_tvec.applyQuaternion(rotation_90_z);


}

function transformWorldPointsUsingPnP(cameraMatrix, distCoeffs, worldPoints, imagePoints) {
    const objectPoints = worldPoints.map((p) => [p.x, p.y, p.z]);
    const imagePointsCv = imagePoints.map((p) => [p.x, p.y]);

    const objectPointsMat = cv.matFromArray(objectPoints.length, 1, cv.CV_32FC3, objectPoints.flat());
    const imagePointsMat = cv.matFromArray(imagePointsCv.length, 1, cv.CV_32FC2, imagePointsCv.flat());
    const cameraMatrixCv = cv.matFromArray(3, 3, cv.CV_64FC1, cameraMatrix);
    const distCoeffsCv = cv.matFromArray(distCoeffs.length, 1, cv.CV_64FC1, distCoeffs);

    const rvec = new cv.Mat();
    const tvec = new cv.Mat();

    const solvePnPMethod = cv.SOLVEPNP_EPNP;

    cv.solvePnP(
        objectPointsMat,
        imagePointsMat,
        cameraMatrixCv,
        distCoeffsCv,
        rvec,
        tvec,
        false,
        solvePnPMethod
    );

    const rotMat = new cv.Mat();
    cv.Rodrigues(rvec, rotMat);

    const CV_64F = cv.CV_64F;
    const transformMat = cv.Mat.eye(4, 4, CV_64F);
    rotMat.copyTo(transformMat.rowRange(0, 3).colRange(0, 3));
    tvec.copyTo(transformMat.rowRange(0, 3).col(3));

    const finalTransform = new THREE.Matrix4().fromArray(transformMat.data64F);

    const transformedWorldPoints = worldPoints.map((point) => {
        const vector = new THREE.Vector4(point.x, point.y, point.z, 1).applyMatrix4(finalTransform);

        if (vector.w !== 0) {
            vector.divideScalar(vector.w);
        }

        return new THREE.Vector3(vector.x, vector.y, vector.z);
    });

    objectPointsMat.delete();
    imagePointsMat.delete();
    cameraMatrixCv.delete();
    distCoeffsCv.delete();
    rvec.delete();
    tvec.delete();
    rotMat.delete();
    transformMat.delete();

    return transformedWorldPoints;
}

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


    //trying to fetch the opengl coordinates, still not correct

    /* const rotationY = new THREE.Quaternion();
    rotationY.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);

    const rotationX = new THREE.Quaternion();
    rotationX.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);

    let transformedQuaternion = quaternion.clone();
    transformedQuaternion.multiply(rotationY).multiply(rotationX);
    transformedQuaternion.x *= -1 */
    return quaternion

}
