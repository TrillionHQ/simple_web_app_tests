import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';

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

let modelHandPose;
let modelEdges;
let video;

document.addEventListener('DOMContentLoaded', async function () {
    // Set up models
    const modelConfig = {
        runtime: 'tfjs',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
        modelType: 'lite'
    };
    modelHandPose = await handPoseDetection.createDetector(handPoseDetection.SupportedModels.MediaPipeHands, modelConfig);

    // Load edge detection model
    modelEdges = await tf.loadGraphModel('teed_model_tfjs_16/model.json');

    // Setup camera
    video = document.getElementById("webcam");
    if (!video) {
        console.error("Video element with id 'webcam' not found");
        return;
    }

    async function setupCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    }

    await setupCamera();
    video.play();

    // Set up canvas and context for edge detection
    const canvasElement = document.getElementById("output_canvas");
    const canvasCtx = canvasElement.getContext("2d", { willReadFrequently: true });

    // Set up Three.js for hand pose visualization
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
    renderer.setSize(width, height);
    threeJsContainer.appendChild(renderer.domElement);

    const geometry = new THREE.CylinderGeometry(0, 0.05, 0.15, 4);
    const material = new THREE.MeshBasicMaterial({ color: 0x0000FF });
    const pyramid = new THREE.Mesh(geometry, material);
    const edges = new THREE.EdgesGeometry(geometry);
    const edgesMaterial = new THREE.LineBasicMaterial({ color: 0xFFFFFF });
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

    // Process each video frame
    async function processVideoFrame() {
        if (!video.paused && !video.ended) {
            video.requestVideoFrameCallback(processVideoFrame);
        }
    
        try {
            // Step 1: Capture the video frame as a tensor and keep it in float32 format
            let tensorImage = tf.tidy(() => {
                let image = tf.browser.fromPixels(video).toFloat(); // Keep it as float32
                
                // Resize the image to match the expected input shape for the edge detection model
                image = tf.image.resizeBilinear(image, [352, 352]);
    
                // Explicitly keep the tensor on GPU
                return tf.keep(image);
            });
    
            // Measure time before handPose model execution
            const startHandPose = performance.now();
    
            // Step 2: Use the tensor for handPose model
            const handPosePredictions = await modelHandPose.estimateHands(tensorImage);
    
            // Measure time after handPose model execution
            const endHandPose = performance.now();
            const handPoseInferenceTime = endHandPose - startHandPose;
            console.log(`HandPose Inference time: ${handPoseInferenceTime.toFixed(2)} ms`);
    
            // Measure time before edge detection model execution
            const startTeed = performance.now();
    
            // Step 3: Apply the necessary transformations on the same tensor for the edge detection model
            const edgeDetectionOutput = tf.tidy(() => {
                const mean = tf.tensor([103.939, 116.779, 123.68], undefined, 'float32');
                let processedImage = tensorImage.sub(mean); // Normalize
                processedImage = processedImage.transpose([2, 0, 1]).expandDims(0); // Reshape as required to [1, 3, 352, 352]
                
                // Pass the processed image to the edge detection model
                return modelEdges.execute({ input: processedImage });
            });
            
    
            // Measure time after edge detection model execution
            const endTeed = performance.now();
            const teedInferenceTime = endTeed - startTeed;
            console.log(`TEED Inference time: ${teedInferenceTime.toFixed(2)} ms`);
    
            if (handPosePredictions.length > 0) {
                const keypoints3D = handPosePredictions[0].keypoints3D;
                const keypoints = handPosePredictions[0].keypoints;
                updatepyramidPosition(keypoints3D, keypoints);
            }
    
            const output = edgeDetectionOutput[3];
            const squeezedOutput = output.squeeze();
            const reshapedOutput = squeezedOutput.expandDims(-1);
    
            const minVal = reshapedOutput.min();
            const maxVal = reshapedOutput.max();
            const normalizedOutput = reshapedOutput.sub(minVal).div(maxVal.sub(minVal));
    
            // Render the normalized result directly to the canvas
            await tf.browser.toPixels(normalizedOutput, canvasElement);
            renderer.render(scene, camera);
    
            // Dispose of any remaining intermediate tensors
            tensorImage.dispose();
            squeezedOutput.dispose();
            reshapedOutput.dispose();
            normalizedOutput.dispose();
            minVal.dispose();
            maxVal.dispose();
    
            if (Array.isArray(edgeDetectionOutput)) {
                edgeDetectionOutput.forEach(tensor => tensor.dispose());
            } else {
                edgeDetectionOutput.dispose();
            }
    
        } catch (err) {
            console.error('Error during inference:', err);
        }
    }
    


    // Function to update the pyramid's position based on landmarks
    function updatepyramidPosition(keypoints3D, keypoints) {
        pyramid.position.set(0, 0, 0);
        pyramid_tvec.position.set(0.1, 0, 0);
        let quaternion = calculateFingerRotationpyramid(keypoints3D, 'INDEX');
        quaternion.x *= -1;
        quaternion.w *= -1;
        pyramid.setRotationFromQuaternion(quaternion);

        let quaternion_tvec = calculateFingerRotationpyramid(transformWorldPointsUsingPnP(cameraMatrix, distortion, keypoints3D, keypoints));
        quaternion_tvec.x *= -1;
        quaternion_tvec.w *= -1;
        pyramid_tvec.setRotationFromQuaternion(quaternion_tvec);

        const rotation_90_z = new THREE.Quaternion();
        rotation_90_z.setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
        pyramid.applyQuaternion(rotation_90_z);
        pyramid_tvec.applyQuaternion(rotation_90_z);
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

        return new THREE.Quaternion().setFromRotationMatrix(rotation_matrix);
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
        cv.solvePnP(objectPointsMat, imagePointsMat, cameraMatrixCv, distCoeffsCv, rvec, tvec, false, cv.SOLVEPNP_EPNP);

        const rotMat = new cv.Mat();
        cv.Rodrigues(rvec, rotMat);

        const transformMat = cv.Mat.eye(4, 4, cv.CV_64F);
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


    console.log('Current TensorFlow.js backend:', tf.getBackend());


    processVideoFrame(); // Start hand pose detection and edge detection
});
