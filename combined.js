import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
// Adds the WASM backend to the global backend registry.
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';




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
    //tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);

    //setWasmPaths('192.168.0.101/node_modules/@tensorflow/tfjs-backend-wasm/dist');

    await tf.setBackend('webgl');
    await tf.ready();

    console.log('Current TensorFlow.js backend:', tf.getBackend());


    // Set up models
    const modelConfig = {
        runtime: 'mediapipe',
        solutionPath: 'models',
        modelType: 'lite'
    };
    modelHandPose = await handPoseDetection.createDetector(handPoseDetection.SupportedModels.MediaPipeHands, modelConfig);

    // Load edge detection model
    modelEdges = await tf.loadGraphModel('teed_model_tfjs/model.json');

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

    async function processVideoFrame() {
        if (!video.paused && !video.ended) {
            video.requestVideoFrameCallback(processVideoFrame);
        }

        try {
            // Step 1: Захват кадра видео в тензор в формате uint8
            let tensorImage = tf.tidy(() => {
                return tf.browser.fromPixels(video); // Оставляем изображение в исходном формате uint8 (0-255)
            });

            // Measure time before handPose model execution
            const startHandPose = performance.now();

            // Step 2: Используем изображение для MediaPipe
            const handPosePredictions = await modelHandPose.estimateHands(video); // Передаем напрямую видео или uint8 тензор

            // Measure time after handPose model execution
            const endHandPose = performance.now();
            const handPoseInferenceTime = endHandPose - startHandPose;
            console.log(`HandPose Inference time: ${handPoseInferenceTime.toFixed(2)} ms`);

            // Measure time before edge detection model execution
            const startTeed = performance.now();

            // Step 3: Применяем edge detection модель (сначала меняем размер тензора)
            const edgeDetectionOutput = tf.tidy(() => {
                const mean = tf.tensor([103.939, 116.779, 123.68]);
                let processedImage = tensorImage.toFloat().sub(mean); // Преобразуем в float32 и нормализуем
                processedImage = tf.image.resizeBilinear(processedImage, [352, 352]); // Меняем размер изображения на [352, 352]
                processedImage = processedImage.transpose([2, 0, 1]).expandDims(0);
                return modelEdges.execute({ input: processedImage });
            });

            // Measure time after edge detection model execution
            const endTeed = performance.now();
            const teedInferenceTime = endTeed - startTeed;
            console.log(`TEED Inference time: ${teedInferenceTime.toFixed(2)} ms`);

            if (handPosePredictions.length > 0) {
                const keypoints = handPosePredictions[0].keypoints;

                // Преобразуем лэндмарки в пиксельные координаты изображения 352x352
                const imageLandmarks = convertLandmarksToImageCoords(keypoints, 640, 480, 352, 352);

                // Применяем фильтрацию контуров на основе преобразованных координат
                let filteredOutput = applyMCP_PIPFiltering(edgeDetectionOutput[3], imageLandmarks);

                // Проверяем, что filteredOutput не содержит недопустимые значения (NaN, Inf)
                const containsNaN = filteredOutput.isNaN().any().dataSync()[0];
                const containsInf = filteredOutput.isInf().any().dataSync()[0];

                if (containsNaN || containsInf) {
                    console.error("Ошибка: filteredOutput содержит недопустимые значения (NaN или Inf).");
                } else {
                    // Масштабируем значения на основе максимального абсолютного значения
                    const absMaxVal = filteredOutput.abs().max();

                    // Масштабируем значения через деление на максимальное абсолютное значение
                    const scaledOutput = filteredOutput.div(absMaxVal);

                    // Приведение значений к диапазону [0, 1] для визуализации
                    const minVal = scaledOutput.min();
                    const maxVal = scaledOutput.max();

                    // Выполняем бинаризацию
                    const binaryOutput = tf.greaterEqual(scaledOutput.sub(minVal).div(maxVal.sub(minVal)), 0.5);

                    // Преобразуем бинарные значения в формат [0, 1]
                    const normalizedOutput = binaryOutput.cast('float32'); // Приводим к float32 для работы с toPixels

                    // Отображаем изображение
                    await tf.browser.toPixels(normalizedOutput, canvasElement);


                    // Освобождаем ресурсы
                    absMaxVal.dispose();
                    minVal.dispose();
                    maxVal.dispose();
                    scaledOutput.dispose();
                    filteredOutput.dispose();
                }
            }
            else {
                // Если нет лэндмарков, просто отображаем результат детекции краев
                const output = edgeDetectionOutput[3];
                const squeezedOutput = output.squeeze();

                // Нормализуем значение в диапазоне от 0 до 1
                const minVal = squeezedOutput.min();
                const maxVal = squeezedOutput.max();
                // Выполняем бинаризацию
                const binaryOutput = tf.greaterEqual(squeezedOutput.sub(minVal).div(maxVal.sub(minVal)), 0.5);

                // Преобразуем бинарные значения в формат [0, 1]
                const normalizedOutput = binaryOutput.cast('float32'); // Приводим к float32 для работы с toPixels

                // Отображаем изображение
                await tf.browser.toPixels(normalizedOutput, canvasElement);


                // Освобождаем ресурсы
                squeezedOutput.dispose();
                normalizedOutput.dispose();
                minVal.dispose();
                maxVal.dispose();
            }

            renderer.render(scene, camera);

            // Освобождаем память от промежуточных тензоров
            tensorImage.dispose();
            if (Array.isArray(edgeDetectionOutput)) {
                edgeDetectionOutput.forEach(tensor => tensor.dispose());
            } else {
                edgeDetectionOutput.dispose();
            }

        } catch (err) {
            console.error('Error during inference:', err);
        }
    }

    // Преобразование нормализованных координат лэндмарков в координаты пикселей изображения 352x352
    function convertLandmarksToImageCoords(landmarks, inputWidth, inputHeight, targetWidth, targetHeight) {
        // Преобразуем нормализованные координаты лэндмарков в координаты в пикселях
        return landmarks.map(landmark => {
            const x = landmark.x;  // Преобразование нормализованной координаты X в пиксели
            const y = landmark.y; // Преобразование нормализованной координаты Y в пиксели

            // Преобразуем координаты исходного изображения в координаты изображения 352x352
            const x_new = (x / inputWidth) * targetWidth;
            const y_new = (y / inputHeight) * targetHeight;

            return { x: x_new, y: y_new };
        });
    }

    // Пример функции для фильтрации контуров на основе MCP-PIP
    function applyMCP_PIPFiltering(edgeTensor, keypoints) {
        const mcpPipVectors = calculateMCP_PIPVectors(keypoints);

        // Убираем лишние измерения тензора, чтобы получить форму [352, 352]
        const squeezedEdgesTensor = edgeTensor.squeeze();

        // Преобразуем тензор детекции краев в массив для постобработки
        const edgesArray = squeezedEdgesTensor.arraySync(); // Получаем данные в массиве для обработки

        // Пробегаемся по каждому пикселю и фильтруем его, если он рядом с MCP-PIP
        for (let y = 0; y < edgesArray.length; y++) {
            for (let x = 0; x < edgesArray[y].length; x++) {
                const pixelPos = { x, y };

                // Проверяем близость пикселя к MCP-PIP вектору каждого пальца
                for (const vector of mcpPipVectors) {
                    const distance = pointToLineDistance(pixelPos, vector.mcp, vector.pip);
                    if (distance < vector.threshold) {
                        edgesArray[y][x] = 0;
                        break;
                    }
                }
            }
        }

        // Преобразуем обратно в тензор с использованием правильной формы
        const filteredTensor = tf.tensor(edgesArray, squeezedEdgesTensor.shape); // Используем форму [352, 352]
        return filteredTensor;
    }



    // Расчет векторов MCP-PIP для каждого пальца
    function calculateMCP_PIPVectors(keypoints) {
        const fingers = ['index', 'middle', 'ring', 'pinky'];
        const vectors = [];

        // Получаем MCP точки среднего и безымянного пальцев
        const middleMCP = getKeyPoint('middle_mcp', keypoints);
        const ringMCP = getKeyPoint('ring_mcp', keypoints);

        const standardDistance = 30;

        if (middleMCP && ringMCP) {
            // Вычисляем текущее расстояние между MCP среднего и безымянного пальцев
            const currentDistance = euclideanDistance(middleMCP, ringMCP);

            // Вычисляем коэффициент масштаба (относительно стандартного расстояния в 30 пикселей)
            const scaleFactor = currentDistance / standardDistance;

            // Пробегаемся по каждому пальцу и вычисляем пороговое значение для MCP-PIP
            for (const finger of fingers) {
                const mcp = getKeyPoint(finger + '_mcp', keypoints);
                const pip = getKeyPoint(finger + '_pip', keypoints);

                if (mcp && pip) {
                    // Пропорционально изменяем порог (threshold)
                    const threshold = 5 * scaleFactor;

                    vectors.push({
                        mcp,
                        pip,
                        threshold
                    });
                }
            }
        }

        return vectors;
    }

    function euclideanDistance(point1, point2) {
        return Math.sqrt(Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2));
    }


    // Получение лэндмарков по имени
    function getKeyPoint(name, keypoints) {
        const fingerMap = {
            'index_mcp': 5,
            'index_pip': 6,
            'middle_mcp': 9,
            'middle_pip': 10,
            'ring_mcp': 13,
            'ring_pip': 14,
            'pinky_mcp': 17,
            'pinky_pip': 18
        };

        const index = fingerMap[name];
        if (index !== undefined && keypoints[index]) {
            return { x: keypoints[index].x, y: keypoints[index].y };
        }
        return null;
    }

    // Функция для расчета расстояния от точки до линии (вектора MCP-PIP)
    function pointToLineDistance(point, lineStart, lineEnd) {
        const a = lineStart.y - lineEnd.y;
        const b = lineEnd.x - lineStart.x;
        const c = lineStart.x * lineEnd.y - lineEnd.x * lineStart.y;
        return Math.abs(a * point.x + b * point.y + c) / Math.sqrt(a * a + b * b);
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



    processVideoFrame(); // Start hand pose detection and edge detection
});
