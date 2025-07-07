// python -m http.server 8000
// ngrok http 8000

const modelPath = "model/ecoq_classifier.onnx";
let session;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
  video.srcObject = stream;
  return new Promise(resolve => video.onloadedmetadata = resolve);
}

async function loadModel() {
  session = await ort.InferenceSession.create(modelPath);
  console.log("ONNX model loaded");
}

function preprocessFrame() {
  const size = 224; // 모델 입력 사이즈
  const cropSize = Math.min(video.videoWidth, video.videoHeight); // 비디오의 짧은 변을 기준으로 중앙 정사각형 크기 설정

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = size;
  tempCanvas.height = size;
  const tempCtx = tempCanvas.getContext("2d");

  const sx = (video.videoWidth - cropSize) / 2;
  const sy = (video.videoHeight - cropSize) / 2;

  // 중앙 영역 자르기 및 리사이즈
  tempCtx.drawImage(video, sx, sy, cropSize, cropSize, 0, 0, size, size);

  const imgData = tempCtx.getImageData(0, 0, size, size).data;
  const float32 = new Float32Array(1 * 3 * size * size);
  for (let i = 0; i < size * size; i++) {
    float32[i] = (imgData[i * 4] / 255) * 2 - 1; // R
    float32[i + size * size] = (imgData[i * 4 + 1] / 255) * 2 - 1; // G
    float32[i + 2 * size * size] = (imgData[i * 4 + 2] / 255) * 2 - 1; // B
  }

  return new ort.Tensor("float32", float32, [1, 3, size, size]);
}


function handleDetections(detections) {
  const banner = document.getElementById("banner");
  if (!banner) return;

  banner.style.display = detections.length > 0 ? "block" : "none";
}

function drawMask() {
  const w = canvas.width;
  const h = canvas.height;

  const boxSize = Math.min(w, h) * 0.6;
  const verticalOffset = 0;
  const left = (w - boxSize) / 2;
  const top = (h - boxSize) / 2 + verticalOffset;

  // ✅ 1. 매 프레임 clear
  ctx.clearRect(0, 0, w, h);

  // ✅ 2. 전체 반투명 검은색 덮기
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
  ctx.fillRect(0, 0, w, h);

  // ✅ 3. 중앙 사각형 영역을 투명하게 뚫음
  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath();
  ctx.rect(left, top, boxSize, boxSize);
  ctx.fill();

  // ✅ 4. 다시 선을 그릴 수 있도록 복구
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  ctx.strokeRect(left, top, boxSize, boxSize);

  ctx.restore();
}



function postprocess(outputTensor) {
  const logit = outputTensor.data[0];
  const score = 1 / (1 + Math.exp(-logit)); // Apply sigmoid to get probability
  const threshold = 0.5; // Adjust as needed

  const isEcoqcodeDetected = score > threshold;
  console.log("ECOQCODE Detection Score:", score, "Detected:", isEcoqcodeDetected);
  return isEcoqcodeDetected;
}


async function detectLoop() {
  if (!session) return;

  const tensor = preprocessFrame();
  const outputMap = await session.run({ input: tensor });

  const outputName = session.outputNames[0];
  const outputTensor = outputMap[outputName];

  const isEcoqcodeDetected = postprocess(outputTensor);
  drawMask();
  handleDetections(isEcoqcodeDetected);

  requestAnimationFrame(detectLoop);
}

async function main() {
  await initCamera();
  await loadModel();

  // 영상 정보 확보를 위한 대기 (1프레임 뒤에야 width/height 접근 가능)
  await new Promise(resolve => setTimeout(resolve, 500));

  // 📌 캔버스 크기 설정
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  detectLoop();
}


main();
