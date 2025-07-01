// python -m http.server 8000
// ngrok http 8000

const modelPath = "model/model_simplified.onnx";
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
  const size = 640; // 모델 입력 사이즈
  const cropSize = 320; // 자를 중앙 영역 사이즈 (px)

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = size;
  tempCanvas.height = size;
  const tempCtx = tempCanvas.getContext("2d");

  // ⚠️ 영상 크기가 아직 설정 안 됐을 수도 있으므로 기본값 보정
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;

  const sx = (vw - cropSize) / 2;
  const sy = (vh - cropSize) / 2;

  // 📌 중앙 영역 자르기 → (sx, sy, cropSize, cropSize)를 (0,0,size,size)로 리사이즈
  tempCtx.drawImage(video, sx, sy, cropSize, cropSize, 0, 0, size, size);

  const imgData = tempCtx.getImageData(0, 0, size, size).data;
  const float32 = new Float32Array(1 * 3 * size * size);
  for (let i = 0; i < size * size; i++) {
    float32[i] = imgData[i * 4] / 255;
    float32[i + size * size] = imgData[i * 4 + 1] / 255;
    float32[i + 2 * size * size] = imgData[i * 4 + 2] / 255;
  }

  return new ort.Tensor("float32", float32, [1, 3, size, size]);
}


function drawBoxes(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawMask();  // ✅ 추가

  const highConfDetections = detections.filter(det => det.score >= 0.5);
  const count = highConfDetections.length;

  document.getElementById("banner").style.display = count >= 3 ? "block" : "none";
}

function drawMask() {
  const w = canvas.width;
  const h = canvas.height;

  // 마스크 크기: 화면 가운데 정사각형
  const boxSize = Math.min(w, h) * 0.6;
  const left = (w - boxSize) / 2;
  const top = (h - boxSize) / 2;

  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.beginPath();
  ctx.rect(0, 0, w, h); // 전체 어둡게
  ctx.rect(left, top, boxSize, boxSize); // 가운데 뚫음
  ctx.fill("evenodd"); // 뚫기 모드
  ctx.restore();
}


function postprocess(outputTensor) {
  const raw = outputTensor.data;
  const numDet = raw.length / 6; // 혹은 85로 나누거나, 구조를 로그로 확인
  const detections = [];

  for (let i = 0; i < raw.length; i += 6) {
    const cx = raw[i];
    const cy = raw[i + 1];
    const w = raw[i + 2];
    const h = raw[i + 3];
    const obj = raw[i + 4];
    const cls = raw[i + 5];  // ❗️이게 확률인지 index인지 확인 필요

    const score = obj * cls;

    // 너무 많은 오탐 방지
    if (obj > 0.5 && cls > 0.8 && w * h > 0.001 && w * h < 0.5) {
      detections.push({
        label: "ECOQCODE",
        score,
        box: [cx - w / 2, cy - h / 2, w, h]
      });
    }
  }

  console.log("📦 Detections:", detections);
  return detections;
}


async function detectLoop() {
  if (!session) return;
  const tensor = preprocessFrame();
  const outputMap = await session.run({ images: tensor });

  // 🔥 여기가 중요
  const outputName = session.outputNames[0];
  const outputTensor = outputMap[outputName];

  const results = postprocess(outputTensor);
  drawBoxes(results);
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
