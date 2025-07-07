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


function handleDetections(detections) {
  const banner = document.getElementById("banner");
  if (!banner) return;

  banner.style.display = detections.length > 0 ? "block" : "none";
}

function drawMask() {
  const w = canvas.width;
  const h = canvas.height;

  const boxSize = Math.min(w, h) * 0.;
  const verticalOffset = -h * 0.1;
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
  const raw = outputTensor.data;
  const numDet = raw.length / 6;
  const detections = [];

  for (let i = 0; i < raw.length; i += 6) {
    const cx = raw[i];
    const cy = raw[i + 1];
    const w = raw[i + 2];
    const h = raw[i + 3];
    const obj = raw[i + 4];
    const cls = raw[i + 5];  // 단일 클래스이므로 거의 항상 1에 가까움

    // 단일 클래스 모델 → cls는 무시하고 obj만 쓰자
    const score = obj;

    // 완화된 조건
    if (score > 0.4 && w * h > 0.0003 && w * h < 0.6) {
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

  const outputName = session.outputNames[0];
  const outputTensor = outputMap[outputName];

  const results = postprocess(outputTensor);  // ✅ 이 줄 추가!
  drawMask();
  handleDetections(results);

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
