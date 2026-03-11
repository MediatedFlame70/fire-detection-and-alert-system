let cameraStream = null;
let currentTab = 'image';

// TAB SWITCHING
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', ['image', 'video', 'camera'][i] === tab);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  hideResults();
}

// DRAG & DROP
function handleDragOver(e, isVid) {
  e.preventDefault();
  document.getElementById(isVid ? 'dropZoneVid' : 'dropZone').classList.add('dragover');
}
function handleDragLeave(e, isVid) {
  document.getElementById(isVid ? 'dropZoneVid' : 'dropZone').classList.remove('dragover');
}
function handleDrop(e, type) {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (type === 'image') loadImageFile(file);
  else loadVideoFile(file);
}

// IMAGE HANDLING
function handleImageFile(input) {
  if (input.files[0]) loadImageFile(input.files[0]);
}
function loadImageFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('imgPreview').classList.add('show');
    document.getElementById('resultCanvas').getContext('2d').clearRect(0, 0, 9999, 9999);
    hideResults();
  };
  reader.readAsDataURL(file);
}

// VIDEO HANDLING
function handleVideoFile(input) {
  if (input.files[0]) loadVideoFile(input.files[0]);
}
function loadVideoFile(file) {
  const url = URL.createObjectURL(file);
  const v = document.getElementById('videoPreview');
  v.src = url;
  v.style.display = 'block';
  document.getElementById('vidPreview').classList.add('show');
  hideResults();
}

// CAMERA
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    const feed = document.getElementById('cameraFeed');
    feed.srcObject = cameraStream;
    feed.style.display = 'block';
    document.getElementById('camPlaceholder').style.display = 'none';
  } catch (e) {
    alert('Camera access denied. Please allow camera permissions and try again.');
  }
}
function stopCamera() {
  if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
  const feed = document.getElementById('cameraFeed');
  feed.style.display = 'none';
  feed.srcObject = null;
  document.getElementById('camPlaceholder').style.display = 'block';
}
async function captureAndAnalyze() {
  const feed = document.getElementById('cameraFeed');
  if (!feed.srcObject) { alert('Please start the camera first!'); return; }
  document.getElementById('camProcessing').classList.add('show');
  await new Promise(r => setTimeout(r, 1800 + Math.random() * 1000));
  document.getElementById('camProcessing').classList.remove('show');
  showResults(getRandomDetection());
}

// ANALYZE IMAGE
async function analyzeImage() {
  const img = document.getElementById('previewImg');
  if (!img.src || img.src === window.location.href) return;
  document.getElementById('imgProcessing').classList.add('show');
  document.getElementById('analyzeImgBtn').disabled = true;
  await new Promise(r => setTimeout(r, 1500 + Math.random() * 1200));
  document.getElementById('imgProcessing').classList.remove('show');
  document.getElementById('analyzeImgBtn').disabled = false;
  const result = getRandomDetection();
  drawBoundingBox(result);
  showResults(result);
}

// ANALYZE VIDEO
async function analyzeVideo() {
  const v = document.getElementById('videoPreview');
  if (!v.src) return;
  document.getElementById('vidProcessing').classList.add('show');
  document.getElementById('analyzeVidBtn').disabled = true;
  const texts = [
    'Scanning frame 1/24... Detecting fire signatures...',
    'Scanning frame 8/24... Running CNN backbone...',
    'Scanning frame 16/24... Vision Transformer analysis...',
    'Scanning frame 24/24... Aggregating detections...',
    'Finalizing results...'
  ];
  for (let t of texts) {
    document.getElementById('vidProcessingText').textContent = t;
    await new Promise(r => setTimeout(r, 600 + Math.random() * 400));
  }
  document.getElementById('vidProcessing').classList.remove('show');
  document.getElementById('analyzeVidBtn').disabled = false;
  showResults(getRandomDetection());
}

// RANDOM DETECTION SIMULATION
function getRandomDetection() {
  const types = ['fire', 'smoke', 'both', 'neutral'];
  const type = types[Math.floor(Math.random() * types.length)];
  let firePct, smokePct, neutralPct;
  if (type === 'fire') {
    firePct = 72 + Math.random() * 25;
    smokePct = Math.random() * 20;
    neutralPct = 100 - firePct - smokePct;
  } else if (type === 'smoke') {
    smokePct = 68 + Math.random() * 28;
    firePct = Math.random() * 18;
    neutralPct = 100 - firePct - smokePct;
  } else if (type === 'both') {
    firePct = 55 + Math.random() * 20;
    smokePct = 30 + Math.random() * 20;
    neutralPct = Math.max(0, 100 - firePct - smokePct);
  } else {
    neutralPct = 78 + Math.random() * 20;
    firePct = Math.random() * 12;
    smokePct = Math.random() * (100 - firePct - neutralPct);
  }
  const bbox = [
    (Math.random() * 0.4 + 0.1).toFixed(2),
    (Math.random() * 0.4 + 0.1).toFixed(2),
    (Math.random() * 0.3 + 0.15).toFixed(2),
    (Math.random() * 0.3 + 0.15).toFixed(2)
  ];
  return { type, firePct: Math.max(0, firePct), smokePct: Math.max(0, smokePct), neutralPct: Math.max(0, neutralPct), bbox };
}

// DRAW BOUNDING BOX ON CANVAS
function drawBoundingBox(result) {
  if (result.type === 'neutral') return;
  const img = document.getElementById('previewImg');
  const canvas = document.getElementById('resultCanvas');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const [bx, by, bw, bh] = result.bbox;
  const x = bx * canvas.width, y = by * canvas.height;
  const w = bw * canvas.width, h = bh * canvas.height;
  const color = result.type === 'smoke' ? '#ff8800' : '#ff2200';
  ctx.strokeStyle = color; ctx.lineWidth = 3;
  ctx.shadowColor = color; ctx.shadowBlur = 12;
  ctx.strokeRect(x, y, w, h);
  ctx.shadowBlur = 0;
  ctx.fillStyle = color;
  ctx.font = 'bold 16px Rajdhani, sans-serif';
  const label = result.type === 'both'
    ? `FIRE+SMOKE ${result.firePct.toFixed(1)}%`
    : result.type === 'fire'
    ? `FIRE ${result.firePct.toFixed(1)}%`
    : `SMOKE ${result.smokePct.toFixed(1)}%`;
  const tw = ctx.measureText(label).width;
  ctx.fillRect(x, y - 24, tw + 12, 24);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, x + 6, y - 6);
}

// SHOW RESULTS
function showResults(result) {
  const banner = document.getElementById('alertBanner');
  const icon = document.getElementById('alertIcon');
  const title = document.getElementById('alertTitle');
  const msg = document.getElementById('alertMsg');
  banner.className = 'alert-banner';

  if (result.type === 'fire') {
    banner.classList.add('alert-fire');
    icon.textContent = '🔥'; title.textContent = '⚠ FIRE DETECTED';
    msg.textContent = 'High confidence fire signature detected. Immediate evacuation and emergency response recommended!';
    showSiren(); showToast('fire', '🔥 FIRE DETECTED', 'High confidence fire identified. Immediate action required!');
  } else if (result.type === 'smoke') {
    banner.classList.add('alert-smoke');
    icon.textContent = '💨'; title.textContent = '⚠ SMOKE DETECTED';
    msg.textContent = 'Smoke patterns identified in the image. Investigate the area for potential fire sources.';
    showSiren(); showToast('smoke', '💨 SMOKE DETECTED', 'Smoke signature found. Investigate for fire sources.');
  } else if (result.type === 'both') {
    banner.classList.add('alert-both');
    icon.textContent = '🔥💨'; title.textContent = '🚨 FIRE & SMOKE DETECTED';
    msg.textContent = 'Both fire and smoke detected simultaneously. Critical situation — trigger emergency protocols immediately!';
    showSiren(); showToast('fire', '🚨 FIRE & SMOKE DETECTED', 'Critical: Both fire and smoke present. Emergency protocols needed!');
  } else {
    banner.classList.add('alert-neutral');
    icon.textContent = '✅'; title.textContent = 'NO FIRE OR SMOKE DETECTED';
    msg.textContent = 'The model classified this input as neutral. No fire or smoke signatures found.';
    showToast('neutral', '✅ CLEAR — No Detection', 'No fire or smoke detected in this input.');
  }

  const cls = result.type === 'both' ? 'Fire + Smoke' : result.type === 'fire' ? 'Fire' : result.type === 'smoke' ? 'Smoke' : 'Neutral';
  document.getElementById('resClass').textContent = cls;
  document.getElementById('resFire').textContent = result.firePct.toFixed(1) + '%';
  document.getElementById('resSmoke').textContent = result.smokePct.toFixed(1) + '%';
  document.getElementById('resBbox').textContent = `[${result.bbox.join(', ')}]`;

  setTimeout(() => {
    const fp = Math.min(result.firePct, 100), sp = Math.min(result.smokePct, 100), np = Math.min(result.neutralPct, 100);
    document.getElementById('barFire').style.width = fp + '%';
    document.getElementById('barSmoke').style.width = sp + '%';
    document.getElementById('barNeutral').style.width = np + '%';
    document.getElementById('barFirePct').textContent = fp.toFixed(1) + '%';
    document.getElementById('barSmokePct').textContent = sp.toFixed(1) + '%';
    document.getElementById('barNeutralPct').textContent = np.toFixed(1) + '%';
  }, 100);

  document.getElementById('resultsSection').classList.add('show');
  document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
  document.getElementById('resultsSection').classList.remove('show');
  hideSiren();
}

// SIREN EFFECT
function showSiren() {
  document.getElementById('siren').classList.add('show');
  setTimeout(hideSiren, 4000);
}
function hideSiren() {
  document.getElementById('siren').classList.remove('show');
}

// TOAST NOTIFICATION
function showToast(type, title, body) {
  const t = document.getElementById('toast');
  document.getElementById('toastIcon').textContent = type === 'fire' ? '🔥' : type === 'smoke' ? '💨' : '✅';
  document.getElementById('toastTitle').textContent = title;
  document.getElementById('toastBody').textContent = body;
  t.className = 'toast show ' + type;
  setTimeout(() => t.classList.remove('show'), 6000);
}

// DISMISS ALERT
function dismissAlert() {
  hideResults();
  hideSiren();
  ['barFire', 'barSmoke', 'barNeutral'].forEach(id => document.getElementById(id).style.width = '0');
  const canvas = document.getElementById('resultCanvas');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}
