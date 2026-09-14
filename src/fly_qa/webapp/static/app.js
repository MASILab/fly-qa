const brainCanvas = document.getElementById("brain-canvas");
const brain = new NeuronCanvas(brainCanvas);
const brainCaption = document.getElementById("brain-caption");

const deskCanvas = document.getElementById("fly-desk-canvas");
const flyDesk = typeof THREE !== "undefined" && deskCanvas ? new FlyDesk3D(deskCanvas) : null;

const tallyEls = {
  pass: document.getElementById("tally-pass"),
  flag: document.getElementById("tally-flag"),
  fail: document.getElementById("tally-fail"),
};
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const currentPath = document.getElementById("current-path");
const currentVerdict = document.getElementById("current-verdict");
const currentScores = document.getElementById("current-scores");
const currentReason = document.getElementById("current-reason");
const currentThumb = document.getElementById("current-thumb");
const resultsBody = document.getElementById("results-body");

const MAX_ROWS = 200;

// Every "result" event received so far, keyed by image path -- lets clicking a
// row in the table re-display that image (and replay its real neuron activity)
// after inference has moved on to later images.
const resultsByPath = new Map();
let selectedPath = null;

fetch("/api/viz-subset")
  .then((r) => r.json())
  .then((data) => {
    brain.setGraph(data.nodes, data.edges);
    const roleCounts = data.nodes.reduce((acc, n) => {
      acc[n.role] = (acc[n.role] || 0) + 1;
      return acc;
    }, {});
    brainCaption.textContent =
      `real MaleCNS v1.0 subset: ${data.nodes.length} neurons ` +
      `(${roleCounts.r1r6 || 0} R1-R6, ${roleCounts.r8 || 0} R8, ` +
      `${roleCounts.hidden || 0} hidden, ${roleCounts.dnp20 || 0} DNp20, ${roleCounts.dnpe017 || 0} DNpe017)`;
  })
  .catch(() => {
    brainCaption.textContent = "connectome subset unavailable";
  });

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    handleEvent(event);
  };

  ws.onclose = () => {
    setTimeout(connect, 1000);
  };
}

function handleEvent(event) {
  if (event.type === "started") {
    progressLabel.textContent = `0 / ${event.total_images}`;
    return;
  }

  if (event.type === "result") {
    tallyEls[event.verdict].textContent = event.tally[event.verdict];
    progressFill.style.width = `${(event.processed_images / event.total_images) * 100}%`;
    progressLabel.textContent = `${event.processed_images} / ${event.total_images}`;

    resultsByPath.set(event.path, event);
    addResultRow(event);
    showResult(event.path); // live-follow the most recent image as it arrives
  }

  if (event.type === "finished") {
    currentPath.textContent = "done. click any row below to review it.";
    currentVerdict.textContent = "";
    currentScores.textContent = "";
    currentReason.textContent = "";
  }
}

// Renders one previously-received result into the "current image" panel and
// replays its real per-step neuron activity. Used both for the live-following
// display and for reviewing an earlier image by clicking its row.
function showResult(path) {
  const event = resultsByPath.get(path);
  if (!event) return;
  selectedPath = path;

  const filename = path.split("/").pop();
  currentPath.textContent = filename;
  currentVerdict.textContent = event.verdict.toUpperCase();
  currentVerdict.className = `current-verdict ${event.verdict}`;

  if (event.fed_to_connectome) {
    currentScores.textContent =
      `confidence=${event.confidence_signal.toExponential(3)} (drives verdict)  ` +
      `defect_score=${event.defect_score.toExponential(3)} (reported only)` +
      (event.calibrated ? "" : "  (UNCALIBRATED thresholds)");
    currentReason.textContent = "";
  } else {
    currentScores.textContent = "not fed to connectome -- precheck failed";
    currentReason.textContent = event.precheck_findings.join("; ");
  }

  currentThumb.src = imageUrl(path);
  currentThumb.hidden = false;
  if (flyDesk) flyDesk.setScreenImage(imageUrl(path));

  if (event.step_activity) {
    brain.playSequence(event.step_activity, event.verdict);
  }

  highlightSelectedRow();
}

function addResultRow(event) {
  const filename = event.path.split("/").pop();
  const score = event.fed_to_connectome ? event.defect_score.toExponential(3) : "n/a";
  const row = document.createElement("tr");
  row.dataset.path = event.path;
  row.innerHTML = `
    <td><button type="button" class="image-link">${escapeHtml(filename)}</button></td>
    <td>${score}</td>
    <td class="status-${event.verdict}">${event.verdict}</td>
  `;
  row.querySelector(".image-link").addEventListener("click", () => showResult(event.path));
  resultsBody.prepend(row);
  while (resultsBody.children.length > MAX_ROWS) {
    resultsBody.removeChild(resultsBody.lastChild);
  }
}

function highlightSelectedRow() {
  for (const row of resultsBody.children) {
    row.classList.toggle("selected", row.dataset.path === selectedPath);
  }
}

function imageUrl(path) {
  return `/api/image?${new URLSearchParams({ path }).toString()}`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

connect();
