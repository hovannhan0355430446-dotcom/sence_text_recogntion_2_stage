let currentFile = null;
let currentSampleName = null;
let currentResult = null;
let activeTab = "annotated";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const runBtn = document.getElementById("runBtn");
const sampleList = document.getElementById("sampleList");
const loadingIndicator = document.getElementById("loadingIndicator");

const statCount = document.getElementById("statCount");
const statDetTime = document.getElementById("statDetTime");
const statRecTime = document.getElementById("statRecTime");
const statTotalTime = document.getElementById("statTotalTime");

const resultImage = document.getElementById("resultImage");
const overlayCanvas = document.getElementById("overlayCanvas");
const fullTextArea = document.getElementById("fullTextArea");
const copyBtn = document.getElementById("copyBtn");
const resultTableBody = document.getElementById("resultTableBody");

browseBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        selectFile(e.target.files[0]);
    }
});

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        selectFile(e.dataTransfer.files[0]);
    }
});

function selectFile(file) {
    currentFile = file;
    currentSampleName = null;
    clearSampleSelection();
    runBtn.disabled = false;
    runBtn.textContent = `Chạy nhận diện: ${file.name}`;
}

async function loadSamples() {
    try {
        const resp = await fetch("/api/samples");
        const data = await resp.json();
        sampleList.innerHTML = "";
        data.samples.forEach((sample) => {
            const card = document.createElement("div");
            card.className = "sample-card";
            card.innerHTML = `
                <img src="${sample.thumbnail}" alt="${sample.name}">
                <span>${sample.name}</span>
            `;
            card.addEventListener("click", () => {
                clearSampleSelection();
                card.classList.add("active");
                currentSampleName = sample.name;
                currentFile = null;
                runBtn.disabled = false;
                runBtn.textContent = `Chạy nhận diện: ${sample.name}`;
            });
            sampleList.appendChild(card);
        });
    } catch (err) {
        console.error("Không thể tải danh sách ảnh mẫu:", err);
    }
}

function clearSampleSelection() {
    document.querySelectorAll(".sample-card").forEach((c) => c.classList.remove("active"));
}

runBtn.addEventListener("click", async () => {
    if (!currentFile && !currentSampleName) return;

    const thresh = document.getElementById("thresh").value;
    const boxThresh = document.getElementById("boxThresh").value;
    const unclipRatio = document.getElementById("unclipRatio").value;

    const formData = new FormData();
    formData.append("thresh", thresh);
    formData.append("box_thresh", boxThresh);
    formData.append("unclip_ratio", unclipRatio);

    runBtn.disabled = true;
    loadingIndicator.style.display = "flex";

    try {
        let resp;
        if (currentFile) {
            formData.append("file", currentFile);
            resp = await fetch("/api/predict", { method: "POST", body: formData });
        } else {
            formData.append("sample_name", currentSampleName);
            resp = await fetch("/api/predict_sample", { method: "POST", body: formData });
        }

        if (!resp.ok) {
            const err = await resp.json();
            alert("Lỗi xử lý: " + (err.detail || "Không rõ nguyên nhân"));
            return;
        }

        currentResult = await resp.json();
        renderResult(currentResult);
    } catch (err) {
        alert("Lỗi kết nối máy chủ: " + err.message);
    } finally {
        runBtn.disabled = false;
        loadingIndicator.style.display = "none";
    }
});

function renderResult(data) {
    statCount.textContent = data.items.length;
    statDetTime.textContent = `${data.timings.detection}s`;
    statRecTime.textContent = `${data.timings.recognition}s`;
    statTotalTime.textContent = `${data.timings.total}s`;

    fullTextArea.value = data.full_text || "";

    updateImageView();

    if (data.items.length === 0) {
        resultTableBody.innerHTML = `<tr><td colspan="5" class="empty-cell">Không phát hiện được văn bản trong ảnh.</td></tr>`;
        return;
    }

    let rowsHtml = "";
    data.items.forEach((item) => {
        const detPct = (item.det_score * 100).toFixed(1);
        const recPct = (item.rec_score * 100).toFixed(1);
        rowsHtml += `
            <tr data-id="${item.id}">
                <td>${item.id}</td>
                <td><img class="crop-thumb" src="${item.crop_base64}" alt="crop"></td>
                <td class="text-cell">${item.text}</td>
                <td>${detPct}%</td>
                <td>${recPct}%</td>
            </tr>
        `;
    });
    resultTableBody.innerHTML = rowsHtml;

    document.querySelectorAll("#resultTableBody tr").forEach((row) => {
        row.addEventListener("mouseenter", () => {
            const id = parseInt(row.getAttribute("data-id"));
            highlightBox(id);
        });
        row.addEventListener("mouseleave", () => {
            clearHighlight();
        });
    });
}

function updateImageView() {
    if (!currentResult) return;
    if (activeTab === "annotated") {
        resultImage.src = currentResult.annotated_base64;
    } else {
        resultImage.src = currentResult.original_base64;
    }
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeTab = btn.getAttribute("data-tab");
        updateImageView();
    });
});

function highlightBox(id) {
    if (!currentResult) return;
    const item = currentResult.items.find((x) => x.id === id);
    if (!item) return;

    const img = resultImage;
    if (!img.naturalWidth || !img.naturalHeight) return;

    overlayCanvas.width = img.clientWidth;
    overlayCanvas.height = img.clientHeight;
    overlayCanvas.style.top = `${img.offsetTop}px`;
    overlayCanvas.style.left = `${img.offsetLeft}px`;

    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const scaleX = img.clientWidth / currentResult.image_size.width;
    const scaleY = img.clientHeight / currentResult.image_size.height;

    ctx.strokeStyle = "#ff0055";
    ctx.lineWidth = 3;
    ctx.fillStyle = "rgba(255, 0, 85, 0.25)";

    ctx.beginPath();
    item.polygon.forEach((pt, idx) => {
        const x = pt[0] * scaleX;
        const y = pt[1] * scaleY;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
}

function clearHighlight() {
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

copyBtn.addEventListener("click", () => {
    if (!fullTextArea.value) return;
    navigator.clipboard.writeText(fullTextArea.value).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = "Đã sao chép!";
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 1500);
    });
});

window.addEventListener("DOMContentLoaded", () => {
    loadSamples();
});

