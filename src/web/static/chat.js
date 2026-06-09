/* Chat page — file upload, SSE streaming, progress tracking */

let runId = null;
let eventSource = null;

const chatView = document.getElementById("chatView");
const progressView = document.getElementById("progressView");
const chatContainer = document.getElementById("chatContainer");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const uploadFilename = document.getElementById("uploadFilename");
const descUploadArea = document.getElementById("descUploadArea");
const descFileInput = document.getElementById("descFileInput");
const descFilename = document.getElementById("descFilename");
const uploadStatus = document.getElementById("uploadStatus");
const uploadStatusFile = document.getElementById("uploadStatusFile");
const btnResults = document.getElementById("btnResults");

let pendingDescFile = null;

/* --- CSV File Upload --- */

uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});
uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

/* --- Description File Upload (optional) --- */

descUploadArea.addEventListener("click", () => descFileInput.click());
descUploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    descUploadArea.classList.add("dragover");
});
descUploadArea.addEventListener("dragleave", () => {
    descUploadArea.classList.remove("dragover");
});
descUploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    descUploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleDescFile(e.dataTransfer.files[0]);
    }
});
descFileInput.addEventListener("change", () => {
    if (descFileInput.files.length > 0) {
        handleDescFile(descFileInput.files[0]);
    }
});

function handleDescFile(file) {
    pendingDescFile = file;
    descFilename.textContent = file.name;
    descUploadArea.classList.add("uploaded");
}

async function handleFile(file) {
    if (!file.name.endsWith(".csv")) {
        alert("Please upload a CSV file.");
        return;
    }

    uploadFilename.textContent = file.name;
    uploadArea.classList.add("uploaded");

    const formData = new FormData();
    formData.append("file", file);
    if (pendingDescFile) {
        formData.append("description_file", pendingDescFile);
    }

    const resp = await fetch("/upload", { method: "POST", body: formData });
    const data = await resp.json();

    if (data.error) {
        alert("Upload failed: " + data.error);
        return;
    }

    runId = data.run_id;
    uploadStatus.style.display = "block";
    uploadStatusFile.textContent = file.name;
    setStep("stepUpload", "done");
    setStep("stepIntake", "active");

    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.focus();

    var msg = "File uploaded.";
    if (data.has_description) {
        msg += " Problem description loaded from " + data.description_filename + ".";
        msg += " I'll review it and ask any clarifying questions.";
    } else {
        msg += " Describe your optimization problem, or just say 'go' and I'll start by inspecting the data.";
    }
    addMessage("agent", msg);
}

/* --- Chat --- */

sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || !runId) return;

    addMessage("user", text);
    chatInput.value = "";
    chatInput.disabled = true;
    sendBtn.disabled = true;

    const resp = await fetch("/chat/respond", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId, message: text }),
    });

    if (!resp.ok) {
        addMessage("agent", "Error sending message. Please try again.");
        chatInput.disabled = false;
        sendBtn.disabled = false;
        return;
    }

    showTypingIndicator();
    startSSE();
}

/* --- Typing Indicator --- */

function showTypingIndicator() {
    removeTypingIndicator();
    const div = document.createElement("div");
    div.className = "message agent typing-indicator";
    div.id = "typingIndicator";
    div.innerHTML = '<span style="font-size:24px;color:#58a6ff;">&#9679; &#9679; &#9679;</span>';
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
}

/* --- SSE --- */

function startSSE() {
    if (eventSource) eventSource.close();

    eventSource = new EventSource("/chat/stream?run_id=" + runId);

    eventSource.addEventListener("message", (e) => {
        const data = JSON.parse(e.data);
        /* Only remove indicator when we have actual content to show */
        if (data.type === "message" || data.type === "config_ready" || data.type === "error") {
            removeTypingIndicator();
        }
        handleEvent(data);
        showTypingIndicator();
    });

    eventSource.addEventListener("done", (e) => {
        removeTypingIndicator();
        eventSource.close();
        eventSource = null;
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    });

    eventSource.addEventListener("waiting", (e) => {
        removeTypingIndicator();
        eventSource.close();
        eventSource = null;
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    });

    eventSource.addEventListener("error", () => {
        removeTypingIndicator();
        eventSource.close();
        eventSource = null;
        chatInput.disabled = false;
        sendBtn.disabled = false;
    });
}

/* --- Progress-mode SSE (after approval) --- */

function startProgressSSE() {
    if (eventSource) eventSource.close();

    eventSource = new EventSource("/chat/stream?run_id=" + runId);

    eventSource.addEventListener("message", (e) => {
        const data = JSON.parse(e.data);
        handleProgressEvent(data);
    });

    eventSource.addEventListener("done", () => {
        eventSource.close();
        eventSource = null;
        document.getElementById("headerStatus").textContent = "Complete";
        document.getElementById("progressBar").style.width = "100%";
        document.getElementById("progressLabel").textContent = "Pipeline complete";
        setStep("stepDone", "done");
        btnResults.href = "/run/" + runId;
        btnResults.classList.add("visible");
    });

    eventSource.addEventListener("error", () => {
        eventSource.close();
        eventSource = null;
    });
}

function handleProgressEvent(event) {
    switch (event.type) {
        case "message":
            /* Ignore long agent text — only show short status updates */
            break;
        case "cleaning_start":
            setStep("stepCleaning", "active");
            activateStage("stageCleaning", "cleaningIcon");
            updateProgress(10, "Cleaning data...");
            break;
        case "cleaning_done":
            setStep("stepCleaning", "done");
            completeStage("stageCleaning", "cleaningIcon");
            setStageSummary("cleaning", "Data cleaned and saved.");
            updateProgress(25, "Data cleaning complete");
            break;
        case "stage_summary":
            /* Summaries are saved to files for the explanation agent — don't render them */
            break;
        case "modeling_start":
            setStep("stepModeling", "active");
            activateStage("stageModeling", "modelingIcon");
            updateProgress(30, "Modeling agent working...");
            break;
        case "modeling_done":
            setStep("stepModeling", "done");
            completeStage("stageModeling", "modelingIcon");
            setStageSummary("modeling", "Prediction and optimization complete.");
            updateProgress(70, "Modeling complete");
            break;
        case "explanation_start":
            setStep("stepExplanation", "active");
            activateStage("stageExplanation", "explanationIcon");
            updateProgress(75, "Generating final report...");
            break;
        case "explanation_done":
            setStep("stepExplanation", "done");
            completeStage("stageExplanation", "explanationIcon");
            setStageSummary("explanation", "Report ready.");
            updateProgress(100, "Report ready");
            break;
        case "error":
            document.getElementById("progressLabel").textContent = "Error: " + event.message;
            document.getElementById("headerStatus").textContent = "Error";
            break;
    }
}

function updateProgress(pct, label) {
    document.getElementById("progressBar").style.width = pct + "%";
    document.getElementById("progressLabel").textContent = label;
}

function activateStage(stageId, iconId) {
    const card = document.getElementById(stageId);
    card.classList.remove("pending");
    card.classList.add("active");
    document.getElementById(iconId).innerHTML = '<span class="spinner"></span>';
}

function completeStage(stageId, iconId) {
    const card = document.getElementById(stageId);
    card.classList.remove("active");
    card.classList.add("completed");
    document.getElementById(iconId).innerHTML = "&#10003;";
    document.getElementById(iconId).classList.add("done-icon");
}

function setStageSummary(stage, summary) {
    const map = {
        "cleaning": "cleaningSummary",
        "modeling": "modelingSummary",
        "explanation": "explanationSummary",
    };
    const el = document.getElementById(map[stage]);
    if (el) el.textContent = summary;
}

/* --- Event Handling (chat phase) --- */

function handleEvent(event) {
    switch (event.type) {
        case "message":
            addMessage("agent", event.text);
            break;
        case "tool_call":
            /* Hidden — typing indicator handles this */
            break;
        case "tool_result":
            /* Hidden */
            break;
        case "config_ready":
            addConfigCard(event.config);
            break;
        case "error":
            addMessage("agent", "Error: " + event.message);
            break;
    }
}

/* --- DOM Helpers --- */

function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = "message " + role;
    div.textContent = text;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addConfigCard(config) {
    const div = document.createElement("div");
    div.className = "config-card";
    div.innerHTML =
        '<h3>&gt; ProblemConfig saved</h3>' +
        '<p>The config has been validated and saved. Review the summary above.</p>' +
        '<button onclick="approveAndRun()">Approve &amp; Begin Optimization &#8594;</button>';
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function approveAndRun() {
    /* Switch to progress view */
    chatView.style.display = "none";
    progressView.style.display = "flex";
    setStep("stepIntake", "done");

    const resp = await fetch("/run/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId }),
    });

    if (resp.ok) {
        startProgressSSE();
    }
}

function setStep(stepId, state) {
    var el = document.getElementById(stepId);
    if (el) el.className = "pipeline-step " + state;
}

document.getElementById("uploadCaretBar").addEventListener("click", function() {
    var row = document.getElementById("uploadRow");
    var caret = document.getElementById("uploadCaret");
    if (row.style.display === "none") {
        row.style.display = "flex";
        caret.innerHTML = "&#9660;";
    } else {
        row.style.display = "none";
        caret.innerHTML = "&#9654;";
    }
});

