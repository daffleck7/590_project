/* Chat page — file upload, SSE streaming, message rendering */

let runId = null;
let eventSource = null;

const chatContainer = document.getElementById("chatContainer");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const uploadFilename = document.getElementById("uploadFilename");
const uploadStatus = document.getElementById("uploadStatus");
const uploadStatusFile = document.getElementById("uploadStatusFile");
const btnResults = document.getElementById("btnResults");

/* --- File Upload --- */

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

async function handleFile(file) {
    if (!file.name.endsWith(".csv")) {
        alert("Please upload a CSV file.");
        return;
    }

    uploadFilename.textContent = file.name;
    uploadArea.classList.add("uploaded");

    const formData = new FormData();
    formData.append("file", file);

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

    addMessage("agent", "File uploaded. Describe your optimization problem, or just say 'go' and I'll start by inspecting the data.");
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

    startSSE();
}

function startSSE() {
    if (eventSource) eventSource.close();

    eventSource = new EventSource("/chat/stream?run_id=" + runId);

    eventSource.addEventListener("message", (e) => {
        const data = JSON.parse(e.data);
        handleEvent(data);
    });

    eventSource.addEventListener("done", (e) => {
        eventSource.close();
        eventSource = null;
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    });

    eventSource.addEventListener("error", () => {
        eventSource.close();
        eventSource = null;
        chatInput.disabled = false;
        sendBtn.disabled = false;
    });
}

function handleEvent(event) {
    switch (event.type) {
        case "message":
            addMessage("agent", event.text);
            break;
        case "tool_call":
            addToolCall(event.tool, event.args);
            break;
        case "tool_result":
            updateToolResult(event.tool, event.result);
            break;
        case "config_ready":
            addConfigCard(event.config);
            break;
        case "cleaning_start":
            setStep("stepIntake", "done");
            setStep("stepCleaning", "active");
            addMessage("agent", "Starting data cleaning...");
            break;
        case "cleaning_done":
            setStep("stepCleaning", "done");
            setStep("stepDone", "done");
            btnResults.href = "/run/" + runId;
            btnResults.classList.add("visible");
            addMessage("agent", "Pipeline complete! Click 'View Results' to see the output.");
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

function addToolCall(toolName, args) {
    const div = document.createElement("div");
    div.className = "message tool-call";
    div.id = "tool-" + Date.now();
    const argsStr = Object.entries(args || {})
        .map(function(entry) { return entry[0] + "=" + JSON.stringify(entry[1]); })
        .join(", ");
    div.innerHTML = '<span class="tool-name">' + toolName + '</span>(' + argsStr + ')<div class="tool-result"></div>';
    div.addEventListener("click", function() { div.classList.toggle("expanded"); });
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateToolResult(toolName, result) {
    const toolCalls = document.querySelectorAll(".message.tool-call");
    for (let i = toolCalls.length - 1; i >= 0; i--) {
        const tc = toolCalls[i];
        if (tc.querySelector(".tool-name").textContent === toolName) {
            const resultDiv = tc.querySelector(".tool-result");
            resultDiv.textContent = result;
            break;
        }
    }
}

function addConfigCard(config) {
    const div = document.createElement("div");
    div.className = "config-card";
    div.innerHTML =
        '<h3>&gt; ProblemConfig</h3>' +
        '<pre>' + JSON.stringify(config, null, 2) + '</pre>' +
        '<button onclick="proceedToCleaning()">Proceed to Data Cleaning &rarr;</button>';
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function proceedToCleaning() {
    const resp = await fetch("/run/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId }),
    });
    if (resp.ok) {
        startSSE();
    }
}

function setStep(stepId, state) {
    var el = document.getElementById(stepId);
    el.className = "pipeline-step " + state;
}
