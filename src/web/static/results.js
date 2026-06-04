/* Results page — loads and renders run data */

function toggleSection(sectionId) {
    var section = document.getElementById(sectionId);
    var body = section.querySelector(".section-body");
    var toggle = section.querySelector(".toggle");
    body.classList.toggle("collapsed");
    toggle.textContent = body.classList.contains("collapsed") ? "\u25B6" : "\u25BC";
}

async function loadResults(runId) {
    var resp = await fetch("/run/" + runId + "/data");
    if (!resp.ok) {
        document.getElementById("configJson").textContent = "Failed to load results.";
        return;
    }

    var data = await resp.json();
    renderConfig(data.config);
    renderDataPreview(data.manifest, data.data_preview);
    renderActivityLog(data.activity_log, data.trace);
}

function renderConfig(config) {
    var el = document.getElementById("configJson");
    if (config) {
        el.textContent = JSON.stringify(config, null, 2);
    } else {
        el.textContent = "No config found.";
    }
}

function renderDataPreview(manifest, preview) {
    var statsEl = document.getElementById("dataStats");
    var headEl = document.getElementById("dataTableHead");
    var bodyEl = document.getElementById("dataTableBody");

    if (!manifest) {
        statsEl.innerHTML = "<p>No data available.</p>";
        return;
    }

    var statsHtml =
        '<div class="stat-card"><div class="stat-label">Rows</div><div class="stat-value">' +
        manifest.row_count.toLocaleString() + '</div></div>' +
        '<div class="stat-card"><div class="stat-label">Columns</div><div class="stat-value">' +
        manifest.columns.length + '</div></div>';

    var nullEntries = Object.entries(manifest.null_counts).filter(function(e) { return e[1] > 0; });
    for (var i = 0; i < nullEntries.length; i++) {
        statsHtml += '<div class="stat-card"><div class="stat-label">Nulls: ' +
            nullEntries[i][0] + '</div><div class="stat-value">' +
            nullEntries[i][1] + '</div></div>';
    }
    statsEl.innerHTML = statsHtml;

    if (preview && preview.length > 0) {
        var columns = Object.keys(preview[0]);
        headEl.innerHTML = "<tr>" + columns.map(function(c) { return "<th>" + c + "</th>"; }).join("") + "</tr>";
        bodyEl.innerHTML = preview.map(function(row) {
            return "<tr>" + columns.map(function(c) {
                var val = row[c];
                return "<td>" + (val != null ? val : "") + "</td>";
            }).join("") + "</tr>";
        }).join("");
    }
}

function renderActivityLog(activityLog, trace) {
    var el = document.getElementById("activityLog");

    if (!activityLog || activityLog.length === 0) {
        el.innerHTML = '<p style="color: var(--text-muted)">No activity recorded.</p>';
        return;
    }

    var html = "";

    for (var i = 0; i < activityLog.length; i++) {
        var entry = activityLog[i];
        var time = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : "";
        var agent = entry.agent || "";

        if (entry.type === "message") {
            html += '<div class="activity-entry message-entry">' +
                '<span class="entry-agent">' + agent + '</span>' +
                '<span class="entry-time">' + time + '</span>' +
                '<div class="entry-content expandable" onclick="this.classList.toggle(\'expanded\')">' +
                escapeHtml(entry.text) + '</div></div>';
        } else if (entry.type === "tool_call") {
            var argsStr = JSON.stringify(entry.args || {});
            html += '<div class="activity-entry tool-entry">' +
                '<span class="entry-agent">' + agent + '</span>' +
                '<span class="entry-time">' + time + '</span>' +
                '<div class="entry-content">' + entry.tool + '(' + escapeHtml(argsStr) + ')</div></div>';
        } else if (entry.type === "config_ready") {
            html += '<div class="activity-entry message-entry">' +
                '<span class="entry-agent">system</span>' +
                '<span class="entry-time">' + time + '</span>' +
                '<div class="entry-content">ProblemConfig produced</div></div>';
        } else if (entry.type === "cleaning_start") {
            html += '<div class="activity-entry message-entry">' +
                '<span class="entry-agent">system</span>' +
                '<span class="entry-time">' + time + '</span>' +
                '<div class="entry-content">Data cleaning started</div></div>';
        } else if (entry.type === "cleaning_done") {
            html += '<div class="activity-entry message-entry">' +
                '<span class="entry-agent">system</span>' +
                '<span class="entry-time">' + time + '</span>' +
                '<div class="entry-content">Data cleaning complete</div></div>';
        } else if (entry.type === "error") {
            html += '<div class="activity-entry error-entry">' +
                '<span class="entry-agent">error</span>' +
                '<span class="entry-time">' + time + '</span>' +
                '<div class="entry-content">' + escapeHtml(entry.message || "") + '</div></div>';
        }
    }

    if (trace && trace.steps) {
        var totalDuration = 0;
        for (var j = 0; j < trace.steps.length; j++) {
            totalDuration += (trace.steps[j].duration_s || 0);
        }
        html += '<div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border);">' +
            '<div class="stats-grid">';
        for (var k = 0; k < trace.steps.length; k++) {
            html += '<div class="stat-card"><div class="stat-label">' + trace.steps[k].agent +
                ' duration</div><div class="stat-value">' + trace.steps[k].duration_s + 's</div></div>';
        }
        html += '<div class="stat-card"><div class="stat-label">Total</div><div class="stat-value">' +
            totalDuration.toFixed(1) + 's</div></div></div></div>';
    }

    el.innerHTML = html;
}

function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
