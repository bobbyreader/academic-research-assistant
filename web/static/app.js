const stageNames = {
  search: "真实检索",
  lit_review: "证据分析",
  writing: "研究写作",
  export: "整理下载",
};

const stateNames = {
  pending: "待开始",
  in_progress: "进行中",
  completed: "已完成",
  failed: "需处理",
  blocked: "需处理",
};

const form = document.querySelector("#research-form");
const startButton = document.querySelector("#start-button");
const message = document.querySelector("#form-message");
const resultsPanel = document.querySelector("#results-panel");
const recentList = document.querySelector("#recent-list");
const statusPill = document.querySelector("#status-pill");
const progressFooter = document.querySelector("#progress-footer");
const progressTitle = document.querySelector("#progress-title-text");
const wizardPages = [...document.querySelectorAll("[data-step-page]")];
const wizardSteps = [...document.querySelectorAll("[data-step-target]")];
let activeStep = 1;

function setMessage(text, type = "") {
  message.textContent = text;
  message.className = `form-message ${type}`.trim();
}

function selectedValues(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((input) => input.value);
}

function updateBrief() {
  const topic = document.querySelector("#topic").value.trim();
  const sources = selectedValues("sources");
  const exports = selectedValues("exports");
  const provider = selectedValues("provider")[0] || "codex_cli";
  const providerNames = { codex_cli: "本机 Codex", gemini: "Gemini", openai_compatible: "兼容 API" };
  document.querySelector("#brief-topic").textContent = topic || "尚未填写";
  document.querySelector("#brief-sources").textContent = sources.length ? `${sources.length} 个已选来源` : "尚未选择";
  document.querySelector("#brief-exports").textContent = exports.length ? `${exports.length} 种输出格式` : "尚未选择";
  document.querySelector("#brief-provider").textContent = providerNames[provider];
  document.querySelector("#launch-summary").textContent = topic && sources.length && exports.length
    ? `“${topic}” · ${sources.length} 个文献来源 · ${exports.length} 种交付文件 · ${providerNames[provider]}`
    : "请先完成主题、资料来源和交付文件的选择。";
}

function updateChoiceCards() {
  document.querySelectorAll(".select-card input").forEach((input) => {
    input.closest(".select-card").classList.toggle("selected", input.checked);
  });
  updateBrief();
}

function focusStep(step) {
  const page = document.querySelector(`[data-step-page="${step}"]`);
  const heading = page.querySelector("h3");
  if (heading) heading.focus({ preventScroll: true });
}

function showStep(step, shouldFocus = false) {
  activeStep = step;
  wizardPages.forEach((page) => {
    const active = Number(page.dataset.stepPage) === step;
    page.hidden = !active;
    page.classList.toggle("is-active", active);
  });
  wizardSteps.forEach((button) => {
    const active = Number(button.dataset.stepTarget) === step;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  setMessage("");
  if (shouldFocus) focusStep(step);
}

function validateStep(step) {
  if (step === 1) {
    const topic = document.querySelector("#topic");
    if (topic.value.trim().length < 3) {
      setMessage("请先输入至少 3 个字符的研究主题。 ");
      topic.focus();
      return false;
    }
  }
  if (step === 2 && !selectedValues("sources").length) {
    setMessage("请至少选择一个文献来源。 ");
    document.querySelector("input[name='sources']").focus();
    return false;
  }
  if (step === 3 && !selectedValues("exports").length) {
    setMessage("请至少选择一种导出格式。 ");
    document.querySelector("input[name='exports']").focus();
    return false;
  }
  return true;
}

function requestStep(target) {
  if (target > activeStep) {
    for (let step = activeStep; step < target; step += 1) {
      if (!validateStep(step)) return;
    }
  }
  showStep(target, true);
}

function setStatus(status) {
  const labels = { queued: "排队中", running: "进行中", completed: "可以下载", failed: "任务失败" };
  const className = status === "completed" ? "done" : status === "failed" ? "error" : status === "running" ? "running" : "idle";
  statusPill.className = `status-pill ${className}`;
  statusPill.textContent = labels[status] || "等待开始";
}

function updateProgress(snapshot) {
  setStatus(snapshot.status);
  document.querySelectorAll("[data-stage]").forEach((item) => {
    const stage = snapshot.stages.find((entry) => entry.id === item.dataset.stage);
    if (!stage) return;
    item.dataset.status = stage.status;
    item.querySelector(".stage-state").textContent = stateNames[stage.status] || stage.status;
  });
  if (snapshot.status === "queued") {
    progressFooter.textContent = "任务已创建，正在准备研究引擎。";
    progressTitle.textContent = "正在为这次研究准备文献检索。";
  } else if (snapshot.status === "running") {
    progressFooter.textContent = `当前阶段：${stageNames[snapshot.current_stage] || "研究处理中"}`;
    progressTitle.textContent = "研究正在进行，完成后会自动显示下载文件。";
  } else if (snapshot.status === "completed") {
    progressFooter.textContent = "研究已完成，结果已保存到本机。";
    progressTitle.textContent = "研究已完成。你可以下载结果或在最近项目中再次查看。";
  } else if (snapshot.status === "failed") {
    progressFooter.textContent = snapshot.error || "任务没有完成，请检查设置后重试。";
    progressTitle.textContent = "任务暂停了。修改设置后可以再次启动。";
  }
}

function showResults(snapshot) {
  document.querySelector("#result-project").textContent = snapshot.project_name;
  document.querySelector("#result-summary-text").textContent = snapshot.warnings.length
    ? "研究草稿已生成，部分格式需要留意。"
    : "研究草稿、证据分析和你选择的文件格式已经整理完成。";
  const downloadList = document.querySelector("#download-list");
  downloadList.replaceChildren();
  snapshot.artifacts.forEach((artifact) => {
    const link = document.createElement("a");
    link.className = "download-link";
    link.href = artifact.url;
    link.textContent = `下载 ${artifact.label}`;
    link.setAttribute("download", "");
    downloadList.append(link);
  });
  const warningBox = document.querySelector("#warning-box");
  const warningList = document.querySelector("#warning-list");
  warningList.replaceChildren();
  snapshot.warnings.forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = warning;
    warningList.append(item);
  });
  warningBox.classList.toggle("hidden", snapshot.warnings.length === 0);
  resultsPanel.classList.remove("hidden");
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function pollJob(jobId) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const snapshot = await response.json();
    if (!response.ok) throw new Error(snapshot.error || "无法读取任务状态");
    updateProgress(snapshot);
    if (snapshot.status === "completed") {
      showResults(snapshot);
      setMessage("研究已完成，可以在下方下载结果。", "success");
      return;
    }
    if (snapshot.status === "failed") throw new Error(snapshot.error || "研究任务失败");
    await sleep(1200);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (![1, 2, 3].every((step) => validateStep(step))) return;
  const topic = document.querySelector("#topic").value.trim();
  const sources = selectedValues("sources");
  const exports = selectedValues("exports");
  const data = new FormData(form);
  data.delete("sources");
  data.delete("exports");
  sources.forEach((source) => data.append("sources", source));
  exports.forEach((format) => data.append("exports", format));
  startButton.disabled = true;
  startButton.querySelector("span").textContent = "研究进行中…";
  resultsPanel.classList.add("hidden");
  setMessage("");
  try {
    const response = await fetch("/api/research", { method: "POST", body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "无法创建研究任务");
    setMessage(`已创建项目“${result.project_name}”，研究引擎正在工作。`, "success");
    setStatus("queued");
    await pollJob(result.job_id);
    await loadProjects();
  } catch (error) {
    setStatus("failed");
    setMessage(error.message || "发生未知错误，请稍后重试。");
  } finally {
    startButton.disabled = false;
    startButton.querySelector("span").textContent = "开始研究";
  }
});

async function loadProjects() {
  const response = await fetch("/api/projects");
  if (!response.ok) return;
  const payload = await response.json();
  recentList.replaceChildren();
  if (!payload.projects.length) {
    const empty = document.createElement("div");
    empty.className = "empty-projects";
    empty.textContent = "还没有项目。完成第一次研究后，它会出现在这里。";
    recentList.append(empty);
    return;
  }
  payload.projects.slice(0, 6).forEach((project) => {
    const card = document.createElement("article");
    card.className = "project-card";
    const title = document.createElement("strong");
    title.textContent = project.name;
    const topic = document.createElement("p");
    topic.textContent = project.topic || "尚未记录研究主题";
    const meta = document.createElement("div");
    meta.className = "project-meta";
    const status = document.createElement("span");
    const exportStatus = project.stage_status?.export;
    status.textContent = ["completed", "ready_with_author_checks"].includes(exportStatus) ? "已完成" : "处理中";
    const count = document.createElement("span");
    count.textContent = `${project.artifacts.length} 个文件`;
    meta.append(status, count);
    card.append(title, topic, meta);
    recentList.append(card);
  });
}

document.querySelectorAll("[data-next-step]").forEach((button) => {
  button.addEventListener("click", () => requestStep(Number(button.dataset.nextStep)));
});

document.querySelectorAll("[data-prev-step]").forEach((button) => {
  button.addEventListener("click", () => showStep(Number(button.dataset.prevStep), true));
});

wizardSteps.forEach((button) => {
  button.addEventListener("click", () => requestStep(Number(button.dataset.stepTarget)));
});

form.addEventListener("input", updateBrief);
form.addEventListener("change", updateChoiceCards);

document.querySelector("#data-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  document.querySelector("#file-name").textContent = file ? file.name : "未选择文件";
});

document.querySelector("#refresh-projects").addEventListener("click", async () => {
  const button = document.querySelector("#refresh-projects");
  button.disabled = true;
  await loadProjects();
  button.disabled = false;
});

updateChoiceCards();
loadProjects();
