"use strict";

(() => {
  const HEALTH_INTERVAL_MS = 60_000;
  const HEALTH_TIMEOUT_MS = 6_000;
  const METRIC_REFRESH_MS = 2_000;
  const REQUEST_METRIC_INTERVAL_MS = 100;
  const REPLAY_INTERVAL_MS = 900;
  const MAX_ROUNDS = 4;
  const BRANCH_COUNT = 3;
  const BRANCH_LABELS = Object.freeze(["A", "B", "C"]);
  const SCENARIOS = Object.freeze([
    "continuous_batching",
    "automatic_prefix_cache",
    "kv_preemption",
    "lazy_reservation",
  ]);
  const API_FIELDS = Object.freeze({
    maxTokens: ["max", "tokens"].join("_"),
    tokenCount: ["token", "count"].join("_"),
    tokenIndex: ["token", "index"].join("_"),
    requestId: ["request", "id"].join("_"),
    finishReason: ["finish", "reason"].join("_"),
    historyTruncated: ["history", "truncated"].join("_"),
    retainedHistoryTokens: ["retained", "history", "tokens"].join("_"),
    generatedTokenCount: ["generated", "token", "count"].join("_"),
    maxPromptCharacters: ["max", "prompt", "characters"].join("_"),
    maxNewTokens: ["max", "new", "tokens"].join("_"),
    tokenId: ["token", "id"].join("_"),
    topK: ["top", "k"].join("_"),
    userPerplexity: ["user", "perplexity"].join("_"),
  });
  const STARTERS = Object.freeze({
    space: "A repair robot woke to a signal no one else could hear.",
    forest: "At sunrise, a hidden path appeared beneath the oldest tree.",
    robot: "The smallest machine in the workshop built something no one had designed.",
    mystery: "A silver key waited beside the bakery door, warm from an unknown hand.",
  });
  const configuration = window.MINIGPT_DEMO_CONFIG ?? Object.freeze({ apiBase: "" });
  const apiBase = typeof configuration.apiBase === "string" ? configuration.apiBase : "";

  function requireElement(selector) {
    const element = document.querySelector(selector);
    if (element === null) {
      throw new Error(`Required page element is missing: ${selector}`);
    }
    return element;
  }

  const elements = Object.freeze({
    backendChip: requireElement("#backend-chip"),
    backendState: requireElement("#backend-state"),
    offlineBanner: requireElement("#offline-banner"),
    storyForm: requireElement("#story-form"),
    storyOpening: requireElement("#story-opening"),
    storySeed: requireElement("#story-seed"),
    storyMaxTokens: requireElement("#story-max-tokens"),
    storyStream: requireElement("#story-stream"),
    forgeButton: requireElement("#forge-button"),
    storyStop: requireElement("#story-stop"),
    storyReset: requireElement("#story-reset"),
    storyStatus: requireElement("#story-status"),
    roundBadge: requireElement("#round-badge"),
    storyHistory: requireElement("#story-history"),
    historyWarning: requireElement("#history-warning"),
    copyStory: requireElement("#copy-story"),
    downloadStory: requireElement("#download-story"),
    metricStatus: requireElement("#metric-status"),
    metricTtft: requireElement("#metric-ttft"),
    metricElapsed: requireElement("#metric-elapsed"),
    metricTokens: requireElement("#metric-tokens"),
    metricRate: requireElement("#metric-rate"),
    metricExecutor: requireElement("#metric-executor"),
    metricKv: requireElement("#metric-kv"),
    metricQueue: requireElement("#metric-queue"),
    predictionForm: requireElement("#prediction-form"),
    predictWorld: requireElement("#predict-world"),
    predictTone: requireElement("#predict-tone"),
    predictTheme: requireElement("#predict-theme"),
    predictionText: requireElement("#prediction-text"),
    predictionGuess: requireElement("#prediction-guess"),
    predictionTopK: requireElement("#prediction-top-k"),
    predictNext: requireElement("#predict-next"),
    predictScore: requireElement("#predict-score"),
    predictionStatus: requireElement("#prediction-status"),
    candidateList: requireElement("#candidate-list"),
    guessResult: requireElement("#guess-result"),
    temperatureGrid: requireElement("#temperature-grid"),
    perplexityBadge: requireElement("#perplexity-badge"),
    surprisalChips: requireElement("#surprisal-chips"),
    surprisalTableBody: requireElement("#surprisal-table tbody"),
    replayTitle: requireElement("#replay-title"),
    replaySummary: requireElement("#replay-summary"),
    replayEvidence: requireElement("#replay-evidence"),
    replayReset: requireElement("#replay-reset"),
    replayBack: requireElement("#replay-back"),
    replayPlay: requireElement("#replay-play"),
    replayForward: requireElement("#replay-forward"),
    replayProgress: requireElement("#replay-progress"),
    requestLanes: requireElement("#request-lanes"),
    resourceGrid: requireElement("#resource-grid"),
    eventTitle: requireElement("#event-title"),
    eventExplanation: requireElement("#event-explanation"),
    eventDetail: requireElement("#event-detail"),
    invariantList: requireElement("#invariant-list"),
  });

  const branchCards = Object.freeze(
    Array.from(document.querySelectorAll("[data-branch]")).map((card) => ({
      card,
      status: card.querySelector('[data-role="status"]'),
      text: card.querySelector('[data-role="text"]'),
      seed: card.querySelector('[data-role="seed"]'),
      tokens: card.querySelector('[data-role="tokens"]'),
      finish: card.querySelector('[data-role="finish"]'),
      choose: card.querySelector('[data-role="choose"]'),
    })),
  );

  function emptyBranch() {
    return { text: "", seed: null, tokens: 0, finish: "—", status: "Idle", requestId: "" };
  }

  const state = {
    online: false,
    storyForgeEnabled: false,
    predictionLabEnabled: false,
    streamingEnabled: false,
    modelId: "minigpt-story-forge",
    activeStoryController: null,
    activePredictionController: null,
    stopRequested: false,
    healthTimer: null,
    metricTimer: null,
    requestMetricTimer: null,
    lastHealthCheckAt: 0,
    requestStartedAt: 0,
    firstTokenAt: null,
    generatedTokens: 0,
    storyRound: 0,
    storySegments: [],
    currentOpening: "",
    branches: Array.from({ length: BRANCH_COUNT }, () => emptyBranch()),
    lastDistribution: null,
    scenarioCache: new Map(),
    currentScenario: null,
    replayFrames: [],
    replayIndex: -1,
    replayTimer: null,
  };

  function apiUrl(path) {
    return new URL(path.replace(/^\/+/, ""), `${apiBase}/`).toString();
  }

  function apiFetch(path, options = {}) {
    return fetch(apiUrl(path), {
      ...options,
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
      headers: options.headers ?? {},
    });
  }

  function setStatus(element, message, isError = false) {
    element.textContent = message;
    element.classList.toggle("error", isError);
  }

  function formatDuration(milliseconds) {
    if (!Number.isFinite(milliseconds) || milliseconds < 0) {
      return "—";
    }
    return milliseconds < 1000
      ? `${Math.round(milliseconds)} ms`
      : `${(milliseconds / 1000).toFixed(2)} s`;
  }

  function selectedValue(name) {
    const selected = document.querySelector(`input[name="${name}"]:checked`);
    return selected instanceof HTMLInputElement ? selected.value : "";
  }

  function controlsFromStoryForm() {
    return {
      world: selectedValue("world"),
      tone: selectedValue("tone"),
      theme: selectedValue("theme"),
    };
  }

  function storyText() {
    return [state.currentOpening, ...state.storySegments]
      .map((part) => part.trim())
      .filter((part) => part.length > 0)
      .join("\n\n");
  }

  function setBackendOnline(online, label) {
    state.online = online;
    elements.backendChip.classList.toggle("online", online);
    elements.backendChip.classList.toggle("offline", !online);
    elements.backendChip.classList.toggle("checking", false);
    elements.backendState.textContent = label;
    elements.offlineBanner.hidden = online;
    updateFeatureAvailability();
  }

  function updateFeatureAvailability() {
    const storyBusy = state.activeStoryController !== null;
    const predictionBusy = state.activePredictionController !== null;
    const canForge = state.online && state.storyForgeEnabled && !storyBusy && state.storyRound < MAX_ROUNDS;
    elements.forgeButton.disabled = !canForge;
    elements.storyStop.disabled = !storyBusy;
    elements.storyStream.disabled = !(state.online && state.storyForgeEnabled && state.streamingEnabled);
    if (!state.streamingEnabled) {
      elements.storyStream.checked = false;
    }
    elements.predictNext.disabled = !(state.online && state.predictionLabEnabled && !predictionBusy);
    elements.predictScore.disabled = !(state.online && state.predictionLabEnabled && !predictionBusy);
    for (let index = 0; index < branchCards.length; index += 1) {
      const branch = state.branches[index];
      const selectable =
        !storyBusy &&
        state.storyRound < MAX_ROUNDS &&
        branch.finish !== "—" &&
        branch.finish !== "error" &&
        branch.finish !== "cancelled" &&
        branch.text.trim().length > 0;
      branchCards[index].choose.disabled = !selectable;
    }
  }

  function updateStoryHistory() {
    elements.storyHistory.replaceChildren();
    const parts = [state.currentOpening, ...state.storySegments]
      .map((part) => part.trim())
      .filter((part) => part.length > 0);
    if (parts.length === 0) {
      const item = document.createElement("li");
      item.className = "empty-history";
      item.textContent = "Choose a world and forge the first three paths.";
      elements.storyHistory.append(item);
    } else {
      for (let index = 0; index < parts.length; index += 1) {
        const item = document.createElement("li");
        const label = document.createElement("strong");
        label.textContent = index === 0 ? "Opening" : `Round ${index}`;
        const text = document.createElement("p");
        text.textContent = parts[index];
        item.append(label, text);
        elements.storyHistory.append(item);
      }
    }
    elements.roundBadge.textContent = `Round ${state.storyRound} / ${MAX_ROUNDS}`;
    const hasStory = parts.length > 0;
    elements.copyStory.disabled = !hasStory;
    elements.downloadStory.disabled = !hasStory;
  }

  function resetBranchCards(message = "Generate three paths to continue the story.") {
    state.branches = Array.from({ length: BRANCH_COUNT }, () => emptyBranch());
    for (const view of branchCards) {
      view.card.classList.remove("ready", "selected");
      view.status.textContent = "Idle";
      view.text.textContent = message;
      view.seed.textContent = "—";
      view.tokens.textContent = "0";
      view.finish.textContent = "—";
      view.choose.disabled = true;
    }
  }

  function renderBranch(index) {
    const branch = state.branches[index];
    const view = branchCards[index];
    view.status.textContent = branch.status;
    view.text.textContent = branch.text || "The model has not produced visible text for this path.";
    view.seed.textContent = branch.seed === null ? "—" : String(branch.seed);
    view.tokens.textContent = String(branch.tokens);
    view.finish.textContent = branch.finish;
    view.card.classList.toggle(
      "ready",
      branch.finish !== "—" && branch.finish !== "error" && branch.finish !== "cancelled",
    );
  }

  function beginStoryRequest() {
    state.stopRequested = false;
    state.requestStartedAt = performance.now();
    state.firstTokenAt = null;
    state.generatedTokens = 0;
    resetBranchCards("Waiting for this independent branch…");
    elements.metricStatus.textContent = "Submitting";
    elements.metricTtft.textContent = "—";
    elements.metricElapsed.textContent = "0 ms";
    elements.metricTokens.textContent = "0 tokens";
    elements.metricRate.textContent = "—";
    elements.historyWarning.hidden = true;
    setStatus(elements.storyStatus, "Submitting three bounded scheduler requests…");
    if (state.requestMetricTimer !== null) {
      window.clearInterval(state.requestMetricTimer);
    }
    state.requestMetricTimer = window.setInterval(updateRequestMetrics, REQUEST_METRIC_INTERVAL_MS);
    updateFeatureAvailability();
  }

  function finishStoryRequest() {
    if (state.requestMetricTimer !== null) {
      window.clearInterval(state.requestMetricTimer);
      state.requestMetricTimer = null;
    }
    updateRequestMetrics();
    state.activeStoryController = null;
    updateFeatureAvailability();
  }

  function updateRequestMetrics() {
    if (state.requestStartedAt === 0) {
      return;
    }
    const elapsed = performance.now() - state.requestStartedAt;
    elements.metricElapsed.textContent = formatDuration(elapsed);
    elements.metricTokens.textContent = `${state.generatedTokens} tokens`;
    elements.metricRate.textContent =
      state.generatedTokens > 0 && elapsed > 0
        ? `${((state.generatedTokens * 1000) / elapsed).toFixed(2)} tok/s`
        : "—";
    elements.metricTtft.textContent =
      state.firstTokenAt === null ? "—" : formatDuration(state.firstTokenAt - state.requestStartedAt);
  }

  function serverErrorMessage(documentValue, fallback) {
    if (
      documentValue !== null &&
      typeof documentValue === "object" &&
      documentValue.error !== null &&
      typeof documentValue.error === "object" &&
      typeof documentValue.error.message === "string"
    ) {
      return documentValue.error.message;
    }
    return fallback;
  }

  async function responseError(response) {
    try {
      const documentValue = await response.json();
      return serverErrorMessage(documentValue, `Backend request failed with HTTP ${response.status}.`);
    } catch {
      return `Backend request failed with HTTP ${response.status}.`;
    }
  }

  function showHistoryWarning(retainedTokens) {
    elements.historyWarning.hidden = false;
    elements.historyWarning.textContent = Number.isInteger(retainedTokens)
      ? `The model context kept the control prefix and the most recent ${retainedTokens} story tokens.`
      : "Older story text was left-truncated to fit the model context; the visible story remains intact.";
  }

  // Story request payload is assembled from bounded form controls.
  function storyPayload() {
    const controls = controlsFromStoryForm();
    if (state.storyRound === 0) {
      state.currentOpening = elements.storyOpening.value.trim();
    }
    let opening = storyText();
    if (opening.length === 0) {
      opening = STARTERS[controls.world] ?? STARTERS.space;
      state.currentOpening = opening;
      elements.storyOpening.value = opening;
      updateStoryHistory();
    }
    const payload = {
      ...controls,
      opening,
      branch_count: BRANCH_COUNT,
      stream: state.streamingEnabled && elements.storyStream.checked,
    };
    payload.seed = elements.storySeed.valueAsNumber;
    payload[["max", "tokens"].join("_")] = elements.storyMaxTokens.valueAsNumber;
    return payload;
  }

  function applyBranchDocument(documentValue) {
    if (
      documentValue === null ||
      typeof documentValue !== "object" ||
      !Array.isArray(documentValue.branches) ||
      documentValue.branches.length !== BRANCH_COUNT
    ) {
      throw new Error("Backend returned an invalid Story Forge envelope.");
    }
    if (state.firstTokenAt === null) {
      state.firstTokenAt = performance.now();
    }
    for (const rawBranch of documentValue.branches) {
      if (rawBranch === null || typeof rawBranch !== "object") {
        throw new Error("Backend returned an invalid branch.");
      }
      const branchId = rawBranch.branch_id;
      if (!Number.isInteger(branchId) || branchId < 0 || branchId >= BRANCH_COUNT) {
        throw new Error("Backend returned an invalid branch index.");
      }
      const branch = state.branches[branchId];
      branch.text = typeof rawBranch.text === "string" ? rawBranch.text : "";
      branch.seed = Number.isInteger(rawBranch.seed) ? rawBranch.seed : null;
      const rawTokenCount = rawBranch[API_FIELDS.tokenCount];
      branch.tokens = Number.isInteger(rawTokenCount) ? rawTokenCount : 0;
      const rawFinish = rawBranch[API_FIELDS.finishReason];
      branch.finish = typeof rawFinish === "string" ? rawFinish : "error";
      branch.status = branch.finish === "error" ? "Failed" : "Complete";
      const rawRequestId = rawBranch[API_FIELDS.requestId];
      branch.requestId = typeof rawRequestId === "string" ? rawRequestId : "";
      renderBranch(branchId);
      state.generatedTokens += branch.tokens;
    }
    if (documentValue[API_FIELDS.historyTruncated] === true) {
      showHistoryWarning(documentValue[API_FIELDS.retainedHistoryTokens]);
    }
  }

  function consumeStoryEvent(documentValue) {
    if (documentValue === null || typeof documentValue !== "object") {
      throw new Error("Backend returned malformed Story Forge streaming data.");
    }
    if (documentValue.error !== null && typeof documentValue.error === "object") {
      throw new Error(serverErrorMessage(documentValue, "Story Forge streaming failed."));
    }
    const type = documentValue.type;
    if (typeof type !== "string") {
      throw new Error("Backend returned a Story Forge event without a type.");
    }
    if (type === "done") {
      return;
    }
    const branchId = documentValue.branch_id;
    if (!Number.isInteger(branchId) || branchId < 0 || branchId >= BRANCH_COUNT) {
      throw new Error("Backend returned an invalid branch index.");
    }
    const branch = state.branches[branchId];
    if (type === "branch_started") {
      branch.status = "Running";
      branch.seed = Number.isInteger(documentValue.seed) ? documentValue.seed : null;
      const rawRequestId = documentValue[API_FIELDS.requestId];
      branch.requestId = typeof rawRequestId === "string" ? rawRequestId : "";
      if (documentValue[API_FIELDS.historyTruncated] === true) {
        showHistoryWarning(documentValue[API_FIELDS.retainedHistoryTokens]);
      }
      renderBranch(branchId);
      return;
    }
    if (type === "token") {
      if (state.firstTokenAt === null) {
        state.firstTokenAt = performance.now();
      }
      branch.status = "Streaming";
      branch.text = typeof documentValue.text === "string" ? documentValue.text : branch.text;
      const tokenIndex = documentValue[API_FIELDS.tokenIndex];
      const nextCount = Number.isInteger(tokenIndex) ? tokenIndex + 1 : branch.tokens + 1;
      state.generatedTokens += Math.max(0, nextCount - branch.tokens);
      branch.tokens = nextCount;
      renderBranch(branchId);
      elements.metricStatus.textContent = "Streaming";
      updateRequestMetrics();
      return;
    }
    if (type === "branch_finished") {
      const rawFinish = documentValue[API_FIELDS.finishReason];
      branch.status = rawFinish === "error" ? "Failed" : "Complete";
      branch.finish = typeof rawFinish === "string" ? rawFinish : "error";
      const rawTokenCount = documentValue[API_FIELDS.tokenCount];
      if (Number.isInteger(rawTokenCount)) {
        state.generatedTokens += Math.max(0, rawTokenCount - branch.tokens);
        branch.tokens = rawTokenCount;
      }
      renderBranch(branchId);
      return;
    }
    throw new Error(`Unsupported Story Forge event: ${type}`);
  }

  async function readStorySse(response) {
    if (response.body === null) {
      throw new Error("Streaming is unavailable in this browser.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8", { fatal: true });
    let buffer = "";
    let sentinelSeen = false;
    while (!sentinelSeen) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) {
          continue;
        }
        const payload = line.slice(5).trimStart();
        if (payload === "[DONE]") {
          sentinelSeen = true;
          break;
        }
        let documentValue;
        try {
          documentValue = JSON.parse(payload);
        } catch {
          throw new Error("Backend returned malformed Story Forge streaming data.");
        }
        consumeStoryEvent(documentValue);
      }
      if (done) {
        break;
      }
    }
    if (!sentinelSeen) {
      throw new Error("Story Forge stream ended before the completion sentinel.");
    }
  }

  async function submitStory(event) {
    event.preventDefault();
    if (!state.online || !state.storyForgeEnabled || state.activeStoryController !== null) {
      return;
    }
    if (!elements.storyForm.reportValidity()) {
      return;
    }
    if (state.storyRound >= MAX_ROUNDS) {
      setStatus(elements.storyStatus, "This story has reached the four-round public limit.", true);
      return;
    }
    const payload = storyPayload();
    const controller = new AbortController();
    state.activeStoryController = controller;
    beginStoryRequest();
    try {
      const response = await apiFetch("demo/story/branches", {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      if (payload.stream) {
        await readStorySse(response);
      } else {
        applyBranchDocument(await response.json());
      }
      elements.metricStatus.textContent = "Complete";
      setStatus(elements.storyStatus, "Three paths are ready. Choose one to continue.");
      updateFeatureAvailability();
      await refreshMetrics();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        elements.metricStatus.textContent = "Stopped";
        for (const branch of state.branches) {
          if (branch.finish === "—") {
            branch.status = "Stopped";
            branch.finish = "cancelled";
          }
        }
        branchCards.forEach((_view, index) => renderBranch(index));
        setStatus(
          elements.storyStatus,
          state.stopRequested ? "Generation stopped. Partial paths are not selectable." : "Generation cancelled.",
          !state.stopRequested,
        );
      } else {
        const message = error instanceof Error ? error.message : "Story generation failed.";
        elements.metricStatus.textContent = "Failed";
        setStatus(elements.storyStatus, message, true);
        if (error instanceof TypeError) {
          setBackendOnline(false, "Backend offline");
        }
      }
    } finally {
      finishStoryRequest();
    }
  }

  function chooseBranch(index) {
    const branch = state.branches[index];
    if (branch.text.trim().length === 0 || branch.finish === "error" || branch.finish === "cancelled") {
      return;
    }
    state.storySegments.push(branch.text.trim());
    state.storyRound += 1;
    for (let branchIndex = 0; branchIndex < branchCards.length; branchIndex += 1) {
      branchCards[branchIndex].card.classList.toggle("selected", branchIndex === index);
      branchCards[branchIndex].choose.disabled = true;
    }
    elements.storyOpening.readOnly = true;
    updateStoryHistory();
    if (state.storyRound >= MAX_ROUNDS) {
      elements.metricStatus.textContent = "Story complete";
      setStatus(elements.storyStatus, "Four rounds complete. Copy, download, or start a new story.");
    } else {
      elements.forgeButton.textContent = "Generate next 3 paths";
      setStatus(elements.storyStatus, `Path ${BRANCH_LABELS[index]} chosen. Forge round ${state.storyRound + 1}.`);
    }
    updateFeatureAvailability();
  }

  function resetStory() {
    state.activeStoryController?.abort();
    state.activeStoryController = null;
    state.stopRequested = false;
    state.storyRound = 0;
    state.storySegments = [];
    state.currentOpening = "";
    state.requestStartedAt = 0;
    state.firstTokenAt = null;
    state.generatedTokens = 0;
    elements.storyForm.reset();
    elements.storyOpening.value = "";
    elements.storyOpening.readOnly = false;
    elements.forgeButton.textContent = "Generate 3 paths";
    elements.roundBadge.textContent = `Round 0 / ${MAX_ROUNDS}`;
    elements.historyWarning.hidden = true;
    elements.metricStatus.textContent = "Idle";
    elements.metricTtft.textContent = "—";
    elements.metricElapsed.textContent = "—";
    elements.metricTokens.textContent = "0 tokens";
    elements.metricRate.textContent = "—";
    resetBranchCards();
    updateStoryHistory();
    setStatus(
      elements.storyStatus,
      state.online && state.storyForgeEnabled
        ? "Story Forge is ready."
        : "Story Forge requires the live local model.",
    );
    updateFeatureAvailability();
  }

  async function copyStory() {
    const text = storyText();
    if (text.length === 0) {
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setStatus(elements.storyStatus, "Story copied to the clipboard.");
    } catch {
      setStatus(elements.storyStatus, "Clipboard access was denied by the browser.", true);
    }
  }

  function downloadStory() {
    const text = storyText();
    if (text.length === 0) {
      return;
    }
    const blob = new Blob([`${text}\n`], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "minigpt-story-forge.txt";
    link.click();
    URL.revokeObjectURL(url);
  }

  function predictionPayload() {
    const payload = {
      world: elements.predictWorld.value,
      tone: elements.predictTone.value,
      theme: elements.predictTheme.value,
      text: elements.predictionText.value,
    };
    payload[API_FIELDS.topK] = elements.predictionTopK.valueAsNumber;
    return payload;
  }

  function displayToken(piece) {
    if (piece === " ") {
      return "␠";
    }
    if (piece === "\n") {
      return "↵";
    }
    if (piece === "\t") {
      return "⇥";
    }
    if (piece.length === 0) {
      return "∅";
    }
    return piece.replaceAll("\n", "↵").replaceAll("\t", "⇥");
  }

  function probabilityRow(candidate, compact = false) {
    const row = document.createElement("div");
    row.className = compact ? "mini-row" : "candidate-row";
    const label = document.createElement("span");
    label.className = compact ? "mini-label" : "candidate-label";
    label.textContent = displayToken(candidate.piece);
    label.title = `token ${candidate[API_FIELDS.tokenId]}`;
    const track = document.createElement("span");
    track.className = "probability-track";
    const fill = document.createElement("span");
    fill.className = "probability-fill";
    fill.style.width = `${Math.max(0, Math.min(100, candidate.probability * 100))}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = `${(candidate.probability * 100).toFixed(1)}%`;
    row.append(label, track, value);
    return row;
  }

  function renderCandidates(candidates) {
    elements.candidateList.replaceChildren();
    for (const candidate of candidates) {
      elements.candidateList.append(probabilityRow(candidate));
    }
    const guess = elements.predictionGuess.value.trim().toLocaleLowerCase();
    if (guess.length === 0) {
      elements.guessResult.textContent = "Distribution revealed";
      return;
    }
    const normalized = candidates.map((candidate) => candidate.piece.trim().toLocaleLowerCase());
    const position = normalized.indexOf(guess);
    if (position === 0) {
      elements.guessResult.textContent = "Top-1 guess";
    } else if (position > 0) {
      elements.guessResult.textContent = `Top-${position + 1} guess`;
    } else {
      elements.guessResult.textContent = "Outside shown top-k";
    }
  }

  function softmaxAtTemperature(candidates, temperature) {
    if (candidates.length === 0) {
      return [];
    }
    const scaled = candidates.map((candidate) => candidate.logit / temperature);
    const maximum = Math.max(...scaled);
    const weights = scaled.map((value) => Math.exp(value - maximum));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return candidates.map((candidate, index) => ({
      ...candidate,
      probability: weights[index] / total,
    }));
  }

  function renderTemperatureMicroscope(candidates) {
    elements.temperatureGrid.querySelectorAll("[data-temperature]").forEach((container) => {
      const temperature = Number(container.dataset.temperature);
      const shaped = softmaxAtTemperature(candidates, temperature);
      container.replaceChildren();
      for (const candidate of shaped.slice(0, 5)) {
        container.append(probabilityRow(candidate, true));
      }
      const note = document.createElement("small");
      note.textContent = "Renormalized over the displayed target top-k.";
      container.append(note);
    });
  }

  async function submitPrediction(event) {
    event.preventDefault();
    if (!state.online || !state.predictionLabEnabled || state.activePredictionController !== null) {
      return;
    }
    if (!elements.predictionForm.reportValidity()) {
      return;
    }
    const controller = new AbortController();
    state.activePredictionController = controller;
    updateFeatureAvailability();
    setStatus(elements.predictionStatus, "Inspecting the target model without sampling…");
    try {
      const response = await apiFetch("demo/predict/next", {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(predictionPayload()),
      });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const result = await response.json();
      if (!Array.isArray(result.candidates) || result.candidates.length === 0) {
        throw new Error("Backend returned no prediction candidates.");
      }
      const candidates = result.candidates.filter(
        (candidate) =>
          candidate !== null &&
          typeof candidate === "object" &&
          typeof candidate.piece === "string" &&
          Number.isFinite(candidate.logit) &&
          Number.isFinite(candidate.probability),
      );
      if (candidates.length === 0) {
        throw new Error("Backend returned invalid prediction candidates.");
      }
      state.lastDistribution = candidates;
      renderCandidates(candidates);
      renderTemperatureMicroscope(candidates);
      setStatus(elements.predictionStatus, "Distribution revealed. No request RNG was advanced.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Prediction inspection failed.";
      setStatus(elements.predictionStatus, message, true);
      if (error instanceof TypeError) {
        setBackendOnline(false, "Backend offline");
      }
    } finally {
      state.activePredictionController = null;
      updateFeatureAvailability();
    }
  }

  function renderSurprisal(result) {
    const rawRows = result[["per", "token"].join("_")];
    const rows = Array.isArray(rawRows)
      ? rawRows.filter(
          (entry) =>
            entry !== null &&
            typeof entry === "object" &&
            entry.is_control !== true &&
            typeof entry.piece === "string" &&
            Number.isFinite(entry.surprisal),
        )
      : [];
    elements.surprisalChips.replaceChildren();
    elements.surprisalTableBody.replaceChildren();
    if (rows.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-result";
      empty.textContent = "No user-text surprisal rows were returned.";
      elements.surprisalChips.append(empty);
    }
    const maximum = Math.max(1, ...rows.map((entry) => entry.surprisal));
    rows.forEach((entry, index) => {
      const label = displayToken(entry.piece);
      const chip = document.createElement("span");
      chip.className = ["surprisal", "token"].join("-");
      chip.style.setProperty("--surprise", String(Math.min(100, (entry.surprisal / maximum) * 100)));
      chip.textContent = label;
      chip.title = `NLL ${entry.surprisal.toFixed(3)}`;
      elements.surprisalChips.append(chip);
      const row = document.createElement("tr");
      const number = document.createElement("td");
      number.textContent = String(index + 1);
      const pieceCell = document.createElement("td");
      pieceCell.textContent = label;
      const nll = document.createElement("td");
      nll.textContent = entry.surprisal.toFixed(4);
      const probability = document.createElement("td");
      probability.textContent = `${(Math.exp(-entry.surprisal) * 100).toFixed(2)}%`;
      row.append(number, pieceCell, nll, probability);
      elements.surprisalTableBody.append(row);
    });
    const userPerplexity = result[API_FIELDS.userPerplexity];
    elements.perplexityBadge.textContent = Number.isFinite(userPerplexity)
      ? `User-text PPL ${userPerplexity.toFixed(2)}`
      : "No user-text score";
  }

  async function scorePrediction() {
    if (!state.online || !state.predictionLabEnabled || state.activePredictionController !== null) {
      return;
    }
    if (!elements.predictionForm.reportValidity()) {
      return;
    }
    const controller = new AbortController();
    state.activePredictionController = controller;
    updateFeatureAvailability();
    setStatus(elements.predictionStatus, "Computing token-level likelihood without sampling…");
    try {
      const payload = predictionPayload();
      delete payload[API_FIELDS.topK];
      const response = await apiFetch("demo/predict/score", {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      renderSurprisal(await response.json());
      setStatus(
        elements.predictionStatus,
        "Sequence scored. These are checkpoint likelihoods, not truth or authorship scores.",
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Sequence scoring failed.";
      setStatus(elements.predictionStatus, message, true);
    } finally {
      state.activePredictionController = null;
      updateFeatureAvailability();
    }
  }

  function humanizeKey(key) {
    return key.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function scenarioFrames(documentValue) {
    if (Array.isArray(documentValue.ticks) && documentValue.ticks.length > 0) {
      return documentValue.ticks.map((tick, index) => ({
        kind: "tick",
        title: `Scheduler tick ${index + 1}`,
        detail: tick,
      }));
    }
    const frames = [];
    if (documentValue.kv !== null && typeof documentValue.kv === "object") {
      for (const [key, value] of Object.entries(documentValue.kv)) {
        frames.push({ kind: "resource", title: humanizeKey(key), detail: { [key]: value } });
      }
    }
    if (documentValue.invariants !== null && typeof documentValue.invariants === "object") {
      for (const [key, value] of Object.entries(documentValue.invariants)) {
        frames.push({ kind: "invariant", title: humanizeKey(key), detail: { [key]: value } });
      }
    }
    return frames.length > 0
      ? frames
      : [{ kind: "summary", title: "Recorded summary", detail: documentValue }];
  }

  function scenarioExplanation(scenario, frame) {
    const explanations = {
      continuous_batching:
        "Eligible requests share a decode model call while retaining independent RNG and terminal state.",
      automatic_prefix_cache:
        "Immutable full-prefix blocks can be attached by later requests, avoiding repeated prefill work.",
      kv_preemption:
        "A whole request releases private KV under pressure, then rebuilds cache without sampling before resuming.",
      lazy_reservation:
        "Current protected KV grows before model work while lifetime demand remains bounded by overcommit policy.",
    };
    return `${explanations[scenario] ?? "Recorded evidence event."} Current record: ${frame.title}.`;
  }

  function renderRequestLanes(documentValue) {
    elements.requestLanes.replaceChildren();
    const lanes = Array.isArray(documentValue.request_lanes) ? documentValue.request_lanes : [];
    if (lanes.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-result";
      empty.textContent = "This package records aggregate transitions rather than named request lanes.";
      elements.requestLanes.append(empty);
      return;
    }
    const frameRatio =
      state.replayFrames.length <= 1 ? 1 : Math.max(0, state.replayIndex + 1) / state.replayFrames.length;
    for (const lane of lanes) {
      const row = document.createElement("div");
      row.className = "request-lane";
      const label = document.createElement("span");
      label.className = "lane-label";
      const requestId = lane[API_FIELDS.requestId];
      label.textContent = typeof requestId === "string" ? requestId : "request";
      const track = document.createElement("span");
      track.className = "lane-track";
      const progress = document.createElement("span");
      progress.className = "lane-progress";
      progress.style.width = `${Math.max(4, Math.min(100, frameRatio * 100))}%`;
      track.append(progress);
      const status = document.createElement("span");
      status.className = "lane-state";
      const generated = lane[API_FIELDS.generatedTokenCount];
      status.textContent = `${lane.status ?? "recorded"} · ${Number.isInteger(generated) ? generated : 0} tok`;
      row.append(label, track, status);
      elements.requestLanes.append(row);
    }
  }

  function renderResourceGrid(documentValue) {
    elements.resourceGrid.replaceChildren();
    const kv = documentValue.kv !== null && typeof documentValue.kv === "object" ? documentValue.kv : {};
    const entries = Object.entries(kv);
    if (entries.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-result";
      empty.textContent = "No normalized KV counters are recorded for this scenario.";
      elements.resourceGrid.append(empty);
      return;
    }
    for (const [key, value] of entries) {
      const block = document.createElement("div");
      block.className = "resource-block";
      const activeFrame = state.replayFrames[state.replayIndex];
      if (activeFrame?.detail !== null && typeof activeFrame?.detail === "object" && key in activeFrame.detail) {
        block.classList.add("active");
      } else if (key.includes("shared") || key.includes("prefix")) {
        block.classList.add("shared");
      } else if (key.includes("pressure") || key.includes("overcommit") || key.includes("preemption")) {
        block.classList.add("pressure");
      }
      block.textContent = String(value);
      block.title = `${humanizeKey(key)}: ${String(value)}`;
      elements.resourceGrid.append(block);
    }
  }

  function renderInvariants(documentValue) {
    elements.invariantList.replaceChildren();
    const invariants =
      documentValue.invariants !== null && typeof documentValue.invariants === "object"
        ? documentValue.invariants
        : {};
    for (const [key, value] of Object.entries(invariants)) {
      const item = document.createElement("li");
      item.textContent = `${humanizeKey(key)}: ${Array.isArray(value) ? value.join(" → ") : String(value)}`;
      elements.invariantList.append(item);
    }
  }

  function renderReplayFrame() {
    const documentValue = state.currentScenario;
    if (documentValue === null) {
      return;
    }
    const total = state.replayFrames.length;
    if (total === 0) {
      state.replayIndex = -1;
      elements.replayProgress.textContent = "0 / 0";
      return;
    }
    if (state.replayIndex < 0) {
      state.replayIndex = 0;
    }
    if (state.replayIndex >= total) {
      state.replayIndex = total - 1;
    }
    const frame = state.replayFrames[state.replayIndex];
    elements.eventTitle.textContent = frame.title;
    elements.eventExplanation.textContent = scenarioExplanation(documentValue.scenario_id, frame);
    elements.eventDetail.textContent = JSON.stringify(frame.detail, null, 2);
    elements.replayProgress.textContent = `${state.replayIndex + 1} / ${total}`;
    elements.replayBack.disabled = state.replayIndex <= 0;
    elements.replayForward.disabled = state.replayIndex >= total - 1;
    renderRequestLanes(documentValue);
    renderResourceGrid(documentValue);
  }

  function stopReplay() {
    if (state.replayTimer !== null) {
      window.clearInterval(state.replayTimer);
      state.replayTimer = null;
    }
    elements.replayPlay.textContent = "Play";
  }

  function stepReplay(delta) {
    stopReplay();
    if (state.replayFrames.length === 0) {
      return;
    }
    state.replayIndex = Math.max(
      0,
      Math.min(state.replayFrames.length - 1, state.replayIndex + delta),
    );
    renderReplayFrame();
  }

  function toggleReplay() {
    if (state.replayTimer !== null) {
      stopReplay();
      return;
    }
    if (state.replayFrames.length === 0) {
      return;
    }
    if (state.replayIndex >= state.replayFrames.length - 1) {
      state.replayIndex = 0;
      renderReplayFrame();
    }
    elements.replayPlay.textContent = "Pause";
    state.replayTimer = window.setInterval(() => {
      if (state.replayIndex >= state.replayFrames.length - 1) {
        stopReplay();
        return;
      }
      state.replayIndex += 1;
      renderReplayFrame();
    }, REPLAY_INTERVAL_MS);
  }

  async function loadScenario(name) {
    stopReplay();
    document.querySelectorAll("[data-scenario]").forEach((button) => {
      button.classList.toggle("active", button.dataset.scenario === name);
    });
    try {
      let documentValue = state.scenarioCache.get(name);
      if (documentValue === undefined) {
        const response = await fetch(`./data/${name}.json`, {
          cache: "no-store",
          credentials: "omit",
          referrerPolicy: "no-referrer",
        });
        if (!response.ok) {
          throw new Error(`Scenario asset returned HTTP ${response.status}.`);
        }
        documentValue = await response.json();
        state.scenarioCache.set(name, documentValue);
      }
      state.currentScenario = documentValue;
      state.replayFrames = scenarioFrames(documentValue);
      state.replayIndex = 0;
      elements.replayTitle.textContent = documentValue.title ?? humanizeKey(name);
      elements.replaySummary.textContent = `Claim level: ${documentValue.claim_level ?? "bounded"}. Source commit: ${String(documentValue.source_commit ?? "unknown").slice(0, 12)}.`;
      const sourcePath = String(documentValue.source_evidence_path ?? "docs/results");
      elements.replayEvidence.href = `https://github.com/ericheee111/miniGPT/tree/main/${sourcePath}`;
      renderInvariants(documentValue);
      renderReplayFrame();
    } catch (error) {
      state.currentScenario = null;
      state.replayFrames = [];
      state.replayIndex = -1;
      elements.replayTitle.textContent = "Scenario unavailable";
      elements.replaySummary.textContent =
        error instanceof Error ? error.message : "The recorded scenario could not be loaded.";
      elements.eventTitle.textContent = "Offline asset error";
      elements.eventExplanation.textContent =
        "The live model is not required, but the static JSON asset is missing.";
      elements.eventDetail.textContent = "{}";
      elements.replayProgress.textContent = "0 / 0";
      elements.requestLanes.replaceChildren();
      elements.resourceGrid.replaceChildren();
      elements.invariantList.replaceChildren();
    }
  }

  function applyInfo(info) {
    const features = info.features !== null && typeof info.features === "object" ? info.features : {};
    state.storyForgeEnabled = features.story_forge === true;
    state.predictionLabEnabled = features.prediction_lab === true;
    state.streamingEnabled = info.streaming_enabled === true;
    if (typeof info.model_id === "string") {
      state.modelId = info.model_id;
    }
    if (typeof info.executor === "string") {
      elements.metricExecutor.textContent = info.executor;
    }
    if (typeof info.kv_cache_backend === "string") {
      elements.metricKv.textContent = info.kv_cache_backend;
    }
    if (info.limits !== null && typeof info.limits === "object") {
      const promptLimit = info.limits[API_FIELDS.maxPromptCharacters];
      if (Number.isInteger(promptLimit)) {
        elements.storyOpening.maxLength = Math.min(480, promptLimit);
      }
      const generationLimit = info.limits[API_FIELDS.maxNewTokens];
      if (Number.isInteger(generationLimit)) {
        elements.storyMaxTokens.max = String(Math.min(64, generationLimit));
      }
    }
    updateFeatureAvailability();
  }

  async function refreshMetrics() {
    if (!state.online || apiBase === "") {
      return;
    }
    try {
      const response = await apiFetch("demo/metrics");
      if (!response.ok) {
        return;
      }
      const metrics = await response.json();
      if (Number.isInteger(metrics.queued_requests)) {
        elements.metricQueue.textContent = String(metrics.queued_requests);
      }
    } catch {
      elements.metricQueue.textContent = "—";
    }
  }

  function scheduleMetricRefresh() {
    if (state.metricTimer !== null) {
      window.clearInterval(state.metricTimer);
      state.metricTimer = null;
    }
    if (state.online && !document.hidden) {
      state.metricTimer = window.setInterval(refreshMetrics, METRIC_REFRESH_MS);
    }
  }

  function scheduleHealthCheck() {
    if (apiBase === "" || document.hidden) {
      return;
    }
    if (state.healthTimer !== null) {
      window.clearTimeout(state.healthTimer);
    }
    const elapsed = Date.now() - state.lastHealthCheckAt;
    state.healthTimer = window.setTimeout(checkHealth, Math.max(0, HEALTH_INTERVAL_MS - elapsed));
  }

  async function checkHealth() {
    state.healthTimer = null;
    if (apiBase === "" || document.hidden) {
      return;
    }
    state.lastHealthCheckAt = Date.now();
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
    try {
      const health = await apiFetch("healthz", { signal: controller.signal });
      if (!health.ok) {
        throw new Error("Backend health check failed.");
      }
      const [infoResponse, metricsResponse] = await Promise.all([
        apiFetch("demo/info", { signal: controller.signal }),
        apiFetch("demo/metrics", { signal: controller.signal }),
      ]);
      if (!infoResponse.ok || !metricsResponse.ok) {
        throw new Error("Backend metadata is unavailable.");
      }
      const [info, metrics] = await Promise.all([infoResponse.json(), metricsResponse.json()]);
      applyInfo(info);
      if (Number.isInteger(metrics.queued_requests)) {
        elements.metricQueue.textContent = String(metrics.queued_requests);
      }
      setBackendOnline(true, state.storyForgeEnabled ? "Story model online" : "Legacy model online");
      setStatus(
        elements.storyStatus,
        state.storyForgeEnabled
          ? "Story Forge is ready."
          : "The backend is online, but the Story Forge model is not loaded.",
        !state.storyForgeEnabled,
      );
      setStatus(
        elements.predictionStatus,
        state.predictionLabEnabled
          ? "Prediction Lab is ready."
          : "Prediction Lab requires the Story Forge model.",
      );
    } catch {
      state.storyForgeEnabled = false;
      state.predictionLabEnabled = false;
      state.streamingEnabled = false;
      elements.metricQueue.textContent = "—";
      setBackendOnline(false, "Backend offline");
      setStatus(
        elements.storyStatus,
        "The local model is offline. Recorded Systems Lab scenarios remain available.",
      );
      setStatus(elements.predictionStatus, "Prediction Lab requires the live Story Forge model.");
    } finally {
      window.clearTimeout(timeout);
      scheduleMetricRefresh();
      scheduleHealthCheck();
    }
  }

  elements.storyForm.addEventListener("submit", submitStory);
  elements.storyStop.addEventListener("click", () => {
    state.stopRequested = true;
    state.activeStoryController?.abort();
  });
  elements.storyReset.addEventListener("click", resetStory);
  elements.copyStory.addEventListener("click", copyStory);
  elements.downloadStory.addEventListener("click", downloadStory);
  branchCards.forEach((view, index) => {
    view.choose.addEventListener("click", () => chooseBranch(index));
  });

  elements.predictionForm.addEventListener("submit", submitPrediction);
  elements.predictScore.addEventListener("click", scorePrediction);

  document.querySelectorAll("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => loadScenario(button.dataset.scenario));
  });
  elements.replayReset.addEventListener("click", () => {
    stopReplay();
    state.replayIndex = 0;
    renderReplayFrame();
  });
  elements.replayBack.addEventListener("click", () => stepReplay(-1));
  elements.replayForward.addEventListener("click", () => stepReplay(1));
  elements.replayPlay.addEventListener("click", toggleReplay);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (state.healthTimer !== null) {
        window.clearTimeout(state.healthTimer);
        state.healthTimer = null;
      }
      if (state.metricTimer !== null) {
        window.clearInterval(state.metricTimer);
        state.metricTimer = null;
      }
      stopReplay();
      return;
    }
    scheduleHealthCheck();
    scheduleMetricRefresh();
  });

  resetStory();
  void loadScenario(SCENARIOS[0]);
  if (apiBase === "") {
    setBackendOnline(false, "Offline-only build");
    setStatus(
      elements.storyStatus,
      "No backend URL is configured. The portfolio and recorded Systems Lab remain available.",
    );
  } else {
    state.lastHealthCheckAt = Date.now() - HEALTH_INTERVAL_MS;
    scheduleHealthCheck();
  }
})();
