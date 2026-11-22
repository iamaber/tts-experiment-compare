const $ = (id) => document.getElementById(id);
const state = { models: [], languages: {}, tryUrl: null };

async function synthesize(model, voice, text, speed) {
  const response = await fetch("/api/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, voice, text, speed }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  const header = (name) => Number(response.headers.get(name));
  return {
    url: URL.createObjectURL(await response.blob()),
    latency: header("X-Latency-Seconds"),
    cold: response.headers.get("X-Cold") === "1",
    duration: header("X-Audio-Seconds"),
    rtf: header("X-RTF"),
    sampleRate: header("X-Sample-Rate"),
  };
}

function option(value, label, disabled = false) {
  const el = document.createElement("option");
  el.value = value;
  el.textContent = label;
  el.disabled = disabled;
  return el;
}

function modelsFor(language) {
  return state.models.filter((m) => m.languages.includes(language));
}

function voicesFor(model, language) {
  return model.voices.filter((v) => v.language === language);
}

function fillLanguages(select) {
  select.replaceChildren(
    ...Object.entries(state.languages).map(([code, lang]) => option(code, lang.name)),
  );
}

function bindSpeed(input, label) {
  const update = () => (label.textContent = `${Number(input.value).toFixed(2)}x`);
  input.addEventListener("input", update);
  update();
}

function selectTab(name) {
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.hidden = panel.id !== `tab-${name}`;
  });
  if (name === "benchmark") loadBenchmark();
}

function renderTryModels() {
  const language = $("try-language").value;
  const previous = $("try-model").value;
  $("try-model").replaceChildren(
    ...modelsFor(language).map((m) =>
      option(m.id, m.available ? m.name : `${m.name} (not installed)`, !m.available),
    ),
  );
  const keep = modelsFor(language).find((m) => m.id === previous && m.available);
  const first = modelsFor(language).find((m) => m.available);
  if (keep || first) $("try-model").value = (keep || first).id;
  renderTryVoices();
}

function renderTryVoices() {
  const language = $("try-language").value;
  const model = state.models.find((m) => m.id === $("try-model").value);
  if (!model) {
    $("try-voice").replaceChildren();
    $("try-model-info").textContent = "No installed model supports this language.";
    $("try-run").disabled = true;
    return;
  }
  const previousVoice = $("try-voice").value;
  $("try-voice").replaceChildren(...voicesFor(model, language).map((v) => option(v.id, v.name)));
  if (voicesFor(model, language).some((v) => v.id === previousVoice)) $("try-voice").value = previousVoice;
  $("try-speed").disabled = !model.speed_control;
  const notes = [model.description, `License: ${model.license}.`];
  if (!model.speed_control) notes.push("This model has no speed control.");
  if (model.online) notes.push("Needs an internet connection.");
  $("try-model-info").textContent = notes.join(" ");
  $("try-run").disabled = false;
}

async function runTry() {
  const button = $("try-run");
  if (button.disabled) return;
  button.disabled = true;
  button.textContent = "Synthesizing...";
  $("try-error").hidden = true;
  try {
    const result = await synthesize(
      $("try-model").value,
      $("try-voice").value,
      $("try-text").value,
      Number($("try-speed").value),
    );
    if (state.tryUrl) URL.revokeObjectURL(state.tryUrl);
    state.tryUrl = result.url;
    $("try-audio").src = result.url;
    $("try-download").href = result.url;
    $("try-download").download = `${$("try-model").value}-${$("try-voice").value}.wav`;
    const metrics = [
      ["Latency", `${result.latency.toFixed(3)} s`],
      ["Audio length", `${result.duration.toFixed(2)} s`],
      ["Real-time factor", result.rtf.toFixed(3)],
      ["Sample rate", `${result.sampleRate} Hz`],
    ];
    if (result.cold) metrics.push(["Note", "First request for this voice, so latency includes loading the weights"]);
    $("try-metrics").replaceChildren(
      ...metrics.flatMap(([term, value]) => {
        const dt = document.createElement("dt");
        const dd = document.createElement("dd");
        dt.textContent = term;
        dd.textContent = value;
        return [dt, dd];
      }),
    );
    $("try-result").hidden = false;
    $("try-audio").play().catch(() => {});
  } catch (error) {
    $("try-error").textContent = error.message;
    $("try-error").hidden = false;
  } finally {
    button.textContent = "Synthesize";
    button.disabled = false;
  }
}

function renderCompareModels() {
  const language = $("compare-language").value;
  $("compare-models").replaceChildren(
    ...modelsFor(language).map((model) => {
      const row = document.createElement("div");
      row.className = "model-row";
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = model.id;
      box.checked = model.available && !model.online;
      box.disabled = !model.available;
      label.append(box, ` ${model.name}`);
      if (!model.available) label.append(` (not installed: ${model.install_hint})`);
      const voice = document.createElement("select");
      voice.dataset.model = model.id;
      voice.setAttribute("aria-label", `${model.name} voice`);
      voice.replaceChildren(...voicesFor(model, language).map((v) => option(v.id, v.name)));
      voice.disabled = !model.available;
      row.append(label, voice);
      return row;
    }),
  );
}

async function runCompare() {
  const button = $("compare-run");
  const picks = [...$("compare-models").querySelectorAll("input:checked")].map((box) => ({
    model: state.models.find((m) => m.id === box.value),
    voice: $("compare-models").querySelector(`select[data-model="${box.value}"]`).value,
  }));
  if (!picks.length) return;
  button.disabled = true;
  const body = $("compare-table").querySelector("tbody");
  body.replaceChildren();
  $("compare-table").hidden = false;
  for (const pick of picks) {
    const row = body.insertRow();
    const cells = Array.from({ length: 6 }, () => row.insertCell());
    cells[0].textContent = pick.model.name;
    cells[1].textContent = pick.model.voices.find((v) => v.id === pick.voice).name;
    cells[2].textContent = "running...";
    try {
      const speed = pick.model.speed_control ? Number($("compare-speed").value) : 1;
      const result = await synthesize(pick.model.id, pick.voice, $("compare-text").value, speed);
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = result.url;
      cells[2].replaceChildren(audio);
      cells[3].textContent = result.latency.toFixed(3) + (result.cold ? " (includes loading)" : "");
      cells[4].textContent = result.duration.toFixed(2);
      cells[5].textContent = result.rtf.toFixed(3);
    } catch (error) {
      cells[2].textContent = error.message;
      cells[2].className = "error";
    }
  }
  button.disabled = false;
}

async function loadBenchmark() {
  const rows = await (await fetch("/api/benchmark")).json();
  $("benchmark-empty").hidden = rows.length > 0;
  $("benchmark-table").hidden = rows.length === 0;
  const names = Object.fromEntries(state.models.map((m) => [m.id, m.name]));
  $("benchmark-table").querySelector("tbody").replaceChildren(
    ...rows.map((r) => {
      const tr = document.createElement("tr");
      const metric = r.language === "en" ? "WER" : "CER";
      const values = r.error
        ? [names[r.model], "", "", `failed: ${r.error}`, "", "", "", ""]
        : [
            names[r.model],
            state.languages[r.language]?.name ?? r.language,
            r.voice,
            r.first_call_s.toFixed(2),
            r.latency_median_s.toFixed(3),
            r.rtf.toFixed(3),
            r.peak_rss_mb.toFixed(0),
            r.error_rate == null ? "not scored" : `${(r.error_rate * 100).toFixed(1)}% ${metric}`,
          ];
      tr.replaceChildren(
        ...values.map((v) => {
          const td = document.createElement("td");
          td.textContent = v;
          return td;
        }),
      );
      return tr;
    }),
  );
}

async function refreshModels() {
  state.models = await (await fetch("/api/models")).json();
}

async function refreshSystem() {
  const info = await (await fetch("/api/system")).json();
  const loaded = info.loaded_models.length ? info.loaded_models.join(", ") : "none";
  $("system").textContent =
    `Server RAM ${Math.round(info.process_rss_mb)} MB, system RAM ${Math.round(info.ram_used_mb / 1000)} of ` +
    `${Math.round(info.ram_total_mb / 1000)} GB, CPU ${Math.round(info.cpu_percent)}%. Loaded models: ${loaded}.`;
}

async function init() {
  const [languages] = await Promise.all([(await fetch("/api/languages")).json(), refreshModels()]);
  state.languages = languages;
  for (const prefix of ["try", "compare"]) {
    fillLanguages($(`${prefix}-language`));
    bindSpeed($(`${prefix}-speed`), $(`${prefix}-speed-value`));
    $(`${prefix}-text`).value = languages[$(`${prefix}-language`).value].sample;
    $(`${prefix}-language`).addEventListener("change", (event) => {
      $(`${prefix}-text`).value = languages[event.target.value].sample;
      prefix === "try" ? renderTryModels() : renderCompareModels();
    });
  }
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  });
  $("try-model").addEventListener("change", renderTryVoices);
  $("try-run").addEventListener("click", runTry);
  $("try-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) runTry();
  });
  $("compare-run").addEventListener("click", runCompare);
  renderTryModels();
  renderCompareModels();
  refreshSystem();
  setInterval(refreshSystem, 5000);
}

init();
