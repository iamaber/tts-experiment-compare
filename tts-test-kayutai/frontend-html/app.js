const API_URL = 'http://localhost:8000';
const CFG_COEF = 2.0;

let selectedVoice = 'default';
let audioContext = null;
let currentAudio = null;
let wordTimings = [];
let isPlaying = false;

const voiceSelect = document.getElementById('voiceSelect');
const voiceUpload = document.getElementById('voiceUpload');
const generateBtn = document.getElementById('generateBtn');
const textInput = document.getElementById('textInput');
const charCount = document.getElementById('charCount');
const audioSection = document.getElementById('audioSection');
const audioPlayer = document.getElementById('audioPlayer');
const playBtn = document.getElementById('playBtn');
const progressBar = document.getElementById('progressBar');
const currentTimeEl = document.getElementById('currentTime');
const durationEl = document.getElementById('duration');
const volumeSlider = document.getElementById('volumeSlider');
const highlightText = document.getElementById('highlightText');
const downloadLink = document.getElementById('downloadLink');

const rtfValue = document.getElementById('rtfValue');
const rtfStatus = document.getElementById('rtfStatus');
const cpuValue = document.getElementById('cpuValue');
const cpuBar = document.getElementById('cpuBar');
const gpuValue = document.getElementById('gpuValue');
const gpuBar = document.getElementById('gpuBar');
const gpuMemValue = document.getElementById('gpuMemValue');
const gpuMemBar = document.getElementById('gpuMemBar');
const genTimeValue = document.getElementById('genTimeValue');

const viewHistoryBtn = document.getElementById('viewHistoryBtn');
const analysisModal = document.getElementById('analysisModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const historyTable = document.getElementById('historyTable').querySelector('tbody');
const exportBtn = document.getElementById('exportBtn');

lucide.createIcons();

async function init() {
    await loadVoices();
    startSystemPolling();
}

async function loadVoices() {
    try {
        const response = await fetch(`${API_URL}/api/voices`);
        const voices = await response.json();
        
        voiceSelect.innerHTML = '';
        voices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice.id;
            option.textContent = `${voice.name} ${voice.is_custom ? '(Custom)' : ''}`;
            voiceSelect.appendChild(option);
        });
        return voices;
    } catch (error) {
        console.error('Failed to load voices:', error);
        return [];
    }
}

voiceSelect.addEventListener('change', (e) => {
    selectedVoice = e.target.value;
});

voiceUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('audio', file);
    const name = file.name.replace(/\.[^/.]+$/, "");

    const knownIds = new Set(Array.from(voiceSelect.options, option => option.value));

    try {
        const res = await fetch(`${API_URL}/api/clone-voice?name=${encodeURIComponent(name)}`, {
            method: 'POST',
            body: formData
        });
        if (res.ok) {
            const voices = await loadVoices();
            const newVoice = voices.find(v => !knownIds.has(String(v.id)))
                ?? voices.find(v => v.is_custom && v.name === name);
            if(newVoice) voiceSelect.value = newVoice.id;
            selectedVoice = newVoice ? newVoice.id : selectedVoice;
            alert('Voice uploaded successfully!');
        }
    } catch (error) {
        alert('Upload failed');
    }
});

textInput.addEventListener('input', (e) => {
    charCount.textContent = e.target.value.length;
});

generateBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) return;

    setLoading(true);
    
    try {
        const response = await fetch(`${API_URL}/api/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text, 
                voice: selectedVoice,
                cfg_coef: CFG_COEF
            })
        });

        if (!response.ok) throw new Error('Generation failed');

        const data = await response.json();
        handleGenerationSuccess(data);
    } catch (error) {
        console.error(error);
        alert('Generation failed. Ensure backend is running.');
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    if (isLoading) {
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i data-lucide="loader-2" class="animate-spin"></i><span>Generating...</span>';
        lucide.createIcons();
    } else {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i data-lucide="play"></i><span>Generate Speech</span>';
        lucide.createIcons();
    }
}

function handleGenerationSuccess(data) {
    currentAudio = new Audio(`${API_URL}${data.audio_url}`);
    audioPlayer.src = `${API_URL}${data.audio_url}`;
    audioSection.classList.remove('hidden');
    downloadLink.href = `${API_URL}${data.audio_url}`;
    downloadLink.download = `tts_${Date.now()}.wav`;

    wordTimings = data.word_timings;
    renderWordHighlighting(data.word_timings);

    updateMetrics(data.metrics);
    updateHistoryTable();

    audioPlayer.play();
    isPlaying = true;
    updatePlayBtn();
}

function renderWordHighlighting(timings) {
    highlightText.innerHTML = '';
    timings.forEach((timing, index) => {
        const span = document.createElement('span');
        span.textContent = timing.word;
        span.className = 'highlight-word';
        span.dataset.start = timing.start;
        span.dataset.end = timing.end;
        span.onclick = () => {
            audioPlayer.currentTime = timing.start;
            audioPlayer.play();
        };
        highlightText.appendChild(span);
    });
}

audioPlayer.addEventListener('timeupdate', () => {
    const current = audioPlayer.currentTime;
    const duration = audioPlayer.duration;
    
    const progress = (current / duration) * 100;
    progressBar.value = progress || 0;
    document.querySelector('.progress-fill').style.width = `${progress}%`;
    
    currentTimeEl.textContent = formatTime(current);
    durationEl.textContent = formatTime(duration);

    const words = highlightText.children;
    for (let word of words) {
        const start = parseFloat(word.dataset.start);
        const end = parseFloat(word.dataset.end);
        if (current >= start && current <= end) {
            word.classList.add('active');
            word.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            word.classList.remove('active');
        }
    }
});

audioPlayer.addEventListener('ended', () => {
    isPlaying = false;
    updatePlayBtn();
});

playBtn.addEventListener('click', () => {
    if (isPlaying) {
        audioPlayer.pause();
    } else {
        audioPlayer.play();
    }
    isPlaying = !isPlaying;
    updatePlayBtn();
});

progressBar.addEventListener('input', (e) => {
    const time = (e.target.value / 100) * audioPlayer.duration;
    audioPlayer.currentTime = time;
});

volumeSlider.addEventListener('input', (e) => {
    audioPlayer.volume = e.target.value;
});

function updatePlayBtn() {
    playBtn.innerHTML = isPlaying ? '<i data-lucide="pause"></i>' : '<i data-lucide="play"></i>';
    lucide.createIcons();
}

function formatTime(seconds) {
    if (isNaN(seconds)) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function updateMetrics(metrics) {
    rtfValue.textContent = `${metrics.rtf.toFixed(3)}x`;
    rtfStatus.textContent = metrics.rtf < 1 ? 'Faster than real-time' : 'Slower than real-time';
    rtfStatus.style.color = metrics.rtf < 1 ? 'var(--accent-green)' : 'var(--accent-yellow)';
    
    genTimeValue.textContent = `${(metrics.generation_time * 1000).toFixed(0)} ms`;
}

async function startSystemPolling() {
    setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/api/system-info`);
            const info = await res.json();
            
            cpuValue.textContent = `${info.cpu_percent.toFixed(1)}%`;
            cpuBar.style.width = `${info.cpu_percent}%`;

            if (info.gpu_percent != null) {
                gpuValue.textContent = `${info.gpu_percent.toFixed(1)}%`;
                gpuBar.style.width = `${info.gpu_percent}%`;
            }

            if (info.gpu_memory_used != null) {
                gpuMemValue.textContent = `${info.gpu_memory_used.toFixed(1)} / ${info.gpu_memory_total.toFixed(1)} GB`;
                const memPercent = (info.gpu_memory_used / info.gpu_memory_total) * 100;
                gpuMemBar.style.width = `${memPercent}%`;
            }
        } catch (error) {
            console.error('System polling failed:', error);
        }
    }, 1000);
}

viewHistoryBtn.addEventListener('click', async () => {
    analysisModal.classList.remove('hidden');
    await updateHistoryTable();
});

closeModalBtn.addEventListener('click', () => {
    analysisModal.classList.add('hidden');
});

async function updateHistoryTable() {
    const res = await fetch(`${API_URL}/api/metrics`);
    const data = await res.json();
    historyTable.innerHTML = '';
    
    data.history.reverse().forEach(entry => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(entry.timestamp).toLocaleTimeString()}</td>
            <td>${entry.text.substring(0, 30)}...</td>
            <td>${entry.audio_duration.toFixed(2)}s</td>
            <td>${entry.generation_time.toFixed(2)}s</td>
            <td>${entry.rtf.toFixed(2)}x</td>
            <td>${entry.device}</td>
        `;
        historyTable.appendChild(row);
    });
}

init();
