# Kyutai TTS Web Application

Full-stack web application for neural text-to-speech generation with real-time word highlighting, voice cloning, and performance monitoring.

## Features

- 🎙️ **Text-to-Speech Generation** - High-quality neural TTS using Kyutai model
- 📝 **Real-time Word Highlighting** - Words highlight as they're spoken
- 🎭 **Voice Cloning** - Upload audio samples to create custom voices
- 📊 **Performance Monitoring** - Real-time metrics and system resource tracking
- 📈 **Historical Analysis** - View and export generation history
- 🌐 **WebSocket Streaming** - Low-latency streaming generation

## Architecture

- **Backend**: FastAPI (Python) with Kyutai TTS model
- **Frontend**: Next.js 14 (TypeScript + React + Tailwind CSS)
- **Model**: Kyutai TTS 1.8B parameters (local)

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- CUDA-capable GPU (recommended) or CPU
- uv (Python package manager)

### 1. Install Backend Dependencies

```bash
cd backend
uv pip install -e .
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Start Backend Server

```bash
cd backend
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Backend API: http://localhost:8000

### 4. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend UI: http://localhost:3000

## Usage

1. Open http://localhost:3000 in your browser
2. Enter text in the input area
3. Select a voice (default or upload custom)
4. Click "Generate Speech"
5. Listen to the audio with synchronized word highlighting
6. View performance metrics in real-time
7. Click "View Details" for historical analysis

## Project Structure

```
tts-test/
├── backend/
│   ├── main.py          # TTS model loader
│   ├── server.py        # FastAPI server
│   ├── pyproject.toml   # Python dependencies
│   ├── voices/          # Custom voice embeddings
│   └── outputs/         # Generated audio files
├── frontend/
│   ├── app/             # Next.js pages
│   ├── components/      # React components
│   ├── lib/             # API client
│   └── package.json     # Node dependencies
└── models/              # Kyutai TTS model files
```

## API Endpoints

### REST API

- `POST /api/tts` - Generate TTS from text
- `POST /api/clone-voice` - Upload voice sample
- `GET /api/voices` - List available voices
- `GET /api/metrics` - Get metrics history
- `GET /api/system-info` - Get system resources
- `GET /api/audio/{filename}` - Download audio file

### WebSocket

- `WS /ws/tts` - Streaming TTS generation

## Performance

- **Real-time Factor**: ~0.78x on CUDA (faster than real-time)
- **Sample Rate**: 24kHz
- **Languages**: English and French
- **Latency**: Low-latency streaming with WebSocket

## Development

### Backend

```bash
cd backend
uv run python main.py  # Test TTS model
uv run uvicorn server:app --reload  # Start API server
```

### Frontend

```bash
cd frontend
npm run dev    # Development server
npm run build  # Production build
npm run lint   # Lint code
```

## License

Model: CC-BY 4.0 (Kyutai)  
Code: MIT

## Credits

- **Kyutai TTS Model**: https://github.com/kyutai-labs/delayed-streams-modeling
- **Moshi Framework**: https://github.com/kyutai-labs/moshi
