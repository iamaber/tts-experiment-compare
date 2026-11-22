# Kyutai TTS web application

A web app for Kyutai text-to-speech with word highlighting during playback, voice cloning, and performance monitoring.

## Features

- Text-to-speech with the Kyutai model
- Words highlight as they're spoken
- Upload audio samples to create custom voices
- Live metrics and system resource tracking
- Generation history with export
- Streaming generation over WebSocket

## Architecture

- Backend: FastAPI (Python) with the Kyutai TTS model
- Frontend: Next.js 14 (TypeScript, React, Tailwind CSS)
- Model: Kyutai TTS, 1.8B parameters, stored locally

## Quick start

### Prerequisites

- Python 3.12+
- Node.js 18+
- CUDA GPU (recommended) or CPU
- uv (Python package manager)

### 1. Install backend dependencies

```bash
cd backend
uv pip install -e .
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
```

### 3. Start backend server

```bash
cd backend
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Backend API: http://localhost:8000

### 4. Start frontend

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
6. View performance metrics as they update
7. Click "View Details" for historical analysis

## Project structure

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

## API endpoints

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

- Real-time factor: ~0.78x on CUDA (faster than real time)
- Sample rate: 24kHz
- Languages: English and French
- Streaming over WebSocket

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

- Kyutai TTS model: https://github.com/kyutai-labs/delayed-streams-modeling
- Moshi framework: https://github.com/kyutai-labs/moshi
