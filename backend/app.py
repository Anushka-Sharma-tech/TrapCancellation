import spaces
from gradio import Server
from fastapi.middleware.cors import CORSMiddleware

from app import main as backend


@spaces.GPU(duration=120)
def model_fake_probability_gpu(audio, sr=16000):
    return backend._model_fake_probability(audio, sr)


# Tell the backend to use the ZeroGPU version
backend.model_fake_probability_gpu = model_fake_probability_gpu


# Gradio server
app = Server(debug=True)

# Add all FastAPI backend routes
app.include_router(backend.app.router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )