import gradio as gr
from app.main import app as fastapi_app

# Create a minimal UI so Hugging Face Spaces knows the app is running
with gr.Blocks() as demo:
    gr.Markdown("<center><h1>🛡️ TrapCancellation Core Engine</h1></center>")
    gr.Markdown("<center>Real-Time Voice Impersonation API is <b>LIVE</b>.</center>")

# This mounts your existing FastAPI app (with /health, /ws/stream, etc.) 
# directly onto the Hugging Face server.
app = gr.mount_gradio_app(fastapi_app, demo, path="/")