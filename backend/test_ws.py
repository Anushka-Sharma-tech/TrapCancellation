import asyncio
import numpy as np
import websockets

URL = "wss://anushka-8-trapcancellation-backend.hf.space/ws/stream"


async def main():
    print("Connecting...")

    try:
        async with websockets.connect(
            URL,
            open_timeout=30,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:

            print("Connected!")

            # 1 second of Float32 PCM, 16 kHz
            audio = np.zeros(16000, dtype=np.float32)

            await ws.send(audio.tobytes())
            print("Audio sent.")

            response = await asyncio.wait_for(
                ws.recv(),
                timeout=60,
            )

            print("Server response:")
            print(response)

    except Exception as e:
        print(f"WebSocket error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())