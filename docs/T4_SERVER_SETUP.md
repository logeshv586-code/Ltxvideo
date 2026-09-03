# Dual NVIDIA T4 Server Setup

This repository now detects GPU model, per-GPU VRAM, GPU count and system RAM at startup.

For a server with **2 × NVIDIA T4 16 GB**, the application treats the cards as **two independent 16 GB generation workers**. It does not add the VRAM together for one render. Two separate generation requests can run concurrently, one on each T4.

## 1. Server requirements

- Linux server
- NVIDIA driver working (`nvidia-smi`)
- 2 × NVIDIA T4 16 GB recommended for this profile
- Python 3.10+
- 32 GB system RAM minimum recommended; 64 GB is preferred for comfortable CPU offload and two concurrent workers
- Enough disk space for model cache and generated videos

## 2. Clone and start

```bash
git clone https://github.com/logeshv586-code/Ltxvideo.git
cd Ltxvideo
chmod +x server_start.sh
./server_start.sh
```

The first run creates `.venv`, installs requirements and downloads the LTX model cache when required.

The UI listens on:

```text
http://0.0.0.0:7860
```

From your own computer, open:

```text
http://SERVER_IP:7860
```

Make sure TCP port `7860` is permitted by the server firewall/security group, or expose the application through your normal VPN/reverse proxy.

## 3. What is adaptive

At startup the app reads:

- CUDA GPU count
- GPU name for every visible card
- VRAM per GPU
- System RAM

It then selects:

- GPU and CPU memory budgets
- safe native frame limits
- adaptive native resolution scaling
- more conservative scaling for long/heavy clips
- Gradio queue concurrency matching the number of GPU workers

### T4 16 GB behavior

The T4 profile keeps roughly 2 GB of per-card VRAM headroom and uses a 14 GiB runtime budget. Normal short renders receive a larger native render size than the old RTX 4050 laptop path. Longer clips use a smaller scaling increase to protect VAE/attention memory peaks.

Example: a UI request planned at `576×320` can be raised to roughly `768×416` on a T4 for a short clip, subject to the pixel safety cap. The final delivery export remains 720p unless the UI/export target is changed.

## 4. Two-GPU operation

By default all visible CUDA GPUs are used as generation workers.

To force only one worker:

```bash
LTX_MAX_GPU_WORKERS=1 ./server_start.sh
```

To explicitly allow two workers:

```bash
LTX_MAX_GPU_WORKERS=2 ./server_start.sh
```

Each worker keeps its own LTX pipeline and is protected by the generator pool, so the same model object is not executed simultaneously from two requests.

## 5. Different port

```bash
PORT=8080 ./server_start.sh
```

or:

```bash
python run.py --server --port 8080
```

## 6. Check hardware first

```bash
nvidia-smi
python run.py --check
```

At normal startup you should see output similar to:

```text
Hardware profile: NVIDIA T4 16 GB server profile
GPU 0: Tesla T4 · ~15 GB VRAM
GPU 1: Tesla T4 · ~15 GB VRAM
System RAM: ... GB
Adaptive GPU workers: 2
```

## 7. Recommended quality usage

For final videos, use **High** quality and 4-second backend clips first. This gives the adaptive T4 path the most room to increase native spatial detail. For long continuous videos, keep the existing sequential continuation workflow so character and motion state can flow from one generated part to the next.

If a T4 reports an out-of-memory error for a difficult scene, the worker unloads the model, clears that GPU's cache and retries once. If needed, reduce concurrency to one worker or use Balanced quality.
