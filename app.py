import os, sys, sysconfig
from faster_whisper import WhisperModel

MODEL = "large-v3-turbo"
cpu = "--cpu" in sys.argv
args = [a for a in sys.argv[1:] if a not in ("--cpu", "--gpu")]
if len(args) != 2:
    sys.exit("usage: py app.py video.mp4 transcript.txt [--gpu | --cpu]   (default: --gpu)")

if cpu:
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
else:
    # expose the cuBLAS/cuDNN DLLs from the nvidia-* wheels in the venv
    for root, _, files in os.walk(os.path.join(sysconfig.get_path("purelib"), "nvidia")):
        if any(f.endswith(".dll") for f in files):
            os.environ["PATH"] = root + os.pathsep + os.environ["PATH"]
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")

segments, _ = model.transcribe(args[0], vad_filter=True)
with open(args[1], "w", encoding="utf-8") as f:
    for s in segments:
        print(s.text.strip())
        f.write(s.text.strip() + "\n")
