import os, sys, sysconfig
from faster_whisper import WhisperModel

if len(sys.argv) != 3:
    sys.exit("an example use: py app.py videoname.mp4 outputfile.txt")

# expose the cuBLAS/cuDNN DLLs from the nvidia-* wheels in the venv
for root, _, files in os.walk(os.path.join(sysconfig.get_path("purelib"), "nvidia")):
    if any(f.endswith(".dll") for f in files):
        os.environ["PATH"] = root + os.pathsep + os.environ["PATH"]

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segments, _ = model.transcribe(sys.argv[1], vad_filter=True)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for s in segments:
        print(s.text.strip())
        f.write(s.text.strip() + "\n")
