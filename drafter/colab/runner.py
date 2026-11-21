import os, subprocess, time
from pathlib import Path

QUEUE = Path("/content/queue")
STATUS = Path("/content/progress.txt")
for d in ("todo", "running", "done"):
    (QUEUE / d).mkdir(parents=True, exist_ok=True)

def status(msg):
    line = time.strftime("%H:%M:%S ") + msg
    with STATUS.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)

jobs = {}
status("runner up")
while True:
    for script in sorted((QUEUE / "todo").glob("*.sh")):
        target = QUEUE / "running" / script.name
        script.rename(target)
        log = open(f"/content/{script.stem}.log", "w")
        jobs[script.stem] = subprocess.Popen(["bash", str(target)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd="/content")
        status(f"started {script.stem} pid {jobs[script.stem].pid}")
    for name, proc in list(jobs.items()):
        code = proc.poll()
        if code is not None:
            (QUEUE / "running" / f"{name}.sh").rename(QUEUE / "done" / f"{name}.sh")
            status(f"finished {name} exit {code}")
            del jobs[name]
    print("tick", flush=True)
    time.sleep(15)