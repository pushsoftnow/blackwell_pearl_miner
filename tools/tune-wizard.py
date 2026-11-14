#!/usr/bin/env python3
# SPDX-License-Identifier: ISC
"""Kernel tuning wizard for bpm (optional, Python 3.8+ standard library only).

The miner itself times the kernel variants compiled into it (kernels/variants.txt) with --tune.
This wizard goes further: it searches new placements of the kernel's operand stores and loads
(knobs K780_P0..P7, PL, MID in kernels/kernel780.cu) and ptxas optimisation levels on your GPU,
compiles each candidate, checks that it computes exactly what the default kernel computes and
measures it against the default. At the end it can add the best result to your local build and
prepares a GitHub issue so the result can go into the variant list for everyone.

Run from the repository: python3 tools/tune-wizard.py   (--help for unattended use)
Root (sudo) is only needed to change the power limit, to pause other users' GPU processes or
for ncu cycle counting without profiling permission; the wizard asks before it does any of that.
"""
import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

REPO_URL = "https://github.com/pushsoftnow/blackwell_pearl_miner"
ROOT = Path(__file__).resolve().parent.parent
KNOBS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "PL", "MID"]
DEFAULT = {"P0": 5, "P1": 10, "P2": 12, "P3": 14, "P4": 25, "P5": 27, "P6": 29, "P7": 31, "PL": 20, "MID": 1}
POSITIONS = 35  # MMA indices 0..34 of a K32 step
PTXAS_LEVELS = [("-O0",), ("-O1",), ("-O2",), ("-O3",), ()]  # () = nvcc's default
MODES = {
    "quick": (40, 4.0, "about 4 minutes"),
    "normal": (150, 6.0, "about 8 minutes"),
    "thorough": (500, 8.0, "about 30 minutes"),
}

# ---------------------------------------------------------------- terminal helpers

AUTO_YES = False


def say(msg=""):
    print(msg, flush=True)


def ask(question, default=None, choices=None):
    """Prompt; returns the default when --yes is given or stdin is not a terminal."""
    hint = f" [{default}]" if default is not None else ""
    if AUTO_YES or not sys.stdin.isatty():
        if default is None:
            sys.exit(f"{question}: no answer possible without a terminal (see --help for options)")
        say(f"{question}{hint}: {default}")
        return str(default)
    while True:
        ans = input(f"{question}{hint}: ").strip() or (str(default) if default is not None else "")
        if not ans:
            continue
        if choices and ans.lower() not in choices:
            say(f"  please answer one of: {', '.join(choices)}")
            continue
        return ans


def yes(question, default=True):
    return ask(question + " (y/n)", "y" if default else "n", ["y", "n", "yes", "no"]).lower().startswith("y")


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.start, self.done = total, label, time.time(), 0
        self.tty = sys.stdout.isatty()
        self.printed = -1

    def update(self, done, text=""):
        self.done = done
        frac = min(1.0, done / max(1, self.total))
        el = time.time() - self.start
        eta = el / frac - el if frac > 0.02 else float("nan")
        eta_s = time.strftime("%M:%S", time.gmtime(eta)) if eta == eta else "--:--"
        width = 30
        bar = "#" * round(frac * width) + "." * (width - round(frac * width))
        line = f"{self.label} [{bar}] {done}/{self.total} | ETA {eta_s} {text}"
        if self.tty:
            sys.stdout.write("\r\x1b[2K" + line[: shutil.get_terminal_size((120, 20)).columns - 1])
            sys.stdout.flush()
        elif done != self.printed and (done >= self.total or done % 10 == 0):
            self.printed = done
            say(line)

    def finish(self):
        if not self.tty and self.printed != self.done:
            self.printed = self.done
            say(f"{self.label}: {self.done}/{self.total} done")
        if self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()


def run(cmd, check=True, **kw):
    return subprocess.run(cmd, check=check, text=True, capture_output=True, **kw)


# ---------------------------------------------------------------- environment


def as_user_cmd(cmd):
    """Under sudo, run build and miner commands as the invoking user (no root-owned files)."""
    user = os.environ.get("SUDO_USER")
    if os.geteuid() == 0 and user and user != "root":
        return ["sudo", "-u", user, "-H", "--preserve-env=PATH,CUDA_HOME,CUDA_PATH,NVCC,CUDA_DEVICE_ORDER"] + cmd
    return cmd


def chown_to_user(path):
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() == 0 and uid and gid:
        try:
            os.chown(path, int(uid), int(gid))
        except OSError:
            pass


def cuda_home():
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return Path("/usr/local/cuda")


def tool(name):
    p = cuda_home() / "bin" / name
    if p.is_file():
        return str(p)
    return shutil.which(name)


def nvcc_version(nvcc):
    out = run([nvcc, "--version"]).stdout
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else "unknown"


def git_commit():
    try:
        return run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).stdout.strip()
    except Exception:
        return "unknown"


def driver_version():
    try:
        t = Path("/proc/driver/nvidia/version").read_text()
        m = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", t)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "unknown"


def gpus():
    out = run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,power.limit,power.min_limit,power.max_limit,power.default_limit",
               "--format=csv,noheader,nounits"]).stdout
    rows = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        rows.append({"index": int(f[0]), "uuid": f[1], "name": f[2], "bus": f[3], "limit": float(f[4]), "min": float(f[5]),
                     "max": float(f[6]), "default": float(f[7])})
    return rows


def compute_apps(uuid):
    out = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], check=False).stdout
    apps = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 4 and f[0] == uuid and f[1].isdigit() and int(f[1]) != os.getpid():
            apps.append({"pid": int(f[1]), "name": f[2], "mem": f[3]})
    return apps


def power_limit_cmd(index, watts):
    cmd = ["nvidia-smi", "-i", str(index), "-pl", f"{watts:g}"]
    if os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    return cmd


# ---------------------------------------------------------------- candidates


def constraints_ok(c):
    """Same order rules as the static_asserts of kernels/kernel780.cu."""
    p = c
    return (0 <= p["P0"] < p["P1"] and p["P2"] >= 10 and p["P4"] >= 20 and p["P1"] <= p["P4"] <= p["P6"]
            and p["P2"] <= p["P3"] <= p["P5"] <= p["P7"] < POSITIONS and 0 <= p["PL"] < POSITIONS and p["MID"] in (0, 1))


def key_of(knobs, ptxas):
    return ",".join(ptxas) + "|" + " ".join(f"{k}={knobs[k]}" for k in KNOBS)


def describe(knobs, ptxas):
    """Variant-list notation: ptxas options, then the knobs that differ from the default."""
    opts = ",".join(ptxas) if ptxas else "-O3"
    diff = [f"{k}={knobs[k]}" for k in KNOBS if knobs[k] != DEFAULT[k]]
    return " ".join([opts] + diff)


def parse_variants():
    seeds = []
    for line in (ROOT / "kernels/variants.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        knobs = dict(DEFAULT)
        for kv in parts[2:]:
            k, v = kv.split("=")
            knobs[k] = int(v)
        seeds.append((parts[0], knobs, tuple(parts[1].split(","))))
    return seeds


def mutate(knobs, rng):
    c = dict(knobs)
    for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
        k = rng.choice(KNOBS)
        if k == "MID":
            c[k] ^= 1
        elif rng.random() < 0.15:
            c[k] = rng.randrange(POSITIONS)
        else:
            c[k] += rng.choice([-3, -2, -1, -1, 1, 1, 2, 3])
    return c


class Builder:
    def __init__(self, work, nvcc, cuobjdump):
        self.work, self.nvcc, self.cuobjdump = work, nvcc, cuobjdump
        pearl = ROOT / "third_party/pearl/miner/pearl-gemm"
        self.inc = [f"-I{pearl / 'csrc'}", f"-I{pearl / 'third_party/cutlass/include'}"]
        cccl = cuda_home() / "include/cccl"
        if cccl.is_dir():
            self.inc.append(f"-I{cccl}")
        self.count = 0

    def build(self, knobs, ptxas):
        self.count += 1
        cid = self.count
        lib = self.work / f"cand_{cid}.so"
        cmd = [self.nvcc, "-O3", "-std=c++20", "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a", "-Xcompiler", "-fPIC",
               "-diag-suppress", "549", "-shared", f"-DK780_NS=cand_{cid}"]
        for o in ptxas:
            cmd += ["-Xptxas", o]
        cmd += [f"-DK780_{k}={knobs[k]}" for k in KNOBS] + self.inc
        cmd += [str(ROOT / "kernels/kernel780.cu"), f"-L{cuda_home() / 'lib64/stubs'}", "-lcuda", "-o", str(lib)]
        r = run(cmd, check=False)
        if r.returncode != 0:
            return None, None, r.stderr[-2000:]
        return lib, self.sass_hash(lib, cid), None

    def sass_hash(self, lib, cid):
        out = run([self.cuobjdump, "-sass", str(lib)]).stdout
        h, on = hashlib.sha256(), False
        for line in out.splitlines():
            if "Function :" in line:
                on = f"cand_{cid}6kernelENS" in line
                continue
            if on:
                m = re.match(r"\s*/\*[0-9a-f]+\*/\s*(.*?)\s*;?\s*(/\*.*)?$", line)
                if m and m.group(1):
                    h.update(m.group(1).encode() + b"\n")
        return h.hexdigest()[:16]


class Measurer:
    def __init__(self, bpm, device, k, seconds, method, ncu):
        self.bpm, self.device, self.k, self.seconds, self.method, self.ncu = bpm, device, k, seconds, method, ncu
        self.env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")

    def timing(self, libs, seconds=None):
        cmd = as_user_cmd([self.bpm, "--device", str(self.device), "--k", str(self.k), "--tune-seconds", str(seconds or self.seconds)])
        for lib in libs:
            cmd += ["--tune-candidate", str(lib)]
        r = run(cmd, check=False, env=self.env)
        if r.returncode != 0:
            raise RuntimeError(f"bpm failed: {r.stderr.strip()[-500:]}")
        d = json.loads(r.stdout.strip().splitlines()[-1])
        return [(c["valid"], c["relative"], c.get("error", 0.0)) for c in d["candidates"]]

    def cycles(self, libs):
        """ncu: elapsed SM cycles of each candidate kernel against the default (K=8192)."""
        base = [self.ncu, "--kernel-name-base", "demangled", "--kernel-name", r"regex:(cand_[0-9]+|k780_v0)::kernel",
                "--metrics", "sm__cycles_elapsed.max", "--csv", self.bpm, "--device", str(self.device), "--k", "8192", "--tune-seconds", "0"]
        for lib in libs:
            base += ["--tune-candidate", str(lib)]
        if self.ncu_sudo and os.geteuid() != 0:
            base = ["sudo", "env", "CUDA_DEVICE_ORDER=PCI_BUS_ID"] + base
        r = run(base, check=False, env=self.env)
        cyc = {}
        for line in r.stdout.splitlines():
            if "sm__cycles_elapsed.max" not in line:
                continue
            f = next(csv.reader([line]))
            name = next((x for x in f if "::kernel" in x), "")
            ns = name.split("::")[0]
            try:
                v = float(f[-1].replace(",", "").replace(".", ""))  # integer cycles, any thousands separator
            except ValueError:
                continue
            cyc[ns] = min(cyc.get(ns, v), v)
        valid_json = None
        for line in r.stdout.splitlines():
            if line.startswith("{"):
                valid_json = json.loads(line)
        if "k780_v0" not in cyc or valid_json is None:
            raise RuntimeError("ncu produced no cycle counts (profiling permission?):\n" + (r.stderr or r.stdout)[-800:])
        out = []
        for lib, c in zip(libs, valid_json["candidates"]):
            ns = Path(lib).stem
            v = cyc.get(ns)
            out.append((c["valid"] and v is not None, (v / cyc["k780_v0"]) if v else 0.0, 0.0))
        return out

    ncu_sudo = False

    def measure(self, libs):
        return self.cycles(libs) if self.method == "ncu" else self.timing(libs)


# ---------------------------------------------------------------- main flow


def main():
    global AUTO_YES
    ap = argparse.ArgumentParser(description="Kernel tuning wizard for bpm. Without options it asks for everything.")
    ap.add_argument("--gpu", type=int, help="GPU index (nvidia-smi)")
    ap.add_argument("--mode", choices=list(MODES) + ["custom"], help="search size")
    ap.add_argument("--candidates", type=int, help="number of measured candidates (with --mode custom)")
    ap.add_argument("--method", choices=["timing", "ncu"], help="timing at the power limit (default) or ncu cycle counts")
    ap.add_argument("--k", type=int, default=16384, help="K used for timing (default 16384)")
    ap.add_argument("--power-limit", type=float, help="power limit in W during the search (default: keep)")
    ap.add_argument("--pause-processes", choices=["pause", "keep", "abort"], help="other GPU processes: pause (SIGSTOP/SIGCONT), keep running, abort")
    ap.add_argument("--seed", type=int, default=None, help="random seed of the search")
    ap.add_argument("--yes", action="store_true", help="accept the default answer of every question")
    ap.add_argument("--no-apply", action="store_true", help="do not add the result to kernels/variants.txt")
    args = ap.parse_args()
    AUTO_YES = args.yes

    say("bpm kernel tuning wizard")
    say("=" * 24)
    say("Searches faster placements of kernel 780 for your GPU and CUDA toolkit. Every candidate is")
    say("checked to compute exactly what the default kernel computes; nothing is sent anywhere.\n")

    nvcc, cuobjdump = tool("nvcc"), tool("cuobjdump")
    if not nvcc or not cuobjdump:
        sys.exit(f"nvcc/cuobjdump not found under {cuda_home()}/bin (set CUDA_HOME)")
    if not (ROOT / "third_party/pearl/miner/pearl-gemm/third_party/cutlass/include/cutlass/cutlass.h").is_file():
        sys.exit("third-party sources missing: run `git submodule update --init --recursive`")
    bpm = ROOT / "target/release/bpm"
    probe = run([str(bpm), "--tune-candidate", "/nonexistent"], check=False) if bpm.is_file() else None
    if probe is None or "cannot read" not in probe.stderr:
        say("The miner binary is missing or older than this wizard.")
        if not yes("Build it now with `cargo build --release`?"):
            sys.exit("bpm with --tune-candidate support is required")
        r = subprocess.run(as_user_cmd(["cargo", "build", "--release", "--manifest-path", str(ROOT / "Cargo.toml")]))
        if r.returncode != 0:
            sys.exit("cargo build failed")
    cuda_ver = nvcc_version(nvcc)

    # GPU
    rows = gpus()
    if not rows:
        sys.exit("nvidia-smi lists no GPU")
    say("GPUs:")
    for g in rows:
        say(f"  {g['index']}: {g['name']} ({g['bus']}, power limit {g['limit']:g} W)")
    gi = args.gpu if args.gpu is not None else int(ask("Which GPU", rows[0]["index"], [str(g["index"]) for g in rows]))
    gpu = next((g for g in rows if g["index"] == gi), None)
    if gpu is None:
        sys.exit(f"no GPU with index {gi}")

    # Search size and method
    mode = args.mode
    if mode is None:
        say("\nSearch size:")
        for name, (n, _, dur) in MODES.items():
            say(f"  {name:<9} {n:>4} candidates, {dur}")
        say("  custom    your own number")
        mode = ask("Mode", "normal", list(MODES) + ["custom"]).lower()
    if mode == "custom":
        n_cand = args.candidates or int(ask("Number of candidates", 100))
        seconds = 6.0
    else:
        n_cand, seconds, _ = MODES[mode]
    method = args.method
    ncu = tool("ncu")
    if method is None:
        if ncu:
            say("\nMeasuring method:")
            say("  timing  kernel time at your power limit, interleaved with the default (recommended)")
            say("  ncu     SM cycle counts from Nsight Compute (independent of clocks; may need root)")
            method = ask("Method", "timing", ["timing", "ncu"]).lower()
        else:
            method = "timing"
    if method == "ncu" and not ncu:
        sys.exit("ncu not found")

    # Power limit
    say(f"\nPower limit of GPU {gi}: {gpu['limit']:g} W (range {gpu['min']:g}-{gpu['max']:g} W). Tune at the limit you mine with.")
    want = args.power_limit
    if want is None:
        want = float(ask("Power limit for the search in W", f"{gpu['limit']:g}"))
    restore_limit = None
    if abs(want - gpu["limit"]) > 0.5:
        if not gpu["min"] <= want <= gpu["max"]:
            sys.exit(f"{want:g} W is outside {gpu['min']:g}-{gpu['max']:g} W")
        cmd = power_limit_cmd(gi, want)
        say(f"Setting the power limit: {' '.join(cmd)}")
        if subprocess.run(cmd).returncode != 0:
            sys.exit("could not set the power limit")
        restore_limit = gpu["limit"]

    # Other processes
    paused = []

    def restore():
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
                say(f"Resumed PID {pid}.")
            except OSError as e:
                say(f"Could not resume PID {pid}: {e} (run: kill -CONT {pid})")
        paused.clear()
        if restore_limit is not None:
            subprocess.run(power_limit_cmd(gi, restore_limit), capture_output=True)
            say(f"Power limit restored to {restore_limit:g} W.")

    def on_signal(signum, _frame):
        say("\nInterrupted.")
        restore()
        sys.exit(130)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    try:
        apps = compute_apps(gpu["uuid"])
        if apps:
            say(f"\nOther processes on GPU {gi} (they distort the measurement while they compute):")
            for a in apps:
                say(f"  PID {a['pid']:>8}  {a['name']}  ({a['mem']} MiB)")
            say("  pause  stop them with SIGSTOP now and continue them with SIGCONT at the end")
            say("         (memory stays allocated; a paused miner reconnects to its pool afterwards)")
            say("  keep   leave them running")
            say("  abort  stop the wizard")
            choice = args.pause_processes or ask("Other processes", "pause", ["pause", "keep", "abort"]).lower()
            if choice == "abort":
                sys.exit("aborted")
            if choice == "pause":
                for a in apps:
                    try:
                        os.kill(a["pid"], signal.SIGSTOP)
                        paused.append(a["pid"])
                        say(f"Paused PID {a['pid']}.")
                    except PermissionError:
                        restore()
                        sys.exit(f"no permission to pause PID {a['pid']}: run the wizard with sudo or stop it yourself")
                    except ProcessLookupError:
                        pass
                time.sleep(2)

        search(args, gpu, gi, cuda_ver, nvcc, cuobjdump, bpm, n_cand, seconds, method, ncu, want)
    finally:
        restore()


def search(args, gpu, gi, cuda_ver, nvcc, cuobjdump, bpm, n_cand, seconds, method, ncu, power):
    rng = random.Random(args.seed)
    work = Path(tempfile.mkdtemp(prefix="bpm-tune-"))
    os.chmod(work, 0o755)
    builder = Builder(work, nvcc, cuobjdump)
    meas = Measurer(str(bpm), gi, args.k, seconds, method, ncu)
    if method == "ncu" and os.geteuid() != 0:
        test = run([ncu, "--metrics", "sm__cycles_elapsed.max", "--kernel-name", "regex:k780_v0::kernel", "--kernel-name-base", "demangled",
                    str(bpm), "--device", str(gi), "--k", "8192", "--tune-seconds", "0", "--tune-candidate", "/nonexistent"], check=False)
        if "ERR_NVGPUCTRPERM" in test.stdout + test.stderr:
            say("ncu needs root on this system; it will run through sudo.")
            meas.ncu_sudo = True

    results = {}      # key -> dict(knobs, ptxas, hash, valid, relative, error)
    by_hash = {}      # sass hash -> (valid, relative, error)
    seen = set()
    built_names = {}  # hash -> name of an existing variant
    jobs = os.cpu_count() or 4
    batch_size = 8

    def evaluate(cands, progress=None, label=""):
        """Builds, de-duplicates by machine code and measures a list of (knobs, ptxas, name)."""
        fresh = []
        for knobs, ptxas, name in cands:
            k = key_of(knobs, ptxas)
            if k in seen or not constraints_ok(knobs):
                continue
            seen.add(k)
            fresh.append((knobs, ptxas, name, k))
        with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
            built = list(ex.map(lambda c: (c, builder.build(c[0], c[1])), fresh))
        todo = []
        for (knobs, ptxas, name, k), (lib, h, err) in built:
            if lib is None:
                results[k] = {"knobs": knobs, "ptxas": ptxas, "hash": None, "valid": False, "relative": 0.0, "error": 0.0, "note": "build failed"}
                continue
            if name:
                built_names.setdefault(h, name)
            results[k] = {"knobs": knobs, "ptxas": ptxas, "hash": h, "lib": lib, "name": name}
            if h not in by_hash and all(t[2] != h for t in todo):
                todo.append((k, lib, h))
        measured = 0
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            for (k, lib, h), r in zip(chunk, meas.measure([c[1] for c in chunk])):
                by_hash[h] = r
            measured += len(chunk)
            if progress:
                progress(len(chunk))
        for k, r in results.items():
            if r.get("hash") in by_hash and "valid" not in r:
                r["valid"], r["relative"], r["error"] = by_hash[r["hash"]]
        return measured

    def best(n=1):
        ok = [r for r in results.values() if r.get("valid")]
        ok.sort(key=lambda r: r["relative"])
        return ok[:n]

    def gain(r):
        return (1.0 / r["relative"] - 1.0) * 100.0 if r["relative"] > 0 else float("nan")

    # 1. Baselines: the variant list and every ptxas level of the default placement.
    say(f"\nWork directory: {work}")
    say("Step 1/3: baselines (kernels/variants.txt and ptxas -O0..-O3 / nvcc default)")
    seeds = [(knobs, ptxas, name) for name, knobs, ptxas in parse_variants()]
    seeds += [(dict(DEFAULT), lvl, f"default{''.join(lvl) or '-nvcc-default'}") for lvl in PTXAS_LEVELS]
    p = Progress(len(seeds), "baselines")
    cnt = [0]

    def tick(n):
        cnt[0] += n
        b = best()
        p.update(cnt[0], f"| best {gain(b[0]):+.2f} %" if b else "")

    measured = evaluate(seeds, tick)
    p.update(p.total)
    p.finish()
    for r in sorted((r for r in results.values() if "relative" in r), key=lambda r: (not r["valid"], r["relative"])):
        state = f"{gain(r):+7.2f} %" if r["valid"] else "WRONG RESULTS" if r.get("hash") else r.get("note", "failed")
        say(f"  {describe(r['knobs'], r['ptxas']):<44} {state}{'  (' + r['name'] + ')' if r.get('name') else ''}")
    levels = sorted({r["ptxas"] for r in results.values() if r.get("valid") and r["relative"] < 1.02}) or [("-O1",)]

    # 2. Local search around the best placements.
    say(f"\nStep 2/3: search ({n_cand} candidates, method {method})")
    p = Progress(n_cand, "search")
    cnt[0] = 0
    stall = 0
    while cnt[0] < n_cand and stall < 40:
        parents = best(4) or [{"knobs": DEFAULT, "ptxas": ("-O1",)}]
        cands = []
        tries = 0
        while len(cands) < batch_size and tries < 2000:
            tries += 1
            par = rng.choice(parents[: max(1, len(parents))])
            knobs = mutate(par["knobs"], rng)
            ptxas = par["ptxas"] if rng.random() > 0.1 else rng.choice(levels)
            k = key_of(knobs, ptxas)
            if k in seen or not constraints_ok(knobs) or any(key_of(c[0], c[1]) == k for c in cands):
                continue
            cands.append((knobs, ptxas, None))
        if not cands:
            break
        before = cnt[0]
        evaluate(cands, tick)
        stall = stall + 1 if cnt[0] == before else 0
    p.finish()

    # 3. Confirmation: the best candidates once more, measured longer.
    top = []
    for r in best(len(results)):
        if r["hash"] not in [t["hash"] for t in top]:
            top.append(r)
        if len(top) == 5:
            break
    say("\nStep 3/3: confirmation of the 5 best (timing, 60 s)")
    conf = meas.timing([t["lib"] for t in top], seconds=60)
    for t, (valid, rel, err) in zip(top, conf):
        t["valid"], t["relative"], t["error"] = valid, rel, err
    top = [t for t in top if t["valid"]]
    top.sort(key=lambda r: r["relative"])
    for t in top:
        say(f"  {describe(t['knobs'], t['ptxas']):<44} {gain(t):+7.3f} ± {t['error'] * 100:.3f} %")
    winner = top[0] if top else None
    # Same rule as the miner (src/tune.rs): beyond twice the measuring error and at least 0.05 %.
    significant = bool(winner) and (1.0 - winner["relative"]) > max(2.0 * winner["error"], 0.0005)

    gpu_slug = re.sub(r"[^a-z0-9]+", "-", gpu["name"].lower().replace("nvidia", "")).strip("-")
    date = datetime.date.today().strftime("%Y%m%d")
    existing = built_names.get(winner["hash"]) if winner else None
    say("")
    if not winner or not significant:
        say("Result: no candidate is measurably faster than the default kernel on this GPU.")
    elif existing:
        say(f"Result: the fastest candidate ({gain(winner):+.2f} %) has the same machine code as the existing variant '{existing}'.")
        say("The miner already contains it and picks it in its own tuning.")
    else:
        say(f"Result: {describe(winner['knobs'], winner['ptxas'])}  {gain(winner):+.2f} % against the default kernel.")

    report = {
        "format": 1,
        "gpu": gpu["name"],
        "driver": driver_version(),
        "cuda": cuda_ver,
        "power_limit_w": power,
        "k": args.k,
        "bpm_commit": git_commit(),
        "method": method,
        "candidates_measured": len(by_hash),
        "date": datetime.date.today().isoformat(),
        "best": {"variant": describe(winner["knobs"], winner["ptxas"]), "gain_percent": round(gain(winner), 3),
                 "error_percent": round(winner["error"] * 100, 3), "significant": significant, "same_as": existing} if winner else None,
        "top": [{"variant": describe(t["knobs"], t["ptxas"]), "gain_percent": round(gain(t), 3)} for t in top],
        "baselines": [{"variant": describe(r["knobs"], r["ptxas"]), "name": r.get("name"), "gain_percent": round(gain(r), 3) if r.get("valid") else None}
                      for r in results.values() if r.get("name")],
    }
    out_dir = ROOT / "tuning/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    chown_to_user(out_dir)
    out = out_dir / f"{gpu_slug}-cuda{cuda_ver}-{power:g}w-{date}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    chown_to_user(out)
    say(f"\nReport written to {out}")

    if winner and significant and not existing and not args.no_apply:
        name = f"{gpu_slug}-{date}"[:32]
        line = f"{name:<10} {describe(winner['knobs'], winner['ptxas'])}"
        say(f"\nVariant line: {line}")
        if yes("Add it to kernels/variants.txt, rebuild the miner and let it measure again?"):
            vt = ROOT / "kernels/variants.txt"
            with vt.open("a") as f:
                f.write(f"# tools/tune-wizard.py on {gpu['name']}, CUDA {cuda_ver}, driver {driver_version()}, {power:g} W, "
                        f"{datetime.date.today().isoformat()}: {gain(winner):+.2f} %\n{line}\n")
            r = subprocess.run(as_user_cmd(["cargo", "build", "--release", "--manifest-path", str(ROOT / "Cargo.toml")]))
            if r.returncode == 0:
                subprocess.run(as_user_cmd([str(bpm), "--device", str(gi), "--k", str(args.k), "--tune-only", "--tune"]),
                               env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"))
                say("Restart the miner to use the new build.")

    title = f"Tuning result: {gpu['name']}, CUDA {cuda_ver}, {power:g} W"
    if report["best"]:
        title += f", {report['best']['gain_percent']:+.2f} %"
    body = ("Result of `tools/tune-wizard.py` (please submit even if nothing faster was found; that is a result too).\n\n"
            "```json\n" + json.dumps(report, indent=1) + "\n```\n")
    url = f"{REPO_URL}/issues/new?" + urllib.parse.urlencode({"title": title, "body": body})
    say("\nPlease share the result so it can go into the variant list for everyone with this GPU:")
    say("open this link (needs a GitHub account), check the text and press 'Create':\n")
    if len(url) > 7500:
        say(f"{REPO_URL}/issues/new  (attach or paste {out})")
    else:
        say(url)
    say("\nThe report contains GPU model, driver, CUDA version, power limit and measured variants only.")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
