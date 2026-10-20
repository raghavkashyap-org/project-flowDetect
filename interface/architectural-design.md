- **project Input Reality:** Live traffic hitting a **FastAPI** backend, where you want live translation into the model's 27 features.
- **project Threat Goal:** **DDoS / Floods, Brute-Force bursts, and Bot traffic** (the exact families this model predicts: `Flood`, `Bot`, `BruteForce`, `Benign`).
- **project Preferred Runtime:** **Live Sidecar Agent + ONNX Runtime** integrated with FastAPI.
- **Current Step:** **Complete Architecture & Design Blueprint first** before writing code.

Below is the complete engineering breakdown covering **(1) whether the model fits & what you almost missed**, **(2) how all 27 features are extracted**, **(3) all ways to run the model**, and **(4) the exact interface & architecture design** connecting the Live Sidecar and FastAPI.

---

## 1. Is This Model Right — Or Did You Miss Something Critical?

### The Short Verdict
- **Right model for project threat goal? YES** — if project goal is catching **volumetric Floods, automated Brute-Force bursts, and Bot polling/C2 patterns**. It will **NOT** catch single-packet payload exploits inside normal human-paced requests (like SQLi, XSS, SSRF, or broken auth logic — those still require input validation / WAF rules).
  **A single stateless HTTP `fastapi.Request` CANNOT directly produce these 27 features.**

### Why a Single FastAPI `Request` Isn't Enough on Its Own
FastAPI operates at **OSI Layer 7 (HTTP Application Layer)**, *after* the Linux kernel has already completed the TCP handshake, stripped TCP headers/flags, and reassembled packets into a single byte stream:
1. **A "Flow" is a Time-Series Across Multiple Packets, Not One Event:**  
   Look at the top 4 features driving this model (from inspecting all 356 trees inside `model.joblib`):
   - `Fwd IAT Std` (**36.80%** of total model gain)
   - `Bwd IAT Tot` (**19.29%** of total model gain)
   - `Flow Duration` (**14.05%** of total model gain)
   - `Flow IAT Mean` (**7.52%** of total model gain)  
   Together, **inter-arrival time (IAT) and duration statistics account for `77.66%` of the model's entire decision power**. A single stateless HTTP request has *no inter-arrival time* and *no backward response packets yet*—passing zeros blinds 78% of the model.
2. **Kernel-Level Floods Never Reach FastAPI:**  
   During a TCP `SYN` flood or UDP flood, packets are absorbed in the OS kernel queue before Uvicorn/FastAPI ever accepts an HTTP request. Only a packet-level sidecar (Layer 3/4) sees them.

### 5 Hidden Model Quirks Uncovered Inside `model.joblib` (Crucial for Production!)
By inspecting the internal histogram bin boundaries (`_bin_mapper.bin_thresholds_`) inside project `model.joblib`, five exact implementation rules come out that silently break production deployments if missed:

| # | Hidden Quirk in `flow-detect-hgb-onnx` | Exact Proof Found in `model.joblib` | Production Rule You MUST Follow |
| :- | :--- | :--- | :--- |
| **1** | **All time/IAT features are in MICROSECONDS ($\mu\text{s}$)**, NOT ms or seconds | `Flow Duration` bins: `10` $\to$ `118,068,000` (`120s` flow timeout); `Idle Mean` starts at `5,000,000` (`5.0s`) | Multiply seconds by **`1_000_000.0`**. Meanwhile, `Flow Pkts/s` and `Flow Byts/s` stay **per second** (`count / (duration_us / 1e6)`). |
| **2** | **Packet lengths are L4 TCP/UDP PAYLOAD bytes (`0..1460`)**, NOT raw Ethernet frame length (`54..1514`) | `Pkt Len Max` & `Bwd Pkt Len Max` top bin is **`1460.0`** (exact TCP MSS = `1500 MTU - 20 IP - 20 TCP`) | Measure **TCP/UDP payload length**, not raw wire frame bytes. Pure TCP `SYN`/`ACK`/`FIN` packets have length **`0`**, not `54`. |
| **3** | **`Down/Up Ratio` is truncated integer `bwd_pkts // fwd_pkts`** | Bin thresholds are `[0.5, 1.5, 2.5, 3.5, ...]` | Compute `float(int(bwd_pkts // max(fwd_pkts, 1)))` (note: **Down = Backward / Server$\to$Client**, **Up = Forward / Client$\to$Server**). |
| **4** | **All 5 TCP Flag columns only split at `0.5` (`0` vs `≥1`)** | Every flag column has 2 bins with threshold `[0.5]`; `Fwd PSH Flags` has **`0` tree splits** (`0.00%` gain) | Any non-zero count (`>= 1`) lands in the same branch as `1`. Division-by-zero when `duration_us == 0` **must** output `0.0` (never `inf`/`NaN`). |
| **5** | ** CICIDS2017 Domain Shift + Unregularized Late Trees (`iters 70–89`)** | `model_card.json` warns single-day lab capture; `model.joblib` peaks at **iter 69** (`-0.01508` loss) then leaf weights spike to `±32,936` at iter 87 | Keep $\tau$ configurable (`default = 0.921055988133361`), start in **Shadow Mode**, AND optionally re-export a **69-tree-iteration clean ONNX model** directly from `model.joblib` (`clf._predictors[:69]`)! |

> **Reverse Proxy Warning:** If FastAPI runs behind Nginx, Cloudflare, or AWS ALB, raw packet capture on FastAPI's internal port sees the **proxy's local IP** unless you either (a) run the sidecar on the edge interface before the proxy, or (b) map `X-Forwarded-For` + sliding HTTP session streams in the fallback collector.

---

## 2. How All 27 Features Are Extracted in Real Time

A network flow is keyed by the bidirectional **5-tuple**:
$$\text{FlowKey} = (\text{client\_ip}, \text{client\_port}, \text{server\_ip}, \text{server\_port}, \text{protocol})$$
- **Forward (`Fwd`) direction:** Client $\longrightarrow$ project FastAPI Server
- **Backward (`Bwd`) direction:** project FastAPI Server $\longrightarrow$ Client

As each packet `(timestamp_us, direction, payload_bytes, tcp_flags)` arrives in the flow window (rolling/active window up to `120s`, emitted periodically e.g. at **packet 4, 8, 16, 32... or every 250 ms** so attacks are caught **mid-flow** before the connection closes!), all **27 features in exact `float32[27]` order** are computed in **$O(1)$ incremental math** (Welford running mean/variance—no need to store thousands of packets per flow!):

| Index | Feature Name | Exact Live Computation Formula |
| ---: | :--- | :--- |
| `0` | `Flow Duration` | $t_{\text{last}} - t_{\text{first}}$ (in **$\mu\text{s}$**) |
| `1` | `Flow Pkts/s` | $N_{\text{total}} / (\text{duration\_us} / 10^6)$ *(if $\text{duration\_us} \le 0 \Rightarrow 0.0$)* |
| `2` | `Flow IAT Mean` | $\text{mean}(\Delta t_{\text{all}})$ across consecutive packets in flow ($\mu\text{s}$) |
| `3` | `Flow IAT Max` | $\max(\Delta t_{\text{all}})$ ($\mu\text{s}$) |
| `4` | `Flow IAT Std` | $\text{std}(\Delta t_{\text{all}})$ ($\mu\text{s}$) |
| `5` | `Tot Fwd Pkts` | Count of client $\to$ server packets $N_{\text{fwd}}$ |
| `6` | `TotLen Fwd Pkts` | $\sum \text{payload\_bytes}_{\text{fwd}}$ |
| `7` | `Flow Byts/s` | $(\sum \text{payload\_bytes}_{\text{fwd}} + \sum \text{payload\_bytes}_{\text{bwd}}) / (\text{duration\_us} / 10^6)$ |
| `8` | `Down/Up Ratio` | $\lfloor N_{\text{bwd}} / \max(N_{\text{fwd}}, 1) \rfloor$ |
| `9` | `Pkt Len Mean` | $\text{mean}(\text{payload\_bytes}_{\text{all}})$ |
| `10` | `Pkt Len Std` | $\text{std}(\text{payload\_bytes}_{\text{all}})$ |
| `11` | `Pkt Len Max` | $\max(\text{payload\_bytes}_{\text{all}})$ |
| `12` | `Fwd Pkt Len Max` | $\max(\text{payload\_bytes}_{\text{fwd}})$ |
| `13` | `Fwd Pkt Len Mean` | $\text{mean}(\text{payload\_bytes}_{\text{fwd}})$ |
| `14` | `Bwd Pkt Len Max` | $\max(\text{payload\_bytes}_{\text{bwd}})$ |
| `15` | `Bwd Pkt Len Mean` | $\text{mean}(\text{payload\_bytes}_{\text{bwd}})$ |
| `16` | `Fwd Act Data Pkts` | Count of `Fwd` packets with $\text{payload\_bytes} \ge 1$ |
| `17` | `Fwd IAT Std` | $\text{std}(\Delta t_{\text{fwd}})$ between consecutive `Fwd` packets ($\mu\text{s}$) |
| `18` | `Bwd IAT Mean` | $\text{mean}(\Delta t_{\text{bwd}})$ between consecutive `Bwd` packets ($\mu\text{s}$) |
| `19` | `Bwd IAT Tot` | $t_{\text{bwd,last}} - t_{\text{bwd,first}}$ ($\mu\text{s}$) |
| `20` | `SYN Flag Cnt` | Count (or `1` if seen) of packets with TCP `SYN` flag |
| `21` | `RST Flag Cnt` | Count (or `1` if seen) of packets with TCP `RST` flag |
| `22` | `PSH Flag Cnt` | Count (or `1` if seen) of packets with TCP `PSH` flag |
| `23` | `ACK Flag Cnt` | Count (or `1` if seen) of packets with TCP `ACK` flag |
| `24` | `Fwd PSH Flags` | `1.0` if `Fwd` `PSH` seen else `0.0` *(unused by model: 0 splits)* |
| `25` | `Idle Mean` | Mean gap duration where inter-packet gap $> 5{,}000{,}000\ \mu\text{s}$ (`5s`), else `0.0` |
| `26` | `Active Mean` | Mean active burst span before each idle gap ($> 5\text{s}$), else `0.0` |

### What If Packet Sniffing (`CAP_NET_RAW`) Isn't Allowed (e.g. Cloud PaaS / Serverless)?
We design **two pluggable collectors** feeding the **exact same `FlowState -> float32[27]` engine**:
1. **Collector Mode A — `LivePacketSidecar` (True L3/L4 Fidelity — Default on Linux VM/Docker Host):**  
   Uses `libpcap` / `Scapy` on project network interface (`eth0` / `lo`, e.g., `tcp port 8000`) to capture real packets, flags, MSS payloads, and microsecond kernel timestamps—**100% faithful to CICIDS2017** and catches **SYN/UDP Floods + HTTP Floods + Brute-Force + Bots**.
2. **Collector Mode B — `AsgiStreamTranslator` (Zero-Privilege Pure FastAPI Fallback):**  
   Hooks into FastAPI's raw ASGI `receive()` and `send()` coroutines keyed by real client IP (`X-Forwarded-For` / socket peer), converting every incoming HTTP request & body chunk into `Fwd` events (segmenting payloads into `1460`-byte MSS chunks) and every outgoing response chunk into `Bwd` events with real microsecond timestamps (`time.perf_counter_ns() // 1000`). Works **anywhere without `sudo`/`CAP_NET_RAW`** to detect **HTTP Floods, API/Login Brute-Force bursts, and automated Bot polling**.

---

## 3. Ways to Run This Model in Production

Look at the latency numbers inside project `onnx_card.json`:
- **Single flow (`batch = 1`):** **`0.0157 ms` (`15.7 microseconds` — ~63,000 flows/sec on 1 CPU core!)**
- **Micro-batch (`batch = 64`):** **`0.124 ms` total (`0.0019 ms` per flow — >500,000 flows/sec!)**

Because the model is tiny (**`815 KB`**, single CPU operator `TreeEnsembleClassifier`, zero GPU required), here are the 4 ways to run it ranked from best fit to most specialized:

1. **In-Process ONNX Runtime (`onnxruntime` CPU Session inside Sidecar / FastAPI Process) — *Recommended***
   - Zero network serialization overhead between feature extraction and scoring (`15 µs` RAM call).
   - Micro-batch every `50–100 ms` during heavy floods to process hundreds of thousands of flows/sec effortlessly.
2. **Standalone Sidecar Container + Redis / Shared Memory Reputation Table (Best for Multi-Worker Gunicorn/Uvicorn)**
   - Sidecar container runs with `network_mode: host` + `cap_add: [NET_RAW, NET_ADMIN]`.
   - Scores flows continuously and writes flagged IPs with TTL (`SETEX block:<ip> 300 '{"family":"Flood","score":0.994}'`) to Redis (and optionally kernel `nftables`/`ipset`).
   - All FastAPI workers read the local/Redis blocklist in $< 0.1\text{ ms}$ without ever slowing down API endpoints.
3. **Pruned Iteration-69 ONNX Model (`model_clean_iter69.onnx`)**
   - Optional upgrade we can generate straight from project `model.joblib`: slices `_predictors[:69]` (the exact peak validation score iteration before late-iteration Newton leaf blowup), cutting latency & model size by **22%** while eliminating logit spikes on boundary inputs.
4. **Native Edge Proxy Plugin (Rust `ort` / Go / C++ ONNX Runtime at Nginx / Envoy)**
   - Same exact `model.onnx` binary loaded directly inside an Envoy filter or Rust reverse proxy if you ever scale beyond `10 Gbps`.

---

## 4. Complete Interface & System Architecture Design

Notice the Golden Rule of this architecture:  
> **Never block an individual HTTP request waiting to accumulate multiple packets.**  
> Instead, **Request Enforcement is synchronous $O(1)$ (`< 5 µs` lookup)**, while **Flow Aggregation & ONNX Scoring run continuously in real time out-of-band** (`~15 µs` per flow snapshot)!

```text
                        INCOMING LIVE CLIENT TRAFFIC
                                     │
            ┌────────────────────────┴────────────────────────┐
            │ (Zero-copy passive tap)                         │ (Normal HTTP path)
            ▼                                                 ▼
┌───────────────────────────────────────┐        ┌──────────────────────────────────┐
│  1. LIVE COLLECTOR LAYER              │        │  4. FASTAPI APPLICATION SERVER   │
│  • Mode A: Libpcap Sidecar Sniffer    │        │                                  │
│    (eth0 / port 8000, L3/L4 packets)  │        │  ┌────────────────────────────┐  │
│  • Mode B: FastAPI ASGI Stream Hook   │──(opt)─┼─▶│ FastAPI Shield Middleware  │  │
│    (receive/send chunk translator)    │        │  │  • Lookup IP in RiskStore  │  │
└───────────────────┬───────────────────┘        │  │    (< 5 µs O(1) memory/Redis) │
                    │ Raw Packet Event           │  │  • Mode = SHADOW (log) or  │  │
                    ▼                            │  │    ENFORCE (HTTP 429/403)  │  │
┌───────────────────────────────────────┐        │  └─────────────┬──────────────┘  │
│  2. STATEFUL FLOW TABLE (In-Memory)   │        │                │ Clean traffic   │
│  • Key: 5-Tuple (ClientIP..Proto)     │        │                ▼                 │
│  • Welford O(1) stats (µs, L4 bytes)  │        │     project Normal API Routes       │
│  • Emits float32[27] snapshot every   │        │     (/login, /orders, /api/...)  │
│    4, 8, 16... pkts or every 250 ms   │        │                                  │
└───────────────────┬───────────────────┘        │  ┌────────────────────────────┐  │
                    │ X: float32[N, 27]          │  │ Control & Monitoring APIs  │  │
                    ▼                            │  │  • GET  /security/status   │  │
┌───────────────────────────────────────┐        │  │  • POST /security/score    │  │
│  3. ONNX DETECTOR & GATE ENGINE       │        │  │  • POST /security/pcap     │  │
│  • onnxruntime (model.onnx, 15 µs)    │        │  │  • PATCH /security/config  │  │
│  • Score = 1.0 - P("Benign")          │        │  │  • WS   /security/live     │  │
│  • Alert if Score >= τ (0.921056)     │───────▶│  └────────────────────────────┘  │
└───────────────────────────────────────┘ Updates└──────────────────────────────────┘
                                       RiskStore /
                                     Optional ipset
```

### Clean Module & API Interface Specification

1. **Module 1 — `FlowFeatureExtractor` (`flow_extractor.py`)**
   - Maintains active flows with LRU eviction + active timeout (`120s`) + idle detection (`5s` = `5,000,000 µs`).
   - Exposes a single clean method:
     ```python
     extractor.ingest_packet(
         client_ip: str, client_port: int,
         server_ip: str, server_port: int,
         proto: int, is_forward: bool,
         payload_len: int, tcp_flags: int, timestamp_us: int
     ) -> Optional[np.ndarray]  # returns float32(27,) whenever snapshot trigger fires
     ```
2. **Module 2 — `ONNXIntrusionEngine` (`detector.py`)**
   - Verifies SHA-256 of `model.onnx` on startup against `model_metadata.json`.
   - Runs batched/single inference and applies the downstream gate outside the ONNX graph:
     ```python
     # Output schema (DecisionResult)
     {
       "alert": True,
       "attack_score": 0.9942,          # 1.0 - P(Benign)
       "threshold_tau": 0.921055988133361,
       "predicted_family": "BruteForce", # argmax over [Benign, Bot, BruteForce, Flood]
       "probabilities": {
         "Benign": 0.0058,
         "Bot": 0.0011,
         "BruteForce": 0.9890,
         "Flood": 0.0041
       },
       "latency_ms": 0.016
     }
     ```
3. **Module 3 — `FastAPIShieldMiddleware` + Sidecar Collector (`middleware.py` & `sidecar_sniffer.py`)**
   - **`FastAPIShieldMiddleware`**: Drop-in one-liner for any existing FastAPI app:
     ```python
     app.add_middleware(FlowShieldMiddleware, engine=engine, mode="shadow")  # or mode="enforce"
     ```
   - Supports **both** live raw packet capture (`sidecar_sniffer.py` watching FastAPI's port via `libpcap`/Scapy) **and** built-in ASGI live stream translation so you can test and run it immediately even without `sudo`!
4. **Module 4 — Live Control Plane & Simulator Endpoints (Mounted at `/security/*` + Live Dashboard)**
   - `GET /security/health` — Model hash verification, opset, active flows count, mode (`shadow` / `enforce`), threshold $\tau$.
   - `POST /security/score-flow` — Score raw 27-feature JSON by name (validates missing fields, NaNs/`inf`, exact ordering).
   - `POST /security/analyze-pcap` — Upload a `.pcap` file to replay packets through the exact same `FlowFeatureExtractor` and inspect all detected attacks.
   - `PATCH /security/config` — Live-tune `tau` (`0.50`–`0.999`), switch between `shadow` and `enforce` mode, whitelist/unban IPs.
   - `WS /security/stream` + `GET /dashboard` — Real-time live traffic & attack monitor showing incoming requests, live extracted 27 features, probabilities across `Benign` / `Flood` / `Bot` / `BruteForce`, and one-click live attack simulators (Normal User vs HTTP Flood vs Login Brute-Force vs Bot Beacon) hitting real FastAPI endpoints!
