# DATA QUALITY VERDICT (gate to feature extraction)

## Row accounting

- Parsed rows (Step 3B): **16,137,183**
- ok-shape rows (Step 3A): 16,137,183
- short-row: 0
- long-row: 0
- terminal-junk: 0
- empty-line: 0
- mid-file-header: 0
- Exact duplicate rows: 437,928
- Missing-label rows: 0
- Label-conflict rows: 12,396
- **Usable rows ≈ 15,686,859**

## Column decisions (all 83)

- DROP (8): Bwd PSH Flags, Bwd URG Flags, Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg
- REVIEW (14): Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, Flow Pkts/s, Flow IAT Mean, Flow IAT Max, Flow IAT Min, Fwd IAT Tot, Fwd IAT Mean, Fwd IAT Max, Fwd IAT Min, Subflow Fwd Pkts, Subflow Bwd Pkts, Fwd Act Data Pkts
- OK (61): rest

## 32-feature status

| feature | missing | inf | neg* | nonnum | verdict |
|---|---|---|---|---|---|
| Flow Duration | 0.00% | 0 | 14 | 0 | REVIEW |
| Tot Fwd Pkts | 0.00% | 0 | 1226 | 0 | REVIEW |
| Tot Bwd Pkts | 0.00% | 0 | 17 | 0 | REVIEW |
| TotLen Fwd Pkts | 0.00% | 0 | 0 | 0 | OK |
| TotLen Bwd Pkts | 0.00% | 0 | 0 | 0 | OK |
| Fwd Pkt Len Max | 0.00% | 0 | 0 | 0 | OK |
| Fwd Pkt Len Mean | 0.00% | 0 | 0 | 0 | OK |
| Bwd Pkt Len Max | 0.00% | 0 | 0 | 0 | OK |
| Bwd Pkt Len Mean | 0.00% | 0 | 0 | 0 | OK |
| Flow Byts/s | 0.00% | 0 | 0 | 0 | OK |
| Flow Pkts/s | 0.00% | 0 | 14 | 0 | REVIEW |
| Flow IAT Mean | 0.00% | 0 | 14 | 0 | REVIEW |
| Flow IAT Std | 0.00% | 0 | 0 | 0 | OK |
| Flow IAT Max | 0.00% | 0 | 3 | 0 | REVIEW |
| Fwd IAT Std | 0.00% | 0 | 0 | 0 | OK |
| Bwd IAT Mean | 0.00% | 0 | 0 | 0 | OK |
| Fwd Header Len | 0.00% | 0 | 0 | 0 | OK |
| Bwd Header Len | 0.00% | 0 | 0 | 0 | OK |
| Pkt Len Mean | 0.00% | 0 | 0 | 0 | OK |
| Pkt Len Std | 0.00% | 0 | 0 | 0 | OK |
| Pkt Size Avg | 0.00% | 0 | 0 | 0 | OK |
| Init Fwd Win Byts | 0.00% | 0 | 0 | 0 | OK |
| Init Bwd Win Byts | 0.00% | 0 | 0 | 0 | OK |
| Fwd Act Data Pkts | 0.00% | 0 | 1220 | 0 | REVIEW |
| Fwd Seg Size Min | 0.00% | 0 | 0 | 0 | OK |
| Down/Up Ratio | 0.00% | 0 | 0 | 0 | OK |
| SYN Flag Cnt | 0.00% | 0 | 0 | 0 | OK |
| RST Flag Cnt | 0.00% | 0 | 0 | 0 | OK |
| PSH Flag Cnt | 0.00% | 0 | 0 | 0 | OK |
| ACK Flag Cnt | 0.00% | 0 | 0 | 0 | OK |
| Idle Mean | 0.00% | 0 | 0 | 0 | OK |
| Active Mean | 0.00% | 0 | 0 | 0 | OK |

*neg excludes the valid -1 sentinel in Init Win columns.

## Labels + class weights

- Benign: 13,390,249 (82.9776%) scale_pos_weight=- nan
- DDOS attack-HOIC: 686,012 (4.2511%) scale_pos_weight=19.52 nan
- DDoS attacks-LOIC-HTTP: 576,191 (3.5706%) scale_pos_weight=23.24 nan
- DoS attacks-Hulk: 461,912 (2.8624%) scale_pos_weight=28.99 nan
- Bot: 286,191 (1.7735%) scale_pos_weight=46.79 nan
- FTP-BruteForce: 193,354 (1.1982%) scale_pos_weight=69.25 nan
- SSH-Bruteforce: 187,589 (1.1625%) scale_pos_weight=71.38 nan
- Infilteration: 160,639 (0.9955%) scale_pos_weight=83.36 nan
- DoS attacks-SlowHTTPTest: 139,890 (0.8669%) scale_pos_weight=95.72 nan
- DoS attacks-GoldenEye: 41,508 (0.2572%) scale_pos_weight=322.59 nan
- DoS attacks-Slowloris: 10,990 (0.0681%) scale_pos_weight=1218.4 nan
- DDOS attack-LOIC-UDP: 1,730 (0.0107%) scale_pos_weight=7740.03 RARE(<5000)
- Brute Force -Web: 611 (0.0038%) scale_pos_weight=21915.3 RARE(<5000)
- Brute Force -XSS: 230 (0.0014%) scale_pos_weight=58218.47 RARE(<5000)
- SQL Injection: 87 (0.0005%) scale_pos_weight=153910.91 RARE(<5000)

## Cleaning rules frozen for extraction

- [ ] Quarantine rejects kept in rejects/ (never silently dropped)
- [ ] Duplicates: drop keep-first; conflicts: quarantine
- [ ] Missing Label rows: drop
- [ ] inf -> cap to finite max + add had_inf flag (do in extraction)
- [ ] -1 in Init Win kept as valid category, not missing
- [ ] Day/Hour/Minute/Second + Dst Port/Protocol: analysis only, train behaviour-only + with-port variants and report both

## Gate

Proceed to feature extraction ONLY if: usable rows counted, every DROP/REVIEW above has an owner decision, and no empty section remains in this file.