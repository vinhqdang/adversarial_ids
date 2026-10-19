# Packet-level replay of FFE adversarial flows

Code: `src/validation/packet_replay.py` (trace generator, CICFlowMeter-style extractor, four packet-level actions, action derivation), `experiments/run_packet_replay.py` (runner), `tests/test_packet_replay.py`. Raw numbers: `results/packet_replay.json`.

Setup: XGBoost target of the main benchmark, FFE with default settings (step budget 20, TSBS on), the benchmark cohort (malicious test flows detected by both XGBoost and LightGBM, 1,000 drawn with seed 42, first N = 300 used). No pcaps: every flow record is expanded into a synthetic packet trace.

## What the replay does

1. Trace generator: from the 12 features, builds n_fwd + n_bwd packets with a forward first packet, random directions, integer-microsecond inter-arrival times that sum to the duration, random payload sizes that sum to `totlen_fwd`/`totlen_bwd`, per-packet transport headers that sum to `fwd_header_len` (20 to 60 B TCP, 8 B UDP), and the initial window on the first forward packet.
2. Extractor: duration = last minus first timestamp, rates = bytes (or packets) over duration, forward mean = forward payload over forward packets, header length = sum of forward transport headers, initial window = window of the first forward packet (0 for UDP).
3. Actions: temporal dilation (all inter-arrival times stretched by one factor, duration grows by exactly the delay), payload padding (spread over forward packets, capped at 1500 - 20 - header bytes per packet), chaff injection (forward packets at random instants inside the flow span, so duration is unchanged), window morphing (first forward packet only). Magnitudes come from the feature displacement exactly as the generators define it: delay = d(duration), N = d(fwd packets), chaff header h0 = 20 B (32 B if the flow already averages >= 26 B per packet, 8 B UDP), window = x_adv[7], padding = d(totlen_fwd) minus what the chaff packets carry. Header growth not explained by chaff is realised as TCP options (never needed on CSE-CIC-IDS2018 and CIC-IDS2017; 4 of 179 successful flows on UNSW-NB15).

Because the generators count every chaff packet as `l0` = 40/52 B (TCP) or 28 B (UDP) of `totlen_fwd`, whereas a minimal chaff packet carries no payload, three conventions are replayed:

* `schema`: chaff carries min(l0, d(totlen)/N) payload bytes, remaining bytes are padding. Reproduces the schema's `totlen_fwd` exactly.
* `empty_compensated`: chaff packets are empty and all of d(totlen) is padding on the other forward packets. Same features as `schema`, with CICFlowMeter-faithful (empty) chaff.
* `empty_literal`: chaff packets are empty and padding is d(totlen) minus N*l0, i.e. the bytes the cost model books to chaff are simply not delivered.

## Round trip, extract(trace(x)) versus x

20,000 random test flows per corpus (benign and malicious), plus the 300 attacked flows.

| Corpus | flows with all 12 features exact (rel. error < 1e-9) | worst-feature error, max |
|---|---|---|
| CSE-CIC-IDS2018 | 100% | 0 |
| CIC-IDS2017 | 100% | 0 |
| UNSW-NB15 | 99.995% (19,999 of 20,000) | 32 (one flow) |

The single failure is a one-packet record whose duration is 1 microsecond: no packet list with one packet has a non-zero duration. The round trip is exact by construction for every feasible record (it tests that the 12 features and the invariants are jointly realisable by a packet list, including the `<= 1500 B` payload bound and the 20-60 B header bound), not an empirical match to a real capture.

## Schema-level FFE versus replayed features

N = 300 per corpus. Schema-level FFE: 99.3 / 99.3 / 59.7% evasion (all feasible), 298 / 298 / 179 successful flows. Feasibility is re-checked with the repository's `verify_physical_feasibility` on the re-extracted vectors: 100% pass in all modes.

Share of successful flows whose replayed 12-vector equals FFE's reported vector within 1% on every feature:

| Corpus | `schema` / `empty_compensated` | `empty_literal` |
|---|---|---|
| CSE-CIC-IDS2018 | 90.9% | 89.6% |
| CIC-IDS2017 | 97.7% | 94.3% |
| UNSW-NB15 | 93.9% | 32.4% |

In `schema` mode the only feature that ever disagrees is `fwd_header_len` (all other features, including both coupled rates, the forward mean, duration and window, match to 1e-6 on every flow). Cause: 27 / 7 / 11 successful flows add packets whose header growth is smaller than N*h0 in the schema vector. TSBS raises the packet count by snapping `tot_fwd_pkts`, and the projection then fills the header sum only up to 20 B per packet, while the original packets averaged more (e.g. 24 B); a real chaff packet adds its own 20 B and leaves the original headers alone. Median error is 0, 99th percentile of the worst-feature error 7.6% / 60% / 40%, maximum 62% / 61% / 40% (a few flows with a large relative header error). Change fidelity for `fwd_header_len` (replayed change versus schema change within 1%) is only 13% / 59% / 97% among flows where it changes; for every other feature it is 100%.

## Do the evasions survive? (XGBoost re-scored on the re-extracted vectors)

| Corpus | schema-level evasions | retained, `schema` and `empty_compensated` | retained, `empty_literal` | replayed ASR (of all 300), `schema` |
|---|---|---|---|---|
| CSE-CIC-IDS2018 | 298 | 99.3% (296; 2 flipped) | 100.0% | 98.7% |
| CIC-IDS2017 | 298 | 100.0% | 99.0% (3 flipped) | 99.3% |
| UNSW-NB15 | 179 | 100.0% | 68.2% (57 flipped) | 59.7% |

The same flows through the unperturbed replayed trace stay detected in 100% of cases (`baseline_trace_still_detected_pct`). Trace seeds change packet sizes and timestamps but not the 12 features, so results are identical across the three seeds run. The two CSE-CIC-IDS2018 flips in `schema` mode are due to the header-length discrepancy above. In `empty_literal` mode the lost evasions come from undelivered chaff bytes (UNSW-NB15 uses chaff on 69% and padding on 86% of its successful flows, so its evasions rely on the exact forward mean and byte rate); if the chaff bytes are carried as extra padding (`empty_compensated`) the evasions are fully retained.

Action usage among successful flows (CSE-CIC-IDS2018 / CIC-IDS2017 / UNSW-NB15): dilation 54 / 34 / 55%, chaff 10 / 6 / 69%, padding 3 / 32 / 86%, window 82 / 98 / 0%.

## Limitations (must be stated)

* The traces are synthetic. Packet sizes, directions and inter-arrival times are random subject to the record's aggregates; no real flow was replayed and nothing was sent over a network. Agreement validates the action semantics and the schema's internal consistency, not that a real sender, receiver, NAT or IDS would produce those features.
* The extractor is my CICFlowMeter-style re-implementation of the 12 features, not CICFlowMeter itself; it ignores flow timeouts and FIN/RST-based flow termination, active/idle timers, TCP retransmission handling, fragmentation, and the exact CICFlowMeter definitions of header length and window (taken from the schema).
* Closure of the loop is partly by construction: `schema` and `empty_compensated` derive the action magnitudes from x_adv so that the packet count, payload total, duration and window are met exactly; the informative checks are that the coupled rates and forward mean recomputed from packets equal the closed-form values, that the MTU headroom suffices (no unmet padding bytes in any replayed flow), that the header bound holds, and the `fwd_header_len` and chaff-payload findings above.
* The schema bounds payload by 1500 B per packet whereas the MTU leaves 1460 B (TCP, 20 B header). The round trip allows the schema's 1500 B; padding in the replay respects the MTU, and never ran out of headroom on the attacked flows. A small fraction of real records still carry packets above the MTU (segmentation offload), which a pure-MTU model would reject.
* Chaff is injected at random instants with a fixed window; a real stack must keep sequence numbers, ACK behaviour and server responses consistent. The server's backward packets and bytes are held fixed, although dilation or chaff would in practice change the server's reaction (retransmissions, RST, duplicate ACKs). Dilation is a uniform stretch; the attack payload's application semantics (does an exploit still work when delayed 100 s, truncated by a timeout, or padded) are not tested.
* Chaff semantics: the generators count 28-52 bytes of `totlen_fwd` per chaff packet although CICFlowMeter's length features count payload only; empty chaff delivers zero payload bytes (`empty_literal`). The evasions survive if those bytes are carried as padding instead. Whether the paper's cost model should price them as chaff or padding is an open modelling choice.
* The window action sets only the first forward packet's window; real stacks advertise a window consistent with later packets and the receive buffer.
* TCP headers are not constrained to multiples of 4 B in the generator, and header-option padding (used on 4 UNSW-NB15 flows) is a fifth action beyond the four generators.
* Only XGBoost on N = 300 flows per corpus was replayed; sampling error on the 100% rates is up to a few points (Wilson 95% lower bound about 98.7% for 298 of 298).

## Suggested supplementary paragraph

**Packet-level replay.** The feasibility notion used in the main text is conformance to the 12-feature flow schema. To check that the four generators correspond to realisable packet operations, we built a synthetic replay. A flow record is expanded into a packet sequence with the stated packet counts, payload and header totals, duration and initial window; a CICFlowMeter-style extractor recomputes the twelve features from the packets. For 20,000 flows per corpus the round trip reproduces the record exactly for all flows on CSE-CIC-IDS2018 and CIC-IDS2017 and for 19,999 of 20,000 on UNSW-NB15 (the exception is a one-packet record with a 1 microsecond duration). We then derived the delay, chaff count, padding and window change from each of the 300 FFE adversarial flows per corpus, applied them to the packets (stretching inter-arrival times, injecting forward packets inside the flow, padding existing packets up to the MTU, morphing the first forward window), and re-extracted the features. All coupled rates, the forward mean, duration and window agree with the schema-level vector to numerical precision; the forward header length differs on 9.1%, 2.3% and 6.1% of the successful flows (up to 62% on single flows) because boundary snapping raises the packet count without accounting for the non-minimal headers of the original packets. Re-scoring the re-extracted vectors with the target XGBoost model preserved 99.3%, 100% and 100% of the schema-level evasions on CSE-CIC-IDS2018, CIC-IDS2017 and UNSW-NB15. The result depends on how chaff bytes are accounted: the generators credit each chaff packet with 28-52 B of forward payload, whereas an empty packet carries none; if those bytes are not delivered, retention falls to 100%, 99.0% and 68.2%, and it is restored when they are sent as padding. The replay validates the action semantics and the schema invariants only: the traces are synthetic, the extractor is our re-implementation, and no TCP state machine, server reaction, offload or middlebox is modelled, so it does not show that an attack payload survives the perturbation on a live network.
