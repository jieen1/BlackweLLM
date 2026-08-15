# Qwen3.6 vs Qwen3.8 — Nebula mission-control evaluation

## Controlled runtime setup

- Backend: qwen36
- Context: 262144 tokens per slot
- External capacity: 1
- Internal slots: 2 (one additional slot required by MTP/CUDA Graph)
- Blocks per slot: 16384, block size 16
- KV cache: FP8 E4M3
- CUDA Graph: enabled
- Prefix cache: enabled
- MTP: enabled, K=3
- Temperature: 0
- Request timeout: 900 seconds
- Prompt: byte-identical source request reused across model runs

Qwen3.6 was originally run at external capacity 4/internal slots 5. Qwen3.8 used
external capacity 1/internal slots 2 as requested. The model/kernel settings are
otherwise identical; throughput here is a single active request, so external
capacity does not change the active decode shape.

## Runtime results

| Model/mode | Result | Prompt tokens | Completion tokens | Wall time | Completion tok/s |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 non-thinking | stop | 1592 | 15896 | 150.40 s | 105.69 |
| Qwen3.6 thinking | stop | 1561 | 37783 | 348.68 s | 108.36 |
| Qwen3.8 non-thinking | stop | 1563 | 39042 | 358.29 s | 108.97 |
| Qwen3.8 thinking, first request | length | 1603 | 65536 | 589.46 s | 111.18 |
| Qwen3.8 thinking continuation | stop | 67172 | 36543 | 356.59 s | 102.48 |

The Qwen3.8 non-thinking run is only about 0.6% faster than the similarly long
Qwen3.6 thinking run. This is effectively the same decode level, not a material
speed improvement. Qwen3.8 thinking required two requests and about 946 seconds
to obtain final HTML.

## Artifact integrity

| Artifact | HTML | JavaScript | Notes |
|---|---|---|---|
| Qwen3.6 non-thinking | complete | parses | Desktop orbital panel overlaps the fleet table |
| Qwen3.6 thinking | complete | parses | Clean light dashboard; minor mobile header clipping |
| Qwen3.8 non-thinking raw | complete | **syntax error** | Missing right parenthesis in moon SVG construction |
| Qwen3.8 non-thinking repaired | complete | parses | One-line repair, preserved separately from raw output |
| Qwen3.8 thinking first request | absent | n/a | 65536 tokens of planning; never produced final code |
| Qwen3.8 thinking continued | complete | parses | Valid two-request recovery artifact |

## Visual assessment

### Qwen3.6 non-thinking

Strong dark mission-control identity, clear KPI hierarchy, dense operational
content, and good mobile card stacking. The desktop orbital panel overlaps the
fleet table, and the 390px header wraps/clips mission metadata.

### Qwen3.6 thinking

Balanced and coherent light dashboard with readable charts, orbital map, and
fleet table. It is less visually distinctive than Qwen3.8 non-thinking, but it
is the cleanest one-request result: complete HTML, valid JavaScript, and usable
desktop/mobile layouts.

### Qwen3.8 non-thinking

The strongest pure visual design: polished cards, excellent typography,
consistent spacing, attractive data graphics, and the best mobile stacking.
However, the raw JavaScript fails to parse. The visual screenshot therefore
uses the explicitly named repaired copy, and the raw model result cannot be
counted as functionally complete.

### Qwen3.8 thinking-aided continuation

Strong desktop information architecture, especially the fleet/recent-events
split and orbital telemetry. Its light palette has weaker contrast, the mobile
top bar is crowded, and the result required a second request after the initial
65K-token thinking loop failed to deliver code.

## Overall ranking

1. **Qwen3.8 non-thinking repaired preview** — best visual quality, but raw output fails JavaScript syntax.
2. **Qwen3.6 thinking** — best clean one-request deliverable.
3. **Qwen3.8 thinking continued** — rich and valid final page, but poor first-request convergence and weaker mobile presentation.
4. **Qwen3.6 non-thinking** — visually strong dark dashboard, but desktop overlap lowers polish.

The main Qwen3.8 gap is not visual capability. It is deliverable reliability:
one-character syntax failure in non-thinking mode and unbounded planning in
thinking mode.
