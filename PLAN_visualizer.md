# Pynsy Analysis Visualizer — Implementation Plan

## Context

Pynsy produces rich execution traces (event records with opcodes, heap IDs, abstract values, call stacks) but outputs flat CSV traces and terminal text. We want a **web-based HUD** inspired by Geoffrey Litt's blog posts — a bespoke visualization that lets you *see* execution flow and analysis reasoning, not just read a trace file.

Primary goal: visualize **tensor shape inference** end-to-end — from raw execution events, through constraint solving, to symbolic shape annotations — including the **hyper_parameter perturbation** that prevents false positive unification of numerically-equal-but-semantically-distinct dimensions.

## Architecture: Python Backend + Single HTML/JS Frontend

**Answering "Python vs JS?"**: Python serves the data. The browser renders it. There's no way around *some* JS for interactive step-through in a browser, but we keep it minimal: ~300-400 lines of vanilla JS embedded in a single HTML file. No React, no npm, no build step. Python `http.server` serves everything.

```
pynsy analysis run                Browser (http://localhost:8080)
       │                                    │
  trace_capture.py                    index.html
  (wrapper analyzer)                  (embedded CSS + JS)
       │                                    │
       ▼                                    │
  visualizer_trace.json ──── server.py ─────┘
  (outdir/visualizer/)     (http.server)
```

## Decisions (locked after code review)

These supersede the original assumptions, having been checked against the actual
framework (`operator_apply.py`, `module_loader.py`, `inference_engine.py`):

- **(A) Perturbation capture via wrapping.** `hyper_parameter()` destroys the
  original value (it bumps `dim_int` in place and only the *final* value reaches
  `record_list`). So `trace_capture` **monkey-patches
  `tensor_shape_inference.hyper_parameter` at import time**. Analyzers are
  imported during `instrument_imports()` — which runs *before* the demo module
  executes — so the wrap is installed before mnist2's top-level
  `hyper_parameter(...)` calls fire. The wrapper records
  `(name, original, perturbed, was_bumped)`.
- **(B) Observe the real solver, don't reimplement it.** We call the genuine
  `CommonUtils.find_solution()`, then derive the "which template held at which
  location" trace in a **read-only second pass** over the produced `solution` +
  `location_id_to_state_list`. This guarantees the visualized constraints match
  the real terminal output (a verification goal). No forked solver.
- **(C) Capture + JSON first, HTML second.** Build and verify
  `visualizer_trace.json` against real mnist2 output before writing any UI.
- **Delegate capture, don't reimplement.** `trace_capture.abstraction` and
  `trace_capture.process_event` call straight through to
  `tensor_shape_inference`'s versions. Dispatch is index-based
  (`retrieve_record_element(record, i)` in operator_apply.py:160), so each
  analyzer must own its abstraction slot; delegating guarantees our captured
  trace is identical to what the real analysis consumes.

## Phase 1: Quick Prototype (what we build now)

### New Files

| File | Purpose |
|------|---------|
| `pynsy/visualizer/__init__.py` | Package init |
| `pynsy/visualizer/trace_capture.py` | Wrapper analyzer — captures enriched trace + analysis reasoning as JSON |
| `pynsy/visualizer/server.py` | HTTP server — serves HTML + JSON trace |
| `pynsy/visualizer/index.html` | Single-page visualizer UI |
| `configs/visualizer.toml` | Config to register the visualizer analyzer |

### 1. `trace_capture.py` — Wrapper Analyzer

Implements the standard 3-function analyzer interface (`abstraction`, `process_event`, `process_termination`) wrapping tensor_shape_inference. Key design: **does not modify inference_engine.py**. Instead, it calls the existing pipeline and captures intermediate state at each stage.

**`process_termination()` captures 5 stages:**

1. **Variable assignment** — After `AbstractState.create_var_ids_and_global_state()`: which fresh variables (v0, v1, ...) map to which source locations and dimensions, what concrete values were observed
2. **Constraint solving** — A reimplementation of `CommonUtils.find_solution()` (~30 lines) with added logging: which template was tried on which variable pair, did it hold for all observations, was it assigned
3. **Equivalence classes** — After `get_equivalence_classes()`: which variables got unified, what annotation each class received
4. **Hyper-parameter perturbation** — From `observed_hyper_parameters` set + record scanning: which values were bumped and why
5. **Annotated source** — Source lines + shape annotations per line (same as existing output but in JSON)

**JSON trace structure:**
```json
{
  "metadata": { "timestamp": "...", "modules": ["pynsy.demos.mnist"] },
  "events": [
    { "index": 0, "module_name": "...", "lineno": 42, "type": "LOAD_FAST",
      "name": "x", "indentation": 2,
      "result_and_args": [{"id": 5, "abs": [128, 784]}] }
  ],
  "source_files": {
    "pynsy.demos.mnist": { "path": "...", "lines": ["import time", ...] }
  },
  "analysis": {
    "var_assignments": [
      { "location_id": 3, "location_key": ["pynsy.demos.mnist", 1, 5, 42, "LOAD_FAST"],
        "var_ids": [0, 1],
        "observed_values": [[128, 784], [128, 784]], "num_observations": 5 }
    ],
    "solving_trace": [
      { "template": "=", "target_var": 0, "candidate_vars": [3],
        "location_id": 3, "held": true, "states_checked": 5 }
    ],
    "equivalence_classes": [
      { "class_id": 0, "var_ids": [0, 3, 7], "annotation": "hidden1" }
    ],
    "hyper_parameters": [
      { "dim_name": "hidden3", "original_value": 1024,
        "perturbed_value": 1025, "was_bumped": true }
    ],
    "annotations": { "pynsy.demos.mnist": { "42": [{"name": "x", "symbolic_shape": ["d0", "d1"], "concrete_shapes": [[128, 784]]}] } }
  }
}
```

### 2. `server.py` — HTTP Server

Python's `http.server`. Serves:
- `/` → `index.html`
- `/api/trace` → the JSON trace file

Run: `python -m pynsy.visualizer.server [--port 8080] [--trace outdir/visualizer/visualizer_trace.json]`

### 3. `index.html` — The HUD

Three-panel layout, dark theme, keyboard-navigable:

```
┌──────────────┬──────────────────────────────────┐
│  Event       │  Source Code                     │
│  Timeline    │  (highlighted current line,      │
│  (scrollable │   shape annotations inline)      │
│   list)      │                                  │
│              ├──────────────────────────────────┤
│              │  Analysis Panel (tabbed):        │
│              │  [Shapes] [Constraints] [FP Fix] │
│              │                                  │
└──────────────┴──────────────────────────────────┘
```

**Tab 1 — Source + Shapes**: Source code with line numbers. Current execution line highlighted. Tensor shapes shown inline below lines (magenta, matching pynsy's terminal convention). Call stack sidebar.

**Tab 2 — Constraint Solving**: Variable assignment table → template matching steps (green=held, red=failed) → equivalence classes with annotations. Cross-highlighting: click a var to highlight all its appearances.

**Tab 3 — False Positive Avoidance**: `hyper_parameter()` calls shown as a before/after comparison. "hidden2=1024, hidden3=1024 → hidden3 bumped to 1025". Shows what would have been falsely unified without perturbation.

**Navigation**: Arrow keys step through events. Timeline scrubber on the left. Keys 1/2/3 switch tabs.

### 4. `configs/visualizer.toml`

```toml
analyzers = ["pynsy.visualizer.trace_capture"]
[instrumentation_rules]
include = ["pynsy.demos."]
exclude = ["pynsy."]
```

### Usage Flow

```bash
# Step 1: Run analysis (produces JSON trace)
python -m pynsy.main --config configs/visualizer.toml --module pynsy.demos.mnist2

# Step 2: View in browser
python -m pynsy.visualizer.server
```

### Key Files to Read/Reuse

| Existing file | What we reuse |
|---|---|
| `pynsy/type_inference/tensor_shape_inference.py` | `abstraction()`, `process_event()`, `hyper_parameter()`, `annotate_shape()`, `observed_hyper_parameters`, source-reading logic (lines 314-363) |
| `pynsy/type_inference/inference_engine.py` | `AbstractState`, `CommonUtils.find_solution()` (lines 141-174, reimplemented with tracing), `get_equivalence_classes()`, `FreshVarIdGenerator` |
| `pynsy/instrumentation/util.py` | `ObjectId` class (for JSON serialization) |
| `pynsy/analyses/util.py` | `get_output_path()` for output directory convention |

### Design Decisions

**Wrapper analyzer vs. modifying inference_engine.py**: The wrapper approach avoids entangling visualization concerns with analysis logic. The inference engine's public API (`get_data()`, `get_equivalence_classes()`) exposes enough intermediate state. Only `find_solution()` (~30 lines) is reimplemented with logging — everything else is delegated.

**Single HTML file vs. separate JS/CSS**: For Phase 1, a single file (~500-800 lines) keeps things simple. No build step, no module bundler. The server can read it from disk or embed it as a string.

**Flat event list vs. tree**: The JSON stores a flat event list. The tree structure (call stack nesting) is reconstructed client-side from `CALL_FUNCTION`/`EXIT_FUNCTION` event types and `indentation` — simpler to serialize and scrub through.

**Custom JSON encoder**: `result_and_args` contains `ObjectId` wrappers and potentially non-serializable objects (JAX arrays). A custom encoder unwraps `ObjectId` to plain ints and converts complex objects to string representations. The `abstraction()` function already reduces objects to shape tuples, so most values serialize trivially.

## Phase 2: Polished Tool (future, high-level)

- **Live execution via WebSocket**: `process_event()` streams events in real-time; analysis panel shows "waiting..." until `process_termination()` fires
- **Shape flow graph**: SVG directed graph showing how tensor shapes propagate through the program (nodes=locations, edges=data flow)
- **Dimension tracking**: Select a dimension (e.g., "batch") to highlight everywhere it appears
- **Timeline zoom**: Canvas-based minimap for large traces (thousands of events), semantic zoom
- **Brushing & linking**: Click anything in any view to cross-highlight related elements in all views
- **Light/dark theme toggle**, syntax highlighting via PrismJS CDN

## Verification

1. Run `python -m pynsy.main --config configs/visualizer.toml --module pynsy.demos.mnist2` — should produce `outdir/visualizer/visualizer_trace.json`
2. Validate JSON structure with `python -c "import json; json.load(open('outdir/visualizer/visualizer_trace.json'))"`
3. Run `python -m pynsy.visualizer.server` — should open browser
4. In browser: step through events, verify source highlighting matches line numbers, verify shapes match existing pynsy terminal output
5. Check constraint solving tab shows the same solution as `tensor_shape_inference` terminal output
6. Check hyper_parameter tab shows perturbation for mnist demo's `layer_sizes`
