# Copyright 2023 The pynsy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper analyzer that captures an enriched tensor-shape-inference trace.

This analyzer does *not* modify the inference engine. It delegates event
capture to ``tensor_shape_inference`` (so the captured trace is identical to
what the real analysis consumes), then reconstructs the five-stage reasoning
pipeline at termination and serializes it to JSON for the web visualizer.
"""

import collections
from datetime import datetime
import functools
import json

from pynsy.analyses import util
from pynsy.instrumentation import logging_utils
from pynsy.instrumentation import module_loader
from pynsy.instrumentation.util import ObjectId
from pynsy.type_inference import inference_engine
from pynsy.type_inference import tensor_shape_inference as tsi

AbstractState = inference_engine.AbstractState
CommonUtils = inference_engine.CommonUtils

log = logging_utils.logger(__name__)


# --- Decision (A): capture hyper-parameter perturbation by wrapping ---------
#
# ``hyper_parameter`` bumps its argument in place and only the *final* value
# reaches the record list, so the original is lost by termination time. We wrap
# it at import (analyzers load before the demo runs) to record the before/after.

perturbation_log = []
_original_hyper_parameter = tsi.hyper_parameter


@functools.wraps(_original_hyper_parameter)
def _wrapped_hyper_parameter(dim_int, dim_name):
  original = dim_int
  perturbed = _original_hyper_parameter(dim_int, dim_name)
  perturbation_log.append({
      "dim_name": dim_name,
      "original_value": original,
      "perturbed_value": perturbed,
      "was_bumped": perturbed != original,
  })
  return perturbed


tsi.hyper_parameter = _wrapped_hyper_parameter


# --- Analyzer interface: delegate capture to tensor_shape_inference ---------


def abstraction(obj):
  return tsi.abstraction(obj)


def process_event(record):
  tsi.process_event(record)


# --- Decision (B): derive the solving trace by *observing* the real solver --


def derive_solving_trace(
    solution,
    templates,
    var_to_location,
    location_id_to_state_list,
):
  """Re-derive which template held at which location, read-only.

  The real solver (``CommonUtils.find_solution``) has already run and produced
  ``solution``: a list indexed by ``var_id`` where ``solution[v]`` is a
  ``TemplateInstance``. A non-identity instance means var ``v`` was explained by
  ``template`` applied to ``[v] + solution[v].vars``. This function does NOT
  re-run the solver; it inspects the finished solution and re-checks the
  predicate against the observed states so the visualizer can show the evidence.

  Args:
    solution: list[TemplateInstance], indexed by var_id (vars are still ints).
    templates: the list of Template objects (the identity template is separate,
      at CommonUtils.identity_template).
    var_to_location: dict[var_id -> location_id], the "home" location whose
      observed states gauge whether a template holds for that var.
    location_id_to_state_list: dict[location_id -> list[state dict]], where each
      state maps var_id -> observed integer dimension.

  Returns:
    list[dict] with keys: template, target_var, candidate_vars, location_id,
    held, states_checked.
  """
  trace = []
  identity = CommonUtils.identity_template
  for target_var, instance in enumerate(solution):
    template = instance.get_template()
    if template is identity:
      continue
    location_id = var_to_location.get(target_var)
    if location_id is None:
      continue
    vars_tuple = [target_var] + list(instance.vars)
    states = location_id_to_state_list.get(location_id, [])
    held = all(template.predicate(state, vars_tuple) for state in states)
    trace.append({
        "template": template.get_name(),
        "target_var": target_var,
        "candidate_vars": list(instance.vars),
        "location_id": location_id,
        "held": held,
        "states_checked": len(states),
    })
  return trace


# --- JSON serialization helpers ---------------------------------------------


def _clean_value(value):
  """Normalize a single result_and_args entry for JSON."""
  vid = value.get("id")
  abs_value = value.get("abs")
  if isinstance(abs_value, (list, tuple)):
    abs_value = [int(x) if isinstance(x, int) else x for x in abs_value]
  return {
      "id": vid.id if isinstance(vid, ObjectId) else vid,
      "abs": abs_value,
  }


def _serialize_events(record_list):
  events = []
  for i, rec in enumerate(record_list):
    result_and_args = [
        _clean_value(v)
        for v in rec.get("result_and_args", [])
        if isinstance(v, dict)
    ]
    events.append({
        "index": i,
        "module_name": rec.get("module_name"),
        "lineno": rec.get("lineno"),
        "type": rec.get("type"),
        "name": rec.get("name", ""),
        "indentation": rec.get("indentation", 0),
        "result_and_args": result_and_args,
    })
  return events


def _read_source_files(module_names):
  source_files = {}
  for module_name in module_names:
    try:
      module = module_loader.import_method_from_module(module_name)
      path = module.__file__
      with open(path, "r") as f:
        lines = f.read().split("\n")
      source_files[module_name] = {"path": path, "lines": lines}
    except Exception as e:  # pragma: no cover - best effort
      log(f"Could not read source for {module_name}: {e}")
  return source_files


# --- Termination: reconstruct the five-stage pipeline -----------------------


def process_termination():
  record_list = tsi.record_list
  if not record_list:
    log("No instructions were instrumented.")
    return

  abstract_state = AbstractState(tsi.TensorShapeInferenceUtils)
  abstract_state.create_var_ids_and_global_state(record_list)
  abstract_state.create_local_states(record_list)
  (
      location_id_to_state_list,
      global_state,
      location_id_to_var_ids_and_values,
      location_to_id,
      fresh_var_generator,
      var_id_to_annotation,
      location_id_to_name,
      _location_id_to_record_list_index,
  ) = abstract_state.get_data()

  # Stage 2: run the genuine solver, then observe it (Decision B).
  solution = CommonUtils.find_solution(
      fresh_var_generator.num_ids(),
      tsi.TensorShapeInferenceUtils.templates,
      global_state,
      location_id_to_state_list,
      location_id_to_var_ids_and_values,
  )
  equivalence_classes = CommonUtils.get_equivalence_classes(solution)
  fresh_var_generator.set_annotations(
      equivalence_classes, var_id_to_annotation, lambda x: f"d{x}"
  )

  var_to_location = {}
  for location_id, vv in location_id_to_var_ids_and_values.items():
    for var_id in vv.var_ids:
      var_to_location[var_id] = location_id

  solving_trace = derive_solving_trace(
      solution,
      tsi.TensorShapeInferenceUtils.templates,
      var_to_location,
      location_id_to_state_list,
  )

  # Stage 1: variable assignments.
  var_assignments = []
  for location_id, vv in location_id_to_var_ids_and_values.items():
    key = location_to_id.get_key(location_id)
    var_assignments.append({
        "location_id": location_id,
        "location_key": list(key) if key is not None else None,
        "var_ids": list(vv.var_ids),
        "observed_values": [list(v["abs"]) for v in vv.values],
        "num_observations": len(vv.values),
    })

  # Stage 3: equivalence classes (capture int membership before mutating).
  eq_classes_json = []
  for class_id, members in enumerate(equivalence_classes):
    if not members:
      continue
    representative = next(iter(members))
    eq_classes_json.append({
        "class_id": class_id,
        "var_ids": sorted(members),
        "annotation": fresh_var_generator.get_annotation(representative),
    })

  # Stage 5: annotated source. Mirror tensor_shape_inference: rewrite each
  # template instance's vars to their annotations, then render per location.
  for rhs in solution:
    rhs.vars = [
        fresh_var_generator.get_annotation(var_id) for var_id in rhs.vars
    ]

  annotations = collections.defaultdict(lambda: collections.defaultdict(list))
  module_names = set()
  for location_id, vv in location_id_to_var_ids_and_values.items():
    module_name, _method_id, _instruction_id, line_number, opcode = (
        location_to_id.get_key(location_id)
    )
    module_names.add(module_name)
    name = location_id_to_name[location_id]
    symbolic_shape = [repr(solution[x]) for x in vv.var_ids]
    annotations[module_name][str(int(line_number))].append({
        "name": CommonUtils.get_nickname(opcode, name),
        "symbolic_shape": symbolic_shape,
        "concrete_shapes": [list(v["abs"]) for v in vv.values],
    })

  trace = {
      "metadata": {
          "timestamp": datetime.now().isoformat(),
          "modules": sorted(module_names),
      },
      "events": _serialize_events(record_list),
      "source_files": _read_source_files(module_names),
      "analysis": {
          "var_assignments": var_assignments,
          "solving_trace": solving_trace,
          "equivalence_classes": eq_classes_json,
          "hyper_parameters": list(perturbation_log),
          "annotations": {m: dict(lines) for m, lines in annotations.items()},
      },
  }

  out_path = util.get_output_path("visualizer", "visualizer_trace.json")
  with open(out_path, "w") as f:
    json.dump(trace, f, indent=2, default=lambda o: o.id
              if isinstance(o, ObjectId) else repr(o))
  log(f"Saved visualizer trace to {out_path}.")
