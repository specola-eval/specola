"""Versioned JSON projection for one selected binding."""

from __future__ import annotations

import json
import math

from .run_state import BindingSnapshot, recovery_transitions


def _number(value):
    if value is None or math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def binding_document(binding: BindingSnapshot) -> dict:
    recoveries = recovery_transitions(binding.transitions)
    return {
        "schema_version": 1,
        "event": "monitor_snapshot",
        "step": binding.step,
        "sim_time": _number(binding.simulator_time),
        "specification": binding.specification,
        "monitor": binding.manager_name,
        "spec": {
            "name": binding.specification,
            "authored_formula": binding.authored_formula,
            "resolved_formula": binding.resolved_formula,
            "backend": binding.backend,
            "window_steps": binding.window_steps,
        },
        "state": {
            "status": binding.state.value,
            "active": binding.active,
            "rho": _number(binding.current_robustness),
            "worst_rho": _number(binding.worst_robustness),
            "first_violation_step": binding.first_violation_step,
            "recovery_count": len(recoveries),
            "last_recovery_step": (
                recoveries[-1].step if recoveries else None
            ),
            "current_verdict_final": binding.final,
            "segment_samples": binding.segment_samples,
            "window_steps": binding.window_steps,
        },
        "binding": {
            "binding_key": binding.binding_key,
            "display": binding.label,
            "entity_ids": list(binding.entity_ids),
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "class": entity.entity_class,
                    "label": entity.label,
                    "role": entity.role,
                    "role_basis": entity.role_basis,
                    "role_provenance": entity.role_provenance,
                    "declared_name": entity.declared_name,
                }
                for entity in binding.entities
            ],
            "components": [
                component.key for component in binding.components
            ],
            "component_details": [
                {
                    "entity_id": component.entity_id,
                    "key": component.key,
                    "value": _number(component.value),
                    "unit": component.unit,
                    "tags": list(component.tags),
                }
                for component in binding.components
            ],
        },
        "robustness": {
            "current": _number(binding.current_robustness),
            "final": _number(binding.final_robustness),
            "final_step": binding.final_step,
            "final_time": _number(binding.final_time),
            "worst": _number(binding.worst_robustness),
            "worst_step": binding.worst_step,
            "worst_time": _number(binding.worst_time),
            "unit": binding.unit,
            "current_verdict_final": binding.final,
            "segment_samples": binding.segment_samples,
            "window_steps": binding.window_steps,
        },
        "error": binding.error_message,
        "history": {
            "first_violation_step": binding.first_violation_step,
            "first_violation_time": _number(binding.first_violation_time),
            "recovery_count": len(recoveries),
            "last_recovery_step": (
                recoveries[-1].step if recoveries else None
            ),
            "transitions": [
                {
                    "step": transition.step,
                    "simulator_time": _number(transition.simulator_time),
                    "from": (
                        transition.from_state.value
                        if transition.from_state is not None else None
                    ),
                    "to": transition.to_state.value,
                }
                for transition in binding.transitions
            ],
        },
    }


def binding_json(binding: BindingSnapshot) -> str:
    return json.dumps(
        binding_document(binding),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
