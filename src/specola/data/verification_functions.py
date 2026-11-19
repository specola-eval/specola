"""Deterministic functions for the packaged installation fixture."""
from specola.spec_functions import FunctionTarget


def prepare(*, inputs, parameters):
    return [FunctionTarget(str(owner), entity_id=owner, label=f"fixture object {owner}")
            for owner in inputs["samples"]]


def twice(*, inputs, parameters, cache, targets):
    return {key: {"verification_twice_mass": 2 * inputs["samples"][target.entity_id]}
            for key, target in targets.items()}


setattr(twice, "prepare", prepare)
