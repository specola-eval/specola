"""Add immutable function operands to the deterministic mock fixture."""
from dataclasses import replace
from collections.abc import Mapping

from specola.adapters.fake import _MockAdapter
from specola.registry import register
from specola.spec_functions import FunctionInputs, immutable


class VerificationMock(_MockAdapter):
    function_input_schemas = {"specola.verification.mass": 1}

    def make_advancer(self, mode, demo=None, seed=0):
        import time
        def advance(step):
            time.sleep(0.02)
            return True
        return advance

    def _observation(self):
        observation = super()._observation()
        if not getattr(self, "_function_input_requests", {}):
            return observation
        masses = {owner: entity.properties["mass"]
                  for owner, entity in observation.perceived_entities.items()
                  if entity.entity_class == "object"}
        values = immutable({"specola.verification.mass": masses})
        assert isinstance(values, Mapping)
        return replace(observation, function_inputs=FunctionInputs(0, values))


register("verification-mock", lambda *args, **kwargs: VerificationMock())
