"""Structural causal model primitives for SignalForge.

The module provides a deterministic, dependency-light kernel for causal queries.
It is intentionally small enough to be embedded in agents while still exposing the
main engineering seams needed by a production causal stack:

* structural equations over exogenous noise terms;
* explicit ``do(...)`` interventions;
* A2P (Abduct-Act-Predict) counterfactual traces; and
* do-calculus helper rules for query rewriting and audit trails.

The implementation does not attempt to be a complete symbolic causal inference
engine. Instead, it supplies audited primitives that higher-level LLM agents can
call while keeping graph validation, intervention semantics, and counterfactual
state transitions deterministic and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

State = dict[str, Any]
StructuralEquation = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class CausalVariable:
    """A node in the SCM graph.

    Args:
        name: Stable variable identifier.
        parents: Endogenous parent variables required by the structural equation.
        description: Optional human-readable semantics for LLM audit prompts.
    """

    name: str
    parents: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CausalVariable.name must be non-empty")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError(f"duplicate parents declared for {self.name!r}")
        if self.name in self.parents:
            raise ValueError(f"{self.name!r} cannot be its own parent")


@dataclass(frozen=True, slots=True)
class Intervention:
    """A Pearl-style ``do`` intervention.

    ``assignments`` replaces structural equations for the named variables during
    the action/prediction phase.
    """

    assignments: Mapping[str, Any]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.assignments:
            raise ValueError("Intervention requires at least one assignment")


@dataclass(frozen=True, slots=True)
class CounterfactualQuery:
    """Counterfactual request executed through the A2P pipeline."""

    evidence: Mapping[str, Any]
    intervention: Intervention
    targets: tuple[str, ...]
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("CounterfactualQuery requires at least one target")


@dataclass(frozen=True, slots=True)
class A2PTrace:
    """Auditable result of an Abduct-Act-Predict counterfactual run."""

    abduced_noise: Mapping[str, Any]
    factual_state: Mapping[str, Any]
    intervention: Intervention
    counterfactual_state: Mapping[str, Any]
    targets: Mapping[str, Any]
    notes: tuple[str, ...] = ()


class DoCalculusRule(str, Enum):
    """Named do-calculus rewrite rules used for explainable query planning."""

    INSERT_DELETE_OBSERVATIONS = "rule_1_insert_delete_observations"
    ACTION_OBSERVATION_EXCHANGE = "rule_2_action_observation_exchange"
    INSERT_DELETE_ACTIONS = "rule_3_insert_delete_actions"


@dataclass(frozen=True, slots=True)
class DoCalculusStep:
    """A single symbolic do-calculus rewrite step."""

    rule: DoCalculusRule
    before: str
    after: str
    justification: str


class StructuralCausalModel:
    """A deterministic structural causal model.

    Variables are evaluated in topological order. Structural equations receive
    two mappings: the partially built endogenous state and the exogenous noise
    values. Equations should be pure functions to keep counterfactual traces
    reproducible.
    """

    def __init__(
        self,
        variables: Sequence[CausalVariable],
        equations: Mapping[str, StructuralEquation],
        *,
        default_noise: Mapping[str, Any] | None = None,
    ) -> None:
        if not variables:
            raise ValueError("StructuralCausalModel requires at least one variable")

        self.variables = tuple(variables)
        self._variable_by_name = {variable.name: variable for variable in self.variables}
        if len(self._variable_by_name) != len(self.variables):
            raise ValueError("variable names must be unique")

        missing_equations = set(self._variable_by_name) - set(equations)
        if missing_equations:
            raise ValueError(f"missing structural equations: {sorted(missing_equations)}")

        unknown_equations = set(equations) - set(self._variable_by_name)
        if unknown_equations:
            raise ValueError(f"equations declared for unknown variables: {sorted(unknown_equations)}")

        for variable in self.variables:
            unknown_parents = set(variable.parents) - set(self._variable_by_name)
            if unknown_parents:
                raise ValueError(
                    f"{variable.name!r} references unknown parents: {sorted(unknown_parents)}"
                )

        self.equations = dict(equations)
        self.default_noise = dict(default_noise or {})
        self._order = self._topological_order()

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return variable names in topological evaluation order."""

        return tuple(variable.name for variable in self._order)

    def evaluate(
        self,
        *,
        noise: Mapping[str, Any] | None = None,
        intervention: Intervention | Mapping[str, Any] | None = None,
    ) -> State:
        """Evaluate the SCM under optional exogenous noise and intervention."""

        merged_noise = {**self.default_noise, **dict(noise or {})}
        assignments = self._normalise_intervention(intervention)
        self._validate_known_variables(assignments, label="intervention")

        state: State = {}
        for variable in self._order:
            if variable.name in assignments:
                state[variable.name] = assignments[variable.name]
                continue

            missing_parents = [parent for parent in variable.parents if parent not in state]
            if missing_parents:
                raise RuntimeError(
                    f"parents for {variable.name!r} were not evaluated: {missing_parents}"
                )
            state[variable.name] = self.equations[variable.name](state, merged_noise)
        return state

    def counterfactual(self, query: CounterfactualQuery) -> A2PTrace:
        """Run Abduct-Act-Predict for a counterfactual query.

        Abduction is intentionally conservative: observed evidence overrides
        matching exogenous noise keys and factual endogenous variables. This lets
        callers inject inferred latent root causes from an LLM or probabilistic
        backend while preserving deterministic SCM execution.
        """

        self._validate_known_variables(query.evidence, label="evidence")
        self._validate_known_variables(query.intervention.assignments, label="intervention")
        self._validate_known_variables({target: None for target in query.targets}, label="target")

        abduced_noise = self.abduce(query.evidence, query.context)
        factual_state = self.evaluate(noise=abduced_noise)
        factual_state.update(dict(query.evidence))
        counterfactual_state = self.evaluate(
            noise=abduced_noise,
            intervention=query.intervention,
        )
        targets = {target: counterfactual_state[target] for target in query.targets}

        notes = (
            "abduct: reconcile evidence with latent context",
            "act: replace intervened structural equations with constants",
            "predict: re-evaluate descendants under the abduced context",
        )
        return A2PTrace(
            abduced_noise=abduced_noise,
            factual_state=factual_state,
            intervention=query.intervention,
            counterfactual_state=counterfactual_state,
            targets=targets,
            notes=notes,
        )

    def abduce(
        self,
        evidence: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> State:
        """Infer latent context from evidence.

        The default implementation is a deterministic hook: it merges
        ``default_noise`` with optional latent ``context`` values and mirrors
        evidence into ``observed_<name>`` keys for downstream structural equations
        that want to condition on observations. Probabilistic abductors can be
        layered above this method without changing the A2P interface.
        """

        abduced = {**self.default_noise, **dict(context or {})}
        for name, value in evidence.items():
            abduced[f"observed_{name}"] = value
        return abduced

    def do_calculus_plan(
        self,
        *,
        outcome: str,
        intervention: Intervention,
        observed: Iterable[str] = (),
    ) -> tuple[DoCalculusStep, ...]:
        """Return an auditable symbolic plan for a causal query.

        This planner records safe, human-readable rewrites. It avoids claiming
        identifiability when graph separation has not been proven by a dedicated
        symbolic engine.
        """

        self._validate_known_variables({outcome: None}, label="outcome")
        self._validate_known_variables(intervention.assignments, label="intervention")
        self._validate_known_variables({name: None for name in observed}, label="observed")

        do_terms = ", ".join(f"do({name}={value!r})" for name, value in intervention.assignments.items())
        observed_terms = ", ".join(observed) or "∅"
        before = f"P({outcome} | {do_terms}, observed={observed_terms})"
        after = before

        steps: list[DoCalculusStep] = []
        if observed_terms != "∅":
            after = f"P({outcome} | {do_terms}, admissible_observations={observed_terms})"
            steps.append(
                DoCalculusStep(
                    rule=DoCalculusRule.INSERT_DELETE_OBSERVATIONS,
                    before=before,
                    after=after,
                    justification=(
                        "Track observations explicitly; deletion requires a verified "
                        "d-separation proof in the mutilated graph."
                    ),
                )
            )

        final = f"A2P[{after}]"
        steps.append(
            DoCalculusStep(
                rule=DoCalculusRule.INSERT_DELETE_ACTIONS,
                before=after,
                after=final,
                justification=(
                    "Represent the intervention by replacing affected structural "
                    "equations before prediction."
                ),
            )
        )
        return tuple(steps)

    def parents_of(self, name: str) -> tuple[str, ...]:
        """Return the direct parents of a variable."""

        self._validate_known_variables({name: None}, label="variable")
        return self._variable_by_name[name].parents

    def _topological_order(self) -> tuple[CausalVariable, ...]:
        ordered: list[CausalVariable] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(variable: CausalVariable) -> None:
            if variable.name in permanent:
                return
            if variable.name in temporary:
                raise ValueError(f"cycle detected at {variable.name!r}")

            temporary.add(variable.name)
            for parent in variable.parents:
                visit(self._variable_by_name[parent])
            temporary.remove(variable.name)
            permanent.add(variable.name)
            ordered.append(variable)

        for variable in self.variables:
            visit(variable)
        return tuple(ordered)

    @staticmethod
    def _normalise_intervention(
        intervention: Intervention | Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if intervention is None:
            return {}
        if isinstance(intervention, Intervention):
            return intervention.assignments
        return intervention

    def _validate_known_variables(self, values: Mapping[str, Any], *, label: str) -> None:
        unknown = set(values) - set(self._variable_by_name)
        if unknown:
            raise ValueError(f"unknown {label} variable(s): {sorted(unknown)}")
