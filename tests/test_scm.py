from signalforge.scm import (
    CausalVariable,
    CounterfactualQuery,
    DoCalculusRule,
    Intervention,
    StructuralCausalModel,
)


def build_incident_model() -> StructuralCausalModel:
    return StructuralCausalModel(
        variables=(
            CausalVariable("phishing_pressure"),
            CausalVariable("analyst_load"),
            CausalVariable("triage_delay", parents=("phishing_pressure", "analyst_load")),
            CausalVariable("breach_impact", parents=("triage_delay",)),
        ),
        equations={
            "phishing_pressure": lambda _state, noise: noise.get("phishing_pressure", 0),
            "analyst_load": lambda _state, noise: noise.get("analyst_load", 0),
            "triage_delay": lambda state, _noise: state["phishing_pressure"] + state["analyst_load"],
            "breach_impact": lambda state, _noise: state["triage_delay"] * 10,
        },
        default_noise={"phishing_pressure": 3, "analyst_load": 2},
    )


def test_evaluate_uses_topological_order_and_structural_equations() -> None:
    scm = build_incident_model()

    assert scm.variable_names == (
        "phishing_pressure",
        "analyst_load",
        "triage_delay",
        "breach_impact",
    )
    assert scm.evaluate() == {
        "phishing_pressure": 3,
        "analyst_load": 2,
        "triage_delay": 5,
        "breach_impact": 50,
    }


def test_intervention_replaces_structural_equation() -> None:
    scm = build_incident_model()

    state = scm.evaluate(intervention=Intervention({"triage_delay": 1}))

    assert state["triage_delay"] == 1
    assert state["breach_impact"] == 10


def test_counterfactual_returns_a2p_trace() -> None:
    scm = build_incident_model()
    query = CounterfactualQuery(
        evidence={"triage_delay": 5},
        intervention=Intervention({"analyst_load": 0}, rationale="add surge analyst capacity"),
        targets=("breach_impact",),
    )

    trace = scm.counterfactual(query)

    assert trace.abduced_noise["observed_triage_delay"] == 5
    assert trace.factual_state["triage_delay"] == 5
    assert trace.counterfactual_state["triage_delay"] == 3
    assert trace.targets == {"breach_impact": 30}
    assert trace.notes == (
        "abduct: reconcile evidence with latent context",
        "act: replace intervened structural equations with constants",
        "predict: re-evaluate descendants under the abduced context",
    )


def test_do_calculus_plan_is_auditable_and_conservative() -> None:
    scm = build_incident_model()

    steps = scm.do_calculus_plan(
        outcome="breach_impact",
        intervention=Intervention({"analyst_load": 0}),
        observed=("phishing_pressure",),
    )

    assert [step.rule for step in steps] == [
        DoCalculusRule.INSERT_DELETE_OBSERVATIONS,
        DoCalculusRule.INSERT_DELETE_ACTIONS,
    ]
    assert "d-separation proof" in steps[0].justification
    assert steps[-1].after.startswith("A2P[")


def test_cycle_detection_rejects_invalid_graph() -> None:
    try:
        StructuralCausalModel(
            variables=(
                CausalVariable("a", parents=("b",)),
                CausalVariable("b", parents=("a",)),
            ),
            equations={"a": lambda _state, _noise: 1, "b": lambda _state, _noise: 1},
        )
    except ValueError as error:
        assert "cycle detected" in str(error)
    else:
        raise AssertionError("cycle should have been rejected")
