"""Consistency checks for the multi-period TNDP operating pipeline."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from math import isclose

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    value: float | str | None = None

@dataclass(frozen=True, slots=True)
class ValidationReport:
    passed: bool
    issues: tuple[ValidationIssue, ...]

    def as_dict(self) -> dict:
        return {"passed": self.passed, "issues": [asdict(x) for x in self.issues]}


def validate_interval_profile(periods, expected_total: float = 14.5, tolerance: float = 1e-9) -> ValidationReport:
    issues = []
    total = sum(float(p.hours) * float(p.frequency_factor) for p in periods)
    if not isclose(total, expected_total, rel_tol=0.0, abs_tol=tolerance):
        issues.append(ValidationIssue("error", "profile_total", f"Сумма профильных часов должна быть {expected_total}", total))
    if not periods:
        issues.append(ValidationIssue("error", "profile_empty", "Профиль периодов пуст"))
    return ValidationReport(not any(i.level == "error" for i in issues), tuple(issues))


def validate_route_plan(plan: dict, periods_count: int = 6) -> ValidationReport:
    issues = []
    periods = plan.get("periods") or []
    if len(periods) != periods_count:
        issues.append(ValidationIssue("error", "period_count", "Число периодов не совпадает с профилем", len(periods)))
    fleets = [int(p.get("fleet", 0)) for p in periods]
    peak = int(plan.get("peak_fleet", 0))
    if fleets and peak != max(fleets):
        issues.append(ValidationIssue("error", "peak_fleet", "Пиковый парк должен быть максимумом по периодам", peak))
    if float(plan.get("annual_mileage_km", 0.0)) < 0 or float(plan.get("annual_hours", 0.0)) < 0:
        issues.append(ValidationIssue("error", "annual_negative", "Годовые показатели не могут быть отрицательными"))
    return ValidationReport(not any(i.level == "error" for i in issues), tuple(issues))


def validate_network_plan(plan: dict) -> ValidationReport:
    issues = []
    routes = plan.get("routes") or []
    network = plan.get("network") or plan
    expected_fleet = sum(int(r.get("peak_fleet", 0)) for r in routes)
    actual_fleet = int(network.get("fleet", 0))
    if expected_fleet != actual_fleet:
        issues.append(ValidationIssue("error", "network_fleet", "Парк сети не соответствует сумме маршрутных пиков", actual_fleet))
    expected_total = sum(float((r.get("costs") or {}).get("total_annual_mln", 0.0)) for r in routes)
    actual_total = float((network.get("costs") or {}).get("total_annual_mln", network.get("total_annual_mln", 0.0)))
    if not isclose(expected_total, actual_total, rel_tol=1e-8, abs_tol=1e-8):
        issues.append(ValidationIssue("error", "network_cost", "Стоимость сети не соответствует сумме маршрутных стоимостей", actual_total))
    return ValidationReport(not any(i.level == "error" for i in issues), tuple(issues))


def validate_evaluation_metadata(metadata: dict) -> ValidationReport:
    issues = []
    unified = metadata.get("unified_operating_plan") or {}
    costs = unified.get("costs") or {}
    components = ["fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln"]
    total = sum(float(costs.get(k, 0.0) or 0.0) for k in components)
    reported = float(costs.get("total_annual_mln", total) or 0.0)
    if not isclose(total, reported, rel_tol=1e-8, abs_tol=1e-8):
        issues.append(ValidationIssue("error", "cost_total", "Сумма статей не совпадает с общей стоимостью", reported))
    if unified and int(metadata.get("fleet", unified.get("fleet", 0)) or 0) != int(unified.get("fleet", 0) or 0):
        issues.append(ValidationIssue("error", "evaluation_fleet", "Парк в Evaluation не совпадает с unified operating plan"))
    return ValidationReport(not any(i.level == "error" for i in issues), tuple(issues))


def validate_pipeline(periods, *, route_plans=(), network_plan=None, evaluation_metadata=None) -> dict:
    checks = {
        "interval_profile": validate_interval_profile(periods),
        "routes": tuple(validate_route_plan(x) for x in route_plans),
        "network": validate_network_plan(network_plan) if network_plan is not None else ValidationReport(True, ()),
        "evaluation": validate_evaluation_metadata(evaluation_metadata or {}),
    }
    issues = []
    for report in checks.values():
        if isinstance(report, tuple):
            for r in report: issues.extend(r.issues)
        else: issues.extend(report.issues)
    return {"passed": not any(i.level == "error" for i in issues), "issues": [asdict(i) for i in issues], "checks": {k: ([r.as_dict() for r in v] if isinstance(v, tuple) else v.as_dict()) for k, v in checks.items()}}
