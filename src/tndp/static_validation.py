"""Static and structural validation helpers for the TNDP pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: str
    code: str
    message: str


def validate_module_graph(*, required_modules: Iterable[str], available_modules: Iterable[str]) -> list[ValidationIssue]:
    available = set(available_modules)
    return [ValidationIssue("error", "missing_module", f"Missing required module: {name}")
            for name in required_modules if name not in available]


def validate_period_results(period_results: Iterable[dict], expected_period_count: int = 6) -> list[ValidationIssue]:
    rows = list(period_results)
    issues: list[ValidationIssue] = []
    if len(rows) != expected_period_count:
        issues.append(ValidationIssue("error", "period_count", f"Expected {expected_period_count} period results, got {len(rows)}"))
    seen = set()
    for row in rows:
        key = str(row.get("period_id", ""))
        if not key:
            issues.append(ValidationIssue("error", "period_id_missing", "Period result has no period_id"))
        elif key in seen:
            issues.append(ValidationIssue("error", "period_duplicate", f"Duplicate period_id: {key}"))
        seen.add(key)
    return issues


def validate_operating_plan(plan: dict, *, tolerance: float = 1e-6) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    fleet = float(plan.get("fleet", 0) or 0)
    annual_km = float(plan.get("annual_mileage_km", 0) or 0)
    annual_hours = float(plan.get("annual_hours", 0) or 0)
    if fleet < -tolerance: issues.append(ValidationIssue("error", "negative_fleet", "Fleet cannot be negative"))
    if annual_km < -tolerance: issues.append(ValidationIssue("error", "negative_mileage", "Annual mileage cannot be negative"))
    if annual_hours < -tolerance: issues.append(ValidationIssue("error", "negative_hours", "Annual hours cannot be negative"))
    periods = plan.get("periods", []) or []
    if periods:
        period_fleets = [float(p.get("fleet", 0) or 0) for p in periods]
        expected_peak = max(period_fleets, default=0.0)
        if fleet + tolerance < expected_peak:
            issues.append(ValidationIssue("error", "peak_fleet_mismatch", f"Network fleet {fleet} is below period peak {expected_peak}"))
    return issues


def validate_costs(costs: dict, *, tolerance: float = 1e-6) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    keys = ("fuel_energy_mln", "repair_mln", "crew_mln", "infrastructure_mln", "dispatch_mln", "contract_mln", "amortization_mln")
    component_sum = sum(float(costs.get(k, 0) or 0) for k in keys)
    total = float(costs.get("total_annual_mln", component_sum) or 0)
    if any(float(costs.get(k, 0) or 0) < -tolerance for k in keys):
        issues.append(ValidationIssue("error", "negative_cost", "Annual cost component cannot be negative"))
    if abs(total - component_sum) > tolerance:
        issues.append(ValidationIssue("error", "cost_sum_mismatch", f"Total cost {total} differs from components {component_sum}"))
    return issues


def validate_pipeline_report(report: dict) -> list[ValidationIssue]:
    """Validate the serialized report without running the expensive assignment."""
    issues = validate_period_results(report.get("period_results", []))
    plan = report.get("unified_operating_plan", {}) or {}
    if plan:
        issues.extend(validate_operating_plan(plan))
        issues.extend(validate_costs(plan.get("costs", {}) or {}))
    if report.get("n_routes", 0) < 0:
        issues.append(ValidationIssue("error", "negative_routes", "Route count cannot be negative"))
    return issues


def summarize_issues(issues: Iterable[ValidationIssue]) -> dict:
    rows = list(issues)
    return {"ok": not any(x.level == "error" for x in rows), "errors": sum(x.level == "error" for x in rows),
            "warnings": sum(x.level == "warning" for x in rows), "issues": [x.__dict__ for x in rows]}
