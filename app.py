"""Streamlit application for data preparation and TNDP route synthesis."""

from pathlib import Path
import json

import geopandas as gpd
import folium
import streamlit as st
from streamlit.components.v1 import html

from config import CACHE_DIR, REPORT_DIR

st.set_page_config(page_title="Tranmodel — TNDP", layout="wide")


@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def report(name: str):
    return load_json(str(CACHE_DIR / name)) or load_json(str(REPORT_DIR / name))


def show_map(m: folium.Map, height: int = 600):
    html(m._repr_html_(), height=height, width=None)


st.sidebar.title("Tranmodel")
st.sidebar.caption("Воронеж · OSM + WorldPop + AequilibraE + TNDP")
section = st.sidebar.radio("Раздел", [
    "Обзор", "Спрос", "Корреспонденции", "TNDP — Синтез маршрутов", "Benchmark",
])

if section == "Обзор":
    st.title("Автоматический синтез маршрутной сети Воронежа")
    st.markdown(
        "Главная задача модели — получить матрицу корреспонденций и на её основе "
        "автоматически сформировать маршрутную сеть. AequilibraE используется "
        "для сетевого расчёта и оценки общественного транспорта."
    )
    p1 = report("phase1_real/phase1_report.json")
    p2 = report("phase2/phase2_report.json")
    tn = report("tndp/tndp_report.json")
    if tn:
        a, b, c, d = st.columns(4)
        a.metric("Маршрутов", f"{tn.get('n_routes', 0):,}")
        b.metric("Кандидатов", f"{tn.get('n_candidates', 0):,}")
        c.metric("Коридоров OD", f"{tn.get('n_corridors', 0):,}")
        d.metric("Обслужено спроса", f"{tn.get('direct_demand_share', 0) * 100:.1f}%")
    elif p1 or p2:
        a, b, c = st.columns(3)
        a.metric("Остановок", f"{(p1 or {}).get('n_stops', 0):,}")
        b.metric("Поездок", f"{(p2 or {}).get('total_trips', 0):,.0f}")
        c.metric("OD-пар", f"{(p2 or {}).get('n_od_pairs', 0):,}")
    else:
        st.info("Сначала подготовьте спрос и матрицу корреспонденций.")

elif section == "Спрос":
    st.title("Спрос")
    rep = report("phase1_real/phase1_report.json")
    if rep:
        a, b, c = st.columns(3)
        a.metric("Остановок", f"{rep['n_stops']:,}")
        b.metric("Население", f"{rep['population_sum_by_stop']:,.0f}")
        c.metric("Рабочих мест", f"{rep['jobs_sum_by_stop']:,.0f}")
        st.markdown((REPORT_DIR / "phase1_real_report.md").read_text(encoding="utf-8") if (REPORT_DIR / "phase1_real_report.md").exists() else "")
    else:
        st.info("Данные спроса ещё не рассчитаны.")

elif section == "Корреспонденции":
    st.title("Матрица корреспонденций")
    rep = report("phase2/phase2_report.json")
    if rep:
        a, b, c, d = st.columns(4)
        a.metric("Поездок", f"{rep.get('total_trips', 0):,.0f}")
        b.metric("OD-пар", f"{rep.get('n_od_pairs', 0):,}")
        c.metric("Средняя дальность", f"{rep.get('avg_dist_km', 0):.2f} км")
        d.metric("Затухание", f"{rep.get('decay_radius_km', 0):.1f} км")
        st.markdown((REPORT_DIR / "phase2_report.md").read_text(encoding="utf-8") if (REPORT_DIR / "phase2_report.md").exists() else "")
    else:
        st.info("Матрица корреспонденций ещё не рассчитана.")

elif section == "TNDP — Синтез маршрутов":
    st.title("TNDP — синтез маршрутной сети")
    st.write(
        "Генератор использует OD-коридоры, терминальные ограничения и дорожную сеть. "
        "Сначала кандидаты проходят быстрый отбор, затем лучшие маршрутные сети "
        "оцениваются через AequilibraE Transit Assignment / Optimal Strategies."
    )

    from src.tndp.model import NetworkDesignConfig
    from src.tndp.run import run_tndp

    a, b, c, d, e = st.columns(5)
    min_routes = a.number_input("Минимум маршрутов", 1, 200, 10)
    max_routes = b.number_input("Максимум маршрутов", 1, 300, 30)
    corridors = c.number_input("OD-коридоров", 10, 2000, 300)
    candidates = d.number_input("Кандидатов/коридор", 1, 30, 8)
    full = e.checkbox("Полная оценка AequilibraE", value=True)

    if st.button("Синтезировать сеть", type="primary"):
        if max_routes < min_routes:
            st.error("Максимум маршрутов должен быть не меньше минимума.")
        else:
            cfg = NetworkDesignConfig(
                min_routes=int(min_routes), max_routes=int(max_routes),
                corridor_top_pairs=int(corridors), candidate_limit_per_corridor=int(candidates),
                full_evaluation=bool(full),
            )
            with st.spinner("Генерируем и оцениваем маршрутные сети..."):
                try:
                    result = run_tndp(cfg, full_assignment=bool(full))
                    st.success("Синтез завершён.")
                    st.json(result)
                    st.cache_data.clear()
                except Exception as exc:
                    st.exception(exc)

    rep = report("tndp/tndp_report.json")
    if rep:
        a, b, c, d, e = st.columns(5)
        a.metric("Маршрутов", f"{rep.get('n_routes', 0):,}")
        b.metric("Кандидатов", f"{rep.get('n_candidates', 0):,}")
        c.metric("Коридоров", f"{rep.get('n_corridors', 0):,}")
        d.metric("Обслужено", f"{rep.get('direct_demand_share', 0) * 100:.1f}%")
        e.metric("Пересадки", f"{rep.get('transfers', 0):.2f}")
        st.caption(f"Оценщик: {rep.get('evaluator', '—')}")
        if (REPORT_DIR / "tndp_report.md").exists():
            st.markdown((REPORT_DIR / "tndp_report.md").read_text(encoding="utf-8"))
        route_path = Path(rep["route_set"])
        if route_path.exists():
            st.download_button("Скачать generated_routes.json", route_path.read_bytes(), "generated_routes.json")
    else:
        st.info("Синтез ещё не выполнялся.")

elif section == "Benchmark":
    st.title("Benchmark TNDP")
    st.write(
        "Раздел подготовлен для проверки решателя на Mandl/Mumford/Rivera из "
        "TransitNetworkDesign. Сравнение выполняется по целевой функции и структуре маршрутов."
    )
    st.code("python -m src.tndp.cli --benchmark mandl", language="powershell")
