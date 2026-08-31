"""Streamlit UI for the Voronezh transport model."""

from pathlib import Path
import json

import folium
import geopandas as gpd
import streamlit as st
from streamlit.components.v1 import html

from config import CACHE_DIR, REPORT_DIR

st.set_page_config(page_title="Воронеж — транспортная модель", layout="wide")


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_named_json(name: str) -> dict | None:
    for base in (CACHE_DIR, REPORT_DIR):
        p = base / name
        if p.exists():
            return load_json(str(p))
    return None


def load_report_md(name: str) -> str:
    p = REPORT_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else "Отчёт не найден."


def show_map(m: folium.Map, height: int = 560) -> None:
    html(m._repr_html_(), height=height, width=None)


st.sidebar.title("Транспортная модель Воронежа")
st.sidebar.caption("OSM + WorldPop + AequilibraE + TNDP")
phase = st.sidebar.radio("Раздел", [
    "Обзор", "TNDP — Синтез маршрутов", "AequilibraE",
    "Фаза 0 — Данные", "Фаза 1 — Спрос", "Фаза 2 — Корреспонденции",
    "Фаза 3 — Маршруты", "Фаза 4 — Пассажиропоток",
])

if phase == "Обзор":
    st.title("Транспортная модель городского округа Воронеж")
    st.markdown(
        "OSM + WorldPop формируют спрос, AequilibraE выполняет сетевые расчёты и назначение, "
        "TNDP синтезирует маршрутную сеть из матрицы корреспонденций."
    )
    rep = load_named_json("tndp/tndp_report.json") or load_named_json("aequilibrae/aequilibrae_report.json")
    if rep:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Маршрутов", f"{rep.get('n_routes', 0):,}")
        c2.metric("Кандидатов", f"{rep.get('n_candidates', rep.get('n_nodes', 0)):,}")
        c3.metric("OD-коридоров", f"{rep.get('n_corridors', 0):,}")
        c4.metric("Обслужено", f"{rep.get('direct_demand_share', 0) * 100:.1f}%")
    st.info("Откройте «TNDP — Синтез маршрутов», чтобы построить новую сеть из матрицы корреспонденций.")

elif phase == "TNDP — Синтез маршрутов":
    st.title("TNDP — автоматический синтез маршрутной сети")
    st.write(
        "Алгоритм выделяет сильные OD-коридоры, строит кандидатные маршруты по дорожной сети "
        "и итеративно выбирает целые маршруты. Каждое принятое изменение может проверяться "
        "полным назначением AequilibraE Optimal Strategies."
    )
    from src.tndp.model import NetworkDesignConfig
    from src.tndp.run import run_tndp

    c1, c2, c3, c4, c5 = st.columns(5)
    min_routes = c1.number_input("Мин. маршрутов", 1, 200, 10)
    max_routes = c2.number_input("Макс. маршрутов", 1, 300, 30)
    corridors = c3.number_input("OD-коридоров", 10, 2000, 300)
    candidates = c4.number_input("Кандидатов/коридор", 1, 30, 8)
    full_assignment = c5.checkbox("AequilibraE", value=True)
    run_button = st.button("Синтезировать маршрутную сеть", type="primary")

    if run_button:
        if max_routes < min_routes:
            st.error("Максимальное число маршрутов должно быть не меньше минимального.")
        else:
            cfg = NetworkDesignConfig(
                min_routes=int(min_routes),
                max_routes=int(max_routes),
                corridor_top_pairs=int(corridors),
                candidate_limit_per_corridor=int(candidates),
            )
            with st.spinner("Генерируем кандидатов и оцениваем маршрутные сети..."):
                try:
                    report = run_tndp(cfg, full_assignment=full_assignment)
                    st.success("Синтез завершён.")
                    st.json(report)
                    st.cache_data.clear()
                except Exception as exc:
                    st.exception(exc)

    report = load_named_json("tndp/tndp_report.json")
    if report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Итоговых маршрутов", f"{report.get('n_routes', 0):,}")
        c2.metric("Обслужено спроса", f"{report.get('direct_demand_share', 0) * 100:.1f}%")
        c3.metric("Средние пересадки", f"{report.get('transfers', 0):.2f}")
        c4.metric("Длина маршрутов", f"{report.get('operator_route_km', 0):.1f} км")
        st.caption(f"Оценщик: {report.get('evaluator', 'не указан')}")
        st.markdown(load_report_md("tndp_report.md"))
        route_path = Path(report["route_set"])
        if route_path.exists():
            st.download_button(
                "Скачать набор маршрутов JSON",
                route_path.read_bytes(),
                "generated_routes.json",
                "application/json",
            )
    else:
        st.info("Синтез ещё не выполнялся.")

elif phase == "AequilibraE":
    st.title("AequilibraE — расчётное ядро")
    st.write("Сетевые скримы, распределение спроса и назначение общественного транспорта.")
    from src.aequilibrae_full import run_full_model
    force = st.checkbox("Пересобрать проект с нуля", value=False)
    if st.button("Запустить полный расчёт AequilibraE", type="primary"):
        with st.spinner("AequilibraE выполняет расчёт..."):
            try:
                report = run_full_model(force=force)
                st.success("Расчёт завершён.")
                st.json(report)
                st.cache_data.clear()
            except Exception as exc:
                st.exception(exc)
    report = load_named_json("aequilibrae/aequilibrae_report.json")
    if report:
        st.markdown(load_report_md("aequilibrae_report.md"))
    else:
        st.info("Расчёт ещё не выполнялся.")

elif phase == "Фаза 0 — Данные":
    st.title("Фаза 0 — Исходные данные")
    rep = load_named_json("phase0_report.json")
    if rep:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Площадь", f"{rep['area_km2']} км²")
        c2.metric("Население", f"{rep['population']['total']:,.0f}")
        c3.metric("Дороги", f"{rep['roads']['total_km']} км")
        c4.metric("Ж/д пути", f"{rep['rail']['km']} км")
        c5.metric("Остановки OSM", f"{rep['stops']['count']}")
    st.markdown(load_report_md("phase0_report.md"))

elif phase == "Фаза 1 — Спрос":
    st.title("Фаза 1 — Спрос")
    rep = load_named_json("phase1_real/phase1_report.json")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Остановок", f"{rep['n_stops']}")
        c2.metric("Население приписано", f"{rep['population_sum_by_stop']:,.0f}")
        c3.metric("Рабочих мест", f"{rep['jobs_sum_by_stop']:,.0f}")
    st.markdown(load_report_md("phase1_real_report.md"))
    stops_path = CACHE_DIR / "phase1_real" / "stops_demand.parquet"
    if stops_path.exists():
        stops = gpd.read_parquet(stops_path).to_crs("EPSG:4326")
        m = folium.Map(location=[51.66, 39.20], zoom_start=11, tiles="CartoDB positron")
        for _, row in stops.iterrows():
            p = row.geometry
            folium.CircleMarker(
                location=[p.y, p.x], radius=4, fill=True, fillOpacity=0.5,
                popup=f"pop={row['population']:.0f} jobs={row['jobs']:.0f}",
            ).add_to(m)
        show_map(m)

elif phase == "Фаза 2 — Корреспонденции":
    st.title("Фаза 2 — Корреспонденции")
    rep = load_named_json("phase2/phase2_report.json") or load_named_json("phase2_report.json")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Поездок", f"{rep.get('total_trips', 0):,.0f}")
        c2.metric("Радиус затухания", f"{rep.get('decay_radius_km', 0)} км")
        c3.metric("Пар OD", f"{rep.get('n_od_pairs', 0):,}")
    st.markdown(load_report_md("phase2_report.md"))

elif phase == "Фаза 3 — Маршруты":
    st.title("Фаза 3 — Маршруты")
    st.caption(
        "Здесь показывается исходная сеть из voronezh_routes_terminals.geojson. "
        "Новая сеть генерируется в разделе TNDP."
    )
    rep = load_named_json("phase3_real/phase3_report.json")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Маршрутов", f"{rep.get('n_routes', 0)}")
        c2.metric("Остановок охвачено", f"{rep.get('n_stops_served', 0)}")
        c3.metric("Суммарная длина", f"{rep.get('total_route_km_air', 0)} км")
    st.markdown(load_report_md("phase3_real_report.md"))

elif phase == "Фаза 4 — Пассажиропоток":
    st.title("Фаза 4 — Пассажиропоток")
    tr = load_named_json("aequilibrae/transit/transit_report.json")
    if tr:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Маршрутов", f"{tr['n_routes']:,}")
        c2.metric("Остановок", f"{tr['n_stops']:,}")
        c3.metric("Спрос", f"{tr['total_demand']:,.0f}")
        c4.metric("Загруженных рёбер", f"{tr['assigned_link_rows']:,}")
        st.markdown(load_report_md("transit_report.md"))
    else:
        rep = load_named_json("phase4/phase4_report.json")
        if rep:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Распределено поездок", f"{rep['assigned_trips']:,.0f}")
            c2.metric("Макс. загрузка", f"{rep['max_segment_load']:,.0f}")
            c3.metric("Коэфф. заполнения", f"{rep['avg_load_factor']:.2f}")
            c4.metric("Средние пересадки", f"{rep['avg_transfers']:.2f}")
        st.markdown(load_report_md("phase4_report.md"))
