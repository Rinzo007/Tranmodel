"""Streamlit UI for the Voronezh transport model.

The AequilibraE backend is exposed as the new computational phase while the
legacy phases remain available for comparison during migration.
"""

import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
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
st.sidebar.caption("OSM + WorldPop + AequilibraE")
phase = st.sidebar.radio(
    "Раздел",
    [
        "Обзор",
        "AequilibraE",
        "Фаза 0 — Данные",
        "Фаза 1 — Спрос",
        "Фаза 2 — Корреспонденции",
        "Фаза 3 — Маршруты",
        "Фаза 4 — Пассажиропоток",
    ],
)

if phase == "Обзор":
    st.title("Транспортная модель городского округа Воронеж")
    st.markdown(
        "Модель использует OSM и WorldPop для подготовки спроса. "
        "Основной новый расчётный контур построен на AequilibraE: "
        "сетевая стоимость, гравитационное распределение и назначение."
    )
    rep = load_named_json("aequilibrae/aequilibrae_report.json")
    p0 = load_named_json("phase0_report.json")
    p1 = load_named_json("phase1_real/phase1_report.json")
    if rep:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Центроидов", f"{rep['n_centroids']:,}")
        c2.metric("Узлов", f"{rep['n_nodes']:,}")
        c3.metric("Рёбер", f"{rep['n_links']:,}")
        c4.metric("Матрица OD", f"{rep['total_demand']:,.0f}")
    elif p0 or p1:
        c1, c2 = st.columns(2)
        c1.metric("Население", f"{(p0 or {}).get('population', {}).get('total', 0):,.0f}")
        c2.metric("Остановок", f"{(p1 or {}).get('n_stops', 0):,}")
    st.info("Для нового расчёта откройте раздел «AequilibraE».")

elif phase == "AequilibraE":
    st.title("AequilibraE — основной расчётный контур")
    st.write(
        "Создаётся проект AequilibraE из кэшированной дорожной сети и остановок, "
        "затем считаются сетевые скримы, гравитационная матрица EXPO/IPF и "
        "назначение BPR/BFW."
    )

    from src.aequilibrae_pipeline import AequilibraEPipelineError, run_all

    c1, c2 = st.columns(2)
    run_button = c1.button("Запустить расчёт AequilibraE", type="primary")
    force = c2.checkbox("Пересобрать проект с нуля", value=False)

    if run_button:
        with st.spinner("AequilibraE выполняет сетевые скримы, распределение и назначение..."):
            try:
                report = run_all(force=force)
                st.success("Расчёт AequilibraE завершён.")
                st.json(report)
                st.cache_data.clear()
            except (AequilibraEPipelineError, FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

    report = load_named_json("aequilibrae/aequilibrae_report.json")
    if report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Центроидов", f"{report['n_centroids']:,}")
        c2.metric("Узлов", f"{report['n_nodes']:,}")
        c3.metric("Рёбер", f"{report['n_links']:,}")
        c4.metric("Объём OD", f"{report['total_demand']:,.0f}")
        st.markdown(load_report_md("aequilibrae_report.md"))
        st.caption(
            f"Проект AequilibraE: `{report['project']}`. "
            "Его можно открыть средствами AequilibraE/QGIS."
        )
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
    st.title("Фаза 1 — Спрос на перевозки")
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
        for _, s in stops.iterrows():
            p = s.geometry.centroid
            folium.CircleMarker(
                location=[p.y, p.x],
                radius=4,
                fill=True,
                fillOpacity=0.5,
                popup=f"pop={s['population']:.0f} jobs={s['jobs']:.0f}",
            ).add_to(m)
        show_map(m)

elif phase == "Фаза 2 — Корреспонденции":
    st.title("Фаза 2 — Корреспонденции (старое ядро)")
    st.caption("Для нового расчёта используйте раздел «AequilibraE». Этот раздел оставлен для сравнения результатов миграции.")
    rep = load_named_json("phase2/phase2_report.json") or load_named_json("phase2_report.json")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Поездок", f"{rep.get('total_trips', 0):,.0f}")
        c2.metric("Радиус затухания", f"{rep.get('decay_radius_km', 0)} км")
        c3.metric("Пар OD", f"{rep.get('n_od_pairs', 0):,}")
    st.markdown(load_report_md("phase2_report.md"))

elif phase == "Фаза 3 — Маршруты":
    st.title("Фаза 3 — Существующая маршрутная сеть")
    st.caption("Маршрутная сеть сохраняется как исходная/эталонная. Её дальнейшая миграция в транспортный модуль AequilibraE требует GTFS с расписанием или частотами.")
    rep = load_named_json("phase3_real/phase3_report.json")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Маршрутов", f"{rep.get('n_routes', 0)}")
        c2.metric("Остановок охвачено", f"{rep.get('n_stops_served', 0)}")
        c3.metric("Суммарная длина", f"{rep.get('total_route_km_air', 0)} км")
    st.markdown(load_report_md("phase3_real_report.md"))

elif phase == "Фаза 4 — Пассажиропоток":
    st.title("Фаза 4 — Пассажиропоток (старое ядро)")
    st.caption("Оставлено для сравнения со старым алгоритмом. Новое сетевое назначение выполняется AequilibraE.")
    rep = load_named_json("phase4/phase4_report.json")
    if rep:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Распределено поездок", f"{rep['assigned_trips']:,.0f}")
        c2.metric("Макс. загрузка", f"{rep['max_segment_load']:,.0f}")
        c3.metric("Коэфф. заполнения", f"{rep['avg_load_factor']:.2f}")
        c4.metric("Средние пересадки", f"{rep['avg_transfers']:.2f}")
    st.markdown(load_report_md("phase4_report.md"))
