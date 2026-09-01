"""Streamlit application for data preparation and TNDP route synthesis."""

from __future__ import annotations

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

def prepare_demand(zone_size_m: float = 750.0, force: bool = False):
    from src.phase1_real import run_phase1_real
    from src.zones import build_transport_zones
    from src.zone_od import run_zone_od
    run_phase1_real(force=force)
    build_transport_zones(size_m=zone_size_m, force=force)
    return run_zone_od(zone_size_m=zone_size_m, force=force)

st.sidebar.title("Tranmodel")
st.sidebar.caption("Воронеж · OSM + WorldPop + AequilibraE + TNDP")
section = st.sidebar.radio("Раздел", ["Обзор", "Спрос", "Транспортные зоны", "Корреспонденции", "TNDP — Синтез маршрутов", "Benchmark"])

if section == "Обзор":
    st.title("Автоматический синтез маршрутной сети Воронежа")
    st.markdown("Главная задача модели — получить матрицу корреспонденций между независимыми транспортными зонами и на её основе автоматически сформировать маршрутную сеть. Остановки ОТ и дорожные узлы являются отдельными сущностями сети.")
    zn, zo, tn = report("zones/zones_report.json"), report("zone_od/zone_od_report.json"), report("tndp/tndp_report.json")
    if tn:
        a, b, c, d = st.columns(4); a.metric("Маршрутов", f"{tn.get('n_routes', 0):,}"); b.metric("Зон", f"{tn.get('n_zones', 0):,}"); c.metric("Коридоров OD", f"{tn.get('n_corridors', 0):,}"); d.metric("Обслужено спроса", f"{tn.get('direct_demand_share', 0) * 100:.1f}%")
    elif zo or zn:
        a, b, c = st.columns(3); a.metric("Транспортных зон", f"{(zn or {}).get('n_zones', 0):,}"); b.metric("Поездок", f"{(zo or {}).get('total_trips', 0):,.0f}"); c.metric("Среднее время", f"{(zo or {}).get('avg_network_time_min', 0):.1f} мин")
    else:
        st.info("Данные спроса ещё не подготовлены.")
        zone_size = st.number_input("Размер транспортной зоны, м", 300, 2000, 750, step=50, key="overview_zone_size")
        force = st.checkbox("Пересчитать существующие данные", key="overview_force")
        if st.button("Подготовить данные спроса", type="primary"):
            with st.spinner("Подготавливаем данные спроса, зоны и сетевую OD-матрицу..."):
                try: st.success("Данные спроса готовы."); st.json(prepare_demand(float(zone_size), bool(force))); st.cache_data.clear()
                except Exception as exc: st.exception(exc)

elif section == "Спрос":
    st.title("Спрос")
    rep = report("phase1_real/phase1_report.json")
    if rep:
        a, b, c = st.columns(3); a.metric("Исходных остановок", f"{rep['n_stops']:,}"); b.metric("Население", f"{rep['population_sum_by_stop']:,.0f}"); c.metric("Рабочих мест", f"{rep['jobs_sum_by_stop']:,.0f}")
        st.info("Этот слой используется только для извлечения WorldPop/POI. Единицами OD являются транспортные зоны, а не остановки.")
    else: st.info("Данные ещё не подготовлены. Перейдите в «Обзор» и нажмите «Подготовить данные спроса».")

elif section == "Транспортные зоны":
    st.title("Транспортные зоны"); st.write("Зоны независимы от остановок ОТ. Они используются как единицы происхождения и назначения поездок.")
    from src.zones import build_transport_zones
    from src.zone_od import run_zone_od
    a, b = st.columns(2); size_m = a.number_input("Размер зоны, м", 300, 2000, 750, step=50); force = b.checkbox("Пересоздать зоны", value=False)
    if st.button("Построить зоны и OD", type="primary"):
        with st.spinner("Подготавливаем исходный спрос, зоны и сетевую OD-матрицу..."):
            try: st.success("Зоны и OD-матрица построены."); st.json(prepare_demand(float(size_m), bool(force))); st.cache_data.clear()
            except Exception as exc: st.exception(exc)
    zn, zo = report("zones/zones_report.json"), report("zone_od/zone_od_report.json")
    if zn:
        a, b, c = st.columns(3); a.metric("Зон", f"{zn.get('n_zones', 0):,}"); b.metric("Население", f"{zn.get('population', 0):,.0f}"); c.metric("Притяжение", f"{zn.get('jobs', 0):,.0f}")
    if zo:
        a, b = st.columns(2); a.metric("Поездок", f"{zo.get('total_trips', 0):,.0f}"); b.metric("Среднее время", f"{zo.get('avg_network_time_min', 0):.1f} мин")

elif section == "Корреспонденции":
    st.title("Матрица корреспонденций"); zo = report("zone_od/zone_od_report.json")
    if zo:
        a, b, c = st.columns(3); a.metric("Поездок", f"{zo.get('total_trips', 0):,.0f}"); b.metric("OD-пар", f"{zo.get('od_pairs', 0):,}"); c.metric("Среднее время", f"{zo.get('avg_network_time_min', 0):.1f} мин")
    else: st.info("Постройте транспортные зоны и OD-матрицу в разделе «Транспортные зоны» или через «Обзор».")

elif section == "TNDP — Синтез маршрутов":
    st.title("TNDP — синтез маршрутной сети")
    st.write("OD-коридоры строятся между транспортными зонами. Кандидатные маршруты проходят по реальному дорожному графу через независимые остановки ОТ. Лучшие сети оцениваются через AequilibraE Transit Assignment / Optimal Strategies.")
    from src.tndp.model import NetworkDesignConfig
    from src.tndp.run import run_tndp
    zo = report("zone_od/zone_od_report.json")
    if not zo:
        st.warning("OD-матрица ещё не подготовлена.")
        if st.button("Подготовить данные спроса и продолжить", type="primary"):
            with st.spinner("Подготавливаем данные спроса, зоны и OD-матрицу..."):
                try: prepare_demand(); st.success("OD-матрица готова. Теперь можно запускать синтез."); st.rerun()
                except Exception as exc: st.exception(exc)
    else:
        a, b, c, d, e = st.columns(5)
        min_routes = a.number_input("Минимум маршрутов", 1, 200, 10); max_routes = b.number_input("Максимум маршрутов", 1, 300, 30); corridors = c.number_input("OD-коридоров", 10, 2000, 300); candidates = d.number_input("Кандидатов/коридор", 1, 30, 8); full = e.checkbox("Полная оценка AequilibraE", value=True)
        if st.button("Синтезировать сеть", type="primary"):
            if max_routes < min_routes: st.error("Максимум маршрутов должен быть не меньше минимума.")
            else:
                cfg = NetworkDesignConfig(min_routes=int(min_routes), max_routes=int(max_routes), corridor_top_pairs=int(corridors), candidate_limit_per_corridor=int(candidates), full_evaluation=bool(full))
                status = st.status("Запуск синтеза маршрутной сети...", expanded=True)
                log_box = st.empty(); messages = []
                def _progress(message: str):
                    messages.append(message); log_box.code("\n".join(messages[-20:]), language="text"); status.update(label=message, state="running")
                try:
                    result = run_tndp(cfg, full_assignment=bool(full), progress=_progress)
                    status.update(label="Синтез завершён", state="complete"); st.success("Синтез завершён."); st.json(result); st.cache_data.clear()
                except Exception as exc:
                    status.update(label="Ошибка синтеза", state="error"); st.exception(exc)
        rep = report("tndp/tndp_report.json")
        if rep:
            a, b, c, d, e = st.columns(5); a.metric("Маршрутов", f"{rep.get('n_routes', 0):,}"); b.metric("Зон", f"{rep.get('n_zones', 0):,}"); c.metric("Коридоров", f"{rep.get('n_corridors', 0):,}"); d.metric("Обслужено", f"{rep.get('direct_demand_share', 0) * 100:.1f}%"); e.metric("Пересадки", f"{rep.get('transfers', 0):.2f}"); st.caption(f"Оценщик: {rep.get('evaluator', '—')}")
            md = REPORT_DIR / "tndp_report.md"
            if md.exists(): st.markdown(md.read_text(encoding="utf-8"))
            route_path = Path(rep.get("route_set", "")); geo_path = Path(rep.get("route_geojson", ""))
            if route_path.exists(): st.download_button("Скачать generated_routes.json", route_path.read_bytes(), "generated_routes.json")
            if geo_path.exists(): st.download_button("Скачать generated_routes.geojson", geo_path.read_bytes(), "generated_routes.geojson")

elif section == "Benchmark":
    st.title("Benchmark TNDP"); st.write("Раздел подготовлен для проверки решателя на Mandl/Mumford/Rivera из TransitNetworkDesign."); st.code("python -m src.tndp.cli --benchmark mandl", language="powershell")
