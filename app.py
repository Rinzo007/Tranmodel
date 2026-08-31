"""Streamlit UI for the Voronezh transport model (phases 0-4).

Run:  streamlit run app.py
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


def load_json(name: str) -> dict | None:
    # try cache dir first (phases write json next to their parquet), then report dir
    for base in (CACHE_DIR, REPORT_DIR):
        p = base / name
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    return None


def load_report_md(name: str) -> str:
    p = REPORT_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else "Отчёт не найден."


def folium_to_html(m: folium.Map) -> str:
    from folium import IFrame
    return m._repr_html_()


def show_map(m: folium.Map, height: int = 560) -> None:
    html(m._repr_html_(), height=height, width=None)


# ---------------------------------------------------------------------------
st.sidebar.title("Транспортная модель Воронежа")
st.sidebar.caption("М.Р. Якимов, 2022 — формирование маршрутной сети")
phase = st.sidebar.radio(
    "Фаза",
    ["Обзор", "Фаза 0 — Данные", "Фаза 1 — Спрос",
     "Фаза 2 — Корреспонденции", "Фаза 3 — Маршруты",
     "Фаза 4 — Пассажиропоток"],
)

# ---------------------------------------------------------------------------
if phase == "Обзор":
    st.title("Транспортная модель городского округа Воронеж")
    st.markdown(
        "Многофазная модель на основе подхода М.Р. Якимова (2022): "
        "данные OSM, население WorldPop, гравитационная матрица "
        "корреспонденций, генерация маршрутов и расчёт пассажиропотока."
    )
    c1, c2, c3, c4 = st.columns(4)
    p0 = load_json("phase0_report.json")
    p1 = load_json("phase2_report.json") or {}
    p2 = load_json("phase3_real/phase3_report.json")
    p3 = load_json("phase3_real/phase3_report.json")
    p4 = load_json("phase4/phase4_report.json")

    c1.metric("Население (WorldPop)",
              f"{p0['population']['total']:,.0f}" if p0 else "-")
    c2.metric("Рабочих мест (N/2)", f"{p1.get('total_jobs', 0):,.0f}")
    c3.metric("Маршрутов", f"{p3['n_routes']}" if p3 else "-")
    c4.metric("Поездок на сеть", f"{p4['assigned_trips']:,.0f}" if p4 else "-")

    st.subheader("Карта маршрутной сети и пассажиропотока")
    if Path(REPORT_DIR / "phase4_map.html").exists():
        show_map(folium.Map(location=[51.66, 39.2], zoom_start=11))
        st.caption("Откройте вкладку «Фаза 4» для карты загрузки.")
    else:
        st.info("Фаза 4 ещё не выполнена.")

# ---------------------------------------------------------------------------
elif phase == "Фаза 0 — Данные":
    st.title("Фаза 0 — Исходные данные")
    rep = load_json("phase0_report.json")
    if rep:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Площадь", f"{rep['area_km2']} км²")
        c2.metric("Население", f"{rep['population']['total']:,.0f}")
        c3.metric("Дороги", f"{rep['roads']['total_km']} км")
        c4.metric("Ж/д пути", f"{rep['rail']['km']} км")
        c5.metric("Остановки (OSM)", f"{rep['stops']['count']}")
    st.markdown(load_report_md("phase0_report.md"))
    show_map(folium.Map(location=[51.66, 39.2], zoom_start=11))
    st.caption("Откройте data/report/phase0_map.html в браузере для полной карты.")

# ---------------------------------------------------------------------------
elif phase == "Фаза 1 — Спрос":
    st.title("Фаза 1 — Спрос на перевозки (реальные остановки)")
    rep = load_json("phase1_real/phase1_report.json")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Остановок", f"{rep['n_stops']}")
        c2.metric("Население приписано", f"{rep['population_sum_by_stop']:,.0f} "
                                          f"({rep['coverage_pop_share']*100:.0f}%)")
        c3.metric("Раб. мест приписано", f"{rep['jobs_sum_by_stop']:,.0f}")
    st.markdown(load_report_md("phase1_real_report.md"))
    stops = gpd.read_parquet(CACHE_DIR / "phase1_real" / "stops_demand.parquet")
    m = folium.Map(location=[51.66, 39.2], zoom_start=11, tiles="CartoDB positron")
    for _, s in stops.iterrows():
        p = s.geometry.centroid
        folium.CircleMarker(
            location=[p.y, p.x], radius=4, fill=True, fillOpacity=0.5,
            popup=f"pop={s['population']:.0f} jobs={s['jobs']:.0f} "
                  f"routes={s['n_routes']} term={s['is_terminal']}",
        ).add_to(m)
    show_map(m)

# ---------------------------------------------------------------------------
elif phase == "Фаза 2 — Корреспонденции":
    st.title("Фаза 2 — Гравитационная матрица корреспонденций")
    rep = load_json("phase2_report.json")
    if rep:
        c1, c2, c3 = st.columns(3)
        c1.metric("Поездок (сумма матрицы)", f"{rep['total_trips']:,.0f}")
        c2.metric("Радиус затухания", f"{rep['decay_radius_km']} км")
        c3.metric("Пар OD", f"{rep['n_od_pairs']:,}")
    st.markdown(load_report_md("phase2_report.md"))
    show_map(folium.Map(location=[51.66, 39.2], zoom_start=11))
    st.caption("data/report/phase2_map.html — линии корреспонденций.")

# ---------------------------------------------------------------------------
elif phase == "Фаза 3 — Маршруты":
    st.title("Фаза 3 — Маршрутная сеть")
    rep = load_json("phase3_real/phase3_report.json")
    if rep:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Маршрутов", f"{rep['n_routes']}")
        c2.metric("Остановок охвачено", f"{rep['n_stops_served']}")
        c3.metric("Ср. длина", f"{rep['avg_route_km_air']} км")
        c4.metric("Суммарная длина", f"{rep['total_route_km_air']} км")
        f = rep.get("formula", {})
        if f:
            st.markdown(
                f"**Формула числа маршрутов:** m = {f.get('k1')}·N/​{f.get('interchange')} "
                f"+ {f.get('k2')}·S/{f.get('interchange')} + {f.get('k3')}·O/{f.get('interchange')} "
                f"= **{rep['route_count_formula_m']:.2f} → {rep['n_routes_formula']} маршрутов**  \n"
                f"N (население) = {f.get('N_thousands'):,.0f} тыс., "
                f"S (площадь) = {f.get('S_km2'):.0f} км², "
                f"O (остановки из файла) = **{f.get('O_stops_file')}**"
            )
    st.markdown(load_report_md("phase3_real_report.md"))

    routes = gpd.read_parquet(CACHE_DIR / "phase3_real" / "routes.parquet")
    flat = pd.read_parquet(CACHE_DIR / "phase3_real" / "routes_flat.parquet")
    stops = gpd.read_parquet(CACHE_DIR / "phase3_real" / "stops_pos.parquet")

    show_all = st.checkbox("Показать все маршруты", value=True)
    rid = st.selectbox("Выбрать маршрут",
                       sorted(flat["route_id"].unique().tolist()),
                       format_func=lambda x: f"Маршрут {x}")

    m = folium.Map(location=[51.66, 39.2], zoom_start=11, tiles="CartoDB positron")
    for _, r in routes.iterrows():
        color = "#e41a1c"
        if not show_all and int(r["route_id"]) != int(rid):
            continue
        if not show_all:
            color = "#2166ac"
        folium.PolyLine(
            [[p[1], p[0]] for p in r.geometry.coords],
            color=color, weight=3 if int(r["route_id"]) == int(rid) else 1.5,
            opacity=0.8,
            popup=f"route {r['route_id']}: {r['n_stops']} ст., {r['length_km']:.1f} км",
        ).add_to(m)
    show_map(m, height=600)

    st.subheader(f"Остановки маршрута {rid}")
    route_stops = flat[flat["route_id"] == int(rid)].sort_values("order")
    names = route_stops["name"].tolist()
    st.write(f"Всего остановок: **{len(names)}**")
    st.write(" → ".join(str(n) for n in names if n is not None))

# ---------------------------------------------------------------------------
elif phase == "Фаза 4 — Пассажиропоток":
    st.title("Фаза 4 — Пассажиропоток на маршрутной сети")
    rep = load_json("phase4/phase4_report.json")
    if rep:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Распределено поездок", f"{rep['assigned_trips']:,.0f} "
                                          f"({rep['assigned_share']*100:.0f}%)")
        c2.metric("Макс. загрузка перегона", f"{rep['max_segment_load']:,.0f}")
        c3.metric("Ср. коэфф. заполнения", f"{rep['avg_load_factor']:.2f}")
        c4.metric("Ср. пересадок", f"{rep['avg_transfers']:.2f}")
    st.markdown(load_report_md("phase4_report.md"))

    seg = pd.read_parquet(CACHE_DIR / "phase4" / "segment_load.parquet")
    routes = gpd.read_parquet(CACHE_DIR / "phase3_real" / "routes.parquet")

    seg_max = float(seg["load"].max()) if len(seg) else 1.0
    look = dict(zip(zip(seg["route_id"], seg["seg_order"]), seg["load"]))
    cmap = folium.LinearColormap(
        ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
        vmin=0, vmax=seg_max, caption="Пассажиропоток на перегоне")

    m = folium.Map(location=[51.66, 39.2], zoom_start=11, tiles="CartoDB positron")
    for _, r in routes.iterrows():
        coords = list(r.geometry.coords)
        for k in range(len(coords) - 1):
            load = look.get((int(r["route_id"]), k), 0.0)
            folium.PolyLine(
                [[coords[k][1], coords[k][0]], [coords[k + 1][1], coords[k + 1][0]]],
                color=cmap(load), weight=3.0, opacity=0.85,
                popup=f"route {r['route_id']} seg {k}: {load:,.0f}",
            ).add_to(m)
    cmap.add_to(m)
    show_map(m, height=620)

    st.subheader("Топ перегонов по загрузке")
    seg_disp = seg.merge(
        routes[["route_id", "n_stops", "length_km"]], on="route_id", how="left")
    st.dataframe(
        seg_disp.sort_values("load", ascending=False).head(100),
        use_container_width=True)
