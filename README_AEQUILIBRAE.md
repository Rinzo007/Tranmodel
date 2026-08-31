# AequilibraE backend

Ветка `aequilibrae-migration` использует AequilibraE как вычислительное ядро транспортной модели.

## Контур модели

`OSM PBF → OSM layers → WorldPop/POI demand → AequilibraE project → network skims → EXPO gravity/IPF → road BFW/BPR + PT Optimal Strategies`

## Исходная маршрутная сеть

Файл `voronezh_routes_terminals.geojson` хранится непосредственно в репозитории. Он содержит точки остановок, принадлежность остановок к маршрутам (`routes`), признак конечной остановки (`is_terminal`) и число маршрутов (`route_count`). `phase1_real` использует его как источник остановок, а `phase3_real` — как источник существующей маршрутной сети.

## Общественный транспорт

`src/aequilibrae_transit.py` выполняет полноценную миграцию этой сети в модуль Public Transport AequilibraE:

`GeoJSON → упорядоченные остановки маршрутов → GTFS → public_transport.sqlite → TransitGraph → Optimal Strategies`

Маршруты получают исходные номера из GeoJSON, а каждое направление моделируется отдельным GTFS-потоком. Остановки, пересадки и пешеходные связи строятся средствами AequilibraE.

Поскольку исходный GeoJSON не содержит расписаний, частот, календаря и времени движения, в GTFS явно заданы параметры модели:

- интервал: 10 мин;
- период работы: 06:00–23:00;
- средняя скорость: 22 км/ч;
- задержка на остановке: 20 с;
- скорость пешего доступа: 4.5 км/ч;
- максимальная длина коннектора зоны: 800 м.

Эти значения являются допущениями модели и сохраняются в `data/cache/aequilibrae/transit/transit_report.json`.

Матрица корреспонденций, полученная AequilibraE EXPO/IPF, используется повторно для назначения спроса на общественный транспорт. Назначение выполняется алгоритмом **Optimal Strategies (Spiess & Florian)**.

## Основные результаты

- `data/cache/aequilibrae/project` — проект AequilibraE;
- `data/cache/aequilibrae/gmns/nodes.csv` и `links.csv` — GMNS-представление дорожной сети;
- `data/cache/aequilibrae/link_load.parquet` — загрузка дорожных рёбер;
- `data/cache/aequilibrae/convergence.parquet` — сходимость дорожного назначения;
- `data/cache/aequilibrae/transit/voronezh_reference_gtfs.zip` — сгенерированный GTFS;
- `data/cache/aequilibrae/transit/transit_link_load.parquet` — загрузка рёбер общественного транспорта;
- `data/cache/aequilibrae/transit/transit_report.json` — отчёт назначения ОТ;
- `data/report/transit_report.md` — человекочитаемый отчёт.

## Запуск

После подготовки OSM/WorldPop слоёв:

```powershell
pip install -r requirements.txt
python -c "from src.aequilibrae_full import run_full_model; print(run_full_model(force=True))"
```

Либо через Streamlit:

```powershell
streamlit run app.py
```

В разделе **AequilibraE** кнопка «Запустить полный расчёт» выполняет построение проекта, распределение спроса, дорожное назначение и назначение пассажиропотока на реальную маршрутную сеть.

## Техническое ограничение

Для детального прогнозирования пассажиропотока на уровне расписаний желательно заменить модельные допущения реальным GTFS/расписанием: частоты по маршрутам и времени суток, время движения между остановками, календарь, тарифы и типы подвижного состава. Текущая реализация не придумывает эти данные внутри GeoJSON и хранит их как явные параметры модели.
