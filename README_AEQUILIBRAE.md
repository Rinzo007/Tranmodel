# AequilibraE backend

Ветка `aequilibrae-migration` добавляет AequilibraE как вычислительное ядро модели.

## Что изменено

Существующие OSM/WorldPop этапы подготовки данных сохраняются. Новый модуль `src/aequilibrae_pipeline.py` заменяет самописный расчёт сетевой стоимости, гравитационное распределение с Furness и сетевое назначение на соответствующие средства AequilibraE:

`OSM PBF → OSM layers → WorldPop/POI demand → AequilibraE project → network skims → EXPO gravity/IPF → BPR/BFW assignment`

Эталонная маршрутная сеть Воронежа теперь хранится непосредственно в репозитории:

`voronezh_routes_terminals.geojson`

Файл содержит точки остановок и их принадлежность к маршрутам (`routes`), признак конечной остановки (`is_terminal`) и число маршрутов (`route_count`). Он используется `phase1_real` для формирования зон спроса и `phase3_real` для построения эталонной маршрутной сети. Зависимость от локального пути вида `D:\...\voronezh_routes_terminals.geojson` устранена.

Основные результаты:

- `data/cache/aequilibrae/project` — полноценный проект AequilibraE;
- `data/cache/aequilibrae/gmns/nodes.csv` и `links.csv` — промежуточное представление GMNS;
- `data/cache/aequilibrae/link_load.parquet` — результаты назначения на рёбра;
- `data/cache/aequilibrae/convergence.parquet` — показатели сходимости;
- `data/cache/aequilibrae/aequilibrae_report.json` и `data/report/aequilibrae_report.md` — отчёт.

## Запуск

После подготовки OSM/WorldPop слоёв:

```powershell
pip install -r requirements.txt
python -m src.aequilibrae_pipeline
```

Для принудительной пересборки проекта:

```powershell
python -c "from src.aequilibrae_pipeline import run_all; run_all(force=True)"
```

## Важное отличие от старого ядра

AequilibraE выполняет сетевое назначение транспортного спроса на дорожный граф. Старый код `phase3_real.py` хранит существующую маршрутную сеть общественного транспорта, но для полноценного назначения именно на общественный транспорт потребуются расписание/частоты, времена ожидания, пересадки и транспортные режимы. Поэтому на первом этапе AequilibraE используется как вычислительное ядро дорожной сети и распределения спроса, а `voronezh_routes_terminals.geojson` используется как эталон существующей маршрутной сети для последующей миграции.
