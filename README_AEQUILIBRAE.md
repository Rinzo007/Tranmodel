# AequilibraE backend

Ветка `aequilibrae-migration` добавляет AequilibraE как вычислительное ядро модели.

## Что изменено

Существующие OSM/WorldPop этапы подготовки данных сохраняются. Новый модуль `src/aequilibrae_pipeline.py` заменяет самописный расчёт сетевой стоимости, гравитационное распределение с Furness и сетевое назначение на соответствующие средства AequilibraE:

`OSM PBF → OSM layers → WorldPop/POI demand → AequilibraE project → network skims → EXPO gravity/IPF → BPR/BFW assignment`

Основные результаты:

- `data/cache/aequilibrae/project` — полноценный проект AequilibraE;
- `data/cache/aequilibrae/gmns/nodes.csv` и `links.csv` — промежуточное представление GMNS;
- `data/cache/aequilibrae/link_load.parquet` — результаты назначения на рёбра;
- `data/cache/aequilibrae/convergence.parquet` — показатели сходимости;
- `data/cache/aequilibrae/aequilibrae_report.json` и `data/report/aequilibrae_report.md` — отчёт.

## Запуск

После подготовки исходных слоёв старой моделью:

```powershell
pip install -r requirements.txt
python -m src.aequilibrae_pipeline
```

Для принудительной пересборки проекта:

```powershell
python -c "from src.aequilibrae_pipeline import run_all; run_all(force=True)"
```

## Важное отличие от старого ядра

AequilibraE выполняет сетевое назначение транспортного спроса на дорожный граф. Старый код `phase3_real.py` хранит существующую маршрутную сеть общественного транспорта, но для полноценного назначения именно на общественный транспорт потребуются расписание/частоты, времена ожидания, пересадки и транспортные режимы. Поэтому на первом этапе AequilibraE используется как вычислительное ядро дорожной сети и распределения спроса, а существующая маршрутная сеть сохраняется для сравнения и последующей миграции.
