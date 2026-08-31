# AequilibraE + TNDP backend

Ветка `aequilibrae-migration` использует AequilibraE как расчётное ядро, а TNDP — как слой синтеза маршрутной сети из матрицы корреспонденций.

## Главный контур модели

`OSM PBF → OSM/WorldPop/POI → OD matrix → demand corridors → candidate routes → TNDP optimization → AequilibraE Transit Assignment → passenger flows`

## TNDP — синтез маршрутов

Новый пакет `src/tndp/` решает задачу проектирования маршрутной сети на основе OD:

- `model.py` — маршруты, набор маршрутов и ограничения;
- `corridors.py` — выделение сильных OD-коридоров;
- `network.py` — построение дорожного графа и компактного графа остановок;
- `candidates.py` — генерация целых кандидатных маршрутов по кратчайшим сетевым путям и через терминалы;
- `optimizer.py` — итеративный отбор маршрутов и локальная замена маршрутов;
- `io.py` — чтение текущей матрицы OD и экспорт решения;
- `run.py` — запуск полного синтеза;
- `benchmark.py` — адаптер формата RenatoArbex/TransitNetworkDesign;
- `cli.py` — запуск из PowerShell/терминала.

Важно: это уже не старый stop-by-stop greedy из `phase3.py`. Оптимизируются целые маршруты относительно всей сети. Текущий оценщик — быстрый суррогат для скрининга кандидатов; архитектура оценщика отделена от генератора, чтобы заменить его полноценным AequilibraE Transit Assignment.

### Запуск TNDP

После создания `data/cache/phase2/matrix_od.parquet` и `data/cache/phase2/stops_matrix.parquet`:

```powershell
python -m src.tndp.cli --min-routes 10 --max-routes 30 --corridors 300 --candidates-per-corridor 8
```

Результаты:

- `data/cache/tndp/generated_routes.json` — сгенерированный набор маршрутов;
- `data/cache/tndp/history.json` — история улучшений;
- `data/cache/tndp/tndp_report.json` — метрики;
- `data/report/tndp_report.md` — отчёт.

Streamlit предоставляет раздел **TNDP — Синтез маршрутов**.

## Источник существующей сети

`voronezh_routes_terminals.geojson` хранится непосредственно в репозитории. Он содержит точки остановок, принадлежность остановок к маршрутам (`routes`), признак конечной остановки (`is_terminal`) и число маршрутов (`route_count`). Этот файл используется как эталон существующей сети и источник реальных ограничений остановочной сети, но не как готовое решение оптимизатора.

## AequilibraE и общественный транспорт

`src/aequilibrae_transit.py` выполняет миграцию заданной маршрутной сети в Public Transport AequilibraE:

`GeoJSON → GTFS → public_transport.sqlite → TransitGraph → Optimal Strategies`

Для модельного запуска без реального GTFS заданы явные параметры: интервал 10 мин, работа 06:00–23:00, скорость 22 км/ч, стоянка 20 с, скорость пешего доступа 4.5 км/ч. Они являются допущениями и сохраняются в отчёте.

## Benchmark

Для проверки алгоритма используется формат `RenatoArbex/TransitNetworkDesign`: узлы, связи, матрица спроса и опубликованные наборы маршрутов. Репозиторий содержит Mandl, Mumford и Rivera — сети разного размера и сложности, а также критерии сравнения решений. После проверки на benchmark-сетях TNDP-решатель переносится на Воронеж.

## Запуск полного расчёта

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Для текущего полного контура AequilibraE:

```powershell
python -c "from src.aequilibrae_full import run_full_model; print(run_full_model(force=True))"
```

Следующая стадия TNDP — заменить суррогатный оценщик в `src/tndp/run.py` на адаптер, который для каждого набора маршрутов строит TransitGraph AequilibraE, выполняет Optimal Strategies и возвращает пользовательские/эксплуатационные показатели. После этого частоты следует калибровать по максимальным загрузкам и итеративно пересчитывать назначение.