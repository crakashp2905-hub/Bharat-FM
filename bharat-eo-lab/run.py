"""bharat-eo-lab orchestrator.

Examples
--------
  python run.py smoke                      # synthetic end-to-end test (no account)
  python run.py export-local --project ee-yourproject
  python run.py analyze --csv data/raw/india_landcover_2024.csv
  python run.py export-drive --project ee-yourproject
"""
from __future__ import annotations
import argparse
import time
import pandas as pd

from src import config as C
from src import utils, benchmark, viz


def analyze(csv_path, tag="india_landcover"):
    """Load a sample table, run Bharat-EO-Bench v0, write figures + report."""
    t0 = time.time()
    df = utils.load_table(csv_path)
    labs, names = utils.present_classes(df)
    print(f"[analyze] {len(df):,} rows | {len(labs)} classes: {names}")

    lb, best = benchmark.run_benchmark(df)
    print("\n=== Bharat-EO-Bench v0 leaderboard ===")
    print(lb.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # report
    md = benchmark.leaderboard_markdown(lb, df)
    rep_path = C.OUT_REP / f"{tag}_leaderboard.md"
    rep_path.write_text(md, encoding="utf-8")
    lb.to_csv(C.OUT_REP / f"{tag}_leaderboard.csv", index=False)
    utils.save_json({k: v for k, v in best.items()
                     if k not in ("model", "test_idx")},
                    C.OUT_REP / f"{tag}_best_metrics.json")

    # figures
    if best is not None:
        cm = viz.confusion_png(best, C.OUT_FIG / f"{tag}_confusion.png")
        f1 = viz.per_class_f1_png(best, C.OUT_FIG / f"{tag}_per_class_f1.png")
        mp = viz.prediction_map_html(df, best, C.OUT_FIG / f"{tag}_pred_map.html")
        print(f"\n[analyze] figures: {cm.name}, {f1.name}"
              + (f", {mp.name}" if mp else ""))
    print(f"[analyze] report: {rep_path}")
    print(f"[analyze] done in {time.time()-t0:.1f}s")
    return lb, best


def main():
    ap = argparse.ArgumentParser(description="bharat-eo-lab pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="synthetic end-to-end test (no GEE account)")

    el = sub.add_parser("export-local", help="pull a small India sample to CSV")
    el.add_argument("--project", default=C.EE_PROJECT)
    el.add_argument("--year", type=int, default=C.YEAR)
    el.add_argument("--points", type=int, default=None)

    ed = sub.add_parser("export-drive", help="launch a large Export to Drive")
    ed.add_argument("--project", default=C.EE_PROJECT)
    ed.add_argument("--year", type=int, default=C.YEAR)
    ed.add_argument("--points", type=int, default=None)

    bo = sub.add_parser("boost", help="targeted rare-class sampling + merge + analyze")
    bo.add_argument("--project", default=C.EE_PROJECT)
    bo.add_argument("--year", type=int, default=C.YEAR)
    bo.add_argument("--per-box", type=int, default=C.PER_BOX_POINTS)

    fl = sub.add_parser("flood", help="Assam 2022 flood susceptibility task")
    fl.add_argument("--project", default=C.EE_PROJECT)
    fl.add_argument("--points", type=int, default=12000)

    gw = sub.add_parser("groundwater", help="GRACE groundwater depletion (Punjab/NW India)")
    gw.add_argument("--project", default=C.EE_PROJECT)

    ds = sub.add_parser("gw-downscale", help="downscale GRACE -> ~1.5km + district ranking")
    ds.add_argument("--project", default=C.EE_PROJECT)

    gv = sub.add_parser("gw-validate", help="validate depletion: GRACE-FO + cross-center + curve")
    gv.add_argument("--project", default=C.EE_PROJECT)

    ec = sub.add_parser("export-chips", help="export S2 image chips + WorldCover masks for Prithvi fine-tune")
    ec.add_argument("--project", default=C.EE_PROJECT)
    ec.add_argument("--n", type=int, default=200)

    ect = sub.add_parser("export-chips-targeted", help="targeted chips in rare-class regions")
    ect.add_argument("--project", default=C.EE_PROJECT)
    ect.add_argument("--per-class", type=int, default=25)

    an = sub.add_parser("analyze", help="run Bharat-EO-Bench v0 on a CSV")
    an.add_argument("--csv", default=str(C.DEFAULT_CSV))
    an.add_argument("--tag", default="india_landcover")

    args = ap.parse_args()

    if args.cmd == "smoke":
        from src import smoke_test
        csv = smoke_test.main()
        analyze(csv, tag="smoke")
    elif args.cmd == "export-local":
        from src import gee_export
        csv = gee_export.export_local(args.project, args.year,
                                      args.points or C.N_POINTS_LOCAL)
        analyze(csv, tag=f"india_landcover_{args.year}")
    elif args.cmd == "export-drive":
        from src import gee_export
        gee_export.export_drive(args.project, args.year,
                                args.points or C.N_POINTS_DRIVE)
    elif args.cmd == "boost":
        from src import gee_export
        boost_csv = gee_export.export_boost(args.project, args.year, args.per_box)
        base = pd.read_csv(C.DEFAULT_CSV)
        boost = pd.read_csv(boost_csv)
        combined = (pd.concat([base, boost], ignore_index=True)
                    .drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True))
        combined.to_csv(C.COMBINED_CSV, index=False)
        print(f"[boost] combined {len(base):,} base + {len(boost):,} boost "
              f"-> {len(combined):,} rows -> {C.COMBINED_CSV}")
        analyze(str(C.COMBINED_CSV), tag=f"india_landcover_{args.year}_boosted")
    elif args.cmd == "flood":
        from src import flood
        csv = flood.export_flood(args.project, args.points)
        flood.run_flood(csv)
    elif args.cmd == "groundwater":
        from src import groundwater
        csv = groundwater.export_gw(args.project)
        groundwater.run_gw(csv, args.project)
    elif args.cmd == "gw-downscale":
        from src import gw_downscale as ds
        tc = ds.export_train(args.project)
        gc = ds.export_grid(args.project)
        ds.run_downscale(tc, gc, args.project)
    elif args.cmd == "gw-validate":
        from src import gw_validate as gv
        gv.run_validate(args.project)
    elif args.cmd == "export-chips":
        from src import export_chips
        export_chips.export_chips(args.project, args.n)
    elif args.cmd == "export-chips-targeted":
        from src import export_chips
        export_chips.export_chips_targeted(args.project, args.per_class)
    elif args.cmd == "analyze":
        analyze(args.csv, tag=args.tag)


if __name__ == "__main__":
    main()
