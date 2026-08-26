"""Command-line entry points; all inputs/outputs are explicit."""

import argparse
import json
from pathlib import Path

from .io import load_config, read_table, record_run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="leg-mri")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dicom-index", help="List reconstructed classic DICOM components")
    p.add_argument("folder")
    p.add_argument("--output", required=True)
    p = sub.add_parser("dicom-convert", help="Convert an explicitly selected component")
    p.add_argument("inventory")
    p.add_argument("group_id")
    p.add_argument("--output", required=True)
    p = sub.add_parser("seg-prepare")
    p.add_argument("manifest")
    p.add_argument("--config", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--splits")
    p = sub.add_parser("seg-plan")
    p.add_argument("--work", required=True)
    p.add_argument("--processes", type=int, default=4)
    p = sub.add_parser("seg-train")
    p.add_argument("--work", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("seg-predict")
    p.add_argument("--work", required=True)
    p.add_argument("--cohort", choices=["development", "test"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--processes", type=int, default=2)
    for command in ("seg-evaluate", "extract"):
        p = sub.add_parser(command)
        p.add_argument("manifest")
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if command == "extract":
            p.add_argument("--quantitative-only", action="store_true")
    p = sub.add_parser("features-combine")
    p.add_argument("folders", nargs="+")
    p.add_argument("--output", required=True)
    p = sub.add_parser("study-run", help="Run a fixed study configuration from NIfTI to ML")
    p.add_argument("--subjects", required=True)
    p.add_argument("--clinical", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--splits")
    p.add_argument("--device", default="cuda")
    p.add_argument("--processes", type=int, default=4)
    p.add_argument(
        "--start-at",
        choices=["prepare", "plan", "train", "predict", "extract", "model"],
        default="prepare",
    )
    for command in ("model-cv", "model-fit"):
        p = sub.add_parser(command)
        p.add_argument("--clinical", required=True)
        p.add_argument("--quantitative")
        p.add_argument("--radiomics")
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        if command == "model-fit":
            p.add_argument("--strategy", required=True)
            p.add_argument("--model", required=True, choices=["LR", "RF", "XGB"])
    p = sub.add_parser("model-predict")
    p.add_argument("model_folder")
    p.add_argument("features")
    p.add_argument("--output", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config) if hasattr(args, "config") else None
    if args.command == "dicom-index":
        from .imaging import dicom_inventory
        from .io import write_json

        write_json(args.output, dicom_inventory(args.folder))
    elif args.command == "dicom-convert":
        from .imaging import convert_dicom_group

        inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        convert_dicom_group(inventory[args.group_id]["files"], args.output)
    elif args.command == "seg-prepare":
        from .segmentation import prepare_dataset

        prepare_dataset(args.manifest, config, args.work, args.splits)
    elif args.command == "seg-plan":
        from .segmentation import plan_and_preprocess

        plan_and_preprocess(args.work, args.processes)
    elif args.command == "seg-train":
        from .segmentation import train_fold

        train_fold(args.work, args.fold, args.device, args.resume)
    elif args.command == "seg-predict":
        from .segmentation import predict_heldout

        predict_heldout(args.work, args.cohort, args.device, args.processes)
    elif args.command == "seg-evaluate":
        from .imaging import evaluate_segmentation

        evaluate_segmentation(args.manifest, config["labels"], args.output)
    elif args.command == "extract":
        from .imaging import extract_features

        extract_features(args.manifest, config, args.output, args.quantitative_only)
    elif args.command == "features-combine":
        from .workflow import combine_features

        combine_features(args.folders, args.output)
    elif args.command == "study-run":
        from .workflow import run_study

        run_study(
            args.subjects,
            args.clinical,
            args.config,
            args.work,
            args.device,
            args.processes,
            args.start_at,
            args.splits,
        )
    elif args.command in ("model-cv", "model-fit"):
        from .modeling import fit_final, load_model_data, run_nested_cv

        df, q, r = load_model_data(args.clinical, args.quantitative, args.radiomics, config)
        if args.command == "model-cv":
            run_nested_cv(df, q, r, config, args.output)
        else:
            fit_final(df, q, r, config, args.strategy, args.model, args.output)
        record_run(
            args.output,
            config,
            [p for p in (args.clinical, args.quantitative, args.radiomics) if p],
        )
    elif args.command == "model-predict":
        import joblib

        model = joblib.load(Path(args.model_folder) / "pipeline.joblib")
        metadata = json.loads((Path(args.model_folder) / "model.json").read_text(encoding="utf-8"))
        id_col = metadata["id_col"]
        df = read_table(args.features, id_col)
        saved_config = json.loads(
            (Path(args.model_folder) / "run.json").read_text(encoding="utf-8")
        )["config"]
        df = df.rename(columns=saved_config.get("feature_aliases", {}))
        result = df[[id_col]].copy()
        result["probability"] = model.predict_proba(df)[:, 1]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    elif args.command == "demo":
        from .demo import run_demo

        run_demo(config, args.output)


if __name__ == "__main__":
    main()
