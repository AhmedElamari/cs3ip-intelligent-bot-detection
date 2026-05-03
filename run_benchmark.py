"""
Bot Detection Benchmark Pipeline
================================
CLI entry point for multi-model benchmarking with XAI analysis.

Usage:
    python run_benchmark.py
    python run_benchmark.py --config config/config.yaml
    python run_benchmark.py --explain
"""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import Config, load_config
from benchmarking import ModelBenchmark
from benchmarking.data_prep import load_data, prepare_data
from benchmarking.model_factory import create_models
from benchmarking.output_utils import save_final_outputs
from benchmarking.robustness import run_robustness_analysis
from benchmarking.run_metadata import BenchmarkRunContext
from benchmarking.time_stratified_results import build_temporal_split_dict, format_protocol_note
from benchmarking.xai_reporting import run_explainability_analysis
from benchmarking.hpo.service import (
    HPOCliOverrides,
    merge_hpo_into_config_params,
    resolve_hpo,
)

REPO_ROOT = Path(__file__).resolve().parent
TWIBOT20_DATA_DIR = REPO_ROOT / "data"


def _balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _reset_model_params_to_base(tuned: Config, base: Config) -> None:
    for name in tuned.get('models', {}):
        params = base.get(f'models.{name}.params')
        if params is not None:
            tuned.set(f'models.{name}.params', copy.deepcopy(params))


def _run_hpo_for_config(
    cfg: Config,
    args: argparse.Namespace,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
) -> dict[str, dict]:
    cw = _balanced_class_weights(y_train)
    scale_flag = bool(cfg.get('preprocessing.scale_features'))
    summaries: dict[str, dict] = {}
    for model_name in list(cfg.get_enabled_models()):
        hpo_enable_scaling = scale_flag and model_name in (
            "logistic_regression",
            "svm",
        )
        hpo_result, audit = resolve_hpo(
            model_name,
            cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names_ordered=list(feature_names),
            data_dir=TWIBOT20_DATA_DIR,
            enable_scaling=hpo_enable_scaling,
            class_weights=cw,
            cli=HPOCliOverrides(
                no_tune=args.no_tune,
                retune=args.retune,
                hpo_trials=args.hpo_trials,
            ),
        )
        summaries[model_name] = {**audit, "best_score": hpo_result.get("best_score")}
        if hpo_result.get("status") != "skipped" and hpo_result.get("best_params"):
            merge_hpo_into_config_params(cfg, model_name, hpo_result["best_params"])
    return summaries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Bot Detection Model Benchmarking Pipeline (TwiBot-20 only)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Run explainability analysis'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to benchmark (e.g., random_forest svm)'
    )
    parser.add_argument(
        '--smote',
        action='store_true',
        help='Use SMOTE for class balancing'
    )
    parser.add_argument(
        '--scale',
        action='store_true',
        help='Scale features'
    )
    parser.add_argument(
        '--skip-statistics',
        action='store_true',
        help='Skip bootstrap confidence intervals and pairwise significance tests'
    )
    parser.add_argument(
        '--statistics-bootstrap-samples',
        type=int,
        default=1000,
        help='Bootstrap resamples for inferential statistics (default: 1000)'
    )
    parser.add_argument(
        '--skip-mcnemar',
        action='store_true',
        help='Skip McNemar paired test in pairwise significance output'
    )
    parser.add_argument(
        '--robustness-analysis',
        action='store_true',
        help='Run optional cost-aware adversarial robustness analysis'
    )
    parser.add_argument(
        '--robustness-profiles',
        type=str,
        nargs='+',
        default=None,
        help='Override robustness profiles (e.g. cheap_only realistic_mixed)'
    )
    parser.add_argument(
        '--frs-model',
        type=str,
        default=None,
        help='Model for FRS / SHAP rank stability / cumulative ablation (default: explainability.poster.model or xgboost)',
    )
    parser.add_argument(
        '--frs-shap-top-k',
        type=int,
        default=None,
        help='SHAP top-K for FRS union (default: config robustness.frs.shap_top_k)',
    )
    parser.add_argument(
        '--frs-ablation-top-ks',
        type=int,
        nargs='+',
        default=None,
        help='Cumulative ablation sizes, e.g. 1 3 5 10 (default: config robustness.frs.ablation_top_ks)',
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter optimisation; use config params only',
    )
    parser.add_argument(
        '--retune',
        action='store_true',
        help='Ignore HPO cache and run a fresh Optuna study',
    )
    parser.add_argument(
        '--hpo-trials',
        type=int,
        default=None,
        help='Override number of Optuna trials per model for this run',
    )
    parser.add_argument(
        '--threshold-analysis',
        action='store_true',
        help='Write validation-selected precision-recall threshold audit artifacts',
    )
    parser.add_argument(
        '--threshold-precision-floor',
        type=float,
        default=0.80,
        help='Validation precision floor for threshold analysis (default: 0.80)',
    )
    parser.add_argument(
        '--time-stratified-results',
        action='store_true',
        help=(
            'Run a second chronological concept-drift benchmark (oldest→train, newest→test) '
            'and write time_stratified_scoreboard.* and concept_drift_delta.*'
        ),
    )
    parser.add_argument(
        '--time-stratified-val-size',
        type=float,
        default=None,
        help='Override concept_drift.val_size (default: config)',
    )
    parser.add_argument(
        '--time-stratified-test-size',
        type=float,
        default=None,
        help='Override concept_drift.test_size (default: config)',
    )
    parser.add_argument(
        '--dissertation-core',
        action='store_true',
        help=(
            'Dissertation-focused full run: tuned HPO + all models (when --models is omitted), '
            'bootstrap CIs and pairwise tests (unless you pass --skip-statistics), and full '
            'benchmark artifacts including dissertation_scoreboard.*; skips slow extras '
            '(performance plots, XAI/SHAP, robustness).'
        ),
    )
    parser.add_argument(
        '--scoreboard-only',
        dest='dissertation_core',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_explainability_audit(args: argparse.Namespace, config: Config) -> dict:
    requested_by_cli = bool(args.explain)
    enabled_in_config = bool(config.get('explainability.enabled'))

    if requested_by_cli and enabled_in_config:
        effective_source = 'cli+config'
    elif requested_by_cli:
        effective_source = 'cli'
    elif enabled_in_config:
        effective_source = 'config'
    else:
        effective_source = 'disabled'

    return {
        'xai_enabled': requested_by_cli or enabled_in_config,
        'xai_requested_by_cli': requested_by_cli,
        'xai_enabled_in_config': enabled_in_config,
        'xai_effective_source': effective_source,
    }


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.no_tune and args.retune:
        parser.error("Cannot use --no-tune together with --retune")

    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    if args.dissertation_core:
        args.explain = False
        args.robustness_analysis = False
        config.set('output.save_plots', False)
        config.set('robustness.enabled', False)

    if args.smote:
        config.set('preprocessing.handle_imbalance', True)
        config.set('preprocessing.imbalance_method', 'smote')
    if args.scale:
        config.set('preprocessing.scale_features', True)
    if args.explain:
        config.set('output.save_plots', True)
    if args.robustness_analysis:
        config.set('robustness.enabled', True)
    if args.robustness_analysis or config.get('robustness.enabled'):
        if args.frs_model is not None:
            config.set('robustness.frs.model', args.frs_model)
        if args.frs_shap_top_k is not None:
            config.set('robustness.frs.shap_top_k', int(args.frs_shap_top_k))
        if args.frs_ablation_top_ks is not None:
            config.set('robustness.frs.ablation_top_ks', [int(x) for x in args.frs_ablation_top_ks])
    if args.robustness_profiles:
        config.set('robustness.profiles', args.robustness_profiles)
    if args.models:
        known_models = set(config.get('models', {}).keys())
        unknown = [m for m in args.models if m not in known_models]
        if unknown:
            parser.error(
                f"Unknown model(s): {unknown}. Available: {sorted(known_models)}\n"
                "Note: 'gradient_boosting' has been replaced by 'xgboost'."
            )
        for model_name in known_models:
            config.set(f'models.{model_name}.enabled', model_name in args.models)
    elif args.dissertation_core:
        for model_name in config.get('models', {}).keys():
            config.set(f'models.{model_name}.enabled', True)

    enabled_models = config.get_enabled_models()
    scale_from_config = config.get('preprocessing.scale_features')
    scaled_models = {'logistic_regression', 'svm'}
    needs_scaling = any(m in enabled_models for m in scaled_models)
    if not scale_from_config and needs_scaling:
        config.set('preprocessing.scale_features', True)
        print(
            "\n[Compatibility] Scaling disabled by config but logistic_regression/svm enabled; "
            "auto-restoring scaling for those models."
        )

    if args.time_stratified_val_size is not None:
        config.set('concept_drift.val_size', float(args.time_stratified_val_size))
    if args.time_stratified_test_size is not None:
        config.set('concept_drift.test_size', float(args.time_stratified_test_size))
    if args.time_stratified_results:
        config.set('concept_drift.enabled', True)

    config_before_hpo = copy.deepcopy(config)

    output_dir = Path(args.output)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f'benchmark_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    data_source_str = str(TWIBOT20_DATA_DIR)

    print("="*60)
    print("BOT DETECTION MODEL BENCHMARK")
    print("="*60)
    if args.dissertation_core:
        stats = "off" if args.skip_statistics else "on"
        print(
            "Mode: --dissertation-core (HPO + all models; inferential stats "
            f"{stats}; no XAI / robustness; still writes dissertation PR/CM figures)"
        )
    print(f"Data:   {data_source_str}")
    print(f"Output: {output_dir}")
    print(f"Models: {config.get_enabled_models()}")

    explainability_audit = _resolve_explainability_audit(args, config)
    if args.dissertation_core:
        explainability_audit = {
            'xai_enabled': False,
            'xai_requested_by_cli': False,
            'xai_enabled_in_config': bool(config.get('explainability.enabled')),
            'xai_effective_source': 'dissertation_core',
        }
    state = "enabled" if explainability_audit['xai_enabled'] else "disabled"
    print(f"Explainability: {state} (source: {explainability_audit['xai_effective_source']})")

    data_splits = load_data(TWIBOT20_DATA_DIR)

    prepared_data = prepare_data(data_splits, config, return_metadata=True)
    if len(prepared_data) == 8:
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_names,
            test_metadata,
        ) = prepared_data
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepared_data
        test_metadata = None

    hpo_summaries = _run_hpo_for_config(
        config,
        args,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )

    summary_path = output_dir / "hpo_summary.json"
    summary_path.write_text(
        json.dumps(
            {"schema_version": "HPOSummaryV1", "models": hpo_summaries},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nHPO summary written to: {summary_path}")

    models = create_models(config)

    benchmark = ModelBenchmark(
        models=models,
        experiment_name=f'benchmark_{timestamp}'
    )
    benchmark.hpo_audit_by_model = hpo_summaries
    if hasattr(benchmark, "set_test_metadata"):
        benchmark.set_test_metadata(test_metadata)

    benchmark.run_benchmark(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=feature_names,
        verbose=True,
        compute_statistics=not args.skip_statistics,
        statistics_bootstrap_samples=args.statistics_bootstrap_samples,
        include_mcnemar=not args.skip_mcnemar,
        enable_scaling=config.get('preprocessing.scale_features'),
    )

    benchmark.print_summary()

    drift_benchmark = None
    drift_protocol_note = ""
    if config.get('concept_drift.enabled'):
        drift_cfg = copy.deepcopy(config)
        _reset_model_params_to_base(drift_cfg, config_before_hpo)
        temporal = build_temporal_split_dict(
            data_splits,
            val_size=float(drift_cfg.get('concept_drift.val_size', 0.2)),
            test_size=float(drift_cfg.get('concept_drift.test_size', 0.1)),
            time_col=str(drift_cfg.get('concept_drift.time_col', 'account_creation_date')),
            random_state=int(drift_cfg.get('random_state', 2112)),
            min_samples_per_split=int(drift_cfg.get('concept_drift.min_samples_per_split', 1)),
        )
        drift_protocol_note = format_protocol_note(
            temporal['train'],
            temporal['val'],
            temporal['test'],
            time_col=str(drift_cfg.get('concept_drift.time_col', 'account_creation_date')),
            reference_date_policy=str(
                drift_cfg.get('concept_drift.reference_date_policy', 'dataset_observation_anchor')
            ),
        )
        prepared_drift = prepare_data(
            temporal,
            drift_cfg,
            return_metadata=False,
            temporal_protocol=True,
        )
        Xd_tr, Xd_va, Xd_te, yd_tr, yd_va, yd_te, fd_names = prepared_drift
        hpo_drift = _run_hpo_for_config(
            drift_cfg,
            args,
            X_train=Xd_tr,
            y_train=yd_tr,
            X_val=Xd_va,
            y_val=yd_va,
            feature_names=fd_names,
        )
        models_drift = create_models(drift_cfg)
        drift_benchmark = ModelBenchmark(
            models=models_drift,
            experiment_name=f'benchmark_{timestamp}_concept_drift'
        )
        drift_benchmark.hpo_audit_by_model = hpo_drift
        drift_benchmark.run_benchmark(
            Xd_tr, yd_tr,
            Xd_va, yd_va,
            Xd_te, yd_te,
            feature_names=fd_names,
            verbose=True,
            compute_statistics=not args.skip_statistics,
            statistics_bootstrap_samples=args.statistics_bootstrap_samples,
            include_mcnemar=not args.skip_mcnemar,
            enable_scaling=drift_cfg.get('preprocessing.scale_features'),
        )
        drift_benchmark.print_summary()

    skip_slow_aux = args.dissertation_core

    if not skip_slow_aux and explainability_audit['xai_enabled']:
        run_explainability_analysis(
            benchmark,
            feature_names,
            config,
            output_dir
        )

    if not skip_slow_aux and config.get('robustness.enabled'):
        run_robustness_analysis(
            benchmark,
            feature_names,
            config,
            output_dir,
        )

    config_path = str(Path(args.config).resolve()) if args.config else None
    run_context = BenchmarkRunContext(
        argv=list(sys.argv[1:]),
        args=vars(args).copy(),
        config_path=config_path,
        repo_root=REPO_ROOT,
        data_dir=TWIBOT20_DATA_DIR,
        output_dir=output_dir,
        explainability=explainability_audit,
    )

    save_final_outputs(
        benchmark,
        output_dir,
        config,
        run_context,
        threshold_analysis_enabled=args.threshold_analysis,
        threshold_precision_floor=args.threshold_precision_floor,
        drift_benchmark=drift_benchmark,
        drift_protocol_note=drift_protocol_note,
    )

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")

    best_name, _, best_metrics = benchmark.get_best_model('f1')
    print(f"\n[BEST] Model: {best_name}")
    print(f"   F1 Score:  {best_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {best_metrics.get('roc_auc', 'N/A')}")

    return benchmark


if __name__ == '__main__':
    main()
