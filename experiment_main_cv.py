



import os
import pandas as pd
import matplotlib.pyplot as plt
from dcmfanet_core8 import load_omics_data, run_dcmf_cv, plot_roc_curves_cv

#


def main():
    data_dir = "D:/Article/dataset/datanew/UCEC/"#/datanew
    mrna_path = data_dir + "exptransp.csv"
    meth_path = data_dir + "methytransp.csv"
    mirna_path = data_dir + "mirnatransp.csv"
    label_path = data_dir + "labels.csv"

    # Balanced feature counts
    X1, X2, X3, y, _ = load_omics_data(
        mrna_path, meth_path, mirna_path, label_path,
        n_features_mrna=2000,
        n_features_meth=2000,
        n_features_mirna=500,
        batch_correct=False,
        batch_labels=None,
        feature_orders=None,
        use_mi=False
    )

    output_base = data_dir + "results_final/"
    os.makedirs(output_base, exist_ok=True)
    out_dir = os.path.join(output_base, "main_cv")
    os.makedirs(out_dir, exist_ok=True)

    loss_weights = (1.0, 0.01, 0.01, 0.001)
    #loss_weights = (1.0, 0.0, 0.0, 0.0)

    results, fused_reps, train_histories, summary, all_test_probs, all_test_labels = run_dcmf_cv(
        X1, X2, X3, y,
        n_folds=5,
        n_trials=25,#100
        epochs=100,
        loss_weights=loss_weights,
        encoder_type='mlp',          # or 'lstm'
        graph_metric='pearson',
        use_smote=True,
        use_focal=True,
        focal_gamma=0.5,             # lowered from 2.0,0.5
        early_stopping_patience=200,  # increased patience
        record_train_val=True,
        save_dir=out_dir
    )

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "dcmfanet_cv_results.csv"), index=False)
    with pd.ExcelWriter(os.path.join(out_dir, "dcmfanet_cv_results.xlsx")) as writer:
        df_results.to_excel(writer, sheet_name="DCMFANet", index=False)
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

    # ROC curves
    plot_roc_curves_cv(all_test_probs, all_test_labels, out_dir)

    # Per‑fold bar plot
    metrics_to_plot = ['accuracy', 'f1', 'auc']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        ax.bar(df_results['fold'], df_results[metric],
               yerr=df_results[metric].std() / 2,
               capsize=5, alpha=0.7, color='skyblue')
        ax.axhline(df_results[metric].mean(), color='red', linestyle='--', label='Mean')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'DCMFANet {metric.upper()} per fold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dcmfanet_per_fold.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # Training curves (last fold)
    if train_histories:
        hist = train_histories[-1]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(hist['train_loss'], color='tab:red', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(hist['train_acc'], color='tab:blue', linestyle='--', label='Train Acc')
        #ax2.plot(hist['val_acc'], color='tab:green', linestyle='-.', label='Val Acc')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title(f'Training Curves (Fold {hist["fold"]}) with Early Stopping')
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.jpg"), dpi=300)
        plt.close()

    print(f"\n Main CV results saved to {out_dir}")


if __name__ == "__main__":
    main()




"""import os
import pandas as pd
import matplotlib.pyplot as plt
from dcmfanet_core9 import load_omics_data, run_dcmf_cv, plot_roc_curves_cv

def main():
    data_dir = "D:/XiangYa2yyPaper/PAADWork/Article/dataset/datanew/liver/"  # change to your path
    mrna_path = data_dir + "exptransp.csv"
    meth_path = data_dir + "methytransp.csv"
    mirna_path = data_dir + "mirnatransp.csv"
    label_path = data_dir + "labels.csv"

    # Load data with variance filtering (unsupervised)
    X1, X2, X3, y, _ = load_omics_data(
        mrna_path, meth_path, mirna_path, label_path,
        n_features_mrna=2000,
        n_features_meth=2000,
        n_features_mirna=500,
        batch_correct=False
    )

    output_base = data_dir + "results_final_v2/"
    os.makedirs(output_base, exist_ok=True)
    out_dir = os.path.join(output_base, "main_cv")
    os.makedirs(out_dir, exist_ok=True)

    loss_weights = (1.0, 0.01, 0.01, 0.001)

    results, fused_reps, train_histories, summary, all_test_probs, all_test_labels = run_dcmf_cv(
        X1, X2, X3, y,
        n_folds=5,
        n_trials=15,
        epochs=500,
        loss_weights=loss_weights,
        encoder_type='mlp',
        graph_metric='pearson',
        use_smote=True,
        use_focal=True,
        focal_gamma=0.5,
        early_stopping_patience=15,
        oversampler='borderline',   # or 'adasyn' (with fallback)
        feature_selection='variance',  # or 'mi' for mutual information per fold
        record_train_val=True,
        save_dir=out_dir
    )

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "dcmfanet_cv_results.csv"), index=False)
    with pd.ExcelWriter(os.path.join(out_dir, "dcmfanet_cv_results.xlsx")) as writer:
        df_results.to_excel(writer, sheet_name="DCMFANet", index=False)
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

    # ROC curves
    plot_roc_curves_cv(all_test_probs, all_test_labels, out_dir)

    # Per‑fold performance bar plot
    metrics_to_plot = ['accuracy', 'f1', 'auc']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15,5))
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        ax.bar(df_results['fold'], df_results[metric],
               yerr=df_results[metric].std()/2,
               capsize=5, alpha=0.7, color='skyblue')
        ax.axhline(df_results[metric].mean(), color='red', linestyle='--', label='Mean')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'DCMFANet {metric.upper()} per fold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dcmfanet_per_fold.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # Training curves (last fold)
    if train_histories:
        hist = train_histories[-1]
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(hist['train_loss'], color='tab:red', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(hist['train_acc'], color='tab:blue', linestyle='--', label='Train Acc')
        ax2.plot(hist['val_acc'], color='tab:green', linestyle='-.', label='Val Acc')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title(f'Training Curves (Fold {hist["fold"]}) with Early Stopping')
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.jpg"), dpi=300)
        plt.close()

    print(f"\n✅ Main CV results saved to {out_dir}")

if __name__ == "__main__":
    main()"""



# experiment_main_cv.py
# Systematic debugging script for DCMFANet
# Allows testing of GCN, oversampling, augmentation individually.

"""import os
import pandas as pd
import matplotlib.pyplot as plt
from dcmfanet_core8 import (
    load_omics_data,
    run_dcmf_cv,
    plot_roc_curves_cv,
    run_augmentation_comparison,
    build_knn_graph          # optional, if you want to try k‑NN
)

def main():
    # ---------------------------
    #  CONFIGURATION
    # ---------------------------
    data_dir = "D:/XiangYa2yyPaper/PAADWork/Article/dataset/datanew/melanoma/"
    mrna_path   = data_dir + "exptransp.csv"
    meth_path   = data_dir + "methytransp.csv"
    mirna_path  = data_dir + "mirnatransp.csv"
    label_path  = data_dir + "labels.csv"

    # Feature counts – keep moderate to avoid overfitting
    N_MRNA  = 2000
    N_METH  = 2000
    N_MIRNA = 500

    # ---------------------------
    #  LOAD DATA
    # ---------------------------
    X1, X2, X3, y, _ = load_omics_data(
        mrna_path, meth_path, mirna_path, label_path,
        n_features_mrna=N_MRNA,
        n_features_meth=N_METH,
        n_features_mirna=N_MIRNA,
        batch_correct=False,
        batch_labels=None,
        feature_orders=None
    )

    print(f"Data shapes: mRNA {X1.shape}, Meth {X2.shape}, miRNA {X3.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

    output_base = data_dir + "results_revised/"
    os.makedirs(output_base, exist_ok=True)

    # ============================================================
    #  OPTIONAL: RUN AUGMENTATION COMPARISON (quick 3‑fold)
    # ============================================================
    RUN_AUG_COMPARISON = False   # set to True if you want to compare strategies
    if RUN_AUG_COMPARISON:
        aug_out_dir = os.path.join(output_base, "augmentation_comparison")
        os.makedirs(aug_out_dir, exist_ok=True)
        print("\n" + "="*70)
        print(" RUNNING AUGMENTATION COMPARISON (3‑fold CV, 30 epochs each)")
        print("="*70)
        run_augmentation_comparison(
            X1, X2, X3, y,
            output_dir=aug_out_dir,
            n_folds=3,
            epochs=30
        )
        print(f"✅ Augmentation comparison saved to {aug_out_dir}\n")

    # ============================================================
    #  STEP 1: QUICK DEBUG – SIMPLEST CONFIGURATION
    #  (No GCN, no augmentation, only Borderline‑SMOTE)
    # ============================================================
    RUN_DEBUG = True   # set to False after debugging
    if RUN_DEBUG:
        debug_dir = os.path.join(output_base, "debug_no_gcn")
        os.makedirs(debug_dir, exist_ok=True)

        print("\n" + "="*70)
        print(" DEBUG RUN: NO GCN, BORDERLINE‑SMOTE ONLY, NO MIXUP/NOISE")
        print("="*70)

        results, _, _, summary, _, _ = run_dcmf_cv(
            X1, X2, X3, y,
            n_folds=3,                  # quick, use 3 folds
            n_trials=5,                 # minimal hyperparameter search
            epochs=50,                  # enough to see trends
            loss_weights=(1.0, 0.01, 0.01, 0.001),
            encoder_type='mlp',
            graph_metric='cosine',
            graph_threshold=0.5,
            use_smote=True,
            use_borderline_smote=True,  # safe oversampling
            use_adasyn=False,
            use_mixup=False,
            noise_factor=0.0,
            use_gcn=False,              # <-- GCN disabled
            early_stopping_patience=15,
            save_dir=debug_dir
        )

        # Print summary
        print("\nDebug run summary:")
        for k, v in summary.items():
            print(f"  {k}: mean={v['mean']:.4f} ± {v['std']:.4f}")
        print(f"✅ Debug results saved to {debug_dir}")

    # ============================================================
    #  STEP 2: FULL CV WITH BEST SETTINGS (if debug looks good)
    # ============================================================
    RUN_MAIN_CV = True   # set to True when you're ready
    if RUN_MAIN_CV:
        out_dir = os.path.join(output_base, "main_cv_final")
        os.makedirs(out_dir, exist_ok=True)

        loss_weights = (1.0, 0.01, 0.01, 0.001)

        print("\n" + "="*70)
        print(" RUNNING FINAL 5‑FOLD CV (choose your settings below)")
        print("="*70)

        # ---- Adjust these parameters based on debug results ----
        results, fused_reps, train_histories, summary, all_test_probs, all_test_labels = run_dcmf_cv(
            X1, X2, X3, y,
            n_folds=5,
            n_trials=10,                # moderate search
            epochs=200,                 # early stopping will cut if needed
            loss_weights=loss_weights,
            encoder_type='mlp',         # or 'lstm' / 'transformer'
            graph_metric='cosine',
            graph_threshold=0.5,
            use_smote=True,
            use_borderline_smote=True,  # start with this; change to ADASYN later
            use_adasyn=False,
            use_mixup=False,            # enable one at a time
            noise_factor=0.0,
            use_gcn=True,               # re‑enable GCN after debug confirms it works
            early_stopping_patience=20,
            save_dir=out_dir
        )

        # ---- Save and plot results ----
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(out_dir, "dcmfanet_cv_results.csv"), index=False)
        with pd.ExcelWriter(os.path.join(out_dir, "dcmfanet_cv_results.xlsx")) as writer:
            df_results.to_excel(writer, sheet_name="DCMFANet", index=False)
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

        plot_roc_curves_cv(all_test_probs, all_test_labels, out_dir)

        # Per‑fold bar plot
        metrics_to_plot = ['accuracy', 'f1', 'auc']
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            ax.bar(df_results['fold'], df_results[metric],
                   yerr=df_results[metric].std() / 2,
                   capsize=5, alpha=0.7, color='skyblue')
            ax.axhline(df_results[metric].mean(), color='red', linestyle='--', label='Mean')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'DCMFANet {metric.upper()} per fold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dcmfanet_per_fold.jpg"), dpi=300, bbox_inches='tight')
        plt.close()

        # Training curves (last fold)
        if train_histories:
            hist = train_histories[-1]
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='tab:red')
            ax1.plot(hist['train_loss'], color='tab:red', label='Train Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Accuracy', color='tab:blue')
            ax2.plot(hist['train_acc'], color='tab:blue', linestyle='--', label='Train Acc')
            ax2.plot(hist['val_acc'], color='tab:green', linestyle='-.', label='Val Acc')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            plt.title(f'Training Curves (Fold {hist["fold"]})')
            fig.tight_layout()
            plt.savefig(os.path.join(out_dir, "training_curves.jpg"), dpi=300)
            plt.close()

        print(f"\n✅ Main CV results saved to {out_dir}")

    print("\n" + "="*70)
    print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f" Results saved under: {output_base}")
    print("="*70)

if __name__ == "__main__":
    main()"""



# experiment_main_cv.py
"""import os
import pandas as pd
import matplotlib.pyplot as plt
from dcmfanet_core6 import load_omics_data, run_dcmf_cv, plot_roc_curves_cv

def main():
    data_dir = "D:/XiangYa2yyPaper/PAADWork/Article/dataset/datanew/PAAD/"
    mrna_path = data_dir + "exptransp.csv"
    meth_path = data_dir + "methytransp.csv"
    mirna_path = data_dir + "mirnatransp.csv"
    label_path = data_dir + "labels.csv"

    # ---- 1. Load data with variance filtering to reduce feature space ----
    # Use moderate values to keep only the most variable features.
    # This is an unsupervised pre‑filter to reduce dimensionality.
    X1, X2, X3, y, processors = load_omics_data(
        mrna_path, meth_path, mirna_path, label_path,
        n_features_mrna=10000,      # keep top 1000 mRNA features
        n_features_meth=4000,       # top 500 methylation features
        n_features_mirna=1000,      # top 200 miRNA features
        batch_correct=False,
        batch_labels=None,
        feature_orders=None
    )

    output_base = data_dir + "results_revised/"
    os.makedirs(output_base, exist_ok=True)
    out_dir = os.path.join(output_base, "main_cv")
    os.makedirs(out_dir, exist_ok=True)

    loss_weights = (1.0, 0.01, 0.01, 0.001)  # can be tuned

    # ---- 2. Run cross‑validation with per‑fold supervised selection ----
    # The function now correctly splits train/val before SMOTE and feature selection.
    # We use a smaller number of trials and epochs to avoid overfitting.
    results, fused_reps, train_histories, summary, all_test_probs, all_test_labels = run_dcmf_cv(
        X1, X2, X3, y,
        n_folds=5,
        n_trials=100,                # fewer trials for faster execution
        epochs=1000,                 # reduced from 1000; use early stopping if needed
        loss_weights=loss_weights,
        encoder_type='mlp',         # MLP is less prone to overfitting than LSTM
        graph_metric='pearson',
        use_smote=True,
        save_dir=out_dir,
        use_supervised_selection=True,          # enable per‑fold selection
        supervised_n_features={'mRNA': 2000, 'Methylation': 2000, 'miRNA': 1050},
        selection_method='anova'                 # or 'mutual_info'
    )

    # ---- 3. Save results ----
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "dcmfanet_cv_results.csv"), index=False)
    with pd.ExcelWriter(os.path.join(out_dir, "dcmfanet_cv_results.xlsx")) as writer:
        df_results.to_excel(writer, sheet_name="DCMFANet", index=False)
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

    # ---- 4. ROC curves ----
    plot_roc_curves_cv(all_test_probs, all_test_labels, out_dir)

    # ---- 5. Per‑fold performance bar plot ----
    metrics_to_plot = ['accuracy', 'f1', 'auc']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        ax.bar(df_results['fold'], df_results[metric],
               yerr=df_results[metric].std() / 2,
               capsize=5, alpha=0.7, color='skyblue')
        ax.axhline(df_results[metric].mean(), color='red', linestyle='--', label='Mean')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'DCMFANet {metric.upper()} per fold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dcmfanet_per_fold.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 6. Training curves (last fold) ----
    if train_histories:
        hist = train_histories[-1]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(hist['train_loss'], color='tab:red', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(hist['train_acc'], color='tab:blue', linestyle='--', label='Train Acc')
        # If validation accuracy was recorded, uncomment the next line:
        # ax2.plot(hist['val_acc'], color='tab:green', linestyle='-.', label='Val Acc')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title(f'Training Curves (Fold {hist["fold"]})')
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.jpg"), dpi=300)
        plt.close()

    print(f"\n✅ Main CV results saved to {out_dir}")

if __name__ == "__main__":
    main()"""




# experiment_main_cv.py
"""import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dcmfanet_core7 import load_omics_data, run_dcmf_cv, plot_roc_curves_cv

def run_single_experiment(selection_method, data_dir, output_base,
                          n_folds=5, n_trials=100, epochs=1000,
                          encoder_type='mlp', graph_metric='pearson',
                          use_smote=True):
    
    print("\n" + "=" * 70)
    print(f"  RUNNING EXPERIMENT WITH SELECTION METHOD: {selection_method.upper()}")
    print("=" * 70)

    # ---- 1. File paths ----
    mrna_path = data_dir + "exptransp.csv"
    meth_path = data_dir + "methytransp.csv"
    mirna_path = data_dir + "mirnatransp.csv"
    label_path = data_dir + "labels.csv"

    # ---- 2. Load data using the chosen selection method ----
    X1, X2, X3, y, _ = load_omics_data(
        mrna_path, meth_path, mirna_path, label_path,
        n_features_mrna=1000,
        n_features_meth=4000,
        n_features_mirna=500,
        batch_correct=False,
        batch_labels=None,
        feature_orders=None,
        selection_method=selection_method   # <-- key parameter
    )

    # ---- 3. Output directory for this method ----
    out_dir = os.path.join(output_base, f"main_cv_{selection_method}")
    os.makedirs(out_dir, exist_ok=True)

    loss_weights = (1.0, 0.01, 0.01, 0.001)

    # ---- 4. Run cross‑validation ----
    results, fused_reps, train_histories, summary, all_test_probs, all_test_labels = run_dcmf_cv(
        X1, X2, X3, y,
        n_folds=n_folds,
        n_trials=n_trials,
        epochs=epochs,
        loss_weights=loss_weights,
        encoder_type=encoder_type,
        graph_metric=graph_metric,
        use_smote=use_smote,
        record_train_val=True,
        save_dir=out_dir
    )

    # ---- 5. Save results ----
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "dcmfanet_cv_results.csv"), index=False)
    with pd.ExcelWriter(os.path.join(out_dir, "dcmfanet_cv_results.xlsx")) as writer:
        df_results.to_excel(writer, sheet_name="DCMFANet", index=False)
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

    # ---- 6. ROC curves ----
    plot_roc_curves_cv(all_test_probs, all_test_labels, out_dir)

    # ---- 7. Per‑fold performance bar plot ----
    metrics_to_plot = ['accuracy', 'f1', 'auc']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        ax.bar(df_results['fold'], df_results[metric],
               yerr=df_results[metric].std() / 2,
               capsize=5, alpha=0.7, color='skyblue')
        ax.axhline(df_results[metric].mean(), color='red', linestyle='--', label='Mean')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{selection_method.upper()} – {metric.upper()} per fold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dcmfanet_per_fold.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 8. Training curves (last fold) ----
    if train_histories:
        hist = train_histories[-1]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(hist['train_loss'], color='tab:red', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(hist['train_acc'], color='tab:blue', linestyle='--', label='Train Acc')
        # Optionally plot validation accuracy if available
        if 'val_acc' in hist:
            ax2.plot(hist['val_acc'], color='tab:green', linestyle='-.', label='Val Acc')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title(f'Training Curves – {selection_method.upper()} (Fold {hist["fold"]})')
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.jpg"), dpi=300)
        plt.close()

    print(f"✅ Results for {selection_method} saved to {out_dir}")
    return df_results, summary


def main():
    # ---- Configuration ----
    data_dir = "D:/XiangYa2yyPaper/PAADWork/Article/dataset/datanew/melanoma/"
    output_base = data_dir + "results_revised/"
    os.makedirs(output_base, exist_ok=True)

    # ---- The three unsupervised selection methods to test ----
    #selection_methods = ['variance', 'std', 'sort_corr']
    selection_methods = ['sort_corr']

    # ---- Common CV parameters (same as in your original script) ----
    n_folds = 5
    n_trials = 10
    epochs = 200
    encoder_type = 'mlp'
    graph_metric = 'pearson'
    use_smote = True

    # ---- Collect summaries for comparison ----
    all_summaries = []

    # ---- Loop over methods ----
    for method in selection_methods:
        df_res, summary = run_single_experiment(
            selection_method=method,
            data_dir=data_dir,
            output_base=output_base,
            n_folds=n_folds,
            n_trials=n_trials,
            epochs=epochs,
            encoder_type=encoder_type,
            graph_metric=graph_metric,
            use_smote=use_smote
        )
        # Store summary (mean, std, CI) for later comparison
        summary_df = pd.DataFrame(summary).T
        summary_df['method'] = method
        all_summaries.append(summary_df)

    # ---- Produce a comparison table across methods ----
    if all_summaries:
        comparison_df = pd.concat(all_summaries)
        comparison_df.to_csv(os.path.join(output_base, "comparison_across_methods.csv"))

        print("\n" + "=" * 70)
        print("  COMPARISON ACROSS UNSUPERVISED SELECTION METHODS")
        print("=" * 70)
        for metric in ['accuracy', 'f1', 'auc']:
            print(f"\n{metric.upper()}:")
            for method in selection_methods:
                row = comparison_df[comparison_df['method'] == method]
                if not row.empty:
                    mean_val = row[metric].values[0]
                    std_val = row[metric + '_std'].values[0] if (metric + '_std') in row.columns else np.nan
                    print(f"  {method:10s} : {mean_val:.4f} ± {std_val:.4f}")
        print("=" * 70)

    print("\n✅ All experiments completed. Results saved under:", output_base)


if __name__ == "__main__":
    main()"""






