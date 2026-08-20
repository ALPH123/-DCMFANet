import os
from dcmfanet import load_omics_data, run_hyperparameter_sensitivity

def main():
    data_dir = "Article/dataset/datanew/PAAD/"
    X1, X2, X3, y, _ = load_omics_data(
        data_dir + "exptransp.csv",
        data_dir + "methytransp.csv",
        data_dir + "mirnatransp.csv",
        data_dir + "labels.csv",
        n_features_mrna=500, n_features_meth=500, n_features_mirna=200
    )
    out_dir = data_dir + "results_final/hyperparameter_sensitivity"
    os.makedirs(out_dir, exist_ok=True)
    run_hyperparameter_sensitivity(X1, X2, X3, y, out_dir)

if __name__ == "__main__":
    main()




