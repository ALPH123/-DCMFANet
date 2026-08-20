import os
from dcmfanet import load_omics_data, run_graph_perturbation

def main():
    data_dir = "D:/Article/dataset/datanew/PAAD/"
    X1, X2, X3, y, _ = load_omics_data(
        data_dir + "exptransp.csv",
        data_dir + "methytransp.csv",
        data_dir + "mirnatransp.csv",
        data_dir + "labels.csv",
        n_features_mrna=2000, n_features_meth=2000, n_features_mirna=500
    )
    out_dir = data_dir + "results_final/graph_perturbation"
    os.makedirs(out_dir, exist_ok=True)
    run_graph_perturbation(X1, X2, X3, y, out_dir, n_folds=5, epochs=200)

if __name__ == "__main__":
    main()