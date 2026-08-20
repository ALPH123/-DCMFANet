import os
from dcmfanet import load_omics_data, run_graph_comparison

def main():
    #data_dir = "Article/dataset/datanew/Liver/"
    #X1, X2, X3, y, _ = load_omics_data(
     #   data_dir + "exptransp.csv",
    #    data_dir + "methytransp.csv",
    #    data_dir + "mirnatransp.csv",
    #    data_dir + "labels.csv",
     #   n_features_mrna=200, n_features_meth=500, n_features_mirna=200
    #)
    data_dir = "D:Article/dataset/datanew/melanoma/"
    mrna_path = data_dir + "exptransp.csv"
    meth_path = data_dir + "methytransp.csv"
    mirna_path = data_dir + "mirnatransp.csv"
    label_path = data_dir + "labels.csv"

    X1, X2, X3, y, processors = load_omics_data(
        mrna_path, meth_path, mirna_path, label_path,
        n_features_mrna=2000,
        n_features_meth=2000,
        n_features_mirna=500,
        batch_correct=False,
        batch_labels=None,
        feature_orders=None,
        use_mi=False  #
    )

    out_dir = data_dir + "results_final/graph_comparison"
    os.makedirs(out_dir, exist_ok=True)
    run_graph_comparison(X1, X2, X3, y, out_dir, n_folds=5, epochs=1000)

if __name__ == "__main__":
    main()





