import os
from dcmfanet import load_omics_data, run_encoder_comparison

def main():
    data_dir = "D:/XiangYa2yyPaper/PAADWork/Article/dataset/datanew/liver/"
    X1, X2, X3, y, _ = load_omics_data(
        data_dir + "exptransp.csv",
        data_dir + "methytransp.csv",
        data_dir + "mirnatransp.csv",
        data_dir + "labels.csv",
        n_features_mrna=500, n_features_meth=500, n_features_mirna=200
    )
    out_dir = data_dir + "results_final/encoder_comparison"
    os.makedirs(out_dir, exist_ok=True)
    run_encoder_comparison(X1, X2, X3, y, out_dir, n_folds=5, epochs=200)

if __name__ == "__main__":
    main()



