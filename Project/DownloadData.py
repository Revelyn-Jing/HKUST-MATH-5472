import os
import numpy as np
import pandas as pd
import deepchem as dc


def download_lipophilicity(save_dir="data"):
    """
    使用 DeepChem 的 MoleculeNet loader 下载 Lipophilicity 数据集，
    并将 (smiles, logD) 保存到 data/lipophilicity_processed.csv
    """

    os.makedirs(save_dir, exist_ok=True)

    print("Downloading Lipophilicity dataset using DeepChem...")

    # 用 Raw featurizer，这样 X 里只是占位，我们只用 ids 和 y
    tasks, datasets, transformers = dc.molnet.load_lipo(featurizer="Raw")
    train, valid, test = datasets

    # 拼接三个子集的 ids 和 y
    ids = list(train.ids) + list(valid.ids) + list(test.ids)
    y = np.concatenate([train.y, valid.y, test.y]).ravel()

    df = pd.DataFrame({"smiles": ids, "logD": y})

    save_path = os.path.join(save_dir, "lipophilicity_processed.csv")
    df.to_csv(save_path, index=False)

    print(f"Processed dataset saved to: {save_path}")
    print("\nPreview:")
    print(df.head())

    return df


if __name__ == "__main__":
    download_lipophilicity()
