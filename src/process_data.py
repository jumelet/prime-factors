import pandas as pd
import numpy as np
import glob
import pickle
import os


def process_df(csv_dir: str):
    conditions = [
        "CORE",
        "SEMSIM_Vonly",
        "SEMSIM_Nall",
        "SEMSIM_all",
        "LEXBOOST_Vonly",
        "LEXBOOST_allN",
        "LEXBOOST_nRand",
        "LEXBOOSTFULL",
        "LEXBOOST_allfunc",
        "ANOMALOUS",
    ]

    mega_df = None

    for condition in conditions:
        df_dict[condition] = {}

        for score_file in glob.glob(os.path.join(csv_dir, condition)):
            model_name = score_file.split("/")[1]
            print(model_name)
            df = read_new_scores(
                score_file, verbose=False, model_name=model_name, condition=condition
            )
            add_semsim_scores(df, condition)

            if mega_df is None:
                mega_df = df
            else:
                mega_df = mega_df.append(df, ignore_index=True)


def read_new_scores(scores_file, verbose=True, model_name=None, condition=None):
    # X: po, y: DO!
    CONVERTERS = dict.fromkeys(
        [
            "all_logp_prime_x",
            "all_logp_prime_y",
            "all_logp_x",
            "all_logp_y",
            "all_logp_x_px",
            "all_logp_x_py",
            "all_logp_y_px",
            "all_logp_y_py",
        ],
        eval,
    )

    df = pd.read_csv(scores_file, sep="\t", converters=CONVERTERS, verbose=verbose)

    df["pe_x"] = df.logp_x_px - df.logp_x_py
    df["pe_y"] = df.logp_y_py - df.logp_y_px

    df["logp_x_px_from4"] = np.array([sum(item[4:]) for item in df.all_logp_x_px])
    df["logp_x_py_from4"] = np.array([sum(item[4:]) for item in df.all_logp_x_py])
    df["logp_y_py_from4"] = np.array([sum(item[4:]) for item in df.all_logp_y_py])
    df["logp_y_px_from4"] = np.array([sum(item[4:]) for item in df.all_logp_y_px])

    df["pe_x_from4"] = df.logp_x_px_from4 - df.logp_x_py_from4
    df["pe_y_from4"] = df.logp_y_py_from4 - df.logp_y_px_from4

    po_template = "DT1 NN1 VB DT2 NN2 PP DT3 NN3 DOT".split()
    do_template = "DT1 NN1 VB DT2 NN2 DT3 NN3 DOT".split()

    po_len = len(po_template)
    do_len = len(do_template)

    # For the token-level scores we set them to NaN if they don't align with the template
    # (due to subword tokenization differences)
    all_logp_x_px = [
        logp_x_px if len(logp_x_px) == po_len else [np.nan] * po_len
        for logp_x_px in df.all_logp_x_px
    ]

    all_logp_x_py = [
        logp_x_py if len(logp_x_py) == po_len else [np.nan] * po_len
        for logp_x_py in df.all_logp_x_py
    ]

    all_logp_y_py = [
        logp_y_py if len(logp_y_py) == do_len else [np.nan] * do_len
        for logp_y_py in df.all_logp_y_py
    ]

    all_logp_y_px = [
        logp_y_px if len(logp_y_px) == do_len else [np.nan] * do_len
        for logp_y_px in df.all_logp_y_px
    ]

    po_token_pe = np.stack(all_logp_x_px) - np.stack(all_logp_x_py)
    do_token_pe = np.stack(all_logp_y_py) - np.stack(all_logp_y_px)

    for token_idx, token in enumerate(po_template):
        df[f"po_{token}_pe"] = po_token_pe[:, token_idx]
    for token_idx, token in enumerate(do_template):
        df[f"do_{token}_pe"] = do_token_pe[:, token_idx]

    token_ids = [
        ("det1", 1),
        ("n1", 2),
        ("verb", 3),
        ("det2", 4),
        ("n2", 5),
        ("prep", 6),
        ("det3", 7),
        ("n3", 8),
    ]

    index_offset = (
        -1
        if (
            "falcon" in model_name
            or ("gpt" in model_name and ("prep" in condition or "det" in condition))
        )
        else 0
    )

    for token, idx in token_ids:
        idx += index_offset
        df[f"prime_{token}"] = [prime.split()[idx] for prime in df.prime_x]
        df[f"target_{token}"] = [target.split()[idx] for target in df.x]

    if model_name is not None:
        df["model_name"] = [model_name] * len(df)
    if condition is not None:
        df["condition"] = [condition] * len(df)

    return df


def add_semsim_scores(df, condition: str):
    """Add sentence+token similarity columns that have been precomputed"""
    with open(f"sen_sim_pickles/sen_sim_po_{condition}.pickle", "rb") as f:
        sen_sim_x = pickle.load(f)

        df["sen_sim_x"] = np.array(sen_sim_x)

    with open(f"sen_sim_pickles/sen_sim_do_{condition}.pickle", "rb") as f:
        sen_sim_y = pickle.load(f)

        df["sen_sim_y"] = np.array(sen_sim_y)

    for sim_name in ["verb_sims", "n1_sims", "n2_sims", "n3_sims"]:
        filename = f"token_sims/{condition}_{sim_name}.pickle"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                token_sims = pickle.load(f)

            df[sim_name] = token_sims
        elif condition == "LEXBOOSTFULL":
            df[sim_name] = np.ones(15_000)
        else:
            print(filename, "semsim scores skipped")
