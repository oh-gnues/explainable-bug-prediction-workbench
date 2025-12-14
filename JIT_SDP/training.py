# %%
import pandas as pd
import numpy as np
import ast
import warnings
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from scipy.stats import ranksums
from cliffs_delta import cliffs_delta
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = Path("test_output_dt")
OUTPUT_DIR.mkdir(exist_ok=True)
TRAIN = True
ACTIONABLE_ANALYSIS = False
K = 5
N = 100
SEED = 42
np.random.seed(SEED)
warnings.simplefilter(action="ignore", category=UserWarning)

kamei_metrics = ["NS", "NF", "LA/LT", "LD/LT", "AGE", "NUC/NF", "EXP", "SEXP"]

cog_metrics = [
    "V",
    "EM/V",
    "DD/V",
    "MDNL",
    "NB",
    "REMC",
    "ENMC",
    "NP",
    "RG",
    "NTM",
    "RTS",
    "RII",
]

actionable_metrics = cog_metrics + ["NS", "NF", "LA/LT", "LD/LT"]


PROJECTS = {
    "zeppelin": ['V', 'MDNL'],
    "zookeeper": ['EM/V', 'DD/V', 'RG'],
    "activemq": ['V', 'REMC', 'NP', 'RG', 'NTM'],
    "camel": ['V', 'MDNL', 'REMC', 'NP', 'NTM'],
    "cassandra": ['V', 'MDNL', 'NB', 'RG', 'NTM',],
    "flink":['V',  'REMC', 'RG', 'NTM', 'RTS', 'RII'],
    "groovy":['V', 'EM/V', 'MDNL',   'NP',  'NTM'],
    "hadoop":['V', 'NTM', 'RTS'],
    "hbase":['V',  'REMC', 'NP', 'NTM', 'RII'],
    "hive":[ 'MDNL', 'NP'],
    "ignite":['V', 'MDNL', 'NB', 'RG', 'NTM', 'RTS'],
    "kafka":['V', 'MDNL',  'REMC',  'NP',  'NTM'],
    "spark":[ 'NTM', 'RII']
}

console = Console()
Path("output").mkdir(exist_ok=True)


def preprocess(df):
    df = df.dropna()
    after_df = df.drop_duplicates()
    return after_df


def pipeline(df, save_name, k=5, n=100, save=True):
    X = df.drop(columns=["buggy"])
    y = df["buggy"]

    model = LogisticRegression(random_state=SEED, max_iter=1000)
    # model = RandomForestClassifier(random_state=SEED)
    # model = XGBClassifier(random_state=SEED)
    # model = svm.SVC(random_state=SEED)
    # model = DecisionTreeClassifier(random_state=SEED)

    pipe = ImbPipeline(steps=[['smote', SMOTE(random_state=SEED)], ['scaler', Normalizer()], ['classifier', model]])
    rstratified_kfold = RepeatedStratifiedKFold(n_splits=k, n_repeats=n, random_state=SEED)
    cv_results = cross_validate(
        pipe,
        X,
        y,
        cv=rstratified_kfold,
        scoring=["f1_macro", "roc_auc", "average_precision", "matthews_corrcoef"],
        verbose=1,
        n_jobs=-1,
        return_estimator=True,
        return_indices=True,
    )
    if save:
        (OUTPUT_DIR / "cache").mkdir(exist_ok=True)
        with open(OUTPUT_DIR / f"cache/{save_name}.pkl", "wb") as f:
            
            pickle.dump(cv_results, f)

    estimators = cv_results["estimator"]
    train_indices = cv_results["indices"]["train"]
    test_indices = cv_results["indices"]["test"]

    # FIND TRUE POSITIVE samples
    tp_samples = []
    for i in range(len(estimators)):
        labels = y.iloc[test_indices[i]]
        preds = estimators[i].predict(X.iloc[test_indices[i]])
        tp_indices = test_indices[i][labels & preds]
        tp_samples.append(list(X.iloc[tp_indices].index))
        
    # Save results to csv
    f1_macro = cv_results["test_f1_macro"]
    roc_auc = cv_results["test_roc_auc"]
    auc_pr = cv_results["test_average_precision"]
    mcc = cv_results["test_matthews_corrcoef"]

    df = pd.DataFrame(
        {"f1_macro": f1_macro, "roc_auc": roc_auc, "auc_pr": auc_pr, "mcc": mcc, "tp_samples": tp_samples}
    )
    df.to_csv(OUTPUT_DIR / f"{save_name}.csv")




kamei = {}
hadoops = []
for file in Path("./results").glob("*_kamei.csv"):
    repo = file.stem.split("_")[0]
    df = pd.read_csv(file, index_col="commit_id")
    if "hadoop" in repo:
        hadoops.append(df)
        continue
    kamei[repo] = df
hadoop = pd.concat(hadoops)
kamei["hadoop"] = hadoop

cog = {}
hadoops = []
for file in Path("./results").glob("*_metrics.csv"):
    repo = file.stem.split("_")[0]
    df = pd.read_csv(file, index_col="commit_id")
    rii_df = pd.read_csv(f"results/{repo}_metrics_rii.csv", index_col="commit_id")
    df["RII"] = rii_df["RII"]
    if "hadoop" in repo:
        hadoops.append(df)
        continue
    cog[repo] = df

hadoop = pd.concat(hadoops)
cog["hadoop"] = hadoop

# console.log("Preprocessing")

kamei_dfs = {}
cog_dfs = {}
kameicogs = {}
for project in PROJECTS:
    cog_df = cog[project][PROJECTS[project] + ["buggy"]]
    kamei_df = kamei[project][kamei_metrics + ["buggy"]]

    kmaei_df = preprocess(kamei_df)
    cog_df = preprocess(cog_df)

    indices = list(set(kamei_df.index) & set(cog_df.index))
    # console.print(f"{project}: {len(kamei_df)} -> {len(indices)}")

    kamei_dfs[project] = kamei_df.loc[indices]
    cog_dfs[project] = cog_df.loc[indices]
    kameicogs[project] = pd.concat([kamei_df.loc[indices], cog_df.loc[indices, PROJECTS[project]]], axis=1)
    
    # console.print(kamei_dfs[project].columns)
    # console.print(cog_dfs[project].columns)
    # console.print(kameicogs[project].columns)

if TRAIN:
    console.log("Training")
    for project in PROJECTS:
        console.log(f"[bold]{project}[/bold]")
        console.log(f"[bold]Kamei[/bold]")
        pipeline(kamei_dfs[project], f"{project}_kamei",k=K, n=N)
        # console.log(f"[bold]COG[/bold]")
        # pipeline(cog_dfs[project],  f"{project}_cog",k=K, n=N)
        console.log(f"[bold]KameiCog[/bold]")
        pipeline(kameicogs[project], f"{project}_kameicog",k=K, n=N)

if not ACTIONABLE_ANALYSIS:
    table1 = Table(title="F1-macro and ROC-AUC")
    table1.add_column("Project")
    table1.add_column("Kamei F1")
    table1.add_column("KameiCog F1")
    table1.add_column("p-value")
    table1.add_column("Result")
    table1.add_column("Kamei AUC")
    table1.add_column("KameiCog AUC")
    table1.add_column("p-value")
    table1.add_column("Result")
    table2 = Table(title="AUC-PR and MCC")
    table2.add_column("Kamei AUC-PR")
    table2.add_column("KameiCog AUC-PR")
    table2.add_column("p-value")
    table2.add_column("Result")
    table2.add_column("Kamei MCC")
    table2.add_column("KameiCog MCC")
    table2.add_column("p-value")
    table2.add_column("Result")

    # table2 = Table(title="Table 5: Intersection and Difference of Defect Inducing Commits (TP) Predicted by Kamei and Cog models")
    # table2.add_column("Project")
    # table2.add_column("Only Kamei (%)")
    # table2.add_column("Both (%)")
    # table2.add_column("Only Cog (%)")

    ktps = []
    ns = []
    ctps = []
    kacts = []
    cacts = []
    for project in PROJECTS:
        df_kamei = pd.read_csv(OUTPUT_DIR / f"{project}_kamei.csv", index_col=0)
        # df_cog = pd.read_csv(OUTPUT_DIR / f"{project}_cog.csv", index_col=0)
        df_kameicog = pd.read_csv(OUTPUT_DIR / f"{project}_kameicog.csv", index_col=0)
        df_kamei["tp_samples"] = df_kamei["tp_samples"].apply(ast.literal_eval)
        # df_cog["tp_samples"] = df_cog["tp_samples"].apply(ast.literal_eval)
        df_kameicog["tp_samples"] = df_kameicog["tp_samples"].apply(ast.literal_eval)

        kamei_f1_macro = []
        kamei_roc_auc = []
        kamei_auc_pr = []
        kamei_mcc = []

        kameicog_f1_macro = []
        kameicog_roc_auc = []
        kameicog_auc_pr = []
        kameicog_mcc = []


        for i in range(0, K*N, K):
            kamei_f1_macro.append(df_kamei["f1_macro"].iloc[i : i + K].mean())
            kamei_roc_auc.append(df_kamei["roc_auc"].iloc[i : i + K].mean())
            kamei_auc_pr.append(df_kamei["auc_pr"].iloc[i : i + K].mean())
            kamei_mcc.append(df_kamei["mcc"].iloc[i : i + K].mean())

            kameicog_f1_macro.append(df_kameicog["f1_macro"].iloc[i : i + K].mean())
            kameicog_roc_auc.append(df_kameicog["roc_auc"].iloc[i : i + K].mean())
            kameicog_auc_pr.append(df_kameicog["auc_pr"].iloc[i : i + K].mean())
            kameicog_mcc.append(df_kameicog["mcc"].iloc[i : i + K].mean())

        # Wilcoxon for F1-macro
        statistic_f1, pvalue_f1 = ranksums(kamei_f1_macro, kameicog_f1_macro)
        delta_f1, res_f1 = cliffs_delta(kamei_f1_macro, kameicog_f1_macro)

        # Wilcoxon for ROC-AUC
        statistic_roc, pvalue_roc = ranksums(kamei_roc_auc, kameicog_roc_auc)
        delta_roc, res_roc = cliffs_delta(kamei_roc_auc, kameicog_roc_auc)

        # Wilcoxon for AUC-PR
        statistic_pr, pvalue_pr = ranksums(kamei_auc_pr, kameicog_auc_pr)
        delta_pr, res_pr = cliffs_delta(kamei_auc_pr, kameicog_auc_pr)

        # # Wilcoxon for MCC
        statistic_mcc, pvalue_mcc = ranksums(kamei_mcc, kameicog_mcc)
        delta_mcc, res_mcc = cliffs_delta(kamei_mcc, kameicog_mcc)

        kf1 = f"{np.mean(kamei_f1_macro):.3}"
        kroc = f"{np.mean(kamei_roc_auc):.3}"

        kcf1 = f"{np.mean(kameicog_f1_macro):.3}"
        kcroc = f"{np.mean(kameicog_roc_auc):.3}"

        kpr = f"{np.mean(kamei_auc_pr):.3}"
        kcpr = f"{np.mean(kameicog_auc_pr):.3}"

        kmcc = f"{np.mean(kamei_mcc):.3}"
        kcmcc = f"{np.mean(kameicog_mcc):.3}"


        if np.mean(kamei_f1_macro) > np.mean(kameicog_f1_macro):
            kf1 = f"[green]{kf1}[/green]"
            kcf1 = f"[red]{kcf1}[/red]"
        else:
            kf1 = f"[red]{kf1}[/red]"
            kcf1 = f"[green]{kcf1}[/green]"

        if np.mean(kamei_roc_auc) > np.mean(kameicog_roc_auc):
            kroc = f"[green]{kroc}[/green]"
            kcroc = f"[red]{kcroc}[/red]"
        else:
            kroc = f"[red]{kroc}[/red]"
            kcroc = f"[green]{kcroc}[/green]"

        if np.mean(kamei_auc_pr) > np.mean(kameicog_auc_pr):
            kpr = f"[green]{kpr}[/green]"
            kcpr = f"[red]{kcpr}[/red]"
        else:
            kpr = f"[red]{kpr}[/red]"
            kcpr = f"[green]{kcpr}[/green]"

        if np.mean(kamei_mcc) > np.mean(kameicog_mcc):
            kmcc = f"[green]{kmcc}[/green]"
            kcmcc = f"[red]{kcmcc}[/red]"
        else:
            kmcc = f"[red]{kmcc}[/red]"
            kcmcc = f"[green]{kcmcc}[/green]"


        # TP samples
        # only_kamei = []
        # both = []
        # only_cog = []
        
        # kamei_all_tp_samples = set()
        # cog_all_tp_samples = set()

        # for i in range(len(df_kamei)):
        #     kamei_samples = set(df_kamei.iloc[i]["tp_samples"])
        #     # cog_samples = set(df_cog.iloc[i]["tp_samples"])

        #     kamei_all_tp_samples |= kamei_samples
        #     cog_all_tp_samples |= cog_samples


        #     if len(kamei_samples) == 0 and len(cog_samples) == 0:
        #         continue

        #     both.append(len(kamei_samples & cog_samples) / len(
        #         kamei_samples | cog_samples
        #     ))
        #     only_kamei.append(len(kamei_samples - cog_samples) / len(kamei_samples | cog_samples))
        #     only_cog.append(len(cog_samples - kamei_samples) / len(cog_samples | kamei_samples))

        # ktps.append(np.mean(only_kamei))
        # ns.append(np.mean(both))
        # ctps.append(np.mean(only_cog))
        
        table1.add_row(
            project,
            f"{kf1}",
            f"{kcf1}",
            f"{pvalue_f1:.3f}" if pvalue_f1 < 0.05 else f"[yellow]{pvalue_f1:.3}[/yellow]",
            res_f1,
            f"{kroc}",
            f"{kcroc}",
            
            f"{pvalue_roc:.3f}" if pvalue_roc < 0.05 else f"[yellow]{pvalue_roc:.3}[/yellow]",
            res_roc,
            # f"{kpr}",
            # f"{kcpr}",
            # f"{pvalue_pr:.3f}" if pvalue_pr < 0.05 else f"[yellow]{pvalue_pr:.3f}[/yellow]",
            # res_pr,
            # f"{kmcc}",
            # f"{kcmcc}",
            # f"{pvalue_mcc:.3f}" if pvalue_mcc < 0.05 else f"[yellow]{pvalue_mcc:.3f}[/yellow]",
            # res_mcc,
        )
        table2.add_row(
            f"{kpr}",
            f"{kcpr}",
            f"{pvalue_pr:.3f}" if pvalue_pr < 0.05 else f"[yellow]{pvalue_pr:.3f}[/yellow]",
            res_pr,
            f"{kmcc}",
            f"{kcmcc}",
            f"{pvalue_mcc:.3f}" if pvalue_mcc < 0.05 else f"[yellow]{pvalue_mcc:.3f}[/yellow]",
            res_mcc,
        )
        # table2.add_row(
        #     project,
        #     str(round(np.mean(only_kamei)*100, 2)),
        #     str(round(np.mean(both)*100, 2)),
        #     str(round(np.mean(only_cog)*100, 2)),
        # )
    console.print(table1)
    console.print(table2)

# Local Explanation
if ACTIONABLE_ANALYSIS:
    sheet = Table(title="Table 7: Prop. of Actionable Features in Local Explanation")
    sheet.add_column("Project")
    sheet.add_column("Kamei (%)")
    sheet.add_column("KameiCog (%)")
    sheet.add_column("Improv. (%)")
    save_dict = {}
    # Load from cache
    with Progress(disable=False) as progress:
        task1 = progress.add_task("[red]Actionable Analysis...", total=len(PROJECTS))
        task2 = progress.add_task("[green]Kamei vs KameiCog...", total=2)
        task3 = progress.add_task("[cyan]Iterating K-folds...", total=500)

        for project in ["kafka"]:
            progress.console.log(f"[bold orange4]START {project}[/bold orange4]")
            df_kamei = pd.read_csv(OUTPUT_DIR / f"{project}_kamei.csv", index_col=0)
            df_kameicog = pd.read_csv(OUTPUT_DIR / f"{project}_kameicog.csv", index_col=0)
            df_kamei["tp_samples"] = df_kamei["tp_samples"].apply(ast.literal_eval)
            df_kameicog["tp_samples"] = df_kameicog["tp_samples"].apply(ast.literal_eval)
            
            save_dict[project] = []
            test_info = {
                "kamei": {
                    "df": kamei_dfs[project],
                    "name": f"{project}_kamei",
                },

                "kameicog": {
                    "df": kameicogs[project],
                    "name": f"{project}_kameicog",
                },
            }
            
            for model in test_info:
                df = test_info[model]["df"]
                name = test_info[model]["name"]
                X = df.drop(columns=["buggy"])
                y = df["buggy"]
    
                with open(OUTPUT_DIR / f"cache/{name}.pkl", "rb") as f:
                    cv_results = pickle.load(f)
                estimators = cv_results["estimator"]
                train_indices = cv_results["indices"]["train"]
                test_indices = cv_results["indices"]["test"]

                tp_samples = []
                ratio_actionables = []
                for i in range(500):
                    # Calculate ratio of actionable features
                    labels = y.iloc[test_indices[i]]
                    preds = estimators[i].predict(X.iloc[test_indices[i]])
                    tp_indices = test_indices[i][labels & preds]
                    if len(tp_indices) == 0:
                        progress.update(task3, advance=1)
                        continue
                    
                    tp_samples.append(list(X.iloc[tp_indices].index))

                    for tp_index in tp_indices:
                        explainer = LimeTabularExplainer(
                            X.iloc[train_indices[i]],
                            feature_names=X.columns,
                            class_names=["not buggy", "buggy"],
                            discretize_continuous=False,
                        )
                        exp = explainer.explain_instance(
                            X.iloc[tp_index],
                            estimators[i].predict_proba,
                            num_features=X.shape[1],
                            num_samples=1000,
                        )
                        
                        top_features = exp.as_map()[1]
                        top_features_index = [x[0] for x in top_features][:5]
                        top_5_features = [X.columns[i] for i in top_features_index]
                        
                        ra = len(set(top_5_features) & set(actionable_metrics)) / len(top_5_features)
                        ratio_actionables.append(ra)
                    progress.update(task3, advance=1)
                save_dict[project].append(np.mean(ratio_actionables))
                progress.console.log(f"[bold orange3]Actionable Features {save_dict[project][-1]*100}[/bold orange3]")
                progress.update(task3, completed=0)
                progress.update(task2, advance=1)
            save_dict[project].append(save_dict[project][1] - save_dict[project][0])
            sheet.add_row(
                project,
                f"{save_dict[project][0]*100:.3}",
                f"{save_dict[project][1]*100:.3}",
                f"{save_dict[project][2]*100:.3}",
            )
            progress.update(task1, advance=1)
            progress.update(task2, completed=0)

        progress.console.print(sheet)
        # Save to csv
        df = pd.DataFrame(save_dict, index=["kamei", "kameicog", "improv"])
        df.to_csv(OUTPUT_DIR / "actionables.csv")

                            


    
# %%
