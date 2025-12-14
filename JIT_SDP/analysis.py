#%%
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ranksums, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from cliffs_delta import cliffs_delta
from rich.console import Console
from rich.table import Table
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from pypair.association import binary_continuous

PROJECTS = [
    "zeppelin",
    "zookeeper",
    "activemq",
    "camel",
    "cassandra",
    "flink",
    "groovy",
    "hadoop",
    "hadoop-hdfs",
    "hadoop-mapreduce",
    "hbase",
    "hive",
    "ignite",
    "kafka",
    "spark",
]

#%%
def logistic_regression(df, metrics):
    X = df[metrics]
    y = df["buggy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("회귀 계수:\n", model.coef_)
    print("절편:", model.intercept_)
    print("\n분류 보고서:\n", classification_report(y_test, y_pred))
    print("혼동 행렬:\n", confusion_matrix(y_test, y_pred))
    
    

def visualize_correlation(df, metrics):
    console = Console()
    no_defects = df[df["buggy"] == False]
    defects = df[df["buggy"] == True]
    table = Table(title="Correlation between metrics and defects")
    table.add_column("Metric")
    
    table.add_column("p-value")
    table.add_column("Effect size")
    table.add_column("Cliff's delta")

    for metric in metrics:
        statistic, pvalue = ranksums(no_defects[metric], defects[metric])
        delta, res = cliffs_delta(no_defects[metric], defects[metric])
        pv_str = f"{pvalue:.3}"
        if pvalue < 0.05:
            pv_str = f"[green]{pv_str}[/green]"
        else:
            pv_str = f"[red]{pv_str}[/red]"

        res_str = res
        if res_str.startswith("neg"):
            res_str = f"[red]{res_str}[/red]"
            metric_str = f"[red]{metric}[/red]"
        elif res_str.startswith("sma"):
            res_str = f"[yellow]{res_str}[/yellow]"
            metric_str = f"[yellow]{metric}[/yellow]"
        else:
            res_str = f"[green]{res_str}[/green]"
            metric_str = f"[cyan]{metric}[/cyan]"

        table.add_row(metric_str, pv_str, f"{delta:.3}", res_str)

    console.print(table)

def visualize_correlation_point_biserial(df, metrics):
    console = Console()
    table = Table(title="Point Biserial Correlation between metrics and defects")
    
    table.add_column("Metric")
    table.add_column("Corr. Coef.")
    table.add_column("p-value")

    for metric in metrics:
        corr, pvalue = pointbiserialr(df['buggy'], df[metric])
        corr_str = f"{corr:.3f}"
        pv_str = f"{pvalue:.3f}"

        if pvalue < 0.05:
            corr_str = f"[green]{corr_str}[/green]"
            pv_str = f"[green]{pv_str}[/green]"
        else:
            corr_str = f"[red]{corr_str}[/red]"
            pv_str = f"[red]{pv_str}[/red]"

        table.add_row(metric, corr_str, pv_str)

    console.print(table)

def visualize_correlation_logistic_regression(df, metrics):
    console = Console()
    table = Table(title="Logistic Regression Coefficients and Log Loss with Scaling")

    table.add_column("Metric")
    table.add_column("Coefficient")
    table.add_column("Log Loss")

    X = df[metrics]
    y = df['buggy']

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)
    coefficients = model.coef_[0]
    y_pred = model.predict_proba(X_scaled)[:, 1]
    logloss = log_loss(y, y_pred)

    for metric, coefficient in zip(metrics, coefficients):
        coeff_str = f"{coefficient:.3}"
        table.add_row(metric, coeff_str, f"{logloss:.3}")

    console.print(table)

def spearman(df, metrics, save_path=None):
    spearman_corr = {}
    for metric1 in metrics:
        spearman_corr[metric1] = []
        for metric2 in metrics:
            corr, pvalue = spearmanr(df[metric1], df[metric2])
            spearman_corr[metric1].append(corr)

    spearman_corr = pd.DataFrame(spearman_corr, index=metrics)
    if save_path:
        spearman_corr.to_csv(save_path)
    return spearman_corr

def hmap(spearman_corr, title, save_path=None):
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
    plt.figure(figsize=(10, 9))
    ax = sns.heatmap(spearman_corr, annot=False, cmap="coolwarm", mask=mask)
    plt.yticks(rotation=0, fontsize=20)
    plt.xticks(rotation=90, fontsize=20)
    for i in range(spearman_corr.shape[0]):
        for j in range(spearman_corr.shape[1]):
            if i != j:
                ax.text(j+0.5, i+0.5, f"{spearman_corr.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=14)
    plt.title(title, pad=10, fontdict={"fontweight": "bold", "fontfamily": "serif", "fontsize": 25})
    if save_path:
        plt.savefig(save_path, format='svg')
# %%

title = "Kamei. before preprocessing (Response)"

kamei_metrics = [ 'NS', 'ND','NF','Entropy', 
                 'LA', 'LD','LT',
                 'NDEV', 'AGE', 'NUC',
                   'EXP', 'REXP', 'SEXP']
kamei = {}
for file in Path("./results").glob("*_kamei_origin.csv"):
    repo = file.stem.split("_")[0]
    df = pd.read_csv(file, index_col="commit_id")
    kamei[repo] = df
hadoop = kamei["hadoop"]
hadoop_hdfs = kamei["hadoop-hdfs"]
hadoop_mapreduce = kamei["hadoop-mapreduce"]
kamei["hadoop"] = pd.concat([hadoop, hadoop_hdfs, hadoop_mapreduce])
del kamei["hadoop-hdfs"]
del kamei["hadoop-mapreduce"]
# kamei = pd.concat(kamei)
# visualize_correlation(kamei, kamei_metrics)
# visualize_correlation_point_biserial(kamei, kamei_metrics)
# visualize_correlation_logistic_regression(kamei, kamei_metrics)

# spear = spearman(kamei, kamei_metrics)
# hmap(spear, title, f"plots/{title}.svg")
# %%
title = "Kamei. after preprocessing (Response)"
kamei_metrics = ['NS', 'NF', 'LA/LT', 'LD/LT', 'AGE', 'NUC/NF', 'EXP', 'SEXP']
kamei = {}
for file in Path("./results").glob("*_kamei.csv"):
    repo = file.stem.split("_")[0]
    df = pd.read_csv(file, index_col="commit_id")
    kamei[repo] = df
hadoop = kamei["hadoop"]
hadoop_hdfs = kamei["hadoop-hdfs"]
hadoop_mapreduce = kamei["hadoop-mapreduce"]
kamei["hadoop"] = pd.concat([hadoop, hadoop_hdfs, hadoop_mapreduce])
del kamei["hadoop-hdfs"]
del kamei["hadoop-mapreduce"]
# kamei = pd.concat(kamei)
# visualize_correlation(kamei, kamei_metrics)
# visualize_correlation_point_biserial(kamei, kamei_metrics)
# visualize_correlation_logistic_regression(kamei, kamei_metrics)
# spear = spearman(kamei, kamei_metrics)
# hmap(spear, title, f"plots/{title}.svg")
# %%
title = "Cog. Comp. before preprocessing (Response)"
cog = {}
for file in Path("./results").glob("*_metrics.csv"):
    repo = file.stem.split("_")[0]
    df = pd.read_csv(file, index_col="commit_id")
    rii_df = pd.read_csv(f"results/{repo}_metrics_rii.csv", index_col="commit_id")
    df["RII"] = rii_df["RII"]
    cog[repo] = df
hadoop = cog["hadoop"]
hadoop_hdfs = cog["hadoop-hdfs"]
hadoop_mapreduce = cog["hadoop-mapreduce"]
cog["hadoop"] = pd.concat([hadoop, hadoop_hdfs, hadoop_mapreduce])
del cog["hadoop-hdfs"]
del cog["hadoop-mapreduce"]

cog_df = pd.concat(cog)
cog_metrics = ["V", "EM/V",  "DD/V", "MDNL", "NB", "REMC", "ENMC", "NP", "RG", "ATL", "RTS", "RII"]
before_cog_metrics = ["V", "EM",  "DD", "MDNL", "NB", "REMC", "ENMC", "NP", "RG", "NTM", "RTS", "RII"]
# visualize_correlation(cog_df, before_cog_metrics)
# visualize_correlation_point_biserial(cog_df, before_cog_metrics)
# visualize_correlation_logistic_regression(cog_df, before_cog_metrics)
# spear = spearman(cog_df, before_cog_metrics)
# hmap(spear, title, f"plots/{title}.svg")
# %%
title = "Cog. Comp. after preprocessing (Response)"
cog_metrics = ["V", "EM/V",  "DD/V", "MDNL", "NB", "REMC", "ENMC", "NP", "RG","NTM", "RTS", "RII"]
# visualize_correlation(cog_df, cog_metrics)
# visualize_correlation_point_biserial(cog_df, cog_metrics)
# visualize_correlation_logistic_regression(cog_df, cog_metrics)
# spear = spearman(cog_df, cog_metrics)
# hmap(spear, title, f"plots/{title}.svg")
# %%
# title = "Kamei. & Cog. Comp. (Response)"
# metrics = kamei_metrics + cog_metrics
# df = pd.concat([kamei[kamei_metrics], cog_df[cog_metrics], cog_df["buggy"]], axis=1)
# visualize_correlation(df, metrics)
# spear = spearman(df, metrics)
# hmap(spear, title, f"plots/{title}.svg")

# %%

# merge hadoop-hdfs, hadoop-mapreduce, hadoop


for repo in cog:
    print(repo)
    df = cog[repo]
    metrics = cog_metrics
    visualize_correlation(df, metrics)
    # visualize_correlation_point_biserial(df, metrics)
    visualize_correlation_logistic_regression(df, metrics)

# %%
console = Console()
table = Table(title="Logistic Regression Coefficients")

table.add_column("Project")
for metric in metrics:
    table.add_column(metric)

PROJECTS = [
    "zeppelin",
    "zookeeper",
    "activemq",
    "camel",
    "cassandra",
    "flink",
    "groovy",
    "hadoop",
    "hbase",
    "hive",
    "ignite",
    "kafka",
    "spark",
]

for repo in PROJECTS:
    df = cog[repo]
    X = df[metrics]
    y = df['buggy']


    # 데이터 스케일링
    scaler = Normalizer()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    coefficients = model.coef_[0]
    coeff_str = [f"{abs(coefficient):.3f}" for coefficient in coefficients]
    table.add_row(repo, *coeff_str)

console.print(table)
# %%

table = Table(title="Rank-Biserial Correlation")

table.add_column("Project")
for metric in metrics:
    table.add_column(metric)

for repo in PROJECTS:
    df = cog[repo]
    X = df[metrics]
    y = df['buggy']
    corr = []
    for metric in metrics:
        corr.append(binary_continuous(y, X[metric], measure="rank_biserial", b_0=False, b_1=True))
    corr_str = [f"{abs(c):.3f}" for c in corr]
    table.add_row(repo, *corr_str)

console.print(table)

    
# %%
console = Console()
table = Table(title="Logistic Regression Coefficients")

table.add_column("Project")
metrics = kamei_metrics
for metric in metrics:
    table.add_column(metric)

PROJECTS = [
    "zeppelin",
    "zookeeper",
    "activemq",
    "camel",
    "cassandra",
    "flink",
    "groovy",
    "hadoop",
    "hbase",
    "hive",
    "ignite",
    "kafka",
    "spark",
]

for repo in PROJECTS:
    df = kamei[repo]
    X = df[metrics]
    y = df['buggy']
    console.print(y.value_counts())


    # 데이터 스케일링
    scaler = Normalizer()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    coefficients = model.coef_[0]
    coeff_str = [f"{abs(coefficient):.3f}" for coefficient in coefficients]
    table.add_row(repo, *coeff_str)

console.print(table)
# %%

table = Table(title="Rank-Biserial Correlation")

table.add_column("Project")
for metric in metrics:
    table.add_column(metric)

for repo in PROJECTS:
    df = kamei[repo]
    X = df[metrics]
    y = df['buggy']
    corr = []
    for metric in metrics:
        corr.append(binary_continuous(y, X[metric], measure="rank_biserial", b_0=False, b_1=True))
    corr_str = [f"{abs(c):.3f}" for c in corr]
    table.add_row(repo, *corr_str)

console.print(table)

# %%
