import pandas as pd
import numpy as np
import javalang
import pickle
from rich.console import Console
from rich.progress import track
from rich.table import Table
from pathlib import Path
from extractor import CommitExtractor, ChangeType
from metrics import get_metrics, aggregate_metrics, identifiers
from argparse import ArgumentParser
from checkstyler import incorrect_indentations

OUTDIR = "results"

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

SAVE_INTERVAL = 50

def mining(repo, dataset_csv="apachejit_total.csv", verbose=True):
    console = Console()
    save_path = Path(f"{OUTDIR}/{repo}.csv")

    if save_path.exists():
        df = pd.read_csv(save_path, index_col="commit_id")

    else:
        df = pd.read_csv(dataset_csv, index_col="commit_id")
        df = df[df["project"] == f"apache/{repo}"]
        df["change_type"] = None

        save_path.parent.mkdir(parents=True, exist_ok=True)

    save_counter = 0
    console.print(f"Total: {len(df)}")
    save_interval = min(SAVE_INTERVAL, len(df) // 20)

    try:
        for commit_id, row in track(
            df.iterrows(),
            description=f"Processing {repo}...",
            total=len(df),
            console=console,
            disable=not verbose,
        ):
            if row.get("change_type") in [
                ChangeType.TRIVIAL.name,
                ChangeType.OVER.name,
                ChangeType.SYNTAX_ERROR.name,
                ChangeType.ERROR.name,
                ChangeType.MODIFY.name,
            ]:
                continue

            else:
                # Process the commit
                extractor = CommitExtractor(repo, commit_id)
                modified_methods = extractor.get_modified_methods(save=True)
                df.loc[commit_id, "change_type"] = extractor.change_type.name
                if extractor.repo != repo:
                    df.loc[commit_id, "project"] = f"apache/{extractor.repo}"
                if extractor.change_type == ChangeType.MODIFY:
                    for method in modified_methods:
                        if len(list(method.ast.filter(javalang.tree.ClassDeclaration))) > 0:
                            df.loc[commit_id, "change_type"] = ChangeType.ERROR.name
                            break

            if save_counter % save_interval == 0 and save_counter > 0:
                df.to_csv(save_path)
                if verbose:
                    console.clear()
                    visualize(df, repo, console)
            save_counter += 1

    except Exception as e:
        console.log(e)

    finally:
        df.to_csv(save_path)
        if verbose:
            visualize(df, repo)


def visualize(df, repo, console=Console()):
    only_modified = df[df["change_type"] == ChangeType.MODIFY.name]
    syntax_errors = df[df["change_type"] == ChangeType.SYNTAX_ERROR.name]
    errors = df[df["change_type"] == ChangeType.ERROR.name]
    adds = df[df["change_type"] == ChangeType.OVER.name]
    trivials = df[df["change_type"] == ChangeType.TRIVIAL.name]
    nones = df[df["change_type"].isnull()]

    table = Table(title=f"[bold green]{repo}[/bold green]")
    table.add_column("Type")
    table.add_column("Bug")
    table.add_column("Clean")
    table.add_column("Total")
    table.add_row(
        "Total",
        str(len(df[df["buggy"] == True])),
        str(len(df[df["buggy"] == False])),
        str(len(df)),
    )
    table.add_row(
        "Modified",
        str(len(only_modified[only_modified["buggy"] == True])),
        str(len(only_modified[only_modified["buggy"] == False])),
        f"{len(only_modified)} ({len(only_modified) / len(df) * 100:.1f}%)",
        style="bold cyan",
    )
    if len(syntax_errors) > 0:
        table.add_row(
            "Syntax Error",
            str(len(syntax_errors[syntax_errors["buggy"] == True])),
            str(len(syntax_errors[syntax_errors["buggy"] == False])),
            str(len(syntax_errors)),
            style="bold red",
        )
    if len(errors) > 0:
        table.add_row(
            "Error",
            str(len(errors[errors["buggy"] == True])),
            str(len(errors[errors["buggy"] == False])),
            str(len(errors)),
            style="bold red",
        )
    if len(adds) > 0:
        table.add_row(
            "Add",
            str(len(adds[adds["buggy"] == True])),
            str(len(adds[adds["buggy"] == False])),
            str(len(adds)),
            style="bold yellow",
        )
    if len(trivials) > 0:
        table.add_row(
            "Trivial",
            str(len(trivials[trivials["buggy"] == True])),
            str(len(trivials[trivials["buggy"] == False])),
            str(len(trivials)),
            style="white",
        )

    if len(nones) > 0:
        table.add_row(
            "None",
            str(len(nones[nones["buggy"] == True])),
            str(len(nones[nones["buggy"] == False])),
            str(len(nones)),
            style="aquamarine3",
        )
    console.print(table)


def compute_kamei(repo):
    save_path = Path(f"{OUTDIR}/{repo}_kamei.csv")
    dataset = Path(f"{OUTDIR}_old/{repo}.csv")
    origin = pd.read_csv(dataset, index_col="commit_id")
    origin = origin[origin["change_type"] == ChangeType.MODIFY.name]

    for commit_id in track(
        origin.index, description=f"Processing ...", total=len(origin), disable=False
    ):

        extractor = CommitExtractor(repo, commit_id)
        if extractor.check_storage(commit_id):
            modified_methods = extractor.load_methods(commit_id)
        else:
            modified_methods = extractor.get_modified_methods(save=True)

        before_code_set = set()
        for method in modified_methods:
            before_code_set.add(method.code)
        lt = 0
        for code in before_code_set:
            lt += len(code.splitlines())

        origin.loc[commit_id, "LT"] = lt

    df = pd.DataFrame(
        columns=[
            "LA/LT",
            "LD/LT",
            "LT/NF",
            "NUC/NF",
            "NS",
            "NF",
            "Entropy",
            "FIX",
            "NDEV",
            "AGE",
            "EXP",
            "SEXP",
            "buggy",
        ],
        index=origin.index,
    )
    df.loc[:, "LA/LT"] = origin["la"] / origin["LT"]
    df.loc[:, "LD/LT"] = origin["ld"] / origin["LT"]
    df.loc[:, "LT/NF"] = origin["LT"] / origin["nf"]
    df.loc[:, "NUC/NF"] = origin["nuc"] / origin["nf"]
    df[["NS", "NF", "Entropy", "FIX", "NDEV", "AGE", "EXP", "SEXP", "buggy"]] = origin[
        ["ns", "nf", "ent", "fix", "ndev", "age", "aexp", "asexp", "buggy"]
    ].copy()
    df.to_csv(save_path)
    origin.rename(
        columns={
            "ns": "NS",
            "nd": "ND",
            "nf": "NF",
            "ent": "Entropy",
            "la": "LA",
            "ld": "LD",
            "ndev": "NDEV",
            "age": "AGE",
            "nuc": "NUC",
            "aexp": "EXP",
            "arexp": "REXP",
            "asexp": "SEXP",
        },
        inplace=True,
    )
    origin.to_csv(f"{OUTDIR}/{repo}_kamei_origin.csv")


def compute_metrics_project(repo, load, rii=False):
    save_path = Path(f"{OUTDIR}/{repo}_metrics{'_rii' if rii else ''}.csv")
    if save_path.exists():
        df = pd.read_csv(save_path, index_col="commit_id")
    else:
        dataset = Path(f"{OUTDIR}/{repo}.csv")
        df = pd.read_csv(dataset, index_col="commit_id")
        modf = df[df["change_type"] == ChangeType.MODIFY.name]
        if rii:
            df = pd.DataFrame(columns=["RII"], index=modf.index)
        else:
            metrics = [
                "V",
                "EM",
                "EM/V",
                "DD",
                "DD/V",
                "MDNL",
                "NB",
                "REMC",
                "ENMC",
                "NP",
                "RG",
                "ATL",
                "RTS",
            ]
            df = pd.DataFrame(columns=metrics, index=modf.index)
            df["buggy"] = modf["buggy"].copy()

    counter = 0
    try:
        for commit_id in track(
            df.index, description=f"Processing ...", total=len(df), disable=False
        ):
            if load and not df.loc[commit_id].isna().any():
                continue

            extractor = CommitExtractor(repo, commit_id)
            if extractor.check_storage(commit_id):
                modified_methods = extractor.load_methods(commit_id)
            else:
                raise Exception(f"Not Saved {commit_id}")
            if rii:
                result = incorrect_indentations(modified_methods, commit_id)
                df.loc[commit_id, "RII"] = result
            else:
                metrics = [get_metrics(method) for method in modified_methods]
                agg_metrics = aggregate_metrics(metrics)
                for metric, value in agg_metrics.items():
                    df.loc[commit_id, metric] = value

            counter += 1
            if counter % SAVE_INTERVAL == 0:
                df.to_csv(save_path)
    except KeyboardInterrupt:
        pass
    df.to_csv(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--mining", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--rii", action="store_true")
    parser.add_argument("--kamei", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--num_methods", action="store_true")
    
    args = parser.parse_args()

    if args.kamei:
        for project in PROJECTS:
            compute_kamei(project)

    if args.metrics and args.project:
        compute_metrics_project(args.project, args.load, rii=args.rii)

    if args.mining and args.project:
        mining(args.project, verbose=args.verbose)

    if args.concat:
        dfs = []
        for project in PROJECTS:
            print(project)

            df = pd.read_csv(f"{OUTDIR}/{project}.csv", index_col="commit_id")
            df = df[df["change_type"] == ChangeType.MODIFY.name]

            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv("{OUTDIR}/total.csv")

        hadoop = pd.concat(
            [
                df[df["project"] == "apache/hadoop"],
                df[df["project"] == "apache/hadoop-hdfs"],
                df[df["project"] == "apache/hadoop-mapreduce"],
            ]
        )

        table = Table(title=f"[bold green]Total[/bold green]")
        table.add_column("Project")
        table.add_column("Defect-inducing")
        table.add_column("Clean")
        table.add_column("Total")

        df = df[df["change_type"] == "MODIFY"]
        df.loc[df["project"] == "apache/hadoop-mapreduce", "project"] = "apache/hadoop"
        df.loc[df["project"] == "apache/hadoop-hdfs", "project"] = "apache/hadoop"
        project_list = df["project"].unique()
        for project in project_list:
            no_defects = df[(df["buggy"] == False) & (df["project"] == project)]
            defects = df[(df["buggy"] == True) & (df["project"] == project)]

            table.add_row(
                project.replace("apache/", ""),
                str(len(defects)),
                str(len(no_defects)),
                str(len(defects) + len(no_defects)),
            )
        table.add_row(
            "Total",
            str(len(df[df["buggy"] == True])),
            str(len(df[df["buggy"] == False])),
            str(len(df)),
        )
        console = Console()
        console.print(table)

    if args.num_methods:
        console = Console()
        df = pd.read_csv("{OUTDIR}/total.csv", index_col="commit_id")
        all_identifiers = {}
        last_commit = None
        total = 0
        saved = None
        if args.load:
            saved = next(Path("{OUTDIR}").glob("*.pkl"))
            with open(saved, "rb") as f:
                all_identifiers = pickle.load(f)
                _, total, last_commit = saved.stem.split("_")
                total = int(total)

        if last_commit != None:
            flag = False
        else:
            flag = True

        if last_commit == df.index[-1]:
            top = sorted(all_identifiers.items(), key=lambda x: x[1], reverse=True)
            top = sorted(top, key=lambda x: len(x[0]))
            console.clear()
            console.print(top[:1000])
            exit()
        for commit_id, row in track(
            df.iterrows(), description=f"Processing ...", total=len(df), console=console
        ):
            if not flag:
                if commit_id != last_commit:
                    continue
                flag = True
                continue

            owner, repo = row["project"].split("/")
            extractor = CommitExtractor(repo, commit_id)
            if extractor.check_storage(commit_id):
                modified_methods = extractor.load_methods(commit_id)
            else:
                modified_methods = extractor.get_modified_methods(save=True)
            for method in modified_methods:
                for identifier in identifiers(method):
                    if identifier not in all_identifiers:
                        all_identifiers[identifier] = 0
                    all_identifiers[identifier] += 1
                total += 1

                if total % 1000 == 0:
                    # top 3 identifiers
                    top = sorted(
                        all_identifiers.items(), key=lambda x: x[1], reverse=True
                    )[:20]
                    console.clear()
                    console.print(f"Total: {total}")
                    console.print(top)

                    with open(
                        f"{OUTDIR}/identifiers_{total}_{commit_id}.pkl", "wb"
                    ) as f:
                        pickle.dump(all_identifiers, f)
                        if saved:
                            saved.unlink(missing_ok=True)
                        saved = Path(f"{OUTDIR}/identifiers_{total}_{commit_id}.pkl")

        console.print(f"Total: {total}")
        with open(f"{OUTDIR}/identifiers_{total}_{commit_id}.pkl", "wb") as f:
            pickle.dump(all_identifiers, f)
            if saved:
                saved.unlink(missing_ok=True)
            saved = Path(f"{OUTDIR}/identifiers_{total}_{commit_id}.pkl")

        top = sorted(all_identifiers.items(), key=lambda x: x[1], reverse=True)
        top = sorted(top, key=lambda x: len(x[0]))
        console.clear()
        console.print(top[:200])

        # plot histogram for length of identifiers
        result = {}
        for identifier, count in all_identifiers.items():
            if result.get(len(identifier)) == None:
                result[len(identifier)] = 0
            result[len(identifier)] += count

        import matplotlib.pyplot as plt

        plt.bar(result.keys(), result.values())
        plt.show()
