from pathlib import Path
import subprocess
from typing import List
from extractor import Method, CommitExtractor
import pandas as pd

CACHE_DIR = "./chekstyle_cache"
CHECKSTYLE_PATH = "./checkstyle.jar"
CONFIG_PATH = "./indentation_config.xml"

def incorrect_indentations(methods: List[Method], commit_id: str):
    temp_foler = Path(CACHE_DIR) / commit_id
    temp_foler.mkdir(parents=True, exist_ok=True)

    ckresult = {}

    for method in methods:
        tmp_file = temp_foler / f"{hash(method.signature)}.java"
        if not tmp_file.exists():
            with open(tmp_file, "w") as f:
                f.write(method.code)
        if tmp_file not in ckresult:
            ckresult[tmp_file] = run_checkstyle(tmp_file)
 
    iis = []
    for method in methods:
        ii = 0
        tmp_file = temp_foler / f"{hash(method.signature)}.java"
        for line in range(method.start_line, method.end_line+1):
            if line in ckresult[tmp_file]:
                ii += 1
        iis.append(ii/method.loc)
    return max(iis)

def indentation_warnings(repo, commit_id):
    extractor = CommitExtractor(repo, commit_id)
    if extractor.check_storage(commit_id):
        methods = extractor.load_methods(commit_id)
    else:
        methods = extractor.get_modified_methods(commit_id)

    temp_foler = Path(CACHE_DIR) / commit_id
    temp_foler.mkdir(parents=True, exist_ok=True)

    ckresult = {}

    for method in methods:
        method.start_line
        method.end_line
        if method.code:
            tmp_file = temp_foler / f"{hash(method.signature)}.java"
            if not tmp_file.exists():
                with open(tmp_file, "w") as f:
                    f.write(method.code)
            if tmp_file not in ckresult:
                ckresult[tmp_file] = []
            ckresult[tmp_file].append((method.start_line, method.end_line))
    
    warnings = 0
    for java_file, lines in ckresult.items():
        result = run_checkstyle(java_file)
        if result:
            ranges = set()
            for start, end in lines:
                ranges.update(range(start, end+1))
            warnings += len(ranges.intersection(set(result)))
    return warnings

def run_checkstyle(java_file):
    checkstyle_path = CHECKSTYLE_PATH
    config_path = CONFIG_PATH
    command = ["java", "-jar", checkstyle_path, "-c", config_path, java_file]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout.startswith("Files to process must be specified"):
        raise Exception("Checkstyle failed")
    # Parse the result
    errors = result.stdout.splitlines()
    errors = errors[1:-1]
    errors = [int(error.split(".java:")[1].split(':')[0]) for error in errors]

    return errors


if __name__ == "__main__":
    df = pd.read_csv("results/total.csv", index_col="commit_id")
    test_rows = df.sample(10)
    for test_commit_id, row in test_rows.iterrows():
        owner, repo = row["project"].split("/")
        print(indentation_warnings(repo, test_commit_id))

