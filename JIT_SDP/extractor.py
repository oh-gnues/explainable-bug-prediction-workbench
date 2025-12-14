import subprocess
from typing import List
import javalang
from javalang.tree import (
    ClassDeclaration,
    MethodDeclaration,
    ConstructorDeclaration,
)

from pydriller import Git
from git import Repo
from enum import Enum
from pathlib import Path
import pickle
from itertools import zip_longest
import os
from rich.console import Console

# sys.setrecursionlimit(10000)  # Be cautious with this, don't set it too high
console = Console()

class Method:
    def __init__(self, ast: javalang.ast.Node, code: str, signature: str):
        self.ast = ast
        self.code = code
        self.documentation = self.ast.documentation
        self.start_line, self.end_line = self._get_position()
        self.signature = signature
        self.added_lines = set()
        self.deleted_lines = set()
        self.code_after = None

    def __eq__(self, __value: object) -> bool:
        if not hasattr(__value, "signature"):
            return False
        return self.signature == __value.signature

    def __hash__(self) -> int:
        return hash(self.signature)

    def __repr__(self) -> str:
        return self.signature

    @property
    def position(self):
        return (self.start_line, self.end_line)
    
    @property
    def nested_level(self):
        return len(self.signature.split("::"))

    def _get_position(self):
        start_line = self.ast.position.line
        end_line = self.ast.position.line

        if self.ast.annotations:
            line = min(
                [annotation._position.line for annotation in self.ast.annotations]
            )
            if line < start_line:
                start_line = line

        if self.ast.documentation:
            length = len(self.ast.documentation.split("\n"))
            maybe_start_line = start_line - length
            valid = True
            tokens = javalang.tokenizer.tokenize(self.code)
            for token in tokens:
                if token.position.line == maybe_start_line:
                    valid = False
                    break
                elif token.position.line > start_line:
                    break
            if valid:
                start_line = maybe_start_line

        for path, node in self.ast:
            if hasattr(node, "position") and node.position:
                line = node.position.line
            elif hasattr(node, "_position") and node._position:
                line = node._position.line
            else:
                continue
            if line > end_line:
                end_line = line

        tokens = javalang.tokenizer.tokenize(self.code)
        smallest_column = 1000
        for token in tokens:
            if token.position.line == self.ast.position.line:
                if smallest_column > token.position.column:
                    smallest_column = token.position.column
            if token.position.line > end_line:
                if token.value == '}' and token.position.column >= smallest_column:
                    end_line = token.position.line
                else:
                    break

        return start_line, end_line
    
    def contains(self, line, type):
        if self.ast.position.line <= line <= self.end_line:
            if type == 'ADD':
                self.added_lines.add(line)
            elif type == 'DELETE':
                self.deleted_lines.add(line)
            return True
        else:
            return False

    @property
    def snippet(self):
        lines = self.code.split("\n")
        return "\n".join(lines[self.start_line - 1 : self.end_line])
    
    @property
    def line_numbered_snippet(self):
        lines = self.code.split("\n")
        return "\n".join(
            [f"{i+self.start_line:4} | {line}" for i, line in enumerate(lines[self.start_line - 1 : self.end_line])]
        )

    @property
    def tokens(self):
        all_tokens = javalang.tokenizer.tokenize(self.code)
        tokens = []
        for token in all_tokens:
            if self.start_line <= token.position.line <= self.end_line:
                tokens.append(token)

        return tokens

    @property
    def loc(self):
        return self.end_line - self.start_line + 1

    @classmethod
    def from_file(cls, code):
        tree = javalang.parse.parse(code)
        for path, node in tree:
            if isinstance(node, (MethodDeclaration, ConstructorDeclaration)):
                if not node.body:
                    continue
                signature = cls._generate_full_signature(path, node)
                yield cls(node, code, signature)
    
    @staticmethod
    def _generate_full_signature(path, node) -> str:
        names = []
        for p in path:
            if isinstance(p, ClassDeclaration):
                names.append(p.name)
            elif isinstance(p, (MethodDeclaration, ConstructorDeclaration)):
                # 메서드나 생성자의 파라미터 타입을 가져와서 시그니처를 생성합니다.
                param_types = [param.type.name for param in p.parameters]
                names.append(f'{p.name}({",".join(param_types)})')

        # 마지막 노드(현재 메서드 또는 생성자)의 시그니처를 추가합니다.
        if isinstance(node, (MethodDeclaration, ConstructorDeclaration)):
            param_types = [param.type.name for param in node.parameters]
            names.append(f'{node.name}({",".join(param_types)})')
        # 계층 구조를 "::"로 연결하여 전체 시그니처를 반환합니다.
        return "::".join(names)
    

class ChangeType(Enum):
    OVER = "OVER"
    MODIFY = "MODIFY"
    TRIVIAL = "TRIVIAL"
    ERROR = "ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"


class CommitExtractor:
    def __init__(self, repo, commit_id, repo_dir=None):
        self.repo_dir = repo_dir if repo_dir else repo
        self.repo = repo
        self.commit_id = commit_id
        self.change_type = None
        self.commit_obj = None
        self._validate()

    def _validate(self):
        if not Path(self.repo_dir).exists():
            Repo.clone_from(f"https://github.com/apache/{self.repo}.git", self.repo_dir)
        try:
            commit_obj = Git(self.repo_dir).get_commit(self.commit_id)
            if commit_obj is None:
                self.change_type = ChangeType.ERROR
            else:
                self.commit_obj = commit_obj

        except Exception as e:
            if self.repo in ["hadoop-hdfs", "hadoop-mapreduce"]:
                self.repo = "hadoop"
                self.repo_dir = self.repo_dir.replace("hadoop-hdfs", "hadoop").replace(
                    "hadoop-mapreduce", "hadoop"
                )
                self._validate()
            else:
                git_lock = Path(self.repo_dir) / ".git" / "config.lock"
                if git_lock.exists():
                    git_lock.unlink()
                    self._validate()
                else:
                    self.change_type = ChangeType.ERROR

    def get_modified_methods(self, save=True) -> List[Method]:
        if self.change_type is ChangeType.ERROR:
            return []

        modified_files = self.commit_obj.modified_files
        modified_files = [
            file
            for file in modified_files
            if file.filename.endswith(".java")
        ]
        if len(modified_files) == 0:
            self.change_type = ChangeType.TRIVIAL
            return []

        non_modify = [
            file
            for file in modified_files
            if file.change_type.name in ["ADD", "RENAME", "DELETE"]
        ]
        if len(non_modify) > 0:
            self.change_type = ChangeType.OVER
            return []

        modified_methods = []
        for i, file in enumerate(modified_files):
            methods = self._get_modified_methods(file)
            if self.change_type == ChangeType.MODIFY:
                modified_methods.extend(methods)
            elif self.change_type in [
                ChangeType.SYNTAX_ERROR,
                ChangeType.OVER,
                ChangeType.ERROR,
            ]:
                return []

        if len(modified_methods) == 0:
            self.change_type = ChangeType.TRIVIAL
            return []
        self.change_type = ChangeType.MODIFY

        if save:
            self.save_methods(methods=modified_methods)

        return modified_methods
    

    def _get_modified_methods(self, file, ignore_comment=False):
        
        is_err_before =  self.check_syntax_error(file.source_code_before)
        if is_err_before in [ChangeType.SYNTAX_ERROR, ChangeType.ERROR]:
            self.change_type = is_err_before
            return []

        is_err =  self.check_syntax_error(file.source_code)
        if is_err in [ChangeType.SYNTAX_ERROR, ChangeType.ERROR]:
            self.change_type = is_err
            return []

        before_methods_gen = Method.from_file(file.source_code_before)
        after_methods_gen = Method.from_file(file.source_code)
            
        modified_methods = set()

        added_lines = set([line[0] for line in file.diff_parsed["added"]])
        deleted_lines = set([line[0] for line in file.diff_parsed["deleted"]])

        for before_method, after_method in zip_longest(before_methods_gen, after_methods_gen):
            if before_method != after_method:
                self.change_type = ChangeType.OVER
                return []
            
            if ignore_comment and set([ str(toekn) for toekn in before_method.tokens ]) == set([ str(toekn) for toekn in after_method.tokens ]):
                continue
 
            method_add_lines = {line for line in added_lines if after_method.contains(line, "ADD")}
            method_del_lines = {line for line in deleted_lines if before_method.contains(line, "DELETE")}

            if method_add_lines or method_del_lines:
                if method_add_lines:
                    before_method.added_lines = after_method.added_lines
                    
                    added_lines -= method_add_lines
                if method_del_lines:
                    deleted_lines -= method_del_lines
                before_method.code_after = after_method.code
                modified_methods.add(before_method)

        if len(modified_methods) == 0:
            self.change_type = ChangeType.TRIVIAL
            return []
        
        self.change_type = ChangeType.MODIFY
        return list(modified_methods)

    def save_methods(self, storage_path="commit_cache", methods=None):
        try:
            file_path = Path(storage_path) / f"{self.commit_id}.pkl"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                if methods is None:
                    methods = self.get_modified_methods()
                pickle.dump(methods, f)
        except Exception as e:
            print(e)
            pass

    @classmethod
    def load_methods(cls, commit_id, storage_path="commit_cache") -> List[Method]:
        file_path = Path(storage_path) / f"{commit_id}.pkl"
        if not file_path.exists():
            return None
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def check_storage(cls, commit_id, storage_path="commit_cache"):
        file_path = Path(storage_path) / f"{commit_id}.pkl"
        return file_path.exists()
    
    @staticmethod
    def check_syntax_error(code):
        try:
            javalang.parse.parse(code)
            return True
        except javalang.parser.JavaSyntaxError:
            return ChangeType.SYNTAX_ERROR
        except Exception:
            return ChangeType.ERROR
        
    def find_file_path_from_code(self, methods: List[Method]):
        modified_files = self.commit_obj.modified_files
        modified_files = [
            file
            for file in modified_files
            if file.filename.endswith(".java") and file.change_type.name == "MODIFY"
        ]

        results = {}
        for file in modified_files:
            for method in methods:
                if method.signature.split('::')[0] == file.old_path.split('/')[-1].split('.')[0]:
                    
                    results[method] = file.old_path
        
        return results
        
    
    def historical_commits(self, methods: List[Method]):
        console = Console()
        method_file_map = self.find_file_path_from_code(methods)
        # console.print(method_file_map)
        prev_commit = self.commit_obj.parents[0]

        # check out to prev_commit and 'git log -L {start_line},{end_line}:{file_path}'
        # find commit id which modified the method
        Git(self.repo_dir).checkout(prev_commit)
        results = {}
        pwd = os.getcwd()
        os.chdir(self.repo_dir)
        for method, file_path in method_file_map.items():
            console.print(f"Method: {method}")
            command = ["git", "log", "-L", f"{method.start_line},{method.end_line}:{file_path}", "--pretty=format:%H" ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stderr:
                console.log(result.stderr)
                exit()
            console.print(result.stdout)
            
        os.chdir(pwd)
        return results
