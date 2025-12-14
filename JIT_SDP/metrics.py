import subprocess
import javalang
import numpy as np
import pandas as pd
from rich.console import Console
from cfg import CFG, create_use_def_graph
from extractor import CommitExtractor, Method
from scipy.stats import skew

console = Console()


def halstead(method: Method):
    operators = []
    operands = []

    for path, node in method.ast:
        if getattr(node, "modifiers", None):
            operators.extend(node.modifiers)
        if getattr(node, "throws", None):
            operators.append("throws")
            operands.extend(node.throws)

        if isinstance(node, javalang.tree.Expression):
            if getattr(node, "operator", None):
                operators.append(node.operator)
            if getattr(node, "prefix_operators", None):
                operators.extend(node.prefix_operators)
            if getattr(node, "postfix_operators", None):
                operators.extend(node.postfix_operators)
            if getattr(node, "qualifier", None):
                operators.append(".")
                operands.append(node.qualifier)

            if isinstance(node, javalang.tree.ArraySelector):
                operators.append("[]")
            elif isinstance(node, javalang.tree.Cast):
                operators.append("()")
                operands.append(node.type.name)
            elif isinstance(node, javalang.tree.Assignment):
                operators.append(node.type)
            elif isinstance(node, javalang.tree.MethodReference):
                operators.append("::")
            elif isinstance(node, javalang.tree.LambdaExpression):
                operators.append("->")
            elif isinstance(node, javalang.tree.ClassReference):
                operators.append(".class")
            elif isinstance(node, javalang.tree.TernaryExpression):
                operators.append("?:")

            elif isinstance(node, javalang.tree.Invocation):
                operators.append("()")

                if getattr(node, "type_arguments", None):
                    operators.append("<>")
                    for type_arg in node.type_arguments:
                        operands.append(type_arg.type)

                if isinstance(
                    node,
                    (
                        javalang.tree.MethodInvocation,
                        javalang.tree.SuperMethodInvocation,
                    ),
                ):
                    operands.append(node.member)
                if isinstance(node, javalang.tree.ExplicitConstructorInvocation):
                    operands.append("this")
                if isinstance(node, javalang.tree.SuperConstructorInvocation):
                    operands.append("super")

            elif isinstance(
                node,
                (javalang.tree.MemberReference, javalang.tree.SuperMemberReference),
            ):
                operands.append(node.member)
            elif isinstance(node, javalang.tree.This):
                operands.append("this")
            elif isinstance(node, javalang.tree.Literal):
                operands.append(node.value)

        if isinstance(node, javalang.tree.Creator):
            operators.append("new")
            operands.append(node.type.name)

        if isinstance(node, javalang.tree.IfStatement):
            if node.then_statement:
                operators.append("if")
            if node.else_statement:
                operators.append("else")
        if isinstance(node, javalang.tree.ForControl):
            operators.append("for")
        if isinstance(node, javalang.tree.EnhancedForControl):
            operators.extend(["for", ":"])
        if isinstance(node, javalang.tree.WhileStatement):
            operators.append("while")
        if isinstance(node, javalang.tree.DoStatement):
            operators.append("do")
        if isinstance(node, javalang.tree.SynchronizedStatement):
            operators.append("synchronized")
        if isinstance(node, javalang.tree.SwitchStatement):
            operators.append("switch")
        if isinstance(node, javalang.tree.SwitchStatementCase):
            if len(node.case) > 0:
                operators.append("case")
            else:
                operators.append("default")
        if isinstance(node, javalang.tree.BreakStatement):
            operators.append("break")
        if isinstance(node, javalang.tree.TryStatement):
            operators.append("try")
            if node.finally_block:
                operators.append("finally")
        if isinstance(node, javalang.tree.CatchClause):
            operators.append("catch")
        if isinstance(node, javalang.tree.CatchClauseParameter):
            operands.append(node.name)
        if isinstance(node, javalang.tree.ThrowStatement):
            operators.append("throw")
        if isinstance(node, javalang.tree.ContinueStatement):
            operators.append("continue")
        if isinstance(node, javalang.tree.AssertStatement):
            operators.append("assert")
        if isinstance(node, javalang.tree.ReturnStatement):
            operators.append("return")
    # Calculate Halstead metrics
    n1 = len(set(operators))  # Number of distinct operators
    n2 = len(set(operands))  # Number of distinct operands
    N1 = len(operators)  # Total number of operators
    N2 = len(operands)  # Total number of operands

    vocabulary = n1 + n2
    length = N1 + N2
    volume = length * np.log2(vocabulary) if vocabulary > 0 else 0
    difficulty = (n1 / 2) * (N2 / n2) if n2 != 0 else 0
    effort = difficulty * volume

    return {
        "vocabulary": vocabulary,
        "length": length,
        "volume": volume,
        "difficulty": difficulty,
        "effort": effort,
    }


def entropy(method: Method):
    tokens = [token.value for token in method.tokens]
    unique_tokens = set(tokens)
    token_count = len(tokens)
    token_entropy = 0
    for token in unique_tokens:
        token_probability = tokens.count(token) / token_count
        token_entropy += token_probability * np.log2(token_probability)
    token_entropy *= -1
    return token_entropy

def token_skewness(method: Method):
    tokens = [token.value for token in method.tokens]
    unique_tokens = set(tokens)
    distribution = [ tokens.count(token) for token in unique_tokens ]
    return skew(distribution)

def terms_per_line(method: Method):
    tokens = [token for token in method.tokens]
    tokens_per_line = {}
    for token in tokens:
        if token.position.line not in tokens_per_line:
            tokens_per_line[token.position.line] = []
        tokens_per_line[token.position.line].append(token)
    terms_per_line = np.max([len(tokens) for tokens in tokens_per_line.values()])
    return terms_per_line


def depdegree(method: Method):
    cfg = CFG(method)
    cfg.compute_reaching_definitions()
    use_def_graph = create_use_def_graph(cfg)
    return use_def_graph.depdegree


def REMC(method: Method):
    called_methods = set()
    local_methods = set()
    for path, node in method.ast:
        if isinstance(node, javalang.tree.MethodInvocation):
            if node.qualifier:
                called_methods.add(f"{node.qualifier}.{node.member}")
            else:
                local_methods.add(node.member)
        elif isinstance(node, javalang.tree.SuperConstructorInvocation):
            called_methods.add(f"super")
        elif isinstance(node, javalang.tree.SuperMethodInvocation):
            called_methods.add(f"super.{node.member}")
        elif isinstance(node, javalang.tree.ExplicitConstructorInvocation):
            local_methods.add(f"this")
    total = len(local_methods) + len(called_methods)
    if total == 0:
        return 0
    return len(called_methods) / total


def enmc(method: Method):
    external_methods = set()
    hierarchy = method.signature.split("::")
    for component in hierarchy:
        if "(" in component:
            external_methods.add(component)
    return len(external_methods) - 1


def num_params(method: Method):
    return len(method.ast.parameters)


def identifiers(method: Method):
    identifiers = [
        token.value
        for token in method.tokens
        if isinstance(token, javalang.tokenizer.Identifier)
    ]
    return identifiers


def rts(method: Method):
    except_tokens = ["i", "e"]
    ids = [
        token.value
        for token in method.tokens
        if isinstance(token, javalang.tokenizer.Identifier) and len(token.value) <= 2 and token.value not in except_tokens
    ]
    return len(ids) / len(identifiers(method))

  
def lines_of_before_code(method: Method):
    all_tokens = javalang.tokenizer.tokenize(method.code)
    lines = set([token.position.line for token in all_tokens])
    return len(lines)

def check_comment(stripped_line: str, state):
    if stripped_line.endswith("*/"):
        return "END"
    if stripped_line.startswith("//"):
        return True
    if stripped_line.startswith("/**"):
        return "JAVADOC_START"
    if stripped_line.startswith("/*"):
        return "START"
    if stripped_line.startswith("*"):
        return "JAVADOC_ING"

    if state in ["JAVADOC_START", "JAVADOC_ING", "START", "ING"]:
        return "ING"
    return False

def consume_tokens(line: str, tokens: list):
    for token in tokens:
        token_index = line.find(token.value)
        if token_index != -1:
            line = line[token_index+len(token.value):]
    return line

def get_comment_lines(method: Method):
    comment_lines = {}
    tokens_by_line = {}
    for token in method.tokens:
        if token.position.line not in tokens_by_line:
            tokens_by_line[token.position.line] = []
        tokens_by_line[token.position.line].append(token)

    codelines = method.code.splitlines()
    last_state = None
    for line in range(method.start_line, method.end_line+1):
        target_line = codelines[line-1]
        if line not in tokens_by_line:
            target_line = target_line.strip()
            if not target_line:
                continue
            last_state = check_comment(target_line, last_state)

            if last_state == False:
                continue
            
            comment_lines[line] = target_line

            if last_state == "END":
                last_state = None
            continue
        last_state = None
        target_line = consume_tokens(target_line, tokens_by_line[line])
        target_line = target_line.strip()
        if not target_line:
            continue
        last_state = check_comment(target_line, last_state)
        if last_state == False:
            continue
        comment_lines[line] = target_line

    return comment_lines


def run_git_blame(file, start_line, end_line):
    result = subprocess.run(
        ['git', 'blame', '-L', f'{start_line},{end_line}', file],
        capture_output=True, text=True
    )
    return result.stdout



def get_metrics(method: Method):
    cfg = CFG(method)
    cfg.compute_reaching_definitions()
    use_def_graph = create_use_def_graph(cfg)
    v = halstead(method)["volume"]
    em = entropy(method)
    dd = use_def_graph.depdegree
    return {
        # Internal
        "V": v,
        "EM": em,
        "EM/V": em / v,
        "DD": dd,
        "DD/V": dd / v,
        "MDNL": cfg.MDNL,
        "NB": cfg.NB,
        # External
        "REMC": REMC(method),
        "ENMC": enmc(method),
        "NP": num_params(method),
        "RG": len(cfg.global_variables) / len(cfg.variables) if len(cfg.variables) > 0 else 0,
        # # Representational
        "MTL": terms_per_line(method),
        "RTS": rts(method),
        "RC": len(get_comment_lines(method)) / method.loc,
    }


def aggregate_metrics(metrics):
    df = pd.DataFrame(metrics)
    max_values = df.max()
    return max_values.to_dict()


if __name__ == "__main__":
    df = pd.read_csv("results/total.csv", index_col="commit_id")
    test_rows = df.sample(1)
    for test_commit_id, row in test_rows.iterrows():
        owner, repo = row["project"].split("/")
        extractor = CommitExtractor(repo, test_commit_id)
        methods = extractor.load_methods(test_commit_id)
        metrics = []
        for method in methods:
            metric = get_metrics(method)
            console.print(metric)
            metrics.append(metric)
        console.print(aggregate_metrics(metrics))

