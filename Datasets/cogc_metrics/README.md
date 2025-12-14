# CogC Metrics

최신의 인지 복잡도에 대한 NeuroSE 연구를 기반으로 구축된 13종류의 오픈소스 프로젝트 및 JIRA 이슈 리포트로부터 추출된 commit의 인지 복잡도 메트릭  데이터셋입니다.

## Dataset Description

사용된 프로젝트 목록은 [`Dataset/kamei_metrics.README.md`](../kamei_metrics/README.md)의 프로젝트 목록과 동일합니다.

아래와 같은 12종류의 인지 복잡도 메트릭을 포함하고 있습니다.

| Dim.    | Name | Definition                                                               |
|---------|------|--------------------------------------------------------------------------|
| Internal | V    | The Halstead’s volume [18] that calculates the minimum number of bits needed to natively represent the method |
|         | EM   | The entropy value representing the relative distribution of the terms (i.e., keywords, identifiers, operators) in the method |
|         | DD   | The DepDegree [4] that calculates the total number of edges in the data-flow graph |
|         | MDNL | The max depth of nesting loop in the method                              |
|         | NB   | The total number of non-structured branch statements (i.e., continue, break) in the method |
| External | REMC | The ratio of external method call expressions (e.g., libraries, APIs) to total method call expressions |
|         | ENMC | The number of externally nested methods from the method                 |
|         | NP   | The number of parameters in the method                                   |
|         | RG   | The ratio of global variables to total variables                         |
| Repre.  | NTM  | The number of terms in the line with the most terms within the method    |
|         | RTS  | The ratio of variables that have short names to total number of variables in the method |
|         | RII  | The ratio of warnings for incorrect indentations examined by CheckStyle [7] to total lines of code |
