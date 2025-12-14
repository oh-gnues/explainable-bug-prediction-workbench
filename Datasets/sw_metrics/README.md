# SW Metrics
9종류의 Apache 오픈소스 프로젝트의 JIRA ITS 이슈들을 분석하여, 소프트웨어 메트릭을 추출한 결함 유발 코드 예측 데이터셋입니다.

## Dataset Description

| Name    | Description                           | #Files        | #Defective files | Studied Releases            |
|---------|---------------------------------------|---------------|------------------|----------------------------|
| ActiveMQ| Messaging and Integration Patterns sever| 1,759-2,822 | 154-292        | 5.0.0, 5.1.0, 5.2.0, 5.3.0, 5.8.0|
| Camel   | Enterprise Integration Framework      | 1,492-8,243  | 189-285        | 1.4.0, 2.9.0, 2.10.0, 2.11.0|
| Derby   | Relational Database                   | 1,953-2,681  | 383-667        | 10.2.1.6, 10.3.1.4, 10.5.1.1|
| Groovy  | Java-syntax-compatible OOP for JAVA   | 749-868      | 26-73          | 1.5.7, 1.6.0_Beta1, 1.6.0_Beta2|
| HBase   | Distributed Scalable Data Store       | 1,059-1,793  | 218-463        | 0.94.0, 0.95.0, 0.95.2     |
| Hive    | Data Warehouse System for Hadoop      | 1,326-2,537  | 176-278        | 0.9.0, 0.10.0, 0.12.0      |
| JRuby   | Ruby Programming Lang for JVM         | 699-1,576    | 82-180         | 1.1, 1.4, 1.5, 1.7         |
| Lucene  | Text Search Engine Library            | 799-2,528    | 84-273         | 2.3.0, 2.9.0, 3.0.0, 3.1.0 |
| Wicket  | Web Application Framework             | 1,613-2,508  | 101-130        | 1.3.0.beta1, 1.3.0.beta2, 1.5.3|

각 릴리즈 디렉토리는 시간 순서대로 0부터 시작하는 숫자로 표현되어 있습니다. 예를 들어, ActiveMQ의 5.0.0 릴리즈는 `activemq@0`, 5.1.0 릴리즈는 `activemq@1`로 표현됩니다.

각 릴리즈 디렉토리에는 `mapping.csv` 파일이 있습니다. 이 파일은 릴리즈에 포함된 파일들의 ID와 실제 파일명을 매핑한 파일입니다.

또한 `train.csv`, `test.csv` 파일이 있습니다. 이 파일들은 각각 학습 데이터셋과 테스트 데이터셋을 나타냅니다. 각 파일은 다음과 같은 소프트웨어 메트릭을 포함하고 있습니다.

| Category         | Metrics Level | Metrics                                                                                                                                                              | Count |
|------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| Code metrics     | File-level    | AvgCyclomaticModified, AvgCyclomaticStrict, AvgEssential, AvgLineBlank, AvgLineComment, CountDeclClass, CountDeclClassMethod, CountDeclClassVariable, CountDeclInstanceMethod, CountDeclInstanceVariable, CountDeclMethodDefault, CountDeclMethodPrivate, CountDeclMethodProtected, CountDeclMethodPublic, CountLineComment, RatioCommentToCode | 16    |
|                  | Class-level   | CountClassBase, CountClassCoupled, CountClassDerived, MaxInheritanceTree, PercentLackOfCohesion                                                                                                  | 5     |
|                  | Method-level  | CountInput_Mean, CountInput_Min, CountOutput_Mean, CountOutput_Min, CountPath_Min, MaxNesting_Mean, MaxNesting_Min                                                                             | 7     |
| Process metrics  |               | ADEV, ADDED_LINES, DEL_LINES                                                                                                                                                                      | 3     |
| Ownership metrics|               | MAJOR_COMMIT, MAJOR_LINE, MINOR_COMMIT, MINOR_LINE, OWN_COMMIT, OWN_LINE                                                                                                                          | 6     |
