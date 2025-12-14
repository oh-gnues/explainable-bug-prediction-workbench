# Kamei Metrics

Kamei et al. (2013)의 논문에서 제안된 commit-level 소프트웨어 메트릭을 추출한 결함 유발 코드 예측 데이터셋입니다.

## Dataset Description

각 프로젝트에 포함된 커밋의 통ㄱ계는 다음과 같습니다.

| Project   | Defect-inducing | Clean | Total |
|-----------|-----------------|-------|-------|
| Zeppelin  | 137 (26%)       | 371   | 508   |
| ZooKeeper | 65 (23%)        | 216   | 281   |
| ActiveMQ  | 288 (11%)       | 2203  | 2491  |
| Camel     | 506 (7%)        | 6465  | 6971  |
| Cassandra | 747 (21%)       | 2657  | 3404  |
| Flink     | 347 (10%)       | 2865  | 3212  |
| Groovy    | 525 (14%)       | 2987  | 3512  |
| Hadoop    | 458 (9%)        | 4309  | 4767  |
| HBase     | 786 (26%)       | 2237  | 3023  |
| Hive      | 1018 (44%)      | 1257  | 2275  |
| Ignite    | 345 (7%)        | 4091  | 4436  |
| Kafka     | 152 (23%)       | 499   | 651   |
| Spark     | 156 (31%)       | 334   | 490   |
| **Total** | **5530 (15%)**  | **30491** | **36021** |

Kamei et al. (2013)의 논문에서 제안된 소프트웨어 메트릭은 다음과 같습니다. 

| Dim.    | Name  | Definition                                       |
|---------|-------|--------------------------------------------------|
| Diffusion | NS    | Number of modified subsystems                   |
|          | ND    | Number of modified directories                  |
|          | NF    | Number of modified files                        |
|          | Entropy | Distribution of modified code across each file  |
| Size    | LA    | Lines of code added                             |
|          | LD    | Lines of code deleted                           |
|          | LT    | Lines of code in files before the change        |
| History | NDEV  | Number of developers who changed the files      |
|          | AGE   | Average time interval between changes           |
|          | NUC   | Number of unique changes to the modified files  |
| Expert  | EXP   | Developer experience                            |
|          | REXP  | Recent developer experience                    |
|          | SEXP  | Developer experience on a subsystem             |

본 데이터셋에는 위의 메트릭에 대해 프로젝트별로 상관계수가 높은 메트릭을 제거하기 위한 전처리가 적용되었습니다. 
즉, LA, LD 를 LT로 나누는 등의 메트릭 변환을 수행하였습니다. 각 프로젝트별로 전처리된 메트릭은 `{project_name}_metrics.csv` 파일에서 확인할 수 있습니다.

