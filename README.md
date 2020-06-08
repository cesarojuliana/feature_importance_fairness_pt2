## Avaliação de Discriminação em ML usando SHAP

Preconceitos presentes na sociedade podem criar vieses em modelos aprendidos a partir de dados. Para avaliar a existência de viés, alguns pesquisadores propõem o uso de definições de "justiça", enquanto outros usam técnicas de interpretabilidade. Porém, parece não existir nenhum estudo que compara as medidas de justiça (através de várias definições de justiça) e os resultados de interpretabilidade (através de várias noções de interpretabilidade). Esse repositório contém a implementação de metodologias para examinar e comparar esses resultados, ele é uma continuação do que foi feito [nesse outro repositório](https://github.com/cesarojuliana/feature_importance_fairness). A ideia da metodologia implementada é avaliar como as medidas de justiça e o resultado de interpretabilidade variam em um modelo com viés e em outro sem viés. Focamos no uso do SHAP (SHapley Additive exPlanations) como técnica de interpretabilidade, que usa conceito da teoria dos jogos cooperativos para calcular a contribuição de cada feature em uma previsão gerada pelo modelo; apresentamos resultados com alguns datasets propensos a injustiça.

Nós usamos as seguintes medidas de justiça: paridade estatística, igualdade de oportunidade, consistência e índice de entropia generalizado.

Nós usamos 5 datasets: Adult, German, Default, COMPAS e PNAD. Todos foram avaliados com 4 modelos: Random Forest, Gradient Boosting, Regressão Logística e SVM. Usamos as seguintes técnicas do SHAP:

- Kernel SHAP: aplicada em todos os modelos
- Tree SHAP: aplicada nos modelos Random Forest e Gradient Boosting
- Linear SHAP: aplicada no modelo de Regressão Logística

### Estrutura do projeto