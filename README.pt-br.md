# MICUT Biometry Deep: Documentação do Projeto

## Visão Geral do Projeto

O projeto **MICUT Biometry Deep** é uma pipeline de software desenhada para analisar e avaliar a qualidade de imagens de impressões digitais. O sistema recebe imagens brutas de impressões digitais como entrada e gera um arquivo CSV com métricas de qualidade detalhadas para cada impressão digital detectada.

A pipeline é estruturada para realizar uma análise em múltiplos estágios, que incluem:

1.  **Pré-processamento de Imagens**: Unifica e transforma as imagens de entrada em um formato mais adequado para a análise.
2.  **Segmentação de Impressões Digitais**: Utiliza um modelo de machine learning para detectar e isolar as impressões digitais individualmente a partir das imagens pré-processadas.
3.  **Avaliação de Qualidade**: Emprega um *ensemble* de deep learning para avaliar a qualidade de cada impressão digital segmentada com base em diversas métricas.
4.  **Geração de Resultados**: Salva os resultados, incluindo os scores de qualidade e as imagens recortadas das impressões digitais, na pasta `data/output`.

## Estrutura de Pastas

O projeto está organizado nos seguintes diretórios:

*   **`bin/`**: Contém o modelo principal de detecção de impressões digitais (`best_detector_model_v2.pth`).
*   **`data/`**: Esta pasta é dividida em `input` e `output`.
    *   `data/input/Fingerprints/`: Local onde você deve colocar as imagens brutas de impressões digitais (arquivos `.bmp`) que deseja analisar.
    *   `data/output/`: Armazena todos os arquivos gerados pela pipeline, incluindo o relatório final em CSV, as imagens recortadas e outros arquivos intermediários.
*   **`micut_deep/`**: O coração do projeto, contendo todo o código-fonte em Python.
*   **`resources/`**: Contém os modelos utilizados para a avaliação de qualidade via deep learning (`model_densenet121.pt` e `pca_fusion_model.pkl`).

## Explicação do Código (Pasta `micut_deep/`)

A seguir, uma análise detalhada de cada um dos arquivos Python na pasta `micut_deep`:

### 1. `pipeline.py`

Este é o script principal que orquestra todo o fluxo de trabalho. Ao executar `python -m micut_deep.pipeline`, a função `main()` deste arquivo é chamada.

Como funciona:

1.  **Configuração**: Primeiramente, a função `setup_directories()` é chamada para criar todas as pastas de saída necessárias, caso ainda não existam.
2.  **Estágio 1: Criação de Colunas**: A função `create_columns_from_cuts()` do arquivo `segmentation.py` é chamada para unificar as imagens de entrada em imagens de "coluna". Este é um pré-processamento para preparar as imagens para o modelo de segmentação.
3.  **Estágio 2: Segmentação com Modelo de ML**: As imagens de coluna são passadas para a função `segment_columns_with_ml()` (também em `segmentation.py`), que utiliza o modelo `best_detector_model_v2.pth` para detectar e recortar as impressões digitais individualmente.
4.  **Estágio 3: Avaliação de Qualidade com Deep Learning**: As impressões digitais recortadas são então passadas para a função `compute_deep_quality_for_crops()`, que utiliza a classe `DeepEnsemble` de `deep_ensemble.py` para calcular uma série de scores de qualidade.
5.  **Resultado Final**: Por fim, o script unifica os resultados dos estágios de segmentação e avaliação de qualidade em um único DataFrame (utilizando a biblioteca Polars) e o salva como `deep_quality.csv` na pasta `data/output`.

### 2. `segmentation.py`

Este arquivo é responsável pelos dois primeiros estágios da pipeline: a criação das imagens de coluna e a segmentação das impressões digitais.

*   **`create_columns_from_cuts()`**: Esta função lê todos os arquivos `.bmp` da pasta de entrada, agrupa-os por ID de pessoa e os unifica em imagens de coluna verticais (uma para cada mão). O objetivo é criar um formato de entrada consistente para o modelo de segmentação.
*   **`segment_columns_with_ml()`**: É aqui que a segmentação por machine learning ocorre. A função carrega o modelo pré-treinado Faster R-CNN (`best_detector_model_v2.pth`) e o utiliza para detectar as caixas delimitadoras (bounding boxes) das impressões digitais nas imagens de coluna. Para cada impressão digital detectada, a função recorta a imagem, a salva na pasta `data/output/crops` e também cria e salva uma máscara correspondente.

### 3. `deep_ensemble.py`

Este arquivo contém o código para o estágio final da pipeline: a avaliação de qualidade. Ele utiliza um "deep ensemble", que é uma combinação de um modelo de deep learning e um modelo de machine learning tradicional (PCA), para produzir um score de qualidade robusto.

*   **`DeepEnsemble` (classe)**: Ao ser inicializada, esta classe carrega dois modelos da pasta `resources`:
    *   `model_densenet121.pt`: Um modelo de deep learning (DenseNet) que prevê quatro métricas de qualidade: `vfq`, `nfq`, `lqm` e `mor`.
    *   `pca_fusion_model.pkl`: Um modelo de PCA (Análise de Componentes Principais) que é usado para unificar as quatro métricas de qualidade em um único score final.
*   **`predict_ensemble()`**: Este método recebe uma imagem de impressão digital recortada, a pré-processa e a envia para o modelo DenseNet para obter as quatro previsões de qualidade.
*   **`fusion()`**: Este método recebe as quatro previsões do modelo DenseNet e utiliza o modelo de PCA para combiná-las em um único score de qualidade final, que varia de 0 a 100.

## Resumo do Fluxo de Trabalho

Em resumo, o projeto funciona da seguinte maneira:

1.  Você coloca as imagens de impressão digital na pasta `data/input/Fingerprints`.
2.  Você executa a pipeline a partir do seu terminal.
3.  A pipeline unifica as impressões digitais em colunas, detecta as impressões digitais individuais nas colunas e as recorta.
4.  Para cada impressão digital recortada, o sistema calcula quatro métricas de qualidade e, em seguida, as unifica em um único score.
5.  Os resultados finais, incluindo os scores de qualidade e os caminhos para as imagens recortadas, são salvos em um arquivo CSV na pasta `data/output`.
