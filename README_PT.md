# TCC: benchmark de redes neurais entre Elixir e Python
Reprodução do experimento feito em _Análise do potencial da linguagem Elixir no setor de redes neurais_; TAVANO, Lucas Campos; UTFPR; 2022. Elixir (Nx/Axon/EXLA) e Python (Keras/TensorFlow) treinam os mesmo modelos (MNIST e CIFAR-10) na mesma GPU enquanto CPU, RAM, GPU e VRAM são amostrados.

>Novo aqui? `RUNBOOK_PT.md` ou `RUNBOOK_EN.md` podem te guiar para uma primeira execução.

## Requisitos
- Uma GPU NVIDIA e um driver que reporte CUDA 13.3 ou superior (driver 610+).
- Docker com NVIDIA Container Toolkit em Linux.
- Em torno de 8GB de RAM.
- Espaço em disco para as duas imagens e os datasets.

>`tools/preflight.sh` existe para checar tudo isso. Execute esse script.

## Layout
- `elixir/`: projeto Mix (`mix.exs`, `config/`, `scripts/train_mnist.exs`, `scripts/train_cifar10.exs`).
- `python/`: `train_mnist.py` e `train_cifar10.py`.
- `bench/`: `run_benchmark_elixir.sh` e `run_benchmark_python.sh`, envelopes que executam um script de treino N vezes e amostram recursos usados para CSVs.
- `docker/`: `Dockerfile.elixir`, `Dockerfile.python`, `entrypoint-python.sh`. `docker-compose.yml` se localiza na raiz do projeto.
- `tools/`: `preflight.sh`, `compile_results.py` (relatório XLSX), `plot_results.py` (gráficos em PNG), `fix_csv_locale.py`, and `save-images.sh` / `load-images.sh` (scripts para arquivar e restaurar as imagens construídas).
- `RUNBOOK_EN.md`, `RUNBOOK_PT.md`: passo a passo de como executar o projeto pela primeira vez.
- `datasets/`: datasets cacheados por ambas as stacks. `scidata/` é o cache para Elixir; `keras/datasets/` é o cache para Python.

## O que uma execução faz
Ambas stacks treinam os mesmos modelos com mesmas configurações:

| Dataset  | Optimizer | Loss                      | Epochs | Batch size |
| -------- | --------- | ------------------------- | ------ | ---------- |
| MNIST    | Adam      | categorical cross entropy | 5      | 128        |
| CIFAR-10 | Adam      | categorical cross entropy | 5      | 64         |

O envelope repete o script `RUNS` vezes e amostra a máquina a cada 0.1s. Sob `results/docker/<BATCH>/<elixir ou python>/<script>/` salva, para cada execução `i`:
- `run_<i>_output.log`: a saída do script.
- `run_<i>_stats.csv`: uma linha por amostra (CPU %, RAM, GPU %, VRAM).
- `summary.csv` um por diretório.

## Uso
Construa uma imagem por vez e as arquive para não precisar repetir esta etapa:
```bash
COMPOSE_PARALLEL_LIMIT=1 docker compose build
tools/save-images.sh  # writes ~/docker-cache/nn-bench-images-<date>.tar.zst
# on a fresh OS, after driver + Docker + NVIDIA toolkit:
tools/load-images.sh ~/docker-cache/nn-bench-images-<date>.tar.zst
```
Reconstrua apenas se `docker/` ou `elixir/mix` mudarem.

Execute uma stack de cada vez, escolhendo os parâmetros da execução:
```bash
BATCH=batch4 docker compose run --rm elixir-bench
BATCH=batch4 docker compose run --rm python-bench
BATCH=batch5 DATASET=cifar10 RUNS=10 docker compose run --rm elixir-bench
```
- `BATCH` (obrigatório): diretório final para os resultados em `results/docker/`.
- `DATASET`: `mnist` (default) ou `cifar10`.
- `RUNS`: número de execuções (default 5).

>`BATCH` é lido em todo comando, sendo necessário o setar sempre.

Notas:
- Execute um serviço de cada vez com SSH e `tmux`, sem desktop (`sudo systemctl isolate multi-user.target`).
- Os containers escrevem como root, `sudo chown -R $USER:$USER results datasets` para mudar o dono do arquivo para seu usuário.
- Uso de rede é mínimo, especialmente após o cacheamento.

## Relatório e gráficos
São executados no host num ambiente virtual com pacotes de plotagem:
```bash
python3 -m venv .venv && .venv/bin/pip install pandas openpyxl matplotlib
```

O relatório xlsx recebe um par de `LABEL=folder` por diretório de resultado, sendo possível usar vários resultados:
```bash
.venv/bin/python tools/compile_results.py -o results/docker/report.xlsx \
  python=results/docker/batch1-5runs/python elixir=results/docker/batch1-5runs/elixir \
  python-extra=results/docker/batch2-10runs/python elixir-extra=results/docker/batch2-10runs/elixir
```

Os gráficos (PNG) usam por padrão os resultados em `results/docker/batch*` e usam como diretório de saída `results/plots/`:
```bash
.venv/bin/python tools/plot_results.py
```

Para execuções que não usem `batch*`, é possível passar o caminho do diretório explicitamente. Precisa possuir `elixir/`e `python/`.
```bash
.venv/bin/python tools/plot_results.py results/docker/exp1 results/docker/exp2
```

Opções úteis:
- `-o <folder>`: diretório de saída.

Para as análises de dados usadas no artigo deste experimento, foram criados dois outros scripts que olham as amostragens mais profundamente. Seus resultados são criados em `results/article/`:
```bash
.venv/bin/python tools/article_stats.py
.venv/bin/python tools/make_figures.py

.venv/bin/python tools/article_stats.py results/docker/exp1 results/docker/exp2 --baremetal none
.venv/bin/python tools/make_figures.py results/docker/exp1 results/docker/exp2 --baremetal none -o results/article/figures-exp
```

Por causa dessa seleção padrão, dê às execuções de teste ou smoke um nome de pasta que não comece com `batch` (o runbook usa `BATCH=mine`); elas são então ignoradas, a menos que você as liste. Toda combinação (ambas as stacks, ambos os datasets) precisa de pelo menos uma execução nas pastas listadas, ou o script para com "No runs found".