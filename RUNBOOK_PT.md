# Rodando o benchmark Elixir vs Python
Este projeto treina modelos com os datasets MNIST e CIFAR-10 usando Elixir (Nx, Axon, EXLA) e Python (Keras, TensorFlow) em Docker, numa GPU NVIDIA, e amostra CPU, RAM, GPU e VRAM.

Este documento mostra como chegar em uma primeira execução com sucesso. Foi testado em uma máquina com a seguinte configuração:
- processador Intel i5-6400 (4 cores, sem hyperthreading)
- 7.7GB de RAM; 2GB de swap
- placa de vídeo RTX 3060 Ti (8GB de VRAM, 8.6 de capacidade de compute)
- Ubuntu 24.04
- driver NVIDIA 610.57.04
- Docker 29.1.3
- Docker Compose 2.40.3
- NVIDIA Container Toolkit 1.20.1

## 1. Checagem
Rode o script the checagem na raiz do projeto. O script não fará alteração alguma no computador ou projeto. PASS, WARN ou FAIL será printado por checagem, juntamente com o motivo e um fix, e por fim, os valores que precisão ser setados (útil na [seção 3](#3-adapte-os-arquivos)). No caso de qualquer FAIL, o script retornará 1. É necessário consertar todos os FAILs antes de continuar.

## 2. Configuração do host
1. **Driver:** Instale um driver NVIDIA onde CUDA seja 13.3 ou superior. Rode `nvidia-smi` e encontre `CUDA Version: 13.3` ou superior no resultado. Como este projeto usa containers, o host não precisa de CUDA Toolkit, cuDNN, NCCL nem NVSHMEM.
2. **Docker:** Instale os pacotes Ubuntu `docker.io`, `docker-compose-v2` e `docker-buildx`. Adicione seu usuário ao grupo `docker` (`sudo usermod -aG docker $USER`) e se logue novamente.
3. **NVIDIA Container Toolkit:** Instale do [repositório apt da NVIDIA][3] e então `sudo nvidia-ctk runtime configure --runtime=docker` e `sudo systemctl restart docker`.
4. **Cheque:** `tools/preflight.sh` deve retornar FAIL nenhum.

Para outras distribuições: os drivers, Docker e toolkit vão ser diferentes dependendo do gerenciador de pacotes. NVIDIA possui [documentação para as principais distribuições][3]. Siga a documentação oficial e rode o script de checagem.

## 3. Adapte os arquivos
Estes são dependentes da configuração de cada máquina, sendo necessário sua edição no caso de diferenças.

**Base image tag**, linha 3 de `docker/Dockerfile.elixir` e linha 3 de `docker/Dockerfile.python` (ambos precisam ser idênticos):
```
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04
```
A imagem carrega `NVIDIA_REQUIRE_CUDA=cuda>=13.3` forçando o host a ter CUDA 13.3 ou superior. CUDA 13.2 (do driver 595.91.07) impediu o container de iniciar. O Dockerfile Elixir também possui configurações atreladas a CUDA 13.3: `libnvshmem3-cuda-13`, os caminhos `nvshmem/13` and `libnvrtc-builtins.so.13.3`.

**`XLA_TARGET`**, linha 57 de `docker/Dockerfile.elixir`:
```
ENV XLA_TARGET=cuda13
```
Use `cuda13` pra imagem CUDA 13 ou `cuda12` para imagem CUDA 12.1 ou superior (esperado que funcione segundo [documentação][4]).

**`mem_limit` e `memswap_limit`**, linhas 6 e 7 do `docker-compose.yml`:
```
mem_limit: 6g
memswap_limit: 6g
```
Mante-los iguais impede swap. Desta forma uma falta de memória aparecerá como uma morte do processo ao invés de um benchmark devagar.

**`MAKEFLAGS`**, linha 58 do `docker/Dockerfile.elixir`:
```
ENV MAKEFLAGS=-j4
```
Afeta apenas a compilação nativa do EXLA. Valores menores são recomendados para máquinas com RAM inferiores a 7GB.

**Display:** desative a interface gráfica com `sudo systemctl isolate multi-user.target`, execute os benchmarks dentro de um `tmux` e reative com `sudo systemctl isolate graphical.target`.

**Datasets:** `datasets/scidata/` (Elixir) e `datasets/keras/datasets/` (Python) são montados no container, e `TMPDIR` e `KERAS_HOME` apontam para lá. Crie estes diretórios se eles não existirem (`mkdir -p datasets/scidata datasets/keras`).

## 4. Primeira execução
Todos os comandos devem ser executados na raiz do projeto. Builde uma imagem por vez pois a compilação do EXLA usa muita RAM. Por conta disto, a imagem Elixir demora para buildar (cerca de 30 minutos).
```bash
tools/preflight.sh
COMPOSE_PARALLEL_LIMIT=1 docker compose build
```
Não rode `docker builder prune` ou `docker system prune -a` após porque isso irá deletar a compilação.

O compose lê o arquivo inteiro a cada comando, então `BATCH` precisa estar definido mesmo nos testes smoke abaixo (qualquer valor serve).

Testes smoke, sem treinamento:
```bash
export BATCH=mine
docker compose run --rm elixir-bench mix run -e 'IO.inspect(Nx.default_backend()); IO.inspect(EXLA.Client.get_supported_platforms())'
docker compose run --rm python-bench python3 -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```
Resultado esperado: `{EXLA.Backend, [client: :cuda]}`; `%{host: 4, cuda: 1}` (host deve contar a quantidade de cores de CPU); `2.21.0`; `PhysicalDevice GPU:0`.

Uma execução curta por stack, cujos resultados são salvos em um diretório separado:
```bash
BATCH=mine RUNS=1 docker compose run --rm elixir-bench
BATCH=mine RUNS=1 docker compose run --rm python-bench
```
`BATCH` é a pasta de resultados dentro de `results/docker/` (obrigatória), `RUNS` o número de execuções (padrão 5) e `DATASET` é `mnist` (padrão) ou `cifar10`; o intervalo de amostragem fica em 0,1 s. Por exemplo `BATCH=batch4 DATASET=cifar10 docker compose run --rm elixir-bench`. Rode apenas um serviço por vez, nunca `docker compose up`.

### Opcional
Arquivo opcional das imagens (`tools/save-images.sh`) que pode ser utilizado em um OS novo facilmente (`tools/load-images.sh <archive.tar.zst>`). Restaurar as imagens levou 6 minutos e 23 segundos.

### Resultados esperados
Tempo de treino, média de 15 execuções Docker, desvio padrão entre parênteses:

|          | Elixir (s)                       | Python (s)   |
| -------- | -------------------------------- | ------------ |
| MNIST    | 7.04 (0.10)                      | 11.25 (0.10) |
| CIFAR-10 | 24.26 (2.73), range 18.9 to 27.0 | 28.25 (0.26) |


## Sources
[1]: https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/ (NVIDIA, "Navigating GPU architecture support")
[2]: https://docs.nvidia.com/cuda/archive/13.3.1/cuda-toolkit-release-notes/index.html (NVIDIA, CUDA Toolkit 13.3 Update 1 release notes)
[3]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html (NVIDIA Container Toolkit install guide: )
[4]: https://github.com/elixir-nx/xla (XLA for Elixir, README)
[5]: https://www.tensorflow.org/install/pip (TensorFlow, pip install)