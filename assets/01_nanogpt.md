De qualquer forma mandar aqui por garantia 
---
layout: default
title:  "NanoGPT ao nanoMoE: Part-1 Nano GPT"
date:   2025-08-23 22:03:16 -0300
categories: moe
---

<!-- 
1. Adicionar figuras da arquitetura
2. nanoGPT to nanoMoE
   1. nanoGPT
      1. Explicar o nanoGPT
      2. Justificar a escolha do modelo nanoGPT para o nanoMoE
         1. opensource, sem dependência externa (como HF/TF)
      3. Adicionar explicações dos parâmetros do GPTconfig
      4. Explicar os resultados obtidos
   2. nanoMoE
 -->

<!-- # Do nanoGPT ao nanoMoE
<INTRODUÇÃO SOBRE A SAGA NANO GPT AO NANO MOE> -->

## nanoGPT

NanoGPT é uma implementação do modelo GPT, foi escrito com foco em simplicidade e didática, tornando o código acessível para quem deseja entender ou modificar os detalhes do funcionamento de um transformer. O projeto se destaca por dispensar dependências externas complexas e por oferecer um código limpo e direto: o arquivo train.py contém um loop de treinamento e o modelo GPT em poucas linhas.

O nanoGPT permite que pesquisadores e entusiastas testem ideias, ajustem hiperparâmetros e explorem o funcionamento interno dos transformers sem a necessidade de grandes infraestruturas. O projeto serve como uma base para experimentos e extensões, como a implementação de arquiteturas MoE (Mixture of Experts), tornando-o uma ferramenta valiosa para quem deseja aprender em modelos de linguagem.

Nos próximos posts, vamos detalhar a arquitetura e o processo de treinamento do nanoGPT, mostrando as principais etapas e decisões técnicas envolvidas. Em seguida, serão apresentadas as modificações necessárias para implementar o MoE (Mixture of Experts) e os principais desafios encontrados durante o desenvolvimento dessa extensão.

## Por que testar um modelo pequeno?

- Iteração rápida: treinos em minutos permitem ajustar código e hiperparâmetros sem esperar horas.
- Custo baixo: consome pouca GPU/CPU e evita otimizações prematuras de infraestrutura.
- Diagnóstico claro: overfitting, instabilidade de loss ou bugs aparecem de forma mais visível.
- Intuição incremental: mudar 1 hiperparâmetro (n_layer, n_head, n_embd, dropout) e observar efeito direto em loss/perplexity.
- Sanidade da pipeline: garantir que tokenização, shift de targets e geração funcionam (overfit controlado de um único parágrafo).
- Baseline mínima: estabelece referência para comparar futuras melhorias (ex: adicionar MoE) e medir ganho real.
- Facilidade de depuração: menos parâmetros reduzem ruído ao investigar gradientes ou explosões numéricas.
- Preparação para escalar: entender limites (quando saturar) antes de investir em modelos maiores.

## Entendendo o nanoGPT através dos parâmetros

A seguir está a implementação da classe GPTConfig, que define os principais parâmetros do modelo. Cada parâmetro influencia diretamente a capacidade, desempenho e comportamento do nanoGPT durante o treinamento e a geração de texto.

``` python
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
```

- *block_size*: define o tamanho do contexto processado pelo modelo. Nesta implementação o modelo de atenção é implementada de forma quadrática, diferente de modelos mais recentes, como o Mixtral, utilizam abordagens diferentes como SMoE. Caso queira entender mais sobre atenção [illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- *vocab_size*: quantidade de tokens distintos que o modelo pode representar.
- *n_layer*: número de camadas do transformer, no caso do gpt-2 apenas blocos decoder. Um número maior de camadas permite que o modelo aprenda padrões mais complexos, entretanto, aumenta o custo das operações divido ao md(o(block_size)^2*n_embd)
- *n_head*: número de cabeças de atenção. Divide n_embd em n_head partes; cada cabeça aprende relações distintas e depois são concatenadas. Com maior número de n_head o modelo aprende mais relações paralelamente.
- *n_embd*: dimensão do embedding dos tokens. Embeddings maiores aumentam a capicidade de representação do modelo.
- *dropout*: taxa de dropout aplicada durante o treinamento para evitar overfitting. Valores maiores podem contribuir na generalização em conjunto de dados pequenos, mas em atrasar a convergência em conjuntos de dados maiores. 
- *bias*: controla o uso de bias nas camadas lineares e de normalização. Redes mais recentes não utilizam, tendo em vista que pode trazer pequenas melhorias de desempenho [ref]()


## Como Avaliar um modelo GPT ? Métricas útilizadas

- Loss: cross-entropy por token
<div align="center">

$$
\mathcal{L}_{CE}
= -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

</div>

- Perplexity: exp(loss) - A perplexidade é utilizada pela sua melhor interpretabilidade, o seu valor inicial, antes do treino, é aproximadamente igual o tamanho do vocabulário. Valores próximos de 1 indicam que o modelo memorizou os dados.
### Perplexity

<div align="center">

$$
\mathrm{PPL} = \exp\!\big(\mathcal{L}_{CE}\big)
$$

</div>

## Estrutura do script

Fluxo básico:
1. Tokeniza um único texto: usamos apenas um exemplo para teste de sanidade; o objetivo inicial é verificar se o modelo consegue memorizar antes de escalar para mais dados, caso não consiga, provavelmente há um bug no modelo ou no processamento dos dados. 
2. Cria os alvos deslocando 1 token: tarefa de next-token prediction (cada posição prevê o próximo). A predição do último token é ignorada pela loss. 
3. Treina por algumas épocas: acompanhamos o comportamento da loss e da perplexity até estabilizar.
4. Por padrão, a cada 25% das épocas fazemos o evaluation. Serão exibidos dois logs:
- reconstrução: indica a previsão token a token em relação ao texto de treino.
- continuação: o modelo recebe um trecho do texto de treino e completa até o número de tokens desejados. 

Esse ciclo valida pipeline (tokenização, shift, forward, loss, geração) antes de aumentar complexidade.

## Preparando dados

``` python
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
targets = input_ids.clone()
targets[:, :-1] = input_ids[:, 1:]
targets[:, -1] = -1  # ignora último na loss
```

O tokenizer utilizado é baseado em uma versão em pt-br do GPT-2 [tokenizer](https://huggingface.co/pierreguillou/gpt2-small-portuguese).

## Parâmetros principais de treinamento

``` python
EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.95)
CONT_PREFIX_TOKENS = 10  # tokens de prefixo para geração
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Resumo rápido:
- EPOCHS: mais iterações = memoriza o parágrafo (perplexity cai e estabiliza)
- LR: taxa de aprendizado do modelo, você pode ajustar para valores maiores se quiser acelerar o aprendizado, podendo gerar mais instabilidade, ou permanecer com valores menores.
- WEIGHT_DECAY: weight decay é responsável por gerar um retardo no overfitting
- BETAS: mexe na inércia do AdamW
- CONT_PREFIX_TOKENS: quantos tokens iniciais mantemos para a continuação (se for 10, o modelo recebe 10 tokens e gera até 25 novos tokens, você pode alterar isso)


## Config mínima do modelo

```python
config = GPTConfig(
	block_size=100,
	vocab_size=tokenizer.vocab_size,
	n_layer=4,
	n_head=4,
	n_embd=128,
	dropout=0.2,
	bias=True,
)
model = GPT(config).to(device)
```

Teste valores diferentes com n_layer / n_head / n_embd para sentir capacidade.
Note que os parâmetros irão começar a escalar à medida que esses valores aumentam.

## Teste inicial

```python
with torch.no_grad():
    logits, loss = model(input_ids, targets=targets)
init_ppl = torch.exp(loss).item() if loss is not None else float("nan")
```

Geração inicial (prefixo curto):

```python
generated_ids = model.generate(
	input_ids[:, :CONT_PREFIX_TOKENS],
	max_new_tokens=10,
	temperature=0.1,
	top_k=1,
)
```

## Loop de treino

``` python
for epoch in range(EPOCHS):
	optimizer.zero_grad()
	_, loss = model(input_ids, targets=targets)
	loss.backward()
	optimizer.step()
	if is_quarter(epoch + 1):
		# imprime métricas + geração
```

Checkpoint a cada 25% do treino:

``` python
def is_quarter(e):
	return e == EPOCHS or e % (EPOCHS // 4) == 0
```

## Execução direta

``` bash
pip install uv
uv sync
uv run -m scripts.01_train_nanogpt
```

## Próximo post

Vamos adicionar a camada MoE: gate, top‑k, métricas de balanceamento e impacto na perplexity.


Código completo: veja [scripts/01_train_nanogpt.py](https://github.com/sagui-nlp/nanoGPT-moe/blob/feat/blog-writing/scripts/01_train_nanogpt.py) no repositório.