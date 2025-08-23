# Do nanoGPT ao nanoMoE

Apesar da alta dos LLMs gigantes, os modelos menores e abertos são extremamente úteis pra testar ideias e entender como as coisas funcionam. Um exemplo é o nanoGPT do Karpathy, dá para explorar o código, testar cada parâmetro e entender como cada coisa funciona (sem precisar de uma infinidade de GPU mas se desejar ainda dá pra usar uma T4 do colab). Nesta trilha do nanoMoE, vamos iniciar com um treino simples em pt-br no nanoGPT para explorar o código e estrutura do modelo antes de prosseguir para o MoE.

## Por que testar um modelo pequeno?

- Treinamento rápido
- Entender como a cada época o modelo passa a gerar textos mais alinhados ao treinamento
- Alterar um parâmetro por vez e entender como ele altera o comportamento do modelo
- Exercitar a intuição antes de escalar

Essas etapas foram muito importantes para alguns experimentos posteriores do nanoMoE.

## Estrutura do script

Fluxo básico:
1. Tokeniza um único texto
   1. nesse caso estamos usando um único texto pois nosso objetivo é testar se o modelo é capaz de aprender um único exemplo antes de escalar para um treinamento mais difícil, com muitos exemplos
2. Cria os alvos deslocando 1 token
   1. lembre-se que na tarefa de NTP next-token-prediction o modelo aprende prevendo o próximo token
3. Treino em algumas épocas
4. A cada 25%: imprime loss, perplexity, reconstrução e continuação gerada
   1. Reconstrução - previsão do modelo para cada token do treino
   2. Continuação - o modelo recebe um conjunto de tokens e gera alguns tokens a partir dele

## Parâmetros principais

Trecho (não o arquivo completo):

```python
# --- Training params ---
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

## Preparando dados

```python
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
targets = input_ids.clone()
targets[:, :-1] = input_ids[:, 1:]
targets[:, -1] = -1  # ignora último na loss
```

shift para previsão do próximo token

## Config mínima do modelo

```python
config = GPTConfig(
	block_size=100,
	vocab_size=tokenizer.vocab_size,
	n_layer=4,
	n_head=4,
	n_embd=128,
	k=1,          # MoE desligado (top-k routing futuro)
	dropout=0.2,
	bias=True,
    use_moe=False, # Desativa MoE e utiliza MLP's já implementadas do nanoGPT
)
model = GPT(config).to(device)
```

Teste valores diferentes com n_layer / n_head / n_embd para sentir capacidade.
Note que os parâmetros irão começar a escalar à medida que esses valores aumentam.

## Teste inicial

```python
with torch.no_grad():
	logits, loss = model(input_ids, targets=targets)
print("Loss inicial:", loss.item())
```

Geração inicial (prefixo curto):

```python
gen_init = model.generate(
	input_ids[:, :CONT_PREFIX_TOKENS],
	max_new_tokens=10,
	temperature=0.1,
	top_k=1,
)
```

temperature baixa + top_k=1 ≈ quase greedy.

## Loop de treino

```python
for epoch in range(EPOCHS):
	optimizer.zero_grad()
	_, loss = model(input_ids, targets=targets)
	loss.backward()
	optimizer.step()
	if is_quarter(epoch + 1):
		# imprime métricas + geração
```

Checkpoint a cada 25% do treino:

```python
def is_quarter(e):
	return e == EPOCHS or e % (EPOCHS // 4) == 0
```

## Métricas

- Loss: cross-entropy por token
- Perplexity: exp(loss) - note que no início do treino temos uma perplexidade próxima ao tamanho do vocabulário, isto é normal já que com os pesos aleatórios o modelo tem aproximadamente a mesma probabilidade de gerar qualquer token do vocabulário
- Model prediction: reconstrução do texto do treinamento
- Generated continuation: gera só com uma parte dos tokens, parâmetro CONT_PREFIX_TOKENS

## Experimentos rápidos

1. Capacidade vs velocidade: subir n_embd e medir epochs até PPL < 2
2. Dropout zero: comparar convergência
3. Prefixo curto (3 tokens): ver deriva
4. LR alto (5e-3): observar oscilação
5. Texto maior: concatenar 3 parágrafos e ajustar profundidade


## Próximo post

Vamos adicionar a camada MoE: gate, top‑k, métricas de balanceamento e impacto na perplexity.

---

Código completo: veja `scripts/01_train_nanogpt.py` no repositório.

## Execução direta

```bash
pip install uv
uv sync
uv run -m scripts.01_train_nanogpt
```