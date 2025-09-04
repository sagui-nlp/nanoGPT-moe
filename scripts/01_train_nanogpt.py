import torch
from transformers import AutoTokenizer

from model import GPT, GPTConfig

# --- Training params ---
EPOCHS = 1000
LR = 1e-4
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.95)
# Number of tokens from the start of the sample kept as prefix for continuation
CONT_PREFIX_TOKENS = 10  # e.g. model conditioned on first 10 tokens
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer & single training text (tiny tutorial example)
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")
text = "Narizinho tinha sete anos e morava no sítio de sua avó, Dona Benta. Era uma menina de nariz arrebitado, muito viva e esperta. Passava os dias brincando às beiras do ribeirão, entre peixinhos e histórias que inventava para a sua boneca de pano, a Emília, que já mostrava ter ideias próprias e uma língua afiada como poucas."

# Prepare inputs & shifted targets
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
targets = input_ids.clone()
targets[:, :-1] = input_ids[:, 1:]
targets[:, -1] = -1  # ignore last position in loss
print("input ids:", input_ids, "shape:", input_ids.shape)

# Minimal config (defaults fill the rest). MoE disabling not exposed here.
config = GPTConfig(
    block_size=100,
    vocab_size=tokenizer.vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=128,
    k=1,
    dropout=0.2,
    bias=True,
)

model = GPT(config).to(device)

# Initial forward (sanity check)
with torch.no_grad():
    logits, loss = model(input_ids, targets=targets)
init_ppl = torch.exp(loss).item() if loss is not None else float("nan")
print(f"\n=== Initial | Loss {loss.item():.4f} | PPL {init_ppl:.4f} ===")
# pred_tokens = logits.argmax(dim=-1)
# print("[Next Token prediction]", tokenizer.decode(pred_tokens[0].tolist()))
generated_ids = model.generate(
    input_ids[:, :CONT_PREFIX_TOKENS].clone(),
    max_new_tokens=25,
    temperature=0.1,
    top_k=1,
)
print(
    "[Received Tokens]",
    tokenizer.decode(input_ids[0, :CONT_PREFIX_TOKENS].tolist()),
)
print(
    "[Generated continuation]",
    tokenizer.decode(generated_ids[0, CONT_PREFIX_TOKENS:].tolist()),
)

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay=WEIGHT_DECAY,
    learning_rate=LR,
    betas=BETAS,
    device_type=device,
)


def is_quarter(epoch_idx: int) -> bool:
    """Return True at ~25%, 50%, 75% and final epoch."""
    if epoch_idx == EPOCHS:
        return True
    if epoch_idx % (EPOCHS // 4) == 0:
        return True
    return False


for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()

    current_epoch = epoch + 1
    if is_quarter(current_epoch):
        ppl = torch.exp(loss).item()
        print(
            f"\n=== Epoch {current_epoch:>3}/{EPOCHS} | Loss {loss.item():.4f} | PPL {ppl:.4f} ==="
        )
        model.eval()
        with torch.no_grad():
            # logits, _ = model(input_ids, targets=targets)
            # pred_tokens = logits.argmax(dim=-1)
            # print("[Model prediction]", tokenizer.decode(pred_tokens[0].tolist()))
            generated_ids = model.generate(
                input_ids[:, :CONT_PREFIX_TOKENS].clone(),
                max_new_tokens=25,
                temperature=0.1,
                top_k=1,
            )
            print(
                "[Received Tokens]",
                tokenizer.decode(input_ids[0, :CONT_PREFIX_TOKENS].tolist()),
            )
            print(
                "[Generated continuation]",
                tokenizer.decode(generated_ids[0, CONT_PREFIX_TOKENS:].tolist()),
            )
