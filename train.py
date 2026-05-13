"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Optional
import evaluate     

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion  = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # 1. Convert raw logits to log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # 2. Create smoothed target distributions
        true_dist = torch.zeros_like(log_probs)

        # 3. Fill with uniform smoothed probability
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))

        # 4. Scatter the high-confidence mass onto the correct class
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # 5. Zero out the probability for the padding index
        true_dist[:, self.pad_idx] = 0.0

        # 6. Zero out entire rows where the target itself is padding
        pad_mask = (target == self.pad_idx).unsqueeze(1)
        true_dist.masked_fill_(pad_mask, 0.0)

        # 7. Compute KL-divergence loss
        return self.criterion(log_probs, true_dist)


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss   = 0.0
    total_tokens = 0

    batches_per_epoch = len(data_iter)
    mode_str          = "Train" if is_train else "Val"
    pbar              = tqdm(data_iter, desc=f"Epoch {epoch_num} [{mode_str}]", leave=False)

    for step, (src, tgt) in enumerate(pbar):

        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher-forcing slices
        tgt_input = tgt[:, :-1]   # feed: <sos> … token_{n-1}
        tgt_y     = tgt[:, 1:]    # gold: token_1 … <eos>

        src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

        # ── Forward ────────────────────────────────────────────────
        logits = model(src, tgt_input, src_mask, tgt_mask)

        loss = loss_fn(
            logits.contiguous().view(-1, logits.size(-1)),
            tgt_y.contiguous().view(-1),
        )

        # ── Backward (train only) ──────────────────────────────────
        if is_train:
            optimizer.zero_grad()
            loss.backward()

            global_step = epoch_num * batches_per_epoch + step

            log_metrics = {
                "train/batch_loss":   loss.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "global_step":        global_step,
            }

            # Prediction-confidence logging (Section 2.5)
            with torch.no_grad():
                probs = F.softmax(
                    logits.contiguous().view(-1, logits.size(-1)), dim=-1
                )
                target_flat  = tgt_y.contiguous().view(-1)
                target_probs = probs.gather(1, target_flat.unsqueeze(-1)).squeeze(-1)
                valid_probs  = target_probs[target_flat != PAD_IDX]
                if valid_probs.numel() > 0:
                    log_metrics["train/prediction_confidence"] = valid_probs.mean().item()

            # Gradient-norm logging for W_q / W_k (Section 2.2, first 1 000 steps)
            if global_step < 1000:
                try:
                    q_grad = model.encoder.layers[0].self_attn.W_q.weight.grad
                    k_grad = model.encoder.layers[0].self_attn.W_k.weight.grad
                    if q_grad is not None and k_grad is not None:
                        log_metrics["gradients/W_q_norm"] = q_grad.norm().item()
                        log_metrics["gradients/W_k_norm"] = k_grad.norm().item()
                except AttributeError:
                    pass

            wandb.log(log_metrics)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # ── Accumulate metrics ─────────────────────────────────────
        non_pad_tokens = (tgt_y != PAD_IDX).sum().item()
        total_loss   += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

        postfix = {"Batch Loss": f"{loss.item():.4f}"}
        if optimizer is not None:
            postfix["LR"] = f"{optimizer.param_groups[0]['lr']:.6f}"
        pbar.set_postfix(postfix)

    # ── Epoch-level W&B logging ────────────────────────────────────
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    split_key = "train" if is_train else "val"
    wandb.log({f"{split_key}/epoch_loss": avg_loss, "epoch": epoch_num})

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)

        with torch.no_grad():
            logits = model.decode(memory, src_mask, ys, tgt_mask)

        next_word = logits[:, -1, :].argmax(dim=-1).item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).fill_(next_word).long().to(device)], dim=1
        )

        if next_word == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Dict mapping int index → token string  (tgt_itos).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()

    hypotheses = []
    references = []

    translation_table = wandb.Table(columns=["Target (Ground Truth)", "Prediction (Model Output)"])
    log_limit = 10
    log_count = 0

    pbar = tqdm(test_dataloader, desc="Evaluating BLEU", leave=False)

    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)

        for i in range(src.size(0)):
            src_seq = src[i].unsqueeze(0)
            tgt_seq = tgt[i].unsqueeze(0)

            src_mask = make_src_mask(src_seq, pad_idx=PAD_IDX).to(device)

            pred_indices = greedy_decode(
                model, src_seq, src_mask, max_len,
                start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device,
            )

            pred_tokens = []
            for idx in pred_indices[0].tolist():
                if idx in (PAD_IDX, SOS_IDX):
                    continue
                if idx == EOS_IDX:
                    break
                pred_tokens.append(tgt_vocab.get(idx, "<unk>"))

            tgt_tokens = []
            for idx in tgt_seq[0].tolist():
                if idx in (PAD_IDX, SOS_IDX):
                    continue
                if idx == EOS_IDX:
                    break
                tgt_tokens.append(tgt_vocab.get(idx, "<unk>"))

            pred_str = " ".join(pred_tokens)
            tgt_str  = " ".join(tgt_tokens)

            hypotheses.append(pred_str)
            references.append([tgt_str])          # sacrebleu expects list-of-lists

            if log_count < log_limit:
                translation_table.add_data(tgt_str, pred_str)
                log_count += 1

    if wandb.run is not None:
        wandb.log({"test/sample_translations": translation_table})


    bleu_metric = evaluate.load("sacrebleu")
    results = bleu_metric.compute(predictions=hypotheses, references=references)

    return float(results["score"])


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.

    Vocab dicts are pulled from model attributes (set during training)
    so the autograder contract signature is preserved exactly.

    Saved keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config',
        'src_stoi', 'tgt_stoi', 'src_itos', 'tgt_itos',
        'src_vocab', 'tgt_vocab'
    """
    model_config = {
        "src_vocab_size": len(model.src_stoi),
        "tgt_vocab_size": len(model.tgt_stoi),
        "d_model": model.d_model,
        "N": len(model.encoder.layers),
        "num_heads": model.encoder.layers[0].self_attn.num_heads,
        "d_ff": model.encoder.layers[0].ffn.linear1.out_features,
        "dropout": model.dropout_rate,
    }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config,
            # Vocab — loaded by Transformer.__init__ in autograder mode
            "src_stoi": model.src_stoi,
            "tgt_stoi": model.tgt_stoi,
            "src_itos": model.src_itos,
            "tgt_itos": model.tgt_itos,
            "src_vocab": model.src_vocab,
            "tgt_vocab": model.tgt_vocab,
        },
        path,
    )
    print(f"[checkpoint] Saved epoch {epoch} → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0)


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1.  Init W&B
        2.  Build dataset / vocabs
        3.  Create DataLoaders for train / val / test
        4.  Instantiate Transformer  (checkpoint_path="SKIP")
        5.  Inject vocab onto model for save_checkpoint
        6.  Instantiate Adam  (β1=0.9, β2=0.98, ε=1e-9)
        7.  Instantiate NoamScheduler  (warmup_steps=4000)
        8.  Instantiate LabelSmoothingLoss  (ε=0.1)
        9.  Training loop with best-BLEU checkpoint saving
        10. Final BLEU on test set
        11. Attention-head heatmap logged to W&B
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    experiment_config = {
        "d_model":              512,
        "num_heads":            8,
        "d_ff":                 2048,
        "N":                    6,
        "dropout":              0.1,
        "batch_size":           128,
        "num_epochs":           10,
        "warmup_steps":         4000,
        "label_smoothing":      0.1,
        "use_noam_scheduler":   True,
        "use_scaling_factor":   True,
        "positional_encoding":  "sinusoidal",
    }

    wandb.init(
        project="DA6401_Assignment_03",
        name=(
            f"run_noam_{experiment_config['use_noam_scheduler']}"
            f"_ls_{experiment_config['label_smoothing']}"
        ),
        config=experiment_config,
    )

    # ── Datasets ──────────────────────────────────────────────────────
    print("Preparing datasets …")
    train_data = Multi30kDataset(split="train", min_freq=2)
    train_data.build_vocab()
    train_data.process_data()

    val_data = Multi30kDataset(split="validation")
    val_data.load_vocab(
        train_data.src_stoi, train_data.tgt_stoi,
        train_data.src_itos, train_data.tgt_itos,
        train_data.src_vocab, train_data.tgt_vocab,
    )
    val_data.process_data()

    test_data = Multi30kDataset(split="test")
    test_data.load_vocab(
        train_data.src_stoi, train_data.tgt_stoi,
        train_data.src_itos, train_data.tgt_itos,
        train_data.src_vocab, train_data.tgt_vocab,
    )
    test_data.process_data()

    train_loader = train_data.get_dataloader(
        batch_size=experiment_config["batch_size"], shuffle=True
    )
    val_loader  = val_data.get_dataloader(
        batch_size=experiment_config["batch_size"], shuffle=False
    )
    test_loader = test_data.get_dataloader(
        batch_size=experiment_config["batch_size"], shuffle=False
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("Building Transformer …")
    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=experiment_config["d_model"],
        N=experiment_config["N"],
        num_heads=experiment_config["num_heads"],
        d_ff=experiment_config["d_ff"],
        dropout=experiment_config["dropout"],
        checkpoint_path="SKIP",   # bypass gdown during training
    ).to(device)

    # Inject vocab onto model so save_checkpoint can read it without
    # needing a dataset argument (preserves autograder contract).
    model.src_stoi  = train_data.src_stoi
    model.tgt_stoi  = train_data.tgt_stoi
    model.src_itos  = train_data.src_itos
    model.tgt_itos  = train_data.tgt_itos
    model.src_vocab = train_data.src_vocab
    model.tgt_vocab = train_data.tgt_vocab

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    base_lr   = 1.0 if experiment_config["use_noam_scheduler"] else 1e-4
    optimizer = optim.Adam(
        model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = None
    if experiment_config["use_noam_scheduler"]:
        scheduler = NoamScheduler(
            optimizer,
            d_model=experiment_config["d_model"],
            warmup_steps=experiment_config["warmup_steps"],
        )

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_data.tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=experiment_config["label_smoothing"],
    ).to(device)

    # ── Training Loop ─────────────────────────────────────────────────
    best_val_bleu  = 0.0
    checkpoint_path = "best_model.pth"

    for epoch in range(experiment_config["num_epochs"]):
        print(f"\n--- Epoch {epoch} ---")

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler, epoch,
            is_train=True, device=device,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                val_loader, model, loss_fn,
                None, None, epoch,
                is_train=False, device=device,
            )
            val_bleu = evaluate_bleu(
                model, val_loader, train_data.tgt_itos, device=device
            )
            wandb.log({"val/bleu": val_bleu, "epoch": epoch})
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val BLEU: {val_bleu:.2f}"
            )

        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            print(f"New best val BLEU {val_bleu:.2f} — saving → {checkpoint_path}")
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

    # ── Final Test Evaluation ──────────────────────────────────────────
    print("\nTraining complete. Loading best model for test evaluation …")
    load_checkpoint(checkpoint_path, model)

    test_bleu = evaluate_bleu(model, test_loader, train_data.tgt_itos, device=device)
    print(f"Final Test BLEU: {test_bleu:.2f}")
    wandb.log({"test/final_bleu": test_bleu})

    # ── Attention Heatmaps (Section 2.3) ──────────────────────────────
    print("Generating attention heatmaps …")
    sample_sentence = "ein kleiner hund rennt über das gras."
    translation     = model.infer(sample_sentence)
    print(f"Source:      {sample_sentence}")
    print(f"Translation: {translation}")

    # attn_weights stored by MHA after each forward pass
    attn_weights = model.encoder.layers[-1].self_attn.attn_weights
    attn_weights = attn_weights.squeeze(0).cpu().numpy()   # [num_heads, seq_q, seq_k]

    num_heads = attn_weights.shape[0]
    cols      = 4
    rows      = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    for head_idx, ax in enumerate(axes.flatten()):
        if head_idx < num_heads:
            im = ax.imshow(attn_weights[head_idx], cmap="viridis", aspect="auto")
            ax.set_title(f"Head {head_idx + 1}")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            fig.colorbar(im, ax=ax)
        else:
            ax.axis("off")   # hide unused subplot panels

    plt.tight_layout()
    wandb.log({"attention_maps/encoder_last_layer": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()