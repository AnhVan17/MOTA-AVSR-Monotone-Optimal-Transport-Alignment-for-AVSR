"""Train a Vietnamese SentencePiece tokenizer on the train-shard transcripts.

    modal run scripts/modal/data_prep/train_sentencepiece.py --vocab-size 2000

Reads ``.txt`` transcripts from ``/mnt/vicocktail_features/vicocktail-avvn-train-*.tar``, trains a
SentencePiece *unigram* model, and saves ``vi_sp_<N>.model``/``.vocab`` to ``/mnt/tokenizer/``.

Why ~2000 vocab: the full Whisper vocab (51865) makes the decoder 71% of the model with mostly dead
rows for Vietnamese. ViCocktail (same 269h dataset) uses SentencePiece 2057. A compact Vietnamese
vocab cuts ~38M params and suits the low-resource setting.

Control ids are fixed so the tokenizer/CTC wiring is deterministic:
  unk=0, bos=1, eos=2, pad=3, ``<blank>``=4 (CTC blank), real pieces from 5.

After training, download for local use + commit to the repo:
  modal volume get avsr-volume /tokenizer/vi_sp_2000.model assets/tokenizer/
  modal volume get avsr-volume /tokenizer/vi_sp_2000.vocab assets/tokenizer/
"""
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import BARE_IMAGE, get_volume

OUTPUT_ROOT = "/mnt/vicocktail_features"
TOKENIZER_DIR = "/mnt/tokenizer"

app = modal.App("avsr-train-spm")
volume = get_volume()
# ship src/ so the container can re-import this module's top-level `from src.infra...`.
image = BARE_IMAGE.pip_install("sentencepiece==0.2.0").add_local_dir("src", remote_path="/root/src")


@app.function(image=image, volumes={"/mnt": volume}, timeout=3600, cpu=8, memory=16384)
def train_spm(vocab_size: int = 2000, subset: str = "train", max_sentences: int = 60000) -> str:
    import glob
    import os
    import tarfile
    import tempfile

    import sentencepiece as spm

    shards = sorted(glob.glob(f"{OUTPUT_ROOT}/vicocktail-avvn-{subset}-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shards matched subset={subset} in {OUTPUT_ROOT}")
    print(f"Collecting up to {max_sentences} transcripts from {len(shards)} shards...")

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    corpus_path = os.path.join(tempfile.gettempdir(), "vi_corpus.txt")
    n = 0
    done = False
    with open(corpus_path, "w", encoding="utf-8") as out:
        for si, shard in enumerate(shards):
            if done:
                break
            with tarfile.open(shard, "r") as tar:
                for member in tar:
                    if not member.name.endswith(".txt"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    text = f.read().decode("utf-8").strip()
                    if text:
                        out.write(text + "\n")
                        n += 1
                        if n >= max_sentences:
                            done = True
                            break
            print(f"  [{si + 1}/{len(shards)}] {os.path.basename(shard)} → {n} transcripts", flush=True)
    # ~60k Vietnamese sentences is ample for a 2000-piece unigram model (subword stats saturate fast).
    print(f"Collected {n} transcripts (cap {max_sentences}) → {corpus_path}")

    model_prefix = os.path.join(TOKENIZER_DIR, f"vi_sp_{vocab_size}")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,                # Vietnamese diacritics → full coverage
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
        user_defined_symbols=["<blank>"],      # id 4 → CTC blank
    )
    volume.commit()

    # Sanity check on real Vietnamese text.
    sp_model = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    sample = "xin chào việt nam hôm nay trời đẹp"
    ids = sp_model.encode(sample)
    print(f"vocab_size={sp_model.vocab_size()} | <blank> id={sp_model.piece_to_id('<blank>')}")
    print(f"encode('{sample}') = {ids}")
    print(f"decode = '{sp_model.decode(ids)}'")
    return f"saved {model_prefix}.model (vocab={sp_model.vocab_size()}, {n} transcripts)"


@app.local_entrypoint()
def main(vocab_size: int = 2000, subset: str = "train", max_sentences: int = 60000):
    print(train_spm.remote(vocab_size=vocab_size, subset=subset, max_sentences=max_sentences))
