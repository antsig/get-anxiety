import json
from tensorflow.keras.preprocessing.text import Tokenizer

# Fungsi untuk memuat tokenizer dari file JSON yang sudah ada
def load_tokenizer_from_json(filepath="tokenizer.json"):
    try:
        with open(filepath, "r") as f:
            tokenizer_json = json.load(f)

        # Validasi struktur file JSON
        if "config" not in tokenizer_json or "word_index" not in tokenizer_json:
            raise ValueError("File JSON tidak memiliki struktur yang benar.")

        config = tokenizer_json.get("config", {})
        tokenizer = Tokenizer(
            num_words=config.get("num_words"),
            filters=config.get("filters"),
            lower=config.get("lower"),
            split=config.get("split"),
            char_level=config.get("char_level"),
            oov_token=config.get("oov_token"),
        )
        tokenizer.word_index = tokenizer_json.get("word_index", {})
        return tokenizer

    except json.JSONDecodeError:
        raise ValueError("File JSON tidak dapat dibaca dengan benar.")
    except FileNotFoundError:
        raise ValueError("File JSON tidak ditemukan.")
    except Exception as e:
        raise ValueError(f"Terjadi kesalahan saat memuat tokenizer: {str(e)}")
