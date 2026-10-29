import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Tokenisation adaptee au code.

    Les accents sont retires et chaque identifiant est garde entier puis
    decoupe : `getUserById` -> getuserbyid, get, user, by, id ;
    `parse_files` -> parse_files, parse, files.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = []
    for word in re.findall(r"\w+", text):
        tokens.append(word.lower())
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ").split()
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens


class BM25Index:
    """Index BM25 Okapi (k1=1.5, b=0.75) sur le texte des noeuds d'un projet."""

    def __init__(self, ids, texts):
        self.ids = list(ids)
        self.texts = dict(zip(self.ids, texts))
        self.bm25 = BM25Okapi([tokenize(text) for text in texts]) if self.ids else None

    def search(self, query, limit):
        """[(node_id, score BM25)] des noeuds ayant au moins un terme en commun avec la requete."""
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:limit]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
