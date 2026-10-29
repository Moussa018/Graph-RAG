class CrossEncoderReranker:
    """Reclasse les candidats en notant chaque paire (question, texte du noeud).

    Modele par defaut : cross-encoder multilingue entraine sur mMARCO (questions
    en francais, code et docstrings souvent en anglais). Charge au premier appel.
    """

    def __init__(self, model_name, max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def rerank(self, query, candidates, top_k):
        """candidates : [(node_id, texte)]. Retourne [(node_id, score)] trie, limite a top_k."""
        if not candidates:
            return []
        scores = self._load().predict([(query, text) for _, text in candidates])
        ranked = sorted(zip((node_id for node_id, _ in candidates), scores), key=lambda x: x[1], reverse=True)
        return [(node_id, float(score)) for node_id, score in ranked[:top_k]]
