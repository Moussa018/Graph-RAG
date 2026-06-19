import spacy
from typing import List, Dict, Any

class DynamicExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            raise RuntimeError("Le modèle français est introuvable. Exécutez : python -m spacy download fr_core_news_sm")
    def analyse_document(self,doc_id: str, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)

        entities = set()
        relations  = []
        
        #extraction entitées
        for ent in doc.ents:
            entities.add(ent.text)
        
        #Extraction Relations logiques 
        for token in doc :
            if token.pos_ == "VERB":
                sujets = [w for w in token.children if w.dep_ in ("nsubj", "nsubj:pass")]
                objets = [w for w in token.children if w.dep_ in ("obj", "obl", "iobj")]
                for sujet in sujets :
                    for objet in objets:
                        relation_type = token.lemma_.upper()
                        