"""Corpus français synthétique à grammaire probabiliste (reproductible, sans téléchargement).

Il contient des dépendances à courte et longue portée (accord sujet-verbe à travers une relative,
parenthèses à refermer, motif « si ... alors ... »), donc l'attention a quelque chose à apprendre.
Limite : ce n'est PAS du langage naturel ; les résultats servent à comparer des variantes, pas à
prédire les performances sur un vrai corpus.
"""
import random

SUJ = [("le chat", "s"), ("le chien", "s"), ("la petite fille", "s"), ("l'ingénieur", "s"), ("le modèle", "s"),
       ("les chats", "p"), ("les chiens", "p"), ("les enfants", "p"), ("les chercheurs", "p"), ("les modèles", "p")]
VERBE = {"s": ["mange", "observe", "cherche", "apprend", "prédit", "regarde"],
         "p": ["mangent", "observent", "cherchent", "apprennent", "prédisent", "regardent"]}
OBJ = ["une pomme", "le texte", "la lune", "un exemple", "la réponse", "un petit jardin", "le signal"]
REL = {"s": "qui dort", "p": "qui dorment"}
ADV = ["souvent", "toujours", "rarement", "ici", "demain", "lentement"]
COND = ["il pleut", "le vent souffle", "la nuit tombe", "le modèle converge"]
CONS = ["on reste au calme", "on sort dehors", "la perte diminue", "on relit le texte"]


def _phrase(rng):
    suj, nb = rng.choice(SUJ)
    r = rng.random()
    if r < 0.35:
        return f"{suj} {rng.choice(VERBE[nb])} {rng.choice(OBJ)} {rng.choice(ADV)}."
    if r < 0.6:  # accord à travers une relative : « les chats qui dorment mangent »
        return f"{suj} {REL[nb]} {rng.choice(VERBE[nb])} {rng.choice(OBJ)}."
    if r < 0.8:
        return f"si {rng.choice(COND)}, alors {rng.choice(CONS)}."
    inner, nb2 = rng.choice(SUJ)  # parenthèse à refermer
    return f"{suj} {rng.choice(VERBE[nb])} ({inner} {rng.choice(VERBE[nb2])}) {rng.choice(ADV)}."


def make_corpus(n_chars: int = 200_000, seed: int = 0) -> str:
    rng = random.Random(seed)
    out, size = [], 0
    while size < n_chars:
        s = _phrase(rng)
        out.append(s)
        size += len(s) + 1
    return "\n".join(out)
