import numpy as np

def diagram_sizes(dgms):
    return ", ".join([f"|$H_{i}$|={len(d)}" for i, d in enumerate(dgms)])

def stda2giotto(persistence_diagram):
    persistence_giotto = []
    for homology_dimension in range(len(persistence_diagram)):
        for homology in persistence_diagram[homology_dimension]:
            persistence_giotto.append([homology[0], homology[1], homology_dimension])
    return np.asarray(persistence_giotto)