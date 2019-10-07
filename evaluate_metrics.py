def K_Necessity(pre_softmax_scores, K=0.1, resolution=1):
    original_score = pre_softmax_scores[0]
    score_drop = K * original_score
    pre_softmax_scores -= score_drop
    for i, s in enumerate(pre_softmax_scores):
        if s < 0:
            break

    pixel_removed = i * resolution

    return  score_drop / (pixel_removed + 1e-16)


def K_Sufficiency(pre_softmax_scores, K=0.9, original_score=None,
                  resolution=1):
    if original_score is None:
        original_score = pre_softmax_scores[-1]
    score_increase = original_score * K
    for i, s in enumerate(pre_softmax_scores):
        if s > score_increase:
            break
    pixel_added = i * resolution
    return score_increase / (1e-16 + pixel_added)
