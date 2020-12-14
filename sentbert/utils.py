def encode_label(label: str) -> int:
    if label == 'negative':
        return 0
    elif label == 'neutral':
        return 1
    elif label == 'positive':
        return 2
    else:
        raise ValueError('Label {} not supported!'.format(label))

def decode_label(label: int) -> str:
    if label == 0:
        return 'negative'
    elif label == 1:
        return 'neutral'
    elif label == 2:
        return 'positive'
    else:
        raise ValueError('Label {} not supported!'.format(label))