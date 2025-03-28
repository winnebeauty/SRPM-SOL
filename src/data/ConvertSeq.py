from esm.utils.constants import esm3 as C


def index_to_sequence(seq_idx):
    """
    将索引列表还原为氨基酸序列，不跳过 <pad>。
    """
    IDX_TO_SEQ = {idx: char for idx, char in enumerate(C.SEQUENCE_VOCAB)}
    sequence = [IDX_TO_SEQ[idx] for idx in seq_idx]  # 忽略 <pad>

    return ''.join(sequence)



def sequence_to_index(sequence):
    """
    将氨基酸序列转换为索引列表。
    """
    SEQ_TO_IDX = {char: idx for idx, char in enumerate(C.SEQUENCE_VOCAB)}
    sequence=[SEQ_TO_IDX[char] for char in sequence]

    return sequence