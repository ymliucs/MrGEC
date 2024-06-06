from typing import Callable, Iterable, List, Optional
import torch
from supar.utils import Field


class MrField(Field):
    r"""
    Defines a datatype together with instructions for converting to :class:`~torch.Tensor`.
    :class:`Field` models common text processing datatypes that can be represented by tensors.
    It holds a :class:`~supar.utils.vocab.Vocab` object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The :class:`Field` object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method.

    Args:
        name (str):
            The name of the field.
        pad_token (str):
            The string token used as padding. Default: ``None``.
        unk_token (str):
            The string token used to represent OOV words. Default: ``None``.
        bos_token (str):
            A token that will be prepended to every example using this field, or ``None`` for no `bos_token`.
            Default: ``None``.
        eos_token (str):
            A token that will be appended to every example using this field, or ``None`` for no `eos_token`.
        lower (bool):
            Whether to lowercase the text in this field. Default: ``False``.
        use_vocab (bool):
            Whether to use a :class:`~supar.utils.vocab.Vocab` object.
            If ``False``, the data in this field should already be numerical.
            Default: ``True``.
        tokenize (function):
            The function used to tokenize strings using this field into sequential examples. Default: ``None``.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(
        self,
        name: str,
        pad: Optional[str] = None,
        unk: Optional[str] = None,
        bos: Optional[str] = None,
        eos: Optional[str] = None,
        lower: bool = False,
        use_vocab: bool = True,
        tokenize: Optional[Callable] = None,
        fn: Optional[Callable] = None,
    ):
        super().__init__(name, pad, unk, bos, eos, lower, use_vocab, tokenize, fn)

    def transform(self, sequences: Iterable[List[str]]) -> Iterable[torch.Tensor]:
        r"""
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (Iterable[List[str]]):
                A list of sequences.

        Returns:
            A list of tensors transformed from the input sequences.
        """

        seqs = []
        for seq in sequences:
            seq = self.preprocess(seq)
            if self.use_vocab:
                seq = [self.vocab[token] for token in seq]
            if self.bos:
                seq = [self.bos_index] + seq
            if self.eos:
                seq = seq + [self.eos_index]
                seqs.append(torch.tensor(seq, dtype=torch.long))
        yield seqs

    def compose(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        r"""
        Composes a batch of sequences into a padded tensor.

        Args:
            batch (Iterable[~torch.Tensor]):
                A list of tensors.

        Returns:
            A padded tensor converted to proper device.
        """
        padded_tensor, ref_nums = pad(batch, self.pad_index)

        return {'padded_tensor': padded_tensor.to(self.device, non_blocking=True), 'ref_nums': ref_nums}


def pad(tensors: List[torch.Tensor], padding_value: int = 0, total_length: int = None, padding_side: str = 'right') -> torch.Tensor:
    ref_nums = [len(l) for l in tensors]
    if isinstance(tensors[0], list):
        tensors = [tensor for l in tensors for tensor in l]
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors) for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return (out_tensor, ref_nums)
