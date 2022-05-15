import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides

# !!!!!!!!!!!!!!!!!ALL copied from allennlp!!!!!!!!!!!!!!!!!
# from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
__all__ = [
    "SelfAttentiveSpanExtractor", "EndpointSpanExtractor",
    "MaxPoolingSpanExtractor", "BidirectionalEndpointSpanExtractor",
    "SpanExtractorWithSpanWidthEmbedding", "SpanExtractor"
]


class SpanExtractor(nn.Module):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.

    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """

    @overrides
    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ):
        """
        Given a sequence tensor, extract spans and return representations of
        them. Span representation can be computed in many different ways,
        such as concatenation of the start and end spans, attention over the
        vectors contained inside the span, etc.

        # Parameters

        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.

        # Returns

        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        """
        Returns the expected final dimension of the `sequence_tensor`.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the expected final dimension of the returned span representation.
        """
        raise NotImplementedError


class SpanExtractorWithSpanWidthEmbedding(SpanExtractor):
    """
    `SpanExtractorWithSpanWidthEmbedding` implements some common code for span
    extractors which will need to embed span width.
    Specifically, we initiate the span width embedding matrix and other
    attributes in `__init__`, leave an `_embed_spans` method that can be
    implemented to compute span embeddings in different ways, and in `forward`
    we concatenate span embeddings returned by `_embed_spans` with span width
    embeddings to form the final span representations.
    We keep SpanExtractor as a purely abstract base class, just in case someone
    wants to build a totally different span extractor.
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    # Returns
    span_embeddings : `torch.FloatTensor`.
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
    """

    def __init__(
            self,
            input_dim: int,
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            bucket_widths: bool=False, ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._span_width_embedding: Optional[nn.Embedding] = None
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = nn.Embedding(
                num_embeddings=num_width_embeddings,
                embedding_dim=span_width_embedding_dim)
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )

    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ):
        """
        Given a sequence tensor, extract spans, concatenate width embeddings
        when need and return representations of them.
        # Parameters
        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.
        # Returns
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        # shape (batch_size, num_spans, embedding_dim)
        span_embeddings = self._embed_spans(sequence_tensor, span_indices,
                                            sequence_mask, span_indices_mask)
        if self._span_width_embedding is not None:
            # width = end_index - start_index + 1 since `SpanField` use inclusive indices.
            # But here we do not add 1 beacuse we often initiate the span width
            # embedding matrix with `num_width_embeddings = max_span_width`
            # shape (batch_size, num_spans)
            widths_minus_one = span_indices[..., 1] - span_indices[..., 0]

            if self._bucket_widths:
                widths_minus_one = bucket_values(
                    widths_minus_one,
                    num_total_buckets=self.
                    _num_width_embeddings  # type: ignore
                )

            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(
                widths_minus_one)
            span_embeddings = torch.cat(
                [span_embeddings, span_width_embeddings], -1)

        if span_indices_mask is not None:
            # Here we are masking the spans which were originally passed in as padding.
            return span_embeddings * span_indices_mask.unsqueeze(-1)

        return span_embeddings

    def get_input_dim(self) -> int:
        return self._input_dim

    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ) -> torch.Tensor:
        """
        Returns the span embeddings computed in many different ways.
        """
        raise NotImplementedError


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        # TODO(brendanr): Is there some reason why we need repr here? It
        # produces horrible output for simple multi-line error messages.
        return self.message


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.

    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```

    # Parameters

    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    # Returns

    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = (get_range_vector(indices.size(0), get_device_of(indices)) *
               sequence_length)
    for _ in range(len(indices.shape) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.reshape(-1)
    return offset_indices


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(
            size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def batched_index_select(
        target: torch.Tensor,
        indices: torch.LongTensor,
        flattened_indices: Optional[torch.LongTensor]=None, ) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/master/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.

    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    # Returns

    selected_targets : `torch.Tensor`
        A tensor with shape [indices.shape, target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices,
                                                            target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.reshape(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.shape) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.reshape(*selected_shape)
    return selected_targets


def batched_span_select(target: torch.Tensor,
                        spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns segmented spans in the target with respect to the provided span indices.

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.

    # Returns

    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(
        max_batch_span_width, get_device_of(target)).reshape(1, 1, -1)
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = (span_mask & (raw_span_indices < target.size(1)) &
                 (0 <= raw_span_indices))
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).max


def weighted_sum(matrix: torch.Tensor,
                 attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions `(batch_size, num_queries, num_words,
    embedding_dim)`.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - `(batch_size, num_queries, num_words)` (distribution over words for each query)
        - `(batch_size, num_documents, num_queries, num_words)` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    `(batch_size, num_queries, embedding_dim)` and
    `(batch_size, num_documents, num_queries, embedding_dim)` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.ndim == 2 and matrix.ndim == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.ndim == 3 and matrix.ndim == 3:
        return attention.bmm(matrix)
    if matrix.ndim - 1 < attention.ndim:
        expanded_size = list(matrix.shape)
        for i in range(attention.ndim - matrix.ndim + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def masked_softmax(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int=-1,
        memory_efficient: bool=False, ) -> torch.Tensor:
    """
    `F.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        while mask.ndim < vector.ndim:
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = F.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) +
                               tiny_value_of_dtype(result.dtype))
        else:
            masked_vector = vector.masked_fill(
                ~mask, min_value_of_dtype(vector.dtype))
            result = F.softmax(masked_vector, dim=dim)
    return result


class SelfAttentiveSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.
    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.
    Registered as a `SpanExtractor` with name "self_attentive".
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    # Returns
    attended_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """

    def __init__(
            self,
            input_dim: int,
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            bucket_widths: bool=False, ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths, )
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_output_dim(self) -> int:
        if self._span_width_embedding is not None:
            return self._input_dim + self._span_width_embedding.get_output_dim(
            )
        return self._input_dim

    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ) -> torch.FloatTensor:
        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # shape (batch_size, sequence_length, embedding_dim + 1)
        concat_tensor = torch.cat([sequence_tensor, global_attention_logits],
                                  -1)

        concat_output, span_mask = batched_span_select(concat_tensor,
                                                       span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = masked_softmax(span_attention_logits,
                                                span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = weighted_sum(span_embeddings,
                                                span_attention_weights)

        return attended_text_embeddings


class TimeDistributed(nn.Module):
    """
    Given an input shaped like `(batch_size, time_steps, [rest])` and a `Module` that takes
    inputs like `(batch_size, [rest])`, `TimeDistributed` reshapes the input to be
    `(batch_size * time_steps, [rest])`, applies the contained `Module`, then reshapes it back.

    Note that while the above gives shapes with `batch_size` first, this `Module` also works if
    `batch_size` is second - we always just combine the first two dimensions, then split them.

    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional `pass_through` iterable.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module

    @overrides
    def forward(self, *inputs, pass_through: List[str]=None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [
            self._reshape_tensor(input_tensor) for input_tensor in inputs
        ]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        new_size = some_input.shape[:2] + reshaped_outputs.shape[1:]
        outputs = reshaped_outputs.contiguous().reshape(new_size)

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.shape
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.reshape(*squashed_shape)


#######################################################################################3
def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise ConfigurationError(
                'Tensor dims must match for operation "{}"'.format(operation))
        return first_tensor_dim


def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with [`combine_tensors`](./util.md#combine_tensors).
    This function computes the resultant dimension when calling `combine_tensors(combination, tensors)`,
    when the tensor dimension is known.  This is necessary for knowing the sizes of weight matrices
    when building models that use `combine_tensors`.

    # Parameters

    combination : `str`
        A comma-separated list of combination pieces, like `"1,2,1*2"`, specified identically to
        `combination` in `combine_tensors`.
    tensor_dims : `List[int]`
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to `combine_tensors`.
    """
    if len(tensor_dims) > 9:
        raise ConfigurationError(
            "Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    return sum(
        _get_combination_dim(piece, tensor_dims)
        for piece in combination.split(","))


def combine_tensors(combination: str,
                    tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    `combination` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like `"1,2,1+2,3-1"`.

    We allow the following kinds of combinations : `x`, `x*y`, `x+y`, `x-y`, and `x/y`,
    where `x` and `y` are positive integers less than or equal to `len(tensors)`.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the `combination` string.  For example, for the input string `"1,2,1*2"`, the result
    would be `[1;2;1*2]`, as you would expect, where `[;]` is concatenation along the last
    dimension.

    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like `torch.cat([x_tensor, y_tensor, x_tensor * y_tensor])`.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.

    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.

    This function also accepts `x` and `y` in place of `1` and `2` in the combination
    string.
    """
    if len(tensors) > 9:
        raise ConfigurationError(
            "Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    to_concatenate = [
        _get_combination(piece, tensors) for piece in combination.split(",")
    ]
    return torch.cat(to_concatenate, dim=-1)


def _get_combination(combination: str,
                     tensors: List[torch.Tensor]) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            return first_tensor * second_tensor
        elif operation == "/":
            return first_tensor / second_tensor
        elif operation == "+":
            return first_tensor + second_tensor
        elif operation == "-":
            return first_tensor - second_tensor
        else:
            raise ConfigurationError("Invalid operation: " + operation)


def bucket_values(distances: torch.Tensor,
                  num_identity_buckets: int=4,
                  num_total_buckets: int=10) -> torch.Tensor:
    """
    Places the given values (designed for distances) into `num_total_buckets`semi-logscale
    buckets, with `num_identity_buckets` of these capturing single values.

    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].

    # Parameters

    distances : `torch.Tensor`, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: `int`, optional (default = `4`).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : `int`, (default = `10`)
        The total number of buckets to bucket values into.

    # Returns

    `torch.Tensor`
        A tensor of the same shape as the input, containing the indices of the buckets
        the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (
        num_identity_buckets - 1)
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


class BidirectionalEndpointSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Represents spans from a bidirectional encoder as a concatenation of two different
    representations of the span endpoints, one for the forward direction of the encoder
    and one from the backward direction. This type of representation encodes some subtlety,
    because when you consider the forward and backward directions separately, the end index
    of the span for the backward direction's representation is actually the start index.
    By default, this `SpanExtractor` represents spans as
    `sequence_tensor[inclusive_span_end] - sequence_tensor[exclusive_span_start]`
    meaning that the representation is the difference between the the last word in the span
    and the word `before` the span started. Note that the start and end indices are with
    respect to the direction that the RNN is going in, so for the backward direction, the
    start/end indices are reversed.
    Additionally, the width of the spans can be embedded and concatenated on to the
    final combination.
    The following other types of representation are supported for both the forward and backward
    directions, assuming that `x = span_start_embeddings` and `y = span_end_embeddings`.
    `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give `x,y,x*y` as the `combination` parameter to this class.
    The computed similarity function would then be `[x; y; x*y]`, which can then be optionally
    concatenated with an embedded representation of the width of the span.
    Registered as a `SpanExtractor` with name "bidirectional_endpoint".
    # Parameters
    input_dim : `int`, required
        The final dimension of the `sequence_tensor`.
    forward_combination : `str`, optional (default = `"y-x"`).
        The method used to combine the `forward_start_embeddings` and `forward_end_embeddings`
        for the forward direction of the bidirectional representation.
        See above for a full description.
    backward_combination : `str`, optional (default = `"x-y"`).
        The method used to combine the `backward_start_embeddings` and `backward_end_embeddings`
        for the backward direction of the bidirectional representation.
        See above for a full description.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    use_sentinels : `bool`, optional (default = `True`).
        If `True`, sentinels are used to represent exclusive span indices for the elements
        in the first and last positions in the sequence (as the exclusive indices for these
        elements are outside of the the sequence boundary). This is not strictly necessary,
        as you may know that your exclusive start and end indices are always within your sequence
        representation, such as if you have appended/prepended <START> and <END> tokens to your
        sequence.
    """

    def __init__(
            self,
            input_dim: int,
            forward_combination: str="y-x",
            backward_combination: str="x-y",
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            bucket_widths: bool=False,
            use_sentinels: bool=True, ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths, )
        self._forward_combination = forward_combination
        self._backward_combination = backward_combination

        if self._input_dim % 2 != 0:
            raise ConfigurationError(
                "The input dimension is not divisible by 2, but the "
                "BidirectionalEndpointSpanExtractor assumes the embedded representation "
                "is bidirectional (and hence divisible by 2).")

        self._use_sentinels = use_sentinels
        if use_sentinels:
            self._start_sentinel = nn.Parameter(
                torch.randn([1, 1, int(input_dim / 2)]))
            self._end_sentinel = nn.Parameter(
                torch.randn([1, 1, int(input_dim / 2)]))

    def get_output_dim(self) -> int:
        unidirectional_dim = int(self._input_dim / 2)
        forward_combined_dim = get_combined_dim(
            self._forward_combination,
            [unidirectional_dim, unidirectional_dim])
        backward_combined_dim = get_combined_dim(
            self._backward_combination,
            [unidirectional_dim, unidirectional_dim])
        if self._span_width_embedding is not None:
            return (forward_combined_dim + backward_combined_dim +
                    self._span_width_embedding.get_output_dim())
        return forward_combined_dim + backward_combined_dim

    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ) -> torch.FloatTensor:

        # Both of shape (batch_size, sequence_length, embedding_size / 2)
        forward_sequence, backward_sequence = sequence_tensor.split(
            int(self._input_dim / 2), dim=-1)
        forward_sequence = forward_sequence.contiguous()
        backward_sequence = backward_sequence.contiguous()

        # shape (batch_size, num_spans)
        span_starts, span_ends = [
            index.squeeze(-1) for index in span_indices.split(
                1, dim=-1)
        ]

        if span_indices_mask is not None:
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask
        # We want `exclusive` span starts, so we remove 1 from the forward span starts
        # as the AllenNLP `SpanField` is inclusive.
        # shape (batch_size, num_spans)
        exclusive_span_starts = span_starts - 1
        # shape (batch_size, num_spans, 1)
        start_sentinel_mask = (exclusive_span_starts == -1).unsqueeze(-1)

        # We want `exclusive` span ends for the backward direction
        # (so that the `start` of the span in that direction is exlusive), so
        # we add 1 to the span ends as the AllenNLP `SpanField` is inclusive.
        exclusive_span_ends = span_ends + 1

        if sequence_mask is not None:
            # shape (batch_size)
            sequence_lengths = get_lengths_from_binary_sequence_mask(
                sequence_mask)
        else:
            # shape (batch_size), filled with the sequence length size of the sequence_tensor.
            sequence_lengths = torch.ones_like(
                sequence_tensor[:, 0, 0],
                dtype=torch.long) * sequence_tensor.size(1)

        # shape (batch_size, num_spans, 1)
        end_sentinel_mask = (exclusive_span_ends >=
                             sequence_lengths.unsqueeze(-1)).unsqueeze(-1)

        # As we added 1 to the span_ends to make them exclusive, which might have caused indices
        # equal to the sequence_length to become out of bounds, we multiply by the inverse of the
        # end_sentinel mask to erase these indices (as we will replace them anyway in the block below).
        # The same argument follows for the exclusive span start indices.
        exclusive_span_ends = exclusive_span_ends * ~end_sentinel_mask.squeeze(
            -1)
        exclusive_span_starts = exclusive_span_starts * ~start_sentinel_mask.squeeze(
            -1)

        # We'll check the indices here at runtime, because it's difficult to debug
        # if this goes wrong and it's tricky to get right.
        if (exclusive_span_starts < 0).any() or (
                exclusive_span_ends > sequence_lengths.unsqueeze(-1)).any():
            raise ValueError(
                f"Adjusted span indices must lie inside the length of the sequence tensor, "
                f"but found: exclusive_span_starts: {exclusive_span_starts}, "
                f"exclusive_span_ends: {exclusive_span_ends} for a sequence tensor with lengths "
                f"{sequence_lengths}.")

        # Forward Direction: start indices are exclusive. Shape (batch_size, num_spans, input_size / 2)
        forward_start_embeddings = batched_index_select(forward_sequence,
                                                        exclusive_span_starts)
        # Forward Direction: end indices are inclusive, so we can just use span_ends.
        # Shape (batch_size, num_spans, input_size / 2)
        forward_end_embeddings = batched_index_select(forward_sequence,
                                                      span_ends)

        # Backward Direction: The backward start embeddings use the `forward` end
        # indices, because we are going backwards.
        # Shape (batch_size, num_spans, input_size / 2)
        backward_start_embeddings = batched_index_select(backward_sequence,
                                                         exclusive_span_ends)
        # Backward Direction: The backward end embeddings use the `forward` start
        # indices, because we are going backwards.
        # Shape (batch_size, num_spans, input_size / 2)
        backward_end_embeddings = batched_index_select(backward_sequence,
                                                       span_starts)

        if self._use_sentinels:
            # If we're using sentinels, we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with either the start sentinel,
            # or the end sentinel.
            forward_start_embeddings = (
                forward_start_embeddings * ~start_sentinel_mask +
                start_sentinel_mask * self._start_sentinel)
            backward_start_embeddings = (
                backward_start_embeddings * ~end_sentinel_mask +
                end_sentinel_mask * self._end_sentinel)

        # Now we combine the forward and backward spans in the manner specified by the
        # respective combinations and concatenate these representations.
        # Shape (batch_size, num_spans, forward_combination_dim)
        forward_spans = combine_tensors(
            self._forward_combination,
            [forward_start_embeddings, forward_end_embeddings])
        # Shape (batch_size, num_spans, backward_combination_dim)
        backward_spans = combine_tensors(
            self._backward_combination,
            [backward_start_embeddings, backward_end_embeddings])
        # Shape (batch_size, num_spans, forward_combination_dim + backward_combination_dim)
        span_embeddings = torch.cat([forward_spans, backward_spans], -1)

        return span_embeddings


class EndpointSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Represents spans as a combination of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.
    The following types of representation are supported, assuming that
    `x = span_start_embeddings` and `y = span_end_embeddings`.
    `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give `x,y,x*y` as the `combination` parameter to this class.
    The computed similarity function would then be `[x; y; x*y]`, which can then be optionally
    concatenated with an embedded representation of the width of the span.
    Registered as a `SpanExtractor` with name "endpoint".
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    combination : `str`, optional (default = `"x,y"`).
        The method used to combine the `start_embedding` and `end_embedding`
        representations. See above for a full description.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    use_exclusive_start_indices : `bool`, optional (default = `False`).
        If `True`, the start indices extracted are converted to exclusive indices. Sentinels
        are used to represent exclusive span indices for the elements in the first
        position in the sequence (as the exclusive indices for these elements are outside
        of the the sequence boundary) so that start indices can be exclusive.
        NOTE: This option can be helpful to avoid the pathological case in which you
        want span differences for length 1 spans - if you use inclusive indices, you
        will end up with an `x - x` operation for length 1 spans, which is not good.
    """

    def __init__(
            self,
            input_dim: int,
            combination: str="x,y",
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            bucket_widths: bool=False,
            use_exclusive_start_indices: bool=False, ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths, )
        self._combination = combination

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            self._start_sentinel = nn.Parameter(
                torch.randn([1, 1, int(input_dim)]))

    def get_output_dim(self) -> int:
        combined_dim = get_combined_dim(self._combination,
                                        [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [
            index.squeeze(-1) for index in span_indices.split(
                1, dim=-1)
        ]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        if not self._use_exclusive_start_indices:
            if sequence_tensor.size(-1) != self._input_dim:
                raise ValueError(
                    f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                    f"received ({self._input_dim}).")
            start_embeddings = batched_index_select(sequence_tensor,
                                                    span_starts)
            end_embeddings = batched_index_select(sequence_tensor, span_ends)

        else:
            # We want `exclusive` span starts, so we remove 1 from the forward span starts
            # as the AllenNLP `SpanField` is inclusive.
            # shape (batch_size, num_spans)
            exclusive_span_starts = span_starts - 1
            # shape (batch_size, num_spans, 1)
            start_sentinel_mask = (exclusive_span_starts == -1).unsqueeze(-1)
            exclusive_span_starts = exclusive_span_starts * ~start_sentinel_mask.squeeze(
                -1)

            # We'll check the indices here at runtime, because it's difficult to debug
            # if this goes wrong and it's tricky to get right.
            if (exclusive_span_starts < 0).any():
                raise ValueError(
                    f"Adjusted span indices must lie inside the the sequence tensor, "
                    f"but found: exclusive_span_starts: {exclusive_span_starts}."
                )

            start_embeddings = batched_index_select(sequence_tensor,
                                                    exclusive_span_starts)
            end_embeddings = batched_index_select(sequence_tensor, span_ends)

            # We're using sentinels, so we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with the start sentinel.
            start_embeddings = (start_embeddings * ~start_sentinel_mask +
                                start_sentinel_mask * self._start_sentinel)

        combined_tensors = combine_tensors(self._combination,
                                           [start_embeddings, end_embeddings])

        return combined_tensors


def masked_max(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int,
        keepdim: bool=False, ) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate max, assume unmasked parts are already zeros
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate max
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the maximum values.
    """
    replaced_vector = vector.masked_fill(~mask,
                                         min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def get_lengths_from_binary_sequence_mask(
        mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    # Parameters

    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    # Returns

    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)


class MaxPoolingSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Represents spans through the application of a dimension-wise max-pooling operation.
    Given a span x_i, ..., x_j with i,j as span_start and span_end, each dimension d
    of the resulting span s is computed via s_d = max(x_id, ..., x_jd).
    Elements masked-out by sequence_mask are ignored when max-pooling is computed.
    Span representations of masked out span_indices by span_mask are set to '0.'
    Registered as a `SpanExtractor` with name "max_pooling".
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    # Returns
    max_pooling_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is the result of a max-pooling operation.
    """

    def __init__(
            self,
            input_dim: int,
            num_width_embeddings: int=None,
            span_width_embedding_dim: int=None,
            bucket_widths: bool=False, ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths, )

    def get_output_dim(self) -> int:
        if self._span_width_embedding is not None:
            return self._input_dim + self._span_width_embedding.get_output_dim(
            )
        return self._input_dim

    def _embed_spans(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
            sequence_mask: torch.BoolTensor=None,
            span_indices_mask: torch.BoolTensor=None, ) -> torch.FloatTensor:

        if sequence_tensor.size(-1) != self._input_dim:
            raise ValueError(
                f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                f"received ({self._input_dim}).")

        if sequence_tensor.shape[1] <= span_indices.max() or span_indices.min(
        ) < 0:
            raise IndexError(
                f"Span index out of range, max index ({span_indices.max()}) "
                f"or min index ({span_indices.min()}) "
                f"not valid for sequence of length ({sequence_tensor.shape[1]})."
            )

        if (span_indices[:, :, 0] > span_indices[:, :, 1]).any():
            raise IndexError("Span start above span end", )

        # Calculate the maximum sequence length for each element in batch.
        # If span_end indices are above these length, we adjust the indices in adapted_span_indices
        if sequence_mask is not None:
            # shape (batch_size)
            sequence_lengths = get_lengths_from_binary_sequence_mask(
                sequence_mask)
        else:
            # shape (batch_size), filled with the sequence length size of the sequence_tensor.
            sequence_lengths = torch.ones_like(
                sequence_tensor[:, 0, 0],
                dtype=torch.long) * sequence_tensor.size(1)

        adapted_span_indices = torch.tensor(
            span_indices, device=span_indices.device)

        for b in range(sequence_lengths.shape[0]):
            adapted_span_indices[b, :, 1][adapted_span_indices[b, :, 1] >=
                                          sequence_lengths[b]] = (
                                              sequence_lengths[b] - 1)

        # Raise Error if span indices were completely masked by sequence mask.
        # We only adjust span_end to the last valid index, so if span_end is below span_start,
        # both were above the max index:

        if (adapted_span_indices[:, :, 0] > adapted_span_indices[:, :, 1]
            ).any():
            raise IndexError(
                "Span indices were masked out entirely by sequence mask", )

        # span_vals <- (batch x num_spans x max_span_length x dim)
        span_vals, span_mask = batched_span_select(sequence_tensor,
                                                   adapted_span_indices)

        # The application of masked_max requires a mask of the same shape as span_vals
        # We repeat the mask along the last dimension (embedding dimension)
        repeat_dim = len(span_vals.shape) - 1
        repeat_idx = [1] * (repeat_dim) + [span_vals.shape[-1]]

        # ext_span_mask <- (batch x num_spans x max_span_length x dim)
        # ext_span_mask True for values in span, False for masked out values
        ext_span_mask = span_mask.unsqueeze(repeat_dim).repeat(repeat_idx)

        # max_out <- (batch x num_spans x dim)
        max_out = masked_max(span_vals, ext_span_mask, dim=-2)

        return max_out
