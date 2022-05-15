import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import overrides

# !!!!!!!!!!!!!!!!!ALL copied from allennlp!!!!!!!!!!!!!!!!!
# from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
__all__ = [
    "SelfAttentiveSpanExtractor", "EndpointSpanExtractor", "SpanExtractor"
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


class SelfAttentiveSpanExtractor(SpanExtractor):
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

    # Returns

    attended_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def forward(
            self,
            sequence_tensor: torch.FloatTensor,
            span_indices: torch.LongTensor,
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

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1)

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


class EndpointSpanExtractor(SpanExtractor):
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
        super().__init__()
        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            self._start_sentinel = nn.Parameter(
                torch.randn([1, 1, int(input_dim)]))

        self._span_width_embedding = None
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = nn.Embedding(
                num_embeddings=num_width_embeddings,
                embedding_dim=span_width_embedding_dim, )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_buckets and span_width_embedding_dim.")

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = get_combined_dim(self._combination,
                                        [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(
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
            exclusive_span_starts = (exclusive_span_starts *
                                     ~start_sentinel_mask.squeeze(-1))

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
        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = bucket_values(
                    span_ends - span_starts,
                    num_total_buckets=self.
                    _num_width_embeddings  # type: ignore
                )
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            combined_tensors = torch.cat(
                [combined_tensors, span_width_embeddings], -1)

        if span_indices_mask is not None:
            return combined_tensors * span_indices_mask.unsqueeze(-1)

        return combined_tensors
