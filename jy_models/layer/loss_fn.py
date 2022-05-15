# https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py
def multilabel_categorical_crossentropy(y_pred,
                                        y_true,
                                        epsilon=1e-7,
                                        infinity=1e12,
                                        framework="torch"):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素是0～1
           的数，表示当前类是目标类的概率；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 和
           https://kexue.fm/archives/9064 。
    """
    if framework == "torch":
        import torch
        cat = torch.cat
        zeros_like = torch.zeros_like
        where = torch.where
        logsumexp = torch.logsumexp
        clip = torch.clip
        log = torch.log
    elif framework == "paddle":
        import paddle
        cat = paddle.concat
        zeros_like = paddle.zeros_like
        where = paddle.where
        logsumexp = paddle.logsumexp
        clip = paddle.clip
        log = paddle.log
    elif framework == "tensorflow":
        import tensorflow as tf
        cat = tf.concat
        zeros_like = tf.zeros_like
        where = tf.where
        logsumexp = tf.reduce_logsumexp
        clip = tf.clip_by_value
        log = tf.math.log
    else:
        raise ValueError(
            "Unknow framework: {} framework should be in ['torch', 'paddle', 'tensorflow']".
            format(framework))

    y_mask = y_pred > -infinity / 10
    n_mask = (y_true < 1 - epsilon) & y_mask
    p_mask = (y_true > epsilon) & y_mask
    y_true = clip(y_true, epsilon, 1 - epsilon)
    infs = zeros_like(y_pred) + infinity
    y_neg = where(n_mask, y_pred, -infs) + log(1 - y_true)
    y_pos = where(p_mask, -y_pred, -infs) + log(y_true)
    zeros = zeros_like(y_pred[..., :1])
    y_neg = cat([y_neg, zeros], axis=-1)
    y_pos = cat([y_pos, zeros], axis=-1)
    neg_loss = logsumexp(y_neg, axis=-1)
    pos_loss = logsumexp(y_pos, axis=-1)
    return neg_loss + pos_loss


def sparse_multilabel_categorical_crossentropy(y_pred,
                                               y_true,
                                               mask_zero=False,
                                               framework="torch",
                                               epsilon=1e-7,
                                               infinity=1e12):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    if framework == "torch":
        import torch
        cat = torch.cat
        zeros_like = torch.zeros_like
        logsumexp = torch.logsumexp
        clip = torch.clip
        log = torch.log

        def batch_gather(x, index, axis=-1):
            return torch.gather(x, dim=axis, index=index)

        exp = torch.exp
    elif framework == "paddle":
        import paddle
        cat = paddle.concat
        zeros_like = paddle.zeros_like
        logsumexp = paddle.logsumexp
        clip = paddle.clip
        log = paddle.log
        batch_gather = paddle.index_sample
        exp = paddle.exp
    elif framework == "tensorflow":
        import tensorflow as tf
        cat = tf.concat
        zeros_like = tf.zeros_like
        logsumexp = tf.reduce_logsumexp
        clip = tf.clip_by_value
        log = tf.math.log

        def batch_gather(x, index, batch_dims=-1):
            return tf.gather(x, index, batch_dims=batch_dims)

        exp = tf.exp
    else:
        raise ValueError(
            "Unknow framework: {} framework should be in ['torch', 'paddle', 'tensorflow']".
            format(framework))

    zeros = zeros_like(y_pred[..., :1])
    y_pred = cat([y_pred, zeros], axis=-1)
    if mask_zero:
        infs = zeros + infinity
        y_pred = cat([infs, y_pred[..., 1:]], axis=-1)
    y_pos_2 = batch_gather(y_pred, index=y_true)
    y_pos_1 = cat([y_pos_2, zeros], axis=-1)
    if mask_zero:
        y_pred = cat([-infs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = batch_gather(y_pred, index=y_true)
    pos_loss = logsumexp(-y_pos_1, axis=-1)
    all_loss = logsumexp(y_pred, axis=-1)
    aux_loss = logsumexp(y_pos_2, axis=-1) - all_loss
    aux_loss = clip(1 - exp(aux_loss), epsilon, 1)
    neg_loss = all_loss + log(aux_loss)
    return pos_loss + neg_loss


def globalpointer_kl_div(logits1, logits2, framework="torch"):
    # logits1 shape: (batch_size, entity_type_num, seqlen, seqlen)
    # logits2 shape: (batch_size, entity_type_num, seqlen, seqlen)
    if framework == "torch":
        import torch
        sigmoid = torch.nn.functional.sigmoid
        reshape = torch.reshape
        reduce_sum = torch.sum
    elif framework == "paddle":
        import paddle
        sigmoid = paddle.nn.functional.sigmoid
        reshape = paddle.reshape
        reduce_sum = paddle.sum
    elif framework == "tensorflow":
        import tensorflow as tf
        sigmoid = tf.sigmoid
        reshape = tf.reshape
        reduce_sum = tf.reduce_sum
    else:
        raise ValueError(
            "Unknow framework: {} framework should be in ['torch', 'paddle', 'tensorflow']".
            format(framework))
    res = (sigmoid(logits1) - sigmoid(logits2)) * (logits1 - logits2)
    return reduce_sum(reshape(res, [1, -1]))


if __name__ == "__main__":
    import paddle
    paddle.set_device("cpu")
    import torch
    import tensorflow as tf
    from bert4keras.backend import (
        multilabel_categorical_crossentropy as
        bert4keras_multilabel_categorical_crossentropy,
        sparse_multilabel_categorical_crossentropy as
        bert4keras_sparse_multilabel_categorical_crossentropy, )
    paddle.seed(42)
    x = paddle.randn((3, 5))
    y = paddle.randn((3, 5))
    yy = paddle.to_tensor([[1, 2, 0], [1, 0, 0], [4, 0, 0]], dtype="int64")
    paddle_out = multilabel_categorical_crossentropy(
        x, y, framework="paddle"), sparse_multilabel_categorical_crossentropy(
            x, yy, mask_zero=True, framework="paddle")

    torch_out = multilabel_categorical_crossentropy(
        torch.from_numpy(x.numpy()),
        torch.from_numpy(y.numpy()),
        framework="torch"), sparse_multilabel_categorical_crossentropy(
            torch.from_numpy(x.numpy()),
            torch.from_numpy(yy.numpy()),
            mask_zero=True,
            framework="torch", )
    tensorflow_out = multilabel_categorical_crossentropy(
        tf.constant(x.numpy()), tf.constant(y.numpy()),
        framework="tensorflow"), sparse_multilabel_categorical_crossentropy(
            tf.constant(x.numpy()),
            tf.constant(yy.numpy()),
            mask_zero=True,
            framework="tensorflow")
    bert4keras_out = bert4keras_multilabel_categorical_crossentropy(
        tf.constant(y.numpy()), tf.constant(x.numpy(
        ))), bert4keras_sparse_multilabel_categorical_crossentropy(
            tf.constant(yy.numpy()), tf.constant(x.numpy()), mask_zero=True)
    print("paddle_out:", paddle_out)
    print("torch_out:", torch_out)
    print("tensorflow_out:", tensorflow_out)
    print("bert4keras_out:", bert4keras_out)
    print("=" * 50)

    paddle_globalpointer_kl_div_loss = globalpointer_kl_div(
        x, y, framework="paddle")
    torch_globalpointer_kl_div_loss = globalpointer_kl_div(
        torch.from_numpy(x.numpy()),
        torch.from_numpy(y.numpy()),
        framework="torch")
    tensorflow_globalpointer_kl_div_loss = globalpointer_kl_div(
        tf.constant(x.numpy()), tf.constant(y.numpy()), framework="tensorflow")
    print("paddle_globalpointer_kl_div_loss:",
          paddle_globalpointer_kl_div_loss)
    print("torch_globalpointer_kl_div_loss:", torch_globalpointer_kl_div_loss)
    print("tensorflow_globalpointer_kl_div_loss:",
          tensorflow_globalpointer_kl_div_loss)
