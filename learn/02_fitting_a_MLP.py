########################################
'''
# 范例: 拟合一个 MLP

让我们使用 optax 来拟合一个参数函数, 
同样，我们需要考虑到是否一个值 是 奇数 还是 偶数 了？

我们创建一个数据集，是一批次的随机的 8bit 的 integers 数据
使用 1-hot 编码表达是 odd 或者 even
[1, 0] 代表 odd，奇数
[0, 1] 代表 even，偶数
'''

import optax
import jax.numpy as jnp
import jax
import numpy as np

BATCH_SIZE = 5
NUM_TRAIN_STEPS = 1_000
RAW_TRAINING_DATA = np.random.randint(255, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 1))

TRAINING_DATA = np.unpackbits(RAW_TRAINING_DATA.astype(np.uint8), axis=-1)
LABLES = jax.nn.one_hot(RAW_TRAINING_DATA % 2, 2).astype(jnp.float32).reshape(NUM_TRAIN_STEPS, BATCH_SIZE, 2)


'''
我们现在使用 JAX 定义了一个参数化的函数。这将会让我们高效的计算梯度
我们的函数中，是一个一层的 MLP, 只有一个简单的 隐藏层，和一个输出层。
我们初始化所有的参数，使用标准的高斯分布
'''
initial_params = {
    "hidden": jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),
    "output": jax.random.normal(shape=[32, 2], key=jax.random.PRNGKey(1)),
}

def net(x: jnp.ndarray, params: optax.Params) -> jnp.ndarray:
    x = jnp.dot(x, params["hidden"])
    x = jax.nn.relu(x)
    x = jnp.dot(x, params["output"])
    return x

def loss(params: optax.Params, batch:jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    params: 是 MLP 的参数 dict
    batch: 是 8 bit 的整形
    """
    y_hat = net(batch, params)    # 使用神经网络做预测， y_hat 的 shape: (batch_size, 2)

    # optax 同样提供过了一系列通用的 loss function
    loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)

    return loss_value.mean()

'''
我们将会使用 optax.adam 来从他们的梯度上面计算参数更新，在每一个优化步骤上

需要注意的是， Optax 的优化器是使用纯函数来实现的，因此我们需要保持优化器状态的跟踪
对于 Adam 优化器，这个状态需要包含冲量值
'''
def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    """
    params: 是待优化的参数，在这里是一个 dict, 里面的 key 是 2 层 MLP 的参数
    optimizer: 是一个优化器，在这里是 optax.adam
    """
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value


    for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABLES)):
        params, opt_state, loss_value = step(params, opt_state, batch, labels)
        if (i % 100 == 0):
            print(f"Step: {i}, Loss: {loss_value}")

    return params, opt_state

# 最后，我们可以 fit 我们的参数函数，使用 optax 提供的 Adam 优化器
optimizer = optax.adam(learning_rate=1e-2)
_ = fit(initial_params, optimizer)


