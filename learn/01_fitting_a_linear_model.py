import jax.numpy as jnp
import jax
import optax
import functools

'''
下面，我们开始设置一个简单的线性模型和一个代价函数。
你可以在任何其他库中看到。
这里，我们用最简单的方式来自己实现它
代价函数，L2 loss, 是从 Optax 库中导入的。
'''

@functools.partial(jax.vmap, in_axes=(None, 0))
def network(params, x):
    return jnp.dot(params, x)

def compute_loss(params, x, y):
    y_pred = network(params, x)
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss



'''
然后，我们在已知的线性模型中，生成一些数据
'''
key = jax.random.PRNGKey(42)
target_params = jnp.array([0.5, 0.5])

# 生成一些数据
xs = jax.random.normal(key, (16, 2))
ys = jnp.sum(xs * target_params, axis = 1)

'''
# 关于 optax 的基本用法
optax 包含很多主流优化器的实现 https://optax.readthedocs.io/en/latest/api/optimizers.html
举个例子，用于 Adam 优化器的传递可以使用 optax.adam,
在这里，我们开始调用 GadientTransformation 对象用于 Adam 优化器
我们初始化 optimizer 状态使用 init 函数 和 network 的 params
'''
start_learning_rate = 0.1
optimizer = optax.adam(start_learning_rate)

# 初始化 model 和 优化器 的参数
params = jnp.array([0.0, 0.0])
opt_state = optimizer.init(params)

'''
下一步就是开始写更新循环，GradientTransformation 对象包含 1 个 update 函数，这个 update 函数
接收当前的优化状态，当前的梯度，斌返回 update 需要的参数

Optax 从少量的参数规则中来，这个规则接收从梯度穿度到当前参数的更新
'''
# 一个简单的循环
for _ in range(500):
    # 计算梯度
    grads = jax.grad(compute_loss)(params, xs, ys)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

print(f"params = {params}, target_params={target_params}")
assert jnp.allclose(params, target_params)


##########################################################
'''
# 自定义优化器
optax 通过 链式梯度传递的方式，可以轻松的创建自定义的优化器。
举个例子，这会基于 Adam 创建一个优化器。
需要注意的是，-learning_rate 是一个重要的细节，在 apply_updates 是添加的
'''
# 学习率的指数衰减
scheduler = optax.exponential_decay(
    init_value=start_learning_rate,
    transition_steps=1000,
    decay_rate=0.99
)

# 使用 optax.chain 来叠加上面的指数学习率衰减
gradient_transform = optax.chain(
    optax.clip_by_global_norm(1.0),        # 使用 global norm 来做梯度截断
    optax.scale_by_adam(),                 # 使用 adam 更新
    optax.scale_by_schedule(scheduler),    # 使用 scheduleer 的学习率
    # 符号设置为 -1, 因为我们期望梯度是下降的
    optax.scale(-1)
)

# 初始化 model + optimizer 的参数
params = jnp.array([0.0, 0.0])    
opt_state = gradient_transform.init(params)

# 继续循环
for _ in range(1000):
    grads = jax.grad(compute_loss)(params, xs, ys)
    updates, opt_state = gradient_transform.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

assert jnp.allclose(params, target_params)


#########################################
'''
# 关于 Optax 的进阶用法

## 修改关于一个调度的超参数

在一些场景中，修改超参数，而不是学习率，更有意义
我们可以轻松的使用 inject_hyperparams
举个例子，这些代码引诱 clip_by_global_norm 的 max_norm, 使用梯度下降作为训练过程

这部分是减小全局范数
'''
decaying_global_norm_tx = optax.inject_hyperparams(optax.clip_by_global_norm)(max_norm=optax.linear_schedule(1.0, 0.0, transition_steps=99))

opt_state = decaying_global_norm_tx.init(None)
assert opt_state.hyperparams["max_norm"] == 1.0, "最大的 norm 应该从 1 开始"

for _ in range(100):
    _, opt_state = decaying_global_norm_tx.update(None, opt_state)

assert opt_state.hyperparams["max_norm"] == 0.0, "最大的 norm 应该到 0 结束"










