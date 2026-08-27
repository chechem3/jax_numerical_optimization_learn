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
target_params = 0.5

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
for _ in range(1000):
    # 计算梯度
    grads = jax.grad(compute_loss)(params, xs, ys)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

assert jnp.allclose(params, target_params, atol=1e-2)





