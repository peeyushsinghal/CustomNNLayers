"""
Custom layers for CNN and Transformers
"""
import numpy as np
import torch
from torch.autograd import Function


class CustomConv2d(Function):
    """
    Custom convolutional layer implementation.
    """
    @staticmethod
    def forward(ctx, input_image, kernel, padding, stride):
        """
        Forward pass for the custom convolutional layer.
        :param ctx: The context object.
        :param input_image: The input image tensor.
        :param kernel: The kernel tensor.
        :param padding: The padding value.
        :param stride: The stride value.
        :return: The output tensor.
        """
        # Save tensors for backward pass
        ctx.save_for_backward(input_image, kernel)
        ctx.padding = padding
        ctx.stride = stride
        
        # Convert to numpy for computation
        input_np = input_image.detach().numpy()
        kernel_np = kernel.detach().numpy()
        
        # Add padding
        padded_input = np.pad(input_np, ((padding, padding), (padding, padding)), mode='edge')
        
        # Get dimensions
        output_height = (padded_input.shape[0] - kernel_np.shape[0]) // stride + 1
        output_width = (padded_input.shape[1] - kernel_np.shape[1]) // stride + 1
        
        # Compute convolution
        output = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                start_h = i * stride
                end_h = start_h + kernel_np.shape[0]
                start_w = j * stride
                end_w = start_w + kernel_np.shape[1]
                output[i, j] = np.sum(padded_input[start_h:end_h, start_w:end_w] * kernel_np)
        
        return torch.from_numpy(output).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom convolutional layer.
        :param ctx: The context object.
        :param grad_output: The gradient of the output tensor.
        :return: The gradients of the input and kernel tensors.
        """
        input_image, kernel = ctx.saved_tensors
        padding, stride = ctx.padding, ctx.stride
        
        # Implement the backward pass (gradient computation)
        grad_input = torch.zeros_like(input_image)
        grad_kernel = torch.zeros_like(kernel)
        
        # Convert to numpy for computation
        grad_output_np = grad_output.numpy()
        input_np = input_image.numpy()
        kernel_np = kernel.numpy()
        
        # Compute gradients (simplified version)
        padded_input = np.pad(input_np, ((padding, padding), (padding, padding)), mode='edge')
        
        for i in range(grad_output.shape[0]):
            for j in range(grad_output.shape[1]):
                start_h = i * stride
                end_h = start_h + kernel.shape[0]
                start_w = j * stride
                end_w = start_w + kernel.shape[1]
                grad_kernel += grad_output_np[i, j] * torch.from_numpy(
                    padded_input[start_h:end_h, start_w:end_w])
        
        return grad_input, grad_kernel, None, None


class CustomReLU(Function):
    """
    Custom ReLU activation function implementation.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass for the custom ReLU activation function.
        :param ctx: The context object.
        :param input: The input tensor.
        :return: The output tensor.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom ReLU activation function.
        :param ctx: The context object.
        :param grad_output: The gradient of the output tensor.
        :return: The gradient of the input tensor.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class CustomMaxPool2d(Function):
    """
    Custom max pooling layer implementation.
    """
    @staticmethod
    def forward(ctx, input, kernel_size=2, stride=2):
        """
        Forward pass for the custom max pooling layer.
        :param ctx: The context object.
        :param input: The input tensor.
        :param kernel_size: The kernel size.
        :param stride: The stride value.
        :return: The output tensor.
        """
        # Save parameters for backward pass
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        
        # Convert to numpy for computation
        input_np = input.detach().numpy()
        height, width = input_np.shape
        pool_height = (height - kernel_size) // stride + 1
        pool_width = (width - kernel_size) // stride + 1
        
        # Compute max pooling
        output = np.zeros((pool_height, pool_width))
        max_indices = np.zeros((pool_height, pool_width, 2), dtype=int)
        
        for i in range(pool_height):
            for j in range(pool_width):
                start_h = i * stride
                end_h = start_h + kernel_size
                start_w = j * stride
                end_w = start_w + kernel_size
                window = input_np[start_h:end_h, start_w:end_w]
                output[i, j] = np.max(window)
                idx = np.unravel_index(window.argmax(), window.shape)
                max_indices[i, j] = [start_h + idx[0], start_w + idx[1]]
        
        ctx.save_for_backward(input, torch.from_numpy(max_indices))
        return torch.from_numpy(output).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom max pooling layer.
        :param ctx: The context object.
        :param grad_output: The gradient of the output tensor.
        :return: The gradient of the input tensor.
        """
        input, max_indices = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        
        max_indices = max_indices.numpy()
        grad_output_np = grad_output.numpy()
        
        for i in range(grad_output.shape[0]):
            for j in range(grad_output.shape[1]):
                h, w = max_indices[i, j]
                grad_input[h, w] += grad_output_np[i, j]
        
        return grad_input, None, None


class CustomLinear(Function):
    """
    Custom linear layer implementation.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        Forward pass for the custom linear layer.
        :param ctx: The context object.
        :param input: The input tensor.
        :param weight: The weight tensor.
        :param bias: The bias tensor.
        :return: The output tensor.
        """
        ctx.save_for_backward(input, weight, bias)
        
        # Convert to numpy for computation
        input_np = input.detach().numpy()
        weight_np = weight.detach().numpy()
        
        # Store original shape for reshaping later
        original_shape = input_np.shape
        
        # Reshape input to 2D if it's 3D: (batch_size * seq_len, input_dim)
        if len(input_np.shape) > 2:
            batch_size, seq_len, input_dim = input_np.shape
            input_np = input_np.reshape(-1, input_dim)
        
        # Ensure weight matrix is properly oriented (output_dim, input_dim)
        if weight_np.shape[1] != input_np.shape[-1]:
            weight_np = weight_np.T
            
        # Compute linear transformation
        output = np.dot(input_np, weight_np.T)  # Result: (batch_size * seq_len, output_dim)
        
        if bias is not None:
            output += bias.detach().numpy()
        
        # Reshape output back to original dimensions if input was 3D
        if len(original_shape) > 2:
            output = output.reshape(original_shape[0], original_shape[1], -1)
        
        return torch.from_numpy(output).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom linear layer.
        :param ctx: The context object.
        :param grad_output: The gradient of the output tensor.
        :return: The gradients of the input and weight tensors.
        """
        input, weight, bias = ctx.saved_tensors
        
        # Convert to numpy for computation
        grad_output_np = grad_output.numpy()
        input_np = input.numpy()
        weight_np = weight.numpy()
        
        # Store original shapes
        original_input_shape = input_np.shape
        original_grad_shape = grad_output_np.shape
        
        # Reshape if input was 3D
        if len(input_np.shape) > 2:
            input_np = input_np.reshape(-1, input_np.shape[-1])
            grad_output_np = grad_output_np.reshape(-1, grad_output_np.shape[-1])
        
        # Ensure weight matrix is properly oriented
        if weight_np.shape[1] != input_np.shape[-1]:
            weight_np = weight_np.T
            
        # Compute gradients
        grad_input = np.dot(grad_output_np, weight_np)  # [batch_size * seq_len, input_dim]
        grad_weight = np.dot(grad_output_np.T, input_np)  # [output_dim, input_dim]
        
        # Ensure grad_weight has the same shape as the original weight
        if weight.shape != grad_weight.shape:
            grad_weight = grad_weight.T
            
        # Reshape grad_input back to original shape if input was 3D
        if len(original_input_shape) > 2:
            grad_input = grad_input.reshape(original_input_shape)
        
        if bias is not None:
            grad_bias = grad_output_np.sum(axis=0)
            return (torch.from_numpy(grad_input).float(),
                   torch.from_numpy(grad_weight).float(),
                   torch.from_numpy(grad_bias).float())
        
        return (torch.from_numpy(grad_input).float(),
                torch.from_numpy(grad_weight).float(),
                None)


class CustomGELU(Function):
    """
    Custom GELU activation function implementation.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass for the custom GELU activation function.
        :param ctx: The context object.
        :param input: The input tensor.
        :return: The output tensor.
        """
        # GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
        input_np = input.detach().numpy()
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (input_np + 0.044715 * input_np**3)))
        result = input_np * cdf
        
        ctx.save_for_backward(input)
        ctx.cdf = cdf
        
        return torch.from_numpy(result).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom GELU activation function.
        :param ctx: The context object.
        :param grad_output: The gradient of the output tensor.
        :return: The gradient of the input tensor.
        """
        input, = ctx.saved_tensors
        cdf = ctx.cdf
        
        input_np = input.numpy()
        grad_output_np = grad_output.numpy()
        
        # Derivative of GELU
        pdf = np.exp(-0.5 * input_np**2) / np.sqrt(2 * np.pi)
        gradient = cdf + input_np * pdf * (0.0356774 * input_np**2 + 0.797885)
        
        grad_input = grad_output_np * gradient
        return torch.from_numpy(grad_input).float()


class CustomLayerNorm(Function):
    """
    Custom layer normalization implementation.
    """
    @staticmethod
    def forward(ctx, input, eps=1e-5):
        """
        Forward pass for the custom layer normalization.
        :param ctx: The context object.
        :param input: The input tensor.
        :param eps: The epsilon value for numerical stability.
        :return: The normalized tensor.
        """
        input_np = input.detach().numpy()
        
        # Calculate mean and variance
        mean = np.mean(input_np, axis=-1, keepdims=True)
        var = np.var(input_np, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (input_np - mean) / np.sqrt(var + eps)
        
        ctx.save_for_backward(torch.from_numpy(x_norm).float(),
                            torch.from_numpy(var).float(),
                            input)
        ctx.eps = eps
        
        return torch.from_numpy(x_norm).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom layer normalization.
        :param ctx: The context object.
        :param grad_output: The gradient of the output tensor.
        :return: The gradients of the input tensor.
        """
        x_norm, var, input = ctx.saved_tensors
        eps = ctx.eps
        
        input_np = input.numpy()
        grad_output_np = grad_output.numpy()
        x_norm_np = x_norm.numpy()
        var_np = var.numpy()
        
        N = input_np.shape[-1]
        
        # Gradient calculations
        dx_norm = grad_output_np
        dvar = -0.5 * np.sum(dx_norm * (input_np - np.mean(input_np, axis=-1, keepdims=True)) 
                            * np.power(var_np + eps, -1.5), axis=-1, keepdims=True)
        dmean = -np.sum(dx_norm / np.sqrt(var_np + eps), axis=-1, keepdims=True)
        
        grad_input = dx_norm / np.sqrt(var_np + eps)
        grad_input += 2 * dvar * (input_np - np.mean(input_np, axis=-1, keepdims=True)) / N
        grad_input += dmean / N
        
        return torch.from_numpy(grad_input).float(), None


class CachedAttention(Function):
    """
    Custom attention mechanism with caching of key-value pairs.
    """
    _cache = {}  # Class variable to store KV cache
    
    @staticmethod
    def forward(ctx, queries, keys, values, mask=None, use_cache=False):
        """
        Forward pass for the custom attention mechanism.
        :param ctx: The context object.
        :param queries: The queries tensor.
        :param keys: The keys tensor.
        :param values: The values tensor.
        :param mask: The mask tensor.
        :param use_cache: Whether to use caching.
        :return: The output tensor.
        """
        # Save tensors for backward pass
        ctx.save_for_backward(queries, keys, values, mask)
        
        # Convert to numpy for computation
        q_np = queries.detach().numpy()  # [batch_size, num_heads, seq_len, head_dim]
        k_np = keys.detach().numpy()     # [batch_size, num_heads, seq_len, head_dim]
        v_np = values.detach().numpy()   # [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(q_np.shape[-1])
        
        # Reshape k_np for matrix multiplication
        k_np_t = np.transpose(k_np, (0, 1, 3, 2))  # [batch_size, num_heads, head_dim, seq_len]
        
        # Compute attention scores
        scores = np.matmul(q_np, k_np_t) * scale  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores + mask.detach().numpy()
        
        # Apply softmax
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / (np.sum(attention_weights, axis=-1, keepdims=True) + 1e-6)
        
        # Compute attention output
        output = np.matmul(attention_weights, v_np)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Store in cache if requested
        if use_cache:
            CachedAttention._cache['k'] = keys
            CachedAttention._cache['v'] = values
        
        return torch.from_numpy(output).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom attention mechanism.
        :param ctx: The context object.
        :param grad_output: The gradient of the output tensor.
        :return: The gradients of the input tensors.
        """
        queries, keys, values, mask = ctx.saved_tensors
        
        # Convert to numpy
        grad_output_np = grad_output.numpy()
        q_np = queries.numpy()
        k_np = keys.numpy()
        v_np = values.numpy()
        
        # Compute gradients (simplified)
        scale = 1.0 / np.sqrt(q_np.shape[-1])
        
        # Reshape for matrix multiplication
        k_np_t = np.transpose(k_np, (0, 1, 3, 2))
        v_np_t = np.transpose(v_np, (0, 1, 3, 2))
        
        # Gradient with respect to queries
        grad_queries = np.matmul(grad_output_np, k_np_t) * scale
        
        # Gradient with respect to keys
        grad_keys = np.matmul(np.transpose(grad_output_np, (0, 1, 3, 2)), q_np) * scale
        
        # Gradient with respect to values
        grad_values = np.matmul(np.transpose(attention_weights, (0, 1, 3, 2)), grad_output_np)
        
        return (torch.from_numpy(grad_queries).float(),
                torch.from_numpy(grad_keys).float(),
                torch.from_numpy(grad_values).float(),
                None,
                None)


# Modified callable functions to include default arguments
def conv2d(x, kernel, padding=0, stride=1):
    """
    Custom convolutional layer implementation.
    """
    return CustomConv2d.apply(x, kernel, padding, stride)


def max_pool2d(x, kernel_size=2, stride=2):
    """
    Custom max pooling layer implementation.
    """
    return CustomMaxPool2d.apply(x, kernel_size, stride)


# relu remains unchanged
relu = CustomReLU.apply


# Add callable functions for the new layers
def linear(input, weight, bias=None):
    """
    Custom linear layer implementation.
    """
    return CustomLinear.apply(input, weight, bias)


def gelu(x):
    """
    Custom GELU activation function implementation.
    """
    return CustomGELU.apply(x)


def layer_norm(x, eps=1e-5):
    """
    Custom layer normalization implementation.
    """
    return CustomLayerNorm.apply(x, eps)


def attention(queries, keys, values, mask=None, use_cache=False):
    """
    Custom attention mechanism with caching of key-value pairs.
    """
    return CachedAttention.apply(queries, keys, values, mask, use_cache)