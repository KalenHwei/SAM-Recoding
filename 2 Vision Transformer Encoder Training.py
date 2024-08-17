import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, Type
import math

# 0.0 定义数据读取块ImageReader

class ImageReader(nn.Module):
    def __init__(self, 
                 size: Tuple = (1024, 1024)
                 ) -> None:
        super().__init__()
        self.reader = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        
    def forward(self, x:str) -> torch.Tensor:
        image = Image.open(x)
        image_tensor = self.reader(image)
        return image_tensor.unsqueeze(0)


# 0.1 前置函数1: 获取相对位置嵌入

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    get_rel_pos函数用于根据q和k的size，获取相对位置嵌入
    它的作用是捕捉输入序列中不同位置之间的相对关系
    在注意力机制中，相对位置嵌入被用来增强模型对不同位置之间的依赖关系的建模能力
    通过计算查询和键之间的相对坐标，然后根据相对坐标从相对位置嵌入中提取相应的位置嵌入
    可以将这些位置嵌入添加到attention map中，从而影响注意力权重的计算
    这有助于模型更好地理解输入序列中不同位置之间的关系，并提高模型在处理序列数据时的性能。
    
    参数解释:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): 相对位置嵌入 (L, C).

    输出:
        是根据查询和键的大小提取的相对位置嵌入
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


#0.2 前置函数2: 根据分解的相对位置嵌入调整attention map

def add_decomposed_rel_pos(
    scores: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    计算分解后的相对位置嵌入
        scores (Tensor): attention map，也就是torch.matmul(q, k_trans) / self.scale
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        scores (Tensor): 加上了相对位置嵌入补偿的attention map
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    scores = (
        scores.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return scores 


#0.3 前置函数3: 定义将图片切割成window的功能函数window_partition

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    作用是将输入的张量（通常代表一个图像或特征图）划分成非重叠的小窗口，并在必要时对输入进行填充（padding），以确保窗口划分是完整的.
    参数:
        x (tensor): 输入，前端处理好的一般shape = (B=1, H, W, C) = (1, 16, 16, 768)
        window_size (int): 每个窗口的大小

    返回值:
        windows: 划分后的窗口，形状为 [B * num_windows, window_size, window_size, C]
        (Hp, Wp): 填充后的高度和宽度，用于后续处理或恢复原始尺寸
    """
    B, H, W, C = x.shape # 首先读取x的shape= (B=1, H, W, C) = (1, 16, 16, 768)

    #得到需要填充的h和w的宽度
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    #如果需要pad的话执行if语句
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


#0.4 前置函数4: 定义将切割的window还原成图片的功能函数window_unpartition

def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    将分割的window还原成原始序列，并移除padding
    参数:
        windows (tensor): 输入window，shape = (B*num_windows, window_size, window_size, C)
        window_size (int): window size
        pad_hw (Tuple): 填充的高和宽，用tuple封装，Tuple = (Hp, Wp)
        hw (Tuple): 在padding之前的图片原始高宽(H, W)

    输出值:
        x: 还原切割之前的初始序列，shape = (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# 1 定义MLP块
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    

# 2 定义LayerNorm块
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    
# 3 定义Patch Embedding类，用卷积做
class PatchEmbed(nn.Module):
    def __init__(
            self, 
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        
        self.projection = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x:torch.Tensor表示输入x是Tensor，括号外面的-> torch.Tensor指函数返回值也是tensor
        x = self.projection(x)
        x = x.permute(0, 2, 3, 1) # 交换维度，即：(B C H W) -> (B H W C)
        return x
    
    
# 4 更新的Attention类，用nn.Linear集成了可学习母参数以提高效率
# 图片进入流程: image-tensorlizer-PatchEmbed-Attention
# 因此最后的图片shape = (B=1, H. W. C) = (1, 64, 64, 768)

class Attention(nn.Module):
    def __init__(self,
                 dmodel: int, # 也就是上面传进来的768
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True,
                 input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        dmodel_per_head = dmodel // num_heads
        self.scale = math.sqrt(dmodel_per_head)
        
        self.qkv = nn.Linear(dmodel, dmodel * 3, bias = qkv_bias) # 定义一个线性层
        self.output_linear = nn.Linear(dmodel, dmodel)
        
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "如果用了相对位置编码，则必须提供输入的size"
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, dmodel_per_head), requires_grad = True)
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, dmodel_per_head), requires_grad = True)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape # X的shape = (B, H, W, dmodel) = (1, 16, 16, 768)
        qkv_combine = self.qkv(x) # shape = (B, H, W, 3*dmodel)，即用一个线性层生成qkv_combine
        qkv_combine = qkv_combine.reshape(B, H*W, 3, self.num_heads, -1) # shape = (B, H*W, 3, num_heads, dmodel_per_head)
        qkv_combine = qkv_combine.permute(2, 0, 3, 1, 4) # shape = (3, B, num_heads, H*W, dmodel_per_head)
        qkv_combine = qkv_combine.reshape(3, B*self.num_heads, H*W, -1) # shape = (3, B*num_heads, H*W, dmodel_per_head)
        q, k, v = qkv_combine.unbind(0) # q, k, v shape = (B*num_heads, H*W, dmodel_per_head)
        
        k_trans = k.transpose(-2, -1) # k_trans shape = (B*num_heads, dmodel_per_head, H*W)
        scores = torch.matmul(q, k_trans) / self.scale # scores shape = (B*num_heads, H*W, H*W)
        
        # 如果使用了相对位置编码则启用
        if self.use_rel_pos:
            scores = add_decomposed_rel_pos(attn=scores, q=q, rel_pos_h=self.rel_pos_h, rel_pos_w=self.rel_pos_w, q_size=(H, W), k_size=(H, W))
        
        weights = torch.softmax(scores, dim = -1) # 在(B*num_heads, H*W, H*W)的最后一个维度上计算softmax。至于为什么已经推过了
        output = torch.matmul(weights, v) # shape = (B*num_heads, H*W, dmodel_per_head)
        output = output.view(B, self.num_heads, H, W, -1) # shape = (B, num_heads, H, W, dmodel_per_head)
        output = output.permute(0, 2, 3, 1, 4) # (B, H, W, num_heads, dmodel_per_head)
        output = output.reshape(B, H, W, -1) # shape = (B, H, W, dmodel)
        output = self.output_linear(output)
        
        return output
    

# 5 定义Vision Transformer Encoder Block

class Block(nn.Module):
    def __init__(self,
                 dmodel: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dmodel)
        self.Attention = Attention(dmodel,
                                   num_heads = num_heads,
                                   qkv_bias = qkv_bias,
                                   use_rel_pos= use_rel_pos,
                                   rel_pos_zero_init = rel_pos_zero_init,
                                   input_size = input_size if window_size == 0 else (window_size, window_size),
                                   )
        self.norm2 = norm_layer(dmodel)
        self.mlp = MLPBlock(embedding_dim = dmodel,
                            mlp_dim = int(dmodel * mlp_ratio),
                            act = act_layer
                            )
        self.window_size = window_size
    
    # 没有window：x - layernorm - Attention - short+x - x + mlp(layernorm(x))
    # 有window：中间加两个变形和还原即可
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x.shape = (B=1, H, W, dmodel) = (1, 16, 16, 768)
        shortcut = x
        
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size) # 返回值为windows = (B * num_windows, window_size, window_size, C)和(Hp, Wp)
        x = self.Attention(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, ((H, W)))
        
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


# 6 Vision Transformer Encoder 完整整合

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList() # 先生成一个空的Module，名称为blocks
        for i in range(depth):
            block = Block(
                dmodel=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block) # 再append进blocks里
        
        #出Block之后，经历卷积-标准化-卷积-标准化
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks: # 执行Block
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
    
