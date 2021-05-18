import torch
from torch import nn
from torch.nn import functional as F
import math

def OrtConv2d(*params, **named_params):
    conv = nn.Conv2d(*params, **named_params)
    nn.init.orthogonal_(conv.weight.data)
    return conv

def OrtLinear(*params, **named_params):
    lin = nn.Linear(*params, **named_params)
    nn.init.orthogonal_(lin.weight.data)
    return lin

def disable_spectral_norm(x):
    return x

class CBN2d(nn.Module):
    def __init__(self, num_features, num_conditions, num_z_features=0):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed_1 = nn.Conv2d(num_conditions + num_z_features, num_features, kernel_size=1, bias=False)
        self.embed_2 = nn.Conv2d(num_conditions + num_z_features, num_features, kernel_size=1, bias=False)

    def forward(self, x, y, z=None):
        y = y.reshape(y.shape[0], y.shape[1], 1, 1)
        if z is not None:
            y = y.repeat(1, 1, z.shape[2], z.shape[3])
            y = torch.cat((y, z), 1)
        gamma = self.embed_1(y)
        beta = self.embed_2(y)
        out = (1.0 + gamma) * self.bn(x) + beta
        return out

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, optimized, sn=disable_spectral_norm):
        super().__init__()
        self.sn = sn
        self.downsample = downsample
        self.optimized = optimized
        self.learnable_sc = in_channels != out_channels or downsample
        
        self.conv1 = sn(OrtConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = sn(OrtConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if self.learnable_sc:
            self.conv_sc = sn(OrtConv2d(in_channels, out_channels, kernel_size=1, bias=False))
        if sn != disable_spectral_norm:
            self.residual_coef = nn.Parameter(torch.zeros(1,), requires_grad=True)
        self.relu = nn.ReLU()
    
    def _residual(self, x):
        if not self.optimized:
            x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x
    
    def _shortcut(self, x):
        if self.learnable_sc:
            if self.optimized:
                x = self.conv_sc(F.avg_pool2d(x, 2)) if self.downsample else self.conv_sc(x)
            else:
                x = F.avg_pool2d(self.conv_sc(x), 2) if self.downsample else self.conv_sc(x)
        
        return x
    
    def forward(self, x):
        if self.sn != disable_spectral_norm:
            s = F.sigmoid(self.residual_coef)
            return self._shortcut(x) * s + self._residual(x) * (1 - s)
        return self._shortcut(x) + self._residual(x)
    
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conditions, z_cond, upsample):
        super().__init__()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.cbn1 = CBN2d(in_channels, num_conditions, z_cond)
        self.cbn2 = CBN2d(out_channels, num_conditions)
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
    
    def _upsample_conv(self, x, conv):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = conv(x)
        
        return x
    
    def _residual(self, x, y, z):
        x = self.relu(self.cbn1(x, y, z))
        x = self._upsample_conv(x, self.conv1) if self.upsample else self.conv1(x)
        x = self.relu(self.cbn2(x, y))
        x = self.conv2(x)
        return x
    
    def _shortcut(self, x):
        if self.learnable_sc:
            x = self._upsample_conv(x, self.conv_sc) if self.upsample else self.conv_sc(x)
            
        return x
    
    def forward(self, x, y, z=None):
        return self._shortcut(x) + self._residual(x, y, z)

class BigEncoder(nn.Module):
    def __init__(self, num_images, image_size, ch):
        super().__init__()
        
        self.image_size = image_size
        self.num_images = num_images
        
        blocks = []
        log_im_size = int(math.log(image_size, 2))
        assert math.log(image_size, 2) == log_im_size and image_size >= 32
        w = self.image_size
        first_block = True
        from_ch = num_images
        assert ch % (2 ** (log_im_size - 4)) == 0
        to_ch = ch // (2 ** (log_im_size - 4))
        
        while w > 4:
            blocks.append(DBlock(from_ch,
                                 to_ch,
                                 downsample=True,
                                 optimized=first_block))
            w //= 2
            first_block = False
            from_ch = to_ch
            if w > 8:
                to_ch *= 2
                
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[1] == self.num_images and x.shape[2] == x.shape[3] and x.shape[3] == self.image_size
        
        skip = []
        
        w = self.image_size
        k = 0
        for block in self.blocks:
            x = block(x)
            if block.downsample:
                w //= 2
            skip.append(x)
        
        return skip
    
class Anonimizer(nn.Module):
    def __init__(self, encoder, filters, reverse=False, last_vector=0):
        super().__init__()
        self.reverse = reverse
        self.encoder = encoder
        self.convs = []
        for ch in filters:
            self.convs.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))
        self.convs = nn.ModuleList(self.convs)
        self.fc = None
        if last_vector > 0:
            self.fc = nn.Linear(filters[-1], last_vector)
        
    def forward(self, x):
        xs = self.encoder(x)
        for i in range(len(xs)):
            xs[i] = self.convs[i](xs[i])
        return xs[::-1] if self.reverse else xs, self.fc(torch.sum(F.relu(xs[-1]), dim=(2, 3)))

class SnCond(nn.Module):
    def __init__(self, sn=False):
        super().__init__()
        self.sn = sn
        if sn:
            self.length = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def forward(self, x, y):
        if self.sn:
            assert len(x.shape) == 2 and len(y.shape) == 2
            y = y / torch.norm(y, dim=1).reshape(-1, 1)
            y = y * F.sigmoid(self.length)
            return torch.sum(y * x, dim=1, keepdim=True)
        else:
            return torch.sum(y * x, dim=1, keepdim=True)
    
class BigRegressor(nn.Module):
    def __init__(self, num_images,
                 image_size,
                 ch=512,
                 num_classes=0,
                 sparse_num_classes=False,
                 encoder_num_images=0,
                 sn=disable_spectral_norm,
                 sn_mul=1,
                 n_outputs=1,
                 a_shape_list=None):
        
        super().__init__()
        self.n_outputs = n_outputs
        self.sn_mul = sn_mul
        self.sn = sn
        self.num_classes = num_classes
        self.sparse_num_classes = sparse_num_classes
        num_images = num_images + encoder_num_images
        self.num_images = num_images
        self.encoder_num_images = encoder_num_images
        self.image_size = image_size
        if num_classes > 0:
            self.sn_cond = SnCond(sn=(sn!=disable_spectral_norm))
            self.sn_sigma = nn.Parameter(torch.zeros(1,), requires_grad=True)

        if encoder_num_images > 0:
            self.z_mul = nn.Parameter(torch.ones(1, encoder_num_images, 1, 1), requires_grad=True)
        
        blocks = []
        log_im_size = int(math.log(image_size, 2))
        assert math.log(image_size, 2) == log_im_size and image_size >= 32
        w = self.image_size
        first_block = True
        from_ch = num_images
        assert ch % (2 ** (log_im_size - 4)) == 0
        to_ch = ch // (2 ** (log_im_size - 4))
        
        while w > 4:
            blocks.append(DBlock(from_ch, to_ch,
                                 downsample=True,
                                 sn=sn,
                                 optimized=first_block))
            w //= 2
            first_block = False
            from_ch = to_ch
            if w > 8:
                to_ch *= 2
                
        self.blocks = nn.ModuleList(blocks)
        
        self.relu = nn.ReLU()
        self.fc = sn(OrtLinear(to_ch, n_outputs, bias=False))
        if a_shape_list is not None:
            self.a_layers = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size=3, padding=1) for ch in a_shape_list])
        if self.num_classes > 0:
            layer = nn.Embedding if sparse_num_classes else nn.Linear
            self.embed = layer(self.num_classes, to_ch)
    
    def forward(self, x, y=None, z=None, a=None):
        assert len(x.shape) == 4
        
        if self.num_classes > 0:
            assert self.n_outputs == 1
            assert y is not None
            assert y.shape[0] == x.shape[0]
            if not self.sparse_num_classes:
                assert len(y.shape) == 2
                assert y.shape[1] == self.num_classes
            else:
                assert len(y.shape) == 1
            y = self.embed(y)

        x = x * self.sn_mul
        if self.encoder_num_images > 0:
            x = torch.cat((x, z * self.z_mul), 1)
        
        i = 0
        a_i = 0
        w = self.image_size
        while i < len(self.blocks):
            x = self.blocks[i](x)
            if self.blocks[i].downsample:
                w //= 2
            i += 1
            if a is not None:
                x = x + self.a_layers[a_i](a[a_i])
            a_i += 1

        x = self.relu(x)
        if self.sn != disable_spectral_norm:
            x = torch.mean(x, dim=(2, 3))
        else:
            x = torch.sum(x, dim=(2, 3))
        
        out = self.fc(x)
        if self.num_classes > 0:
            if self.sn == disable_spectral_norm:
                out = out + self.sn_cond(x=x, y=y)
            else:
                s = F.sigmoid(self.sn_sigma)
                out = out * s + self.sn_cond(x=x, y=y) * (1 - s)
        
        return out

class BigGenerator(nn.Module):
    def __init__(self, num_images,
                 image_size,
                 latent_dim_mul=20,
                 ch=512,
                 num_classes=0,
                 sparse_num_classes=False,
                 encoder_image_size=0,
                 encoder_num_images=0,
                 bottleneck=nn.Tanh(),
                 a_shape_list=None):
        super().__init__()
        
        log_im_size = int(math.log(image_size, 2))
        assert math.log(image_size, 2) == log_im_size and image_size >= 32
        
        self.ch = ch
        self.num_images = num_images
        self.encoder_num_images = encoder_num_images
        self.sparse_num_classes = sparse_num_classes
        self.image_size = image_size
        encoder_image_size = image_size
        self.encoder_image_size = encoder_image_size
        self.latent_dim = latent_dim_mul * (log_im_size - 1)
        self.num_classes = num_classes
        self.num_chunk = log_im_size - 1
        
        if self.num_classes > 0:
            layer = nn.Embedding if sparse_num_classes else nn.Linear
            self.embed = layer(self.num_classes, self.num_classes)

        if self.encoder_num_images > 0:
            self.encoder = BigEncoder(num_images=self.encoder_num_images,
                                      image_size=self.encoder_image_size,
                                      ch=ch)

        num_latents = self.__get_num_latents()
        
        blocks = []
        w = 4
        from_ch = ch
        to_ch = from_ch
        
        self.fc = nn.Linear(num_latents[0], from_ch*4*4, bias=False)
        idx = 1
        while w < self.image_size:
            assert to_ch % 2 == 0
            blocks.append(GBlock(from_ch, to_ch,
                                 num_latents[idx],
                                 from_ch if self.encoder_num_images > 0 else 0 + (a_shape_list[idx - 1] if a_shape_list is not None else 0),
                                 upsample=True))
            w *= 2
            
            from_ch = to_ch
            to_ch //= 2
            idx += 1
        
        assert from_ch >= self.num_images * 4
            
        self.blocks = nn.ModuleList(blocks)
        
        self.bn = nn.BatchNorm2d(from_ch)
        self.relu = nn.ReLU()
        self.conv_last = nn.Conv2d(from_ch, self.num_images, kernel_size=3, padding=1, bias=False)
        self.tanh = bottleneck
        
        nn.init.constant_(self.bn.weight.data, 1.0)
        nn.init.constant_(self.bn.bias.data, 0.0)
    
    def __get_num_latents(self):
        xs = torch.empty(self.latent_dim).chunk(self.num_chunk)
        num_latents = [x.size(0) for x in xs]
        for i in range(0, self.num_chunk):
            num_latents[i] += self.num_classes
        return num_latents
    
    def forward(self, x=None, y=None, z=None, a=None):
        if self.num_classes > 0:
            assert y is not None
            if not self.sparse_num_classes:
                assert len(y.shape) == 2
                assert y.shape[1] == self.num_classes
            else:
                assert len(y.shape) == 1
            y = self.embed(y)

        if self.encoder_num_images > 0:
            assert z is not None
            encoded = self.encoder(z)
        else:
            encoded = []

        if x is not None:
            xs = x.chunk(self.num_chunk, dim=1)
        if self.num_classes > 0:
            if self.latent_dim > 0:
                h = self.fc(torch.cat([y, xs[0]], dim=1))
            else:
                h = self.fc(y)
        else:
            h = self.fc(xs[0])
        h = h.view(h.shape[0], self.ch, 4, 4)
        
        i = 0
        j = len(encoded) - 1
        k = 0
        a_i = 0
        w = 4
        while i < len(self.blocks):
            if self.num_classes > 0:
                if self.latent_dim > 0:
                    cond = torch.cat([y, xs[i + 1]], dim=1)
                else:
                    cond = y
            else:
                cond = xs[i + 1]
                
            cond_z = encoded[j] if len(encoded) > 0 else None
            if a is not None:
                if cond_z is None:
                    cond_z = a[a_i]
                else:
                    cond_z = torch.cat([cond_z, a[a_i]], 1)
            
            h = self.blocks[i](h, cond, cond_z)
                
            if self.blocks[i].upsample:
                w *= 2
            i += 1
            if j >= 0:
                j -= 1
            
            a_i += 1
        
        assert i == len(self.blocks) and j == max(len(encoded) - len(self.blocks) - 1, -1)
        
        return self.tanh(self.conv_last(self.relu(self.bn(h))))
