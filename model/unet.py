#这是一个2d_unet卷积模型
import torch
import torch.nn as nn

#这是 PyTorch 中用于定义神经网络模块的基类。
#继承它可以方便地将自定义的模块集成到 PyTorch 的神经网络架构中。
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(   #将一系列的神经网络层按顺序组合在一起
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),#这个参数数值设置使得卷积操作后图像尺寸不变
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)

#该类用于构建整个神经网络模型。
class Model(nn.Module):
    def __init__(self, in_features, out_features, init_features):
        super(Model, self).__init__()
        features = init_features
        #初始化一个变量 features 为传入的 init_features
        #这个变量将用于在构建网络各层时确定通道数等参数

        #1.定义一系列编码层:用于捕捉不同尺度的特征信息
        self.encode_layer1 = UNetBlock(in_features, features)
        self.encode_layer2 = UNetBlock(features, features * 2)
        self.encode_layer3 = UNetBlock(features * 2, features * 4)
        self.encode_layer4 = UNetBlock(features * 4, features * 8)
        self.encode_decode_layer = UNetBlock(features * 8, features * 16)

        #2.定义最大池化过程
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #3.定义一系列上卷积层（上采样）
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)

        #4.定义一系列解码层
        self.decode_layer4 = UNetBlock(features * 16, features * 8)
        self.decode_layer3 = UNetBlock(features * 8, features * 4)
        self.decode_layer2 = UNetBlock(features * 4, features * 2)
        self.decode_layer1 = UNetBlock(features * 2, features)

        #5.定义输出层
        self.out_layer = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid()#通过Sigmoid函数将输出值映射0-1范围内
        )

    #定义了整个模型的前向传播逻辑
    def forward(self, x):

        #1.编码阶段
        enc1 = self.encode_layer1(x)
        enc2 = self.encode_layer2(self.pool(enc1))
        enc3 = self.encode_layer3(self.pool(enc2))
        enc4 = self.encode_layer4(self.pool(enc3))
        # 将最深层的编码结果 enc4 先经过池化层进行最后一次下采样，
        # 然后传入 self.encode_decode_layer 进行进一步的特征提取
        # 得到瓶颈层（bottleneck）的结果，这是编码阶段的最后一步，此时数据的特征被高度抽象化。
        bottleneck = self.encode_decode_layer(self.pool(enc4))

        #2.解码阶段
        # ①上采样
        dec4 = self.upconv4(bottleneck)
        # ②跳越拼接：
        # 将上采样后的结果 dec4 与对应的编码层结果 enc4 在通道维度（dim=1）上进行拼接，
        # 这样可以将在编码过程中丢失的部分信息重新组合起来，以便后续更好地还原数据。
        dec4 = torch.cat((dec4, enc4), dim=1)
        # ③解码层
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decode_layer1(dec1)

        #3. 输出阶段
        out = self.out_layer(dec1)
        
        return out


    '''关于一些模块功能区分
    1.上采样层
    ①功能：
    上采样层主要用于增加数据的空间尺寸（如二维图像的高度和宽度），为后续的操作提供合适的空间分辨率，使得网络能够在更高分辨率的特征图上进行操作。
    *在 U - Net 架构中，它帮助恢复在编码阶段丢失的空间信息，为最终的像素级预测（如分割任务中的每个像素的类别预测）提供足够的细节。
    ②解释：
    在深度学习的架构中，特别是像 U - Net 这种用于图像分割的网络，编码过程通常会包含下采样操作（如使用池化层）来减小图像尺寸并提取更抽象的特征。
    上采样层的作用就是逆转这个过程的一部分，将经过下采样后变小的特征图还原到较大的尺寸。
    例如，在代码中的 nn.ConvTranspose2d 层就是一种上采样层，它通过转置卷积（Transposed Convolution）的方式来实现上采样。
    
    2.解码层
    ①功能：
    用于对经过上采样后的特征进行处理和转换。
    解码层的重点在于对特征的语义信息进行调整和优化，而不仅仅是改变数据的空间尺寸。
    解码层的目的是将上采样后的特征图转换为更适合最终输出的形式。
    ②解释：
    以包含卷积操作的解码层为例，它会根据卷积核在特征图上滑动，对每个局部区域进行加权求和，从而提取新的特征。
    这些特征会经过归一化层（如批量归一化）来稳定训练过程，然后通过激活函数（如 ReLU）引入非线性特性。
    在这个过程中，解码层会综合考虑上采样后的空间信息以及从编码阶段传递过来的特征信息，对特征进行重新组合和优化。
    
    在图像分割任务中，它要将特征图中的语义信息（如物体的边界、类别等）进行细化和调整，以便输出层能够准确地生成分割结果。
    它不仅仅是恢复空间分辨率，还涉及到对特征的语义理解和转换，与上采样层的功能相互补充，共同完成从抽象特征到最终输出的转换过程。
    '''