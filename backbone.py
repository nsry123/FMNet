from modules import conv3x3, conv1x1, UpSampleLayer

class UBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(UBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        last = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            last = self.downsample(x)

        out += last
        out = self.relu(out)

        return out
    
class FM(nn.Module):
    def __init__(self, block, layers, output_stride, pretrained=False):
        self.inplanes = 128
        super(UNet, self).__init__()
        if output_stride == 16:
            strides = [2, 2, 1]
            dilations = [1, 1, 2]
        elif output_stride == 8:
            strides = [2, 1, 1]
            dilations = [1, 2, 4]
        else:
            raise NotImplementedError
        
        channels = [128,256,512,1024,1024]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0,bias=False)
        self.conv3 = nn.Conv2d(512,512, kernel_size=3, stride=2, padding=1,bias=False)
        self.layer0 = nn.Sequential(self.conv1,self.bn1,self.relu,self.maxpool)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.edge = nn.Parameter(torch.zeros((2,1,256,256)))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[0], dilation=dilations[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[1], dilation=dilations[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[2], dilation=dilations[2])
        
        
        #c,h,index,in_channels
        self.p1 = Adaptive_w(256, 128, 1, channels)
        self.p2 = Adaptive_w(512, 64, 2, channels)
        self.p3 = Adaptive_w(1024, 32, 3, channels)
        self.p4 = Adaptive_w(1024, 32, 4, channels)
        
        self.g1 = Pointw_guid(256, 128, 1, channels)
        self.g2 = Pointw_guid(512, 64, 2, channels)
        self.g3 = Pointw_guid(1024, 32, 3, channels)
        self.g4 = Pointw_guid(1024, 32, 4, channels)
        
        self.up1 = UpSampleLayer(1024,1024)
        self.up2 = UpSampleLayer(2048,512)
        self.up3 = UpSampleLayer(1024,256)
        self.up4 = UpSampleLayer(512,128)
        
        self.convb = nn.Conv2d(256,1,1)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        a1 = self.layer0(x)
        a2 = self.layer1(a1)
        a3 = self.layer2(a2)
        a4 = self.layer3(a3)
        a5 = self.layer4(a4)
        a5 = self.conv2(a5)
        
        edge1 = F.interpolate(self.edge,mode='bilinear',size=(128,128))
        edge2 = F.interpolate(self.edge,mode='bilinear',size=(64,64))
        edge3 = F.interpolate(self.edge,mode='bilinear',size=(32,32))
        edge4 = F.interpolate(self.edge,mode='bilinear',size=(32,32))
        
        a2 = self.g1(a2,a3,a4,a5)
        a3 = self.g2(a3,a2,a4,a5)
        a4 = self.g3(a4,a2,a3,a5)
        a5 = self.g4(a5,a2,a3,a4)
        
        a2 = self.p1(a2,a3,a4,a5)+edge1
        a3 = self.p2(a3,a2,a4,a5)+edge2
        a4 = self.p3(a4,a2,a3,a5)+edge3
        a5 = self.p4(a5,a2,a3,a4)+edge4
        
        out1 = self.up1(a5,a4)
        out2 = self.up2(out1,a3)
        out3 = self.up3(out2,a2)
        out3 = self.conv3(out3)
        out4 = self.up4(out3,a1)
        
        return out4
