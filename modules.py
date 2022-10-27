from backbone import FM, UBlock

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def makeFM(output_stride=16, pretrained=False):
    
    model = FM(UBlock, [3, 4, 23, 3], output_stride, pretrained=pretrained)
    return model

  

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):

        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch*2,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        
    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        #print(x.shape)
        x_out=self.Conv_BN_ReLU_2(x)
        #print(x_out.shape)
        #print(out.shape)
        cat_out=torch.cat((x_out,out),dim=1)
        x_out=self.upsample(cat_out)
        return x_out
