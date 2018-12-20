class conv_bn(torch.nn.Module):
    def __init__(self, inp_ch, outp_ch):
        super(conv_bn, self).__init__()
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inp_ch, outp_ch, 3, padding=1),
                torch.nn.BatchNorm2d(outp_ch),
                torch.nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class encode(torch.nn.Module):
    def __init__(self, inp_ch, outp_ch):
        super(encode, self).__init__()
        self.mpconv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            conv_bn(inp_ch, outp_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

# Unet based, can change
class decode(torch.nn.Module):
    def __init__(self, inp_ch, outp_ch, bilinear=False):
        super(decode, self).__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = torch.nn.ConvTranspose2d(inp_ch//2, inp_ch//2, 2, stride=2)
        self.conv = conv_bn(inp_ch, outp_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        x = x1
        if(type(x2) == torch.Tensor):
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

        # self.inc = conv_bn(n_channels, 64)
        # self.e1 = encode(64, 128)
        # self.e2 = encode(128, 256)
        # self.e3 = encode(256, 256)
        # #self.e4 = encode(512, 512)
        # #self.d1 = decode(1024, 256)
        # self.d2 = decode(512, 128)
        # self.d3 = decode(256, 64)
        # self.d4 = decode(128, 64)
        # self.outc = torch.nn.Conv2d(64, n_channels, 1)

                #self.e4 = encode(512, 512)
        #self.d1 = decode(1024, 256)
                print(h1.size(), h2.size(), h3.size(), h4.size())
        #h5 = self.e4(h4)
        #h = self.d1(h5, h4)
