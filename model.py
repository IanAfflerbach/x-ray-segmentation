import torch

class SegmentationModel(torch.nn.Module):
    class ConvBlock(torch.nn.Module):
        def __init__(self, n_input_channel, n_output_channel):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(n_input_channel, n_output_channel, kernel_size=3, padding=1),
                # TODO - DropBlock
                torch.nn.BatchNorm2d(n_output_channel),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.network(x)

    class EncoderBlock(torch.nn.Module):
        def __init__(self, n_input_channel, n_output_channel):
            super().__init__()

            self.conv1 = SegmentationModel.ConvBlock(n_input_channel, n_output_channel)
            self.conv2 = SegmentationModel.ConvBlock(n_output_channel, n_output_channel)

        def forward(self, x):
            return self.conv2(self.conv1(x))
        
    class DecoderBlock(torch.nn.Module):
        def __init__(self, n_input_channels, n_output_channels):
            super().__init__()

            self.trans = torch.nn.ConvTranspose2d(n_input_channels, n_output_channels, 2, stride=2)
            self.conv1 = SegmentationModel.ConvBlock(n_input_channels, n_output_channels)
            self.conv2 = SegmentationModel.ConvBlock(n_output_channels, n_output_channels)

        def forward(self, x, x_copy):
            z_trans = self.trans(x)
            z_cat = torch.cat((x_copy, z_trans), dim=1)
            return self.conv2(self.conv1(z_cat))

    
    def __init__(self):
        super().__init__()
        # encoder convolutions
        self.down1 = self.EncoderBlock(3, 16)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.down2 = self.EncoderBlock(16, 32)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.down3 = self.EncoderBlock(32, 64)
        self.pool3 = torch.nn.MaxPool2d(2)
        
        # bottom layer convolutions
        self.down_final = self.ConvBlock(64, 128)
        # TODO - Spatial Attention Module
        self.up_first = self.ConvBlock(128, 128)

        # decoder convolutions
        self.up1 = self.DecoderBlock(128, 64)
        self.up2 = self.DecoderBlock(64, 32)
        self.up3 = self.DecoderBlock(32, 16)

        # output layer
        self.conv_final = torch.nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # print("X", x.shape)
        z_down1 = self.down1(x)
        # print("Down 1", z_down1.shape)
        z_down2 = self.down2(self.pool1(z_down1))
        # print("Down 2", z_down2.shape)
        z_down3 = self.down3(self.pool2(z_down2))
        # print("Down 3", z_down3.shape)

        z = self.down_final(self.pool3(z_down3))
        # print("Before SAM", z.shape)
        z = self.up_first(z)
        # print("After SAM", z.shape)

        z = self.up1(z, z_down3)
        # print("Up 1", z.shape)
        z = self.up2(z, z_down2)
        # print("Up 2", z.shape)
        z = self.up3(z, z_down1)
        # print("Up 3", z.shape)

        z = self.conv_final(z)
        # print("1x1 Conv", z.shape)
        z = self.sigmoid(z)
        # print("Sigmoid", z.shape)

        return z