""" Script for attentional 2D convolution layer.
"""
import torch
from torch import nn
from typing import Union, Tuple

import pdb


class AttentionConv2D(nn.Module):
  def __init__(self,
              in_channels: int,
              out_channels: int,
              kernel_size: Union[int, Tuple[int, int]],
              stride: Union[int, Tuple[int, int]] = 1,
              padding: Union[int, Tuple[int, int]] = 0,
              dilation: Union[int, Tuple[int, int]] = 1,
              groups: int = 1,
              bias: bool = True,
              padding_mode: str = 'zeros',
              input_size: Union[int, Tuple[int, int]] = None,
              runtime_mesh: bool = True,
              double_mesh: bool = True,
              return_attention: bool = False) -> None:
    super(AttentionConv2D, self).__init__()
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._kernel_size = kernel_size
    self._stride = stride
    self._padding = padding
    self._dilation = dilation
    self._groups = groups
    self._bias = bias
    self._padding_mode = padding_mode
    self._input_size = input_size
    self._runtime_mesh = runtime_mesh
    self._double_mesh = double_mesh
    self._return_attention = return_attention

    self._conv_a = nn.Conv2d(
        self._in_channels, self._out_channels, self._kernel_size,
        stride=self._stride, padding=self._padding, dilation=self._dilation,
        groups=self._groups, bias=self._bias, padding_mode=self._padding_mode)
    self._conv_b = nn.Conv2d(
        self._in_channels, self._out_channels, self._kernel_size,
        stride=self._stride, padding=self._padding, dilation=self._dilation,
        groups=self._groups, bias=self._bias, padding_mode=self._padding_mode)

    #add channel 2
    conv_att_in_channels = self._in_channels + 4 if self._double_mesh \
        else self._in_channels
    self._conv_att = nn.Conv2d(conv_att_in_channels, 1, 1)
    # add channel 2
    self._conv_attention = nn.Sequential(
        nn.Conv2d(7, 1, self._kernel_size, stride=self._stride,
            padding=self._padding, dilation=self._dilation),
        nn.BatchNorm2d(1))
    self._sigmoid = nn.Sigmoid()

    if not self._runtime_mesh:
      assert self._input_size != None
      if isinstance(self._input_size, int):
        h, w = self._input_size, self._input_size
      elif isinstance(self._input_size, tuple):
        h, w = self._input_size
      else:
        raise ValueError("Invalid input_size.")
      lin_h = torch.linspace(0, h - 1, steps=h).cuda()
      lin_w = torch.linspace(0, w - 1, steps=w).cuda()
      self._grid_h, self._grid_w = torch.meshgrid(lin_h, lin_w)
      # print(self._grid_h, self._grid_w)
      self._polar_r = torch.sqrt(torch.square(self._grid_h - h // 2) + torch.square(self._grid_w))
      self._polar_r = self._polar_r / float(self._polar_r[h - 1, w - 1])
      self._polar_theta = torch.abs(torch.arctan((self._grid_h - h // 2) / self._grid_w) / torch.arctan((self._grid_h[0][0] - h//2) / self._grid_w[0][0]))
      self._polar_theta[h//2][0] = 0
      # for i in range(self._input_size[0]):
      #   print(self._polar_theta[i])
      if h % 2 == 0:
        self._grid_h = torch.abs(self._grid_h - h // 2 + 0.5)
      else:
        self._grid_h = torch.abs(self._grid_h - h // 2)
      self._grid_h = self._grid_h / float(h // 2)
      self._grid_w = self._grid_w / float(w)
      output_size = self.get_output_size()
      self._range_map = torch.sqrt(
          self._grid_h * self._grid_h + self._grid_w * self._grid_w)
      self._range_map = self._range_map / torch.max(self._range_map)
      self._range_map = self._range_map.unsqueeze(0).unsqueeze(0)
      self._range_map = nn.functional.interpolate(
          self._range_map, output_size)

    self._global_count = 0

  def forward(self, x, fix_attention=False):
    """
      Forard function for Conv2D.
      Parameters:
      ----------
      fix_attention: bool
        Fix attention map if turned on.
    """
    if self._runtime_mesh:
      h, w = x.shape[2], x.shape[3]

      lin_h = torch.linspace(0, h - 1, steps=h).cuda()
      lin_w = torch.linspace(0, w - 1, steps=w).cuda()
      grid_h, grid_w = torch.meshgrid(lin_h, lin_w)

      if h % 2 == 0:
        grid_h = torch.abs(grid_h - h // 2 + 0.5)
      else:
        grid_h = torch.abs(grid_h - h // 2)
      grid_h = grid_h / float(h // 2)
      grid_w = grid_w / float(w)

      polar_r = torch.sqrt(torch.square(grid_h) + torch.square(grid_w))
      polar_r = polar_r / float(polar_r[h - 1, w - 1])
      polar_theta = torch.abs(torch.arctan( grid_h / grid_w) / torch.arctan(
        (grid_h[0][0] - h // 2) / grid_w[0][0]))
      if h%2 != 0:
        polar_theta[h // 2][0] = 0

    else:
      grid_h, grid_w = self._grid_h, self._grid_w
      polar_r, polar_theta = self._polar_r, self._polar_theta

    grid_h_batch = grid_h.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)
    grid_w_batch = grid_w.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)
    polar_r_batch = polar_r.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)
    polar_theta_batch = polar_theta.unsqueeze(0).unsqueeze(0).repeat(
      x.shape[0], 1, 1, 1)
    # print(grid_h_batch)

    if not self._double_mesh:
      conv_attr = self._conv_att(x)
    else:
      # print(grid_h_batch.shape, polar_theta_batch.shape)
      conv_attr = self._conv_att(
          torch.cat([x, grid_h_batch, grid_w_batch, polar_r_batch, polar_theta_batch], dim=1))

    if not fix_attention:
      attention_maps = torch.cat(
          [conv_attr,
          torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1),
          grid_h_batch, grid_w_batch, polar_r_batch, polar_theta_batch], dim=1)
      attention = self._conv_attention(attention_maps)
      attention = self._sigmoid(attention).repeat(1, self._out_channels, 1, 1)
    else:
      if self._runtime_mesh:
        output_size = self._get_output_size(x.shape[-2:])
        range_map = torch.sqrt(grid_h * grid_h + grid_w * grid_w)
        range_map = range_map / torch.max(range_map)
        range_map = range_map.unsqueeze(0).unsqueeze(0)
        range_map = nn.functional.interpolate(range_map, output_size)
      else:
        range_map = self._range_map
      attention = range_map.repeat(x.shape[0], self._out_channels, 1, 1)

    conv_a = self._conv_a(x)
    conv_b = self._conv_b(x)
    out = attention * conv_a + (1. - attention) * conv_b
    if self._return_attention:
      return out, attention
    else:
      return out

  def get_output_size(self):
    """ Get size of output tensor.
    """
    if type(self._input_size) is not tuple:
      input_size = (self._input_size, self._input_size)
    else:
      input_size = self._input_size
    return self._get_output_size(input_size)

  def _get_output_size(self, input_size):
    if type(self._kernel_size) is not tuple:
      kernel_size = (self._kernel_size, self._kernel_size)
    else:
      kernel_size = self._kernel_size

    if type(self._stride) is not tuple:
      stride = (self._stride, self._stride)
    else:
      stride = self._stride

    if type(self._padding) is not tuple:
      pad = (self._padding, self._padding)
    else:
      pad = self._pad

    if type(self._dilation) is not tuple:
      dilation = (self._dilation, self._dilation)
    else:
      dilation = self._dilation

    out_size_h = (input_size[0] + (2 * pad[0]) - \
        (dilation[0] * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    out_size_w = (input_size[1] + (2 * pad[1]) - \
        (dilation[1] * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    if type(input_size) is int:
      return out_size_h
    else:
      return (out_size_h, out_size_w)


class AttentionTransposeConv2D(nn.Module):
  def __init__(self,
              in_channels: int,
              out_channels: int,
              kernel_size: Union[int, Tuple[int, int]],
              stride: Union[int, Tuple[int, int]] = 1,
              padding: Union[int, Tuple[int, int]] = 0,
              output_padding: Union[int, Tuple[int, int]] = 0,
              dilation: Union[int, Tuple[int, int]] = 1,
              groups: int = 1,
              bias: bool = True,
              padding_mode: str = 'zeros',
              input_size: Union[int, Tuple[int, int]] = None,
              runtime_mesh: bool = True,
              double_mesh: bool = True,
              return_attention: bool = False) -> None:
    super(AttentionConv2D, self).__init__()
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._kernel_size = kernel_size
    self._stride = stride
    self._padding = padding
    self._dilation = dilation
    self._groups = groups
    self._bias = bias
    self._padding_mode = padding_mode
    self._input_size = input_size
    self._runtime_mesh = runtime_mesh
    self._double_mesh = double_mesh
    self._return_attention = return_attention

    self._conv_a = nn.Conv2d(
        self._in_channels, self._out_channels, self._kernel_size,
        stride=self._stride, padding=self._padding, dilation=self._dilation,
        groups=self._groups, bias=self._bias, padding_mode=self._padding_mode)
    self._conv_b = nn.Conv2d(
        self._in_channels, self._out_channels, self._kernel_size,
        stride=self._stride, padding=self._padding, dilation=self._dilation,
        groups=self._groups, bias=self._bias, padding_mode=self._padding_mode)

    conv_att_in_channels = self._in_channels + 2 if self._double_mesh \
        else self._in_channels
    self._conv_att = nn.Conv2d(conv_att_in_channels, 1, 1)
    self._conv_attention = nn.Sequential(
        nn.Conv2d(5, 1, self._kernel_size, stride=self._stride,
            padding=self._padding, dilation=self._dilation),
        nn.BatchNorm2d(1))
    self._sigmoid = nn.Sigmoid()

    if not self._runtime_mesh:
      assert self._input_size != None
      if isinstance(self._input_size, int):
        h, w = self._input_size, self._input_size
      elif isinstance(self._input_size, tuple):
        h, w = self._input_size
      else:
        raise ValueError("Invalid input_size.")
      lin_h = torch.linspace(0, h - 1, steps=h).cuda()
      lin_w = torch.linspace(0, w - 1, steps=w).cuda()
      self._grid_h, self._grid_w = torch.meshgrid(lin_h, lin_w)
      if h % 2 == 0:
        self._grid_h = torch.abs(self._grid_h - h // 2 + 0.5)
      else:
        self._grid_h = torch.abs(self._grid_h - h // 2)
      self._grid_h = self._grid_h / float(h // 2)
      self._grid_w = self._grid_w / float(w)

      output_size = self.get_output_size()
      self._range_map = torch.sqrt(
          self._grid_h * self._grid_h + self._grid_w * self._grid_w)
      self._range_map = self._range_map / torch.max(self._range_map)
      self._range_map = self._range_map.unsqueeze(0).unsqueeze(0)
      self._range_map = nn.functional.interpolate(
          self._range_map, output_size)

    self._global_count = 0

  def forward(self, x, fix_attention=False):
    """
      Forard function for Conv2D.
      Parameters:
      ----------
      fix_attention: bool
        Fix attention map if turned on.
    """
    if self._runtime_mesh:

      h, w = x.shape[2], x.shape[3]
      lin_h = torch.linspace(0, h - 1, steps=h).cuda()
      lin_w = torch.linspace(0, w - 1, steps=w).cuda()
      grid_h, grid_w = torch.meshgrid(lin_h, lin_w)
      if h % 2 == 0:
        grid_h = torch.abs(grid_h - h // 2 + 0.5)
      else:
        grid_h = torch.abs(grid_h - h // 2)
      grid_h = grid_h / float(h // 2)
      grid_w = grid_w / float(w)
    else:

      grid_h, grid_w = self._grid_h, self._grid_w
    grid_h_batch = grid_h.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)
    grid_w_batch = grid_w.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)

    if not self._double_mesh:
      conv_attr = self._conv_att(x)
    else:
      conv_attr = self._conv_att(
          torch.cat([x, grid_h_batch, grid_w_batch], dim=1))

    if not fix_attention:
      attention_maps = torch.cat(
          [conv_attr,
          torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1),
          grid_h_batch, grid_w_batch], dim=1)
      attention = self._conv_attention(attention_maps)
      attention = self._sigmoid(attention).repeat(1, self._out_channels, 1, 1)
    else:
      if self._runtime_mesh:
        output_size = self._get_output_size(x.shape[-2:])
        range_map = torch.sqrt(grid_h * grid_h + grid_w * grid_w)
        range_map = range_map / torch.max(range_map)
        range_map = range_map.unsqueeze(0).unsqueeze(0)
        range_map = nn.functional.interpolate(range_map, output_size)
      else:
        range_map = self._range_map
      attention = range_map.repeat(x.shape[0], self._out_channels, 1, 1)

    conv_a = self._conv_a(x)
    conv_b = self._conv_b(x)
    out = attention * conv_a + (1. - attention) * conv_b
    if self._return_attention:
      return out, attention
    else:
      return out

  def get_output_size(self):
    """ Get size of output tensor.
    """
    if type(self._input_size) is not tuple:
      input_size = (self._input_size, self._input_size)
    else:
      input_size = self._input_size
    return self._get_output_size(input_size)

  def _get_output_size(self, input_size):
    if type(self._kernel_size) is not tuple:
      kernel_size = (self._kernel_size, self._kernel_size)
    else:
      kernel_size = self._kernel_size

    if type(self._stride) is not tuple:
      stride = (self._stride, self._stride)
    else:
      stride = self._stride

    if type(self._padding) is not tuple:
      pad = (self._padding, self._padding)
    else:
      pad = self._pad

    if type(self._dilation) is not tuple:
      dilation = (self._dilation, self._dilation)
    else:
      dilation = self._dilation

    out_size_h = (input_size[0] + (2 * pad[0]) - \
        (dilation[0] * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    out_size_w = (input_size[1] + (2 * pad[1]) - \
        (dilation[1] * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    if type(input_size) is int:
      return out_size_h
    else:
      return (out_size_h, out_size_w)



if __name__ == "__main__":
  # print(10.52//2)
  att_conv = AttentionConv2D(10, 20, 3).cuda()
  # att_conv = AttentionConv2D(10, 20, 3, input_size=(45, 65),
  #     runtime_mesh=False, double_mesh=True).cuda()
  print(att_conv(torch.randn(4, 10, 44, 65).cuda(), fix_attention=False))