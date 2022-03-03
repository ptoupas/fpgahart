import numpy as np
import random
import torch
from torch import nn
from collections import deque
import random
import argparse
import os
import sys

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

np.set_printoptions(suppress=True, precision=5, threshold=sys.maxsize)

def create_queue(data_in):
  batch, height, width, depth, channel = data_in.shape
  data_stream = deque([])

  for b in range(batch):
    for h in range(height):
      for w in range(width):
        for d in range(depth):
          for c in range(channel):
            # print(f"{b}, {h}, {w}, {d}, {c} = {data_in[b, h, w, d, c]}")
            data_stream.appendleft(data_in[b, h, w, d, c])
  
  return data_stream

def show_wb(window_buffer):
  print("\nWindow Buffer")
  print(f"[[[{list(window_buffer[0][0][0])}, {list(window_buffer[0][1][0])}, {list(window_buffer[0][2][0])}]")
  print(f"  [{list(window_buffer[1][0][0])}, {list(window_buffer[1][1][0])}, {list(window_buffer[1][2][0])}]")
  print(f"  [{list(window_buffer[2][0][0])}, {list(window_buffer[2][1][0])}, {list(window_buffer[2][2][0])}]]")
  print()
  print(f"[[[{list(window_buffer[0][0][1])}, {list(window_buffer[0][1][1])}, {list(window_buffer[0][2][1])}]")
  print(f"  [{list(window_buffer[1][0][1])}, {list(window_buffer[1][1][1])}, {list(window_buffer[1][2][1])}]")
  print(f"  [{list(window_buffer[2][0][1])}, {list(window_buffer[2][1][1])}, {list(window_buffer[2][2][1])}]]")
  print()
  print(f" [[{0}, {0}, {0}]")
  print(f"  [{0}, {0}, {0}]")
  print(f"  [{0}, {0}, {0}]]]")

def show_lb(line_buffer, extra_buffer):
  print("\nLine Buffer 0")
  print(list(line_buffer[0][0]), len(line_buffer[0][0]))
  print(list(line_buffer[0][1]), len(line_buffer[0][1]))

  print("\nLine Buffer 1")
  print(list(line_buffer[1][0]), len(line_buffer[1][0]))
  print(list(line_buffer[1][1]), len(line_buffer[1][1]))

  print("\nLine Buffer 2")
  print(list(line_buffer[2][0]), len(line_buffer[2][0]))
  print(list(line_buffer[2][1]), len(line_buffer[2][1]))

  print("\nExtra Buffer 0")
  print(list(extra_buffer[0]), len(extra_buffer[0]))

  print("\nExtra Buffer 1")
  print(list(extra_buffer[1]), len(extra_buffer[1]))

def showres(out_buffer, shape, file_format, kh=3, kw=3, kd=3, pad_h=0, pad_w=0, pad_d=0, stride_h=1, stride_w=1, stride_d=1):
  print("\nInterprenting SW Output results\n")
  results = []
  batch, channel, depth, height, width = shape

  for b in range(0, batch):
    for h in range(0, height+2*pad_h):
      for w in range(0, width+2*pad_w):
        for d in range(0, depth+2*pad_d):
          for c in range(0, channel):
            print(f"b={b}  |  row={h}  |  col={w}  |  d={d}  |  c={c}")

            if not ((h < kh-1) or (h==kh-1 and w<kw-1) or (h==kh-1 and w==kw-1 and d<kd-1)):
              for i in range(kh):
                for j in range(kw):
                  for k in range(kd):
                    # for c in range(0, channel): # for 1 to 1 comparisson with HLS
                    tmp = out_buffer[i][j][k].pop()
                    if d >= kd-1 and (d-kd+1)%stride_d == 0 and w >= kw-1 and (w-kw+1)%stride_w == 0 and h >= kh-1 and (h-kh+1)%stride_h == 0:
                      results.append(tmp)
                      if k == kd-1:
                        print(f"{tmp:.4f}")
                      else:
                        print(f"{tmp:.4f}", end=' ')
                if d >= kd-1 and w >= kw-1 and h >= kh-1:
                  print()
              if d >= kd-1 and w >= kw-1 and h >= kh-1:
                print("*"*30)

  for k in range(kd):
    for i in range(kh):
      for j in range(kw):
        assert len(out_buffer[i][j][k]) == 0, "There are still values in buffers"

  write_output_binary = np.array(results, dtype=np.float32)
  file_format = "txt"
  if file_format == "bin":
    write_output_binary.tofile("sw_output.dat")
  elif file_format == "txt":
    np.savetxt("sw_output.dat", write_output_binary, fmt='%.8f')
  else:
    raise Exception("Format not supported")

def array_init(arr):
	prev = arr
	for i in range(len(arr.shape)-1):
		curr = np.zeros((prev.shape[0:-1]),dtype='object')
		for index,_ in np.ndenumerate(curr):
			if i==0:
				curr[index] ="\t"*len(curr.shape) + "{" + ",".join([ str(val) for val in prev[index]]) +"}"
			else:
				curr[index] ="\t"*len(curr.shape) + "{\n" + ",\n".join([ str(val) for val in prev[index]]) + "\n" + "\t"*len(curr.shape) + "}"
		prev = curr
	return ",\n".join([i for i in prev]) 

def transform_weights(weights_raw,coarse_in,coarse_out,wr_factor=1,coarse_group=1,groups=1):
	# parameters
	num_filters  = int(weights_raw.shape[0]/(groups*coarse_out*wr_factor))
	num_channels = int(weights_raw.shape[1]/coarse_in)
	k_size_k       = weights_raw.shape[2]
	k_size_x       = weights_raw.shape[3]
	k_size_y       = weights_raw.shape[4]
	print(f"num_filters={num_filters}  |  num_channels={num_channels}  |  k_size_k={k_size_k}  |  k_size_x={k_size_x}  |  k_size_y={k_size_y}")
	# correct output shape for weights
	weights = np.ndarray(
		shape=(
			wr_factor,
			coarse_group,
			coarse_in,
			coarse_out,
			int(groups/coarse_group),
			num_channels,
			num_filters,
			k_size_x,k_size_y,k_size_k),dtype=float,order='C')

	# transform weights raw shape
	for index,_ in np.ndenumerate(weights):
		# print(f"index={index}, weights_raw[{index[4]*coarse_group*num_filters*wr_factor*coarse_out+index[1]*num_filters*wr_factor*coarse_out+index[6]*wr_factor*coarse_out+index[0]*coarse_out+index[3]}][{index[5]*coarse_in+index[2]}][{index[9]}][{index[7]}][{index[8]}]")
		weights[index] = weights_raw[
					index[4]*coarse_group*num_filters*wr_factor*coarse_out+index[1]*num_filters*wr_factor*coarse_out+index[6]*wr_factor*coarse_out+index[0]*coarse_out+index[3],
					index[5]*coarse_in+index[2],
					index[9],
					index[7],
					index[8]]
	# merge channel and filter dimensions
	print("*"*30)
	print(weights.shape)
	print([wr_factor,coarse_in*coarse_group,coarse_out,int(groups/coarse_group)*num_channels,num_filters,k_size_x,k_size_y,k_size_k])
	weights = np.reshape(weights,[wr_factor,coarse_in*coarse_group,coarse_out,int(groups/coarse_group)*num_channels,num_filters,k_size_x,k_size_y,k_size_k])
	# print(weights)
	print(weights.shape)
	print("*"*30)
	# return transformed weights
	return weights

def gap_3d(input_shape, coarse_in, coarse_out, file_format, prefix="data"):
	if not os.path.exists(prefix + '/gap_3d'):
		os.makedirs(prefix + '/gap_3d')

	x = torch.randn(input_shape)
	print(x.numpy())
	print(x.numpy().shape)
	write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary.tofile(prefix + "/gap_3d/input.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/gap_3d/input.dat", write_input_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	gap = torch.nn.AdaptiveAvgPool3d(1)
	out = gap(x)

	print(out.detach().numpy())
	print(out.detach().numpy().shape)

	write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/gap_3d/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/gap_3d/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

def elemwise_3d(input_shape, input_shape_2, coarse_in, elemwise_op_type, file_format, prefix="data"):
	if not os.path.exists(prefix + '/elemwise_3d'):
		os.makedirs(prefix + '/elemwise_3d')

	x = torch.randn(input_shape)
	# print(x.numpy())
	print(x.numpy().shape)
	y = torch.randn(input_shape_2)
	# print(y.numpy())
	print(y.numpy().shape)

	write_input_binary_1 = x.numpy().transpose(0, 3, 4, 2, 1)
	write_input_binary_2 = y.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary_1.tofile(prefix + "/elemwise_3d/input_1.dat")
		write_input_binary_2.tofile(prefix + "/elemwise_3d/input_2.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/elemwise_3d/input_1.dat", write_input_binary_1.flatten(), fmt='%.8f')
		np.savetxt(prefix + "/elemwise_3d/input_2.dat", write_input_binary_2.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	if elemwise_op_type == "add":
		out = torch.add(x, y)
	elif elemwise_op_type == "mul":
		out = torch.mul(x, y)
	# print(out.detach().numpy())
	print(out.detach().numpy().shape)

	write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/elemwise_3d/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/elemwise_3d/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

def shish_3d(input_shape, coarse_in, file_format, prefix="data"):
	if not os.path.exists(prefix + '/shish_3d'):
		os.makedirs(prefix + '/shish_3d')

	x = torch.randn(input_shape)
	print(x.numpy())
	print(x.numpy().shape)
	write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary.tofile(prefix + "/shish_3d/input.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/shish_3d/input.dat", write_input_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	sigmoid = torch.nn.Sigmoid()
	out = sigmoid(x)
	out = out * x

	print(out.detach().numpy())
	print(out.detach().numpy().shape)

	write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/shish_3d/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/shish_3d/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

def shish_3d(input_shape, coarse_in, file_format, prefix="data"):
	if not os.path.exists(prefix + '/shish_3d'):
		os.makedirs(prefix + '/shish_3d')

	x = torch.randn(input_shape)
	print(x.numpy())
	print(x.numpy().shape)
	write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary.tofile(prefix + "/shish_3d/input.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/shish_3d/input.dat", write_input_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	sigmoid = torch.nn.Sigmoid()
	out = sigmoid(x)
	out = out * x

	print(out.detach().numpy())
	print(out.detach().numpy().shape)

	write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/shish_3d/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/shish_3d/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

def sigmoid_3d(input_shape, coarse_in, file_format, prefix="data"):
	if not os.path.exists(prefix + '/sigmoid_3d'):
		os.makedirs(prefix + '/sigmoid_3d')

	x = torch.randn(input_shape)
	print(x.numpy())
	print(x.numpy().shape)
	write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary.tofile(prefix + "/sigmoid_3d/input.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/sigmoid_3d/input.dat", write_input_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	sigmoid = torch.nn.Sigmoid()
	out = sigmoid(torch.permute(x, (0, 3, 4, 2, 1)))

	print(out.detach().numpy())
	print(out.detach().numpy().shape)

	write_out_binary = out.detach().numpy()#.transpose(1, 0, 2, 3, 4)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/sigmoid_3d/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/sigmoid_3d/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")
		
def relu_3d(input_shape, coarse_in, file_format, prefix="data"):
	if not os.path.exists(prefix + '/relu_3d'):
		os.makedirs(prefix + '/relu_3d')

	x = torch.randn(input_shape)
	print(x.numpy())
	print(x.numpy().shape)
	write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary.tofile(prefix + "/relu_3d/input.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/relu_3d/input.dat", write_input_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	relu = torch.nn.ReLU()
	out = relu(torch.permute(x, (0, 3, 4, 2, 1)))

	print(out.detach().numpy())
	print(out.detach().numpy().shape)

	write_out_binary = out.detach().numpy()#.transpose(1, 0, 2, 3, 4)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/relu_3d/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/relu_3d/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

def part_3d(file_format, prefix):
	class X3d_m_layer(nn.Module):
		def __init__(self):
			super().__init__()

			channels_1 = 8
			filters_1 = 12
			kernel_shape_1 = (1, 1, 1)
			padding_1 = (0, 0, 0)
			stride_1 = (1, 1, 1)
			self.groups_1 = 1
			self.coarse_in_1 = 4
			self.coarse_out_1 = 6

			channels_2 = 12
			filters_2 = 12
			kernel_shape_2 = (3, 3, 3)
			padding_2 = (1, 1, 1)
			stride_2 = (1, 1, 1)
			self.groups_2 = 12
			self.coarse_in_2 = 6
			self.coarse_out_2 = 1

			channels_3 = 12
			filters_3 = 8
			kernel_shape_3 = (1, 1, 1)
			padding_3 = (0, 0, 0)
			stride_3 = (1, 1, 1)
			self.groups_3 = 1
			self.coarse_in_3 = 3
			self.coarse_out_3 = 4
			
			channels_4 = 8
			filters_4 = 12
			kernel_shape_4 = (1, 1, 1)
			padding_4 = (0, 0, 0)
			stride_4 = (1, 1, 1)
			self.groups_4 = 1
			self.coarse_in_4 = 4
			self.coarse_out_4 = 3

			channels_5 = 12
			filters_5 = 8
			kernel_shape_5 = (1, 1, 1)
			padding_5 = (0, 0, 0)
			stride_5 = (1, 1, 1)
			self.groups_5 = 1
			self.coarse_in_5 = 3
			self.coarse_out_5 = 4

			self.relu1 = nn.ReLU()
			self.conv1 = nn.Conv3d(channels_1, filters_1, kernel_shape_1, stride=stride_1, padding=padding_1, groups=self.groups_1, bias=False)
			self.weights_1 = self.conv1.weight
			self.relu2 = nn.ReLU()
			self.conv2 = nn.Conv3d(channels_2, filters_2, kernel_shape_2, stride=stride_2, padding=padding_2, groups=self.groups_2, bias=False)
			self.weights_2 = self.conv2.weight

			self.gap1  = nn.AdaptiveAvgPool3d(1)
			self.conv3 = nn.Conv3d(channels_3, filters_3, kernel_shape_3, stride=stride_3, padding=padding_3, groups=self.groups_3, bias=False)
			self.weights_3 = self.conv3.weight
			self.relu3 = nn.ReLU()
			self.conv4 = nn.Conv3d(channels_4, filters_4, kernel_shape_4, stride=stride_4, padding=padding_4, groups=self.groups_4, bias=False)
			self.weights_4 = self.conv4.weight
			self.sigmoid1 = nn.Sigmoid()

			self.sigmoid2 = nn.Sigmoid()
			self.conv5 = nn.Conv3d(channels_5, filters_5, kernel_shape_5, stride=stride_5, padding=padding_5, groups=self.groups_5, bias=False)
			self.weights_5 = self.conv5.weight

		def swish(self, x):
			return x * torch.sigmoid(x)

		def forward(self, x):
			print("input shape: ", x.shape)
			x = self.relu1(x)
			print("relu1 shape: ", x.shape)
			relu1_out = x
			x = self.conv1(x)
			print("conv1 shape: ", x.shape)
			x = self.relu2(x)
			print("relu2 shape: ", x.shape)
			x = self.conv2(x)
			print("conv2 shape: ", x.shape)
			conv2_out = x
			x = self.gap1(x)
			print("gap1 shape: ", x.shape)
			x = self.conv3(x)
			print("conv3 shape: ", x.shape)
			x = self.relu3(x)
			print("relu3 shape: ", x.shape)
			x = self.conv4(x)
			print("conv4 shape: ", x.shape)
			x = self.sigmoid1(x)
			print("sigmoid1 shape: ", x.shape)
			x = x * conv2_out
			print("multiply shape: ", x.shape)
			x = self.swish(x)
			print("swish shape: ", x.shape)
			x = self.conv5(x)
			print("conv5 shape: ", x.shape)
			x = x + relu1_out
			print("add shape: ", x.shape)
			return x
		
		def save_weights(self):
			print("weights_1_cin{}_cout{}:".format(self.coarse_in_1, self.coarse_out_1), self.weights_1.detach().numpy().shape)
			weights_transformed = transform_weights(
								self.weights_1.detach().numpy(),
								self.coarse_in_1,
								self.coarse_out_1,
								1, 1, groups=self.groups_1)
			with open(prefix + '/weights_1_cin{}_cout{}.csv'.format(self.coarse_in_1, self.coarse_out_1), 'w') as f:
				f.write(array_init(weights_transformed[0]))	

			print("weights_2_cin{}_cout{}:".format(self.coarse_in_2, self.coarse_out_2), self.weights_2.detach().numpy().shape)
			weights_transformed = transform_weights(
								self.weights_2.detach().numpy(),
								1,
								self.coarse_out_2,
								1, coarse_group=self.coarse_in_2, groups=self.groups_2)
			with open(prefix + '/weights_2_cin{}_cout{}.csv'.format(self.coarse_in_2, self.coarse_out_2), 'w') as f:
				f.write(array_init(weights_transformed[0]))

			print("weights_3_cin{}_cout{}:".format(self.coarse_in_3, self.coarse_out_3), self.weights_3.detach().numpy().shape)
			weights_transformed = transform_weights(
								self.weights_3.detach().numpy(),
								self.coarse_in_3,
								self.coarse_out_3,
								1, 1, groups=self.groups_3)
			with open(prefix + '/weights_3_cin{}_cout{}.csv'.format(self.coarse_in_3, self.coarse_out_3), 'w') as f:
				f.write(array_init(weights_transformed[0]))	

			print("weights_4_cin{}_cout{}:".format(self.coarse_in_4, self.coarse_out_4), self.weights_4.detach().numpy().shape)
			weights_transformed = transform_weights(
								self.weights_4.detach().numpy(),
								self.coarse_in_4,
								self.coarse_out_4,
								1, 1, groups=self.groups_4)
			with open(prefix + '/weights_4_cin{}_cout{}.csv'.format(self.coarse_in_4, self.coarse_out_4), 'w') as f:
				f.write(array_init(weights_transformed[0]))	

			print("weights_5_cin{}_cout{}:".format(self.coarse_in_5, self.coarse_out_5), self.weights_5.detach().numpy().shape)
			weights_transformed = transform_weights(
								self.weights_5.detach().numpy(),
								self.coarse_in_5,
								self.coarse_out_5,
								1, 1, groups=self.groups_5)
			with open(prefix + '/weights_5_cin{}_cout{}.csv'.format(self.coarse_in_5, self.coarse_out_5), 'w') as f:
				f.write(array_init(weights_transformed[0]))	

	if not os.path.exists(prefix):
		os.makedirs(prefix)
	
	input_shape = (1, 8, 6, 12, 12)
	
	x = torch.randn(input_shape)
	print("*"*30)
	print("input:", x.numpy().shape)
	print("*"*30)
	write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary.tofile(prefix + "/input.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/input.dat", write_input_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	partition_model = X3d_m_layer()
	partition_model.save_weights()

	out = partition_model(x)

	print("output:", out.detach().numpy().shape)
	print("*"*30)
	write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")


def conv_3d(input_shape, kernel_shape, filters, padding, stride, groups, depthwise, coarse_in, coarse_out, file_format, prefix="data"):
	if not os.path.exists(prefix + '/conv_3d'):
		os.makedirs(prefix + '/conv_3d')

	x = torch.randn(input_shape)
	# print(x.numpy())
	print(x.numpy().shape)

	write_input_binary = x.numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_input_binary.tofile(prefix + "/conv_3d/input.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/conv_3d/input.dat", write_input_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	conv = torch.nn.Conv3d(x.shape[1], filters, kernel_shape, stride=stride, padding=padding, groups=groups, bias=False)
	weights = conv.weight
	# print(weights.detach().numpy())
	print(weights.detach().numpy().shape)
	out = conv(x)
	
	write_weights_binary = weights.detach().numpy()#.transpose(1, 0, 2, 3, 4)
	# if file_format == "bin":
	# 	write_weights_binary.tofile(prefix + "/conv_3d/weights.dat")
	# elif file_format == "txt":
	# 	np.savetxt(prefix + "/conv_3d/weights.dat", write_weights_binary.flatten(), fmt='%.8f')
	# else:
	# 	raise Exception("Format not supported")

	if depthwise:
		weights_transformed = transform_weights(
							write_weights_binary,
							1,
							coarse_out,
							1, coarse_group=coarse_in, groups=groups)
	else:
		weights_transformed = transform_weights(
								write_weights_binary,
								coarse_in,
								coarse_out,
								1, 1, groups=groups)
	with open(prefix + '/conv_3d/weights.csv', 'w') as f:
			f.write(array_init(weights_transformed[0]))
	
	print("="*40)
	print("="*40)
	# print(out.detach().numpy())
	print(out.detach().numpy().shape)

	write_out_binary = out.detach().numpy().transpose(0, 3, 4, 2, 1)
	if file_format == "bin":
		write_out_binary.tofile(prefix + "/conv_3d/output.dat")
	elif file_format == "txt":
		np.savetxt(prefix + "/conv_3d/output.dat", write_out_binary.flatten(), fmt='%.8f')
	else:
		raise Exception("Format not supported")

	exit()
	stream = create_queue(x.numpy().transpose(0, 3, 4, 2, 1))

	batch = x.shape[0]
	channel = x.shape[1]
	depth = x.shape[2]
	height = x.shape[3]
	width = x.shape[4]

	kd, kh, kw = kernel_shape
	pad_d, pad_h, pad_w = padding
	stride_d, stride_h, stride_w = stride

	# window_buffer = [kh][kw][kd-1]
	window_buffer = []
	for i in range(kh):
		width_buffers = []
		for j in range(kw):
			depth_buffers = []
			for k in range(kd-1):
				depth_buffers.append(deque([]))
			width_buffers.append(depth_buffers)
		window_buffer.append(width_buffers)

	# line_buffer = [kh][kw-1]
	line_buffer = []
	for i in range(kh):
		width_buffers = []
		for j in range(kw-1):
			width_buffers.append(deque([]))
		line_buffer.append(width_buffers)

	# extra_buffer = [kh-1]
	extra_buffer = []
	for i in range(kh-1):
		extra_buffer.append(deque([]))

	# window_buffer = [kh][kw][kd]
	out_buffer = []
	for i in range(kh):
		width_buffers = []
		for j in range(kw):
			depth_buffers = []
			for k in range(kd):
				depth_buffers.append(deque([]))
			width_buffers.append(depth_buffers)
		out_buffer.append(width_buffers)

	frame_cache = np.zeros(shape=(kh, kw, kd))

	count = 0
	for b in range(0, batch):
		for h in range(0, height+2*pad_h):
			for w in range(0, width+2*pad_w):
				for d in range(0, depth+2*pad_d):
					for c in range(0, channel):
						count += 1
						if h < pad_h:
							tmp=0
						elif h > height+pad_h-1:
							tmp=0
						elif w < pad_w:
							tmp=0
						elif w > width+pad_w-1:
							tmp=0
						elif d < pad_d:
							tmp=0
						elif d > depth+pad_d-1:
							tmp=0
						else:
							tmp = stream.pop()

						print(f"b={b}  |  row={h}  |  col={w}  |  d={d}  |  c={c}  |  => tmp={tmp}")

						if h < kh-1:
							if w < kw-1:
								if d < kd-1:
									window_buffer[h][w][d].appendleft(tmp)
									frame_cache[h][w][d] = tmp
									if w > 0:
										line_buffer[h][w-1].appendleft(tmp)
								else:
									line_buffer[h][w].appendleft(tmp)
							elif (w == kw-1 or w == kw) and d < kd-1:
								if w == kw-1:
									window_buffer[h][w][d].appendleft(tmp)
									line_buffer[h][w-1].appendleft(tmp)
								elif w == kw:
									extra_buffer[h].appendleft(tmp)
							elif w == kw-1 and d >= kd-1:
								extra_buffer[h].appendleft(tmp)
							else:
								extra_buffer[h].appendleft(tmp)
						elif h == kh-1 and w < kw-1:
							if d < kd-1:
								window_buffer[h][w][d].appendleft(tmp)
								frame_cache[h][w][d] = tmp
								if w > 0:
									line_buffer[h][w-1].appendleft(tmp)
							else:
								line_buffer[h][w].appendleft(tmp)
						elif h == kh-1 and w == kw-1 and d < kd-1:
							window_buffer[h][w][d].appendleft(tmp)
							line_buffer[h][w-1].appendleft(tmp)
						else:

							for i in range(0, kh):
								for j in range(0, kw):
									for k in range(0, kd-1):
										curr = window_buffer[i][j][k].pop()
										frame_cache[i][j][k] = curr

							for i in range(0, kh):
								for j in range(0, kw-1):
									curr = line_buffer[i][j].pop()
									frame_cache[i][j][-1] = curr

							for i in range(0, kh-1):
								curr = extra_buffer[i].pop()
								frame_cache[i][-1][-1] = curr

							frame_cache[-1][-1][-1] = tmp
							
							print(f"WINDOW_BUFFER\t(POP):\t\tWindow buffer depth: {len(window_buffer[0][0][0])}")
							if not (h == height+2*pad_h-1 and w == width+2*pad_w-1 and d == depth+2*pad_d-1):
								for i in range(0, kh):
									for j in range(0, kw):
										for k in range(0, kd-1):
											curr = frame_cache[i][j][k+1]
											window_buffer[i][j][k].appendleft(curr)
								print(f"WINDOW_BUFFER\t(APPEND):\tWindow buffer depth: {len(window_buffer[0][0][0])}")
							else:
								print("WINDOW_BUFFER:\t\t\tSkip appending...")

							print(f"LINE_BUFFER\t(POP):\t\tLine buffer depth: {len(line_buffer[0][0])}")
							if not (h == height+2*pad_h-1 and w == width+2*pad_w-1):
								for i in range(0, kh):
									for j in range(0, kw-1):
										curr = frame_cache[i][j+1][kd-1]
										line_buffer[i][j].appendleft(curr)
								print(f"LINE_BUFFER\t(APPEND):\tLine buffer depth: {len(line_buffer[0][0])}")
							else:
								print("LINE_BUFFER:\t\t\tSkip appending...")

							print(f"EXTRA_BUFFER\t(POP):\t\tExtra buffer depth: {len(extra_buffer[0])}")
							if not ((h == height+2*pad_h-1 and w == kw-1 and d >= kd-1) or (h == height+2*pad_h-1 and w > kw-1)):
								for i in range(0, kh-1):
									curr = frame_cache[i+1][0][0]
									extra_buffer[i].appendleft(curr)
								print(f"EXTRA_BUFFER\t(APPEND):\tExtra buffer depth: {len(extra_buffer[0])}")
							else:
								print("EXTRA_BUFFER:\t\t\tSkip appending...")

							for i in range(kh):
								for j in range(kw):
									for k in range(kd):
										curr = frame_cache[i][j][k]
										out_buffer[i][j][k].appendleft(curr)
							print(count)
							exit()
							# print(frame_cache.transpose((2, 0, 1)))

						print(f"Window buffer: {len(window_buffer[0][0][0])}, Line buffer: {len(line_buffer[0][0])}, Extra buffer: {len(extra_buffer[0])}")
						# show_wb(window_buffer)
						# show_lb(line_buffer, extra_buffer)
						print("="*40)

	for i in range(kh):
		for j in range(kw):
			for k in range(kd-1):
				assert len(window_buffer[i][j][k]) == 0, "There are still %d values in window_buffer [%d][%d][%d]" % (len(window_buffer[i][j][k]), i, j, k)

	for i in range(kh):
		for j in range(kw-1):
			assert len(line_buffer[i][j]) == 0, "There are still %d values in line_buffer [%d][%d]" % (len(line_buffer[i][j]), i, j)

	for i in range(kh-1):
		assert len(extra_buffer[i]) == 0, "There are still %d values in line_buffer [%d]" % (len(extra_buffer[i]), i)

	showres(out_buffer, input_shape, file_format, kh, kw, kd, pad_h, pad_w, pad_d, stride_h, stride_w, stride_d)

def parse_args():
  parser = argparse.ArgumentParser(description='3D sliding window prototying script')
  parser.add_argument('op_type', type=str)
  parser.add_argument('prefix', type=str)
  parser.add_argument('--input_shape', nargs='+', default=[1, 24, 16, 64, 64], type=int)
  parser.add_argument('--input_shape_2', nargs='+', default=[1, 24, 16, 64, 64], type=int)
  parser.add_argument('--kernel_shape', nargs='+', default=[3, 3, 3], type=int)
  parser.add_argument('--filters', default=1, type=int)
  parser.add_argument('--groups', default=1, type=int)
  parser.add_argument('--padding', nargs='+', default=[0, 0, 0], type=int)
  parser.add_argument('--stride', nargs='+', default=[1, 1, 1], type=int)
  parser.add_argument('--depthwise', default=False, action='store_true')
  parser.add_argument('--coarse_in', default=1, type=int)
  parser.add_argument('--coarse_out', default=1, type=int)
  parser.add_argument('--elemwise_op_type', choices=['add', 'mul'], default='add', type=str)
  parser.add_argument('--format', choices=['txt', 'bin'], default='bin', type=str)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
	args = parse_args()

	op_type = args.op_type

	if op_type == '3d_conv':
		conv_3d(args.input_shape, args.kernel_shape, args.filters, args.padding, args.stride, args.groups, args.depthwise, args.coarse_in, args.coarse_out, args.format)
	elif op_type == '3d_relu':
		relu_3d(args.input_shape, args.coarse_in, args.format)
	elif op_type == '3d_swish':
		shish_3d(args.input_shape, args.coarse_in, args.format)
	elif op_type == '3d_sigmoid':
		sigmoid_3d(args.input_shape, args.coarse_in, args.format)
	elif op_type == '3d_elemwise':
		elemwise_3d(args.input_shape, args.input_shape_2, args.coarse_in, args.elemwise_op_type, args.format)
	elif op_type == '3d_elemwise_bc_relu':
		elemwise_broadcast_3d(args.input_shape, args.input_shape_2, args.coarse_in, args.elemwise_op_type, args.format)
	elif op_type == '3d_gap':
		gap_3d(args.input_shape, args.coarse_in, args.coarse_out, args.format)
	elif op_type == '3d_part':
		part_3d(args.format, "data/"+args.prefix)
	else:
		print("Invalid op_type: %s" % op_type)