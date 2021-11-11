import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style("whitegrid")

# gap_NHWDC = [188416, 188416, 77824, 77824, 77824, 34816, 34816, 34816, 34816, 34816, 34816, 16384, 16384, 16384, 16384, 16384]
# plt.plot(gap_NHWDC, label='NHWDC Mem Bounded IN', linewidth=3, marker='o')

# gap_NHWDC = [65537, 65537, 16385, 16385, 16385, 4097, 4097, 4097, 4097, 4097, 4097, 1025, 1025, 1025, 1025, 1025]
# plt.plot(gap_NHWDC, label='NHWDC No Mem Bounded', linewidth=3, marker='s')

# gap_NCHWD = [8053067776, 8053067776, 1006633984, 1006633984, 1006633984, 125829376, 125829376, 125829376, 125829376, 125829376, 125829376, 15728704, 15728704, 15728704, 15728704, 15728704]
# plt.plot(gap_NCHWD, label='NCHWD Mem Bounded IN', linewidth=3, marker='v')

# gap_NCHWD = [225280, 225280, 111616, 111616, 111616, 55552, 55552, 55552, 55552, 55552, 55552, 27712, 27712, 27712, 27712, 27712]
# plt.plot(gap_NCHWD, label='NCHWD No Mem Bounded', linewidth=3, marker='d')

# gap_NCHWD_DW = [8053063744, 8053063744, 1006632992, 1006632992, 1006632992, 125829136, 125829136, 125829136, 125829136, 125829136, 125829136, 15728648, 15728648, 15728648, 15728648, 15728648]
# plt.plot(gap_NCHWD_DW, label='NCHWD-DW Mem Bounded IN', linewidth=3, marker='x')

# gap_NCHWD_DW = [3520, 3520, 3488, 3488, 3488, 3472, 3472, 3472, 3472, 3472, 3472, 3464, 3464, 3464, 3464, 3464]
# plt.plot(gap_NCHWD_DW, label='NCHWD-DW No Mem Bounded', linewidth=3, marker='h')

# plt.title("Data ordering impact on Global Average Pooling Layers")
# plt.legend()
# plt.yscale('log')
# plt.xlabel('Global Average Pooling Layer')
# plt.ylabel('Latency (cycles)')
# plt.show()

def serialize(array):
    result = []
    dim1, dim2, dim3, dim4 = array.shape
    for d1 in range(dim1):
        for d2 in range(dim2):
            for d3 in range(dim3):
                for d4 in range(dim4):
                    result.append(array[d1, d2, d3, d4])
    return result

in_fmap_origin = np.random.rand(54, 16, 64, 64)
in_fmap = np.transpose(in_fmap_origin, (0, 2, 3, 1))
channels, height, width, depth = in_fmap.shape
out_fmap = np.zeros((channels, 1, 1, 1))

print("Depth*Height*Width = {}".format(depth*height*width))
print("Depth*Height*Width*Channels = {}".format(depth*height*width*channels))
for c in range(channels):
    avg = 0
    for h in range(height):
        for w in range(width):
            for d in range(depth):
                avg += in_fmap[c, h, w, d]
    out_fmap[c, 0, 0, 0] = avg/(depth*height*width)

streaming_data_in = serialize(in_fmap)
streaming_data_out = []
par_factor = 1
count = 0
count_ops = 0
cycles = 0
for c in range(channels):
    avg = 0
    for h in range(height):
        # for d in range((depth*width)//par_factor):
        #     for parallel_d in range(par_factor):
        #         avg += streaming_data_in[count]
        #         count += 1
        #     cycles += 1
        for w in range(width):
            for d in range(depth//par_factor):
                for parallel_d in range(par_factor):
                    avg += streaming_data_in[count]
                    count += 1
                cycles += 1
    print("{} result(s) at cycle {}".format(1, cycles))
    res = avg/(depth*height*width)
    count_ops += 1
    streaming_data_out.append(res)
print("Final cycles {}. Total ops {}".format(cycles, count_ops))
print(list(out_fmap.flatten()) == streaming_data_out)


in_fmap_reordered = np.transpose(in_fmap_origin, (2, 3, 1, 0))
out_fmap_reordered = np.zeros((channels, 1, 1, 1))

height, width, depth, channels = in_fmap_reordered.shape
print("Depth*Height*Width = {}".format(depth*height*width))
print("Depth*Height*Width*Channels = {}".format(depth*height*width*channels))
avg = np.zeros((channels))
for h in range(height):
    for w in range(width):
        for d in range(depth):
            for c in range(channels):
                avg[c] += in_fmap_reordered[h, w, d, c]
                if d == depth-1 and h == height-1 and w == width-1:
                    out_fmap_reordered[c, 0, 0, 0] = avg[c]/(depth*height*width)

print(list(out_fmap_reordered.flatten()) == streaming_data_out)



streaming_data_in_reordered = serialize(in_fmap_reordered)
streaming_data_out_reordered = []
par_factor = channels
count = 0
count_ops = 0
cycles = 0
avg = np.zeros((channels))
for h in range(height):
    for w in range(width):
        for d in range(depth):
            for c in range(channels//par_factor):
                for parallel_c in range(par_factor):
                    avg[c*par_factor+parallel_c] += streaming_data_in_reordered[count]
                    count += 1
                    if d == depth-1 and h == height-1 and w == width-1:
                        streaming_data_out_reordered.append(avg[c*par_factor+parallel_c]/(depth*height*width))
                        count_ops += 1
                        print("{} result(s) at cycle {}".format(1, cycles))
                cycles += 1
print("Final cycles {}. Total ops {}".format(cycles, count_ops))
print(list(out_fmap_reordered.flatten()) == streaming_data_out_reordered)

print(streaming_data_out_reordered == streaming_data_out)
