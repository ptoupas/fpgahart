import numpy as np

def get_produce_points(matrix, node, debug=False):
  points = []
  for i in range(0, matrix.shape[0]):
    if i == node:
      non_zero_points = list(np.where(matrix[i, :] != 0)[0])
      for p in non_zero_points:
        if matrix[i, p] > 0:
          points.append(p)

  if len(points) == 1:
    return points[0]
  else:
    if debug:
      print("This node does not produce any values at any rate. Aborting the balancing")
    return None

def get_consume_points(matrix, node):
    points = []
    for i in range(0, matrix.shape[1]):
        if i == node:
            non_zero_points = list(np.where(matrix[:, i] != 0)[0])
            for p in non_zero_points:
                if matrix[p, i] < 0:
                    points.append(p)

    return points

def get_rate_ratio(matrix):
    rate_ratio = []
    for i in range(matrix.shape[1] -1):
        consume_points = get_consume_points(matrix, i)
        produce_points = get_produce_points(matrix, i)
        if len(consume_points) > 0:
            min_rate = 100000000
            for c in consume_points:
                min_rate = min(abs(matrix[c, i]), min_rate)
            rate_ratio.append(matrix[produce_points, i] / min_rate)
        else:
            rate_ratio.append(-1)
    rate_ratio.append(-1)
    return rate_ratio

def balance_multiport_rates(matrix):
    for i in range(matrix.shape[1] -1):
        consume_points = get_consume_points(matrix, i)
        if len(consume_points) > 1:
            min_rate = 100000000
            for c in consume_points:
                min_rate = min(abs(matrix[c, i]), min_rate)
            for c in consume_points:
                matrix[c, i] = -min_rate
    return matrix

def connected_with_node(matrix, layer, prev_layer):
  produce_edge_point = None
  for i in range(matrix.shape[1]):
    if i == prev_layer:
      for j in range(matrix.shape[0]):
        if matrix[j, i] > 0:
          produce_edge_point = j
          break
  if matrix[produce_edge_point, layer] >= 0:
    return False
  else:
    return True

def balance_matrix(matrix, debug=False):
  rate_ratio = get_rate_ratio(matrix)

  for i in range(0, matrix.shape[1]):
    layer = matrix.shape[1] - 1 - i
    count_edges = np.count_nonzero(matrix[:,layer])

    if count_edges > 1:
      consume_points = get_consume_points(matrix, layer)
      produce_points = get_produce_points(matrix, layer)
      if debug:
        print(f"Layer {layer} has {count_edges} edges. Consume points at {consume_points} and produce points at {produce_points}")

      for cp in consume_points:
        node_produce_point = get_produce_points(matrix, cp)
        if debug:
          print(f"Checking layer {layer} with previous layer {cp} at point {node_produce_point}, matrix values = [{matrix[cp,layer]}, {matrix[cp,node_produce_point]}]")

        if abs(matrix[cp,layer]) > matrix[cp,node_produce_point]:
          # propogate forward
          prev_layer = layer
          for j in range(layer,matrix.shape[0]):
            if not j == prev_layer and not connected_with_node(matrix, j, prev_layer):
              continue
            node_consume_point_internal = get_consume_points(matrix, j)
            if debug:
              print(f"Propogating forward from {layer} to {j} at point {node_consume_point_internal}")
            min_rate = 200
            for cpi in node_consume_point_internal:
              node_produce_point_internal = get_produce_points(matrix, cpi)
              if debug:
                print(f"At point {cpi} we have reduce point {node_produce_point_internal}")
                print(f"Matrix values = [{matrix[cpi,j]}, {matrix[cpi,node_produce_point_internal]}]")
              if(abs(matrix[cpi,j]) <= matrix[cpi,node_produce_point_internal]):
                prev_layer = j
                continue
              min_rate = min(abs(matrix[cpi,node_produce_point_internal]), min_rate)
            for cpi in node_consume_point_internal:
              node_produce_point_internal = get_produce_points(matrix, cpi)
              if not matrix[cpi,node_produce_point_internal] == min_rate:
                multiport_produce_point = get_consume_points(matrix, node_produce_point_internal)
                assert len(multiport_produce_point) <= 1, "This graph cannot be balanced"
                if len(multiport_produce_point) == 1:
                  multiport_produce_point = multiport_produce_point[0]
                  matrix[multiport_produce_point,node_produce_point_internal] = -min_rate/rate_ratio[node_produce_point_internal]
              matrix[cpi,cpi] = min_rate
              matrix[cpi,j] = -min_rate
            
            if debug:
              print(f"Matrix values = [{matrix[j,j]}], rr: {rate_ratio[j]}")
            matrix[j,j] = min_rate*rate_ratio[j]
            prev_layer = j

        elif abs(matrix[cp,layer]) < matrix[cp,node_produce_point]:
          # propogate backward
          for j in range(0,layer):
            node_consume_point_internal = get_consume_points(matrix, layer-j)
            if debug:
              print(f"Propogating backward from {layer} to {layer-j} at point {node_consume_point_internal}")
            for cpi in node_consume_point_internal:
              node_produce_point_internal = get_produce_points(matrix, cpi)
              if debug:
                print(f"At point {cpi} we have reduce point {node_produce_point_internal}")
              if(abs(matrix[cpi,layer-j]) >= matrix[cpi,node_produce_point_internal]):
                continue
              matrix[cpi,node_produce_point_internal] = abs(matrix[cpi,layer-j])
              if cpi > 0:
                for pp in get_consume_points(matrix, node_produce_point_internal):
                  matrix[pp,node_produce_point_internal] = -matrix[cpi,node_produce_point_internal]/rate_ratio[cpi]

    if debug:
      print("="*40)
  return matrix