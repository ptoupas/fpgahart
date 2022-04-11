class MemoryNode:
    def __init__(self, mem_type, shape):
        if mem_type == "in":
            self.output_shape = shape
        elif mem_type == "out":
            self.input_shape = shape
        else:
            raise ValueError("Memory type must be 'in' or 'out'")
