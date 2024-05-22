import io
import json
import base64
import numpy as np


class MatrixWrapper:
    def __init__(self, matrix):
        if isinstance(matrix, np.ndarray):
            self.matrix = matrix.tolist()
        else:
            self.matrix = matrix

    def to_json(self):
        return json.dumps(self.matrix)  # Convert list of lists into JSON string


def compress_and_encode_matrix(matrix):
    buf = io.BytesIO()
    np.savez_compressed(buf, matrix=matrix)
    buf.seek(0)
    compressed_data = buf.read()
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    return encoded_data


def load_and_decompress_matrix(encoded_data):
    compressed_data = base64.b64decode(encoded_data)
    data_buffer = io.BytesIO(compressed_data)
    with np.load(data_buffer, allow_pickle=True) as data:
        matrix = data['matrix']
    return matrix