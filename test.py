import functools
import operator
import struct
import array
import numpy as np
import glob
import sys
from fann2 import libfann

class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def parse_idx(fd):
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        files = []
        for file in glob.glob("net/*.net"):
            files.append(file)
        files.sort(reverse=True)
        filename = files[0]
    else:
        filename = sys.argv[1]

    print('Opening net: ' + filename)
    ann = libfann.neural_net()
    ann.create_from_file(filename)

    print('Reading labels')
    with open('source/t10k-labels.idx1-ubyte', 'rb') as fd:
        labels = parse_idx(fd)
    print('Reading images')
    with open('source/t10k-images.idx3-ubyte', 'rb') as fd:
        images = parse_idx(fd)

    print('Testing')
    error_rate = 0
    images = iter(images)
    for label in labels:
        image = next(images)
        input = []
        for row in image:
            for value in row:
                input.append((float(value) - 127.0) / 127.0)
        output = int(label)
        result_raw = ann.run(input)
        result = int(result_raw.index(max(result_raw)))
        if output == result:
            pass
        else:
            # print(output)
            # print(result)
            # print(result_raw)
            error_rate = error_rate + 1
    print('Error rate = ' + ('%.2f' % (float(error_rate)/float(len(labels))*100.0)) + '%')
