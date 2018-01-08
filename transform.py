import functools
import operator
import struct
import array
import numpy as np
import sys
import datetime


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
    print('Reading labels')
    with open('source/train-labels.idx1-ubyte', 'rb') as fd:
        labels = parse_idx(fd)
    print('Reading images')
    with open('source/train-images.idx3-ubyte', 'rb') as fd:
        images = parse_idx(fd)

    print('Parsing')
    k = 0
    images = iter(images)
    print(len(labels))
    filename = 'data/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.data'
    trainfd = open(filename, 'w')
    trainfd.write(str(len(labels)) + ' ' + str(28*28) + ' ' + str(10) + '\n')
    for label in labels:
        image = next(images)
        sys.stdout.write("=================================================\n")
        sys.stdout.write('%4d' % label)
        sys.stdout.write("\n")
        sys.stdout.write("=================================================\n")
        sys.stdout.write("\n")
        input = ''
        for row in image:
            for value in row:
                input = input + str('%.4f' % ((float(value) - 127.0) / 127.0)) + ' '
                sys.stdout.write('%4d' % (int(value)))
                sys.stdout.write(' ')
            sys.stdout.write("\n")
        trainfd.write(input.strip() + '\n')
        output = ''
        for value in range(10):
            if value == int(label):
                output = output + '0.99 '
            else:
                output = output + '0.01 '
        trainfd.write(output.strip() + '\n')
        sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        k = k + 1
