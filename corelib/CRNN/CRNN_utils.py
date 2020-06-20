import numpy as np
import json

from logger.logging import RekogntionLogger

logger = RekogntionLogger(name="tf_io_pipeline_fast_tools")


class _FeatureIO(object):
    """
    Feature IO Base Class
    """
    def __init__(self, char_dict_path, ord_map_dict_path):
        """

        :param char_dict_path:
        :param ord_map_dict_path:
        """

        logger.info(msg="_FeatureIO called")
        self._char_dict = CharDictBuilder.read_char_dict(char_dict_path)
        self._ord_map = CharDictBuilder.read_ord_map_dict(ord_map_dict_path)
        return

    def int_to_char(self, number):
        """
        convert the int index into char
        :param number: Can be passed as string representing the integer value to look up.
        :return: Character corresponding to 'number' in the char_dict
        """

        logger.info(msg="int_to_char called")
        # 1 is the default value in sparse_tensor_to_str() This will be skipped when building the resulting strings
        if number == 1 or number == '1':
            return '\x00'
        else:
            return self._char_dict[str(number) + '_ord']

    def sparse_tensor_to_str_for_tf_serving(self, decode_indices, decode_values, decode_dense_shape):
        """

        :param decode_indices:
        :param decode_values:
        :param decode_dense_shape:
        :return:
        """

        logger.info(msg="sparse_tensor_to_str_for_tf_serving")
        indices = decode_indices
        values = decode_values
        values = np.array([self._ord_map[str(tmp) + '_index'] for tmp in values])
        dense_shape = decode_dense_shape

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]
        for number_list in number_lists:
            str_lists.append([self.int_to_char(val) for val in number_list])
        for str_list in str_lists:
            res.append(''.join(c for c in str_list if c != '\x00'))
        return res


class CharDictBuilder(object):
    """
        Build and read char dict
    """
    def __init__(self):
        logger.info(msg="CharDictBuilder")
        pass

    @staticmethod
    def read_char_dict(dict_path):
        """

        :param dict_path:
        :return: a dict with ord(char) as key and char as value
        """

        logger.info(msg="read_char_dict")
        with open(dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path):
        """

        :param ord_map_dict_path:
        :return:
        """

        logger.info(msg="read_ord_map_dict")
        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res
