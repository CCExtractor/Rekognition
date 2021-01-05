import numpy as np
import json

from logger.logging import RekogntionLogger


logger = RekogntionLogger(name="CRNN_utils")


class _FeatureIO:
    """
        Feature IO Class
    """
    def __init__(self, char_dict_path, ord_map_dict_path):
        """     Initializes Feature IO Class
        Args:
                *   char_dict_path: path to character dictionary
                *   ord_map_dict_path: path to Ord Map dictionary
        Workflow:
                *   Initialize the class with value of the two paths
        """

        logger.info(msg="_FeatureIO called")
        self._char_dict = CharDictBuilder.read_char_dict(char_dict_path)
        self._ord_map = CharDictBuilder.read_ord_map_dict(ord_map_dict_path)

    def int_to_char(self, number):
        """     Convert the int index into char
        Args:
                *   number: string representing the integer value to look up
        Workflow:
                *   if the number is 1, '\x00' is returned as we need to
                    skip 1 because its the default value in
                    parse_tensor_to_str_for_tf_serving
                *   else the corresponding character is returned
        Returns:
                *   Character corresponding to 'number' in the char_dict
        """

        logger.info(msg="int_to_char called")
        if number == 1 or number == '1':
            return '\x00'
        else:
            return self._char_dict[str(number) + '_ord']

    def sparse_tensor_to_str_for_tf_serving(self, decode_indices, decode_values, decode_dense_shape):

        logger.info(msg="sparse_tensor_to_str_for_tf_serving called")
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


class CharDictBuilder:
    """
        To read char dict
    """

    def __init__(self):
        """
        Workflow:
            *   Initializes CharDictBuilder class
        """

        logger.info(msg="CharDictBuilder called")
        pass

    @staticmethod
    def read_char_dict(dict_path):
        """     Reads Character Dictionary
        Args:
                *   dict_path: path to pre stored character dictionary
        Workflow:
                *   Reads the dictionary stored in the path
        Returns:
                *   a dict with ord(char) as key and char as value
        """

        logger.info(msg="read_char_dict called")
        with open(dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path):
        """     Reads Ord Map Dictionary
        Args:
                *   ord_map_dict_path: path to pre stored Ord Map dictionary
        Workflow:
                *   Reads the dictionary stored in the path
        Returns:
                *   the dictionary as read from the path
        """

        logger.info(msg="read_ord_map_dict called")
        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res
