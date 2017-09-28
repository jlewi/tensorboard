# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2

import unittest

class TensorsTest(unittest.TestCase):

  def testInsertTagId(self):
    tensor = tensor_pb2.TensorProto()
    tensor.dtype = types_pb2.DT_INT64
    tensor.version_number = 15
    tensor.int64_val.extend([1, 2, 3])

    stored = tensor_pb2.TensorProto()

    tensor_data = tensor.SerializeToString()
    # TODO(jlewi): On Python 2.7 we need to encode as utf-8 to convert
    # from unicode to str. Otherwise ParseFromString returns an empty proto.
    # This is only a problem inside the Travis 2.7 environment.
    uni_data = unicode(tensor_data)
    stored.ParseFromString(uni_data)
    self.assertEqual(tensor, stored,
                     'Got {0} want {1}'.format(stored, tensor))


if __name__ == '__main__':
  unittest.main()
