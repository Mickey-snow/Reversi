package com.reversi.experimental;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.reversi.common.Board;
import com.reversi.common.Player;

public class OnnxIO {

  public static float[] Encode(Board board, Player player) {
    var data = new float[1 * 2 * 8 * 8];
    int idx = 0;
    final int offset = 8 * 8;

    for (int row = 0; row < 8; ++row)
      for (int col = 0; col < 8; ++col) {
        if (board.get(row, col) == player) {
          data[idx] = 1;
          data[idx + offset] = 0;
        } else {
          data[idx] = 0;
          data[idx + offset] = 1;
        }
        ++idx;
      }

    return data;
  }

  public static float[][] GetPolicy(OrtSession.Result results)
      throws OrtException {
    float[][] ret = new float[8][8];
    float[][] policyOut = (float[][])results.get(0).getValue();
    int idx = 0;

    for (int row = 0; row < 8; ++row)
      for (int col = 0; col < 8; ++col) {
        ret[row][col] = policyOut[0][idx++];
      }

    return ret;
  }

  public static float[][] GetValue(OrtSession.Result results)
      throws OrtException {
    float[][] ret = new float[8][8];
    float[][] valueOut = (float[][])results.get(1).getValue();
    int idx = 0;

    for (int row = 0; row < 8; ++row)
      for (int col = 0; col < 8; ++col) {
        ret[row][col] = valueOut[0][idx++];
      }

    return ret;
  }
}
