package com.reversi.experimental;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.reversi.common.Board;
import com.reversi.common.Player;
import java.nio.FloatBuffer;
import java.util.*;

public class AlphaZeroONNX {
  public static void main(String[] args) throws OrtException {
    // 2.1 Create environment & session
    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
         SessionOptions opts = new SessionOptions();
         OrtSession session = env.createSession(
             "/home/kqx/Documents/Reversi/src/main/resources/alphazero.onnx",
             opts)) {

      // 2.2 Prepare board-state tensor (1×2×8×8)
      Board board = Board.createDefault();
      float[] inputData = OnnxIO.Encode(board, Player.Black);

      OnnxTensor inputTensor = OnnxTensor.createTensor(
          env, FloatBuffer.wrap(inputData), new long[] {1, 2, 8, 8});

      // 2.3 Run the model
      Map<String, OnnxTensor> inputs =
          Collections.singletonMap("input", inputTensor);
      try (OrtSession.Result results = session.run(inputs)) {
        // outputs in order of output_names: ["policy", "value"]
        // float[][] policyOut = (float[][])results.get(0).getValue();
        // float[][] valueOut = (float[][])results.get(1).getValue();

        // policyOut[0] is a length‑64 vector of log‑probs (or probs)
        // valueOut[0][0] is the scalar value in [–1,1]
        System.out.println("Policy logits:");
        var out = OnnxIO.GetPolicy(results);
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j)
            System.out.print(out[i][j] + "  ");
          System.out.println();
        }
      }
    }
  }
}
