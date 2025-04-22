package com.reversi.client;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.reversi.common.Player;
import com.reversi.common.ReversiGame;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Controller implements IController {
  private static final Logger logger =
      LoggerFactory.getLogger(Controller.class);

  private final ReversiGame game;
  private final GameView view;
  private final Player us, bot;
  private final OrtEnvironment env;
  private final OrtSession session;

  public Controller(GameView view, Player us, String modelPath) {
    this.view = view;
    this.us = us;
    this.bot = us.opponent();
    this.game = new ReversiGame();
    this.view.setUs(us);

    // initialize ONNX Runtime once
    try {
      env = OrtEnvironment.getEnvironment();
      SessionOptions opts = new SessionOptions();
      session = env.createSession(modelPath, opts);
    } catch (OrtException e) {
      throw new RuntimeException("Failed to load ONNX model", e);
    }

    // draw the initial position
    this.view.updateGame(game);
  }

  @Override
  public void sendMove(int row, int col) {
    // 1) Human’s turn
    if (game.getCurrentPlayer() != us ||
        !game.getBoard().isValidMove(row, col, us)) {
      logger.warn("Invalid move ({},{})", row, col);
      return;
    }

    game.makeMove(row, col);
    view.updateGame(game);
    if (game.isGameOver()) {
      view.showGameOver(game.getWinner() == us ? "you win" : "you lose");
      return;
    }

    // 2) Bot’s turn
    playBotMove();
  }

  private void playBotMove() {
    try {
      // encode 1×2×8×8 input
      float[] inputData = OnnxIO.Encode(game.getBoard(), bot);
      OnnxTensor inputTensor = OnnxTensor.createTensor(
          env, FloatBuffer.wrap(inputData), new long[] {1, 2, 8, 8});

      Map<String, OnnxTensor> inputs =
          Collections.singletonMap("input", inputTensor);
      try (OrtSession.Result results = session.run(inputs)) {
        float[][] policy = OnnxIO.GetPolicy(results);
        float argMax = -Float.MAX_VALUE;
        int r = -1, c = -1;

        var board = game.getBoard();
        for (int i = 0; i < 8; ++i)
          for (int j = 0; j < 8; ++j) {
            if (board.isValidMove(i, j, bot) && policy[i][j] > argMax) {
              argMax = policy[i][j];
              r = i;
              c = j;
            }
          }

        if (board.isValidMove(r, c, bot)) {
          game.makeMove(r, c);
          view.updateGame(game);
          if (game.isGameOver()) {
            view.showGameOver(game.getWinner() == us ? "you win" : "you lose");
          }
        } else {
          logger.error("bot failed to make a move. ({},{})", r, c);
        }
      }
    } catch (OrtException e) {
      e.printStackTrace();
    }
  }
}
