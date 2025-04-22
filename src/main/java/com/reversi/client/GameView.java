package com.reversi.client;

import com.reversi.common.Player;
import com.reversi.common.ReversiGame;
import java.util.Timer;
import java.util.TimerTask;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;

public class GameView {

  private BorderPane gamePane;
  private Button[][] boardButtons = new Button[8][8];
  private Label statusLabel;
  // additional fields to preserve last status content if needed
  private String baseStatus = "";

  private IController controller;
  private Player us;
  private long blackTime, whiteTime;

  private Timer timer;

  public GameView() {
    createComponents();
    layoutComponents();
    this.controller = null;
    this.blackTime = this.whiteTime = 0;

    this.timer = new Timer();
    timer.scheduleAtFixedRate(new TimerTask() {
      @Override
      public void run() {
        blackTime -= 100;
        whiteTime -= 100;
        if (blackTime < 0)
          blackTime = 0;
        if (whiteTime < 0)
          whiteTime = 0;

        Platform.runLater(() -> {
          final String dispblackTime = formatTime(blackTime);
          final String dispwhiteTime = formatTime(whiteTime);

          // Append time information to the base status.
          statusLabel.setText(baseStatus + "\nTime - Black: " + dispblackTime +
                              " | White: " + dispwhiteTime);
        });
      }

      // Helper method to format milliseconds into a mm:ss string.
      private String formatTime(long ms) {
        long totalSeconds = ms / 1000;
        long minutes = totalSeconds / 60;
        long seconds = totalSeconds % 60;
        return String.format("%02d:%02d", minutes, seconds);
      }
    }, 0, 100);
  }

  public void setController(IController controller) {
    this.controller = controller;
  }

  public void setUs(Player us) { this.us = us; }

  private void createComponents() {
    gamePane = new BorderPane();
    gamePane.getStyleClass().add("game-pane");

    statusLabel = new Label("Game has not started.");
    statusLabel.getStyleClass().add("game-status");
    baseStatus = statusLabel.getText();

    // Initialize the board buttons.
    GridPane boardGrid = new GridPane();
    boardGrid.setAlignment(Pos.CENTER);
    boardGrid.getStyleClass().add("board-grid");

    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        Button btn = new Button();
        btn.getStyleClass().add("board-button");
        btn.setPrefSize(50, 50);
        final int row = i, col = j;
        btn.setOnAction(e -> { controller.sendMove(row, col); });
        boardButtons[i][j] = btn;
        boardGrid.add(btn, j, i); // Note: column index then row index
      }
    }
    gamePane.setCenter(boardGrid);
    gamePane.setTop(statusLabel);
    BorderPane.setAlignment(statusLabel, Pos.CENTER);
  }

  private void layoutComponents() {}

  public BorderPane getGamePane() { return gamePane; }

  /**
   * Updates the board buttons and status label based on the current game state.
   * This implementation assumes that the Board object inside ReversiGame
   * provides a method such as getDiscAt(row, col) which returns a Player enum
   * (or null for an empty cell).
   */
  public void updateGame(ReversiGame game) {
    // Iterate over each board cell.
    Platform.runLater(() -> {
      for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
          Player disc = game.getBoard().get(i, j);
          Button btn = boardButtons[i][j];
          if (disc == Player.Black) {
            btn.setText("B");
            btn.getStyleClass().removeAll("white-disc", "possible-disc");
            if (!btn.getStyleClass().contains("black-disc"))
              btn.getStyleClass().add("black-disc");

          } else if (disc == Player.White) {
            btn.setText("W");
            btn.getStyleClass().removeAll("black-disc", "possible-disc");
            if (!btn.getStyleClass().contains("white-disc"))
              btn.getStyleClass().add("white-disc");

          } else if (game.getCurrentPlayer() == us &&
                     game.getBoard().isValidMove(i, j, us)) {
            btn.setText(".");
            btn.getStyleClass().removeAll("black-disc", "white-disc");
            if (!btn.getStyleClass().contains("possible-disc"))
              btn.getStyleClass().add("possible-disc");

          } else {
            btn.setText("");
            btn.getStyleClass().removeAll("black-disc", "white-disc",
                                          "possible-disc");
          }
        }
      }

      // Update the base status with current player's turn.
      String currentTurn =
          game.getCurrentPlayer() == us ? "Your's turn" : "Opponent's turn";
      baseStatus = currentTurn;
      statusLabel.setText(baseStatus);
    });
  }

  /**
   * Updates the time display on the UI.
   * Converts black and white times (milliseconds) to a mm:ss format and appends
   * it to the status label.
   */
  public void updateTime(long blackTimeMs, long whiteTimeMs) {
    this.blackTime = blackTimeMs;
    this.whiteTime = whiteTimeMs;
  }

  /**
   * Displays the game-over message on the status label and disables the board
   * buttons.
   */
  public void showGameOver(String displayString) {
    Platform.runLater(() -> {
      statusLabel.setText("Game Over: " + displayString);
      // Disable all board buttons to prevent further moves.
      for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
          boardButtons[i][j].setDisable(true);
        }
      }
    });
  }
}
