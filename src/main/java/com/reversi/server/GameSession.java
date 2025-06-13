package com.reversi.server;

import com.reversi.common.EventBus;
import com.reversi.common.EventListener;
import com.reversi.common.FischerClock;
import com.reversi.common.Message;
import com.reversi.common.Player;
import com.reversi.common.ReversiGame;
import com.reversi.common.Ticker;
import com.reversi.common.GameRecord;
import com.reversi.common.Board;
import com.reversi.server.HistoryStore;

public class GameSession {
  private ReversiGame game;
  private ClientSocket blackPlayer;
  private ClientSocket whitePlayer;
  private boolean gameOver = false;

  private FischerClock clock;
  private HistoryStore historyStore;

  public GameSession(ClientSocket black, ClientSocket white, HistoryStore store) {
    this.game = new ReversiGame();
    this.blackPlayer = black;
    this.whitePlayer = white;
    this.historyStore = store;

    this.clock = new FischerClock(100000, 1000, false);
    var eventBus = new EventBus();
    eventBus.register(FischerClock.TimeoutEvent.class,
                      new EventListener<FischerClock.TimeoutEvent>() {
                        @Override
                        public void onEvent(FischerClock.TimeoutEvent e) {
                          onTimeout();
                        }
                      });
    this.clock.setEventBus(eventBus);
    this.clock.setTicker(new Ticker());

    this.clock.start();
  }

  private GameRecord createRecord() {
    Board board = game.getBoard();
    int black = 0, white = 0;
    for (int i = 0; i < Board.BOARD_SIZE; i++) {
      for (int j = 0; j < Board.BOARD_SIZE; j++) {
        var p = board.get(i, j);
        if (p == Player.Black)
          black++;
        else if (p == Player.White)
          white++;
      }
    }
    char w = game.getWinner() == Player.None ? '.' : game.getWinner().toChar();
    return new GameRecord(w, black, white, System.currentTimeMillis());
  }

  private synchronized void finishGame(String blackMsg, String whiteMsg) {
    if (gameOver)
      return;
    gameOver = true;
    clock.stop();
    blackPlayer.sendMessage(new Message(new Message.GameOver(blackMsg)));
    whitePlayer.sendMessage(new Message(new Message.GameOver(whiteMsg)));
    if (historyStore != null)
      historyStore.addRecord(createRecord());
  }

  private synchronized void onTimeout() {
    if (gameOver)
      return;
    Player current = game.getCurrentPlayer();
    if (current == Player.Black) {
      finishGame("Time expired, you lose", "Opponent timed out, you win");
    } else if (current == Player.White) {
      finishGame("Opponent timed out, you win", "Time expired, you lose");
    }
  }

  // Helper method to check if a client is part of this game.
  public boolean containsClient(ClientSocket handler) {
    return handler.equals(blackPlayer) || handler.equals(whitePlayer);
  }

  public int getBlackId() { return blackPlayer.getClientId(); }
  public int getWhiteId() { return whitePlayer.getClientId(); }

  public Player getClientPlayer(ClientSocket handler) {
    if (handler == blackPlayer)
      return Player.Black;
    else if (handler == whitePlayer)
      return Player.White;
    else
      return Player.None;
  }

  public synchronized boolean makeMove(int row, int col, ClientSocket client) {
    if (gameOver)
      return false;
    Player player = getClientPlayer(client);
    if (player != game.getCurrentPlayer())
      return false;

    boolean moveMade = game.makeMove(row, col);
    if (moveMade) {
      clock.swap();
      if (game.isGameOver()) {
        Player winner = game.getWinner();
        String blackMsg, whiteMsg;
        if (winner == Player.Black) {
          blackMsg = "you win";
          whiteMsg = "you lose";
        } else if (winner == Player.White) {
          blackMsg = "you lose";
          whiteMsg = "you win";
        } else {
          blackMsg = "draw";
          whiteMsg = "draw";
        }
        finishGame(blackMsg, whiteMsg);
      }
    }
    return moveMade;
  }

  public boolean isValidMove(int row, int col, ClientSocket client) {
    Player player = getClientPlayer(client);
    if (player != game.getCurrentPlayer())
      return false;

    return game.isValidMove(row, col);
  }

  public ReversiGame getGame() { return game; }
  public FischerClock getClock() { return clock; }
}
