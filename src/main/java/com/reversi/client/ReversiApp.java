package com.reversi.client;

import com.reversi.common.EventBus;
import com.reversi.common.EventListener;
import com.reversi.common.Message;
import com.reversi.common.Player;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

public class ReversiApp extends Application {

  private EventBus eventBus;
  private ServerSocket serverSocket;

  private StackPane rootPane;
  private LobbyView lobbyView;
  private Scene scene;
  private GameView gameView;

  // the event listener
  private Listener listener;

  @Override
  public void start(Stage primaryStage) {
    this.eventBus = new EventBus();
    this.serverSocket = new ServerSocket(eventBus);
    this.listener = this.new Listener();
    eventBus.register(ServerMessage.class, listener);

    this.rootPane = new StackPane();
    var stylesCss =
        getClass().getClassLoader().getResource("styles.css").toExternalForm();

    // Create an instance of the lobby view.
    this.lobbyView = new LobbyView();
    lobbyView.setServerSocket(serverSocket);
    lobbyView.getMainPane().setVisible(true);

    // Create an instance of the game view.
    this.gameView = new GameView();
    gameView.setController(serverSocket);
    gameView.getGamePane().setVisible(false);

    rootPane.getChildren().addAll(lobbyView.getMainPane(),
                                  gameView.getGamePane());
    this.scene = new Scene(rootPane, 800, 600);
    // Load the external CSS file for styling.
    scene.getStylesheets().add(stylesCss);

    primaryStage.setTitle("Reversi Game Lobby");
    primaryStage.setScene(scene);
    primaryStage.show();

    serverSocket.connectToServer();
  }

  class Listener implements EventListener<ServerMessage> {
    @Override
    public void onEvent(ServerMessage e) {
      Message msg = e.getMessage();
      switch (msg.getType()) {
      case LobbyUpdate:
        Message.LobbyUpdate update = (Message.LobbyUpdate)msg.getMessage();
        lobbyView.update(update.getLobbyRooms());
        break;

      case Start:
        lobbyView.getMainPane().setVisible(false);
        gameView.getGamePane().setVisible(true);
        Message.Start start = (Message.Start)msg.getMessage();
        gameView.setUs(Player.from(start.getColor()));
        break;
      case GameUpdate:
        Message.GameUpdate upd = (Message.GameUpdate)msg.getMessage();
        gameView.updateGame(upd.getGame());
        gameView.updateTime(upd.getBlackTimeMs(), upd.getWhiteTimeMs());
        break;
      case GameOver:
        Message.GameOver over = (Message.GameOver)msg.getMessage();
        gameView.showGameOver(over.getReason());
        break;
      case Invalid:
        // TODO: no-op for now
        break;

      default:
        break;
      }
    }
  }

  public static void main(String[] args) { launch(args); }
}
