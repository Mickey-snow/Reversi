package com.reversi.client;

import com.reversi.common.LobbyRoom;
import com.reversi.common.Message;
import java.util.Map;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.*;

public class LobbyView {

  private BorderPane mainPane;
  private TextField roomNameField;
  private Button joinButton;
  private Button createButton;
  private Label lobbyStatusLabel;
  private ListView<String> roomsListView;

  private ServerSocket serverSocket;

  public LobbyView() {
    createComponents();
    layoutComponents();
    attachListeners();
    this.serverSocket = null;
  }

  public void setServerSocket(ServerSocket serverSocket) {
    this.serverSocket = serverSocket;
  }

  private void createComponents() {
    // Top status label.
    lobbyStatusLabel = new Label("Not connected to any room");
    lobbyStatusLabel.getStyleClass().add("status-label");

    // Create room input field and buttons.
    roomNameField = new TextField();
    roomNameField.setPromptText("Enter Room Number");

    joinButton = new Button("Join Room");
    createButton = new Button("Create Room");

    // List view for available rooms.
    roomsListView = new ListView<>();
    roomsListView.getStyleClass().add("rooms-list");
  }

  private void layoutComponents() {
    mainPane = new BorderPane();
    mainPane.setPadding(new Insets(10));

    // Left side: rooms list.
    VBox leftBox = new VBox(10, new Label("Available Rooms:"), roomsListView);
    leftBox.setPadding(new Insets(10));
    leftBox.getStyleClass().add("left-pane");
    mainPane.setLeft(leftBox);

    // Center: form to join or create a room.
    GridPane centerGrid = new GridPane();
    centerGrid.setAlignment(Pos.CENTER);
    centerGrid.setHgap(10);
    centerGrid.setVgap(10);
    centerGrid.add(new Label("Room:"), 0, 0);
    centerGrid.add(roomNameField, 1, 0);
    centerGrid.add(joinButton, 0, 1);
    centerGrid.add(createButton, 1, 1);
    mainPane.setCenter(centerGrid);

    // Top: status label.
    HBox topBox = new HBox(lobbyStatusLabel);
    topBox.setAlignment(Pos.CENTER);
    topBox.setPadding(new Insets(10));
    mainPane.setTop(topBox);
  }

  private void attachListeners() {
    // Handle join room via button.
    joinButton.setOnAction(e -> {
      String roomId = roomNameField.getText().trim();
      if (!roomId.isEmpty()) {
        joinRoom(roomId);
      } else {
        lobbyStatusLabel.setText("Please enter a valid room ID");
      }
    });

    // Handle create room button.
    createButton.setOnAction(e -> {
      String roomId = roomNameField.getText().trim();
      if (!roomId.isEmpty()) {
        createRoom(roomId);
      } else {
        lobbyStatusLabel.setText("Room ID cannot be empty.");
      }
    });

    // Enable joining via double click on a room in the list.
    roomsListView.addEventFilter(MouseEvent.MOUSE_CLICKED, event -> {
      if (event.getClickCount() == 2) {
        String selectedRoom =
            roomsListView.getSelectionModel().getSelectedItem();
        if (selectedRoom != null) {
          joinRoom(selectedRoom);
        }
      }
    });
  }

  private void joinRoom(String roomId) {
    serverSocket.send(new Message(new Message.LobbyJoin(roomId)));
    lobbyStatusLabel.setText("Joining room: " + roomId + " ...");
  }

  private void createRoom(String roomId) {
    // TODO: add room config support
    var message = new Message(new Message.LobbyCreate(new LobbyRoom(roomId)));
    serverSocket.send(message);
    lobbyStatusLabel.setText("Creating room: " + roomId +
                             ", waiting players to join ...");
  }

  public BorderPane getMainPane() { return mainPane; }

  public void update(Map<String, LobbyRoom> lobbyRooms) {
    roomsListView.getItems().clear();
    roomsListView.getItems().addAll(lobbyRooms.keySet());
  }
}
