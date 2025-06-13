package com.reversi.server;

import com.reversi.common.EventBus;
import com.reversi.common.EventListener;
import com.reversi.common.FischerClock;
import com.reversi.common.LobbyRoom;
import com.reversi.common.Message;
import com.reversi.common.PlayerStatus;
import com.reversi.common.ReversiGame;
import com.reversi.server.events.GameStateChange;
import com.reversi.server.HistoryStore;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SessionHub {
  private static final Logger logger =
      LoggerFactory.getLogger(SessionHub.class);

  // Maps to manage clients, lobby rooms, and active game sessions.
  private final Map<Integer, ClientSocket> clients = new HashMap<>();
  private final Map<String, LobbyRoom> lobbyRooms = new HashMap<>();
  private final Map<String, GameSession> activeGameSessions = new HashMap<>();

  private final HistoryStore historyStore;

  private final EventBus eventBus = new EventBus();
  private final List<Object> listeners = new ArrayList<>();

  public SessionHub(HistoryStore store) {
    this.historyStore = store;
    // Register event listeners.
    ClientMessageListener clientListener = new ClientMessageListener();
    GameSessionUpdateListener gameListener = new GameSessionUpdateListener();
    listeners.add(clientListener);
    listeners.add(gameListener);
    eventBus.register(ClientMessage.class, clientListener);
    eventBus.register(GameStateChange.class, gameListener);
  }

  /**
   * Adds a new client to the managed clients.
   *
   * @param client The ClientSocket instance.
   */
  public void registerClient(ClientSocket client) {
    synchronized (clients) { clients.put(client.getClientId(), client); }
    sendLobbyUpdate(client);
  }

  public EventBus getEventBus() { return this.eventBus; }

  private void sendLobbyUpdate(ClientSocket client) {
    var message = new Message(new Message.LobbyUpdate(lobbyRooms));

    if (client != null)
      client.sendMessage(message);
    else {
      for (var it : clients.values())
        it.sendMessage(message);
    }
  }

  private void sendHistory(ClientSocket client) {
    var message = new Message(new Message.HistoryData(historyStore.getHistory()));
    if (client != null) {
      client.sendMessage(message);
    }
  }

  // --- Inner classes for event listeners ---
  class ClientMessageListener implements EventListener<ClientMessage> {
    @Override
    public void onEvent(ClientMessage e) {
      Message msg = e.getMessage();
      ClientSocket handler = e.getHandler();

      switch (msg.getType()) {
      case LobbyCreate: {
        Message.LobbyCreate lobbyCreate = (Message.LobbyCreate)msg.getMessage();
        LobbyRoom room = lobbyCreate.getRoom();
        room.addPlayer(new PlayerStatus(handler.getClientId()));
        synchronized (lobbyRooms) {
          if (lobbyRooms.containsKey(room.getRoomName())) {
            handler.sendMessage(new Message(new Message.Invalid(
                "Room " + room.getRoomName() + " already exists.")));
            break;
          }
          lobbyRooms.put(room.getRoomName(), room);
        }

        sendLobbyUpdate(null);
        logger.info("Client {} created room {}", handler.getClientId(),
                    room.getRoomName());
        break;
      }
      case LobbyJoin: {
        Message.LobbyJoin lobbyJoin = (Message.LobbyJoin)msg.getMessage();
        String roomId = lobbyJoin.getRoomNumber();
        LobbyRoom room;
        synchronized (lobbyRooms) { room = lobbyRooms.get(roomId); }
        if (room == null) {
          handler.sendMessage(
              new Message(new Message.Invalid("Room " + roomId + " invalid.")));
          break;
        }

        room.addPlayer(new PlayerStatus(handler.getClientId()));
        sendLobbyUpdate(null);
        logger.info("Client {} joined room {}", handler.getClientId(), roomId);

        if (room.isReadyToStart()) {
          var players = room.getPlayers().keySet().toArray();
          if (players.length == 2) {
            ClientSocket blackPlayer = clients.get(players[0]);
            ClientSocket whitePlayer = clients.get(players[1]);
            GameSession gameSession = new GameSession(blackPlayer, whitePlayer, historyStore);
            synchronized (activeGameSessions) {
              activeGameSessions.put(room.getRoomName(), gameSession);
            }
            // Notify players that the game just started.
            blackPlayer.sendMessage(new Message(new Message.Start('B')));
            whitePlayer.sendMessage(new Message(new Message.Start('W')));
            eventBus.post(new GameStateChange(gameSession));
            logger.info("Game session started for room {}", room.getRoomName());
            synchronized (lobbyRooms) { lobbyRooms.remove(room.getRoomName()); }
          }
        }

        break;
      }
      case Move: {
        // Route move messages to the appropriate game session.
        GameSession session = null;
        synchronized (activeGameSessions) {
          for (GameSession gs : activeGameSessions.values()) {
            if (gs.containsClient(handler)) {
              session = gs;
              break;
            }
          }
        }
        if (session != null) {
          Message.Move move = (Message.Move)msg.getMessage();
          boolean valid =
              session.makeMove(move.getRow(), move.getCol(), handler);
          if (!valid) {
            handler.sendMessage(
                new Message(new Message.Invalid("Invalid move")));
          }
          eventBus.post(new GameStateChange(session));
        } else {
          logger.error(
              "Received move from client {} with no active game session.",
              handler.getClientId());
        }
        break;
      }
      case HistoryRequest: {
        sendHistory(handler);
        break;
      }
      default:
        logger.warn("Message ignored: {}", msg.toString());
        break;
      }
    }
  }

  class GameSessionUpdateListener implements EventListener<GameStateChange> {
    @Override

    public void onEvent(GameStateChange e) {
      GameSession session = e.getSession();
      ReversiGame game = session.getGame();
      FischerClock clock = session.getClock();

      var gameUpd = new Message.GameUpdate(game, clock.getBlackTimeMillis(),
                                           clock.getWhiteTimeMillis());
      var message = new Message(gameUpd);
      ClientSocket blackPlayer;
      ClientSocket whitePlayer;
      synchronized (clients) {
        blackPlayer = clients.get(session.getBlackId());
        whitePlayer = clients.get(session.getWhiteId());
      }
      if (blackPlayer != null) {
        blackPlayer.sendMessage(message);
      }
      if (whitePlayer != null) {
        whitePlayer.sendMessage(message);
      }
    }
  }
}
