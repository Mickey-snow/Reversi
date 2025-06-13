package com.reversi.common;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import com.reversi.common.GameRecord;
import org.junit.jupiter.api.Test;

public class MessageTest {
  static String serialize(Message m) throws IOException {
    var mapper = JacksonObjMapper.get();
    return mapper.writeValueAsString(m);
  }

  static Message deserialize(String s) throws IOException {
    var mapper = JacksonObjMapper.get();
    return mapper.readValue(s, Message.class);
  }

  @Test
  void testSerializeDeserializeMove() {
    Message.Move move = new Message.Move(3, 2);
    Message msg = new Message(move);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.Move, deserialized.getType(),
                 "Message type should be Move");

    Message.Move moveDeserialized = (Message.Move)deserialized.getMessage();
    assertEquals(3, moveDeserialized.getRow(), "Row value not matching");
    assertEquals(2, moveDeserialized.getCol(), "Column value not matching");
  }

  @Test
  void testSerializeDeserializeInvalid() {
    Message.Invalid invalid = new Message.Invalid("Invalid operation");
    Message msg = new Message(invalid);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.Invalid, deserialized.getType(),
                 "Message type should be Invalid");

    Message.Invalid invalidDeserialized =
        (Message.Invalid)deserialized.getMessage();
    assertEquals("Invalid operation", invalidDeserialized.getReason(),
                 "Invalid reason does not match");
  }

  @Test
  void testSerializeDeserializeStart() {
    Message.Start start = new Message.Start('B');
    Message msg = new Message(start);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.Start, deserialized.getType(),
                 "Message type should be Start");

    Message.Start startDeserialized = (Message.Start)deserialized.getMessage();
    assertEquals('B', startDeserialized.getColor(),
                 "Starting color does not match");
  }

  @Test
  void testSerializeDeserializeLobbyJoin() {
    Message.LobbyJoin lobbyJoin = new Message.LobbyJoin("1234");
    Message msg = new Message(lobbyJoin);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON for LobbyJoin should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.LobbyJoin, deserialized.getType(),
                 "Message type should be LobbyJoin");

    Message.LobbyJoin lobbyJoinDeserialized =
        (Message.LobbyJoin)deserialized.getMessage();
    assertEquals("1234", lobbyJoinDeserialized.getRoomNumber(),
                 "Room number does not match");
  }

  @Test
  void testSerializeDeserializeLobbyReady() {
    Message.LobbyReady lobbyReady = new Message.LobbyReady(true);
    Message msg = new Message(lobbyReady);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON for LobbyReady should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.LobbyReady, deserialized.getType(),
                 "Message type should be LobbyReady");

    Message.LobbyReady lobbyReadyDeserialized =
        (Message.LobbyReady)deserialized.getMessage();
    assertTrue(lobbyReadyDeserialized.getIsReady(),
               "Ready status does not match");
  }

  @Test
  void testSerializeDeserializeGameUpdate() {
    var game = new ReversiGame();

    Message.GameUpdate gameUpdate = new Message.GameUpdate(game, 100, 200);
    Message msg = new Message(gameUpdate);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON for GameUpdate should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.GameUpdate, deserialized.getType(),
                 "Message type should be GameUpdate");

    Message.GameUpdate deserializedMessage =
        (Message.GameUpdate)deserialized.getMessage();
    assertTrue(game.equals(deserializedMessage.getGame()));
    assertEquals(100, deserializedMessage.getBlackTimeMs());
    assertEquals(200, deserializedMessage.getWhiteTimeMs());
  }

  @Test
  void testSerializeDeserializeLobbyCreate() {
    // Create a LobbyRoom instance to be used in LobbyCreate
    LobbyRoom room = new LobbyRoom("TestRoom");
    Message.LobbyCreate lobbyCreate = new Message.LobbyCreate(room);
    Message msg = new Message(lobbyCreate);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON for LobbyCreate should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json), json);
    assertEquals(Message.Type.LobbyCreate, deserialized.getType(),
                 "Message type should be LobbyCreate");

    Message.LobbyCreate lobbyCreateDeserialized =
        (Message.LobbyCreate)deserialized.getMessage();
    assertEquals("TestRoom", lobbyCreateDeserialized.getRoom().getRoomName(),
                 "LobbyRoom name does not match");
  }

  @Test
  void testSerializeDeserializeLobbyUpdate() {
    // Create a map of LobbyRooms for LobbyUpdate
    Map<String, LobbyRoom> lobbyRooms = new HashMap<>();
    LobbyRoom room1 = new LobbyRoom("Room1");
    lobbyRooms.put("room1", room1);
    Message.LobbyUpdate lobbyUpdate = new Message.LobbyUpdate(lobbyRooms);
    Message msg = new Message(lobbyUpdate);

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json, "Serialized JSON for LobbyUpdate should not be null");

    Message deserialized = assertDoesNotThrow(() -> deserialize(json), json);
    assertEquals(Message.Type.LobbyUpdate, deserialized.getType(),
                 "Message type should be LobbyUpdate");

    Message.LobbyUpdate lobbyUpdateDeserialized =
        (Message.LobbyUpdate)deserialized.getMessage();
    Map<String, LobbyRoom> deserializedRooms =
        lobbyUpdateDeserialized.getLobbyRooms();
    assertTrue(deserializedRooms.containsKey("room1"),
               "LobbyUpdate should contain key 'room1'");
    assertEquals("Room1", deserializedRooms.get("room1").getRoomName(),
                 "LobbyRoom name does not match");
  }

  @Test
  void testSerializeDeserializeGameOver() {
    String reason = "time out";
    var msg = new Message(new Message.GameOver(reason));

    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json);

    Message deserialized = assertDoesNotThrow(() -> deserialize(json), json);
    assertEquals(Message.Type.GameOver, deserialized.getType());
    Message.GameOver gameoverDeserialized =
        (Message.GameOver)deserialized.getMessage();
    assertTrue(reason.equals(gameoverDeserialized.getReason()));
  }

  @Test
  void testSerializeDeserializeHistoryRequest() {
    var msg = new Message(new Message.HistoryRequest());
    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json);
    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.HistoryRequest, deserialized.getType());
  }

  @Test
  void testSerializeDeserializeHistoryData() {
    java.util.List<GameRecord> list = new java.util.ArrayList<>();
    list.add(new GameRecord('B', 10, 5, 1L));
    var msg = new Message(new Message.HistoryData(list));
    String json = assertDoesNotThrow(() -> serialize(msg));
    assertNotNull(json);
    Message deserialized = assertDoesNotThrow(() -> deserialize(json));
    assertEquals(Message.Type.HistoryData, deserialized.getType());
    Message.HistoryData data = (Message.HistoryData)deserialized.getMessage();
    assertEquals(1, data.getRecords().size());
    assertEquals('B', data.getRecords().get(0).getWinner());
  }
}
