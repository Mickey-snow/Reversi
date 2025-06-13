package com.reversi.common;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import java.util.List;
import java.io.IOException;
import java.util.Map;
import com.reversi.common.GameRecord;

@JsonSerialize(using = Message.Serializer.class)
@JsonDeserialize(using = Message.Deserializer.class)
public class Message {
  // Nested message types for Client -> Server
  public static class LobbyJoin {
    private final String roomNumber;
    @JsonCreator
    public LobbyJoin(@JsonProperty("roomNumber") String roomNumber) {
      this.roomNumber = roomNumber;
    }
    public String getRoomNumber() { return roomNumber; }
  }

  public static class LobbyCreate {
    private final LobbyRoom room;
    @JsonCreator
    public LobbyCreate(@JsonProperty("room") LobbyRoom room) {
      this.room = room;
    }
    public LobbyRoom getRoom() { return room; }
  }

  public static class LobbyReady {
    private final boolean isReady;
    @JsonCreator
    public LobbyReady(@JsonProperty("isReady") boolean isReady) {
      this.isReady = isReady;
    }
    public boolean getIsReady() { return isReady; }
  }

  public static class Move {
    private final int row, col;
    @JsonCreator
    public Move(@JsonProperty("row") int row, @JsonProperty("col") int col) {
      this.row = row;
      this.col = col;
    }
    public int getRow() { return row; }
    public int getCol() { return col; }
  }

  // Nested message types for Server -> Client
  public static class Invalid {
    private final String reason;
    @JsonCreator
    public Invalid(@JsonProperty("reason") String reason) {
      this.reason = reason;
    }
    public String getReason() { return reason; }
  }

  public static class Start {
    private final char color;
    @JsonCreator
    public Start(@JsonProperty("color") char color) {
      this.color = color;
    }
    public char getColor() { return color; }
  }

  public static class GameOver {
    private final String reason;
    @JsonCreator
    public GameOver(@JsonProperty("reason") String reason) {
      this.reason = reason;
    }
    public String getReason() { return reason; }
  }

  public static class GameUpdate {
    private final ReversiGame game;
    private final long blackTimeMs, whiteTimeMs;
    @JsonCreator
    public GameUpdate(@JsonProperty("game") ReversiGame game,
                      @JsonProperty("blackTimeMs") long blackTimeMs,
                      @JsonProperty("whiteTimeMs") long whiteTimeMs) {
      this.game = game;
      this.blackTimeMs = blackTimeMs;
      this.whiteTimeMs = whiteTimeMs;
    }
    public ReversiGame getGame() { return game; }
    public long getBlackTimeMs() { return blackTimeMs; }
    public long getWhiteTimeMs() { return whiteTimeMs; }
  }

  public static class LobbyUpdate {
    private final Map<String, LobbyRoom> lobbyRooms;
    @JsonCreator
    public LobbyUpdate(@JsonProperty("lobbyRooms")
                       Map<String, LobbyRoom> lobbyRooms) {
      this.lobbyRooms = lobbyRooms;
    }
    public Map<String, LobbyRoom> getLobbyRooms() { return lobbyRooms; }
  }

  public static class HistoryRequest {}

  public static class HistoryData {
    private final java.util.List<GameRecord> records;

    @JsonCreator
    public HistoryData(@JsonProperty("records") java.util.List<GameRecord> records) {
      this.records = records;
    }

    public java.util.List<GameRecord> getRecords() { return records; }
  }

  // Tagged union storage
  private final Object msg;
  private final Type type;
  public enum Type {
    Move,
    Invalid,
    Start,
    GameOver,
    Turn,
    Board,
    LobbyJoin,
    LobbyReady,
    GameUpdate,
    LobbyCreate,
    LobbyUpdate,
    HistoryRequest,
    HistoryData
  }

  // Constructors for different message types.
  public Message(Move msg) {
    this.msg = msg;
    this.type = Type.Move;
  }
  public Message(Invalid msg) {
    this.msg = msg;
    this.type = Type.Invalid;
  }
  public Message(Start msg) {
    this.msg = msg;
    this.type = Type.Start;
  }
  public Message(LobbyJoin msg) {
    this.msg = msg;
    this.type = Type.LobbyJoin;
  }
  public Message(LobbyReady msg) {
    this.msg = msg;
    this.type = Type.LobbyReady;
  }
  public Message(GameUpdate msg) {
    this.msg = msg;
    this.type = Type.GameUpdate;
  }
  public Message(LobbyCreate msg) {
    this.msg = msg;
    this.type = Type.LobbyCreate;
  }
  public Message(LobbyUpdate msg) {
    this.msg = msg;
    this.type = Type.LobbyUpdate;
  }
  public Message(GameOver msg) {
    this.msg = msg;
    this.type = Type.GameOver;
  }
  public Message(HistoryRequest msg) {
    this.msg = msg;
    this.type = Type.HistoryRequest;
  }
  public Message(HistoryData msg) {
    this.msg = msg;
    this.type = Type.HistoryData;
  }

  // No-arg constructor for Jackson
  protected Message() {
    this.msg = null;
    this.type = null;
  }

  public Type getType() { return type; }
  public Object getMessage() { return msg; }

  // ===================== Serialization Support =====================
  static class Serializer extends JsonSerializer<Message> {
    @Override
    public void serialize(Message m, JsonGenerator gen,
                          SerializerProvider serializers) throws IOException {
      // Begin writing the JSON object
      gen.writeStartObject();

      // Write the "type" field using the enum name.
      gen.writeStringField("type", m.getType().name());

      // Write the "body" field.
      // Here we indicate to the generator that the "body" field will contain
      // a nested JSON object. Jackson will use the default serializer for this.
      gen.writeFieldName("body");
      serializers.defaultSerializeValue(m.getMessage(), gen);

      // Complete the JSON object.
      gen.writeEndObject();
    }
  }

  static class Deserializer extends JsonDeserializer<Message> {
    @Override
    public Message deserialize(JsonParser p, DeserializationContext ctxt)
        throws IOException {
      ObjectMapper mapper = JacksonObjMapper.get();

      JsonNode node = p.getCodec().readTree(p);
      String typeStr = node.get("type").asText();
      Message.Type type = Message.Type.valueOf(typeStr);
      JsonNode msgNode = node.get("body");

      switch (type) {
      case Move:
        Move move = mapper.treeToValue(msgNode, Move.class);
        return new Message(move);

      case Invalid:
        Invalid invalid = mapper.treeToValue(msgNode, Invalid.class);
        return new Message(invalid);

      case Start:
        Start start = mapper.treeToValue(msgNode, Start.class);
        return new Message(start);

      case GameOver:
        GameOver gameOver = mapper.treeToValue(msgNode, GameOver.class);
        return new Message(gameOver);

      case LobbyJoin:
        LobbyJoin lobbyJoin = mapper.treeToValue(msgNode, LobbyJoin.class);
        return new Message(lobbyJoin);

      case LobbyReady:
        LobbyReady lobbyReady = mapper.treeToValue(msgNode, LobbyReady.class);
        return new Message(lobbyReady);

      case GameUpdate:
        GameUpdate gameUpdate = mapper.treeToValue(msgNode, GameUpdate.class);
        return new Message(gameUpdate);

      case LobbyCreate:
        LobbyCreate lobbyCreate =
            mapper.treeToValue(msgNode, LobbyCreate.class);
        return new Message(lobbyCreate);

      case LobbyUpdate:
        LobbyUpdate lobbyUpdate =
            mapper.treeToValue(msgNode, LobbyUpdate.class);
        return new Message(lobbyUpdate);

      case HistoryRequest:
        HistoryRequest historyReq = mapper.treeToValue(msgNode, HistoryRequest.class);
        return new Message(historyReq);

      case HistoryData:
        HistoryData historyData = mapper.treeToValue(msgNode, HistoryData.class);
        return new Message(historyData);

      default:
        throw new IllegalStateException("Unexpected type: " + typeStr);
      }
    }
  }
}
