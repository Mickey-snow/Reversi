package com.reversi.common;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import java.io.IOException;

/**
 * {@code ReversiGame} encapsulates the core logic of a Reversi game including
 * board state management, move validation, disc flipping, and player turn
 * management. It is responsible for executing game moves and
 * serializing/deserializing game state to/from JSON.
 *
 * <p>The class leverages Jackson annotations to enable JSON serialization and
 * deserialization using custom serializer and deserializer classes.
 */
@JsonSerialize(using = ReversiGame.Serializer.class)
@JsonDeserialize(using = ReversiGame.Deserializer.class)
public class ReversiGame {
  // The game board representing the state of play.
  private Board board;

  // The player whose turn it is. Uses the enum: 'B' for Black, 'W' for White.
  private Player currentPlayer;

  /**
   * Default constructor that initializes the game with the standard starting
   * board and sets the initial player to Black.
   */
  public ReversiGame() {
    board = Board.createDefault();
    currentPlayer = Player.Black; // Black always starts first.
  }

  /**
   * Constructs a game instance with a specific board state and current player.
   *
   * @param board         the initial board configuration
   * @param currentPlayer the player whose turn is first in this configuration
   */
  public ReversiGame(Board board, Player currentPlayer) {
    this.board = board;
    this.currentPlayer = currentPlayer;
  }

  /**
   * Retrieves the current state of the game board.
   *
   * @return the {@link Board} representing the current board state
   */
  public Board getBoard() { return board; }

  /**
   * Retrieves the current player whose turn is to make a move.
   *
   * @return the {@link Player} currently active
   */
  public Player getCurrentPlayer() { return currentPlayer; }

  /**
   * Determines whether placing a disc at the specified row and column is a
   * valid move for the current player.
   *
   * @param row the 0-indexed row position on the board
   * @param col the 0-indexed column position on the board
   * @return {@code true} if the move is valid; {@code false} otherwise
   */
  public boolean isValidMove(int row, int col) {
    return board.isValidMove(row, col, currentPlayer);
  }

  /**
   * Attempts to execute a move for the current player at the specified
   * position. <p> If the move is valid, the board state is updated (including
   * flipping any opponent discs) and the turn is switched to the opposing
   * player.
   * </p>
   *
   * @param row the 0-indexed row position where the disc is placed
   * @param col the 0-indexed column position where the disc is placed
   * @return {@code true} if the move was successfully executed; {@code false}
   *     if invalid
   */
  public boolean makeMove(int row, int col) {
    if (!board.makeMove(row, col, currentPlayer))
      return false;

    // Switch the turn to the opponent.
    currentPlayer = currentPlayer.opponent();
    return true;
  }

  /**
   * Returns true if neither Black nor White has any valid moves remaining.
   *
   * @return whether the game is finished
   */
  public boolean isGameOver() {
    // When both players have no moves, the game ends.
    return board.getValidMoves(Player.Black).isEmpty() &&
        board.getValidMoves(Player.White).isEmpty();
  }

  /**
   * Determines the winner by counting discs on the board.
   *
   * @return Player.Black if Black has more discs,
   *         Player.White if White has more,
   *         or Player.None if tied
   */
  public Player getWinner() {
    int blackCount = 0, whiteCount = 0;
    for (int i = 0; i < Board.BOARD_SIZE; i++) {
      for (int j = 0; j < Board.BOARD_SIZE; j++) {
        Player p = board.get(i, j);
        if (p == Player.Black) {
          blackCount++;
        } else if (p == Player.White) {
          whiteCount++;
        }
      }
    }
    if (blackCount > whiteCount) {
      return Player.Black;
    } else if (whiteCount > blackCount) {
      return Player.White;
    } else {
      return Player.None;
    }
  }

  /**
   * Compares the provided object with this game instance for equality. Two game
   * instances are considered equal if they have the same board state and the
   * same current player.
   *
   * @param o the object to be compared for equality with this game
   * @return {@code true} if the specified object is equal to this game; {@code
   *     false} otherwise
   */
  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;
    ReversiGame other = (ReversiGame)o;
    return currentPlayer == other.currentPlayer && board.equals(other.board);
  }

  /**
   * Loads the state of this game from another game instance.
   *
   * @param o the other {@link ReversiGame} whose state is to be copied
   */
  public void loadFrom(ReversiGame o) {
    this.board = o.board;
    this.currentPlayer = o.currentPlayer;
  }

  // ===================== Serialization Support =====================

  /**
   * {@code Serializer} provides a custom JSON serialization for {@link
   * ReversiGame}. It is used by Jackson to write the game state to a JSON
   * structure.
   */
  public static class Serializer extends JsonSerializer<ReversiGame> {
    /**
     * Serializes the given {@code ReversiGame} object into JSON.
     *
     * @param game         the {@link ReversiGame} instance to serialize
     * @param gen          the {@link JsonGenerator} used to write JSON content
     * @param serializers  the {@link SerializerProvider} that can be used for
     *     obtaining serializers for
     *                     serializing objects contained within {@code game}
     * @throws IOException if an I/O error occurs during serialization
     */
    @Override
    public void serialize(ReversiGame game, JsonGenerator gen,
                          SerializerProvider serializers) throws IOException {
      gen.writeStartObject();

      // Delegate the serialization of the board to its configured serializer.
      gen.writeFieldName("board");
      serializers.defaultSerializeValue(game.getBoard(), gen);

      // Write the current player as a single character string.
      gen.writeStringField("current_player",
                           String.valueOf(game.getCurrentPlayer().toChar()));

      gen.writeEndObject();
    }
  }

  /**
   * {@code Deserializer} handles custom JSON deserialization for {@link
   * ReversiGame}. It reads JSON data and reconstructs a {@code ReversiGame}
   * instance accordingly.
   */
  public static class Deserializer extends JsonDeserializer<ReversiGame> {
    /**
     * Deserializes JSON content into a {@link ReversiGame} object.
     *
     * @param p      the {@link JsonParser} used for reading JSON content
     * @param ctxt   the {@link DeserializationContext} that can be used to
     *     access deserializers for handling
     *               complex types
     * @return a fully constructed {@code ReversiGame} instance reflecting the
     *     JSON data
     * @throws IOException if the JSON content is malformed or missing required
     *     fields
     */
    @Override
    public ReversiGame deserialize(JsonParser p, DeserializationContext ctxt)
        throws IOException {
      Board board = null;
      Player currentPlayer = null;

      // Validate that the JSON structure begins with an object.
      if (p.currentToken() != JsonToken.START_OBJECT) {
        throw new IOException("Expected START_OBJECT token, but found: " +
                              p.currentToken());
      }

      // Iterate over all fields within the JSON object.
      while (p.nextToken() != JsonToken.END_OBJECT) {
        String fieldName = p.getCurrentName();
        p.nextToken(); // Move to the field's value.

        if ("board".equals(fieldName)) {
          // Deserialize the 'board' field using Board's custom deserializer.
          board = ctxt.readValue(p, Board.class);
        } else if ("current_player".equals(fieldName)) {
          String cpStr = p.getValueAsString();
          if (cpStr == null || cpStr.isEmpty()) {
            throw new IOException(
                "Invalid or missing value for field 'current_player'");
          }
          // Convert the string representation to the corresponding Player enum.
          currentPlayer = Player.from(cpStr.charAt(0));
        } else {
          // If an unexpected field is encountered, trigger an error.
          throw new IOException("Unexpected field: " + fieldName);
        }
      }

      // Ensure that all mandatory fields have been provided.
      if (board == null) {
        throw new IOException("Missing required field: board");
      }
      if (currentPlayer == null) {
        throw new IOException("Missing required field: current_player");
      }

      return new ReversiGame(board, currentPlayer);
    }
  }
}
