package com.reversi.common;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

/** Represents a completed game result. */
public class GameRecord {
  private final char winner;
  private final int blackScore;
  private final int whiteScore;
  private final long timestamp;

  @JsonCreator
  public GameRecord(
      @JsonProperty("winner") char winner,
      @JsonProperty("blackScore") int blackScore,
      @JsonProperty("whiteScore") int whiteScore,
      @JsonProperty("timestamp") long timestamp) {
    this.winner = winner;
    this.blackScore = blackScore;
    this.whiteScore = whiteScore;
    this.timestamp = timestamp;
  }

  public char getWinner() { return winner; }
  public int getBlackScore() { return blackScore; }
  public int getWhiteScore() { return whiteScore; }
  public long getTimestamp() { return timestamp; }
}
